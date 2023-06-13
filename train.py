import os
import warnings
import fire
import torch
import wandb
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from eval import evaluate


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train(
    wandb_api_key: str = None,
    model_name="tiiuae/falcon-7b",
    quantize: str = None,
    lora: bool = False,
    lora_layers: list[str] = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout: float = 0.05,
    mixed_precision: str = "bf16",
    hidden_dropout_prob: float = 0.1,
    attn_dropout_prob: float = 0.1,
    gradient_checkpointing: bool = True,
    max_lr: float = 1e-5,
    lr_warmup_pct: float = 0.005,
    lr_warmup_factor: float = 25.0,
    lr_decay_factor: float = 10.0,
    num_epochs: int = 1,
    effective_batch_size: int = 32,
    microbatch_size: int = 4,
    wandb_project_name: str = 'assistant',
    data_dir: str = 'data',
    save_dir: str = 'checkpoints',
):
    # Validate parameters
    if quantize not in [None, "4bit", "8bit"]:
        raise ValueError("quantize must be one of [None, '4bit', '8bit']")
    
    if lora: 
        if hidden_dropout_prob + attn_dropout_prob > 0.0:
            warnings.warn("You've enabled dropout on parameters that are frozen for LoRA. You may want to set hidden_dropout_prob and attn_dropout_prob to 0.0.")
        if lora_layers is None or len(lora_layers) == 0:
            raise ValueError("You must specify which layers to apply LoRA to.")
        
    else:
        if quantize:
            raise RuntimeError("Quantization is not supported for full finetuning. Please enable LoRA to train a quantized model.")
        if lora_layers is not None and len(lora_layers) > 0:
            warnings.warn("You've specified layers to apply LoRA to, but have not enabled LoRA. The LoRA layers will be ignored.")
        if lora_dropout is not None:
            warnings.warn("LoRA dropout rate of {} will be ignored since LoRA is not enabled.".format(lora_dropout))

    # create save dir
    if not os.path.exists(save_dir):
        print(f"Creating save directory at {save_dir}")
        os.makedirs(save_dir)

    # set up accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=effective_batch_size // microbatch_size,
    )

    # set up 4 bit config
    if quantize == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True
        )
    elif quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # get train dataloader
    if os.path.exists(os.path.join(data_dir, "train_dataloader.pt")):
        train_dataloader = torch.load(os.path.join(data_dir, "train_dataloader.pt"))
        datasets = train_dataloader['datasets']
        train_dataloader = train_dataloader['dataloader']
    else:
        raise RuntimeError("No train_dataloader.pt found in data_dir. Make sure to run data.py first.")
    
    # get eval dataloaders
    if os.path.exists(os.path.join(data_dir, "eval_dataloaders.pt")):
        eval_dataloaders = torch.load(os.path.join(data_dir, "eval_dataloaders.pt"))
    else:
        raise RuntimeError("No eval_dataloaders.pt found in data_dir. Make sure to run data.py first.")

    # get model
    with accelerator.main_process_first():
        # make sure to do huggingface-cli login if model is not public
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.hidden_dropout = hidden_dropout_prob
        model_config.attention_dropout = attn_dropout_prob
        model_config.use_cache=False
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if quantize is not None:
            model = prepare_model_for_kbit_training(model)

        if lora:
            lora_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["query_key_value", "dense_h_to_4h", "dense_4h_to_h"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        
        print_trainable_parameters(model)

    # get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    # get scheduler
    scheduler_kwargs = {
        "max_lr": max_lr,
        "total_steps": len(train_dataloader) * num_epochs,
        "pct_start": lr_warmup_pct,
        "div_factor": lr_warmup_factor,
        "final_div_factor": lr_decay_factor,
        "anneal_strategy": "linear",
        "three_phase": False,
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, scheduler
    )

    # prepare eval dataloaders
    for key in eval_dataloaders:
        eval_dataloaders[key] = accelerator.prepare_data_loader(eval_dataloaders[key])

    # set up wandb (main process only, grouped makes wandb crazy slow ime)
    if accelerator.is_main_process:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project_name,
            config = {
                "model_name": model_name,
                "model_config": model_config.__dict__,
                "scheduler": scheduler_kwargs,
                "datasets": datasets
            }
        )
    criterion = torch.nn.CrossEntropyLoss()

    # train
    model.train()
    if accelerator.is_main_process:
        print("Beginning training...")
    total_tokens = 0
    running_losses = []
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            total_tokens += batch["input_ids"].numel() * accelerator.num_processes
            logits = model(input_ids=batch['input_ids']).logits
            loss = criterion(logits.view(-1, logits.shape[-1]), batch["targets"].view(-1))
            if accelerator.is_main_process:
                running_losses.append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                # running loss averages over the same # of batches
                # as the actual loss used to compute gradient, even if not the same
                # specific batches, since this is all just on one process.
                if len(running_losses) >= effective_batch_size // microbatch_size:
                    wandb.log({
                        "epoch": epoch,
                        "microbatch_loss": loss.item(),
                        "running_loss": sum(running_losses) / len(running_losses),
                        "lr": scheduler.get_last_lr()[0],
                        "total_tokens": total_tokens,
                    })
                    running_losses = []
                else:
                    wandb.log({
                        "epoch": epoch,
                        "microbatch_loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "total_tokens": total_tokens,
                    })
        # At end of epoch, evaluate and save model. :)
        

        if accelerator.is_main_process:
            print(f"Finished epoch {epoch}. Saving checkpoint...")
            if not lora:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pt"))
            else:
                model.save_pretrained(os.path.join(save_dir, f"model_{epoch}"))
            print("Model saved!")

# eventually, maybe just want to make like a SFTTrainer class where you just call trainer.fit()
if __name__ == '__main__':
    fire.Fire(train)