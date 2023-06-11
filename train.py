import os
import fire
import torch
import wandb
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM# , BitsAndBytesConfig

def train(
    model_name="tiiuae/falcon-7b",
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
    hf_hub_token: str = None,
    wandb_project_name: str = 'assistant',
    data_dir: str = 'data',
    save_dir: str = 'checkpoints',
):
    # no 4bit bs for now
    # # set up 4 bit config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # set up accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=effective_batch_size // microbatch_size,
    )

    # get train dataloader
    if os.path.exists(os.path.join(data_dir, "train_dataloader.pt")):
        train_dataloader = torch.load(os.path.join(data_dir, "train_dataloader.pt"))
        datasets = train_dataloader['datasets']
        train_dataloader = train_dataloader['dataloader']
    else:
        raise RuntimeError("No train_dataloader.pt found in data_dir. Make sure to run data.py first.")
    

    # get model
    with accelerator.main_process_first():
        config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_hub_token, trust_remote_code=True)
        config.hidden_dropout = hidden_dropout_prob
        config.attention_dropout = attn_dropout_prob
        # otherwise you get a really annoying warning on every training step
        config.use_cache=False
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            config=config,
            use_auth_token=hf_hub_token, 
            trust_remote_code=True
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

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

    # set up wandb (main process only, grouped makes wandb crazy slow ime)
    # wandb.init(
    #     project=wandb_project_name,
    #     config = {
    #         "model_name": model_name,
    #         "model_config": config.__dict__,
    #         "scheduler": scheduler_kwargs,
    #         "datasets": datasets
    #     }
    # )
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
            optimizer.zero_grad()
            # if accelerator.is_main_process:
            #     # reporting this running loss should average over the same # of batches
            #     # as the actual loss used to compute gradient, even if not the same
            #     # specific batches, since this is all just on one process.
            #     if len(running_losses) >= effective_batch_size // microbatch_size:
            #         wandb.log({
            #             "epoch": epoch,
            #             "microbatch_loss": loss.item(),
            #             "running_loss": sum(running_losses) / len(running_losses),
            #             "lr": scheduler.get_last_lr()[0],
            #             "total_tokens": total_tokens,
            #         })
            #         running_losses = []
            #     else:
            #         wandb.log({
            #             "epoch": epoch,
            #             "microbatch_loss": loss.item(),
            #             "lr": scheduler.get_last_lr()[0],
            #             "total_tokens": total_tokens,
            #         })
        if accelerator.is_main_process:
            print(f"Finished epoch {epoch}. Saving checkpoint...")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pt"))
            print("Model saved!")



    # accelerate training loop

# eventually, maybe just want to make like a SFTTrainer class where you just call trainer.fit()
if __name__ == '__main__':
    fire.Fire(train)