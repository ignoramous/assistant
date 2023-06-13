import os
import fire
import torch
import numpy as np
import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator

@torch.no_grad()
def evaluate(accelerator, model, eval_dataloaders):
    model.eval()
    metrics = {}
    for key in eval_dataloaders:
        print(f"Evaluating on {key}...")
        loader = eval_dataloaders[key]
        losses = []
        for batch in tqdm.tqdm(loader):
            logits = model(input_ids=batch['input_ids']).logits.float()
            batch_losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), batch["targets"].view(-1), reduction="none"
            )
            losses.extend(accelerator.gather(batch_losses).cpu().numpy())
        losses = losses[:len(loader.dataset)]
        metrics[f"{key}_loss"] = np.mean(losses)

    return metrics

def run_evals(
    model_name: str,
    checkpoint_path: str,
    lora: bool = False,
    quantize: str = None,
    mixed_precision: str = "bf16",
    data_dir: str = "data",
):
    # validate args
    if quantize not in [None, "4bit", "8bit"]:
        raise ValueError("quantize must be one of [None, '4bit', '8bit']")
    
    if not lora and quantize:
        raise ValueError("Full model checkpoints aren't quantized so quantizing is not supported right now.")

    # accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )

    # set up 4/8 bit config
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

    # load model
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # load checkpoint
    if lora:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
        )
    else:
        model.load_state_dict(torch.load(checkpoint_path))

    # get eval dataloaders
    if os.path.exists(os.path.join(data_dir, "eval_dataloaders.pt")):
        eval_dataloaders = torch.load(os.path.join(data_dir, "eval_dataloaders.pt"))
    else:
        raise RuntimeError("No eval_dataloaders.pt found in data_dir. Make sure to run data.py first.")
    
    # evaluate
    metrics = evaluate(accelerator, model, eval_dataloaders)

    for key in metrics:
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    fire.Fire(run_evals)