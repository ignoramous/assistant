import os
import fire
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM# , BitsAndBytesConfig
from datasets import load_dataset

def train(
    model_name="tiiuae/falcon-7b",
    hf_hub_token: str = None,
    wandb_project_name: str = 'assistant',
    data_dir: str = 'data',
):
    # no 4bit bs for now
    # # set up 4 bit config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # get model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_hub_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_hub_token, trust_remote_code=True)

    # get train dataloader
    if os.path.exists(os.path.join(data_dir, "train_dataloader.pt")):
        train_dataloader = torch.load(os.path.join(data_dir, "train_dataloader.pt"))
    else:
        raise RuntimeError("No train_dataloader.pt found in data_dir. Make sure to run data.py first.")

# eventually, maybe just want to make like a SFTTrainer class where you just call trainer.fit()
if __name__ == '__main__':
    fire.Fire(train)