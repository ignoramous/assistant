import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def train(
    hf_hub_token: str = None,
    wandb_project_name: str = 'assistant',
    data_dir: str = 'data',
):
    # get model
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderplus", use_auth_token=hf_hub_token)
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderplus", use_auth_token=hf_hub_token)

    # get train dataloader
    if os.path.exists(os.path.join(data_dir, "train_dataloader.pt")):
        train_dataloader = torch.load(os.path.join(data_dir, "train_dataloader.pt"))
    else:
        raise RuntimeError("No train_dataloader.pt found in data_dir. Make sure to run data.py first.")

# eventually, maybe just want to make like a SFTTrainer class where you just call trainer.fit()
if __name__ == '__main__':
    fire.Fire(train)