import os
import fire
import torch
import wandb
from typing import Any
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from data import tokenize_function
    

def infer(
    conversation: list[str],
    model: Any,
    tokenizer: Any,
    seq_len: int = 1024,
    human_tokens: list[int] = [6], # falcon >>QUESTION<< token
    assistant_tokens: list[int] = [5] # falcon >>ANSWER<< token
):
    tokenize_fn_input = {"messages": [conversation]}
    tokenized = tokenize_function(tokenize_fn_input, tokenizer, seq_len=seq_len, human_tokens=human_tokens, assistant_tokens=assistant_tokens)
    input_ids = [tokenized["input_ids"][0][:7]]

    print(input_ids)
    print(tokenizer.decode(input_ids[0]))
    
    while True:
        # decode next token
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]

def chat():
    message = "Hello, world!"
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    infer([message], model, tokenizer)

if __name__ == "__main__":
    fire.Fire(chat)