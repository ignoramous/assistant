import os
import fire
import torch
import wandb
from typing import Any
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data import tokenize_function
    
@torch.no_grad()
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
    input_ids = tokenized["input_ids"][0][:7]
    
    model.eval()
    model.cuda()
    for i in range(100):
        # decode next token
        input_tensor = torch.tensor(input_ids).view(1, -1).cuda()
        logits = model(input_tensor).logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        input_ids.append(next_token)
        if next_token in human_tokens:
            break

    # decode response
    response = tokenizer.decode(input_ids[7:])
    print(response)

def chat(
    model_name: str,
    checkpoint_path: str,
    lora: bool = False,
    quantize: str = None,
):
    # validate args
    if quantize not in [None, "4bit", "8bit"]:
        raise ValueError("quantize must be one of [None, '4bit', '8bit']")
    
    if not lora and quantize:
        raise ValueError("Full model checkpoints aren't quantized so quantizing is not supported right now.")

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    
    message = "Hello, world!"
    infer([message], model, tokenizer)

if __name__ == "__main__":
    fire.Fire(chat)