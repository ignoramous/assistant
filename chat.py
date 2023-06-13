import os
import fire
import torch
import wandb
from typing import Any
from accelerate import Accelerator
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from data import tokenize_conversation


@torch.no_grad()
def infer(
    conversation: list[str],
    # accelerator: Accelerator,
    model: Any,
    tokenizer: Any,
    human_tokens: list[int] = [6],  # falcon >>QUESTION<< token
    assistant_tokens: list[int] = [5],  # falcon >>ANSWER<< token
):
    input_ids, _, _ = tokenize_conversation(
        conversation,
        tokenizer,
        human_tokens=human_tokens,
        assistant_tokens=assistant_tokens,
    )
    # find the first assistant token
    idx = 0
    for i, token in enumerate(input_ids):
        if token in assistant_tokens:
            idx = i
            break
    input_ids = input_ids[: idx + 1]
    # preview the inputs
    print(f"Generating completion for input: {tokenizer.decode(input_ids)}")

    generated_tokens = []
    model.eval()
    for i in range(100):
        # decode next token
        print("|", end="", flush=True)
        input_tensor = torch.tensor(input_ids).view(1, -1).cuda()
        next_token_logits = model(input_tensor).logits[:, -1, :]
        probs = torch.softmax(
            next_token_logits, dim=-1
        )  # Compute softmax probabilities
        next_token_index = torch.multinomial(
            probs, num_samples=1
        )  # Sample from the multinomial distribution
        next_token = next_token_index.item()
        input_ids.append(next_token)
        generated_tokens.append(next_token)
        if generated_tokens[-len(human_tokens) :] == human_tokens:
            break

    # decode response
    print(f"\nGenerated {len(generated_tokens)} tokens.\n\n")
    response = tokenizer.decode(generated_tokens)
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
        raise ValueError(
            "Full model checkpoints aren't quantized so quantizing is not supported right now."
        )

    # set up 4/8 bit config
    if quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=True)
    elif quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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
        trust_remote_code=True,
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

    while True:
        message = input("Enter a message: ")
        infer([message], model, tokenizer)


if __name__ == "__main__":
    fire.Fire(chat)
