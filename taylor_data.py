import os
import re
import fire
import torch
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import AutoTokenizer, DefaultDataCollator
from typing import Any
from functools import partial
from data import tokenize_function, get_datasets

def process_cgy(example):
    return {
        "messages": [example['prompt'], example['response']]
    }

def process_cgy_qa(example):
    return {
        "messages": [example['human'], example['assistant']],
    }

def process_dolly(example):
    return {
        "messages": [example['instruction'], example['response']]
    }

def get_tailored_datasets(): 
    result = {}
    generic_datasets = get_datasets(["guanaco-filter", "lima-filter", "taylor-dialogues"], train=True)
    generic_concat = concatenate_datasets([v for _, v in generic_datasets.items()])

    tailored1 = load_dataset("andersonbcdefg/cognigy-finetune", split="train").map(
        process_cgy, remove_columns=["prompt", "response"]
    )
    tailored2 = load_dataset("andersonbcdefg/cognigy-qa", split="train").map(
        process_cgy_qa, remove_columns=["human", "assistant"]
    )
    tailored_concat = concatenate_datasets([tailored1, tailored2])
    
    eval_subsample = 10 
    tailored_eval = tailored_concat.filter(
        lambda example, idx: idx % eval_subsample == 0, with_indices=True
    )
    tailored_train = tailored_concat.filter(
        lambda example, idx: idx % eval_subsample != 0, with_indices=True
    )

    probabilities = [0.6, 0.4]
    train_dataset = interleave_datasets(
        [generic_concat, tailored_train], probabilities=probabilities, seed=42, stopping_strategy="all_exhausted"
    )
    
    result["eval"] = tailored_eval
    result["train"] = train_dataset
    
    for key, dataset in result.items():
        print(f"Dataset {key} has {len(dataset)} examples.")
    return result

def to_dataloader(dataset, microbatch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=DefaultDataCollator(),
        num_workers=0,
    )

def prepare_data(
    tokenizer_name="tiiuae/falcon-7b",
    train_microbatch_size: int = 4,
    eval_microbatch_size: int = 16,
    data_dir: str = 'data',
):  
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    datasets_dict = get_tailored_datasets()
    tokenized = {}

    # tokenize all datasets
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    for key, dataset in datasets_dict.items():
        print(f"Tokenizing {key}...")
        tokenized[key] = dataset.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )
    print("=== BUILDING TRAIN DATALOADER ===")
    # save train dataloader
    print("Saving train dataloader...")
    dataloader = torch.utils.data.DataLoader(
        tokenized['train'],
        batch_size=train_microbatch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=DefaultDataCollator(),
        num_workers=0,
    )
    torch.save({"datasets": ["generic", "cognigy"], "dataloader": dataloader}, os.path.join(data_dir, "train_dataloader.pt"))

    # get eval datasets
    print("=== BUILDING EVAL DATALOADERS ===")
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized["eval"],
        batch_size=eval_microbatch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=DefaultDataCollator(),
        num_workers=0,
    )
    
    # save eval dataloaders
    print("Saving eval dataloaders...")
    torch.save({
        "eval": eval_dataloader,
    }, os.path.join(data_dir, "eval_dataloaders.pt"))

if __name__ == '__main__':
    fire.Fire(prepare_data)