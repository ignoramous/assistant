import os
import re
import fire
import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, DefaultDataCollator
from typing import Any
from functools import partial

def process_cgy(example):
    return {
        "messages": [example['prompt'], example['response']]
    }

def process_dolly(example):
    return {
        "messages": [example['instruction'], example['response']]
    }

# this tokenizes a single conversation, with no padding or truncation.
# also useful for inference/chat.
def tokenize_conversation(
    conversation: list[str],
    tokenizer: Any,
    human_tokens: list[int] = [6], # falcon >>QUESTION<< token
    assistant_tokens: list[int] = [5], # falcon >>ANSWER<< token
):
    input_ids = [tokenizer.eos_token_id, *human_tokens]
    targets = [-100] * len(input_ids) # we will roll targets over by 1 later
    attention_mask = [1] * len(input_ids)
    for idx, message in enumerate(conversation):
        tokenized = tokenizer(message, add_special_tokens=False, padding=False, truncation=False)
        if idx % 2 == 0:
            # user message ends by prompt for assistant reply. don't calculate loss on user messages.
            input_ids.extend([*tokenized.input_ids, *assistant_tokens])
            targets.extend([-100] * (len(tokenized.input_ids) + len(assistant_tokens)))
            attention_mask.extend([1] * (len(tokenized.input_ids) + len(assistant_tokens)))
        else:
            # assistant message ends by prompt for user reply. calculate loss on assistant messages,
            # including the 'human tokens', since this is how we know the assistant is finished.
            input_ids.extend([*tokenized.input_ids, *human_tokens])
            targets.extend([*tokenized.input_ids, *human_tokens])
            attention_mask.extend([1] * (len(tokenized.input_ids) + len(human_tokens)))

    return input_ids, targets, attention_mask

def tokenize_function(
    examples: dict, 
    tokenizer: Any, 
    filter_max_prompt_length: int = 768, 
    seq_len: int = 1024,
    human_tokens: list[int] = [6], # falcon >>QUESTION<< token
    assistant_tokens: list[int] = [5], # falcon >>ANSWER<< token
):
    all_input_ids = []
    all_targets = []
    all_attention_masks = []
    
    for conversation in examples["messages"]:
        # if len(conversation) % 2 != 0:
        #     conversation = conversation[:-1]
        
        # if the initial prompt is too long, skip it
        if len(tokenizer(conversation[0]).input_ids) > filter_max_prompt_length:
            continue

        else:
            input_ids, targets, attention_mask = tokenize_conversation(
                conversation,
                tokenizer,
                human_tokens=human_tokens,
                assistant_tokens=assistant_tokens,
            )

        # assert shapes match
        assert len(input_ids) == len(attention_mask), "you done fucked up, cowboy"
        assert len(input_ids) == len(targets), "you done fucked up, cowboy 2"

        # make sure that the input_ids end with the human token or assistant token
        assert input_ids[-1] == human_tokens[-1] or input_ids[-1] == assistant_tokens[-1]

        # handle padding, and truncation
        pad_token = tokenizer.pad_token_id
        if pad_token is None: pad_token = tokenizer.eos_token_id

        if len(input_ids) == seq_len:
            targets = targets[1:] + [-100]
        elif len(input_ids) < seq_len:
            input_ids.extend([pad_token] * (seq_len - len(input_ids)))
            targets.extend([-100] * (seq_len - len(targets) + 1))
            targets = targets[1:]
            attention_mask.extend([0] * (seq_len - len(attention_mask)))
        elif len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            targets = targets[1:seq_len+1]
            attention_mask = attention_mask[:seq_len]
        
        # add to list of all input ids/attention masks
        assert len(input_ids) == seq_len, "you done fucked up pad/trunc with targets"
        assert len(input_ids) == len(targets), "you done fucked up padding and trunc with targets"
        assert len(input_ids) == len(attention_mask), "you done fucked up pad/trunc with attn mask"
        all_input_ids.append(input_ids)
        all_targets.append(targets)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "targets": all_targets,
        "attention_mask": all_attention_masks,
    }


def get_datasets():  
    result = {}

    dolly_1k = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42).flatten_indices().filter(
        lambda x: x['context'] == ""
    ).select(range(1000))
    dolly_1k = dolly_1k.map(process_dolly, remove_columns=dolly_1k.column_names)

    tailored = load_dataset("andersonbcdefg/cognigy-finetune", split="train").shuffle(seed=42).flatten_indices()
    tailored = tailored.map(process_cgy, remove_columns=tailored.column_names)
    eval_size = len(tailored) // 10
    tailored_eval = tailored.select(range(eval_size))
    tailored_train = tailored.select(range(eval_size, len(tailored)))
    
    result["dolly"] = dolly_1k
    result["train"] = tailored_train
    result["eval"] = tailored_eval
    
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

    datasets_dict = get_datasets()
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
    # interleave train datasets
    print("Interleaving datasets (this may take a while)...")
    lengths = [len(tokenized["dolly"]), len(tokenized["train"])]
    probabilities = [length / sum(lengths) for length in lengths]
    interleaved =  interleave_datasets([tokenized["dolly"], tokenized["train"]], probabilities=probabilities, seed=42)

    # save train dataloader
    print("Saving train dataloader...")
    dataloader = torch.utils.data.DataLoader(
        interleaved,
        batch_size=train_microbatch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=DefaultDataCollator(),
        num_workers=0,
    )
    torch.save({"datasets": ["dolly", "cgy"], "dataloader": dataloader}, os.path.join(data_dir, "train_dataloader.pt"))

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