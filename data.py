import os
import re
import fire
import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, DefaultDataCollator
from typing import Any
from functools import partial

# For this training pipeline, preprocessing should simply return a list of strings.
# The first string is assumed to be from the user, the second from the assistant,
# and so on. If messages don't alternate, you've gotta process them so they do.
def process_lima(example):
    return {
        "messages": example['conversations']
    }

def process_oasst_guanaco(example):
    text = example['text']
    messages = re.split(r'### Human:|### Assistant:', text)
    messages = [m.strip() for m in messages if m.strip() != '']
    return {
        "messages": messages
    }

def process_dolly(example):
    return {
        "messages": [example['instruction'], example['response']]
    }

def process_alpaca(example):
    return {
        "messages": [example['instruction'], example['output']]
    }

TRAIN_REGISTRY = {
    "lima": {
        "hub_url": "GAIR/lima",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_lima,
    },
    "oasst_guanaco": {
        "hub_url": "timdettmers/openassistant-guanaco",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_oasst_guanaco,
    },
    "hh_dialogue": {
        "hub_url": "andersonbcdefg/hh-dialogue",
        "split": "train",
        "filter_fn": None,
        "processing_fn": None,
    }
}

EVAL_REGISTRY = {
    "dolly": {
        "hub_url": "databricks/databricks-dolly-15k",
        "split": "train",
        "filter_fn": lambda example: example['context'] == "",
        "processing_fn": process_dolly,
    },
    "oasst_guanaco": {
        "hub_url": "timdettmers/openassistant-guanaco",
        "split": "test",
        "filter_fn": None,
        "processing_fn": process_oasst_guanaco,
    },
    "alpaca": {
        "hub_url": "yahma/alpaca-cleaned",
        "split": "train",
        "filter_fn": lambda example: example['input'] == "",
        "processing_fn": process_alpaca,
    }
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


def get_datasets(
    datasets="all",
    train=True
):  
    registry = TRAIN_REGISTRY if train else EVAL_REGISTRY
    
    if datasets == "all":
        datasets = list(registry.keys())

    print(f"Getting datasets: {datasets}")

    result = {}
    
    # process all datasets to have the same 2 columns: prompt, response
    for ds_name in datasets:
        if "subset" in registry[ds_name]:
            subset = registry[ds_name]["subset"]
            loaded = load_dataset(registry[ds_name]["hub_url"], subset, split=registry[ds_name]["split"])
        else:
            loaded = load_dataset(registry[ds_name]["hub_url"], split=registry[ds_name]["split"])
        if registry[ds_name]["filter_fn"] is not None:
            loaded = loaded.filter(registry[ds_name]["filter_fn"])
        if registry[ds_name]["processing_fn"] is not None:
            print(loaded[0].keys())
            loaded = loaded.map(registry[ds_name]["processing_fn"], remove_columns=loaded.column_names)
        
        # keep eval datasets small
        if not train and len(loaded) > 1000:
            loaded = loaded.shuffle(seed=42).flatten_indices().select(range(1000))
        result[ds_name] = loaded
    
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
    train_datasets="all",
    eval_datasets="all",
    tokenizer_name="tiiuae/falcon-7b",
    train_microbatch_size: int = 4,
    eval_microbatch_size: int = 16,
    hf_hub_token: str = None,
    data_dir: str = 'data',
):  
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

     # get train datasets
    print("=== BUILDING TRAIN DATALOADERS ===")
    train_datasets_dict = get_datasets(
        datasets=train_datasets,
        train=True
    )

    train_tokenized = {}

    # tokenize all datasets
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=hf_hub_token, trust_remote_code=True)
    
    for key, dataset in train_datasets_dict.items():
        print(f"Tokenizing {key}...")
        train_tokenized[key] = dataset.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )

    # interleave datasets
    if len(train_tokenized) > 1:
        print("Interleaving datasets (this may take a while)...")
        lengths = [len(dataset) for dataset in train_tokenized.values()]
        probabilities = [length / sum(lengths) for length in lengths]
        print([f"{n}: {l}" for n, l in zip(train_tokenized.keys(),lengths)])
        interleaved =  interleave_datasets([d for _, d in train_tokenized.items()], probabilities=probabilities, seed=42)
    else:
        interleaved = train_tokenized[list(train_tokenized.keys())[0]]

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
    torch.save({"datasets": train_datasets, "dataloader": dataloader}, os.path.join(data_dir, "train_dataloader.pt"))

    # get eval datasets
    print("=== BUILDING EVAL DATALOADERS ===")
    eval_datasets_dict = get_datasets(
        datasets=eval_datasets,
        train=False
    )

    # tokenize all datasets
    eval_tokenized = {}
    for key, dataset in eval_datasets_dict.items():
        print(f"Tokenizing {key}...")
        eval_tokenized[key] = dataset.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )
    
    # save eval dataloaders
    print("Saving eval dataloaders...")
    torch.save({
        k: to_dataloader(v, eval_microbatch_size, shuffle=False) for k, v in eval_tokenized.items()
    }, os.path.join(data_dir, "eval_dataloaders.pt"))

if __name__ == '__main__':
    fire.Fire(prepare_data)