import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Any, Callable
from functools import partial

# For this training pipeline, preprocessing should simply return a list of strings.
# The first string is assumed to be from the user, the second from the assistant,
# and so on. If messages don't alternate, you've gotta process them so they do.
def process_lima(example):
    return {
        "messages": example['conversations']
    }

TRAIN_REGISTRY = {
    "lima": {
        "hub_url": "GAIR/lima",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_lima,
    }
}

def tokenize_function(
    examples: dict, 
    tokenizer: Any, 
    filter_max_prompt_length: int = 768, 
    seq_len: int = 1024,
    human_tokens: list[int] = [6], # falcon >>QUESTION<< token
    assistant_tokens: list[int] = [5] # falcon >>ANSWER<< token
):
    all_input_ids = []
    all_attention_masks = []
    
    for conversation in examples["messages"]:
        if len(conversation) % 2 != 0:
            conversation = conversation[:-1]
        
        # if the initial prompt is too long, skip it
        if len(tokenizer(conversation[0]).input_ids) > filter_max_prompt_length:
            continue

        # otherwise, tokenize each message in the conversation.
        # loss shouldn't be calculated on user messages, this is reflected in the attention mask
        input_ids = [tokenizer.eos_token_id, *human_tokens]
        attention_mask = [0] * len(input_ids)
        for idx, message in enumerate(conversation):
            tokenized = tokenizer(message, add_special_tokens=False, padding=False, truncation=False)
            if idx % 2 == 0:
                # user message ends by prompt for assistant reply. don't calculate loss on user messages.
                input_ids.extend([tokenized.input_ids, *assistant_tokens])
                attention_mask.extend([0] * (len(tokenized.input_ids) + len(assistant_tokens)))
            else:
                # assistant message ends by prompt for user reply. calculate loss on assistant messages,
                # including the 'human tokens', since this is how we know the assistant is finished.
                input_ids.extend([tokenized.input_ids, *human_tokens])
                attention_mask.extend([1] * (len(tokenized.input_ids) + len(human_tokens)))

        # handle padding, and truncation
        if len(input_ids) < seq_len:
            input_ids.extend([tokenizer.pad_token_id] * (seq_len - len(input_ids)))
            attention_mask.extend([0] * (seq_len - len(attention_mask)))
        elif len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            attention_mask = attention_mask[:seq_len]
        
        # add to list of all input ids/attention masks
        print(len(input_ids), len(attention_mask))
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
    }


# def get_datasets(
#         datasets=["lima"], 
#         add_human_assistant_labels=[],
#         min_length_in_tokens=None, 
#         max_length_in_tokens=None, 
#         tokenizer=None,
#         train=True
# ):  
#     registry = TRAIN_REGISTRY if train else None
#     if tokenizer is None:
#         if min_length_in_tokens is not None or max_length_in_tokens is not None:
#             raise ValueError("Cannot specify min_length_in_tokens or max_length_in_tokens without specifying tokenizer.")
    
#     if datasets == "all":
#         datasets = list(registry.keys())

#     result = {}
    
#     # process all datasets to have the same 2 columns: prompt, response
#     for ds_name in datasets:
#         if "subset" in registry[ds_name]:
#             subset = registry[ds_name]["subset"]
#             loaded = load_dataset(registry[ds_name]["hub_url"], subset, split=registry[ds_name]["split"])
#         else:
#             loaded = load_dataset(registry[ds_name]["hub_url"], split=registry[ds_name]["split"])
#         if registry[ds_name]["filter_fn"] is not None:
#             loaded = loaded.filter(registry[ds_name]["filter_fn"])
#         if registry[ds_name]["processing_fn"] is not None:
#             print(loaded[0].keys())
#             loaded = loaded.map(registry[ds_name]["processing_fn"], remove_columns=loaded.column_names).filter(
#                 lambda example: example["preferred"] != "" and example["dispreferred"] != ""
#             )
#         result[ds_name] = loaded

#     # add human and assistant to prompt if applicable
#     for key in result.keys():
#         if key in add_human_assistant_labels:
#             result[key] = result[key].map(
#                 lambda example: {"prompt": add_human_and_assistant_to_prompt(example['prompt'])}
#             )

#     # filter all datasets for length
#     if min_length_in_tokens is not None or max_length_in_tokens is not None:
#         for key in result.keys():
#             result[key] = result[key].map(
#                 lambda example: {"length": len(tokenizer(example['prompt']).input_ids) + \
#                                 max(len(tokenizer(example['preferred']).input_ids), 
#                                     len(tokenizer(example['dispreferred']).input_ids))},
#             )
#             if min_length_in_tokens is not None:
#                 result[key] = result[key].filter(lambda example: example['length'] >= min_length_in_tokens)
#             if max_length_in_tokens is not None:
#                 result[key] = result[key].filter(lambda example: example['length'] <= max_length_in_tokens)
    
#     for key, dataset in result.items():
#         print(f"Dataset {key} has {len(dataset)} examples.")
#     return result

# def get_train_dataloader(
#         datasets,
#         add_human_assistant_labels,
#         batch_size, 
#         tokenizer,
#         filter_min_length_in_tokens=None,
#         filter_max_length_in_tokens=None,
#         seq_len=1024,
# ):
#     if tokenizer is None:
#         raise ValueError("Must specify tokenizer.")

#     datasets = get_datasets(
#         datasets,
#         add_human_assistant_labels,
#         min_length_in_tokens=filter_min_length_in_tokens, 
#         max_length_in_tokens=filter_max_length_in_tokens, 
#         tokenizer=tokenizer,
#         train=True
#     )
    
#     if len(datasets) > 1:
#         print("Interleaving datasets (this may take a while)...")
#         lengths = [len(dataset) for dataset in datasets.values()]
#         probabilities = [length / sum(lengths) for length in lengths]
#         print([f"{n}: {l}" for n, l in zip(datasets.keys(),lengths)])
#         interleaved =  interleave_datasets([d for _, d in datasets.items()], probabilities=probabilities, seed=42)
#     else:
#         interleaved = datasets[list(datasets.keys())[0]]
    
#     print("Tokenizing...")
#     tokenized = interleaved.map(
#         partial(tokenize_function, tokenizer=tokenizer, max_len=seq_len),
#         batched=True,
#         batch_size=1000,
#         remove_columns=interleaved.column_names,
#     )

#     dataloader = torch.utils.data.DataLoader(
#         tokenized,
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         collate_fn=DefaultDataCollator(),
#         num_workers=4,
#     )
    
#     return dataloader

def prepare_data(
    hf_hub_token: str = None,
    data_dir: str = 'data',
):
     # get LIMA dataset
    dataset = load_dataset("GAIR/lima", split="train", use_auth_token=hf_hub_token)

    # process LIMA dataset
    dataset = dataset.map(process_lima, remove_columns=dataset.column_names)

    # tokenize LIMA dataset
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_auth_token=hf_hub_token, trust_remote_code=True)
    tokenized = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
    )

    print(tokenized[0]['input_ids'].shape, tokenized[0]['attention_mask'].shape)



if __name__ == '__main__':
    fire.Fire(prepare_data)