import re

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
    },
    "dolly-filter": {
        "hub_url": "andersonbcdefg/dolly-ai-filtered",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_dolly,
    },
    "guanaco-filter": {
        "hub_url": "andersonbcdefg/guanaco-ai-filtered",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_oasst_guanaco,
    },
    "lima-filter": {
        "hub_url": "andersonbcdefg/lima-ai-filtered",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_lima,
    },
    "taylor-dialogues": {
        "hub_url": "andersonbcdefg/taylor-dialogues",
        "split": "train",
        "filter_fn": None,
        "processing_fn": None,
    }
}

# eventually, include company-specific datasets here
FINETUNE_REGISTRY = {}

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