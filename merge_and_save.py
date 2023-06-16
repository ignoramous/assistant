import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "tiiuae/falcon-7b"
CHECKPOINT_PATH = "/Users/ben/Downloads/taylor assistant checkpoints/checkpoints-qlora-lima-oasst-5ep/model_4"

model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_config.use_cache = False
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=model_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    model,
    CHECKPOINT_PATH,
)

model = model.merge_and_unload()
# model.save_pretrained("./merged")
model.push_to_hub("my_cool_new_model")