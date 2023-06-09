import fire
from datasets import load_dataset

def prepare_data(
        
):
     # get LIMA dataset
    dataset = load_dataset("GAIR/lima", split="train")

if __name__ == '__main__':
    fire.Fire(prepare_data)