import fire
import torch
import numpy as np
import tqdm

@torch.no_grad()
def evaluate(accelerator, model, eval_dataloaders):
    metrics = {}
    for key in eval_dataloaders:
        print(f"Evaluating on {key}...")
        loader = eval_dataloaders[key]
        losses = []
        for batch in tqdm.tqdm(loader):
            logits = model(input_ids=batch['input_ids']).logits
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), batch["targets"].view(-1), reduction="none"
            )
            losses.extend(accelerator.gather(losses).cpu().numpy())
        losses = losses[:len(loader.dataset)]
        metrics[f"{key}_loss"] = np.mean(losses)

    return metrics

def run_evals():
    pass


if __name__ == "__main__":
    fire.Fire(run_evals)