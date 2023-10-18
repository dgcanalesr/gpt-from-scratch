import hydra
import os
import torch

from omegaconf import DictConfig
from src.baseline.model import BigramLanguageModel
from src.utils.load_data import get_batch


# Seed and Device
torch.manual_seed(77)
device = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train_model(cfg: DictConfig) -> None:

    # Load data
    train_data = torch.load(cfg.data.train_data_path)
    val_data = torch.load(cfg.data.val_data_path)

    # Model
    model = BigramLanguageModel(cfg.train.vocab_size)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    # Loss estimation
    @torch.no_grad()
    def estimate_loss() -> dict:
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.train.eval_iters)
            for k in range(cfg.train.eval_iters):
                X, Y = get_batch(
                    split=split,
                    train_data=train_data,
                    val_data=val_data,
                    block_size=cfg.train.block_size,
                    batch_size=cfg.train.batch_size
                )
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training
    for iter in range(cfg.train.max_iters):

        # Evaluate loss on train and val sets
        if iter % cfg.train.eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: Train loss {losses['train']:.4f}, \
                Val loss {losses['val']:.4f}")
        
        # Data batch
        xb, yb = get_batch(
            split="train",
            train_data=train_data,
            val_data=val_data,
            block_size=cfg.train.block_size,
            batch_size=cfg.train.batch_size
        )
        xb, yb = xb.to(device), yb.to(device)

        # Loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save last model
    torch.save(model, cfg.models.baseline_path)


if __name__ == "__main__":
    train_model()
