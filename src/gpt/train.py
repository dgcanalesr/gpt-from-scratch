import hydra
import torch

from omegaconf import DictConfig
from src.gpt.model import GPTModel
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
    model = GPTModel(
        vocab_size=cfg.train_gpt.vocab_size, 
        n_layer=cfg.train_gpt.n_layer,
        n_embd=cfg.train_gpt.n_embd, 
        n_head=cfg.train_gpt.n_head,
        block_size=cfg.train_gpt.block_size, 
        dropout=cfg.train_gpt.dropout,
        device=device
    ) 
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.train_gpt.learning_rate
    )

    # Loss estimation
    @torch.no_grad()
    def estimate_loss() -> dict:
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.train_gpt.eval_iters)
            for k in range(cfg.train_gpt.eval_iters):
                X, Y = get_batch(
                    split=split,
                    train_data=train_data,
                    val_data=val_data,
                    block_size=cfg.train_gpt.block_size,
                    batch_size=cfg.train_gpt.batch_size
                )
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training
    for iter in range(cfg.train_gpt.max_iters):

        # Evaluate loss on train and val sets
        if iter % cfg.train_gpt.eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: Train loss {losses['train']:.4f}, \
                Val loss {losses['val']:.4f}")
        
        # Data batch
        xb, yb = get_batch(
            split="train",
            train_data=train_data,
            val_data=val_data,
            block_size=cfg.train_gpt.block_size,
            batch_size=cfg.train_gpt.batch_size
        )
        xb, yb = xb.to(device), yb.to(device)

        # Loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save last model
    torch.save(model, cfg.models.gpt_path)


if __name__ == "__main__":
    train_model()
