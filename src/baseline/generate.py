import hydra
import pickle
import torch 

from omegaconf import DictConfig
from src.utils.load_data import decode


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def generate_text(cfg: DictConfig) -> None:

    # Load model
    model = torch.load(cfg.models.baseline_path)

    # Load decoding dict
    with open(cfg.data.decoding_dict_path, "rb") as file:
        decoding_dict = pickle.load(file)

    # Generate from the model
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated_raw = model.generate(context, cfg.generate.max_new_tokens)
    generated_raw = generated_raw[0].tolist()
    generated_text = decode(generated_raw, decoding_dict)
    print(generated_text)

    # Save text to outputs
    with open(cfg.generate.baseline_output, "w") as file:
        file.write(generated_text)


if __name__ == "__main__":
    generate_text()