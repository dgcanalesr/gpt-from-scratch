import hydra
import pickle
import requests
import torch

from omegaconf import DictConfig

# Data loading
def get_batch(
        split: str, 
        train_data: torch.tensor, 
        val_data: torch.tensor, 
        block_size: int, 
        batch_size: int
) -> (torch.tensor, torch.tensor):

    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        return
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

def encode(string: str, encoding_dict: dict) -> list[int]:
    return [encoding_dict[char] for char in string]

def decode(list_of_int: list[int], decoding_dict: dict) -> int:
    return "".join([decoding_dict[i] for i in list_of_int])

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train_val_split(cfg: DictConfig) -> None:

    # Download data
    response = requests.get(cfg.data.raw_data_url)
    if response.status_code == 200:
        with open(cfg.data.raw_data_path, 'w', encoding="utf-8") as file:
            file.write(response.text)

    # Read data
    with open(cfg.data.raw_data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Characters mapping for encoding-decoding
    chars = sorted(list(set(text)))
    encoding_dict = {ch:i for i,ch in enumerate(chars)}
    decoding_dict = {i:ch for i,ch in enumerate(chars)}

    # Save encoding-decoding dicts
    with open(cfg.data.encoding_dict_path, 'wb') as file: 
        pickle.dump(encoding_dict, file)

    with open(cfg.data.decoding_dict_path, 'wb') as file: 
        pickle.dump(decoding_dict, file)
    
    # Train and test split
    data = torch.tensor(encode(text, encoding_dict), dtype=torch.long)
    n = int(cfg.data.split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Save train-val data
    torch.save(train_data, cfg.data.train_data_path)
    torch.save(val_data, cfg.data.val_data_path)


if __name__ == "__main__":
    train_val_split()

