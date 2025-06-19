from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader, random_split
import utils
from dataset import TransformerDataset 

def get_data_loaders(
    data_dir: Union[str, Path] = "data",
    batch_size: int = 32,
    enc_seq_len: int = 48,
    dec_seq_len: int = 48,
    target_seq_len: int = 48,
    step_size: int = 1,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
):
    df = utils.read_data(data_dir)
    df_norm, mean, std = utils.normalize_data(df)
    data_tensor = torch.tensor(df_norm.values, dtype=torch.float32)
    indices = utils.get_indices_entire_sequence(df_norm, enc_seq_len + target_seq_len, step_size)
    dataset = TransformerDataset(data_tensor, indices, enc_seq_len, dec_seq_len, target_seq_len)

    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean, std
