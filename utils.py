import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import pandas as pd
from typing import Union

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
    stop_position = len(data) - 1
    subseq_first_idx = 0
    subseq_last_idx = window_size
    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
    return indices

def read_data(data_dir: Union[str, Path] = "data", timestamp_col_name: str = "timestamp") -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) != 1:
        raise ValueError("The data folder must contain exactly one CSV file.")
    df = pd.read_csv(csv_files[0], parse_dates=[timestamp_col_name])
    df.set_index(timestamp_col_name, inplace=True)
    df.sort_index(inplace=True)
    return df

def normalize_data(df: pd.DataFrame):
    mean = df.mean()
    std = df.std()
    df_norm = (df - mean) / std
    return df_norm, mean, std
