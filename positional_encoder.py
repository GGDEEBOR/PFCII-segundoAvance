import torch
import torch.nn as nn
import math
from torch import Tensor

class PositionalEncoder(nn.Module):
    """
    Positional encoding layer injecting positional info into embeddings.
    """

    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = False  # debe ser False para coincidir con checkpoint
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))


        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
