import torch
from torch.utils.data import Dataset
from typing import Tuple

class TransformerDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        indices: list,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int
    ) -> None:
        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_idx, end_idx = self.indices[index]
        sequence = self.data[start_idx:end_idx]
        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
        )
        return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(sequence) == enc_seq_len + target_seq_len, \
            f"Sequence length {len(sequence)} != enc_len+target_len {enc_seq_len+target_seq_len}"

        src = sequence[:enc_seq_len]
        trg = sequence[enc_seq_len - 1 : enc_seq_len - 1 + target_seq_len]
        assert len(trg) == target_seq_len
        trg_y = sequence[-target_seq_len:]
        assert len(trg_y) == target_seq_len

        return src, trg, trg_y
