import torch
from torch import nn, Tensor
import positional_encoder as pe

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        dec_seq_len: int,
        out_seq_len: int,
        dim_val: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        dropout_encoder: float = 0.1,
        dropout_decoder: float = 0.1,
        dropout_pos_enc: float = 0.1,
        batch_first: bool = True
    ):
        super().__init__()
        self.batch_first = batch_first

        # Capas de transformación de entrada
        self.encoder_input_layer = nn.Linear(input_size, dim_val)
        self.decoder_input_layer = nn.Linear(input_size, dim_val)

        # Codificación posicional
        self.positional_encoding_layer = pe.PositionalEncoder(
            dropout=dropout_pos_enc,
            max_seq_len=5000,
            d_model=dim_val,
            batch_first=batch_first
        )

        # Capas Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dropout=dropout_encoder,
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dropout=dropout_decoder,
            batch_first=batch_first
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Capa de salida
        self.linear_mapping = nn.Linear(dim_val, 1)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        # Procesamiento del encoder
        src_emb = self.encoder_input_layer(src)
        src_emb = self.positional_encoding_layer(src_emb)
        memory = self.encoder(src_emb, mask=src_mask)

        # Procesamiento del decoder
        tgt_emb = self.decoder_input_layer(tgt)
        tgt_emb = self.positional_encoding_layer(tgt_emb)
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # Salida
        output = self.linear_mapping(decoder_out)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output.squeeze(-1)
