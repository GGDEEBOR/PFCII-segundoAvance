import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os

import pandas as pd
from dataset import TransformerDataset
from model.transformer_timeseries import TimeSeriesTransformer
import utils

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros
csv_folder = "data"
input_variables = ["FCR_N_PriceEUR"]
enc_seq_len = 168
target_seq_len = 48
window_size = enc_seq_len + target_seq_len
step_size = 1

batch_size = 32
epochs = 15
learning_rate = 1e-4

# 1. Lectura y preprocesamiento
df = utils.read_data(data_dir=csv_folder)
df = df[input_variables].ffill()
df_norm, mean, std = utils.normalize_data(df)

tensor_data = torch.tensor(df_norm.values, dtype=torch.float32)

# 2. Creación de  índices y dataset
indices = utils.get_indices_entire_sequence(df_norm, window_size, step_size)
dataset = TransformerDataset(tensor_data, indices, enc_seq_len, dec_seq_len=target_seq_len, target_seq_len=target_seq_len)

# 3. Split train/val
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# 4. Modelo, loss y optimizador
model = TimeSeriesTransformer(
    input_size = 1,
    dec_seq_len = enc_seq_len,
    out_seq_len = target_seq_len,
    dim_val = 512,
    n_encoder_layers = 4,
    n_decoder_layers = 4,
    n_heads = 8,
    batch_first = True
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Entrenamiento
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for src, tgt, tgt_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} — Training"):
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
        src_mask = utils.generate_square_subsequent_mask(enc_seq_len, enc_seq_len).to(device)
        tgt_mask = utils.generate_square_subsequent_mask(target_seq_len, target_seq_len).to(device)

        optimizer.zero_grad()
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.squeeze(-1)
        loss = criterion(output, tgt_y.transpose(0, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch:02d}/{epochs}] Train Loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for src, tgt, tgt_y in val_loader:
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            src_mask = utils.generate_square_subsequent_mask(enc_seq_len, enc_seq_len).to(device)
            tgt_mask = utils.generate_square_subsequent_mask(target_seq_len, target_seq_len).to(device)

            output= model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            output = output.squeeze(-1)
            val_loss = criterion(output, tgt_y.transpose(0, 1))
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"[Epoch {epoch:02d}/{epochs}] Validation Loss: {avg_val_loss:.4f}")

# 6. Guardado del modelo
torch.save(model.state_dict(), "transformer_timeseries_model.pth")
print("Modelo guardado en transformer_timeseries_model.pth")
