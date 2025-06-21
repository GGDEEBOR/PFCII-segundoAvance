import torch
import matplotlib.pyplot as plt
import numpy as np
from model.transformer_timeseries import TimeSeriesTransformer
from nbeats import NBeats
from data_utils import get_data_loaders

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modificacion de  la función de predicción para Transformer
def get_transformer_predictions(model, test_loader):
    model.eval()
    actuals, preds = [], []
    
    with torch.no_grad():
        for src, _, tgt_y in test_loader:
            # Seleccionar solo la primera característica para Transformer
            src_univariate = src[:, :, 0].unsqueeze(-1).to(device).float()
            
            # Asegurar dimensiones correctas [batch_size, seq_len, 1]
            output = model(src_univariate, src_univariate)
            
            preds.append(output.cpu().numpy())
            actuals.append(tgt_y.numpy()[:, :, 0])  # Solo primera característica
    
    return np.concatenate(actuals), np.concatenate(preds)

# Función de predicción para N-BEATS (sin cambios)
def get_nbeats_predictions(model, test_loader):
    model.eval()
    actuals, preds = [], []
    
    with torch.no_grad():
        for src, _, tgt_y in test_loader:
            src_flat = src.view(src.shape[0], -1).to(device).float()
            output = model(src_flat)
            preds.append(output.cpu().numpy())
            actuals.append(tgt_y.numpy()[:, :, 0])
    
    return np.concatenate(actuals), np.concatenate(preds)

# Cargar datos
_, _, test_loader, mean, std = get_data_loaders(
    batch_size=32,
    enc_seq_len=168,
    target_seq_len=48
)

# Cargar modelos
transformer = TimeSeriesTransformer(
    input_size=1,
    dec_seq_len=168,
    out_seq_len=48,
    dim_val=512,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=8
).to(device)
transformer.load_state_dict(torch.load("transformer_timeseries_model.pth", map_location=device))

nbeats = NBeats(
    input_dim=336,  # 168*2 (seq_len * num_features)
    horizon=48,
    num_blocks=3,
    theta_dim=64
).to(device)
nbeats.load_state_dict(torch.load("nbeats_model.pth", map_location=device))

# Obtener predicciones
y_true, y_transformer = get_transformer_predictions(transformer, test_loader)
y_true_nbeats, y_nbeats = get_nbeats_predictions(nbeats, test_loader)

# Visualización (5 primeras muestras)
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(5, 1, i+1)
    
    # Desnormalizar
    true = y_true[i] * std[0] + mean[0]
    pred_t = y_transformer[i] * std[0] + mean[0]
    pred_n = y_nbeats[i] * std[0] + mean[0]
    
    plt.plot(true, 'k--', label='Real')
    plt.plot(pred_t, 'r-', label='Transformer')
    plt.plot(pred_n, 'b-', label='N-BEATS')
    
    if i == 0:
        plt.legend()
    plt.ylabel(f'Muestra {i+1}')
    plt.grid(True)

plt.xlabel('Pasos temporales')
plt.suptitle('Comparación Transformer vs N-BEATS', y=1.02)
plt.tight_layout()
plt.savefig('comparacion_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Métricas
def print_metrics(true, pred, name):
    mse = np.mean((true - pred)**2)
    mae = np.mean(np.abs(true - pred))
    print(f"{name}: MSE={mse:.2f}, MAE={mae:.2f}")

print("\n Métricas Finales:")
print_metrics(y_true, y_transformer, "Transformer")
print_metrics(y_true_nbeats, y_nbeats, "N-BEATS")
