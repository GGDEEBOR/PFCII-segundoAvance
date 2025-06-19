import torch
import matplotlib.pyplot as plt
import numpy as np
from model.transformer_timeseries import TimeSeriesTransformer
from data_utils import get_data_loaders

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = TimeSeriesTransformer(
    input_size=1,
    dec_seq_len=168,
    out_seq_len=48,
    dim_val=512,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=8,
    dropout_encoder=0.1,
    dropout_decoder=0.1,
    dropout_pos_enc=0.1,
    batch_first=True
).to(device)


# Cargar pesos entrenados
model.load_state_dict(torch.load("transformer_timeseries_model.pth", map_location=device))
model.eval()

# Obtener datos
train_loader, val_loader, test_loader, mean, std = get_data_loaders(
    data_dir="data",
    batch_size=1,
    enc_seq_len=168,
    dec_seq_len=48,
    target_seq_len=48,
    step_size=1
)

# Función de desnormalización
def denormalize(data, mean, std):
    if hasattr(mean, '__len__'):
        mean = mean[0]
    if hasattr(std, '__len__'):
        std = std[0]
    return data * std + mean


# Evaluación
with torch.no_grad():
    src, tgt, tgt_y = next(iter(val_loader))
    src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

    src = src[:, :, :1]
    tgt = tgt[:, :, :1]
    tgt_y = tgt_y[:, :, :1]


    # Verificación de shapes
    print("Shapes de entrada:")
    print(f"src: {src.shape} (debería ser [1, 168, 1])")
    print(f"tgt: {tgt.shape} (debería ser [1, 48, 1])")
    print(f"tgt_y: {tgt_y.shape} (debería ser [1, 48, 1])")

    # Predicción
    output = model(src, tgt)
    print(f"\nShape de salida: {output.shape} (debería ser [1, 48])")

    # Desnormalización
    src = denormalize(src.squeeze().cpu().numpy(), mean, std)
    tgt_y = denormalize(tgt_y.squeeze().cpu().numpy(), mean, std)
    output = denormalize(output.squeeze().cpu().numpy(), mean, std)

# Visualización
plt.figure(figsize=(14, 7))
plt.plot(src, label='Histórico', color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(range(len(src), len(src)+len(tgt_y)), tgt_y, 
         label='Valores Reales', color='#2ca02c', linewidth=3)
plt.plot(range(len(src), len(src)+len(output)), output, 
         label='Predicciones', color='#ff7f0e', linestyle='--', linewidth=3)
plt.axvline(x=len(src), color='#d62728', linestyle=':', linewidth=2, label='Inicio Predicción')
plt.title("Comparación: Predicciones vs Valores Reales", fontsize=14, pad=20)
plt.xlabel("Pasos Temporales", fontsize=12)
plt.ylabel("Valor Normalizado", fontsize=12)
plt.legend(fontsize=12, framealpha=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Métricas
mse = np.mean((tgt_y - output)**2)
mae = np.mean(np.abs(tgt_y - output))
print("\nMétricas de rendimiento:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# Guardar y mostrar
plt.savefig('resultados_prediccion.png', dpi=300, bbox_inches='tight')
plt.show()
