import torch
from torch.utils.data import DataLoader
from nbeats import NBeats
from data_utils import get_data_loaders
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def train_nbeats(data_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuración
    seq_len = 168
    num_features = 2
    input_dim = seq_len * num_features  # 168 * 2 = 336
    
    model = NBeats(
        input_dim=input_dim,
        horizon=48,
        num_blocks=3,
        theta_dim=64
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for src, _, tgt_y in progress_bar:
            # Aplanar las características
            batch_size = src.shape[0]
            src_flat = src.view(batch_size, -1)  # [32, 336]
            tgt_y = tgt_y[:, :, 0]
            
            src_flat = src_flat.to(device)
            tgt_y = tgt_y.to(device)
            
            optimizer.zero_grad()
            output = model(src_flat)
            loss = criterion(output, tgt_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')
    
    return model

if __name__ == '__main__':
    # Configuración
    batch_size = 32
    epochs = 15
    learning_rate = 1e-3
    
    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader, test_loader, mean, std = get_data_loaders(
        batch_size=batch_size,
        enc_seq_len=168,
        target_seq_len=48
    )
    
    # Verificar dimensiones
    sample_batch = next(iter(train_loader))
    print(f"\nDimensiones del batch:")
    print(f"Input: {sample_batch[0].shape}")
    print(f"Target: {sample_batch[2].shape}")
    
    # Entrenar
    print("\nEntrenando modelo N-BEATS...")
    model = train_nbeats(train_loader, epochs=epochs, lr=learning_rate)
    
    # Guardar
    torch.save(model.state_dict(), "nbeats_model.pth")
    print("\n Modelo guardado como 'nbeats_model.pth'")