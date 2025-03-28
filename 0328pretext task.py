import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_PATH = "preprocessed_data_segments/X_segments.npy"  # shape: (N, B, C, T)
y_PATH = "preprocessed_data_segments/y_segments.npy"
BATCH_SIZE = 16
EPOCHS = 30
FEATURE_DIM = 64
SPECTRAL_DIM = 20  # 예시 spectral feature dimension

# ==================== LOAD DATA ====================
print("\n📥 Loading EEG data...")
X = np.load(X_PATH, allow_pickle=True)
y = np.load(y_PATH, allow_pickle=True)

N, B, C, T = X.shape
X_reshaped = X.reshape(N, B * C, T)
X_reshaped = (X_reshaped - X_reshaped.mean()) / X_reshaped.std()

# Spectral feature 예시: log-variance per band
spectral_features = []
for trial in X:
    bands = []
    for band_data in trial:
        log_var = np.log(np.var(band_data, axis=1) + 1e-6)  # (C,)
        bands.extend(log_var)
    spectral_features.append(bands)  # 총 B*C 개의 feature
spectral_features = np.array(spectral_features)
scaler = StandardScaler()
spectral_features = scaler.fit_transform(spectral_features)

X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
spec_tensor = torch.tensor(spectral_features, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"✅ Spatial input shape: {X_tensor.shape}, Spectral shape: {spec_tensor.shape}")

# ==================== DATASET ====================
class SSLEEGDataset(Dataset):
    def __init__(self, spatial_data, spectral_data):
        self.spatial_data = spatial_data
        self.spectral_data = spectral_data
    def __len__(self):
        return len(self.spatial_data)
    def __getitem__(self, idx):
        return self.spatial_data[idx], self.spectral_data[idx]

train_dataset = SSLEEGDataset(X_tensor, spec_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==================== MODEL ====================
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class SpectralEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class SSLDualBranch(nn.Module):
    def __init__(self, in_channels, spectral_dim, feature_dim=64):
        super().__init__()
        self.spatial_encoder = SpatioTemporalEncoder(in_channels, feature_dim)
        self.spectral_encoder = SpectralEncoder(spectral_dim, feature_dim)

    def forward(self, spatial_input, spectral_input):
        z_spatial = self.spatial_encoder(spatial_input)
        z_spectral = self.spectral_encoder(spectral_input)
        return z_spatial, z_spectral

# ==================== CONTRASTIVE LOSS ====================
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    batch_size = z1.size(0)

    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=1)
    positives = torch.cat([positives, positives], dim=0)
    logits = similarity_matrix / temperature

    loss = F.cross_entropy(logits, labels)
    return loss

# ==================== TRAINING ====================
model = SSLDualBranch(in_channels=B*C, spectral_dim=spectral_features.shape[1], feature_dim=FEATURE_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n🚀 Starting SSL training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for spatial_batch, spectral_batch in train_loader:
        spatial_batch = spatial_batch.to(DEVICE)
        spectral_batch = spectral_batch.to(DEVICE)

        z1, z2 = model(spatial_batch, spectral_batch)
        loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📉 Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("\n✅ SSL training complete.")
