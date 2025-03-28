import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ===== LOAD PRETRAINED ENCODER STRUCTURE =====
import torch.nn as nn

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=64):
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

# ===== CONFIG =====
X_PATH = "preprocessed_data_segments/X_segments.npy"
y_PATH = "preprocessed_data_segments/y_segments.npy"
MODEL_PATH = None  # (선택) 저장된 encoder가 있다면 여기 경로
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD DATA =====
X = np.load(X_PATH)
y = np.load(y_PATH)
N, B, C, T = X.shape
X_reshaped = X.reshape(N, B * C, T)
X_reshaped = (X_reshaped - X_reshaped.mean()) / X_reshaped.std()

X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ===== LOAD ENCODER =====
encoder = SpatioTemporalEncoder(in_channels=B*C, feature_dim=64).to(DEVICE)
if MODEL_PATH:
    encoder.load_state_dict(torch.load(MODEL_PATH))
encoder.eval()

# ===== EXTRACT EMBEDDINGS =====
embeddings = []
labels = []
loader = DataLoader(list(zip(X_tensor, y_tensor)), batch_size=64)

with torch.no_grad():
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        z = encoder(X_batch)
        embeddings.append(z.cpu().numpy())
        labels.append(y_batch.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

# ===== t-SNE =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_proj = tsne.fit_transform(X_scaled)

# ===== Plot =====
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    idx = labels == label
    plt.scatter(X_proj[idx, 0], X_proj[idx, 1], label=f"Class {label}", alpha=0.6)
plt.legend()
plt.title("t-SNE of SSL Encoder Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_ssl_embedding.png")
plt.show()
