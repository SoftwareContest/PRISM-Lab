import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ===== CONFIG =====
X_PATH = "preprocessed_data_segments/X_segments.npy"
y_PATH = "preprocessed_data_segments/y_segments.npy"
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001
N_SPLITS = 5
LABEL_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD DATA =====
X = np.load(X_PATH)
y = np.load(y_PATH)
N, B, C, T = X.shape
X_reshaped = X.reshape(N, B * C, T)
X_reshaped = (X_reshaped - X_reshaped.mean()) / X_reshaped.std()
X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ===== DATASET =====
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== PRETRAINED ENCODER =====
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

# ===== CLASSIFIER HEAD =====
class EEGClassifier(nn.Module):
    def __init__(self, encoder, feature_dim=64, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, n_classes)
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

# ===== 5-FOLD CV + FINE-TUNING =====
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, y_tensor)):
    print(f"\n📂 Fold {fold+1}/{N_SPLITS} 시작")

    # === Semi-supervised: only LABEL_RATE% used ===
    train_labels = y_tensor[train_idx].numpy()
    label_mask = np.zeros(len(train_labels), dtype=bool)
    for label in np.unique(train_labels):
        label_indices = np.where(train_labels == label)[0]
        n_labeled = int(len(label_indices) * LABEL_RATE)
        selected = np.random.choice(label_indices, n_labeled, replace=False)
        label_mask[selected] = True
    labeled_idx = np.array(train_idx)[label_mask]

    # === Datasets and Loaders ===
    train_dataset = EEGDataset(X_tensor[labeled_idx], y_tensor[labeled_idx])
    test_dataset = EEGDataset(X_tensor[test_idx], y_tensor[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # === Load encoder (fine-tunable) ===
    encoder = SpatioTemporalEncoder(in_channels=B*C).to(DEVICE)
    model = EEGClassifier(encoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_accuracies = []

    # === Training ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)

        # Test accuracy per epoch (optional, just for visualization)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                output = model(X_batch)
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                trues.extend(y_batch.numpy())
        acc = accuracy_score(trues, preds)
        test_accuracies.append(acc)

        print(f"📉 Fold {fold+1} Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f} | Test Acc: {acc:.4f}")

    # Final fold accuracy
    accuracies.append(test_accuracies[-1])
    print(f"✅ Fold {fold+1} Accuracy: {test_accuracies[-1]:.4f}")

    # === Plot loss and accuracy curve ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.title(f"Fold {fold+1} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Acc")
    plt.title(f"Fold {fold+1} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fold_{fold+1}_curve.png")
    plt.close()

print(f"\n🎯 Fine-tune avg acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
