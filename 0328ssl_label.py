import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# ===== Dataset =====
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== SSL Encoder =====
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

# ===== Classifier =====
class EEGClassifier(nn.Module):
    def __init__(self, encoder, feature_dim=64, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, n_classes)
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

# ===== Config =====
X_PATH = "preprocessed_data_segments/X_segments.npy"
y_PATH = "preprocessed_data_segments/y_segments.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001
LABEL_RATIOS = [0.1, 0.3, 0.5, 1.0]
N_SPLITS = 5

# ===== Load Data =====
X = np.load(X_PATH)
y = np.load(y_PATH)
N, B, C, T = X.shape
X_reshaped = X.reshape(N, B * C, T)
X_reshaped = (X_reshaped - X_reshaped.mean()) / X_reshaped.std()
X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ===== Main Loop =====
results = {}

for ratio in LABEL_RATIOS:
    print(f"\n🔍 Fine-tuning with {int(ratio*100)}% labeled data")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, y_tensor)):
        train_labels = y_tensor[train_idx].numpy()
        label_mask = np.zeros(len(train_labels), dtype=bool)

        for label in np.unique(train_labels):
            label_indices = np.where(train_labels == label)[0]
            n_labeled = int(len(label_indices) * ratio)
            selected = np.random.choice(label_indices, n_labeled, replace=False)
            label_mask[selected] = True

        labeled_idx = np.array(train_idx)[label_mask]
        train_dataset = EEGDataset(X_tensor[labeled_idx], y_tensor[labeled_idx])
        test_dataset = EEGDataset(X_tensor[test_idx], y_tensor[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        encoder = SpatioTemporalEncoder(in_channels=B*C).to(DEVICE)
        model = EEGClassifier(encoder).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                output = model(X_batch)
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                trues.extend(y_batch.numpy())
        acc = accuracy_score(trues, preds)
        accs.append(acc)
        print(f"✅ Fold {fold+1}: Accuracy = {acc:.4f}")

    results[ratio] = (np.mean(accs), np.std(accs))

# ===== Plotting =====
x_vals = [int(r * 100) for r in results.keys()]
y_means = [results[r][0] for r in results.keys()]
y_stds = [results[r][1] for r in results.keys()]

plt.figure(figsize=(8, 5))
plt.errorbar(x_vals, y_means, yerr=y_stds, fmt='-o', capsize=5)
plt.title("SSL Fine-tune Accuracy vs Label Ratio")
plt.xlabel("Labeled Data Ratio (%)")
plt.ylabel("Accuracy")
plt.xticks(x_vals)
plt.grid(True)
plt.tight_layout()
plt.savefig("ssl_label_ratio_curve.png")
plt.show()

# ===== Summary Print =====
print("\n📊 결과 요약:")
for r in results:
    print(f"{int(r*100)}% labeled: {results[r][0]:.4f} ± {results[r][1]:.4f}")