import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# ===== 설정 =====
X_PATH = "preprocessed_data_segments/X_segments.npy"
y_PATH = "preprocessed_data_segments/y_segments.npy"
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001
N_SPLITS = 5
LABEL_RATE = 0.1  # 라벨링된 비율 (10%)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 데이터 로딩 =====
print("📥 데이터 로딩 중...")
X = np.load(X_PATH)
y = np.load(y_PATH)

N, B, C, T = X.shape
X_reshaped = X.reshape(N, B * C, T)
X_reshaped = (X_reshaped - X_reshaped.mean()) / X_reshaped.std()
X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
print(f"✅ 데이터 shape: {X_tensor.shape}, labels: {y_tensor.shape}")

# ===== Dataset 클래스 =====
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== CNN 모델 정의 =====
class EEGCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(EEGCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ===== Cross Validation =====
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, y_tensor)):
    print(f"\n📂 Fold {fold+1}/{N_SPLITS} 시작")

    # === labeled subset 구성 ===
    train_labels = y_tensor[train_idx].numpy()
    label_mask = np.zeros(len(train_labels), dtype=bool)

    for label in np.unique(train_labels):
        label_indices = np.where(train_labels == label)[0]
        n_labeled = int(len(label_indices) * LABEL_RATE)
        selected = np.random.choice(label_indices, n_labeled, replace=False)
        label_mask[selected] = True

    labeled_idx = np.array(train_idx)[label_mask]
    unlabeled_idx = np.array(train_idx)[~label_mask]

    print(f"🔖 라벨링된 데이터: {len(labeled_idx)} / 전체: {len(train_idx)}")

    # === Dataset 구성 ===
    train_dataset = EEGDataset(X_tensor[labeled_idx], y_tensor[labeled_idx])
    test_dataset = EEGDataset(X_tensor[test_idx], y_tensor[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = EEGCNN(in_channels=B*C, n_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === 학습 ===
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
        print(f"📉 Fold {fold+1} Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

    # === 평가 ===
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            trues.extend(y_batch.numpy())

    acc = accuracy_score(trues, preds)
    accuracies.append(acc)
    print(f"✅ Fold {fold+1} 정확도: {acc:.4f}")

print(f"\n🎯 Semi-Supervised 평균 정확도: {np.mean(accuracies):.4f}")
