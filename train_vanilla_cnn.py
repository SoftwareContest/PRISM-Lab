# train_vanilla_cnn.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ===== 하이퍼파라미터 =====
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== CNN 모델 정의 =====
class CNNBackbone(nn.Module):
    def __init__(self, in_channels=21):
        super(CNNBackbone, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 128, 1)
        x = x.squeeze(-1)              # (B, 128)
        return x

class Classifier2(nn.Module):
    def __init__(self, in_features=128, n_classes=2):
        super(Classifier2, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

# ===== 데이터 로딩 =====
X = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/X_segments_general.npy",allow_pickle=True).astype(np.float32)  # (840, 60, 1000)
y = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/y_segments_general.npy").astype(np.int64)  # (840,)

# ===== 학습/검증 분할 =====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===== 모델 정의 =====
backbone = CNNBackbone(in_channels=21).to(DEVICE)
classifier = Classifier2(in_features=128, n_classes=2).to(DEVICE)

model = nn.Sequential(backbone, classifier).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== 학습 루프 =====
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb = xb  # (B, 60, 1000)

        outputs = model(xb)  # (B, 2)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == yb).sum().item()
        total += yb.size(0)

    acc = correct / total * 100
    print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# ===== 저장 =====
torch.save(model.state_dict(), "vanilla_CNN_PD_classifier.pth")
print("Vanilla CNN PD 분류기 모델이 저장되었습니다.")
