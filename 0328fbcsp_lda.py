import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mne.decoding import CSP

# ===== 데이터 경로 설정 =====
X_PATH = "/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_fbcsp/X_segments_fbcsp.npy"
y_PATH = "/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_fbcsp/y_segments_fbcsp.npy"

# ===== 하이퍼파라미터 =====
N_SPLITS = 5
N_CSP_COMPONENTS = 4
TSNE_VISUALIZE = True

# ===== 데이터 불러오기 =====
X = np.load(X_PATH, allow_pickle=True)  # shape: (N, B, C, T)
y = np.load(y_PATH)
X = X.astype(np.float64)  # CSP가 float64 필요함

N, B, C, T = X.shape
print(f"Data loaded: {X.shape}, Labels: {y.shape}")

# ===== FBCSP + LDA 함수 정의 =====
def apply_fbcsp_lda(X_data, y_data, n_splits=5, tsne_plot=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, loss_list = [], []
    all_features, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        X_train_banks, X_test_banks = [], []

        for b in range(B):  # 5개의 필터 뱅크
            Xb_train = X_train[:, b]  # (N, C, T)
            Xb_test = X_test[:, b]

            csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
            Xb_train_csp = csp.fit_transform(Xb_train, y_train)
            Xb_test_csp = csp.transform(Xb_test)

            X_train_banks.append(Xb_train_csp)
            X_test_banks.append(Xb_test_csp)

        X_train_concat = np.concatenate(X_train_banks, axis=1)
        X_test_concat = np.concatenate(X_test_banks, axis=1)

        clf = LDA()
        clf.fit(X_train_concat, y_train)
        preds = clf.predict(X_test_concat)
        probs = clf.predict_proba(X_test_concat)

        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)

        acc_list.append(acc)
        loss_list.append(loss)

        print(f"Fold {fold+1} Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")
        #Fold 1 Accuracy: 0.9524 | Log Loss: 0.1071
        #Fold 2 Accuracy: 0.9821 | Log Loss: 0.0824
        #Fold 3 Accuracy: 0.9762 | Log Loss: 0.0672
        #Fold 4 Accuracy: 0.9881 | Log Loss: 0.0447
        #Fold 5 Accuracy: 0.9702 | Log Loss: 0.0836
        if tsne_plot:
            all_features.append(X_test_concat)
            all_labels.append(y_test)

    # ===== t-SNE 시각화 =====
    if tsne_plot:
        X_all = np.concatenate(all_features, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_all)

        plt.figure(figsize=(6, 6))
        for label in np.unique(y_all):
            idx = y_all == label
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Class {label}", alpha=0.6)
        plt.legend()
        plt.title("t-SNE of FBCSP+LDA Features (5-Fold CV)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("tsne_fbcsp_lda.png")
        plt.show()

    # log loss 시각화
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_splits + 1), loss_list, color='salmon')
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.title("Fold-wise Log Loss - FBCSP+LDA")
    plt.xticks(range(1, n_splits + 1))
    plt.ylim(0, max(loss_list) * 1.1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("log_loss_fbcsp_lda.png")
    plt.show()

    return acc_list, loss_list

# ===== 모델 실행 =====
acc_list, loss_list = apply_fbcsp_lda(X, y, n_splits=N_SPLITS, tsne_plot=TSNE_VISUALIZE)

print(f"\nFBCSP+LDA 평균 정확도: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"FBCSP+LDA 평균 Log Loss: {np.mean(loss_list):.4f} ± {np.std(loss_list):.4f}")
# FBCSP+LDA 평균 정확도: 0.9738 ± 0.0123
# FBCSP+LDA 평균 Log Loss: 0.0770 ± 0.0206
# Fold 1 Accuracy: 0.9524 | Log Loss: 0.1071
# Fold 2 Accuracy: 0.9821 | Log Loss: 0.0824
# Fold 3 Accuracy: 0.9762 | Log Loss: 0.0672
# Fold 4 Accuracy: 0.9881 | Log Loss: 0.0447
# Fold 5 Accuracy: 0.9702 | Log Loss: 0.0836