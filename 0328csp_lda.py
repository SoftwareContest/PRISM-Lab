import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

# ===== 설정 =====
X_PATH = "preprocessed_data_segments/X_segments.npy"
y_PATH = "preprocessed_data_segments/y_segments.npy"
N_SPLITS = 5
BANDS = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 50)]  # Filter Banks
ALL_BAND = True  # 0.5-55Hz CSP 한번 포함 여부

# ===== 데이터 로딩 =====
X = np.load(X_PATH, allow_pickle=True)  # shape: (N, B, C, T)
y = np.load(y_PATH)
N, B, C, T = X.shape

# ===== CSP + LDA 함수 정의 =====
def apply_csp_lda(X_data, y_data, band_idx_list, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        X_train_fb, X_test_fb = [], []

        for band_idx in band_idx_list:
            Xb_train = X_train[:, band_idx]  # (N, C, T)
            Xb_test = X_test[:, band_idx]
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            Xb_train_csp = csp.fit_transform(Xb_train, y_train)
            Xb_test_csp = csp.transform(Xb_test)
            X_train_fb.append(Xb_train_csp)
            X_test_fb.append(Xb_test_csp)

        X_train_concat = np.concatenate(X_train_fb, axis=1)
        X_test_concat = np.concatenate(X_test_fb, axis=1)

        clf = LDA()
        clf.fit(X_train_concat, y_train)
        preds = clf.predict(X_test_concat)
        acc = accuracy_score(y_test, preds)
        acc_list.append(acc)
        print(f"✅ Fold {fold+1} Accuracy: {acc:.4f}")

    return acc_list

# ===== Filter Bank CSP+LDA =====
print("\n🚀 Filter Bank CSP+LDA 실행 중...")
band_indices = list(range(len(BANDS)))  # [0, 1, 2, 3, 4]
acc_fb = apply_csp_lda(X, y, band_indices, N_SPLITS)
print(f"\n🎯 Filter Bank 평균 정확도: {np.mean(acc_fb):.4f} ± {np.std(acc_fb):.4f}")

# ===== 전체 대역 CSP+LDA (0.5–55Hz) =====
if ALL_BAND:
    print("\n🚀 전체 대역 CSP+LDA 실행 중...")
    # 대역 0~4 평균 → 전체 대역으로 가정
    X_all_band = np.mean(X, axis=1, keepdims=True)  # (N, 1, C, T)
    X_all_band = X_all_band[:, 0]  # (N, C, T)
    acc_all = apply_csp_lda(X_all_band[:, np.newaxis], y, [0], N_SPLITS)
    print(f"\n🎯 전체 대역 평균 정확도: {np.mean(acc_all):.4f} ± {np.std(acc_all):.4f}")
