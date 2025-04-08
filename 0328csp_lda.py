from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import CSP

X = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/X_segments_general.npy", allow_pickle=True)
y = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/y_segments_general.npy")
X = X.astype(np.float64)

def apply_csp_lda_general(X_data, y_data, n_splits=5, tsne_plot=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list = []
    loss_list = []
    all_features = []
    all_labels = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # ===== CSP 적용 =====
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)

        # ===== LDA 학습 =====
        clf = LDA()
        clf.fit(X_train_csp, y_train)
        y_pred = clf.predict(X_test_csp)
        y_prob = clf.predict_proba(X_test_csp)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)

        acc_list.append(acc)
        loss_list.append(loss)

        print(f"Fold {fold+1} Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")

        if tsne_plot:
            all_features.append(X_test_csp)
            all_labels.append(y_test)

    # ===== t-SNE 시각화 =====
    if tsne_plot:
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(6, 6))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Class {label}", alpha=0.6)
        plt.legend()
        plt.title("t-SNE of CSP+LDA Features (5-Fold CV)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("tsne_csp_lda_general.png")
        plt.show()

    # ===== Log Loss Bar Plot 추가 =====
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_splits + 1), loss_list, color='mediumseagreen')
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.title("Fold-wise Log Loss - CSP+LDA")
    plt.xticks(range(1, n_splits + 1))
    plt.ylim(0, max(loss_list) * 1.1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("log_loss_csp_lda.png")
    plt.show()

    return acc_list, loss_list

X = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/X_segments_general.npy", allow_pickle=True)
y = np.load("/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_general/y_segments_general.npy")

X = X.astype(np.float64)

acc_list, loss_list = apply_csp_lda_general(X, y, n_splits=5, tsne_plot=True)

print(f"\nCSP+LDA 평균 정확도: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"CSP+LDA 평균 Log Loss: {np.mean(loss_list):.4f} ± {np.std(loss_list):.4f}")
#CSP+LDA 평균 정확도: 0.7690 ± 0.0444
#CSP+LDA 평균 Log Loss: 0.4982 ± 0.0394
