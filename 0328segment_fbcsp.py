import numpy as np
import os

# ===== 설정 =====
INPUT_PATH = "/Users/macbookair/[2025]학부연구생/preprocessed_data_fbcsp/X_fbcsp.npy"
LABEL_PATH = "/Users/macbookair/[2025]학부연구생/preprocessed_data_fbcsp/y_fbcsp.npy"
SAVE_DIR = "/Users/macbookair/[2025]학부연구생/preprocessed_data_segments_fbcsp"
SEGMENT_LEN = 1000  # 자를 segment 길이

# ===== 데이터 불러오기 =====
X = np.load(INPUT_PATH, allow_pickle=True)
y = np.load(LABEL_PATH, allow_pickle=True)

segment_data = []
segment_labels = []

# === 1. 최소 segment 수 계산 ===
segment_counts = [bank[0].shape[1] // SEGMENT_LEN for bank in X]
min_segments = min(segment_counts)
print(f"모든 subject에서 사용할 segment 수: {min_segments}")

# === 2. 동일 개수의 segment 생성 ===
for subj_idx, (banks, label) in enumerate(zip(X, y)):
    for s in range(min_segments):
        segment = []
        for bank in banks:
            segment_slice = bank[:, s*SEGMENT_LEN:(s+1)*SEGMENT_LEN]  # (C, T)
            segment.append(segment_slice)
        try:
            segment = np.stack(segment, axis=0)  # shape: (B, C, T)
            segment_data.append(segment)
            segment_labels.append(label)
        except Exception as e:
            print(f"segment stacking 실패 (subject {subj_idx}, segment {s}): {e}")
    print(f"Subject {subj_idx}: {min_segments} segments 생성")

# === 3. shape 불일치한 segment 제거 ===
print("\nsegment shape 일치 여부 확인 중...")
base_shape = segment_data[0].shape
filtered_segments = []
filtered_labels = []

for idx, (s, l) in enumerate(zip(segment_data, segment_labels)):
    if s.shape == base_shape:
        filtered_segments.append(s)
        filtered_labels.append(l)
    else:
        subj_idx = idx // min_segments
        segment_idx = idx % min_segments
        print(f"제외된 segment: subject {subj_idx}, segment {segment_idx}, shape: {s.shape}")

# === 4. 배열로 변환 및 저장 ===
segment_data = np.stack(filtered_segments)
segment_labels = np.array(filtered_labels)
print(f"\n최종 shape: {segment_data.shape}, labels: {segment_labels.shape}") #최종 shape: (840, 5, 60, 1000), labels: (840,)

# === 5. 저장 ===
os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "X_segments_fbcsp.npy"), segment_data)
np.save(os.path.join(SAVE_DIR, "y_segments_fbcsp.npy"), segment_labels)
print("segment 단위 저장 완료 (FBCSP 모델용)")
