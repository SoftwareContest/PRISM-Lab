import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ===== 설정 =====
RAW_DATA_DIR = r"C:\Users\User\PycharmProjects\PRISM-Lab\Data and Code\Dataset\IowaDataset\Raw data"
SAVE_PATH = r"C:\Users\User\PycharmProjects\PRISM-Lab\fixed_5fold_splits_from_vhdr.npz"

# ===== .vhdr 파일만 가져오기 =====
all_files = os.listdir(RAW_DATA_DIR)
vhdr_files = [f for f in all_files if f.endswith(".vhdr")]

# ===== Control과 PD 분류 =====
normal_files = sorted([f for f in vhdr_files if f.startswith("Control")])
pd_files = sorted([f for f in vhdr_files if f.startswith("PD")])

print(f"✅ Control 파일 수: {len(normal_files)}")
print(f"✅ PD 파일 수: {len(pd_files)}")

# ===== subject 리스트와 label 생성 =====
subjects = normal_files + pd_files
labels = [0] * len(normal_files) + [1] * len(pd_files)  # 0: Control, 1: PD

# ===== 예외 처리 =====
if len(subjects) == 0:
    raise ValueError("❌ .vhdr 파일이 하나도 없습니다. 경로를 다시 확인하세요.")
if len(normal_files) != 14 or len(pd_files) != 14:
    print("⚠️ 경고: Control 또는 PD 데이터 수가 14이 아닙니다. 확인해보세요.")

# ===== Stratified 5-Fold Split =====
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(subjects, labels))

# ===== 저장할 딕셔너리 구성 =====
split_dict = {
    "subjects": np.array(subjects),
    "labels": np.array(labels),
}
for i, (train_idx, test_idx) in enumerate(splits):
    split_dict[f"train_{i}"] = train_idx
    split_dict[f"test_{i}"] = test_idx

# ===== 저장 =====
np.savez(SAVE_PATH, **split_dict)
print(f"\n✅ 5-Fold splits 저장 완료: {SAVE_PATH}")
