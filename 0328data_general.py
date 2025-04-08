import os
import numpy as np
import mne

# ===== 설정 =====
DATA_DIR = "/Users/macbookair/[2025]학부연구생/Data and Code/Dataset/IowaDataset/Raw data"
SPLIT_PATH = "/Users/macbookair/[2025]학부연구생/fixed_5fold_splits_from_vhdr.npz"
SAVE_DIR = "/Users/macbookair/[2025]학부연구생/preprocessed_data_general"
COMMON_CHANNEL_PATH = "common_10_20_channels.npy"  # 공통 채널 저장된 경로
FS = 250  # Resample target frequency
BANDPASS = (0.5, 55)

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 공통 채널 불러오기 =====
common_channels = np.load(COMMON_CHANNEL_PATH)

# ===== 분할 정보 불러오기 =====
split = np.load(SPLIT_PATH, allow_pickle=True)
subjects = split["subjects"]
labels = split["labels"]

# ===== 전처리 실행 =====
all_data = []
all_labels = []

for i, filename in enumerate(subjects):
    full_path = os.path.join(DATA_DIR, filename)
    try:
        raw = mne.io.read_raw_brainvision(full_path, preload=True, verbose=False)

        # ===== 공통 채널로 제한  =====
        raw.pick_channels(common_channels)

        # ===== 나머지 전처리 =====
        raw.resample(FS)
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)
        data = raw.get_data()  # shape: (channels, time)

        all_data.append(data)
        all_labels.append(labels[i])

        print(f"전처리 완료 (General): {filename} | shape: {data.shape}")
    except Exception as e:
        print(f"에러 발생 ({filename}): {e}")

# ===== 길이 통일 (시간 길이, 샘플 수가 각 subject 마다 달라서 np.array로 묶는데 실패함)=====
min_length = min([data.shape[1] for data in all_data])
print(f"모든 subject 데이터를 {min_length} 샘플로 잘라서 저장합니다.") #결과: 모든 subject 데이터를 30300 샘플로 잘라서 저장합니다.

all_data = [data[:, :min_length] for data in all_data]

# ===== 저장 =====
np.save(os.path.join(SAVE_DIR, "X_general.npy"), np.array(all_data, dtype=object))
np.save(os.path.join(SAVE_DIR, "y_general.npy"), np.array(all_labels))
print(f"\n전처리된 일반 모델용 데이터 저장 완료: {SAVE_DIR}")

