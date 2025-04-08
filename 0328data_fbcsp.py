import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt

# ===== 설정 =====
DATA_DIR = "/Users/macbookair/[2025]학부연구생/Data and Code/Dataset/IowaDataset/Raw data"
SPLIT_PATH = "/Users/macbookair/[2025]학부연구생/fixed_5fold_splits_from_vhdr.npz"
SAVE_DIR = "/Users/macbookair/[2025]학부연구생/preprocessed_data_fbcsp"
COMMON_CHANNEL_PATH = "common_channels.npy"
FS = 250
BANDPASS = (0.5, 55)
FILTER_BANKS = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 50)]

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 필터 함수 정의 =====
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_filter_banks(data, fs):
    return [bandpass_filter(data, low, high, fs) for (low, high) in FILTER_BANKS]

# ===== 공통 채널 불러오기 =====
common_channels = np.load(COMMON_CHANNEL_PATH)

# ===== 분할 정보 불러오기 =====
split = np.load(SPLIT_PATH, allow_pickle=True)
subjects = split["subjects"]
labels = split["labels"]

# ===== 1차 패스: 전체 최소 시간 길이 측정 =====
data_lengths = []

for filename in subjects:
    try:
        raw = mne.io.read_raw_brainvision(os.path.join(DATA_DIR, filename), preload=True, verbose=False)
        raw.pick_channels(common_channels)
        raw.resample(FS)
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)
        data = raw.get_data()
        data_lengths.append(data.shape[1])
    except Exception as e:
        print(f"[길이 측정 실패] {filename}: {e}")

global_min_len = min(data_lengths)
print(f"\n▶ 공통 최소 길이(global_min_len): {global_min_len} samples")

# ===== 2차 패스: 전처리, 자르기, 필터뱅크 적용 및 저장 =====
all_data = []
all_labels = []

for i, filename in enumerate(subjects):
    full_path = os.path.join(DATA_DIR, filename)
    try:
        raw = mne.io.read_raw_brainvision(full_path, preload=True, verbose=False)
        # === Resp 채널을 misc로 지정하여 경고 방지 ===
        if 'Resp' in raw.ch_names:
            raw.set_channel_types({'Resp': 'misc'})
        raw.pick(common_channels)
        raw.resample(FS)
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)
        data = raw.get_data()[:, :global_min_len]  # ← 시간 길이 맞춤

        filtered_banks = apply_filter_banks(data, FS)
        all_data.append(filtered_banks)
        all_labels.append(labels[i])

        print(f"전처리 완료 (FBCSP): {filename} | shape per bank: {filtered_banks[0].shape}")
    except Exception as e:
        print(f"[전처리 실패] {filename}: {e}")

# ===== 저장 =====
np.save(os.path.join(SAVE_DIR, "X_fbcsp.npy"), np.array(all_data, dtype=object))
np.save(os.path.join(SAVE_DIR, "y_fbcsp.npy"), np.array(all_labels))
print(f"\n전처리된 FBCSP 전용 데이터 저장 완료: {SAVE_DIR}")
