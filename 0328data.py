import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt

# ===== 설정 =====
DATA_DIR = r"C:\Users\User\PycharmProjects\PRISM-Lab\Data and Code\Dataset\IowaDataset\Raw data"
SPLIT_PATH = r"C:\Users\User\PycharmProjects\PRISM-Lab\fixed_5fold_splits_from_vhdr.npz"
SAVE_DIR = r"C:\Users\User\PycharmProjects\PRISM-Lab\preprocessed_data"
FS = 250  # Resample target frequency
FILTER_BANKS = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 50)]
BANDPASS = (0.5, 55)

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 필터 함수 =====
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_filter_banks(data, fs):
    return [bandpass_filter(data, low, high, fs) for (low, high) in FILTER_BANKS]

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
        raw.resample(FS)
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)
        data = raw.get_data()  # shape: (channels, time)

        filtered_banks = apply_filter_banks(data, FS)  # 리스트 of (channels, time)
        all_data.append(filtered_banks)
        all_labels.append(labels[i])

        print(f"✅ 전처리 완료: {filename}")
    except Exception as e:
        print(f"❌ 에러 발생 ({filename}): {e}")

# ===== 저장 =====
np.save(os.path.join(SAVE_DIR, "X_filtered_banks.npy"), np.array(all_data, dtype=object))
np.save(os.path.join(SAVE_DIR, "y_labels.npy"), np.array(all_labels))
print(f"\n📁 전처리된 데이터 저장 완료: {SAVE_DIR}")
