import os
import mne
from functools import reduce
import numpy as np

# 설정
DATA_DIR = "/Users/macbookair/[2025]학부연구생/Data and Code/Dataset/IowaDataset/Raw data"
vhdr_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".vhdr")]

# standard 10-20 기준 채널 alias
STANDARD_WITH_ALIASES = {
    'Fp1': ['Fp1'],
    'Fp2': ['Fp2'],
    'F7': ['F7'],
    'F3': ['F3'],
    'Fz': ['Fz'],
    'F4': ['F4'],
    'F8': ['F8'],
    'T3': ['T3', 'FT7', 'FT9'],
    'C3': ['C3'],
    'Cz': ['Cz'],
    'C4': ['C4'],
    'T4': ['T4', 'FT8', 'FT10'],
    'T5': ['T5', 'TP7', 'TP9'],
    'P3': ['P3'],
    'Pz': ['Pz', 'POz', 'CPz'],
    'P4': ['P4'],
    'T6': ['T6', 'TP8', 'TP10'],
    'O1': ['O1'],
    'O2': ['O2'],
    'A1': ['A1', 'M1', 'TP9', 'FT9'],
    'A2': ['A2', 'M2', 'TP10', 'FT10']
}

def resolve_channel_names(actual_ch_names, alias_dict):
    resolved = {}
    for std_name, aliases in alias_dict.items():
        for alias in aliases:
            if alias in actual_ch_names:
                resolved[std_name] = alias
                break
    return resolved

# 각 파일에서 standard 10-20 채널 매칭
all_resolved_sets = []

for fname in vhdr_files:
    try:
        raw = mne.io.read_raw_brainvision(os.path.join(DATA_DIR, fname), preload=False, verbose=False)
        actual_ch_names = raw.info['ch_names']
        resolved = resolve_channel_names(actual_ch_names, STANDARD_WITH_ALIASES)
        resolved_set = set(resolved.values())
        all_resolved_sets.append(resolved_set)
        print(f"{fname}: {len(resolved_set)} matched 10-20 channels")
    except Exception as e:
        print(f"Error loading {fname}: {e}")

# 교집합 추출
common_channels = sorted(list(reduce(set.intersection, all_resolved_sets)))
print(f"\n공통 standard 10-20 채널 수: {len(common_channels)}")
print(common_channels)

# 저장
np.save("common_10_20_channels.npy", np.array(common_channels))
