import sys
from pathlib import Path

import soundfile as sf
import soxr

TARGET_SR = 16000

root = Path("/data/qualcom-robotic/mira_model/mira")
dirs = ["positive_train", "positive_test", "negative_train", "negative_test"]

total = 0
for d in dirs:
    folder = root / d
    wavs = list(folder.glob("*.wav"))
    print(f"{d}: {len(wavs)} files")
    for i, wav_path in enumerate(wavs):
        data, sr = sf.read(wav_path)
        if sr != TARGET_SR:
            data = soxr.resample(data, sr, TARGET_SR)
            sf.write(wav_path, data, TARGET_SR)
        total += 1
        if (i + 1) % 2000 == 0:
            print(f"  {d}: {i+1}/{len(wavs)}")

print(f"Done. Resampled/checked {total} files to {TARGET_SR}Hz.")
