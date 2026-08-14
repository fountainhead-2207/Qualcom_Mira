"""
Collect real "Mira" wake-word positive samples on the board's own mic, for
mixing into a retrain of the wake-word model (currently trained only on
synthetic piper TTS voices - this closes the synthetic-to-real gap for this
speaker specifically).

Same BOYAMIC hardware handling as board_wakeword.py: S24_3LE/2ch/48000Hz,
real signal on the right channel only, downsampled to 16kHz mono for output
(wake-word training samples are expected at 16kHz).

Usage: python collect_mira_samples.py [num_samples] [output_dir]
Prompts "Say: Mira" N times, records a fixed 1.6s window each time (no
manual start/stop - simpler for saying one word many times quickly),
reports peak/rms so obviously-bad takes are visible immediately.
"""
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")

ALSA_DEVICE = "hw:CARD=BOYAMIC,DEV=0"
DEVICE_FS = 48000
FS = 16000
CHANNELS = 2
BYTES_PER_SAMPLE = 3
RIGHT_CHANNEL = 1
CLIP_S = 1.6

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/arduino/mira_real_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_24bit_right_channel(raw_bytes):
    n_total = len(raw_bytes) // BYTES_PER_SAMPLE
    if n_total == 0:
        return np.zeros(0, dtype=np.int32)
    arr = np.frombuffer(raw_bytes[: n_total * BYTES_PER_SAMPLE], dtype=np.uint8).reshape(-1, 3)
    vals = arr[:, 0].astype(np.int32) | (arr[:, 1].astype(np.int32) << 8) | (arr[:, 2].astype(np.int32) << 16)
    vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
    return vals[RIGHT_CHANNEL::CHANNELS]


def to_16k_int16(samples_24bit_48k):
    f32 = samples_24bit_48k.astype(np.float32) / 8388608.0
    f32_16k = resample_poly(f32, 1, DEVICE_FS // FS)
    return np.clip(f32_16k * 32767.0, -32768, 32767).astype(np.int16)


def save_wav(path, audio_i16, fs):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_i16.tobytes())


print(f"Will record {N} samples of the word 'Mira' to {OUT_DIR}", flush=True)
print("Say ONLY 'Mira' each time, right after the beep-less 'GO'. ~1.6s per take.\n", flush=True)

kept = 0
i = 0
while kept < N:
    i += 1
    print(f"[{kept+1}/{N}] Get ready...", flush=True)
    time.sleep(0.6)
    print("  GO - say 'Mira' now!", flush=True)

    n_bytes = int(CLIP_S * DEVICE_FS) * CHANNELS * BYTES_PER_SAMPLE
    # NOTE: arecord's -d/--duration only accepts an integer number of seconds -
    # passing "1.6" made it exit 0 immediately with "invalid duration argument
    # '1.6'" on stderr and zero bytes of audio, which is exactly the peak=0
    # rms=0.0 failure seen on every take (confirmed 2026-08-14). Recording
    # without -d and reading exactly n_bytes off the pipe sidesteps this and
    # matches the approach already proven working in board_wakeword.py.
    proc = subprocess.Popen(
        ["arecord", "-D", ALSA_DEVICE, "-f", "S24_3LE", "-c", str(CHANNELS),
         "-r", str(DEVICE_FS), "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    raw = proc.stdout.read(n_bytes)
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    err = proc.stderr.read().decode(errors="replace").strip()
    if len(raw) < n_bytes:
        print(f"  WARNING: only got {len(raw)}/{n_bytes} bytes from arecord", flush=True)
    if err:
        print(f"  arecord stderr: {err}", flush=True)
    right = decode_24bit_right_channel(raw)
    audio_i16 = to_16k_int16(right)

    peak = int(np.abs(audio_i16).max()) if audio_i16.size else 0
    rms = float(np.sqrt(np.mean(audio_i16.astype(np.float64) ** 2))) if audio_i16.size else 0.0
    print(f"  peak={peak} rms={rms:.1f}", flush=True)

    if rms < 150:
        print("  too quiet, not saving - will retry this one", flush=True)
        continue

    path = OUT_DIR / f"mira_real_{kept:03d}.wav"
    save_wav(path, audio_i16, FS)
    print(f"  saved {path.name}\n", flush=True)
    kept += 1

print(f"Done. {kept} samples saved in {OUT_DIR}", flush=True)
