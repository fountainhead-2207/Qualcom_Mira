"""
Voice data collection for measuring/tuning STT accuracy, recorded straight
from the board's own mic (BOYAMIC) - the same hardware path board_wakeword.py
and collect_mira_samples.py already use in production.

Ports collect_voice_data.py (the original Windows-side, PhoWhisper-era
version, kept as reference) to run board-native: that version relied on
interactive ENTER-to-stop keypresses over sounddevice, which doesn't work
over a non-interactive SSH command. This version instead records a fixed
window per prompt (auto "GO" cue, no keypress needed), same as
collect_mira_samples.py, and applies the same 24-bit-right-channel decode +
48kHz->16kHz downsample BOYAMIC needs.

Saves audio + the expected (ground-truth) text side by side, for measuring
real Zipformer transcription accuracy against known-correct text.

Usage: python collect_voice_data_board.py [output_dir]
"""
import json
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
MIN_RMS = 150

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/arduino/voice_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# every distinct motion word/phrase actually in COMMAND_MAP (board_wakeword.py),
# plus the wake word alone and a few full "wake word + command" sentences,
# since that's the real shape of what gets transcribed live. (prompt, seconds)
PROMPTS = [
    ("Mira", 1.6),
    ("vẫy tay", 1.8), ("múa", 1.6), ("nhảy", 1.6), ("dọn dẹp", 1.8), ("quét", 1.6),
    ("lắc đầu", 1.8), ("gật đầu", 1.8), ("đồng ý", 1.8), ("không", 1.6),
    ("Mira vẫy tay", 2.6),
    ("Mira nhảy", 2.4),
    ("Mira dọn dẹp", 2.8),
    ("Mira quét", 2.4),
    ("Mira lắc đầu", 2.8),
    ("Mira gật đầu", 2.8),
    ("Mira đồng ý", 2.8),
    ("Mira không", 2.4),
]


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


def record(seconds):
    n_bytes = int(seconds * DEVICE_FS) * CHANNELS * BYTES_PER_SAMPLE
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
    return to_16k_int16(decode_24bit_right_channel(raw))


def main():
    manifest_path = OUT_DIR / "manifest.jsonl"
    manifest = []
    if manifest_path.exists():
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_prompts = {m["prompt"] for m in manifest}

    print(f"Saving to {OUT_DIR}", flush=True)
    print(f"{len(PROMPTS)} prompts, {len(done_prompts)} already recorded in a previous run.\n", flush=True)

    i = 0
    while i < len(PROMPTS):
        prompt, seconds = PROMPTS[i]
        if prompt in done_prompts:
            i += 1
            continue

        print(f"[{i+1}/{len(PROMPTS)}] Say:  \"{prompt}\"", flush=True)
        time.sleep(0.6)
        print(f"  GO - say it now! (~{seconds:.1f}s)", flush=True)

        audio = record(seconds)
        peak = int(np.abs(audio).max()) if audio.size else 0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
        print(f"  captured peak={peak} rms={rms:.1f}", flush=True)

        if rms < MIN_RMS:
            print("  too quiet, not saving - will retry this one\n", flush=True)
            continue

        clip_id = f"{i:03d}"
        wav_path = OUT_DIR / f"{clip_id}.wav"
        save_wav(wav_path, audio, FS)
        manifest.append({
            "id": clip_id,
            "prompt": prompt,
            "wav": wav_path.name,
            "fs": FS,
            "peak": peak,
            "rms": rms,
        })
        with manifest_path.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        done_prompts.add(prompt)
        print(f"  saved {wav_path.name}\n", flush=True)
        i += 1

    print(f"\nDone. {len(manifest)} clips saved in {OUT_DIR}, manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
