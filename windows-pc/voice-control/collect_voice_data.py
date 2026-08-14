"""
Voice data collection for tuning PhoWhisper to this speaker.

Reads a fixed list of prompts (the wake word + every command word/phrase in
COMMAND_MAP, plus a few full sentences), shows one at a time, records a clip
for it, and saves audio + the expected (ground-truth) text side by side.

Two uses for the output:
  1. Immediate: re-run wakeword_to_whisper.py's transcription over these
     clips to measure real accuracy against known-correct text, instead of
     guessing from live sessions one alias at a time.
  2. Later: this is the shape of data a LoRA/fine-tune pass over PhoWhisper
     would need (audio, correct_text) pairs, if accuracy tuning by aliases
     alone stops being enough.

Usage:
    python collect_voice_data.py [output_dir]

Controls: press ENTER to start recording each prompt, speak, press ENTER
again to stop (or it auto-stops after MAX_CLIP_S). Type 's' + ENTER to skip
a prompt, 'r' + ENTER to redo the previous one.
"""
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.stdout.reconfigure(encoding="utf-8")

DEVICE = "Microphone Array (Intel Smart Sound Technology for Digital Microphones), Windows WASAPI"
DEVICE_FS = 48000
FS = 16000
MAX_CLIP_S = 6.0

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("d:/Comp/Qualcom/voice-control/voice_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# every distinct motion word/phrase actually in COMMAND_MAP, plus the wake
# word alone and a few full "wake word + command" sentences, since that's
# the real shape of what gets transcribed live.
PROMPTS = [
    "Mira",
    "vẫy tay", "múa", "nhảy", "dọn dẹp", "quét", "lắc đầu", "gật đầu", "đồng ý", "không",
    "Mira vẫy tay",
    "Mira nhảy",
    "Mira dọn dẹp",
    "Mira quét",
    "Mira lắc đầu",
    "Mira gật đầu",
    "Mira đồng ý",
    "Mira không",
]


def record_clip(fs_device):
    print(f"  [recording... press ENTER to stop, auto-stops at {MAX_CLIP_S:.0f}s]", flush=True)
    chunks = []
    stop_flag = threading.Event()

    def waiter():
        input()
        stop_flag.set()

    threading.Thread(target=waiter, daemon=True).start()

    def callback(indata, frames, time_info, status):
        chunks.append(indata[:, 0].copy())

    t0 = time.perf_counter()
    with sd.InputStream(samplerate=fs_device, channels=1, dtype="int16", device=DEVICE, callback=callback):
        while not stop_flag.is_set() and (time.perf_counter() - t0) < MAX_CLIP_S:
            time.sleep(0.05)

    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def save_wav(path, audio_i16, fs):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_i16.tobytes())


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
        prompt = PROMPTS[i]
        if prompt in done_prompts:
            i += 1
            continue

        print(f"[{i+1}/{len(PROMPTS)}] Say:  \"{prompt}\"", flush=True)
        print("  Press ENTER when ready to start recording...", flush=True)
        input()

        audio = record_clip(DEVICE_FS)
        if audio.size == 0:
            print("  (nothing captured, retrying this prompt)", flush=True)
            continue

        peak = int(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(f"  captured {audio.size / DEVICE_FS:.1f}s, peak={peak}, rms={rms:.1f}", flush=True)

        action = input("  ENTER = keep, 's' = skip, 'r' = redo: ").strip().lower()
        if action == "s":
            i += 1
            continue
        if action == "r":
            continue

        clip_id = f"{i:03d}"
        wav_path = OUT_DIR / f"{clip_id}.wav"
        save_wav(wav_path, audio, DEVICE_FS)
        manifest.append({
            "id": clip_id,
            "prompt": prompt,
            "wav": wav_path.name,
            "device_fs": DEVICE_FS,
            "peak": peak,
            "rms": rms,
        })
        with manifest_path.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        done_prompts.add(prompt)
        i += 1

    print(f"\nDone. {len(manifest)} clips saved in {OUT_DIR}, manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
