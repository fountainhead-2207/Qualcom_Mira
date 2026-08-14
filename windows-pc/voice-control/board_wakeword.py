"""
Board-side wake word -> Zipformer STT -> local motion execution.

Runs entirely on the UNO Q board: no SSH, no network hop, no Windows PC in
the loop for the canned-command path. The board captures its own mic
(BOYAMIC, via raw `arecord` piping since it only offers S24_3LE/48kHz/stereo
with real signal on the RIGHT channel only - see debugging notes below),
runs the "mira" wake-word model, and on trigger transcribes the following
~4s with the on-board Zipformer-30M-RNNT model before calling
`mira-robot replay <motion>` directly as a local subprocess.

Hardware notes (found empirically 2026-08-14):
  - BOYAMIC only accepts S24_3LE, 2 channels, 48000Hz - no other format/rate.
  - Real signal is on the RIGHT channel; LEFT is silent.
  - Board CPU (aarch64) STT latency measured at ~700ms for a 3-4s clip,
    comfortably under a second.
"""
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from openwakeword.model import Model
import sherpa_onnx
from prometheus_client import Gauge, Counter, start_http_server

METRICS_PORT = 9103
WAKE_SCORE = Gauge("board_wakeword_score", "Latest openWakeWord 'mira' score")
WAKE_TRIGGERS = Counter("board_wakeword_triggers_total", "Wake word trigger count")
COMMAND_AUDIO_PEAK = Gauge("board_command_audio_peak", "Peak amplitude of last command window (24-bit scale)")
COMMAND_AUDIO_RMS = Gauge("board_command_audio_rms", "RMS amplitude of last command window (24-bit scale)")
COMMAND_RESULTS = Counter("board_command_recognized_total", "Command recognition outcomes", ["motion"])

MIRA_ONNX = "/home/arduino/.local/share/mira-so101/mira.onnx"
ZIPFORMER_DIR = "/home/arduino/zipformer-vi"
ALSA_DEVICE = "hw:CARD=BOYAMIC,DEV=0"
DEVICE_FS = 48000
FS = 16000  # target rate for wake-word + STT
FRAME = 1280  # 80ms at 16kHz
DEVICE_FRAME_SAMPLES = FRAME * DEVICE_FS // FS  # 80ms at 48kHz, per channel
BYTES_PER_SAMPLE = 3  # S24_3LE
CHANNELS = 2
RIGHT_CHANNEL = 1  # confirmed: LEFT is silent on this mic
DEVICE_CHUNK_BYTES = DEVICE_FRAME_SAMPLES * CHANNELS * BYTES_PER_SAMPLE

COMMAND_WINDOW_S = 2.8
THRESHOLD = 0.5
# audio_i16 below is already downsampled/renormalized to 16-bit scale by
# to_16k_int16(), so this floor is directly comparable to the Windows one.
MIN_COMMAND_RMS = 200.0

# No duration argument -> listen indefinitely (until the sleep command or
# Ctrl-C), matching how this is actually meant to run day-to-day. Pass a
# number of seconds explicitly for a bounded test session.
TOTAL_DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else float("inf")

SLEEP = "__sleep__"

DEBUG_DIR = Path("/home/arduino/voice_debug_logs")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
_debug_counter = [0]

COMMAND_MAP = {
    "vẫy tay": "wave", "quay tay": "wave", "bảy tay": "wave", "bẫy tay": "wave", "vẫy": "wave", "wave": "wave",
    "nhảy": "dance", "nháy": "dance", "múa": "dance", "mua": "dance", "mu": "dance", "dance": "dance",
    "quét": "scan", "scan": "scan",
    "lắc đầu": "shake", "lắt đầu": "shake", "lắc đào": "shake", "lắc": "shake", "shake": "shake",
    "gật đầu": "nod", "gập đầu": "nod", "giật đầu": "nod", "gật": "nod", "nod": "nod",
    "đồng ý": "yes", "có": "yes", "yes": "yes",
    "không": "no", "no": "no",
    "ngủ": SLEEP, "đi ngủ": SLEEP, "nghỉ ngơi": SLEEP, "nghỉ": SLEEP, "sleep": SLEEP,
}


def save_debug_wav(samples_i16, fs):
    _debug_counter[0] += 1
    path = DEBUG_DIR / f"cmd_capture_{_debug_counter[0]}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(samples_i16.tobytes())
    return str(path)


def decode_24bit_right_channel(raw_bytes):
    """raw interleaved S24_3LE stereo -> mono int32 samples from the right channel."""
    n_total = len(raw_bytes) // BYTES_PER_SAMPLE
    if n_total == 0:
        return np.zeros(0, dtype=np.int32)
    arr = np.frombuffer(raw_bytes[: n_total * BYTES_PER_SAMPLE], dtype=np.uint8).reshape(-1, 3)
    vals = arr[:, 0].astype(np.int32) | (arr[:, 1].astype(np.int32) << 8) | (arr[:, 2].astype(np.int32) << 16)
    vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
    return vals[RIGHT_CHANNEL::CHANNELS]


def to_16k_int16(samples_24bit_48k):
    """Downsample 48kHz 24-bit-range samples to 16kHz int16 for the models."""
    f32 = samples_24bit_48k.astype(np.float32) / 8388608.0  # normalize 24-bit full scale
    f32_16k = resample_poly(f32, 1, DEVICE_FS // FS)
    return np.clip(f32_16k * 32767.0, -32768, 32767).astype(np.int16)


print("Loading wake-word model...", flush=True)
oww = Model(wakeword_model_paths=[MIRA_ONNX])
wake_key = list(oww.models.keys())[0]

print("Loading Zipformer-30M-RNNT (int8, CPU)...", flush=True)
stt = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens=f"{ZIPFORMER_DIR}/tokens.txt",
    encoder=f"{ZIPFORMER_DIR}/encoder-epoch-20-avg-10.int8.onnx",
    decoder=f"{ZIPFORMER_DIR}/decoder-epoch-20-avg-10.int8.onnx",
    joiner=f"{ZIPFORMER_DIR}/joiner-epoch-20-avg-10.int8.onnx",
    num_threads=2,
    sample_rate=FS,
    feature_dim=80,
    decoding_method="greedy_search",
)


def transcribe(samples_i16, fs):
    f32 = samples_i16.astype(np.float32) / 32768.0
    stream = stt.create_stream()
    stream.accept_waveform(fs, f32)
    stt.decode_stream(stream)
    return stream.result.text.strip().lower()


robot_busy = threading.Event()


def run_command(motion):
    def _worker():
        print(f"  >>> mira-robot replay {motion} (local, no SSH) ...", flush=True)
        try:
            result = subprocess.run(
                ["mira-robot", "replay", motion, "--yes"],
                capture_output=True, text=True, timeout=45,
            )
            last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "(no output)"
            print("  ", last_line, flush=True)
            if result.returncode != 0:
                print("  error:", result.stderr.strip(), flush=True)
        finally:
            robot_busy.clear()

    robot_busy.set()
    threading.Thread(target=_worker, daemon=True).start()


start_http_server(METRICS_PORT)
print(f"Metrics on :{METRICS_PORT}/metrics", flush=True)
duration_label = "indefinitely (say 'Mira ngủ' or 'Mira nghỉ ngơi' to stop)" if TOTAL_DURATION == float("inf") else f"for {TOTAL_DURATION}s"
print(f"Listening {duration_label} (device={ALSA_DEVICE}) ...", flush=True)

arecord = subprocess.Popen(
    ["arecord", "-D", ALSA_DEVICE, "-f", "S24_3LE", "-c", str(CHANNELS), "-r", str(DEVICE_FS), "-t", "raw"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)


def read_frame_16k():
    raw = arecord.stdout.read(DEVICE_CHUNK_BYTES)
    if len(raw) < DEVICE_CHUNK_BYTES:
        return None
    right = decode_24bit_right_channel(raw)
    return to_16k_int16(right)


def record_window_16k(seconds):
    n_frames = int(seconds * FS / FRAME)
    chunks = []
    for _ in range(n_frames):
        chunk = read_frame_16k()
        if chunk is not None:
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


import collections
preroll = collections.deque(maxlen=int(1.2 * FS / FRAME) + 1)
t_start = time.perf_counter()

try:
    _dbg_last = time.perf_counter()
    _dbg_scores, _dbg_rms = [], []
    while time.perf_counter() - t_start < TOTAL_DURATION:
        chunk = read_frame_16k()
        if chunk is None:
            print("arecord stream ended unexpectedly", flush=True)
            break
        preroll.append(chunk)
        score = oww.predict(chunk)[wake_key]
        WAKE_SCORE.set(score)
        _dbg_scores.append(score)
        _dbg_rms.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if chunk.size else 0.0)
        if time.perf_counter() - _dbg_last > 1.0:
            print(f"  [debug] max_score={max(_dbg_scores):.4f} max_rms={max(_dbg_rms):.0f} frames={len(_dbg_scores)}", flush=True)
            _dbg_scores.clear()
            _dbg_rms.clear()
            _dbg_last = time.perf_counter()
        t = time.perf_counter() - t_start
        if score > 0.3:
            print(f"  {t:5.2f}s  wake_score={score:.3f}", flush=True)

        if score > THRESHOLD and robot_busy.is_set():
            print(f"  {t:5.2f}s  heard but robot busy - ignoring", flush=True)
            oww.reset()
            continue

        if score > THRESHOLD:
            WAKE_TRIGGERS.inc()
            print(f"  {t:5.2f}s  WAKE WORD DETECTED -> recording command window...", flush=True)
            pre_audio = np.concatenate(list(preroll)) if preroll else np.zeros(0, dtype=np.int16)
            preroll.clear()
            audio_i16 = np.concatenate([pre_audio, record_window_16k(COMMAND_WINDOW_S)])
            peak = int(np.abs(audio_i16).max()) if audio_i16.size else 0
            rms = float(np.sqrt(np.mean(audio_i16.astype(np.float64) ** 2))) if audio_i16.size else 0.0
            COMMAND_AUDIO_PEAK.set(peak)
            COMMAND_AUDIO_RMS.set(rms)
            wav_path = save_debug_wav(audio_i16, FS)
            print(f"  captured peak={peak} rms={rms:.1f} saved={wav_path}", flush=True)

            if rms < MIN_COMMAND_RMS:
                print(f"  too quiet (rms={rms:.1f}) - skipping STT", flush=True)
                COMMAND_RESULTS.labels(motion="skipped_too_quiet").inc()
                oww.reset()
                continue

            t0 = time.perf_counter()
            heard = transcribe(audio_i16, FS)
            stt_ms = (time.perf_counter() - t0) * 1000
            print(f"  heard: '{heard}' ({stt_ms:.0f}ms)", flush=True)

            heard_words = set(heard.strip(".,!? ").split())
            matched = None
            for word, motion in COMMAND_MAP.items():
                if len(word) <= 4 and " " not in word:
                    if word in heard_words:
                        matched = motion
                        break
                elif word in heard:
                    matched = motion
                    break

            if matched == SLEEP:
                COMMAND_RESULTS.labels(motion="sleep").inc()
                print("  >>> going to sleep - playing rest pose, then stopping the listening loop", flush=True)
                subprocess.run(["mira-robot", "replay", "rest", "--yes"], capture_output=True, text=True, timeout=45)
                oww.reset()
                break
            elif matched:
                COMMAND_RESULTS.labels(motion=matched).inc()
                run_command(matched)
            else:
                COMMAND_RESULTS.labels(motion="unmatched").inc()
                print("  (no known command word)", flush=True)
            oww.reset()
finally:
    arecord.terminate()

print("\nDone listening.", flush=True)
