import collections
import queue
import subprocess
import sys
import threading
import time
import wave

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model
import sherpa_onnx
from prometheus_client import Gauge, Counter, start_http_server

METRICS_PORT = 9102
WAKE_SCORE = Gauge("wakeword_score", "Latest openWakeWord 'mira' score for the current audio chunk")
WAKE_TRIGGERS = Counter("wakeword_triggers_total", "Number of times the wake word crossed the trigger threshold")
COMMAND_AUDIO_PEAK = Gauge("command_audio_peak", "Peak int16 amplitude of the last captured command-window audio")
COMMAND_AUDIO_RMS = Gauge("command_audio_rms", "RMS int16 amplitude of the last captured command-window audio")
COMMAND_RESULTS = Counter("command_recognized_total", "Command-word recognition outcomes", ["motion"])

# Whisper hallucinates plausible-sounding full sentences on near-silent audio
# (observed live: a false wake-word trigger on background noise, rms~3.6,
# produced a long invented Vietnamese sentence that happened to contain "co"
# and incorrectly fired the "yes" motion on the real robot). Skip transcription
# entirely below this floor rather than trust the matcher to catch it.
# Tuned from live measurement of this user's real speech (2026-08-14):
# 7 samples ranged 230.9-607.5 RMS; 200 sits safely below the observed
# minimum while filtering more ambient noise than the original 150 floor.
MIN_COMMAND_RMS = 200.0

DEBUG_DIR = "d:/Comp/Qualcom/voice-control/debug_logs"
import os
os.makedirs(DEBUG_DIR, exist_ok=True)
_debug_counter = [0]


def save_debug_wav(audio_i16, fs):
    _debug_counter[0] += 1
    path = f"{DEBUG_DIR}/cmd_capture_{_debug_counter[0]}.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_i16.tobytes())
    return path

MIRA_ONNX = "d:/Comp/Qualcom/voice-control/mira.onnx"
DEVICE = "Microphone Array (Intel Smart Sound Technology for Digital Microphones), Windows WASAPI"
# No duration argument -> listen indefinitely (until the sleep command or
# Ctrl-C). Pass a number of seconds explicitly for a bounded test session.
TOTAL_DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
SLEEP = "__sleep__"
FS = 16000  # target rate for the wake-word model and Whisper
# WASAPI shared-mode streams generally refuse to open at an arbitrary
# samplerate (unlike MME, which resamples internally) - capture at the
# device's native rate and downsample ourselves instead.
DEVICE_FS = 48000
FRAME = 1280  # 80ms for wake-word, at FS (16kHz)
DEVICE_FRAME = FRAME * DEVICE_FS // FS  # 80ms at DEVICE_FS (48kHz)
COMMAND_WINDOW_S = 2.8
THRESHOLD = 0.5

LANGUAGE = "vi"

# substring (as Whisper is likely to transcribe it) -> exact mira-robot motion name
# Vietnamese phrases first, English words kept as fallback in case Whisper still
# outputs them (e.g. mixed-language speech).
COMMAND_MAP = {
    "vẫy tay": "wave",
    "quay tay": "wave",
    "bảy tay": "wave",
    "bẫy tay": "wave",
    "vẫy": "wave",
    "wave": "wave",
    "nhảy": "dance",
    "nháy": "dance",
    "múa": "dance",
    "mua": "dance",
    "mu": "dance",
    "dance": "dance",
    "dọn dẹp": "clean",
    "dọn": "clean",
    "clean": "clean",
    "quét": "scan",
    "scan": "scan",
    "lắc đầu": "shake",
    "lắt đầu": "shake",
    "lắc đào": "shake",
    "lắc": "shake",
    "shake": "shake",
    "gật đầu": "nod",
    "gập đầu": "nod",
    "giật đầu": "nod",
    "gật": "nod",
    "nod": "nod",
    "đồng ý": "yes",
    "có": "yes",
    "yes": "yes",
    "không": "no",
    "no": "no",
    "ngủ": SLEEP, "đi ngủ": SLEEP, "nghỉ ngơi": SLEEP, "nghỉ": SLEEP, "sleep": SLEEP,
}

start_http_server(METRICS_PORT)
print(f"Metrics exposed on :{METRICS_PORT}/metrics", flush=True)

print("Loading wake-word model...", flush=True)
oww = Model(wakeword_model_paths=[MIRA_ONNX])
wake_key = list(oww.models.keys())[0]

ZIPFORMER_DIR = "d:/Comp/Qualcom/zipformer-vi"
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


def transcribe(audio_i16, fs):
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    stream = stt.create_stream()
    stream.accept_waveform(fs, audio_f32)
    stt.decode_stream(stream)
    return stream.result.text.strip().lower()

q = queue.Queue()


def callback(indata, frames, time_info, status):
    mono_48k = indata[:, 0]
    mono_16k = resample_poly(mono_48k, 1, DEVICE_FS // FS).astype(np.int16)
    q.put(mono_16k)


robot_busy = threading.Event()


def run_command(motion):
    def _worker():
        print(f"  >>> Running mira-robot replay {motion} ...", flush=True)
        cmd = [
            "ssh", "-i", "C:/Users/LENOVO/.ssh/id_ed25519_unoq",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "arduino@192.168.1.41", f"mira-robot replay {motion} --yes",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "(no output)"
            print("  board:", last_line, flush=True)
            if result.returncode != 0:
                print("  error:", result.stderr.strip(), flush=True)
        finally:
            drain_stale_audio()
            robot_busy.clear()

    robot_busy.set()
    threading.Thread(target=_worker, daemon=True).start()


def drain_stale_audio():
    """Discard any audio that piled up in the queue while we were busy recording/
    transcribing, so the main loop resumes on fresh real-time audio instead of
    working through a backlog of stale chunks (this alone caused multi-second-plus
    unresponsiveness to a new 'Mira' even when the previous utterance didn't match
    any command and no robot motion ran at all)."""
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except queue.Empty:
            break
    if drained:
        print(f"  (dropped {drained} stale audio chunks)", flush=True)


def record_command_window(stream_getter, seconds):
    chunks = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        try:
            chunks.append(stream_getter(timeout=0.3))
        except queue.Empty:
            continue
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


_motions = sorted(set(COMMAND_MAP.values()) - {SLEEP})
duration_label = "indefinitely (say 'Mira ngủ' or 'Mira nghỉ ngơi' to stop)" if TOTAL_DURATION == float("inf") else f"for {TOTAL_DURATION}s"
print(f"Listening {duration_label}. Say 'Mira' then a command word "
      f"({', '.join(_motions)}) ...", flush=True)
t_start = time.perf_counter()
# keep ~1.2s of recent audio so we don't lose the start of a command word
# that overlaps with wake-word detection latency
preroll = collections.deque(maxlen=int(1.2 * FS / FRAME) + 1)

with sd.InputStream(samplerate=DEVICE_FS, channels=1, dtype="int16", blocksize=DEVICE_FRAME, device=DEVICE, callback=callback) as stream:
    while time.perf_counter() - t_start < TOTAL_DURATION:
        try:
            chunk = q.get(timeout=1)
        except queue.Empty:
            continue
        preroll.append(chunk)
        prediction = oww.predict(chunk)
        score = prediction[wake_key]
        WAKE_SCORE.set(score)
        t = time.perf_counter() - t_start
        if score > 0.3:
            print(f"  {t:5.2f}s  wake_score={score:.3f}", flush=True)
        if score > THRESHOLD and robot_busy.is_set():
            print(f"  {t:5.2f}s  wake word heard but robot is still busy with the previous "
                  f"command - ignoring (say it again once it finishes)", flush=True)
            oww.reset()
            continue
        if score > THRESHOLD:
            WAKE_TRIGGERS.inc()
            print(f"  {t:5.2f}s  WAKE WORD DETECTED -> recording command window...", flush=True)
            pre_audio = np.concatenate(list(preroll)) if preroll else np.zeros(0, dtype=np.int16)
            preroll.clear()
            audio_i16 = np.concatenate([pre_audio, record_command_window(q.get, COMMAND_WINDOW_S)])
            peak = int(np.abs(audio_i16).max()) if audio_i16.size else 0
            rms = float(np.sqrt(np.mean(audio_i16.astype(np.float64) ** 2))) if audio_i16.size else 0.0
            COMMAND_AUDIO_PEAK.set(peak)
            COMMAND_AUDIO_RMS.set(rms)
            wav_path = save_debug_wav(audio_i16, FS)
            print(f"  captured {audio_i16.size} samples, peak={peak}, rms={rms:.1f}, saved={wav_path}", flush=True)

            if rms < MIN_COMMAND_RMS:
                print(f"  audio too quiet (rms={rms:.1f} < {MIN_COMMAND_RMS}) - skipping transcription "
                      f"(likely a false wake-word trigger on noise, not real speech)", flush=True)
                COMMAND_RESULTS.labels(motion="skipped_too_quiet").inc()
                oww.reset()
                drain_stale_audio()
                continue

            t0 = time.perf_counter()
            heard = transcribe(audio_i16, FS)
            stt_ms = (time.perf_counter() - t0) * 1000
            print(f"  heard: '{heard}' ({stt_ms:.0f}ms)", flush=True)
            heard_clean = heard.strip(".,!? ")
            heard_words = set(heard_clean.split())
            matched = None
            for word, motion in COMMAND_MAP.items():
                # ultra-common short function words (co/khong/yes/no) must match
                # a whole word, not just any substring, to avoid false positives
                # on unrelated Vietnamese sentences that happen to contain them
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
                subprocess.run(
                    ["ssh", "-i", "C:/Users/LENOVO/.ssh/id_ed25519_unoq",
                     "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                     "arduino@192.168.1.41", "mira-robot replay play-dead --yes"],
                    capture_output=True, text=True, timeout=45,
                )
                oww.reset()
                break
            elif matched:
                COMMAND_RESULTS.labels(motion=matched).inc()
                run_command(matched)
            else:
                COMMAND_RESULTS.labels(motion="unmatched").inc()
                print("  (no known command word recognized)", flush=True)
                drain_stale_audio()
            oww.reset()

print("\nDone listening.", flush=True)
