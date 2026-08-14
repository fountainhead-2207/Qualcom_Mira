"""
Board-native voice control via continuous rolling-window ASR.

Replaces the openWakeWord + separate-command-window design (board_wakeword.py,
kept as a fallback). That design failed in practice: the custom "mira" model was
trained on synthetic Piper TTS clips of the isolated word, so it only fired on
an isolated, fully-articulated "Mira" and scored a flat ~0.0005 whenever "mira"
appeared inside a fluent phrase like "mira vẫy tay" - measured 0/40 recall on
real samples of this speaker, with the base feature pipeline verified healthy.

Instead, the Zipformer ASR model that already runs here transcribes those same
phrases exactly right, so it does both jobs at once: a rolling window is
transcribed continuously and a trigger requires the wake word AND a command
word in the same transcript. Measured on the board (4 cores, num_threads=2):

    window 1.0s -> 182ms    window 2.0s -> 339ms    window 3.0s -> 500ms
    window 1.5s -> 263ms    window 2.5s -> 426ms

so a 2.5s window on a 0.5s hop leaves headroom, and end-to-end latency is one
hop plus one decode (~0.9s from the end of speech) instead of wake-detect plus
a fixed 2.8s recording plus a decode.

Audio capture is unchanged: BOYAMIC only offers S24_3LE/2ch/48kHz with real
signal on the RIGHT channel only, read from a persistent `arecord` pipe and
downsampled to 16kHz mono.
"""
import base64
import collections
import difflib
import json
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx
from prometheus_client import Gauge, Counter, start_http_server

# mira_chat_server.py on the RTX 4090. The board's own network blocks outbound
# SSH (port 234, and even port 22) to anywhere but port 443, so it can't tunnel
# there directly - reached instead through the Windows PC on the same LAN,
# which already has SSH access and opens `ssh -N -L 0.0.0.0:8766:localhost:8766
# 61.28.228.23` (bound to all interfaces, not just localhost, specifically so
# the board can reach it).
CHAT_SERVER_URL = "http://192.168.1.32:8766/chat"
CHAT_TIMEOUT_S = 8

# JBL Go 4 speaker, paired via `bluealsa` (bluez-alsa-utils) - PipeWire/
# WirePlumber never registered an A2DP sink profile with bluetoothd on this
# board (bluetoothd's own log: "a2dp-sink profile connect failed... Protocol
# not available") even with Experimental mode on and a fresh re-pair;
# bluealsa did immediately. PipeWire is now stopped/masked entirely to avoid
# it competing with bluealsa for the audio hardware. The `plug:` wrapper
# auto-converts Piper's 22050Hz mono output to the link's 48kHz stereo -
# no manual resampling needed.
BT_SPEAKER_MAC = "4C:3C:8F:3C:42:EE"
BT_APLAY_DEVICE = f"plug:'bluealsa:DEV={BT_SPEAKER_MAC},PROFILE=a2dp'"

# Played automatically while waiting on the chat server, not something the LLM
# gets to choose - see chat_and_speak().
THINKING_MOTION = "thinking"

METRICS_PORT = 9103
AUDIO_RMS = Gauge("board_audio_rms", "RMS of the most recent audio hop")
DECODE_MS = Gauge("board_decode_ms", "Latency of the most recent rolling-window ASR decode")
TRIGGERS = Counter("board_voice_triggers_total", "Transcripts containing the wake word")
RESULTS = Counter("board_voice_recognized_total", "Command recognition outcomes", ["motion"])

ZIPFORMER_DIR = "/home/arduino/zipformer-vi"
ALSA_DEVICE = "hw:CARD=BOYAMIC,DEV=0"
DEVICE_FS = 48000
FS = 16000
CHANNELS = 2
BYTES_PER_SAMPLE = 3  # S24_3LE
RIGHT_CHANNEL = 1  # LEFT is silent on this mic

WINDOW_S = 2.5
HOP_S = 0.5
READ_CHUNK_S = 0.25
READ_CHUNK_BYTES = int(READ_CHUNK_S * DEVICE_FS) * CHANNELS * BYTES_PER_SAMPLE

# Skip decoding near-silent windows: cheap, and stops the ASR from inventing
# plausible text on room noise (a hallucinated sentence containing "có" once
# fired a real motion with nobody speaking).
MIN_RMS = 120.0

TOTAL_DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else float("inf")

SLEEP = "__sleep__"

# "Mira" is a foreign name to a Vietnamese ASR model, so it comes back mangled
# and inconsistent - observed in one 10s recording: mira, vira, bida, mi rươ,
# willer, willia, quy lơ, quy lê. An exact string list can't keep up, so the
# wake word is matched fuzzily (below) with these as known-good anchors.
WAKE_WORDS = ("mira", "mi ra", "mê ra", "mưa ra", "mila", "my ra", "vira",
              "bida", "willer", "willia", "mi rươ", "vi ra", "bi ra")
WAKE_FUZZ = 0.6  # difflib ratio against "mira" for a single token or token pair

COMMAND_MAP = {
    "vẫy tay": "wave", "quay tay": "wave", "bảy tay": "wave", "bẫy tay": "wave", "vẫy": "wave", "wave": "wave",
    "nhảy": "dance", "nháy": "dance", "múa": "dance", "mua": "dance", "dance": "dance",
    "dọn dẹp": "clean", "dọn": "clean", "clean": "clean",
    "quét": "scan", "scan": "scan",
    "lắc đầu": "shake", "lắt đầu": "shake", "lắc đào": "shake", "lắc": "shake", "shake": "shake",
    "gật đầu": "nod", "gập đầu": "nod", "giật đầu": "nod", "gật": "nod", "nod": "nod",
    "đồng ý": "yes", "có": "yes", "yes": "yes",
    "không": "no", "no": "no",
    "ngủ": SLEEP, "đi ngủ": SLEEP, "nghỉ ngơi": SLEEP, "nghỉ": SLEEP, "sleep": SLEEP,
}

DEBUG_DIR = Path("/home/arduino/voice_debug_logs")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
_debug_counter = [0]


def save_debug_wav(samples_i16, fs):
    _debug_counter[0] += 1
    path = DEBUG_DIR / f"trigger_{_debug_counter[0]}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(samples_i16.tobytes())
    return str(path)


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


def transcribe(samples_i16):
    stream = stt.create_stream()
    stream.accept_waveform(FS, samples_i16.astype(np.float32) / 32768.0)
    stt.decode_stream(stream)
    return stream.result.text.strip().lower()


def match_command(text):
    """Returns (motion, is_multiword_phrase) or (None, False)."""
    words = set(text.strip(".,!? ").split())
    for word, motion in COMMAND_MAP.items():
        # Short function words (có/không/gật/lắc) must match a whole word, or
        # any sentence containing them incidentally fires a motion.
        if len(word) <= 4 and " " not in word:
            if word in words:
                return motion, False
        elif word in text:
            return motion, " " in word
    return None, False


def heard_wake_word(text):
    if any(w in text for w in WAKE_WORDS):
        return True
    tokens = text.strip(".,!? ").split()
    for i, tok in enumerate(tokens):
        if difflib.SequenceMatcher(None, tok, "mira").ratio() >= WAKE_FUZZ:
            return True
        if i + 1 < len(tokens):
            pair = tok + tokens[i + 1]
            if difflib.SequenceMatcher(None, pair, "mira").ratio() >= WAKE_FUZZ:
                return True
    return False


chat_busy = threading.Event()


def chat_and_speak(heard_text):
    """Pose "thinking", ask mira_chat_server.py for a reply + gesture, speak the
    reply, then run that gesture.

    "thinking" is fired here rather than being offered to the LLM as one of the
    gestures it can pick: its whole purpose is to fill the round-trip, so it has
    to start before the request, not arrive with the answer. The arm holds the
    pose while it waits - the geared STS3215s keep a balanced posture even after
    mira-robot disconnects and drops torque (verified on the real arm).

    Ordering is forced by mira-robot's exclusive robot lock: the reply's gesture
    can't start until the thinking replay has released it, so it's joined below.
    Speech doesn't contend for that lock, so the answer still plays as soon as
    it arrives rather than waiting for the arm.
    """
    robot_busy.set()

    def _pose_thinking():
        try:
            replay_motion(THINKING_MOTION)
        except Exception as e:
            print(f"  (thinking pose failed: {e})", flush=True)

    thinking = threading.Thread(target=_pose_thinking, daemon=True)
    thinking.start()

    result = None
    try:
        req = urllib.request.Request(
            CHAT_SERVER_URL,
            data=json.dumps({"text": heard_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  chat server unreachable ({e}) - skipping", flush=True)
    finally:
        chat_busy.clear()

    if result is not None:
        print(f"  Mira: {result['reply_text']}", flush=True)
        wav_bytes = base64.b64decode(result["audio_wav_base64"])
        wav_path = DEBUG_DIR / "chat_reply.wav"
        wav_path.write_bytes(wav_bytes)
        subprocess.run(["aplay", "-D", BT_APLAY_DEVICE, str(wav_path)], capture_output=True)

    thinking.join(timeout=90)
    robot_busy.clear()

    if result is None:
        return
    motion = result.get("motion")
    if motion and not result.get("motion_is_proposed"):
        run_command(motion)
    elif motion:
        print(f"  (would run '{motion}' but it hasn't been recorded yet)", flush=True)


robot_busy = threading.Event()


def replay_motion(motion):
    """Run one motion to completion. mira-robot takes an exclusive lock on the
    robot, so only one of these can be in flight at a time."""
    print(f"  >>> mira-robot replay {motion} ...", flush=True)
    result = subprocess.run(
        ["mira-robot", "replay", motion, "--yes"],
        capture_output=True, text=True, timeout=90,
    )
    last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "(no output)"
    print("  ", last, flush=True)
    if result.returncode != 0:
        print("  error:", result.stderr.strip(), flush=True)


def run_command(motion):
    def _worker():
        try:
            replay_motion(motion)
        finally:
            robot_busy.clear()

    robot_busy.set()
    threading.Thread(target=_worker, daemon=True).start()


start_http_server(METRICS_PORT)
print(f"Metrics on :{METRICS_PORT}/metrics", flush=True)
label = ("indefinitely (say 'Mira ngủ' to stop)" if TOTAL_DURATION == float("inf")
         else f"for {TOTAL_DURATION}s")
print(f"Listening {label} - rolling {WINDOW_S}s window, {HOP_S}s hop "
      f"(device={ALSA_DEVICE})", flush=True)
print(f"Say the wake word and a command together, e.g. 'Mira vẫy tay'.", flush=True)

arecord = subprocess.Popen(
    ["arecord", "-D", ALSA_DEVICE, "-f", "S24_3LE", "-c", str(CHANNELS),
     "-r", str(DEVICE_FS), "-t", "raw"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)

rolling = collections.deque(maxlen=int(WINDOW_S * FS))
t_start = time.perf_counter()
last_decode = 0.0
last_text = ""

try:
    while time.perf_counter() - t_start < TOTAL_DURATION:
        raw = arecord.stdout.read(READ_CHUNK_BYTES)
        if len(raw) < READ_CHUNK_BYTES:
            print("arecord stream ended unexpectedly", flush=True)
            break
        rolling.extend(to_16k_int16(decode_24bit_right_channel(raw)))

        if time.perf_counter() - last_decode < HOP_S:
            continue
        last_decode = time.perf_counter()

        if len(rolling) < int(1.0 * FS):
            continue
        window = np.fromiter(rolling, dtype=np.int16, count=len(rolling))
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        AUDIO_RMS.set(rms)
        if rms < MIN_RMS:
            continue

        t0 = time.perf_counter()
        text = transcribe(window)
        DECODE_MS.set((time.perf_counter() - t0) * 1000)
        if not text or text == last_text:
            continue
        last_text = text

        t = time.perf_counter() - t_start
        heard_wake = heard_wake_word(text)
        matched, is_phrase = match_command(text)
        # A two-word command phrase ("vẫy tay", "lắc đầu") is specific enough to
        # act on by itself; the ASR mangles the wake word far more often than it
        # mangles these. Single short command words still need the wake word so
        # overheard conversation can't drive the arm. Free-form chat needs the
        # wake word too, for the same reason.
        should_act = (matched is not None and (heard_wake or is_phrase)) or heard_wake
        print(f"  {t:6.2f}s  heard: '{text}'"
              f"{'  [WAKE]' if heard_wake else ''}"
              f"{'  cmd=' + matched if matched else ''}", flush=True)
        if not should_act:
            continue
        TRIGGERS.inc()

        if robot_busy.is_set():
            print("  robot still busy with the previous command - ignoring", flush=True)
            continue

        if matched == SLEEP:
            RESULTS.labels(motion="sleep").inc()
            print("  >>> going to sleep - resting, then stopping", flush=True)
            save_debug_wav(window, FS)
            subprocess.run(["mira-robot", "replay", "play-dead", "--yes"],
                           capture_output=True, text=True, timeout=60)
            break
        elif matched:
            RESULTS.labels(motion=matched).inc()
            save_debug_wav(window, FS)
            run_command(matched)
            # Drop the buffer so the same utterance can't retrigger on the
            # next overlapping window.
            rolling.clear()
            last_text = ""
        elif chat_busy.is_set():
            print("  already waiting on a chat reply - ignoring", flush=True)
        else:
            RESULTS.labels(motion="chat").inc()
            save_debug_wav(window, FS)
            chat_busy.set()
            threading.Thread(target=chat_and_speak, args=(text,), daemon=True).start()
            rolling.clear()
            last_text = ""
finally:
    arecord.terminate()

print("\nDone listening.", flush=True)
