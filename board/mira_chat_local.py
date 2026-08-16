#!/usr/bin/env python3
"""Bản chat server chạy NGAY TRÊN BOARD - không cần 4090, không cần mạng.

Thay thế cho `server/mira_chat_server.py` khi muốn cả hệ thống gói trong một
board. Giữ nguyên hợp đồng HTTP để `board_voice_control.py` không phải sửa gì
ngoài một dòng URL:

    POST /chat {"text": "..."} -> {reply_text, motion, task, llm_s, tts_s,
                                   audio_wav_base64}

Hai thứ được giữ nguyên có chủ ý, vì cả hai đều là kết quả đo/nghe thật chứ
không phải lựa chọn tuỳ tiện:

  * **Prompt y hệt bản trên server** (đọc từ /llm/mira_sys.txt và
    /llm/mira_fewshot.txt). Đổi prompt thì hành vi lệch khỏi bộ eval 86 câu đã
    chấm, và không so sánh board với server được nữa.
  * **Toàn bộ phần hậu kỳ âm thanh** của bản server: tách "Mira" thành "Mi ra"
    (model tiếng Việt phát âm "Mira" thành "mỉa"), 0.15s lặng đầu (A2DP nuốt
    ~100ms đầu), 0.3s lặng cuối (chống tiếng "tịt" do ALSA cạn buffer), ghép
    mệnh đề với khe 0.10s và length_scale 1.12 - đây là bản người dùng đã chọn
    sau khi nghe thử 5 biến thể.

Điều kiện chạy: llama-server đã bật sẵn với model và **cache_prompt** - trên CPU
này prompt 1231 token mất ~70s để nhai, nếu nhai lại mỗi câu thì vô dụng. Xem
`docs/benchmarks_board_vs_server.md`.

    /llm/llama-b10444/llama-server -m /llm/qwen2.5-0.5b-instruct-q4_0.gguf \
        -t 4 -c 2048 --host 127.0.0.1 --port 8099 &
    python3 mira_chat_local.py
"""
import base64
import io
import json
import os
import re
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from piper import PiperVoice, SynthesisConfig

PORT = int(os.environ.get("MIRA_LOCAL_CHAT_PORT", "8770"))
LLM_URL = os.environ.get("MIRA_LLM_URL", "http://127.0.0.1:8099/completion")
PIPER_MODEL = os.environ.get(
    "MIRA_PIPER", "/llm/voices/vi_VN-vais1000-medium.onnx")
SYS_PATH = os.environ.get("MIRA_SYS_PROMPT", "/llm/mira_sys.txt")
FEWSHOT_PATH = os.environ.get("MIRA_FEWSHOT", "/llm/mira_fewshot.txt")

# Giống hệt bản server. Model nào trả về cử chỉ ngoài danh sách này thì coi như
# "none" - động tác không có thật trên board sẽ làm mira-robot lỗi.
EXISTING_MOTIONS = ["wave", "dance", "nod", "shake", "play-dead", "scan",
                    "shrug", "point", "bow", "celebrate", "curious_tilt"]

TASK_SHAPE = re.compile(r"^pick up (the|a|an) \S+", re.IGNORECASE)
TASK_VAGUE = re.compile(
    r"^pick up (the|a|an) "
    r"(item|items|object|objects|thing|things|stuff|it|that|this|one)\b",
    re.IGNORECASE)

SYSTEM_PROMPT = open(SYS_PATH, encoding="utf-8").read()
FEW_SHOT = open(FEWSHOT_PATH, encoding="utf-8").read()

print("Loading Piper voice...", flush=True)
voice = PiperVoice.load(PIPER_MODEL)

TRAIL_SILENCE_S = 0.3
LEAD_SILENCE_S = 0.15
_UNSPEAKABLE = re.compile("[^" r"0-9A-Za-zÀ-ỹ" r"\s.,!?:;'\"()\-" "]+")


def strip_unspeakable(text):
    return re.sub(r"\s{2,}", " ", _UNSPEAKABLE.sub(" ", text)).strip()


def _trim_edges(a, rate, thresh=250, keep=0.02, trim_head=True):
    if a.size == 0:
        return a
    loud = np.where(np.abs(a) > thresh)[0]
    if loud.size == 0:
        return a[:0]
    pad = int(keep * rate)
    lo = max(0, loud[0] - pad) if trim_head else 0
    return a[lo:min(a.size, loud[-1] + pad)]


def synthesize(text):
    speakable = strip_unspeakable(re.sub(r"\bmira\b", "Mi ra", text, flags=re.I))
    parts, rate = [], voice.config.sample_rate
    for chunk in voice.synthesize(speakable, syn_config=SynthesisConfig(length_scale=1.12)):
        rate = chunk.sample_rate
        parts.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

    gap = np.zeros(int(0.10 * rate), dtype=np.int16)
    joined = [np.zeros(int(LEAD_SILENCE_S * rate), dtype=np.int16)]
    for i, part in enumerate(parts):
        if i:
            joined.append(gap)
        joined.append(_trim_edges(part, rate, trim_head=(i > 0)))
    samples = np.concatenate(joined) if joined else np.zeros(0, dtype=np.int16)
    samples = np.concatenate([samples, np.zeros(int(TRAIL_SILENCE_S * rate), dtype=np.int16)])

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return out.getvalue()


def extract_reply_json(text):
    """Giống bản server: Qwen hay bỏ dấu nháy đóng ở giá trị cuối, nên khi
    json.loads thất bại thì bóc bằng regex - regex không quan tâm thiếu nháy."""
    brace = text.find("{")
    prose = (text if brace == -1 else text[:brace]).strip()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "reply" in obj:
                        return obj, prose
                    break
    m = re.search(r'"reply"\s*:\s*"(.*?)"\s*[,}]', text, re.DOTALL)
    if m:
        motion = re.search(r'"motion"\s*:\s*"?([A-Za-z_\-]+)', text)
        return {"reply": m.group(1), "motion": motion.group(1) if motion else "none"}, prose
    return None, prose


def complete(heard_text, max_tokens=60):
    body = json.dumps({
        "prompt": f"{SYSTEM_PROMPT}\n\n{FEW_SHOT}\nNgười dùng: {heard_text}\nMira:",
        "n_predict": max_tokens,
        "temperature": 0.6, "top_p": 0.8, "top_k": 20,
        # Bắt buộc: prompt 1231 token, nhai lại mỗi câu là ~70s trên CPU này.
        "cache_prompt": True,
        "stop": ["\nNgười dùng:", "\n\n"],
    }).encode("utf-8")
    req = urllib.request.Request(LLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp).get("content", "")


def generate_reply(heard_text):
    text = complete(heard_text)
    obj, prose = extract_reply_json(text)
    if obj is None:
        print(f"  [không bóc được JSON] {text!r}", flush=True)
        return (prose or "Xin lỗi, mình chưa nghĩ ra câu trả lời."), "none", None
    reply = obj.get("reply", "").strip()
    motion = str(obj.get("motion", "none")).strip()
    if motion not in EXISTING_MOTIONS:
        motion = "none"
    task = obj.get("task")
    task = task.strip() if isinstance(task, str) and task.strip() else None
    if task and (not TASK_SHAPE.match(task) or TASK_VAGUE.match(task)):
        print(f"  [bỏ task không hợp lệ] {task!r}", flush=True)
        task = None
    return (reply or prose or "Xin lỗi, mình chưa nghĩ ra câu trả lời."), motion, task


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}", flush=True)

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        heard = body.get("text", "").strip()
        if not heard:
            self.send_response(400)
            self.end_headers()
            return

        t0 = time.perf_counter()
        reply, motion, task = generate_reply(heard)
        llm_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        wav = synthesize(reply)
        tts_s = time.perf_counter() - t0

        payload = json.dumps({
            "reply_text": reply,
            "motion": None if motion == "none" else motion,
            "motion_is_proposed": False,
            "llm_s": round(llm_s, 3),
            "tts_s": round(tts_s, 3),
            "task": task,
            "audio_wav_base64": base64.b64encode(wav).decode("ascii"),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    # Nạp sẵn cache prompt lúc khởi động, để câu ĐẦU TIÊN của người dùng không
    # phải trả giá ~70s nhai prompt.
    print("Đang hâm nóng cache prompt (một lần, có thể mất ~1 phút)...", flush=True)
    t0 = time.perf_counter()
    try:
        complete("chào mira", max_tokens=8)
        print(f"  xong sau {time.perf_counter()-t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"  KHÔNG hâm được cache: {exc}", flush=True)
    print(f"Ready. Listening on :{PORT}/chat", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
