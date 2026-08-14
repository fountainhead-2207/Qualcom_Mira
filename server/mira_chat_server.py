"""
Conversational fallback server for the voice pipeline, run on the RTX 4090.

board_voice_control.py already recognizes canned commands (wave, dance, ...)
locally with no network hop. When the wake word is heard but no canned
command matched - a free-form remark, question, or small talk directed at
"Mira" - the board instead POSTs the transcribed text here. This server:

  1. asks a small instruct LLM (Qwen2.5-3B-Instruct) for a short Vietnamese
     reply, forcing it to also pick one gesture (existing motion, one of the
     proposed new ones, or "none") to pair with that reply via a JSON schema
  2. synthesizes the reply with Piper (vi_VN-vais1000-medium)
  3. returns {reply_text, motion, audio_wav_base64} for the board to play
     over Bluetooth/aplay while (or just before) running the motion

Not yet wired to real hardware execution or tested with a live speaker -
see docs/uno-q-board.md for status. Motions this can request must already
exist as canned motions on the board (mira-robot replay <name>) - the
proposed new ones (thinking, shrug, ...) need to be recorded first, see
docs/gesture_proposals.md.
"""
import base64
import io
import json
import os
import re
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", "/data/qualcom-robotic/hf-cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from piper import PiperVoice

PORT = 8766
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"
PIPER_MODEL = "/data/qualcom-robotic/piper-voices/vi_VN/vi_VN-vais1000-medium.onnx"

# Existing canned motions plus the proposed-but-not-yet-recorded ones (see
# docs/gesture_proposals.md) - the LLM is told which are real vs aspirational
# so it can be steered toward "none" until those are recorded.
# "thinking" is deliberately absent: board_voice_control.py fires it itself the
# moment a free-form remark is heard, to fill this server's own round-trip. If
# the LLM could also pick it, it would arrive with the answer instead - after
# the waiting it exists to cover.
EXISTING_MOTIONS = ["wave", "dance", "nod", "shake", "play-dead", "clean", "scan",
                    "shrug", "point", "bow", "celebrate", "curious_tilt"]
PROPOSED_MOTIONS = []  # all of docs/gesture_proposals.md is recorded now

SYSTEM_PROMPT = (
    "Bạn là Mira, một cánh tay robot thân thiện, trả lời ngắn gọn bằng tiếng Việt "
    "(1-2 câu, tối đa khoảng 30 từ). Giọng điệu vui vẻ, tự nhiên, như một người bạn nhỏ. "
    "Sau khi trả lời, chọn ĐÚNG MỘT cử chỉ phù hợp nhất với câu trả lời từ danh sách sau, "
    f"hoặc \"none\" nếu không cử chỉ nào phù hợp: {', '.join(EXISTING_MOTIONS + PROPOSED_MOTIONS + ['none'])}.\n"
    "Trả lời CHỈ bằng JSON đúng định dạng: {\"reply\": \"...\", \"motion\": \"...\"}"
)

print(f"Loading {LLM_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
print("Loading Piper voice...", flush=True)
voice = PiperVoice.load(PIPER_MODEL)
print("Ready.", flush=True)


def extract_reply_json(text):
    """Recover {"reply": ..., "motion": ...} from a generation, plus any prose
    that preceded it.

    Two model quirks make naive parsing fail most of the time here:
      - a plain `\\{.*\\}` search is greedy, so when more than one object is
        emitted it spans from the first brace to the last and yields garbage;
        each candidate object is brace-matched individually instead.
      - Qwen very often drops the closing quote on the last value, e.g.
        `{"reply": "Chào!", "motion": "wave}` - malformed JSON that no amount
        of careful extraction will parse. Measured 4-in-6 failures on one
        prompt before this. So when JSON parsing fails, the two fields are
        pulled out by regex, which doesn't care about the missing quote.

    Returns (obj, prose); obj is None only if both routes fail. The prose is
    returned separately because it's the spoken fallback and must never carry
    the raw braces - the TTS reads them out loud otherwise.
    """
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
                        break  # not valid; try the next opening brace
                    if isinstance(obj, dict) and "reply" in obj:
                        return obj, prose
                    break

    reply = re.search(r'"reply"\s*:\s*"(.*?)"\s*[,}]', text, re.DOTALL)
    if reply:
        motion = re.search(r'"motion"\s*:\s*"?([A-Za-z_\-]+)', text)
        return {"reply": reply.group(1),
                "motion": motion.group(1) if motion else "none"}, prose
    return None, prose


def generate_reply(heard_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": heard_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7,
                              pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    obj, prose = extract_reply_json(text)
    if obj is None:
        # Log the raw generation - a silent fallback here was hiding a 2-in-3
        # parse failure rate, which looked like the model refusing to answer.
        print(f"  [unparsed LLM output] {text!r}", flush=True)
        return (prose or "Xin lỗi, tôi chưa nghĩ ra câu trả lời."), "none"
    reply = obj.get("reply", "").strip()
    motion = str(obj.get("motion", "none")).strip()
    if motion not in EXISTING_MOTIONS + PROPOSED_MOTIONS:
        motion = "none"
    return (reply or prose or "Xin lỗi, tôi chưa nghĩ ra câu trả lời."), motion


TRAIL_SILENCE_S = 0.3


def synthesize(text):
    # "Mira" written plainly gets phonemized with English phones (schwa + ɹ)
    # that this Vietnamese acoustic model never saw in training, and it
    # improvises something close to the real word "mỉa" instead. Splitting
    # into two space-separated words forces espeak to read it as the plain
    # Vietnamese syllables "mi" + "ra" the model actually knows - confirmed by
    # comparing EspeakPhonemizer's output for both spellings before making
    # this change. A space reads at a natural pace; a hyphen produces the
    # same phonemes but the duration model rushes it, like a single word.
    speakable = re.sub(r"\bmira\b", "Mi ra", text, flags=re.IGNORECASE)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(speakable, wf)

    # A2DP playback pops audibly at the very end of the clip - classic ALSA
    # buffer-underrun-on-drain artifact when the PCM stream stops right as
    # the last real samples are consumed. Padding with trailing silence gives
    # the Bluetooth link something harmless to drain instead.
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    pad_frames = int(TRAIL_SILENCE_S * params.framerate)
    frames += b"\x00" * (pad_frames * params.nchannels * params.sampwidth)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(frames)
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        heard_text = body.get("text", "").strip()
        if not heard_text:
            self.send_response(400)
            self.end_headers()
            return

        reply, motion = generate_reply(heard_text)
        wav_bytes = synthesize(reply)
        response = {
            "reply_text": reply,
            "motion": None if motion == "none" else motion,
            "motion_is_proposed": motion in PROPOSED_MOTIONS,
            "audio_wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"Listening on :{PORT}/chat", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
