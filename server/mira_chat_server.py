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
# Qwen2.5-3B followed the persona rules poorly - it slipped into "tao", mixed
# Chinese mid-sentence, invented a bitcoin price, and picked gestures close to
# randomly. This is a generation newer at a similar size, and the -Instruct-2507
# variant has no thinking mode, so replies stay short without extra prompting.
#
# The 8B/14B FP8 checkpoints already sitting in /data/KV_cache/LMbench are a
# dead end for plain transformers: FP8 needs a Triton kernel from the `kernels`
# package, and installing it broke transformers outright (its newer
# LayerRepository API is incompatible with transformers 5.5.4). AWQ 4-bit would
# fit VRAM better than this does but needs `gptqmodel`, another compiled-CUDA
# dependency in the same venv that serves MolmoAct2 - not worth the risk. Going
# bigger cleanly means vLLM in its own venv serving those FP8 files.
LLM_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# Only matters for models with a thinking mode (Qwen3.5 emits <think>...</think>
# by default, which would blow the latency budget several times over). Harmless
# for -Instruct-2507, which has no such mode.
ENABLE_THINKING = False
PIPER_MODEL = "/data/qualcom-robotic/piper-voices/vi_VN/vi_VN-vais1000-medium.onnx"

# Existing canned motions plus the proposed-but-not-yet-recorded ones (see
# docs/gesture_proposals.md) - the LLM is told which are real vs aspirational
# so it can be steered toward "none" until those are recorded.
# "thinking" is deliberately absent: board_voice_control.py fires it itself the
# moment a free-form remark is heard, to fill this server's own round-trip. If
# the LLM could also pick it, it would arrive with the answer instead - after
# the waiting it exists to cover.
EXISTING_MOTIONS = ["wave", "dance", "nod", "shake", "play-dead", "scan",
                    "shrug", "point", "bow", "celebrate", "curious_tilt"]
PROPOSED_MOTIONS = []  # all of docs/gesture_proposals.md is recorded now

# Spelling out when each gesture applies matters more than listing them: with
# only names to go on, the model reached for "wave" on 11 of 24 test prompts and
# never once picked shrug, bow or point - including for "I don't know which
# company to choose", which is the textbook shrug.
GESTURE_GUIDE = [
    ("wave", "chào hỏi lúc mới gặp"),
    ("bow", "chào tạm biệt, hoặc khi được cảm ơn"),
    ("celebrate", "người dùng có tin vui, thành công, đáng chúc mừng"),
    ("dance", "người dùng rủ nhảy, hoặc cực kỳ vui"),
    ("shrug", "KHÔNG biết, không chắc, không có thông tin, hoặc không làm được"),
    ("curious_tilt", "tò mò, ngạc nhiên, hoặc đang hỏi lại người dùng"),
    ("point", "nhắc tới một vật cụ thể ở đâu đó"),
    ("nod", "đồng ý, xác nhận, hoặc đang lắng nghe an ủi"),
    ("shake", "phủ định, từ chối"),
    ("scan", "đang tìm, đang quan sát xung quanh"),
    ("play-dead", "giả vờ hết pin, làm trò cho vui"),
]

SYSTEM_PROMPT = (
    "Bạn là Mira - một cánh tay robot nhỏ, tinh nghịch và hài hước. Bạn thích trêu "
    "vui, hay pha một câu tếu, nhưng luôn tử tế và không bao giờ thô.\n"
    "\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. Chỉ nói TIẾNG VIỆT. Tuyệt đối không dùng tiếng Anh, tiếng Trung, emoji, "
    "hay ký hiệu lạ - câu trả lời sẽ được đọc thành tiếng nên những thứ đó nghe rất kỳ.\n"
    "2. Ngắn gọn: 1-2 câu, tối đa khoảng 25 từ.\n"
    "3. Gọi người dùng là 'bạn'. MIRA LÀ TÊN CỦA BẠN - không bao giờ gọi người "
    "dùng là Mira.\n"
    "4. Xưng 'mình' hoặc 'tôi'. Không dùng 'mày', 'tao'.\n"
    "5. Bạn có một cánh tay robot đặt trên bàn làm việc. Bạn CÓ THỂ thử cầm, nhặt, "
    "hoặc di chuyển những vật nhỏ đang ở trên bàn đó. Khi người dùng nhờ làm việc "
    "như vậy, hãy nhận lời một cách vui vẻ và điền thêm trường \"task\" (xem dưới).\n"
    "   Nhưng bạn KHÔNG rời khỏi bàn, không đi lấy đồ ở nơi khác, không bật đèn, "
    "không gọi điện, không điều khiển thiết bị nào khác. Những việc đó thì thừa nhận "
    "là không làm được (và chọn cử chỉ shrug).\n"
    "6. Nếu không biết điều gì đó (thời tiết, giá cả, tin tức, chuyện tương lai), "
    "hãy thật thà nói là không biết theo kiểu tinh nghịch. TUYỆT ĐỐI KHÔNG bịa số "
    "liệu hay thông tin.\n"
    "7. Khi nhận làm việc gì, nói ở thể SẮP LÀM (\"để mình nhặt cho\", \"mình thử "
    "nha\"). TUYỆT ĐỐI KHÔNG nói như đã làm xong (\"mình vừa nhặt rồi\", \"đã tìm "
    "thấy rồi\") - lúc trả lời thì bạn chưa làm gì cả.\n"
    "8. Bạn KHÔNG nhìn thấy gì cả (không có mắt). Nếu được hỏi vật gì ở đâu, đừng "
    "nói là đã thấy - hãy thừa nhận không biết, hoặc đề nghị thử nhặt nếu người "
    "dùng chỉ chỗ.\n"
    "\n"
    "Sau đó chọn ĐÚNG MỘT cử chỉ khớp nhất với câu trả lời:\n"
    + "\n".join(f"- {name}: {when}" for name, when in GESTURE_GUIDE
                if name in EXISTING_MOTIONS + PROPOSED_MOTIONS)
    + "\n- none: không cử chỉ nào phù hợp\n"
    "\n"
    "Đừng mặc định chọn wave - chỉ dùng wave khi thật sự đang chào hỏi.\n"
    "\n"
    "NẾU người dùng nhờ cầm/nhặt/di chuyển một vật trên bàn, thêm trường \"task\": "
    "mô tả việc đó bằng MỘT CÂU TIẾNG ANH, BẮT BUỘC bắt đầu bằng \"pick up the\", "
    "theo mẫu \"pick up the <vật> and put it on the table\". Cánh tay chỉ biết gắp "
    "và đặt - không có việc gì khác điền vào đây được.\n"
    "Bỏ hẳn trường \"task\" khi người dùng chỉ hỏi, chỉ trò chuyện, rủ nhảy, hay "
    "nhờ việc mà bạn không làm được.\n"
    "\n"
    "Trả lời CHỈ bằng JSON đúng định dạng, không thêm gì khác:\n"
    "{\"reply\": \"...\", \"motion\": \"...\"}\n"
    "hoặc khi cần thao tác vật thể:\n"
    "{\"reply\": \"...\", \"motion\": \"...\", \"task\": \"pick up the ... and put it on the table\"}"
)

# A 3B model follows demonstrated behaviour far more reliably than stated rules.
# Rules alone still produced "tao đang chơi điện thoại", a stray "Chị", Chinese
# mid-sentence, and shake for a goodbye; each example below pins down one of
# those, plus the JSON shape.
# Balance matters as much as content: an earlier set with two "I can't do that"
# examples out of six taught it to refuse almost everything - it answered "hôm
# nay tôi mệt quá" with "mình chỉ là cánh tay robot, không có phép màu đâu", and
# read a plain goodbye as a request it couldn't fulfil. One refusal example is
# enough, and the warm ones sit last because the closest examples pull hardest.
FEW_SHOT = [
    # Off-table requests are the refusal case now that on-table manipulation is
    # real. The old example here was "fetch me a glass of water", which became
    # ambiguous - a glass sitting on the table is exactly what it can pick up.
    ("mira ơi bật đèn lên đi",
     '{"reply": "Mình có tay chứ không có công tắc, cái đó bạn tự bật nha!", '
     '"motion": "shrug"}'),
    ("tôi đánh rơi cái tua vít mà đang đau lưng không nhặt được, bạn nhặt giúp mình nhé",
     '{"reply": "Để mình nhặt cho, bạn ngồi nghỉ đi nha!", "motion": "nod", '
     '"task": "pick up the screwdriver and put it on the table"}'),
    ("chào mira",
     '{"reply": "Chào bạn! Mình đây, sẵn sàng quậy rồi!", "motion": "wave"}'),
    ("tôi vừa thi đậu rồi mira ơi",
     '{"reply": "Oa, giỏi quá! Cho mình ăn mừng một cái nhé!", "motion": "celebrate"}'),
    ("mira ơi hôm nay trời đẹp không",
     '{"reply": "Mình ở trong nhà suốt nên không biết, bạn kể mình nghe đi!", '
     '"motion": "curious_tilt"}'),
    ("tạm biệt mira hẹn mai gặp lại",
     '{"reply": "Mai gặp lại nha bạn, mình chờ đó!", "motion": "bow"}'),
    ("mira ơi hôm nay tôi mệt quá",
     '{"reply": "Thương bạn quá, nghỉ chút đi rồi mình kể chuyện vui cho nghe!", '
     '"motion": "nod"}'),
]

print(f"Loading {LLM_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
print("Loading Piper voice...", flush=True)
voice = PiperVoice.load(PIPER_MODEL)
print("Ready.", flush=True)


# MolmoAct2's fine-tune was trained on instructions of the form "pick up the
# screwdriver and put it on the black workspace" - see docs/gpu-training-server.md.
# Anything not shaped like that is not something it can act on.
TASK_SHAPE = re.compile(r"^pick up (the|a|an) \S+", re.IGNORECASE)


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


CLASSIFY_PROMPT = (
    "Bạn chọn cử chỉ cho một cánh tay robot. Dưới đây là câu robot vừa nói.\n"
    "Chọn ĐÚNG MỘT cử chỉ khớp nhất:\n"
    + "\n".join(f"- {name}: {when}" for name, when in GESTURE_GUIDE
                if name in EXISTING_MOTIONS + PROPOSED_MOTIONS)
    + "\n- none: không cử chỉ nào phù hợp\n"
    "\nChỉ trả về tên cử chỉ, không giải thích, không thêm gì khác."
)

CLASSIFY_EXAMPLES = [
    ("Chào bạn! Mình đây, sẵn sàng quậy rồi!", "wave"),
    ("Mai gặp lại nha bạn, mình chờ đó!", "bow"),
    ("Ngủ ngon nha bạn, mai lại chơi tiếp!", "bow"),
    ("Oa, giỏi quá! Chúc mừng bạn nha!", "celebrate"),
    ("Mình mù tịt luôn, không biết đâu bạn ơi!", "shrug"),
    ("Bưng nước thì mình chịu rồi, chỉ có một cánh tay thôi!", "shrug"),
    ("Thương bạn quá, nghỉ chút đi nha!", "nod"),
    ("Ủa thật hả bạn? Kể mình nghe đi!", "curious_tilt"),
    ("Cái kia kìa, ngay đó đó!", "point"),
    ("Nhảy luôn đi, mình quậy với bạn!", "dance"),
]


def choose_gesture(reply_text):
    """Pick the gesture in a separate pass, given only the finished reply.

    Asking one 3B model to compose a reply and classify it into a dozen gesture
    labels in the same breath made the classification near-random: shake (a
    head-shake, i.e. "no") came back for goodbyes, a pay rise and words of
    comfort alike. Split out, with greedy decoding so the same reply always maps
    to the same gesture, it has one easy job instead of two.
    """
    messages = [{"role": "system", "content": CLASSIFY_PROMPT}]
    for example_reply, example_motion in CLASSIFY_EXAMPLES:
        messages.append({"role": "user", "content": example_reply})
        messages.append({"role": "assistant", "content": example_motion})
    messages.append({"role": "user", "content": reply_text})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=ENABLE_THINKING)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().lower()
    for name in EXISTING_MOTIONS + PROPOSED_MOTIONS:
        if name.replace("_", "").replace("-", "") in answer.replace("_", "").replace("-", ""):
            return name
    return "none"


def generate_reply(heard_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_reply in FEW_SHOT:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_reply})
    messages.append({"role": "user", "content": heard_text})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=ENABLE_THINKING)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # 0.6 rather than 0.7: the examples pin the persona down, and the extra
        # randomness mostly showed up as rule violations rather than variety.
        out = model.generate(**inputs, max_new_tokens=140, do_sample=True, temperature=0.6,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    obj, prose = extract_reply_json(text)
    if obj is None:
        # Log the raw generation - a silent fallback here was hiding a 2-in-3
        # parse failure rate, which looked like the model refusing to answer.
        print(f"  [unparsed LLM output] {text!r}", flush=True)
        return (prose or "Xin lỗi, tôi chưa nghĩ ra câu trả lời."), "none", None
    reply = obj.get("reply", "").strip()
    motion = str(obj.get("motion", "none")).strip()
    if motion not in EXISTING_MOTIONS + PROPOSED_MOTIONS:
        motion = "none"
    # A manipulation request routes to MolmoAct2 instead of a canned gesture.
    # Validated rather than trusted: the model emitted "dance with the user" as a
    # task, and MolmoAct2 only does pick-and-place - anything else handed to it is
    # meaningless. Its training tasks all read "pick up the X and put it ...", so
    # that prefix is the gate.
    task = obj.get("task")
    task = task.strip() if isinstance(task, str) and task.strip() else None
    if task and not TASK_SHAPE.match(task):
        print(f"  [dropped malformed task] {task!r}", flush=True)
        task = None
    return (reply or prose or "Xin lỗi, tôi chưa nghĩ ra câu trả lời."), motion, task


TRAIL_SILENCE_S = 0.3

# The system prompt forbids emoji and non-Vietnamese script, but the model
# leaked both in testing ("À yeah, let's dance! 🎉", "làm很多事情呢！"), and
# whatever slips through gets read aloud. Strip it here so a prompt-following
# lapse can't reach the speaker.
_UNSPEAKABLE = re.compile(
    "[^"
    r"0-9A-Za-zÀ-ỹ"          # latin + the full Vietnamese range
    r"\s.,!?:;'\"()\-"       # punctuation espeak handles sensibly
    "]+"
)


def strip_unspeakable(text):
    cleaned = _UNSPEAKABLE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


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
    speakable = strip_unspeakable(speakable)
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

        reply, motion, task = generate_reply(heard_text)
        wav_bytes = synthesize(reply)
        response = {
            "reply_text": reply,
            "motion": None if motion == "none" else motion,
            "motion_is_proposed": motion in PROPOSED_MOTIONS,
            # Set when the request needs real manipulation rather than a canned
            # gesture - an English task string for MolmoAct2. The board decides
            # whether to act on it; today it can only shadow-log, because
            # MolmoAct2 needs two cameras and the wrist one is dead hardware.
            "task": task,
            "audio_wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
        }
        if task:
            print(f"  [manipulation task] {task!r}", flush=True)
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
