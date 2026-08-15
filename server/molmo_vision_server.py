#!/usr/bin/env python3
"""Phục vụ phần NHÌN của MolmoAct2 qua HTTP, tách khỏi phần sinh hành động.

Vì sao tách khỏi molmoact2_server.py: hai thứ này có số phận khác nhau. Phần
sinh hành động đã đo là bỏ qua câu lệnh (probe_molmoact2_official_sample.py -
ba lệnh trái ngược ra cùng một quỹ đạo, ngay trên ảnh mẫu của chính checkpoint),
nên hiện không dùng được. Phần nhìn thì ngược lại: hỏi "Point at the screwdriver"
là nó trả về toạ độ đúng vật, ~1s một câu. Bảng theo dõi dùng được ngay phần đó,
và đó cũng là nền cho hướng "định vị bằng Molmo rồi tự tính đường gắp".

    POST /detect  {"image": "<base64 jpeg>", "targets": ["screwdriver", ...]}
      -> {"detections": [{"label", "x", "y", "box": [x1,y1,x2,y2]|null, "ms"}],
          "total_ms": ...}

Toạ độ trả về theo TỈ LỆ 0-1 của khung hình, không phải pixel, để bên hiển thị
tự nhân với kích thước thật - khung stream và khung gửi đi không nhất thiết
cùng kích thước.
"""
import base64
import io
import json
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "allenai/MolmoAct2-SO100_101"
PORT = 8767
MAX_NEW_TOKENS = 64

# Molmo trả `<points coords="1 1 361 691">screwdriver</points>` hoặc
# `<boxes coords="1 1 160 230 480 590">objects</boxes>`. Toạ độ theo phần nghìn.
# Hai số đầu là siêu dữ liệu (số điểm / chỉ số), toạ độ nằm ở cuối - đã kiểm
# chứng bằng cách vẽ cả hai cách đọc lên ảnh rồi nhìn: đọc (x, y) mới trúng vật.
TAG = re.compile(r'<(points|boxes)\s+coords="([^"]+)"[^>]*>')

print(f"nạp {MODEL}...", flush=True)
t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"Ready sau {time.perf_counter()-t0:.0f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB. "
      f"Listening on :{PORT}", flush=True)

# Một GPU, một model - nối tiếp các yêu cầu thay vì để chúng giẫm lên nhau.
gpu_lock = threading.Lock()


def ask(image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def parse(text):
    """-> (x, y, box) theo tỉ lệ 0-1; box là None nếu model chỉ trả điểm."""
    m = TAG.search(text)
    if not m:
        return None, None, None
    kind, raw = m.group(1), m.group(2)
    nums = [float(v) for v in raw.split() if v.replace(".", "").isdigit()]
    if kind == "boxes" and len(nums) >= 4:
        x1, y1, x2, y2 = (v / 1000 for v in nums[-4:])
        return (x1 + x2) / 2, (y1 + y2) / 2, [x1, y1, x2, y2]
    if len(nums) >= 2:
        return nums[-2] / 1000, nums[-1] / 1000, None
    return None, None, None


def detect(image, targets):
    out = []
    for target in targets:
        t0 = time.perf_counter()
        try:
            # "Detect" cho khung khi model có; "Point at" thì luôn ra điểm. Hỏi
            # Detect trước rồi lùi về Point, để có khung khi nào có thể.
            text = ask(image, f"Detect the {target}.")
            x, y, box = parse(text)
            if x is None:
                text = ask(image, f"Point at the {target}.")
                x, y, box = parse(text)
        except Exception:
            traceback.print_exc()
            x = y = box = None
            text = ""
        if x is not None:
            out.append({"label": target, "x": round(x, 4), "y": round(y, 4),
                        "box": [round(v, 4) for v in box] if box else None,
                        "ms": round((time.perf_counter() - t0) * 1000)})
        else:
            print(f"  [không định vị được] {target}: {text[:80]!r}", flush=True)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/detect":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            image = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
            targets = body.get("targets") or ["screwdriver"]
            t0 = time.perf_counter()
            with gpu_lock:
                detections = detect(image, targets)
            self._send(200, {"detections": detections,
                             "total_ms": round((time.perf_counter() - t0) * 1000),
                             "image_size": list(image.size)})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": str(exc)})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
