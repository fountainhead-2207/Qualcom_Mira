import base64
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

CKPT = "/data/qualcom-robotic/mira_molmoact2_step2000"
PORT = 8765

INFER_REQUESTS = Counter("molmoact2_infer_requests_total", "Total inference requests", ["status"])
INFER_LATENCY = Histogram(
    "molmoact2_infer_latency_seconds",
    "Inference latency (model forward pass only, excludes HTTP overhead)",
    buckets=(0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20),
)
INFER_WARNINGS = Counter("molmoact2_infer_warnings_total", "Warnings emitted during inference", ["kind"])

print("Loading policy config + pre/post processors...", flush=True)
cfg = PreTrainedConfig.from_pretrained(CKPT)
preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT)

print("Loading policy weights...", flush=True)
t0 = time.perf_counter()
policy = MolmoAct2Policy.from_pretrained(CKPT)
policy.to("cuda")
policy.eval()
print(f"Ready in {time.perf_counter() - t0:.1f}s. Listening on :{PORT}", flush=True)


def decode_image(b64_str: str) -> torch.Tensor:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((320, 240))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, 0-1
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,240,320
    return tensor


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/infer":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            warnings = []
            camera2_is_placeholder = bool(body.get("camera2_is_placeholder", False))
            if camera2_is_placeholder:
                warnings.append(
                    "camera2 is a placeholder (wrist cam unavailable) - predictions are PLUMBING-TEST ONLY, not a real diagnostic"
                )
                INFER_WARNINGS.labels(kind="camera2_placeholder").inc()

            batch = {
                "observation.images.camera1": decode_image(body["camera1"]),
                "observation.images.camera2": decode_image(body["camera2"]),
                "observation.state": torch.tensor([body["state"]], dtype=torch.float32),
                "task": [body["task"]],
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                processed = preprocessor(batch)
                action = policy.predict_action_chunk(processed, inference_action_mode="continuous")
                action = postprocessor(action)
            latency_s = time.perf_counter() - t0
            INFER_LATENCY.observe(latency_s)

            action_list = action.cpu().float().tolist()
            has_nan = bool(torch.isnan(action).any().item())
            has_inf = bool(torch.isinf(action).any().item())
            if has_nan:
                warnings.append("action contains NaN")
                INFER_WARNINGS.labels(kind="nan").inc()
            if has_inf:
                warnings.append("action contains Inf")
                INFER_WARNINGS.labels(kind="inf").inc()

            response = {
                "action": action_list,
                "latency_ms": latency_s * 1000,
                "warnings": warnings,
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            INFER_REQUESTS.labels(status="ok").inc()
        except Exception as exc:
            traceback.print_exc()
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            INFER_REQUESTS.labels(status="error").inc()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
