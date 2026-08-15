#!/usr/bin/env python3
"""Serve the Mira dashboard and proxy the board's own endpoints to it.

Why a proxy rather than pointing the page straight at Prometheus: Prometheus
scrapes the board every 5s, but board_voice_control.py updates board_audio_rms
every audio hop (HOP_S = 0.5s). Reading through Prometheus therefore shows an
audio meter that lags by up to 5s and misses most of what the mic did. The page
needs the board's live values, and the board's :9103 (prometheus_client) sends
no CORS headers, so a browser can't fetch it cross-origin.

Serving the page and the proxied data from one origin solves both: the meter
polls the board directly at its own hop rate, and no CORS is involved.

    python3 dashboard_server.py            # http://localhost:8090

Prometheus + Grafana stay as they are - they keep the history; this is the
live view.
"""
import base64
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOARD = os.environ.get("MIRA_BOARD", "192.168.1.41")
VOICE_METRICS = f"http://{BOARD}:9103/metrics"   # board_voice_control.py
CAMERA_METRICS = f"http://{BOARD}:9101/metrics"  # camera_exporter.py
LOG_URL = f"http://{BOARD}:9104/log"             # log_tail_server.py
OVERHEAD_SNAPSHOT = f"http://{BOARD}:8080/snapshot"
PORT = int(os.environ.get("MIRA_DASHBOARD_PORT", "8090"))
# Bind to the LAN when serving phones directly from the board; loopback is the
# safe default for a laptop that is only showing it to itself.
BIND = os.environ.get("MIRA_BIND", "127.0.0.1")
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "live_dashboard.html")

# molmo_vision_server.py on the 4090, reached through the local SSH tunnel
# (ssh -L 127.0.0.1:8767:localhost:8767 -p 234 duyetnt2@61.28.228.23). The 4090
# is rented and cannot reach this LAN, so frames are pushed from here rather
# than pulled from there.
# Empty disables detection entirely. That is the right setting when this runs ON
# the board: the board's network blocks outbound traffic to the rented 4090
# (only port 443 gets out), so it can never reach the vision server. Everything
# else on the page is board-local and works there unchanged.
DETECT_URL = os.environ.get("MIRA_DETECT_URL", "http://127.0.0.1:8767/detect")
DETECT_TARGETS = [t.strip() for t in os.environ.get(
    "MIRA_DETECT_TARGETS", "screwdriver,robot arm,speaker").split(",") if t.strip()]
# Each target costs roughly a second on the GPU and the scene changes slowly;
# polling faster would just queue work behind the previous round.
DETECT_INTERVAL_S = float(os.environ.get("MIRA_DETECT_INTERVAL", "6"))

# Camera health moves slowly (its own checker loop runs every 10s) and the page
# polls several times a second, so re-fetching it every tick would be pure load
# on a board that is also running the voice pipeline.
CAMERA_TTL_S = 4.0
_camera_cache = {"at": 0.0, "value": {}}

# name{label="v"} 12.3  ->  (name, {label: v}, 12.3)
SAMPLE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$')
LABEL = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus(text):
    """Flatten Prometheus text format into {metric: value | {label: value}}.

    Only what this dashboard needs: single-value gauges collapse to a number,
    labelled series collapse to a dict keyed by the first label value (motion,
    camera). Anything with no labels and no interest to us costs one dict entry.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        name, labels, raw = m.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if labels:
            pairs = dict(LABEL.findall(labels))
            # instance/job come from Prometheus scrapes, not from the exporter
            key = next((v for k, v in pairs.items() if k not in ("instance", "job", "machine")), "")
            out.setdefault(name, {})[key] = value
        else:
            out[name] = value
    return out


def fetch(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def camera_health():
    now = time.time()
    if now - _camera_cache["at"] < CAMERA_TTL_S:
        return _camera_cache["value"]
    try:
        value = parse_prometheus(fetch(CAMERA_METRICS))
    except Exception:
        value = {}
    _camera_cache.update(at=now, value=value)
    return value


# --- MolmoAct2 object detection, polled in the background -------------------
# Detection runs at its own slow pace (seconds per round) while the page polls
# state four times a second, so it cannot sit in the request path. The page
# reads whatever the last completed round found, plus how old it is.
_detect = {"at": 0.0, "detections": [], "error": None, "ms": 0}


def detect_loop():
    while True:
        try:
            with urllib.request.urlopen(OVERHEAD_SNAPSHOT, timeout=8) as resp:
                frame = resp.read()
            body = json.dumps({
                "image": base64.b64encode(frame).decode("ascii"),
                "targets": DETECT_TARGETS,
            }).encode("utf-8")
            req = urllib.request.Request(
                DETECT_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.load(resp)
            _detect.update(at=time.time(), detections=result.get("detections", []),
                           ms=result.get("total_ms", 0), error=None)
        except Exception as exc:
            # Keep the last good detections on screen rather than blanking them:
            # the vision server being restarted should not look like the objects
            # vanished from the table.
            _detect.update(error=str(exc))
        time.sleep(DETECT_INTERVAL_S)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # the page polls several times a second; logging it is noise

    def _send(self, code, body, content_type):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html", "/live_dashboard.html"):
            try:
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError as e:
                self._send(500, f"cannot read {PAGE}: {e}", "text/plain; charset=utf-8")
            return

        if path == "/api/state":
            try:
                voice = parse_prometheus(fetch(VOICE_METRICS))
                payload = {"ok": True, "voice": voice, "camera": camera_health()}
            except Exception as e:
                payload = {"ok": False, "error": str(e)}
            self._send(200, json.dumps(payload), "application/json; charset=utf-8")
            return

        if path == "/api/detections":
            age = time.time() - _detect["at"] if _detect["at"] else None
            self._send(200, json.dumps({
                "detections": _detect["detections"],
                "age_s": round(age, 1) if age is not None else None,
                "ms": _detect["ms"],
                "targets": DETECT_TARGETS,
                "error": _detect["error"],
            }), "application/json; charset=utf-8")
            return

        if path == "/api/log":
            n = "120"
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    if part.startswith("n="):
                        n = part[2:]
            try:
                payload = fetch(f"{LOG_URL}?n={n}")
            except Exception as e:
                payload = json.dumps({"lines": [f"(không đọc được log từ board: {e})"]})
            self._send(200, payload, "application/json; charset=utf-8")
            return

        self._send(404, "not found", "text/plain; charset=utf-8")


if __name__ == "__main__":
    if DETECT_URL:
        threading.Thread(target=detect_loop, daemon=True).start()
    print(f"Mira dashboard on http://localhost:{PORT}  (board {BOARD})", flush=True)
    print(f"  nhận dạng vật: {DETECT_TARGETS} mỗi {DETECT_INTERVAL_S}s qua {DETECT_URL}"
          if DETECT_URL else "  nhận dạng vật: TẮT (không đặt MIRA_DETECT_URL)", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
