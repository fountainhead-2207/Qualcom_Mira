"""
Lightweight camera-health Prometheus exporter for the UNO Q board.
Stdlib only (no extra dependencies) to stay within the board's tight
RAM/disk budget.

Exposes, per camera:
  camera_up{camera="..."}                            1/0, from re-fetching that
                                                     camera's ustreamer snapshot
  camera_last_success_timestamp_seconds{camera="..."}
  camera_last_frame_bytes{camera="..."}

Both cameras are probed the same way now. The wrist slot used to report a
hardcoded 0 plus a camera_known_broken marker, because the Sunplus SPCA2085
mounted there was dead at the protocol level (see docs/uno-q-board.md). That
unit has since been replaced with a working UVC camera, so a hardcoded 0 would
now report a live camera as broken - it has to be measured, like the other one.
"""
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# name -> ustreamer snapshot endpoint. Each camera needs its own ustreamer
# instance; the port here must match the one it was started on.
CAMERAS = {
    "overhead": "http://127.0.0.1:8080/snapshot",
    "wrist": "http://127.0.0.1:8081/snapshot",
}
CHECK_INTERVAL_S = 10
PORT = 9101

state_lock = threading.Lock()
state = {name: {"up": 0, "last_success": 0.0, "last_size": 0} for name in CAMERAS}


def checker_loop():
    while True:
        for name, url in CAMERAS.items():
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = resp.read()
                with state_lock:
                    state[name].update(up=1, last_success=time.time(), last_size=len(data))
            except Exception:
                with state_lock:
                    state[name]["up"] = 0
        time.sleep(CHECK_INTERVAL_S)


def render_metrics() -> str:
    with state_lock:
        snapshot = {name: dict(s) for name, s in state.items()}

    lines = [
        "# HELP camera_up Whether the camera is currently producing frames (1) or not (0)",
        "# TYPE camera_up gauge",
    ]
    lines += [f'camera_up{{camera="{n}"}} {s["up"]}' for n, s in snapshot.items()]
    lines += [
        "# HELP camera_last_success_timestamp_seconds Unix timestamp of the last successful frame fetch",
        "# TYPE camera_last_success_timestamp_seconds gauge",
    ]
    lines += [f'camera_last_success_timestamp_seconds{{camera="{n}"}} {s["last_success"]}'
              for n, s in snapshot.items()]
    lines += [
        "# HELP camera_last_frame_bytes Size in bytes of the last successfully fetched frame",
        "# TYPE camera_last_frame_bytes gauge",
    ]
    lines += [f'camera_last_frame_bytes{{camera="{n}"}} {s["last_size"]}'
              for n, s in snapshot.items()]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep quiet, this runs unattended on a constrained board

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    threading.Thread(target=checker_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"camera_exporter listening on :{PORT}", flush=True)
    server.serve_forever()
