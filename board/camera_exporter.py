"""
Lightweight camera-health Prometheus exporter for the UNO Q board.
Stdlib only (no extra dependencies) to stay within the board's tight
RAM/disk budget.

Exposes:
  camera_up{camera="overhead"}   1/0, based on periodically re-fetching the
                                  ustreamer snapshot endpoint
  camera_last_success_timestamp_seconds{camera="overhead"}
  camera_up{camera="wrist"}      always 0 -- known hardware/protocol-level
                                  failure (Sunplus SPCA2085), see reason label
"""
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OVERHEAD_SNAPSHOT_URL = "http://127.0.0.1:8080/snapshot"
CHECK_INTERVAL_S = 10
PORT = 9101

state_lock = threading.Lock()
state = {
    "overhead_up": 0,
    "overhead_last_success": 0.0,
    "overhead_last_size": 0,
}


def checker_loop():
    while True:
        try:
            with urllib.request.urlopen(OVERHEAD_SNAPSHOT_URL, timeout=5) as resp:
                data = resp.read()
            with state_lock:
                state["overhead_up"] = 1
                state["overhead_last_success"] = time.time()
                state["overhead_last_size"] = len(data)
        except Exception:
            with state_lock:
                state["overhead_up"] = 0
        time.sleep(CHECK_INTERVAL_S)


def render_metrics() -> str:
    with state_lock:
        s = dict(state)
    lines = [
        "# HELP camera_up Whether the camera is currently producing frames (1) or not (0)",
        "# TYPE camera_up gauge",
        f'camera_up{{camera="overhead"}} {s["overhead_up"]}',
        'camera_up{camera="wrist"} 0',
        "# HELP camera_last_success_timestamp_seconds Unix timestamp of the last successful frame fetch",
        "# TYPE camera_last_success_timestamp_seconds gauge",
        f'camera_last_success_timestamp_seconds{{camera="overhead"}} {s["overhead_last_success"]}',
        "# HELP camera_last_frame_bytes Size in bytes of the last successfully fetched frame",
        "# TYPE camera_last_frame_bytes gauge",
        f'camera_last_frame_bytes{{camera="overhead"}} {s["overhead_last_size"]}',
        "# HELP camera_known_broken Static marker for cameras with a known unfixable hardware/protocol failure",
        "# TYPE camera_known_broken gauge",
        'camera_known_broken{camera="wrist",reason="SPCA2085_uvc_protocol_failure"} 1',
    ]
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
