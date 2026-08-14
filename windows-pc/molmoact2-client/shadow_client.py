"""
MolmoAct2 shadow-mode inference client.

SHADOW MODE ONLY: captures a real camera1 frame + real follower state, sends
them (plus a camera2 placeholder, since the wrist camera is currently broken)
to the RTX 4090 inference server, and logs the predicted actions to a
timestamped JSON file.

This script NEVER sends anything to the robot's motors. It is diagnostic
logging only, per the safety plan in runpod_handoff.md.
"""
import base64
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

BOARD_HOST = "192.168.1.41"
OVERHEAD_SNAPSHOT_URL = f"http://{BOARD_HOST}:8080/snapshot"
INFER_URL = "http://127.0.0.1:8765/infer"  # via SSH -L tunnel to 61.28.228.23:8765
SSH_KEY = "C:/Users/LENOVO/.ssh/id_ed25519_unoq"
LOG_DIR = "d:/Comp/Qualcom/molmoact2-client/shadow_logs"

TASK = "pick up the screwdriver and put it on the black workspace"


def fetch_overhead_jpeg() -> bytes:
    with urllib.request.urlopen(OVERHEAD_SNAPSHOT_URL, timeout=10) as resp:
        return resp.read()


def placeholder_camera2_jpeg() -> bytes:
    # Wrist camera (Sunplus SPCA2085) is currently non-functional (confirmed
    # hardware/protocol-level failure). Use a neutral gray placeholder frame
    # so the request is well-formed; results are plumbing-test only.
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def read_follower_state() -> list[float]:
    result = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "arduino@" + BOARD_HOST, "~/.local/share/mira-so101/venv/bin/python /home/arduino/read_follower_state.py"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read follower state: {result.stderr.strip()}")
    state = json.loads(result.stdout.strip())
    order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
             "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    return [state[k] for k in order]


def main():
    import os
    os.makedirs(LOG_DIR, exist_ok=True)

    print("Fetching overhead camera1 frame...", flush=True)
    camera1_jpeg = fetch_overhead_jpeg()
    print(f"  got {len(camera1_jpeg)} bytes", flush=True)

    print("Building camera2 placeholder (wrist cam unavailable)...", flush=True)
    camera2_jpeg = placeholder_camera2_jpeg()

    print("Reading real follower state via SSH (read-only)...", flush=True)
    state = read_follower_state()
    print(f"  state: {state}", flush=True)

    body = {
        "camera1": base64.b64encode(camera1_jpeg).decode("ascii"),
        "camera2": base64.b64encode(camera2_jpeg).decode("ascii"),
        "camera2_is_placeholder": True,
        "state": state,
        "task": TASK,
    }
    payload = json.dumps(body).encode("utf-8")

    print(f"Sending inference request to {INFER_URL} ...", flush=True)
    req = urllib.request.Request(INFER_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        response = json.loads(resp.read())
    wall_latency_s = time.perf_counter() - t0

    print(f"Response received in {wall_latency_s:.2f}s (server-side latency: {response.get('latency_ms', '?')} ms)", flush=True)
    for w in response.get("warnings", []):
        print(f"  WARNING: {w}", flush=True)

    action = response.get("action")
    if action is not None:
        arr = np.array(action)
        print(f"  action shape: {arr.shape}", flush=True)
        print(f"  action[0] (first predicted step): {arr[0][0].tolist()}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"{LOG_DIR}/shadow_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "task": TASK,
            "state": state,
            "camera2_is_placeholder": True,
            "server_response": response,
            "wall_latency_s": wall_latency_s,
        }, f, indent=2)
    print(f"Logged to {log_path}", flush=True)
    print("\nNOTE: SHADOW MODE ONLY. No command was sent to the robot's motors.", flush=True)


if __name__ == "__main__":
    main()
