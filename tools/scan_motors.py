"""
Ping every servo ID on one arm's bus and report which ones answer.

For diagnosing `Failed to write '...' on id_=N ... There is no status packet!` -
that means servo N stopped answering, and the usual causes are the 12V supply
being off, or a loose daisy-chain connector at or before that servo. Since the
chain is serial, a break at motor N typically silences N and everything after
it, which is what makes the pattern of live-vs-dead IDs worth reading.

Deliberately does not write anything or touch torque, so it's safe to run on a
bus that's already misbehaving.

Usage:
    python scan_motors.py COM8        # follower
    python scan_motors.py COM7        # leader
"""
import sys

sys.path.insert(0, r"C:\Users\LENOVO\AppData\Roaming\uv\tools\lelab\Lib\site-packages")

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

bus = FeetechMotorsBus(
    port=PORT,
    motors={name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
            for i, name in enumerate(JOINTS, 1)},
)
bus.connect(handshake=False)  # skip the handshake so a dead servo can't abort the scan

print(f"Scanning {PORT}...\n", flush=True)
alive, dead = [], []
try:
    for i, name in enumerate(JOINTS, 1):
        model = bus.ping(i, num_retry=2)
        if model is None:
            dead.append((i, name))
            print(f"  id {i}  {name:15} NO RESPONSE", flush=True)
        else:
            alive.append((i, name))
            pos = bus.read("Present_Position", name, normalize=False)
            print(f"  id {i}  {name:15} ok (model {model}, raw pos {pos})", flush=True)
finally:
    bus.disconnect()

print(f"\n{len(alive)}/6 servos answered.", flush=True)
if dead:
    first = dead[0][0]
    print(f"Dead: {', '.join(f'{i}:{n}' for i, n in dead)}", flush=True)
    print(f"\nThe chain goes 1 -> 2 -> ... -> 6, so check the cable going INTO servo "
          f"{first} ({dead[0][1]}) and that servo's own connectors first.", flush=True)
    if len(dead) > 1 and [i for i, _ in dead] == list(range(first, 7)):
        print("Everything from that point on is silent, which fits a single break there "
              "rather than several failed servos.", flush=True)
    print("Also confirm the arm's 12V supply is plugged in and on.", flush=True)
else:
    print("Whole chain is healthy - if a write still fails, retry it; that would be "
          "a transient bus error rather than a wiring fault.", flush=True)
