"""
Watch the leader arm's RAW encoder counts and report which joints actually move.

Event-driven on purpose: earlier fixed-window versions of this kept missing the
movement entirely because the "move now" prompt reached the operator after the
window had already elapsed, which reads identically to a dead encoder. This
version watches for a long stretch and prints only when a joint actually
changes, so the operator can move whenever they're ready.

What the result distinguishes:
  - a joint whose raw count never budges while the joint physically rotates
    -> the bus isn't reading that motor (ID mismatch / wiring / dead encoder)
  - raw moves but the normalized value doesn't -> calibration maps it wrong

Usage:
    python read_arm_raw.py [seconds]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\LENOVO\AppData\Roaming\uv\tools\lelab\Lib\site-packages")

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
MOVE_THRESHOLD = 8  # raw counts; above the few-count idle jitter of these servos
CAL_DIR = Path(r"C:\Users\LENOVO\.cache\huggingface\lerobot\calibration\teleoperators\so_leader")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]

leader = SO101Leader(SO101LeaderConfig(port="COM7", id="c2", calibration_dir=CAL_DIR))
leader.bus.connect()
leader.bus.write_calibration(leader.calibration)
leader.bus.disable_torque()

print("Leader (COM7) connected, torque disabled - the arm should move freely.", flush=True)
print(f"You have {DURATION:.0f}s. Move EVERY joint through its range, one at a time,", flush=True)
print("especially SHOULDER-LIFT and ELBOW. Take your time.\n", flush=True)

start = leader.bus.sync_read("Present_Position", normalize=False)
lo = {j: int(start[j]) for j in JOINTS}
hi = {j: int(start[j]) for j in JOINTS}
reported = set()

t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < DURATION:
        raw = leader.bus.sync_read("Present_Position", normalize=False)
        for j in JOINTS:
            v = int(raw[j])
            lo[j] = min(lo[j], v)
            hi[j] = max(hi[j], v)
            if j not in reported and hi[j] - lo[j] > MOVE_THRESHOLD:
                reported.add(j)
                print(f"  [{time.perf_counter() - t0:5.1f}s] MOVEMENT detected on "
                      f"{j} (raw {lo[j]}..{hi[j]})", flush=True)
        time.sleep(0.05)
finally:
    leader.bus.disconnect()

print("\n" + "=" * 64, flush=True)
print("RAW count range per joint:", flush=True)
print("     " + "".join(f"{s:>10}" for s in SHORT), flush=True)
print("     " + "".join(f"{hi[j] - lo[j]:>10}" for j in JOINTS), flush=True)
dead = [j for j in JOINTS if hi[j] - lo[j] <= MOVE_THRESHOLD]
if dead:
    print(f"\nNEVER MOVED: {', '.join(dead)}", flush=True)
    print("If you did physically rotate those, the bus is not reading them.", flush=True)
else:
    print("\nAll six joints reported real movement - the bus reads every motor fine.", flush=True)
