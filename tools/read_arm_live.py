"""
Print live joint positions from BOTH arms on Windows, with torque disabled so
either can be moved freely by hand.

Diagnostic for "I moved the arm but the recording captured no movement". Two
things this distinguishes:
  - which physical arm is on which COM port (move one, see which side reacts)
  - whether a joint is mechanically locked by leftover enabled torque (a joint
    frozen at an identical value while others jitter is locked, not still)

Reads only - never commands a position. Torque is left disabled on exit, which
is the safe state for a leader arm but means a follower arm will go limp.

Usage:
    python read_arm_live.py [seconds]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\LENOVO\AppData\Roaming\uv\tools\lelab\Lib\site-packages")

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
CAL_ROOT = Path(r"C:\Users\LENOVO\.cache\huggingface\lerobot\calibration")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]

leader = SO101Leader(SO101LeaderConfig(
    port="COM7", id="c2", calibration_dir=CAL_ROOT / "teleoperators/so_leader"))
follower = SO101Follower(SO101FollowerConfig(
    port="COM8", id="c2", calibration_dir=CAL_ROOT / "robots/so_follower"))

leader.bus.connect()
leader.bus.write_calibration(leader.calibration)
follower.bus.connect()
follower.bus.write_calibration(follower.calibration)

# Leftover enabled torque from a previous record/teleop session locks the big
# joints so they can't be posed by hand at all - which looks exactly like "the
# recording captured no movement". Free both arms before reading.
leader.bus.disable_torque()
follower.bus.disable_torque()

print("Both arms connected, torque DISABLED (both should now move freely by hand).", flush=True)
print(f"Move ONE arm for the next {DURATION:.0f}s - watch which side changes.\n", flush=True)
header = "  COM7(leader) " + "".join(f"{s:>7}" for s in SHORT) + "   |  COM8(follower) " + "".join(f"{s:>7}" for s in SHORT)
print(header, flush=True)

mins = {"COM7": [float("inf")] * 6, "COM8": [float("inf")] * 6}
maxs = {"COM7": [float("-inf")] * 6, "COM8": [float("-inf")] * 6}

t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < DURATION:
        lead = leader.get_action()
        foll = follower.get_observation()
        lvals = [lead[f"{j}.pos"] for j in JOINTS]
        fvals = [foll[f"{j}.pos"] for j in JOINTS]
        for port, vals in (("COM7", lvals), ("COM8", fvals)):
            for i, v in enumerate(vals):
                mins[port][i] = min(mins[port][i], v)
                maxs[port][i] = max(maxs[port][i], v)
        print("               " + "".join(f"{v:>7.1f}" for v in lvals)
              + "   |                 " + "".join(f"{v:>7.1f}" for v in fvals), flush=True)
        time.sleep(0.35)
finally:
    leader.bus.disconnect()
    follower.bus.disconnect()

print("\nrange moved per joint:", flush=True)
print("            " + "".join(f"{s:>7}" for s in SHORT), flush=True)
for port in ("COM7", "COM8"):
    rng = [mx - mn for mn, mx in zip(mins[port], maxs[port])]
    print(f"  {port}    " + "".join(f"{r:>7.1f}" for r in rng), flush=True)
print("\n(0.0 = that joint never reported any movement)", flush=True)
