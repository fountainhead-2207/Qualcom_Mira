import json
import sys

sys.path.insert(0, "/home/arduino/.local/share/mira-so101")
from runtime import detect_single_role, make_arm  # noqa: E402

ROLE = "follower"
port = detect_single_role(ROLE)
arm = make_arm(ROLE, port)
arm.bus.connect()
try:
    obs = arm.get_observation()
finally:
    arm.bus.disconnect(disable_torque=True)

# get_observation() may include non-serializable/camera keys; keep only floats
state = {k: float(v) for k, v in obs.items() if isinstance(v, (int, float))}
print(json.dumps(state))
