import sys

sys.path.insert(0, "/home/arduino/.local/share/mira-so101")
from runtime import calibration_path, detect_single_role, make_arm  # noqa: E402

ROLE = "follower"

port = detect_single_role(ROLE)
print(f"Found {ROLE} on {port}")
arm = make_arm(ROLE, port)
arm.bus.connect()
try:
    arm.calibrate()
finally:
    arm.bus.disconnect(disable_torque=True)

print("Saved to:", calibration_path(ROLE))
