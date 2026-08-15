"""Calibrate the SO-101 LEADER arm with its port pinned explicitly.

detect_single_role() can't find a leader whose saved calibration is stale
(chicken-and-egg: detection matches against the very calibration being
regenerated), so this pins the port by serial. Run interactively (needs a TTY),
type 'c' at the prompt, sweep every joint AND the gripper trigger fully.

Written 2026-08-15 the night the leader had to be recalibrated on the board.
"""
import os
import sys

sys.path.insert(0, "/home/arduino/.local/share/mira-so101")
from runtime import calibration_path, make_arm  # noqa: E402

PORT = os.environ.get(
    "MIRA_LEADER_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C118584-if00",
)

arm = make_arm("leader", PORT)
arm.bus.connect()
try:
    arm.calibrate()
finally:
    arm.bus.disconnect(disable_torque=True)
print("Saved to:", calibration_path("leader"))
