#!/usr/bin/env python3
"""Minimal SO-101 teleoperation and recorded-motion runtime for Arduino UNO Q."""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
MOTION_ALIASES = {
    "wave": "motion_hi_wave",
    "dance": "motion_dance",
    "play-dead": "motion_play_dead",
    "play_dead": "motion_play_dead",
    "yes": "motion_nod_yes",
    "nod": "motion_nod_yes",
    "no": "motion_shake_no",
    "shake": "motion_shake_no",
    "clean": "motion_cleaning",
    "scan": "motion_scan_10s",
}
LOCK_PATH = Path("/tmp/mira-so101-motion.lock")
PID_PATH = Path("/tmp/mira-so101-motion.pid")


def lerobot_root() -> Path:
    return Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def calibration_path(role: str) -> Path:
    if role == "follower":
        return lerobot_root() / "calibration/robots/so101_follower/my_follower.json"
    return lerobot_root() / "calibration/teleoperators/so101_leader/my_leader.json"


def load_calibration(role: str) -> dict[str, dict[str, int]]:
    path = calibration_path(role)
    if not path.is_file():
        raise RuntimeError(f"Missing {role} calibration: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data) != MOTOR_NAMES:
        raise RuntimeError(f"Invalid motor names in {path}")
    required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    for expected_id, name in enumerate(MOTOR_NAMES, 1):
        values = data[name]
        if set(values) != required or values["id"] != expected_id:
            raise RuntimeError(f"Invalid calibration entry for {name} in {path}")
        if values["range_min"] >= values["range_max"]:
            raise RuntimeError(f"Invalid calibration range for {name} in {path}")
    return data


def motion_names() -> list[str]:
    return sorted(set(MOTION_ALIASES.values()))


def resolve_motion(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    dataset = MOTION_ALIASES.get(normalized, value)
    if dataset.startswith("local/"):
        dataset = dataset.split("/", 1)[1]
    if dataset not in motion_names():
        choices = ", ".join(sorted(MOTION_ALIASES))
        raise RuntimeError(f"Unknown motion '{value}'. Choose: {choices}")
    return dataset


def load_motion(dataset_name: str, episode: int = 0) -> tuple[list[dict[str, float]], int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = lerobot_root() / "local" / dataset_name
    info_path = root / "meta/info.json"
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not info_path.is_file() or not files:
        raise RuntimeError(f"Incomplete motion dataset: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = tuple(info.get("features", {}).get("action", {}).get("names", ()))
    if names != ACTION_NAMES:
        raise RuntimeError(f"Unexpected joint layout in {info_path}: {names}")
    table = pa.concat_tables([pq.read_table(path, columns=["action", "episode_index"]) for path in files])
    actions: list[dict[str, float]] = []
    for values, row_episode in zip(table["action"].to_pylist(), table["episode_index"].to_pylist()):
        if int(row_episode) != episode:
            continue
        if len(values) != 6 or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Invalid action row in {dataset_name}")
        action = {name: float(value) for name, value in zip(ACTION_NAMES, values)}
        if any(not -100.001 <= action[name] <= 100.001 for name in ACTION_NAMES[:-1]):
            raise RuntimeError(f"Body joint outside normalized range in {dataset_name}")
        if not 0 <= action["gripper.pos"] <= 100.001:
            raise RuntimeError(f"Gripper outside normalized range in {dataset_name}")
        actions.append(action)
    if not actions:
        raise RuntimeError(f"Episode {episode} is missing from {dataset_name}")
    fps = int(info.get("fps", 30))
    if not 1 <= fps <= 120:
        raise RuntimeError(f"Invalid FPS in {info_path}: {fps}")
    return actions, fps


def candidate_ports() -> list[str]:
    paths = glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    unique: list[str] = []
    real_seen: set[str] = set()
    for path in sorted(paths):
        real = os.path.realpath(path)
        if real not in real_seen:
            real_seen.add(real)
            unique.append(path)
    return unique


def make_arm(role: str, port: str, *, max_relative_target: float | None = None):
    if role == "follower":
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=port,
            id="my_follower",
            calibration_dir=calibration_path(role).parent,
            disable_torque_on_disconnect=True,
            max_relative_target=max_relative_target,
        )
        return SO101Follower(config)
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    config = SO101LeaderConfig(
        port=port,
        id="my_leader",
        calibration_dir=calibration_path(role).parent,
    )
    return SO101Leader(config)


def calibration_matches(role: str, port: str) -> tuple[bool, str]:
    arm = make_arm(role, port)
    try:
        arm.bus.connect()
        matches = bool(arm.is_calibrated)
        return matches, "match" if matches else "calibration mismatch"
    except Exception as exc:
        return False, str(exc).strip().splitlines()[-1]
    finally:
        if arm.bus.is_connected:
            try:
                arm.bus.disconnect(disable_torque=True)
            except Exception:
                pass


def detect_roles() -> tuple[str, str, list[str]]:
    ports = candidate_ports()
    if len(ports) < 2:
        raise RuntimeError(f"Need two connected SO-101 USB serial devices; found {ports or 'none'}")
    follower_matches: list[str] = []
    leader_matches: list[str] = []
    details: list[str] = []
    for port in ports:
        follower_ok, follower_note = calibration_matches("follower", port)
        leader_ok, leader_note = calibration_matches("leader", port)
        details.append(f"{port}: follower={follower_note}; leader={leader_note}")
        if follower_ok:
            follower_matches.append(port)
        if leader_ok:
            leader_matches.append(port)
    pairs = [(follower, leader) for follower in follower_matches for leader in leader_matches if follower != leader]
    if len(pairs) != 1:
        raise RuntimeError("Could not uniquely identify leader and follower from calibration.\n" + "\n".join(details))
    return pairs[0][0], pairs[0][1], details


def detect_single_role(role: str) -> str:
    ports = candidate_ports()
    if not ports:
        raise RuntimeError("No USB serial devices found. Connect the SO-101 controller board.")
    matches: list[str] = []
    details: list[str] = []
    for port in ports:
        matched, note = calibration_matches(role, port)
        details.append(f"{port}: {note}")
        if matched:
            matches.append(port)
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify the {role} from its saved calibration.\n" + "\n".join(details)
        )
    return matches[0]


def resolve_port(role: str, requested: str) -> str:
    if requested != "auto":
        if not Path(requested).exists():
            raise RuntimeError(f"Missing {role} port: {requested}")
        matches, note = calibration_matches(role, requested)
        if not matches:
            raise RuntimeError(f"{requested} does not match the saved {role} calibration: {note}")
        return requested
    return detect_single_role(role)


@contextmanager
def exclusive_robot() -> Iterator[None]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another teleoperation or replay process already owns the robot.") from exc
        PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")
        try:
            yield
        finally:
            PID_PATH.unlink(missing_ok=True)


def require_calibrated(arm: Any, role: str) -> None:
    arm.bus.connect()
    try:
        if not arm.is_calibrated:
            raise RuntimeError(
                f"The connected {role} does not match {calibration_path(role)}. "
                "Refusing to enable motion."
            )
    finally:
        arm.bus.disconnect(disable_torque=True)


def interpolate_to(follower: Any, target: dict[str, float], seconds: float, fps: int = 50) -> None:
    current = follower.get_observation()
    steps = max(1, int(seconds * fps))
    print(f"Safely aligning follower for {seconds:.1f}s...")
    for index in range(1, steps + 1):
        ratio = index / steps
        action = {name: current[name] + (target[name] - current[name]) * ratio for name in ACTION_NAMES}
        started = time.perf_counter()
        follower.send_action(action)
        time.sleep(max(0.0, 1 / fps - (time.perf_counter() - started)))


def confirm_motion(description: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    print(f"About to {description}. Keep the emergency stop/power switch accessible.")
    if input("Type MOVE to continue: ").strip() != "MOVE":
        raise RuntimeError("Cancelled; robot did not move.")


def run_replay(args: argparse.Namespace) -> None:
    dataset = resolve_motion(args.motion)
    actions, fps = load_motion(dataset, args.episode)
    follower_port = resolve_port("follower", args.follower_port)
    confirm_motion(f"replay {dataset} ({len(actions) / fps:.1f}s)", args.yes)
    follower = make_arm("follower", follower_port, max_relative_target=8.0)
    with exclusive_robot():
        require_calibrated(follower, "follower")
        follower.connect(calibrate=False)
        try:
            interpolate_to(follower, actions[0], args.align_seconds)
            print(f"Replaying {dataset} at {fps} FPS on {follower_port}. Press Ctrl-C to stop.")
            next_frame = time.perf_counter()
            for action in actions:
                follower.send_action(action)
                next_frame += 1 / fps
                time.sleep(max(0.0, next_frame - time.perf_counter()))
            print("Motion complete.")
        finally:
            if follower.is_connected:
                follower.disconnect()


def run_teleop(args: argparse.Namespace) -> None:
    if args.follower_port == "auto" and args.leader_port == "auto":
        follower_port, leader_port, _details = detect_roles()
    else:
        follower_port = resolve_port("follower", args.follower_port)
        leader_port = resolve_port("leader", args.leader_port)
    if os.path.realpath(follower_port) == os.path.realpath(leader_port):
        raise RuntimeError("Leader and follower ports resolve to the same device.")
    confirm_motion("start leader-to-follower teleoperation", args.yes)
    follower = make_arm("follower", follower_port, max_relative_target=5.0)
    leader = make_arm("leader", leader_port)
    with exclusive_robot():
        require_calibrated(leader, "leader")
        require_calibrated(follower, "follower")
        leader.connect(calibrate=False)
        try:
            follower.connect(calibrate=False)
            try:
                first_action = leader.get_action()
                interpolate_to(follower, first_action, args.align_seconds)
                print(f"TELEOP ACTIVE: {leader_port} -> {follower_port}. Press Ctrl-C to stop.")
                next_frame = time.perf_counter()
                started = next_frame
                while args.duration <= 0 or time.perf_counter() - started < args.duration:
                    follower.send_action(leader.get_action())
                    next_frame += 1 / args.fps
                    time.sleep(max(0.0, next_frame - time.perf_counter()))
            finally:
                if follower.is_connected:
                    follower.disconnect()
        finally:
            if leader.is_connected:
                leader.disconnect()
    print("Teleoperation stopped; follower torque disabled.")


def run_doctor(args: argparse.Namespace) -> None:
    print("Mira SO-101 core doctor")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Architecture: {os.uname().machine}")
    import draccus
    import pyarrow
    import serial
    import scservo_sdk
    from lerobot.robots.so101_follower import SO101Follower
    from lerobot.teleoperators.so101_leader import SO101Leader

    _ = (draccus, serial, scservo_sdk, SO101Follower, SO101Leader)
    print(f"  PyArrow:      {pyarrow.__version__}")
    print("  Motor driver: LeRobot SO-101 core imports OK")
    for role in ("follower", "leader"):
        load_calibration(role)
        print(f"  Calibration:  {role} OK")
    for dataset in motion_names():
        actions, fps = load_motion(dataset)
        print(f"  Motion:       {dataset} ({len(actions)} frames @ {fps} FPS)")
    if not args.no_hardware:
        follower, leader, details = detect_roles()
        for line in details:
            print(f"  Probe:        {line}")
        print(f"  Follower:     {follower}")
        print(f"  Leader:       {leader}")
    print("CORE READY")


def run_list(_args: argparse.Namespace) -> None:
    for alias, dataset in sorted(MOTION_ALIASES.items()):
        if "_" not in alias and alias not in {"play-dead"}:
            actions, fps = load_motion(dataset)
            print(f"{alias:10} {len(actions) / fps:5.1f}s  {dataset}")


def run_status(_args: argparse.Namespace) -> None:
    if not PID_PATH.is_file():
        print("idle")
        return
    pid_text = PID_PATH.read_text(encoding="utf-8").strip()
    if pid_text.isdigit() and Path(f"/proc/{pid_text}").exists():
        print(f"moving (pid {pid_text})")
    else:
        PID_PATH.unlink(missing_ok=True)
        print("idle")


def run_stop(_args: argparse.Namespace) -> None:
    if not PID_PATH.is_file():
        print("Robot is already idle.")
        return
    pid_text = PID_PATH.read_text(encoding="utf-8").strip()
    if not pid_text.isdigit():
        raise RuntimeError("Invalid robot PID file; refusing to signal anything.")
    pid = int(pid_text)
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.is_file() or b"runtime.py" not in cmdline_path.read_bytes():
        PID_PATH.unlink(missing_ok=True)
        raise RuntimeError("The recorded PID is not the motion runtime; refusing to signal it.")
    os.kill(pid, signal.SIGINT)
    print(f"Stop requested for robot process {pid}.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mira-robot", description="Minimal SO-101 motion control")
    commands = root.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="Replay one recorded joint motion")
    replay.add_argument("motion")
    replay.add_argument("--episode", type=int, default=0)
    replay.add_argument("--follower-port", default=os.getenv("FOLLOWER_PORT", "auto"))
    replay.add_argument("--align-seconds", type=float, default=3.0)
    replay.add_argument("--yes", action="store_true")
    replay.set_defaults(func=run_replay)
    teleop = commands.add_parser("teleop", help="Mirror the leader arm to the follower")
    teleop.add_argument("--follower-port", default=os.getenv("FOLLOWER_PORT", "auto"))
    teleop.add_argument("--leader-port", default=os.getenv("LEADER_PORT", "auto"))
    teleop.add_argument("--fps", type=int, default=50)
    teleop.add_argument("--duration", type=float, default=0, help="Seconds; zero runs until stopped")
    teleop.add_argument("--align-seconds", type=float, default=3.0)
    teleop.add_argument("--yes", action="store_true")
    teleop.set_defaults(func=run_teleop)
    doctor = commands.add_parser("doctor", help="Verify software, files, calibration, and hardware")
    doctor.add_argument("--no-hardware", action="store_true")
    doctor.set_defaults(func=run_doctor)
    list_cmd = commands.add_parser("list", help="List bundled motions")
    list_cmd.set_defaults(func=run_list)
    status = commands.add_parser("status", help="Show whether a motion process is active")
    status.set_defaults(func=run_status)
    stop = commands.add_parser("stop", help="Safely interrupt the active motion process")
    stop.set_defaults(func=run_stop)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped by operator.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
