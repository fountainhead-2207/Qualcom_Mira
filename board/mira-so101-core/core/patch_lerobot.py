#!/usr/bin/env python3
"""Remove LeRobot's AI-only imports from the motor-only runtime path."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def main() -> None:
    path = Path(sysconfig.get_paths()["purelib"]) / "lerobot/motors/motors_bus.py"
    source = path.read_text(encoding="utf-8")
    heavy_import = "from lerobot.utils.utils import enter_pressed, move_cursor_up"
    lean_replacement = """# Mira UNO Q core does not install Torch/Accelerate/Datasets. These two\n# terminal helpers are needed only by LeRobot's interactive calibration UI,\n# which this motion-only runtime never invokes.\ndef enter_pressed() -> bool:\n    return False\n\ndef move_cursor_up(_num_lines: int) -> None:\n    return None"""
    if heavy_import in source:
        path.write_text(source.replace(heavy_import, lean_replacement, 1), encoding="utf-8")
    elif "Mira UNO Q core does not install" not in source:
        raise RuntimeError(f"Unexpected LeRobot 0.4.1 motor source at {path}; refusing an unsafe patch.")
    print(f"Patched lean motor runtime: {path}")


if __name__ == "__main__":
    main()
