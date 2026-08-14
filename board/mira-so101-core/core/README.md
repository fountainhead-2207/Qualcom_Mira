# SO-101 UNO Q Motion Core

This is the stripped deployment for moving the robot today. It contains only:

- SO-101 leader-to-follower teleoperation
- Seven recorded joint-motion datasets
- The `my_follower` and `my_leader` calibration files
- Feetech serial motor support
- Safety alignment, process locking, STOP, and hardware checks

It does **not** install Torch, AI models, ACT, DLA, training, inference, cameras,
OpenCV, Agora, Gemini, FastAPI, or the Mira web interface. The LeRobot wheel is
installed without its normal dependencies; only its 0.7 MB SO-101 motor code is
used. A small patch prevents that motor path from importing unused AI packages.
All pinned Python 3.13 ARM64 packages are included in the archive, so pip does
not use the network or compile anything on the UNO Q. Apt is used only to make
sure Debian's standard virtual-environment package is present if the image is
missing it; current full UNO Q images normally already include it.

## UNO Q requirements

- Current official UNO Q Debian 13 image
- At least 1 GB free (the installed core is far smaller)
- Externally powered USB hub for the two SO-101 controller boards
- The exact physical arms that produced the bundled calibration files

## Install

Transfer directly from this Ubuntu workstation:

```bash
cd /home/viz/Downloads/agora/mira
./uno_q/transfer.sh YOUR_UNO_Q_USER@UNO_Q_IP
```

Or build the portable archive:

```bash
./uno_q/build_bundle.sh
```

Copy `dist/mira-so101-uno-q-core.tar.gz` to the UNO Q, then:

```bash
tar -xzf mira-so101-uno-q-core.tar.gz
./mira-so101-uno-q-core/core/install.sh
```

Log out and back in once after installation so the `dialout` group applies.

## Operate

Connect both arms and verify them without enabling movement:

```bash
mira-robot doctor
```

Teleoperate:

```bash
mira-robot teleop
```

Replay a motion:

```bash
mira-robot replay wave
mira-robot replay dance
mira-robot replay clean
```

Every movement asks for `MOVE`, safely aligns the follower for three seconds,
and disables follower torque on exit. For a future wake-word process, add
`--yes` after the command, for example `mira-robot replay wave --yes`.

Emergency software stop from another terminal:

```bash
mira-robot stop
```

The physical power switch remains the real emergency stop.

If automatic role detection cannot uniquely match the saved calibrations:

```bash
mira-robot teleop --follower-port /dev/ttyACM1 --leader-port /dev/ttyACM0
```

Never reuse these calibration files on newly assembled or mechanically changed
arms. In that case, recalibrate before using this motion kit.
