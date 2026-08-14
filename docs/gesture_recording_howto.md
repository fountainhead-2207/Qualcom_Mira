# Recording a gesture yourself, start to finish

Everything runs from a normal terminal on the Windows PC (Git Bash or PowerShell — commands below are Git Bash style). Do it at your own pace; nothing is timed except the 5-second recording window itself, and that only starts when you run the record command.

Set these once per terminal session so you don't repeat them:

```bash
LR="C:/Users/LENOVO/AppData/Roaming/uv/tools/lelab/Scripts"
PY="$LR/python.exe"
CAL="C:/Users/LENOVO/.cache/huggingface/lerobot/calibration"
export PYTHONIOENCODING=utf-8      # required, or emoji in lerobot's logs crash the console
```

**Before anything**: if LeLab's web UI is open and has a recording or teleop session running, stop it — it holds COM7/COM8 and the CLI can't open them (`SerialException: Access is denied`). `curl -s -X POST http://127.0.0.1:8000/stop-recording` does it, or just close LeLab.

---

## Step 0 (only if a recording comes out looking dead) — check the arm reads

```bash
cd d:/Comp/Qualcom
$PY tools/read_arm_raw.py 45
```

Torque gets disabled so the arm moves freely by hand. Move every joint through its range during the 45 seconds; it prints a line the moment it sees each joint move, and a summary at the end. **All six joints should report movement.** Any joint listed as `NEVER MOVED` while you were physically rotating it is a real hardware/bus problem worth chasing before recording anything else.

`tools/read_arm_live.py 15` is the same idea but shows both arms side by side with normalized values — useful to confirm which physical arm is on COM7 vs COM8.

Note both scripts leave torque **disabled** on exit, so the follower arm goes limp (support it if it's raised) and won't mirror the leader until you run a record/teleop command again.

---

## Step 1 — record

Replace `NAME` (used for the folder) and `TASK` (a short English description, matching how the existing motions are labeled, e.g. `"wave hello"`):

```bash
$PY -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower --robot.port=COM8 --robot.id=c2 \
  --robot.calibration_dir="$CAL/robots/so_follower" \
  --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=c2 \
  --teleop.calibration_dir="$CAL/teleoperators/so_leader" \
  --dataset.repo_id=mira/NAME --dataset.single_task="TASK" \
  --dataset.num_episodes=1 --dataset.episode_time_s=5 --dataset.reset_time_s=1 \
  --dataset.fps=30 --dataset.video=false --dataset.push_to_hub=false \
  --play_sounds=false --display_data=false
```

Recording starts the moment you see `Recording episode 0` — you get exactly 5 seconds (150 frames at 30fps, matching every existing motion). **Move the LEADER arm**; the follower mirrors it and the leader's positions are what get saved. No key press needed: when the 5s elapses it saves and exits on its own.

Two things worth knowing:
- `--dataset.repo_id` **must contain a `/`** (`mira/NAME`, not just `NAME`). LeRobot splits on it internally and crashes with `not enough values to unpack` otherwise. This is also why LeLab's web UI can't be used for this — its name field silently strips the `/` as you type.
- The saved folder gets a timestamp appended (`mira/NAME_20260815_001032`). That's expected; you rename it in step 3.

Want a longer gesture? Raise `--dataset.episode_time_s`. Keep it at 5 unless the gesture genuinely needs more, so it stays consistent with the others.

---

## Step 2 — play it back and judge it

```bash
$PY -m lerobot.scripts.lerobot_replay \
  --robot.type=so101_follower --robot.port=COM8 --robot.id=c2 \
  --robot.calibration_dir="$CAL/robots/so_follower" \
  --dataset.repo_id=mira/NAME_<timestamp> --dataset.episode=0 \
  --dataset.fps=30 --play_sounds=false
```

This drives the real follower arm through what you recorded. If it doesn't look right, delete the folder and redo step 1 — nothing else has been touched yet:

```bash
rm -rf "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_<timestamp>"
```

A quick numeric sanity check on whether the recording actually captured motion (useful when replay looks like nothing happened):

```bash
"d:/Comp/Qualcom/lerobot-local-env/Scripts/python.exe" -c "
import json,glob
f=glob.glob('C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_*')[0]
s=json.load(open(f+'/meta/stats.json'))['action']
print('range per joint:', [round(b-a,1) for a,b in zip(s['min'],s['max'])])"
```

For reference, the original `wave` has per-joint ranges of roughly `[31, 61, 54, 101, 15, 33]`. Ranges near zero mean nothing was captured — check step 0.

---

## Step 3 — normalize, rename, deploy

Every gesture recorded so far came out with a small constant overshoot on `shoulder_lift` (about 1.5–2.5% past the ±100 bound the board enforces). The board refuses to even *list* a motion that's out of range (`Body joint outside normalized range`), so clip it first:

```bash
"d:/Comp/Qualcom/lerobot-local-env/Scripts/python.exe" -c "
import pyarrow.parquet as pq, pyarrow as pa, numpy as np, glob
f=glob.glob('C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_*')[0]
p=f+'/data/chunk-000/file-000.parquet'
t=pq.read_table(p); a=np.array(t['action'].to_pylist()); c=a.copy()
c[:,:5]=np.clip(c[:,:5],-100,100); c[:,5]=np.clip(c[:,5],0,100)
print('clipped values:', int((c!=a).sum()))
t=t.set_column(t.schema.get_field_index('action'),'action',
               pa.array(c.tolist(), type=t.schema.field('action').type))
pq.write_table(t,p); print(f)"
```

Then rename to the `motion_` convention and copy to the board:

```bash
mv "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_<timestamp>" \
   "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/motion_NAME"

scp -i C:/Users/LENOVO/.ssh/id_ed25519_unoq -r \
  "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/motion_NAME" \
  arduino@192.168.1.41:/home/arduino/.cache/huggingface/lerobot/local/
```

`/home/arduino/.cache/huggingface/lerobot/local/` is the path that actually matters — `~/mira-so101-uno-q-core/hf_lerobot/local/` is just the original install bundle and changing it has no effect on what runs.

---

## Step 4 — register the name on the board

`mira-robot` resolves names from a hardcoded table, so a new folder isn't visible until you add it. Edit `/home/arduino/.local/share/mira-so101/runtime.py` and add a line to `MOTION_ALIASES`:

```python
"NAME": "motion_NAME",
```

Only add names whose folder actually exists — `mira-robot list` loads every alias's dataset up front to show its duration, so one missing folder breaks the entire listing.

Then verify and try it on the arm (needs the arms plugged into the board, not Windows):

```bash
ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 "mira-robot list"
ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 "mira-robot replay NAME --yes"
```

## Step 5 — let Mira actually choose it

For the conversational layer to pair the gesture with a reply, move the name from `PROPOSED_MOTIONS` to `EXISTING_MOTIONS` in `voice-control/mira_chat_server.py`, then redeploy and restart it on the GPU box:

```bash
scp voice-control/mira_chat_server.py 61.28.228.23:/data/qualcom-robotic/
ssh 61.28.228.23 "tmux kill-session -t mira_chat 2>/dev/null; \
  tmux new-session -d -s mira_chat 'cd /data/qualcom-robotic && \
  molmoact2-env/bin/python -u mira_chat_server.py > mira_chat_server.log 2>&1'"
```

(tmux, not `nohup` — that box kills bare background processes when the SSH session ends.)

---

## Still to record

`point`, `bow`, `celebrate`, `curious_tilt` — see `gesture_proposals.md` for what each is meant to look like and why. Already done: `thinking`, `shrug`, plus `nod` re-recorded under the current calibration.
