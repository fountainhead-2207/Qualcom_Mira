---
generated_from_commit: NO_GIT_REPO
generated_at: 2026-08-14T03:50:00+07:00
covers_paths:
  - d:\Comp\Qualcom\lerobot-local-env\
  - C:\Users\LENOVO\.local\bin\ (uv, lelab)
  - C:\Users\LENOVO\.cache\huggingface\lerobot\calibration\
  - C:\Users\LENOVO\.ssh\config
  - d:\Comp\Qualcom\phowhisper-medium-ct2\
  - d:\Comp\Qualcom\monitoring\
verified_by_human: no
---

# Module: Windows PC (this machine)

Role: mid-tier compute (train/test policies locally with a GPU) and the operator's own LeLab UI. Not the board, not the big GPU box.

## Hardware
- 32GB RAM, NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM), driver reports CUDA 13.2.

## Software installed
- `uv` (`C:\Users\LENOVO\.local\bin\uv.exe`), added to PATH.
- `lerobot-local-env` (`d:\Comp\Qualcom\lerobot-local-env\`): `lerobot==0.4.4`, `torch==2.13.0+cu132` (CUDA confirmed working, `torch.cuda.get_device_name(0)` → RTX 3070 Laptop GPU).
- **LeLab** (`uv tool install git+https://github.com/huggingface/leLab.git`) — installed as an isolated uv tool, own venv, `torch==2.11.0` (separate copy from `lerobot-local-env`, uv hardlinks share disk where possible). Launch: `lelab` → serves on `http://localhost:8000`.
- SSH config (`~/.ssh/config`) has: `unoq` (the board, key-only), and two pre-existing entries the user set up independently — `61.28.228.23` (jump host = the actual RTX 4090 box, port 234) and `103.196.237.13` (a *different* GPU server reached via `ProxyJump 61.28.228.23`, ***do not touch*** per explicit user instruction).

## Known constraint
LeLab's live calibrate/teleoperate needs the physical arms plugged into whichever machine runs it. The arms are physically on the UNO Q board, not here — so LeLab on this PC is currently useful only for: viewing/training on datasets already recorded elsewhere, not live control. The Windows copy of the follower/leader calibration (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`, `.../teleoperators/so_leader/so.json`) was produced by LeLab calibration sessions done directly on this PC (arms were briefly connected here) — this is the source of the "fresh" leader calibration pushed to the board earlier, and of the `so_follower/c2.json` follower calibration synced to the board's `my_follower.json` on 2026-08-14 (after diagnosing that the bundled canned motions' amplitude drift was a calibration-mismatch issue, not a code bug — see `uno-q-board.md`).

## Recording/recalibrating motions: use `lerobot-record` directly, not LeLab's web UI (2026-08-15)
Re-recording "nod" (to fix its calibration drift) and recording the first new gesture ("thinking") through LeLab's own Record page hit two real bugs, both confirmed by reading LeLab's own source (`C:\Users\<user>\AppData\Roaming\uv\tools\lelab\Lib\site-packages\lelab\`):
- **The dataset-name field silently strips `/` as you type it** (frontend bug), but the backend's `sanity_check_dataset_name` requires `repo_id.split("/")` to produce exactly 2 parts — so a plain name like `nod` crashes immediately with `ValueError: not enough values to unpack`.
- **Any status/log line containing an emoji crashes** on this machine's default `cp1252` console encoding (`UnicodeEncodeError`), including ones inside the actual recording loop itself (`print(f"🎬 STATUS CHANGE: Starting recording phase...")` in `lelab/record.py`) — meaning a session could crash *before ever calling `record_loop()`*, so nothing was actually captured, while the UI just showed a stuck spinner with no useful error.

**Fix**: launch LeLab (or `lerobot-record` directly) with `PYTHONIOENCODING=utf-8` in the environment — forces UTF-8 regardless of the console's codepage, and every emoji print stops crashing. For recording itself, bypass the web UI entirely and call the standalone CLI (installed as `lerobot-record.exe` inside LeLab's own venv, `...\lelab\Scripts\`):

```
PYTHONIOENCODING=utf-8 lerobot-record.exe ^
  --robot.type=so101_follower --robot.port=COM8 --robot.id=c2 ^
  --robot.calibration_dir="...\calibration\robots\so_follower" ^
  --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=c2 ^
  --teleop.calibration_dir="...\calibration\teleoperators\so_leader" ^
  --dataset.repo_id=mira/<name> --dataset.single_task="<description>" ^
  --dataset.num_episodes=1 --dataset.episode_time_s=5 --dataset.reset_time_s=1 ^
  --dataset.fps=30 --dataset.video=false --dataset.push_to_hub=false ^
  --play_sounds=false --display_data=false
```

No keyboard interaction needed for a single episode — it just records for `episode_time_s` seconds and saves (unlike LeLab's own wrapper, which auto-discards and re-records if the timer runs out without an explicit "advance" signal — a LeLab-specific behavior, not standard `lerobot-record`). **Must stop any LeLab-side recording/teleoperation session first** (`POST /stop-recording`) — both processes fight over the same COM ports otherwise (`SerialException: could not open port 'COM8': PermissionError`). The CLI still timestamps `repo_id` the same way LeLab does; rename the resulting folder (strip the `_<timestamp>` suffix, e.g. `motion_thinking`) before deploying to the board.

**Known artifact, not a calibration mismatch**: both "nod" and "thinking" recorded with a small, *constant* overshoot on `shoulder_lift.pos` only (~1.5-2.3% past the ±100 normalized bound, present in every one of the 150 frames, not a brief spike) — the board's `load_motion()` validation (`runtime.py`) rejects anything past ±100.001 and refuses to list *or* replay the motion at all until fixed. Root cause looks like the leader's calibrated `range_min` for that one joint being very slightly tighter than where it naturally sits during these motions, not a wrong calibration file. Fixed post-hoc by clipping the recorded parquet's `action` column to [-100, 100] (gripper to [0, 100]) rather than re-recording — a numpy one-liner, see the commit history for the exact script. Worth watching for on every new gesture recorded this way.

**Recalibrating the board's arm without unplugging it from the board**: use the board's own `lerobot-calibrate`-equivalent instead of LeLab — LeLab pulls in heavy deps (`accelerate`, `datasets`, likely `torch`) unsuitable for the board's 2GB RAM. The board's `SO101Follower.calibrate()` method only needs `self.bus` (no camera/torch imports) and runs as a plain interactive SSH session. See `uno-q-board.md` for `calibrate_follower.py`.

## Voice command pipeline (wake-word "Mira" + open-vocabulary STT)
**Superseded 2026-08-14**: this Windows-side pipeline was the first working version; the mic has since moved physically to the UNO Q board, which now runs the *entire* wake-word→STT→motion loop natively with no PC involved for the canned-command path (STT model also swapped there — see "Board-native voice pipeline" in `uno-q-board.md`). This section is kept as a fallback/reference build and because the STT swap (PhoWhisper → Zipformer) happened here first before being ported to the board.
- `openWakeWord` custom "mira" model (`mira.onnx`, trained on the RTX 4090, see `gpu-training-server.md`) triggers on the wake word; venv is `lerobot-local-env` (openwakeword 0.6.0 — needed `openwakeword.utils.download_models()` once to fetch bundled base models, since 0.6.0 doesn't ship them like the board's 0.4.0 did).
- STT: **`hynt/Zipformer-30M-RNNT-6000h`** via `sherpa-onnx` (`OfflineRecognizer.from_transducer`, int8 ONNX, files at `d:\Comp\Qualcom\zipformer-vi\`, `tokens.txt` generated from the repo's `bpe.model` via `sentencepiece` since the repo doesn't ship it). **Replaced `vinai/PhoWhisper-medium`** (previously used, itself a swap from generic Whisper-medium for better Vietnamese accent handling) at explicit user request: 30M params vs 769M, and markedly faster (~90-110ms per ~4s clip on the RTX 3070's CPU path alone, vs PhoWhisper's ~1-2s), with several exact-match transcriptions in live testing that PhoWhisper needed alias-table workarounds for. `faster-whisper`/CTranslate2 and the `phowhisper-medium-ct2` conversion are no longer used by this script but left on disk.
- Language is Vietnamese throughout.
- `sys.stdout.reconfigure(encoding="utf-8")` is required at the top of any script printing Vietnamese text on Windows, or it crashes with `UnicodeEncodeError` under the default cp1252 console codepage.
- Command matching is substring-based with a hand-maintained alias table per motion (`wave`, `dance`, `clean`, `scan`, `shake`, `nod`, `yes`, `no`) covering observed real mis-transcriptions (e.g. `"bảy tay"→wave`, `"lắt đầu"→shake`, `"mu"→dance`) — ultra-common short Vietnamese function words (`có`, `không`) require a whole-word match, not substring, to avoid false-positives on unrelated sentences.
- A ~1.2s pre-roll audio buffer is kept and prepended to the post-trigger recording window, because wake-word detection latency means the command word can start being spoken before the trigger actually fires.
- **Mic device gotcha (2026-08-14)**: the mic in use (a Boya USB mic, later moved to the board) showed near-zero signal via its **MME** device index despite Windows' own input-level meter confirming strong signal — MME can carry an independent, separately-muted gain path from the modern **WASAPI** device of the same physical mic. Fix: select the device by its WASAPI name string, not a bare MME index. WASAPI shared-mode streams also refuse arbitrary sample rates (unlike MME, which resamples internally) — `sd.InputStream` must open at the device's native rate (48000Hz here) and the script downsamples to 16kHz itself via `scipy.signal.resample_poly` before feeding the wake-word/STT models. After the mic moved to the board, this script's `DEVICE` falls back to the laptop's built-in "Microphone Array (Intel Smart Sound Technology...)" WASAPI device.
- **Real-time responsiveness bug, found and fixed live (2026-08-14)**: calling "Mira" repeatedly got no response, and the user observed the whole loop taking ~20s from speech to execution despite the model itself inferring in under a second. Root cause: `mira-robot replay` was invoked via a *blocking* `subprocess.run` over SSH — while it waited for the SSH call (which itself waits for the physical motion to finish playing, 4-33s depending on the motion), the microphone's background callback kept filling an unbounded queue that nothing was draining, so the next "Mira" was answered off several-seconds-stale audio. A second, smaller version of the same bug existed even when no command matched at all, from the ~4-6s recording+transcribe window alone. Fixed by (1) moving the SSH/motion call to a background thread so the main loop keeps consuming audio in real time, (2) explicitly draining the queue after every trigger cycle — matched, unmatched, or rejected as too quiet — and (3) ignoring (not queuing) new wake-word triggers while a motion is still in flight, with a clear "robot busy" message instead of silence.
- Script: `d:\Comp\Qualcom\voice-control\wakeword_to_whisper.py` (moved out of scratchpad into a permanent project location on 2026-08-14, alongside its `mira.onnx` model; debug capture WAVs go to `voice-control\debug_logs\`). Exposes Prometheus metrics on `:9102` (`wakeword_score`, `wakeword_triggers_total`, `command_audio_peak`/`_rms`, `command_recognized_total{motion}`) while running — only up in Prometheus while someone has it open. Run directly: `lerobot-local-env\Scripts\python.exe voice-control\wakeword_to_whisper.py <seconds>`.
- `d:\Comp\Qualcom\voice-control\collect_voice_data.py`: prompts the wake word + every command word/phrase, records one clip each, saves audio+expected-text pairs to `voice-control\voice_data\manifest.jsonl` — for measuring real STT accuracy against known-correct text, and as the shape of data a future fine-tune would need. Not yet run for a full session.

**Safety fix (2026-08-14)**: a live false-positive was observed and fixed — a wake-word trigger on near-silent background noise (rms≈3.6) caused the STT model (PhoWhisper at the time) to hallucinate a full, plausible Vietnamese sentence that happened to contain "có", which matched the `"yes"` alias and fired a real `nod` motion on the physical arm with no one having spoken. Fixed with a `MIN_COMMAND_RMS` floor (raised from an initial 150 to **200** after measuring 7 live real-speech samples ranging 230.9-607.5) — captured command-window audio below that RMS skips transcription entirely rather than trusting the word-matcher to filter hallucinated text.

## MolmoAct2 shadow-inference client (Windows side)
- `shadow_client.py` (scratchpad): fetches a real overhead-camera1 JPEG from the board's `ustreamer` snapshot endpoint (`http://192.168.1.41:8080/snapshot`), builds a clearly-flagged gray placeholder for camera2 (wrist cam is hardware-broken, see `uno-q-board.md`), reads the follower's real current joint state via a read-only SSH call (`read_follower_state.py` on the board — connects, calls `get_observation()`, disconnects with `disable_torque=True`, never writes to motors), and POSTs all three to the RTX 4090 inference server.
- **Never sends anything to the robot's motors.** Every response's `camera2_is_placeholder` warning is preserved through to the JSON log so downstream readers can't mistake this for a real evaluation.
- Verified end-to-end: real action shape `[1,10,6]`, no NaN/Inf, first predicted action step numerically close to the real current state (a good sanity signal per `runpod_handoff.md`'s acceptance criteria).
- Logs to `shadow_logs/shadow_<timestamp>.json`.
- Depends on an SSH local port-forward tunnel (`ssh -N -L 8765:localhost:8765 61.28.228.23`, currently a bare background process, not a service) because the 4090 box only exposes its SSH port publicly — port 8765 is not directly reachable from the internet.

## Monitoring stack (Prometheus + Grafana, both on this PC)
- `d:\Comp\Qualcom\monitoring\prometheus\` (v3.0.1 Windows binary) + `d:\Comp\Qualcom\monitoring\grafana-v11.4.0\` (v11.4.0 Windows binary) — both run as bare background processes (`prometheus.exe`, `grafana-server.exe`), **not installed as Windows services**, so they will not survive a reboot or Windows Update restart without being manually relaunched.
- Prometheus (`:9090`) scrapes two targets: `192.168.1.41:9101` (board's `camera_exporter.py`, stdlib-only, no deps — see `uno-q-board.md`) and `127.0.0.1:8765/metrics` (the MolmoAct2 server on the 4090, reached through the SSH tunnel above — this target goes down if the tunnel dies, independent of the 4090 server's own health).
- Grafana (`:3000`, default `admin`/`admin` — **not changed from default, do this before exposing beyond localhost**) has one dashboard, `mira-overview` (`d:\Comp\Qualcom\monitoring\dashboard.json`, provisioned via the HTTP API, not file-based provisioning): overhead/wrist camera up-down, overhead camera last-success age, MolmoAct2 p50/p95 latency, request rate by status, warning rate by kind.
- Verified with real traffic (not just target-up checks): one real shadow-inference call's latency (~2.16s) round-tripped correctly into the `molmoact2_infer_latency_seconds` histogram.
- Wake-word/STT side exports metrics too now (`wakeword_to_whisper.py` on `:9102`, added 2026-08-14: `wakeword_score`, `wakeword_triggers_total`, `command_audio_peak`/`_rms`, `command_recognized_total{motion}`) — Prometheus scrapes it as job `wakeword_whisper` (5s interval, since it's only ever up while someone has the script running in the foreground). The board-native pipeline's equivalent (`board_wakeword.py`, `:9103` on the board) is not yet added as a scrape target. Dashboard panels for these were added to `mira-overview` (wake-word score timeseries, command audio level, trigger rate, recognition outcomes).

## Human corrections
<!-- survives regeneration. agents must carry this block forward verbatim -->
(none yet)
