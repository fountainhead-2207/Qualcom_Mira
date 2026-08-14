---
generated_from_commit: NO_GIT_REPO
generated_at: 2026-08-14T03:50:00+07:00
covers_paths:
  - (remote) duyetnt2@61.28.228.23:/data/qualcom-robotic/
verified_by_human: no
---

# Module: GPU training server ("duyetnt2-ai-lab", 61.28.228.23)

Personal RTX 4090 (24GB) box, reached by SSH jump: `ssh 61.28.228.23` from the Windows PC (port 234, user `duyetnt2`, key `~/.ssh/id_ed25519`, config pre-existing before this session). `/data` = `/dev/sdb1`, 148G total.

**Adjacent, unrelated, do-not-touch:** `~/.ssh/config` also has `103.196.237.13` (hostname `soc-gpu-test-01`, 4× RTX 5090, reached via `ProxyJump 61.28.228.23`) — a *different* shared server with its own `/data/LMbench`. Explicit user instruction this session: do not modify anything there. (A same-named-but-different `LMbench` also exists locally on *this* box at `/data/KV_cache/LMbench` — that one is in scope, see below.)

## Project directory: `/data/qualcom-robotic/`
- `wakeword-train-env/` — Python 3.10 venv (`uv venv --python 3.10`).
- `piper-sample-generator/` — cloned from github.com/rhasspy/piper-sample-generator, installed via `pip install piper-sample-generator` (PyPI, not editable — editable install from a cloned path was blocked once by the Claude Code auto-mode classifier; plain PyPI install worked).
- `mira_samples/` (3000 synthetic "Mira" TTS clips, LibriTTS-R multi-speaker generator, `models/en-us-libritts-high.pt`) and `mira_samples_augmented/` (volume/impulse-response/16kHz-resample augmented versions).

## Negative training data
Downloaded to `~/openwakeword-negative-data/` (home dir, on `/` — NOT `/data`, which only had 21GB free at the time): `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (17.3GB, precomputed embeddings, shape `(5625000, 16, 96)` float16) + `validation_set_features.npy` (185MB). Source: `huggingface.co/datasets/davidscripka/openwakeword_features` (CC-BY-NC-SA-4.0 — flag if this project ever goes commercial).

## Freed space
Deleted `/data/KV_cache/LMbench/models/Qwen3-32B-FP8` (32GB) at explicit user request, after user confirmed this — not the `103.196.237.13` `LMbench` — was the intended target. `/data` went from 21G → 53G free as a result.

## Training environment version pin (unresolved as of last sync)
`torch` auto-resolved via `uv pip install torch --torch-backend=auto` to `2.13.0+cu132`, but `torchaudio` (an openwakeword `train.py` dependency, via `openwakeword/data.py`) has no matching `cu132` build — latest available is `2.11.0+cu130`, and PyTorch's own runtime check hard-fails on the mismatch. Fix in progress: reinstall both `torch` and `torchaudio` together from `--index-url https://download.pytorch.org/whl/cu130` so they share a CUDA build tag. This was interrupted at least once by a timeout fetching `nvidia-cusparse` from `pypi.nvidia.com` — retry with `UV_HTTP_TIMEOUT=300` if it recurs.

## Unverified — needs human confirmation
- Whether cu130 vs cu132 makes any real performance difference on this RTX 4090 for the (small) openWakeWord training job — likely not, but not measured.
- Full list of `openwakeword/train.py` transitive deps — installed reactively (torchinfo, torchmetrics, pronouncing, torch_audiomentations, speechbrain, mutagen, acoustics) one ModuleNotFoundError at a time; there may be more not yet hit.

## Wake-word training: resolved outcome
The cu130/cu132 mismatch above was resolved (torch+torchaudio reinstalled together from the cu130 index). The trained "mira" model initially exported via `torch.onnx.export`'s default **dynamo** exporter (PyTorch ≥2.9 default) and was silently non-functional (scored ~0.000-0.002 even on its own training-positive samples despite 94.9% reported training accuracy) — fixed by passing `dynamo=False, external_data=False` explicitly. Final model: `mira_model/mira.onnx` (205KB, single self-contained file), verified live at 0.976-0.982 confidence against real speech on both this deployment path and the board.

## `/data` vs `/` partition trap (disk space)
`/data` (`/dev/sdb1`, 148G) is the large partition and is where all project work should live. **`/` (`/dev/sda3`, only 75G) is a separate, much smaller partition** that includes `/home/duyetnt2` and therefore the *default* HuggingFace cache location (`~/.cache/huggingface`) — any `from_pretrained(...)`/`snapshot_download(...)` call that doesn't have `HF_HOME` explicitly set will silently fill up the small `/` partition instead of `/data`, and did: a MolmoAct2 base-checkpoint download filled `/` to 99% (760MB free) and crashed mid-download with `No space left on device` even though `/data` had 36GB free at the time. Fixed by moving `~/.cache/huggingface/hub` to `/data/qualcom-robotic/hf-cache/hub` and symlinking it back, **and** by setting `export HF_HOME=/data/qualcom-robotic/hf-cache` before running anything that touches the HF cache on this box going forward — always set this env var here.

## MolmoAct2 inference environment (`/data/qualcom-robotic/molmoact2-env/`)
- Needs **Python ≥3.12** (`lerobot[molmoact2]` extra requires it) — the box's system Python is 3.10, so a standalone Python 3.12.13 was installed via `uv python install 3.12`, kept self-contained under `/data/qualcom-robotic/tools/uv-python/` (not system-wide). `uv` itself was installed as a single portable binary at `/data/qualcom-robotic/tools/uv-x86_64-unknown-linux-gnu/uv` (no system package changes), per explicit user instruction to keep everything contained in the project folder.
- Venv created with `uv venv --python 3.12` (note: **`uv venv` does not bundle `pip`** — use `uv pip install --python <venv>/bin/python ...`, not `<venv>/bin/pip`, or the venv has no pip binary at all).
- Install command that actually worked: `uv pip install 'lerobot[molmoact2]>=0.6.1' --python ./molmoact2-env/bin/python --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`. The plain `--extra-index-url` alone (without `--index-strategy unsafe-best-match`) fails with a dependency-confusion-guard error, because uv finds an old `requests` on the PyTorch index and refuses to look at PyPI for a newer one by default. Also: don't `pip install lerobot[molmoact2]` without a version floor — PyPI's default resolve landed on `lerobot==0.4.4`, which predates the `molmoact2` extra entirely and fails with "does not provide the extra."
- Resulting versions: `lerobot==0.6.1`, `torch==2.11.0+cu130`, `transformers==5.5.4`, `peft==0.20.0`.
- Checkpoint: `/data/qualcom-robotic/mira_molmoact2_step2000/` — must have all **7** files (`config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_preprocessor_step_3_molmoact2_masked_normalizer.safetensors`, `policy_postprocessor.json`, `policy_postprocessor_step_1_molmoact2_masked_unnormalizer.safetensors`, `train_config.json`). An earlier relay from Windows only copied 4 of the 7 (missing the two `*.json` pipeline-definition files and `train_config.json`) — this breaks `make_pre_post_processors(..., pretrained_path=...)` with a `ProcessorMigrationError` claiming "requires migration," which is a red herring; the real problem is just missing files, copy them over from the Windows copy (`d:\Comp\Qualcom\mira_molmoact2_step2000\`) and it works.
- **Do not manually apply the "SO100/101 zero-shot" joint-frame correction** (`joint_signs=[1,-1,1,1,1,1]`, `joint_offsets=[0,90,90,0,0,0]`) documented on the LeRobot MolmoAct2 model card as needed for the *base* `allenai/MolmoAct2-SO100_101` checkpoint. This fine-tune's own `train_config.json` and saved `policy_preprocessor.json` both have `joint_signs: null, joint_offsets: null` — i.e. it was fine-tuned *without* that correction, so applying it now would introduce a new train/inference mismatch, not fix one. Always load via `MolmoAct2Policy.from_pretrained(CKPT)` + `make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT)` so the checkpoint's own saved (uncorrected) pipeline is what actually runs.
- Inference call shape: `predict_action_chunk` requires `inference_action_mode="continuous"` explicitly (raises otherwise) and expects an *already-preprocessed* batch — i.e. call `preprocessor(batch)` first, not the raw observation dict, or `_model_inputs` silently produces an empty dict and `predict_action_chunk` dies with `StopIteration`.
- Verified: model loads in ~150s cold (mostly CPU-side instantiation after weights are on disk, not GPU-bound), ~11GB VRAM, forward pass ~2-3.4s on synthetic input, output shape `[1,10,6]`, no NaN/Inf.

## Persistent inference server + monitoring
- `/data/qualcom-robotic/molmoact2_server.py`: stdlib-only `ThreadingHTTPServer` (no Flask/FastAPI dependency), loads the policy once at startup (avoid the ~150s cold-load per request), serves `POST /infer` (base64 camera1/camera2 JPEGs + 6-float state + task string → `{action: [1,10,6], latency_ms, warnings}`) and `GET /metrics` (Prometheus format via `prometheus_client`: `molmoact2_infer_requests_total{status}`, `molmoact2_infer_latency_seconds` histogram, `molmoact2_infer_warnings_total{kind}`). Run via bare `nohup`, not a systemd service — restart manually after any reboot or code change (`pgrep -f molmoact2_server.py` to find the PID, `kill -9` it, relaunch with `export HF_HOME=/data/qualcom-robotic/hf-cache && nohup .../python -u molmoact2_server.py > molmoact2_server.log 2>&1 &`).
- Port 8765 (and every port besides SSH) is **not** publicly reachable on this box — the Windows-side client/Prometheus reach it only through an SSH local port-forward (`ssh -N -L 8765:localhost:8765 61.28.228.23`, itself a bare background process on the Windows PC, not a service). If MolmoAct2 metrics/inference suddenly stop working from Windows, check the tunnel process first before assuming the server itself died.

## Human corrections
<!-- survives regeneration. agents must carry this block forward verbatim -->
(none yet)
