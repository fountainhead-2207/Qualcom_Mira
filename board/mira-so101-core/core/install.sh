#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_HOME="${MIRA_TARGET_HOME:-$HOME}"
INSTALL_DIR="$TARGET_HOME/.local/share/mira-so101"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$TARGET_HOME/.local/bin"
HF_DIR="$TARGET_HOME/.cache/huggingface/lerobot"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

case "$(uname -m)" in
  aarch64|arm64) ;;
  *) [[ "${ALLOW_NON_ARM:-}" == "1" ]] || fail "Expected UNO Q ARM64 Linux; detected $(uname -m)." ;;
esac
[[ -f "$BUNDLE_DIR/SHA256SUMS" ]] || fail "Bundle checksum manifest is missing."
[[ -d "$BUNDLE_DIR/wheelhouse" ]] || fail "Bundled ARM64 Python packages are missing."
(
  cd "$BUNDLE_DIR"
  sha256sum -c SHA256SUMS
) || fail "Bundle integrity check failed."

available_kb="$(df -Pk "$TARGET_HOME" | awk 'NR==2 {print $4}')"
(( available_kb >= 1048576 )) || fail "At least 1 GB of free storage is required."
python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$python_version" == "3.13" ]] || fail \
  "This verified bundle targets the current UNO Q Debian 13 Python 3.13; detected Python $python_version. Update the UNO Q OS image first."

printf '%s\n' \
  'Installing the SO-101 motion core only.' \
  'No Torch, AI models, training, inference, cameras, or web stack will be installed.'
if ! python3 -c 'import ensurepip, venv' >/dev/null 2>&1; then
  printf '%s\n' 'Debian Python venv support is missing; installing that OS package now.'
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3 python3-venv ca-certificates
  sudo apt-get clean
fi
if getent group dialout >/dev/null; then
  sudo usermod -aG dialout "$(id -un)"
fi

mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$HF_DIR/local" \
  "$HF_DIR/calibration/robots/so101_follower" \
  "$HF_DIR/calibration/teleoperators/so101_leader"
cp -a "$BUNDLE_DIR/core/runtime.py" "$BUNDLE_DIR/core/mira-robot" "$INSTALL_DIR/"
cp -a "$BUNDLE_DIR/hf_lerobot/local/." "$HF_DIR/local/"

for relative in \
  calibration/robots/so101_follower/my_follower.json \
  calibration/teleoperators/so101_leader/my_leader.json; do
  source_file="$BUNDLE_DIR/hf_lerobot/$relative"
  target_file="$HF_DIR/$relative"
  if [[ -f "$target_file" ]] && ! cmp -s "$source_file" "$target_file"; then
    cp -a "$target_file" "$target_file.backup.$(date +%Y%m%d-%H%M%S)"
  fi
  cp -a "$source_file" "$target_file"
  chmod 600 "$target_file"
done

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --no-cache-dir --no-index \
  --find-links "$BUNDLE_DIR/wheelhouse" -r "$BUNDLE_DIR/core/requirements-core.txt"
"$VENV_DIR/bin/python" -m pip install --no-cache-dir --no-index \
  --find-links "$BUNDLE_DIR/wheelhouse" --no-deps 'lerobot==0.4.1'
"$VENV_DIR/bin/python" "$BUNDLE_DIR/core/patch_lerobot.py"

chmod +x "$INSTALL_DIR/mira-robot" "$INSTALL_DIR/runtime.py"
ln -sfn "$INSTALL_DIR/mira-robot" "$BIN_DIR/mira-robot"
sudo ln -sfn "$INSTALL_DIR/mira-robot" /usr/local/bin/mira-robot

"$VENV_DIR/bin/python" "$INSTALL_DIR/runtime.py" doctor --no-hardware

printf '%s\n' \
  '' \
  'SO-101 MOTION CORE INSTALLED' \
  'Log out and back in once so serial-port permissions apply.' \
  '' \
  'Then connect both arms and run:' \
  '  mira-robot doctor' \
  '  mira-robot teleop' \
  '  mira-robot replay wave' \
  '  mira-robot list' \
  '  mira-robot stop'
