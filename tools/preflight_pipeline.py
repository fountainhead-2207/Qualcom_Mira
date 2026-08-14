"""
Kiểm tra toàn bộ hạ tầng trước khi test pipeline giọng nói bằng người.

Chạy cái này trước khi nói "Mira ..." vào mic. Nó kiểm tra từng mắt trong chuỗi
và chỉ ra đúng chỗ hỏng kèm lệnh sửa, thay vì để phát hiện giữa lúc đang test
rồi phải dò ngược.

Không điều khiển tay máy, không phát tiếng - chỉ đọc trạng thái.

Usage:
    python preflight_pipeline.py
"""
import json
import subprocess
import sys
import urllib.request

BOARD = "arduino@192.168.1.41"
SSH_KEY = "C:/Users/LENOVO/.ssh/id_ed25519_unoq"
GPU_HOST = "61.28.228.23"
CHAT_URL = "http://127.0.0.1:8766/chat"
BT_MAC = "4C:3C:8F:3C:42:EE"
EXPECTED_MOTIONS = {"wave", "dance", "nod", "yes", "no", "shake", "scan",
                    "thinking", "shrug", "point", "bow", "celebrate",
                    "curious-tilt", "rest"}

results = []


def check(name, ok, detail="", fix=""):
    results.append((name, ok, detail, fix))
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    if not ok and fix:
        print(f"       fix: {fix}", flush=True)
    return ok


def board(cmd, timeout=25):
    return subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
         BOARD, cmd],
        capture_output=True, text=True, timeout=timeout + 10)


def gpu(cmd, timeout=30):
    return subprocess.run(["ssh", GPU_HOST, cmd],
                          capture_output=True, text=True, timeout=timeout + 10)


print("=" * 64)
print("Mira pipeline preflight")
print("=" * 64)

# 1. board reachable
try:
    r = board("echo alive")
    check("Board reachable qua SSH", r.returncode == 0 and "alive" in r.stdout,
          fix="kiểm tra board đã bật và cùng mạng; IP có thể đã đổi (hostname -I)")
except Exception as e:
    check("Board reachable qua SSH", False, str(e))

# 2. arm plugged into the board
r = board("ls /dev/serial/by-id/ 2>/dev/null | grep -c USB_Single_Serial || echo 0")
n_arms = int((r.stdout.strip() or "0").splitlines()[-1])
check("Tay máy cắm vào board", n_arms >= 1, f"{n_arms} thiết bị serial",
      fix="rút tay máy khỏi Windows và cắm vào hub của board")

# 3. motions present
r = board("mira-robot list 2>&1")
listed = {line.split()[0] for line in r.stdout.splitlines() if line.strip() and not line.startswith(" ")}
missing = EXPECTED_MOTIONS - listed
check("Đủ 14 động tác", not missing and r.returncode == 0,
      f"{len(listed)} động tác" + (f", thiếu: {sorted(missing)}" if missing else ""),
      fix="xem docs/gesture_recording_howto.md bước 3-4")

# 4. mic present
r = board("arecord -l 2>/dev/null | grep -ci boyamic || echo 0")
check("Mic BOYAMIC được nhận", int((r.stdout.strip() or "0").splitlines()[-1]) > 0,
      fix="cắm lại mic; kiểm tra lsusb có 10d6:b011")

# 5. bluetooth speaker
r = board(f"bluetoothctl info {BT_MAC} 2>/dev/null | grep -c 'Connected: yes' || echo 0")
bt_ok = int((r.stdout.strip() or "0").splitlines()[-1]) > 0
check("Loa JBL đã kết nối", bt_ok,
      fix=f"bật loa rồi: ssh ... \"bluetoothctl connect {BT_MAC}\"")

# 6. bluealsa running (PipeWire is masked - see uno-q-board.md)
r = board("systemctl is-active bluealsa 2>/dev/null || echo inactive")
check("bluealsa đang chạy", "active" in r.stdout,
      fix="sudo systemctl start bluealsa  (KHÔNG dùng PipeWire, nó đã bị mask)")

# 7. windows -> gpu tunnel
try:
    r = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=30)
    check("Tunnel 8766 mở trên Windows", ":8766" in r.stdout and "LISTENING" in r.stdout,
          fix="ssh -f -N -L 0.0.0.0:8766:localhost:8766 61.28.228.23")
except Exception as e:
    check("Tunnel 8766 mở trên Windows", False, str(e))

# 8. chat server alive on the GPU box
r = gpu("ss -tlnp 2>/dev/null | grep -c 8766 || echo 0")
check("Chat server sống trên 4090", int((r.stdout.strip() or "0").splitlines()[-1]) > 0,
      fix="ssh 61.28.228.23 \"tmux new-session -d -s mira_chat 'cd /data/qualcom-robotic "
          "&& molmoact2-env/bin/python -u mira_chat_server.py > mira_chat_server.log 2>&1'\"")

# 9. board can actually reach the chat server through the tunnel
r = board("curl -s -m 20 -o /dev/null -w '%{http_code}' -X POST "
          "http://192.168.1.32:8766/chat -H 'Content-Type: application/json' "
          "-d '{\"text\": \"chào mira\"}'", timeout=40)
check("Board gọi được chat server", r.stdout.strip().endswith("200"),
      f"HTTP {r.stdout.strip()}",
      fix="tunnel phải bind 0.0.0.0 (không phải 127.0.0.1) để board thấy được")

# 10. manipulation routing works
try:
    body = json.dumps({"text": "tôi đánh rơi cái tua vít không nhặt được, "
                               "bạn nhặt giúp mình nhé"}).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        out = json.loads(resp.read())
    check("Routing sang MolmoAct2", bool(out.get("task")),
          f"task={out.get('task')!r}",
          fix="xem SYSTEM_PROMPT trong mira_chat_server.py")
except Exception as e:
    check("Routing sang MolmoAct2", False, str(e))

# 11. no stale arecord holding the mic
r = board("pgrep -c arecord 2>/dev/null || echo 0")
stale = int((r.stdout.strip() or "0").splitlines()[-1])
check("Không có arecord treo", stale == 0, f"{stale} tiến trình",
      fix="pkill arecord  (nếu không thì board_voice_control.py sẽ không mở được mic)")

print("\n" + "=" * 64)
failed = [name for name, ok, _, _ in results if not ok]
if failed:
    print(f"{len(failed)}/{len(results)} bước lỗi: {', '.join(failed)}")
    print("Sửa các bước trên rồi chạy lại preflight.")
    sys.exit(1)
print(f"Tất cả {len(results)} bước OK. Chạy test:")
print(f'  ssh -i {SSH_KEY} {BOARD} \\')
print('    "cd /home/arduino && ~/.local/share/mira-so101/venv/bin/python -u '
      'board_voice_control.py 180"')
print("\nRồi làm theo kịch bản trong docs/pipeline_test_plan.md")
