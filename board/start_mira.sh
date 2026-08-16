#!/bin/bash
# Khởi động mọi thứ chạy trên board, theo đúng thứ tự phụ thuộc.
#
# Vì sao cần: tất cả các dịch vụ trên board đều được bật bằng `nohup` trần, nên
# reboot là mất sạch - trừ bluetooth (đã `systemctl enable`). Trước đây phải nhớ
# 5 lệnh rời rạc, và mỗi lệnh có một cái bẫy riêng đã mắc phải thật:
#   - pgrep -x không khớp ustreamer (nó đổi tên thread), nên lệnh kill cũ không
#     giết được gì và tiến trình mới chết vì cổng bận
#   - camera_exporter phải chạy SAU ustreamer, không thì nó báo camera hỏng
#   - bluetoothd không chạy thì loa im, mà bluetoothctl lại treo chứ không báo lỗi
#
#   ./start_mira.sh          # khởi động những gì chưa chạy
#   ./start_mira.sh restart  # giết hết rồi bật lại
set -u

BOARD_IP=$(hostname -I | awk '{print $1}')
VENV=/home/arduino/wakeword-env/bin/python
SPEAKER=4C:3C:8F:3C:42:EE          # JBL Go 4
MODE=${1:-start}

say() { printf "%-34s %s\n" "$1" "$2"; }

running() { pgrep -f "$1" >/dev/null 2>&1; }

# Giết theo dòng lệnh đầy đủ chứ không theo tên tiến trình: ustreamer đổi tên
# thread nên `pgrep -x ustreamer` trả về rỗng và không giết được gì.
stop_all() {
    for pat in "board_voice_control.py" "dashboard_server.py" "log_tail_server.py" \
               "camera_exporter.py" "ustreamer --device"; do
        pgrep -f "$pat" | while read -r pid; do kill "$pid" 2>/dev/null; done
    done
    sleep 3
}

start_one() {
    local name=$1 pattern=$2 cmd=$3 wait_s=${4:-2}
    if running "$pattern"; then
        say "$name" "đã chạy sẵn"
        return
    fi
    setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
    sleep "$wait_s"
    running "$pattern" && say "$name" "OK" || say "$name" "LỖI - xem log"
}

[ "$MODE" = "restart" ] && { echo "dừng hết..."; stop_all; }

echo "=== camera ==="
# Gán theo ID THIẾT BỊ, không theo /dev/videoN. Số thứ tự phụ thuộc thứ tự cắm:
# một lần cắm cam tay máy trước, nó chiếm video0 và hai camera đổi chỗ cho nhau -
# bảng theo dõi hiện cam tay máy trong ô "trên cao" và ngược lại, trông y như
# camera hỏng. Symlink trong /dev/v4l/by-id/ thì gắn với chính thiết bị.
OVERHEAD=/dev/v4l/by-id/usb-Generic_HD_video_20210901000000-video-index0
WRIST=/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0

# NOOP = đẩy thẳng MJPEG gốc của camera, không nén lại bằng CPU. Đo được:
# 8fps -> 25fps, và board không tốn CPU cho việc mã hoá.
for cam in "trên cao|$OVERHEAD|8080|ustreamer.log" "tay máy|$WRIST|8081|ustreamer_wrist.log"; do
    IFS='|' read -r label dev port log <<< "$cam"
    if [ ! -e "$dev" ]; then
        say "ustreamer $label :$port" "KHÔNG THẤY CAMERA (chưa cắm?)"
        continue
    fi
    start_one "ustreamer $label :$port" "port=$port" \
        "ustreamer --device=$dev --format=MJPEG --encoder=NOOP --resolution=640x480 --desired-fps=30 --host=0.0.0.0 --port=$port > /home/arduino/$log 2>&1" 4
done

echo "=== theo dõi ==="
# Sau ustreamer: exporter kiểm tra camera bằng cách gọi /snapshot, chạy trước
# thì nó báo camera hỏng cho tới vòng kiểm tra sau (10s).
start_one "camera_exporter :9101" "camera_exporter.py" \
    "python3 /home/arduino/camera_exporter.py > /home/arduino/camera_exporter.log 2>&1"
start_one "log_tail_server :9104" "log_tail_server.py" \
    "python3 /home/arduino/log_tail_server.py > /home/arduino/log_tail_server.log 2>&1"
start_one "bảng theo dõi :8088" "dashboard_server.py" \
    "cd /home/arduino/dashboard && MIRA_BOARD=127.0.0.1 MIRA_BIND=0.0.0.0 MIRA_DETECT_URL= MIRA_DASHBOARD_PORT=8088 python3 dashboard_server.py > /home/arduino/dashboard/server.log 2>&1"

echo "=== loa bluetooth ==="
if ! systemctl is-active --quiet bluetooth; then
    sudo systemctl start bluetooth && sleep 4
fi
sudo hciconfig hci0 up 2>/dev/null
if printf 'info %s\nquit\n' "$SPEAKER" | timeout 15 bluetoothctl 2>/dev/null | grep -q "Connected: yes"; then
    say "JBL Go 4" "đã nối"
else
    printf 'connect %s\nquit\n' "$SPEAKER" | timeout 40 bluetoothctl >/dev/null 2>&1
    sleep 3
    printf 'info %s\nquit\n' "$SPEAKER" | timeout 15 bluetoothctl 2>/dev/null | grep -q "Connected: yes" \
        && say "JBL Go 4" "OK" || say "JBL Go 4" "CHƯA NỐI - bật loa lên rồi chạy lại"
fi

echo "=== giọng nói (bật cuối, nó cần mic + loa sẵn sàng) ==="
start_one "board_voice_control :9103" "board_voice_control.py" \
    "$VENV /home/arduino/board_voice_control.py >> /home/arduino/voice_run.log 2>&1" 20

echo
echo "=== kiểm tra cổng ==="
for p in 8080 8081 8088 9101 9103 9104; do
    printf "  :%-5s " "$p"
    ss -tln 2>/dev/null | grep -q ":$p " && echo "UP" || echo "DOWN"
done
echo
echo "Bảng theo dõi (mở từ điện thoại cùng WiFi):  http://$BOARD_IP:8088"
