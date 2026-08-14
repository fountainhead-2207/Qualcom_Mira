# Kế hoạch test pipeline hoàn chỉnh

Phần LLM đã test xong bằng máy (xem `llm_model_selection.md`). Phần dưới đây cần
người có mặt vì nó điều khiển tay máy thật và phát tiếng qua loa.

## Chuẩn bị

1. **Cắm tay máy về board** (hiện đang cắm vào Windows để ghi gesture). Sau khi
   cắm, kiểm tra:
   ```bash
   ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 "mira-robot list"
   ```
   Phải thấy 13 động tác. Nếu báo "No USB serial devices found" thì tay chưa được
   board nhận.

2. **Loa JBL Go 4** bật, đã ghép đôi sẵn (`4C:3C:8F:3C:42:EE`). Kiểm tra:
   ```bash
   ssh ... arduino@192.168.1.41 "bluetoothctl info 4C:3C:8F:3C:42:EE | grep Connected"
   ```
   Nếu `Connected: no` thì `bluetoothctl connect 4C:3C:8F:3C:42:EE`. Lưu ý dùng
   **bluealsa**, không phải PipeWire (PipeWire đã bị mask - xem `uno-q-board.md`).

3. **Tunnel từ Windows sang 4090** phải đang chạy (board không tự ra Internet
   được, chỉ port 443):
   ```bash
   netstat -an | grep 8766     # phải thấy 0.0.0.0:8766 LISTENING
   ```
   Nếu không có: `ssh -f -N -L 0.0.0.0:8766:localhost:8766 61.28.228.23`

4. **Chat server trên 4090** phải sống:
   ```bash
   ssh 61.28.228.23 "ss -tlnp | grep 8766"
   ```
   Nếu chết: `ssh 61.28.228.23 "tmux new-session -d -s mira_chat 'cd /data/qualcom-robotic && molmoact2-env/bin/python -u mira_chat_server.py > mira_chat_server.log 2>&1'"`
   (dùng tmux - máy đó kill process nền khi phiên SSH đóng)

## Chạy

```bash
ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 \
  "cd /home/arduino && ~/.local/share/mira-so101/venv/bin/python -u board_voice_control.py 180"
```

Bỏ số giây ở cuối để nó chạy vô thời hạn (dừng bằng cách nói "Mira ngủ").

## Các trường hợp cần thử

| # | Nói gì | Mong đợi |
|---|---|---|
| 1 | "Mira vẫy tay" | Nhận diện ngay, tay vẫy. Không gọi chat server. |
| 2 | "Mira gật đầu" / "Mira nhảy" / "Mira lắc đầu" / "Mira quét" | Từng động tác tương ứng |
| 3 | "Mira ơi hôm nay tôi vui quá" | Tay vào **tư thế suy tư trước**, giữ đó → rồi phát tiếng trả lời qua loa → rồi làm cử chỉ (`celebrate`) |
| 4 | "Mira lấy giúp tôi ly nước" | Trả lời kiểu tinh nghịch là không làm được + `shrug` |
| 5 | "Mira biết giá vàng không" | Thừa nhận không biết, **không bịa số** |
| 6 | Nói chuyện bình thường không có chữ "Mira" | **Không** phản ứng gì (đúng thiết kế) |
| 7 | Gọi "Mira" 2 lần liền nhau | Lần 2 bị bỏ qua với thông báo "already waiting on a chat reply" |
| 8 | "Mira ngủ" | Chạy `rest` rồi dừng vòng lặp |

## Cần để ý

- **Độ trễ**: `thinking` dài 5s nhưng chat server chỉ mất ~1.2s. Vì `mira-robot`
  giữ khóa robot độc quyền, cử chỉ trả lời phải đợi hết 5s đó. Tiếng nói thì
  phát ngay khi có (không tranh khóa). Nếu thấy chậm quá thì ghi lại `thinking`
  ngắn hơn (~2-3s) - xem `gesture_recording_howto.md`.
- **Vọng âm**: chưa có triệt vọng âm. Nếu tiếng Mira còn đang phát mà mic đã
  nghe lại thì có thể nó tự nghe chính mình. Rủi ro thấp vì muốn chạy động tác
  vẫn cần đủ wake word + từ lệnh, nhưng nếu thấy nó tự trigger thì đây là lý do.
- **Lỗi bus nhất thời**: `Failed to write ... There is no status packet!` đã gặp
  2 lần trong lúc ghi gesture. Cả 2 lần `tools/scan_motors.py COM8` đều báo 6/6
  servo bình thường → chỉ cần chạy lại. Không phải lỗi cáp.
- Log wake word/ASR in ra stdout; metrics ở `:9103/metrics`.

## Việc còn tồn

- `board_voice_control.py` chưa phải systemd service, chạy tay qua SSH và không
  sống qua reboot.
- `:9103` chưa được thêm vào `prometheus.yml` (Windows) nên chưa có panel Grafana
  cho pipeline giọng nói phía board.
- Cam wrist (Sunplus) hỏng phần cứng, đã xác nhận 3 lần trên 2 OS - cần thay mới
  trước khi MolmoAct2 chạy thật (nó cần 2 camera).
- `clean` đã bị bỏ khỏi mọi đường gọi (sẽ do MolmoAct2 làm thật sau này).
