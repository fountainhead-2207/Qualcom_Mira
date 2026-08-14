# Hướng dẫn tự ghi gesture, từ đầu đến cuối

Toàn bộ chạy từ terminal thường trên máy Windows (lệnh dưới viết theo kiểu Git Bash). Làm theo nhịp của bạn — không có gì bị tính thời gian trừ đúng cửa sổ ghi 5 giây, và cửa sổ đó chỉ bắt đầu khi bạn chạy lệnh ghi.

Đặt sẵn mấy biến này một lần cho mỗi terminal để không phải gõ lại:

```bash
LR="C:/Users/LENOVO/AppData/Roaming/uv/tools/lelab/Scripts"
PY="$LR/python.exe"
CAL="C:/Users/LENOVO/.cache/huggingface/lerobot/calibration"
export PYTHONIOENCODING=utf-8      # bắt buộc, không có thì emoji trong log của lerobot làm crash console
```

**Trước khi làm gì**: nếu LeLab web UI đang mở và đang có phiên record/teleop chạy, phải dừng nó — nó giữ COM7/COM8 nên CLI không mở được cổng (`SerialException: Access is denied`). Chạy `curl -s -X POST http://127.0.0.1:8000/stop-recording`, hoặc đóng hẳn LeLab.

---

## Bước 0 (chỉ khi bản ghi ra kết quả "chết", không có chuyển động) — kiểm tra tay máy có đọc được không

```bash
cd d:/Comp/Qualcom
$PY tools/read_arm_raw.py 45
```

Script tự tắt torque nên tay xoay tự do bằng tay. Trong 45 giây đó, xoay từng khớp qua hết tầm; script in ra ngay khi phát hiện mỗi khớp chuyển động, và in tổng kết ở cuối. **Cả 6 khớp đều phải báo có chuyển động.** Khớp nào bị liệt vào `NEVER MOVED` mà bạn thật sự đã xoay nó thì đó là lỗi phần cứng/bus thật, cần xử lý trước khi ghi tiếp.

`tools/read_arm_live.py 15` cũng tương tự nhưng hiện cả 2 tay cạnh nhau kèm giá trị normalized — hữu ích để xác nhận tay nào đang ở COM7, tay nào ở COM8.

Lưu ý: cả 2 script này khi kết thúc đều **để torque ở trạng thái tắt**, nên tay follower sẽ mềm oặt (đỡ nó nếu đang giơ cao) và không mirror theo leader nữa cho tới khi bạn chạy lệnh record/teleop lần sau.

---

## Bước 1 — ghi

Thay `NAME` (dùng làm tên thư mục) và `TASK` (mô tả ngắn bằng tiếng Anh, theo đúng cách các motion cũ được đặt, ví dụ `"wave hello"`):

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

Việc ghi bắt đầu **ngay khi bạn thấy dòng `Recording episode 0`** — bạn có đúng 5 giây (150 frame ở 30fps, khớp với mọi motion có sẵn). **Xoay tay LEADER**; tay follower sẽ mirror theo, và chính vị trí của tay leader là thứ được lưu. Không cần bấm phím gì: hết 5 giây nó tự lưu và tự thoát.

Hai điểm cần biết:
- `--dataset.repo_id` **bắt buộc phải có dấu `/`** (`mira/NAME`, không được chỉ `NAME`). LeRobot cắt chuỗi theo dấu đó, thiếu là crash với lỗi `not enough values to unpack`. Đây cũng là lý do không dùng được LeLab web UI cho việc này — ô nhập tên của nó tự động xóa dấu `/` khi bạn gõ.
- Thư mục lưu ra sẽ bị thêm timestamp vào tên (`mira/NAME_20260815_001032`). Bình thường, bạn đổi tên ở bước 3.

Muốn gesture dài hơn? Tăng `--dataset.episode_time_s`. Cứ để 5 giây trừ khi gesture thật sự cần dài hơn, để đồng nhất với các cái khác.

---

## Bước 2 — phát lại và tự đánh giá

```bash
$PY -m lerobot.scripts.lerobot_replay \
  --robot.type=so101_follower --robot.port=COM8 --robot.id=c2 \
  --robot.calibration_dir="$CAL/robots/so_follower" \
  --dataset.repo_id=mira/NAME_<timestamp> --dataset.episode=0 \
  --dataset.fps=30 --play_sounds=false
```

Lệnh này điều khiển tay follower thật chạy lại đúng những gì bạn vừa ghi. Nếu thấy không ổn, xóa thư mục đó rồi làm lại bước 1 — chưa có gì khác bị ảnh hưởng:

```bash
rm -rf "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_<timestamp>"
```

Cách kiểm tra nhanh bằng số xem bản ghi có thật sự bắt được chuyển động không (hữu ích khi phát lại mà thấy như không có gì xảy ra):

```bash
"d:/Comp/Qualcom/lerobot-local-env/Scripts/python.exe" -c "
import json,glob
f=glob.glob('C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_*')[0]
s=json.load(open(f+'/meta/stats.json'))['action']
print('range per joint:', [round(b-a,1) for a,b in zip(s['min'],s['max'])])"
```

Để so sánh: motion `wave` gốc có biên độ mỗi khớp khoảng `[31, 61, 54, 101, 15, 33]`. Nếu các số gần bằng 0 thì bản ghi không bắt được gì cả — xem lại bước 0.

---

## Bước 3 — chuẩn hóa, đổi tên, đẩy lên board

Mọi gesture ghi được tới giờ đều bị vượt biên một chút và rất ổn định ở khớp `shoulder_lift` (khoảng 1.5–2.5% quá giới hạn ±100 mà board bắt buộc). Board từ chối **kể cả việc liệt kê** một motion vượt biên (`Body joint outside normalized range`), nên phải clip trước:

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

Rồi đổi tên theo quy ước `motion_` và copy sang board:

```bash
mv "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/NAME_<timestamp>" \
   "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/motion_NAME"

scp -i C:/Users/LENOVO/.ssh/id_ed25519_unoq -r \
  "C:/Users/LENOVO/.cache/huggingface/lerobot/mira/motion_NAME" \
  arduino@192.168.1.41:/home/arduino/.cache/huggingface/lerobot/local/
```

`/home/arduino/.cache/huggingface/lerobot/local/` mới là đường dẫn có tác dụng thật — `~/mira-so101-uno-q-core/hf_lerobot/local/` chỉ là bundle cài đặt gốc, sửa ở đó không ảnh hưởng gì đến cái đang chạy.

---

## Bước 4 — đăng ký tên trên board

`mira-robot` tra tên từ một bảng cứng trong code, nên thư mục mới sẽ không được nhận diện cho tới khi bạn thêm vào. Sửa file `/home/arduino/.local/share/mira-so101/runtime.py`, thêm một dòng vào `MOTION_ALIASES`:

```python
"NAME": "motion_NAME",
```

Chỉ thêm những tên mà thư mục đã thật sự tồn tại — `mira-robot list` load dataset của **mọi** alias ngay từ đầu để hiện thời lượng, nên một tên thiếu thư mục sẽ làm hỏng toàn bộ danh sách.

Sau đó kiểm tra và thử trên tay máy thật (cần tay cắm vào board, không phải vào Windows):

```bash
ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 "mira-robot list"
ssh -i C:/Users/LENOVO/.ssh/id_ed25519_unoq arduino@192.168.1.41 "mira-robot replay NAME --yes"
```

## Bước 5 — cho Mira tự chọn được cử chỉ đó

Để lớp hội thoại ghép cử chỉ này với câu trả lời, chuyển tên đó từ `PROPOSED_MOTIONS` sang `EXISTING_MOTIONS` trong `voice-control/mira_chat_server.py`, rồi deploy lại và restart trên máy GPU:

```bash
scp voice-control/mira_chat_server.py 61.28.228.23:/data/qualcom-robotic/
ssh 61.28.228.23 "tmux kill-session -t mira_chat 2>/dev/null; \
  tmux new-session -d -s mira_chat 'cd /data/qualcom-robotic && \
  molmoact2-env/bin/python -u mira_chat_server.py > mira_chat_server.log 2>&1'"
```

(dùng tmux, không dùng `nohup` — máy đó kill process nền khi phiên SSH đóng.)

---

## Còn phải ghi

`point`, `bow`, `celebrate`, `curious_tilt` — xem `gesture_proposals.md` để biết mỗi cái nên trông như thế nào và tại sao cần nó. Đã xong: `thinking`, `shrug`, cùng với `nod` đã ghi lại theo calibration hiện tại.

**Lưu ý**: 2 bản `thinking` và `shrug` ghi trước đó gần như không có chuyển động thật (do lệch thời gian giữa lúc báo và lúc thao tác), nên nên ghi lại cả 2 theo hướng dẫn này.
