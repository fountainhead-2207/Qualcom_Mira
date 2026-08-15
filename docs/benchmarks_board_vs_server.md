# Đo thật: chạy trên board so với chạy trên server

Đo đêm 15→16/08/2026, trên chính phần cứng của dự án. Mục đích: trả lời câu hỏi
"gói hết được vào một board không?" bằng số chứ không bằng cảm tính.

Phần cứng:
- **Board**: Arduino UNO Q, 4× Kryo-V2 @2.0GHz (lớp Cortex-A53, **ARMv8.0 — không
  có dotprod/i8mm**), 1737MB RAM, **không có swap**
- **Server**: RTX 4090 thuê, 24.5GB VRAM

## Kết luận ngắn

| | board | server | chênh |
|---|---|---|---|
| Nghe + nhận dạng tiếng (ASR) | **189ms** | không cần | board thắng, đã chạy sẵn |
| Tổng hợp tiếng (TTS) | **1.04s** | 0.12s | board chậm 9× nhưng **vẫn dùng tốt** |
| Sinh câu trả lời (LLM) | **3.8–4.8s** | **0.45s** | server nhanh **~10×** |
| Chất lượng câu trả lời | 2/4 đúng (0.5B) | 95% (82/86) | server thắng áp đảo |

**Nghe, nói, cử chỉ, camera, bảng theo dõi: board làm được hết.**
**Hội thoại thì chưa** — không phải vì chậm, mà vì model đủ nhỏ để chạy thì trả
lời sai quá nửa.

## LLM trên board

### Ảnh hưởng của kích thước model

Qwen2.5 Instruct, Q4_K_M, 4 luồng (`llama-bench`, pp32/tg32):

| model | dung lượng | đọc prompt | sinh chữ |
|---|---|---|---|
| 0.5B (630M thật) | 463 MB | 16.65 tok/s | 10.26 tok/s |
| 1.5B (1.78B thật) | 1.04 GB | 5.45 tok/s | 3.69 tok/s |

Gấp 2.8× tham số thì chậm đi ~2.8×. Con 1.5B ra **~13.5s mỗi câu** — quá chậm.

### Ảnh hưởng của mức lượng tử hoá

Qwen2.5-0.5B, 4 luồng:

| | dung lượng | đọc prompt | sinh chữ |
|---|---|---|---|
| **Q4_0** | **403 MB** | **20.09** | **11.89** |
| Q4_K_M | 463 MB | 16.65 | 10.26 |
| Q8_0 | 639 MB | 18.84 | 10.43 |

**Q4_K_M là lựa chọn tệ nhất ở đây**: chậm hơn cả hai bản kia mà chất lượng chỉ ở
giữa. Q4_0 nhanh nhất vì llama.cpp repack riêng cho ARM. Q8_0 gần bằng Q4_K_M về
tốc độ nhưng chất lượng cao hơn — đáng cân nhắc nếu RAM cho phép.

### Ảnh hưởng của số luồng

Qwen2.5-0.5B Q4_K_M: 3 luồng 12.58/7.88 → 4 luồng 16.66/10.30 (**+30%**).
Dùng cả 4 nhân thì tranh với pipeline giọng nói; đo thấy pipeline vẫn sống, nhưng
đây là đánh đổi cần biết.

### Đòn bẩy lớn nhất: cache prompt

`SYSTEM_PROMPT` + few-shot của Mira = **1231 token**. Trên CPU này nó là phần đắt
nhất, đắt hơn cả việc sinh câu trả lời:

| | thời gian |
|---|---|
| Lần đầu, chưa cache | **71.6s** |
| Các lần sau, đã cache (`cache_prompt`) | **3.8 – 4.8s** |

`llama-server` giữ KV cache giữa các lượt nên chỉ phải trả giá một lần lúc khởi
động. Không có cache thì mỗi câu mất hơn một phút — vô dụng.

**Hệ quả cho hướng distill**: fine-tune nướng tính cách và quy tắc vào trọng số,
prompt 1231 token biến mất, chỉ còn ~15 token người dùng nói. Đó không phải tiết
kiệm nhỏ mà là bỏ hẳn khoản chi lớn nhất.

### Chất lượng — chỗ thật sự gãy

Cùng prompt, cùng few-shot, hỏi 4 câu qua `llama-server` (0.5B Q4_0):

| hỏi | đáp | |
|---|---|---|
| chào mira | `{"reply": "Chào bạn! Mình đây, sẵn sàng quậy rồi!", "motion": "wave"}` | ✅ nhưng chép nguyên ví dụ few-shot |
| hôm nay tôi mệt quá | `{"reply": "Thương bạn quá, ngủ đi!", "motion": "nod"}` | ✅ |
| mira **nhảy** đi | `{"reply": "Mình **nhặt cho rồi**...", "motion": "wave"}` | ❌ nhại nhầm ví dụ, sai cử chỉ, vi phạm quy tắc 7 |
| trời hôm nay đẹp không | `{"reply": "Oa, trời đẹp quá!"...}` | ❌ **bịa** — vi phạm quy tắc 6 |

Hai kiểu hỏng đều đúng như dự đoán cho model 0.6B: **nhại ví dụ gần nhất** và
**phá quy tắc**. Đó cũng chính là hai thứ mà distill từ teacher 4B sửa được —
và `tools/eval_chat.py` với 86 ca sẵn có sẽ chấm được bằng con số.

## Các thành phần khác trên board

| | đo được | RAM |
|---|---|---|
| ASR Zipformer-30M int8 | 189ms (gauge), 228–667ms mỗi câu | trong 287MB pipeline |
| TTS Piper vi_VN-vais1000-medium | **1.04s tổng hợp → 2.0s tiếng (2.0× thời gian thực)** | model 60MB, nạp 9.3s |
| `board_voice_control.py` | | 287 MB |
| `camera_exporter.py` | | 19 MB |
| `dashboard_server.py` | | **14 MB** |
| `log_tail_server.py` | | 12 MB |
| `ustreamer` ×2 (NOOP passthrough) | 25fps / 20fps | 3 + 2 MB |

RAM còn trống khi chạy tất cả: **1131 MB**.

**ustreamer**: chuyển từ mã hoá CPU sang `--encoder=NOOP` (đẩy thẳng MJPEG gốc
của camera) đưa cam trên cao từ **8fps lên 25fps** mà không tốn thêm CPU.

## Server

| | đo được |
|---|---|
| LLM Qwen3-4B-Instruct-2507 FP8 (vLLM) | 0.33 – 0.60s |
| TTS Piper (cùng giọng) | 0.07 – 0.23s |
| **Trọn vòng board → PC → 4090 → board** | **0.47 – 1.04s** |
| MolmoAct2 chỉ vào một vật | 0.84 – 1.63s |
| VRAM: vLLM 9.6GB, MolmoAct2 10.9GB | |

### MolmoAct2: mắt tốt, tay hỏng

Đo có đối chứng — chạy mỗi câu lệnh 3 lần để biết nhiễu của chính model, rồi so
chênh lệch giữa các lệnh với nhiễu đó:

| | nhiễu cùng 1 lệnh | chênh giữa các lệnh | tỉ lệ |
|---|---|---|---|
| 2 camera | 8.16 | 9.65 | **1.18×** |
| 1 camera | 6.13 | 6.62 | **1.08×** |

**Action head không phân biệt được với nhiễu ngẫu nhiên**, kể cả trên ảnh mẫu của
chính checkpoint. Cùng một lệnh, lần này quỹ đạo đi 30.00 đơn vị, lần sau 0.18.
Số camera không thay đổi gì (README của checkpoint cũng ghi "camera order does
not matter").

`lerobot/smolvla_base` chạy cùng phép thử: chênh lệch 4.26 — cũng không đạt. Nhưng
đó là **đúng thiết kế**: HF ghi rõ bản base phải fine-tune trên dữ liệu teleop của
chính mình.

Ngược lại, **phần nhìn thì dùng được**: hỏi "Point at the screwdriver" trả về toạ
độ đúng vật, 0.84–1.63s. Định dạng là `<points coords="1 1 X Y">` với X, Y theo
**phần nghìn, thứ tự (x, y)** — đã kiểm bằng cách vẽ cả hai cách đọc lên ảnh rồi
nhìn. `Detect all objects.` trả về `<boxes coords="1 1 x1 y1 x2 y2">`; còn hỏi
"bounding box of X" thì nó lặp vô hạn, đừng dùng.

## Cách tái lập

Trên board (`/llm` đã có sẵn llama.cpp b10444 và các model):

```bash
export LD_LIBRARY_PATH=/llm/llama-b10444
/llm/llama-b10444/llama-bench -m /llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 4 -p 32 -n 32 -r 2

# server thường trú + cache prompt (đây mới là cấu hình thật)
/llm/llama-b10444/llama-server -m /llm/qwen2.5-0.5b-instruct-q4_0.gguf \
  -t 4 -c 2048 --host 127.0.0.1 --port 8099
```

Trên server: `probe_controlled.py` (đối chứng nhiễu), `probe_molmo_points.py`
(chấm toạ độ lên ảnh), `probe_smolvla2.py` — đều nằm ở `/data/qualcom-robotic/`.
