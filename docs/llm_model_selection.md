# Chọn model cho lớp hội thoại của Mira

Ghi lại những gì đã thử và tại sao, để không phải dò lại.

## Yêu cầu

- Nói tiếng Việt tự nhiên, tính cách **playful** (khớp với bộ gesture đã ghi)
- Chọn đúng 1 trong ~12 cử chỉ cho mỗi câu trả lời
- Độ trễ đủ thấp: cử chỉ `thinking` chỉ lấp được ~5s, nên round-trip nên dưới ~2s
- Chạy trên RTX 4090 (24.5GB) **cùng lúc** với MolmoAct2 server (đang chiếm ~12.3GB)
  → ngân sách VRAM thực tế khoảng **12GB**
- `/data` trên máy 4090 thường xuyên gần đầy (94-95%) → cẩn thận với model lớn

## Kết quả đo (tools/eval_chat.py)

| Model | Bộ test | Chính xác cử chỉ | Lỗi quy tắc | Độ trễ |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | - | gần như chọn bừa (định tính) | nhiều | ~0.9s |
| Qwen3-4B-Instruct-2507 | 24 câu | 94% (45/48) | 2/48 | 1.2s |
| Qwen3-4B-Instruct-2507 | 43 câu | 95% (82/86) | 3/86 | 1.2s |
| Qwen3.5-4B | 43 câu | 91% (78/86) | 1/86 | 1.7s |
| Qwen3-4B-Instruct-2507 + routing | 86 câu | 90% (154/172) | 5/172 | 1.2s |
| **cùng trên, sau khi sửa** | **86 câu** | **93%** (160/172) | **1/172** | **1.1s** |

Hai dòng cuối cho thấy việc thêm nhánh routing sang MolmoAct2 ban đầu làm tụt độ
chính xác (95%→90%) và tăng lỗi quy tắc, vì model trở nên quá "nhiệt tình" - nói
như đã nhặt xong, và sinh cả task không phải gắp-đặt. Thêm quy tắc phải nói ở thể
sắp làm, phải thừa nhận không nhìn thấy gì, cộng với việc server tự chặn task sai
dạng, đưa lên 93% và lỗi quy tắc xuống còn 1/172.

Đáng chú ý: cả 4 lần "sai" ở bộ 43 câu đều là cử chỉ **khớp với chính câu trả
lời model vừa viết** - ví dụ *"Sẽ quơ quơ vui hơn, nhưng vẫn không thể cầm đồ
được!"* + `shrug`, hay *"Mình có thể quơ quơ một cái để làm bạn vui lên đấy!"* +
`wave`. Nghĩa là model tự nhất quán, chỉ là bộ nhãn mong đợi của harness hẹp hơn
mức sáng tạo của nó. Con số thực tế vì vậy còn cao hơn 95%.

Qwen2.5-3B không đo bằng harness (harness viết sau khi đã đổi model), nhưng các
lỗi định tính rất rõ: gọi người dùng là "Mira", dùng "tao/mày", chèn tiếng Trung
giữa câu, **bịa giá bitcoin**, nhận làm những việc nó không làm được ("để mình
lấy nước cho bạn"), và chọn `shake` (lắc đầu = phủ định) cho cả lời tạm biệt lẫn
tin tăng lương.

Riêng việc chọn cử chỉ, Qwen2.5-3B yếu tới mức đã phải viết một bước phân loại
riêng (`choose_gesture`, greedy decoding) để tách "soạn câu trả lời" khỏi "phân
loại cử chỉ". Với Qwen3-4B-Instruct-2507 thì **không cần** - nó làm cả hai trong
một lượt ở mức 94%.

## Những hướng đã thử và bỏ

**FP8 (Qwen3-8B-FP8, 14B-FP8, 4B-2507-FP8 — đã có sẵn trong `/data/KV_cache/LMbench`)**
Không load được bằng transformers thuần: cần Triton kernel `w8a8_fp8_matmul` lấy
qua gói `kernels`. Cài `kernels` (0.16.0) **làm hỏng transformers hoàn toàn** -
`transformers/integrations/hub_kernels.py` gọi `LayerRepository(...)` không có
`revision`, mà kernels mới bắt buộc phải có → `ValueError` ngay khi import
`transformers.models.qwen3`. Đã gỡ để phục hồi. Các file FP8 đó vốn dành cho
vLLM/SGLang, không phải transformers.

**AWQ 4-bit (Qwen/Qwen3-8B-AWQ, Qwen3-14B-AWQ — chính thức, hàng triệu lượt tải)**
Vừa VRAM hơn hẳn (8B AWQ ~6GB) nhưng `quantizer_awq.py` trong transformers 5.5.4
đòi gói `gptqmodel`. Đây lại là một gói biên dịch CUDA kernel, cài vào đúng cái
venv đang chạy MolmoAct2 server → rủi ro lặp lại sự cố `kernels`. Không đáng đổi.

**Model thuần Việt** (PhoGPT-4B-Chat của VinAI, Vistral-7B-Chat, SeaLLMs-v3-7B,
Sailor2-8B) — đều dựa trên kiến trúc 2024 và lượt tải chỉ 669-997, so với
Qwen3.5-4B là 7.5 triệu. Chúng ra đời khi model đa ngữ chung còn yếu tiếng Việt;
khoảng cách đó nay đã gần như bị xóa. Chưa test trực tiếp, nhưng thứ tự ưu tiên
là model đời mới nhất vừa VRAM, không phải model chuyên ngữ đời cũ.

**Fine-tune** - không có dataset công khai nào khớp (playful + tiếng Việt + 12
nhãn cử chỉ + JSON). Gần nhất là `bkai-foundation-models/vietnamese-roleplay-realm`
nhưng nó không dạy nhãn cử chỉ. Muốn tune thì phải tự sinh dataset (dùng model
mạnh sinh ra các cặp `câu người dùng → câu trả lời playful + cử chỉ`, rồi LoRA).
Ở mức 94% thì chưa cần.

## Lượng tử hóa INT4 - đo trước khi tin

**Đã đo, và INT4 làm CHẬM hơn** (`tools/bench_int4.py`, Qwen3-4B-Instruct-2507
trên chính máy 4090 này):

| | median | tok/s | VRAM |
|---|---|---|---|
| bf16 | **0.83s** | **29.3** | 8.0 GB |
| bitsandbytes NF4 | 1.12s | 20.8 | **2.7 GB** |

NF4 **chậm hơn 35%**, đổi lại tiết kiệm 5.4GB VRAM. Đúng như lý thuyết: ở batch 1
trên GPU rời, trọng số nhỏ đi giúp đỡ VRAM nhưng mỗi token phải giải nén thêm nên
độ trễ tăng. INT4 chỉ thắng khi thật sự bị giới hạn băng thông bộ nhớ - model 4B
trên 4090 thì không phải vậy.

Kết luận: **không lượng tử hóa model 4B này**. VRAM đang đủ (20.9/24.5GB kể cả
MolmoAct2), mà độ trễ thì quan trọng - cử chỉ `thinking` chỉ lấp được vài giây.
Lượng tử hóa chỉ đáng xét khi muốn nhét model **lớn hơn** vào cùng ngân sách.

`bitsandbytes` cài được an toàn vào `molmoact2-env` (wheel có sẵn binary, không
biên dịch, đã xác nhận transformers vẫn import bình thường sau khi cài) - khác
hẳn `kernels` và `gptqmodel`.

**Nếu sau này thật sự cần lượng tử hóa** (để nhét 8B/14B vào ngân sách VRAM):
đừng dùng bitsandbytes NF4. Theo tổng hợp 2026, **AWQ và AutoRound dẫn đầu về độ
chính xác ở 4-bit**; AWQ bảo vệ khoảng 1% trọng số quan trọng nhất dựa trên độ
lớn activation nên giữ được gần như nguyên độ chính xác, và đã thành mặc định cho
serving GPU. SmoothQuant vốn nhắm W8A8 nên không phải lựa chọn cho W4.

Quan trọng hơn cả định dạng: **kernel quyết định tốc độ**. Cùng một model AWQ,
kernel thường cho ~68 tok/s còn **Marlin cho ~741 tok/s** trên cùng GPU. Đây mới
là lý do NF4 trong transformers chậm - không phải vì 4-bit. Nên nếu mục tiêu là
tốc độ thì đường đúng là **AWQ + Marlin qua vLLM**, không phải bnb trong
transformers.

`transformers` 5.5.4 ở đây có sẵn `quantizer_auto_round.py`, `quantizer_hqq.py`,
`quantizer_higgs.py`, `quantizer_sinq.py` - tức là AutoRound và vài phương pháp
mới đều dùng được, chỉ cần gói tương ứng.

## Hướng nâng cấp nếu cần

- **Qwen3.5-4B** (9.3GB, kiến trúc `qwen3_5` - transformers 5.5.4 đã hỗ trợ) là
  bản cùng cỡ đời mới hơn. Lưu ý: **mặc định bật thinking mode**, phải
  `enable_thinking=False` khi apply chat template, và nó là
  `Qwen3_5ForConditionalGeneration` có `image_token_id` nên có thể cần class khác
  `AutoModelForCausalLM`. Disk chỉ đủ cho một model tại một thời điểm.
- **Qwen3.5-9B / 8B AWQ** nếu muốn mạnh hơn: cách sạch nhất là **vLLM trong venv
  riêng** phục vụ chính các file FP8 đã có trên đĩa (không phải tải gì), rồi
  `mira_chat_server.py` gọi qua HTTP. Nhớ giới hạn `gpu_memory_utilization` để
  không tranh VRAM với MolmoAct2.

## Cách đo

```bash
scp tools/eval_chat.py 61.28.228.23:/data/qualcom-robotic/
ssh 61.28.228.23 "cd /data/qualcom-robotic && molmoact2-env/bin/python eval_chat.py --repeat 2"
```

Đừng đánh giá bằng cách đọc vài câu trả lời - trong lúc tinh chỉnh prompt, mỗi
lần sửa lại đánh đổi lỗi này bằng lỗi khác (làm giọng ấm hơn thì cử chỉ sai lại;
thêm ví dụ từ chối thì nó từ chối cả lời chào). Harness cho con số từng trục nên
mới biết là tốt lên hay xấu đi.
