"""
Đo xem lượng tử hóa INT4 (bitsandbytes NF4) có đáng đổi hay không.

Giả thiết thường gặp là "INT4 thì nhanh hơn". Với batch 1 trên GPU rời thì
thường KHÔNG: trọng số nhỏ đi (đỡ VRAM) nhưng mỗi token phải giải nén thêm, nên
độ trễ hay tăng. Chỉ khi bị giới hạn băng thông bộ nhớ thật sự thì INT4 mới
thắng. Script này đo cả hai chế độ trên đúng phần cứng này thay vì đoán.

In ra: VRAM, thời gian nạp, độ trễ sinh (median trên nhiều câu), và câu trả lời
để kiểm tra chất lượng không bị tụt.

Usage:
    python bench_int4.py            # so sánh bf16 vs nf4
    python bench_int4.py nf4        # chỉ đo nf4
"""
import gc
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HOME", "/data/qualcom-robotic/hf-cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPTS = [
    "chào mira",
    "mira ơi tôi vừa thi đậu rồi",
    "mira ơi bật đèn lên đi",
    "tôi đánh rơi cái tua vít không nhặt được bạn nhặt giúp mình nhé",
    "mira ơi hôm nay tôi đi làm về mệt quá mà nhà thì bừa bộn tôi chẳng muốn dọn gì cả",
]
SYSTEM = (
    "Bạn là Mira - cánh tay robot nhỏ, tinh nghịch. Trả lời 1-2 câu tiếng Việt, "
    "rồi chọn một cử chỉ. CHỈ trả JSON: {\"reply\": \"...\", \"motion\": \"...\"}. "
    "Cử chỉ: wave, bow, celebrate, dance, shrug, curious_tilt, point, nod, shake, "
    "scan, play-dead, none"
)


def bench(mode):
    print(f"\n{'=' * 60}\n{mode.upper()}\n{'=' * 60}", flush=True)
    kwargs = {"device_map": "cuda"}
    if mode == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    model.eval()
    load_s = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"nạp {load_s:.0f}s, VRAM {vram:.1f} GB", flush=True)

    latencies, tok_counts = [], []
    for prompt in PROMPTS:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        # bỏ lần đầu: còn tốn thời gian khởi tạo kernel/cache
        for run in range(2):
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            dt = time.perf_counter() - t1
        n = out.shape[1] - inputs["input_ids"].shape[1]
        latencies.append(dt)
        tok_counts.append(n)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {dt:5.2f}s  {n:3d} tok  {reply.strip()[:90]}", flush=True)

    med = statistics.median(latencies)
    tok_per_s = sum(tok_counts) / sum(latencies)
    print(f"\n  median {med:.2f}s, {tok_per_s:.1f} tok/s, VRAM {vram:.1f} GB", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"mode": mode, "median_s": med, "tok_per_s": tok_per_s,
            "vram_gb": vram, "load_s": load_s}


modes = [sys.argv[1]] if len(sys.argv) > 1 else ["bf16", "nf4"]
results = [bench(m) for m in modes]

print(f"\n{'=' * 60}\nKẾT QUẢ\n{'=' * 60}")
print(f"{'mode':6} {'median':>9} {'tok/s':>8} {'VRAM':>8} {'nạp':>7}")
for r in results:
    print(f"{r['mode']:6} {r['median_s']:8.2f}s {r['tok_per_s']:8.1f} "
          f"{r['vram_gb']:7.1f}G {r['load_s']:6.0f}s")
if len(results) == 2:
    bf16, nf4 = results
    faster = "nhanh hơn" if nf4["median_s"] < bf16["median_s"] else "CHẬM hơn"
    pct = abs(nf4["median_s"] - bf16["median_s"]) / bf16["median_s"] * 100
    print(f"\nNF4 {faster} {pct:.0f}%, tiết kiệm "
          f"{bf16['vram_gb'] - nf4['vram_gb']:.1f} GB VRAM")
