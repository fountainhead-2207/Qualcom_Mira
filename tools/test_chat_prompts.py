"""
Batch-test what mira_chat_server.py replies, and which gesture it picks, across
a spread of things the operator would realistically say.

Checks two things that are hard to judge from one-off prompts: whether the
persona actually comes through consistently, and whether gesture selection is
sensible rather than defaulting to "wave" for everything. Audio is requested as
usual but discarded - this is for reading, not listening.

Usage:
    python test_chat_prompts.py [url] [--repeat N]
"""
import argparse
import json
import time
import urllib.request

# Grouped by the gesture that would be the obvious right answer, so a wrong
# pick is easy to spot. Wording is deliberately colloquial and a bit sloppy,
# the way it arrives from the ASR rather than as clean written Vietnamese.
PROMPTS = [
    ("greeting", "chào mira"),
    ("greeting", "mira ơi dậy chưa"),
    ("farewell", "thôi mira nhé tôi đi ngủ"),
    ("farewell", "tạm biệt mira hẹn mai gặp lại"),
    ("good news", "mira ơi tôi vừa được nhận vào làm intern"),
    ("good news", "tôi vừa thi đậu rồi mira ơi"),
    ("good news", "mira ơi tôi mới được tăng lương"),
    ("uncertainty", "mira có biết ngày mai trời mưa không"),
    ("uncertainty", "mira ơi tôi nên chọn công ty nào"),
    ("uncertainty", "mira biết giá bitcoin bao nhiêu không"),
    ("question back", "mira biết cái này là gì không"),
    ("question back", "mira thấy cái kia lạ không"),
    ("about itself", "mira ơi bạn tên gì"),
    ("about itself", "mira bạn làm được gì"),
    ("about itself", "mira ơi bạn có buồn không"),
    ("emotional", "mira ơi hôm nay tôi mệt quá"),
    ("emotional", "tôi buồn quá mira à"),
    ("complex", "tôi mới được nhận vào làm intern tôi vui quá bạn có thể nhảy với tôi không mira"),
    ("complex", "mira ơi hôm nay trời đẹp mà tôi phải làm việc cả ngày mệt lắm"),
    ("complex", "mira này nếu tôi cho bạn một cánh tay nữa thì bạn sẽ làm gì"),
    ("cannot do", "mira lấy giúp tôi cái ly nước"),
    ("cannot do", "mira ơi bật đèn lên đi"),
    ("small talk", "mira ơi hôm nay trời đẹp không"),
    ("small talk", "mira ăn cơm chưa"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8766/chat")
    ap.add_argument("--repeat", type=int, default=1,
                    help="ask each prompt N times - the model samples, so one "
                         "answer per prompt doesn't show how stable it is")
    args = ap.parse_args()

    counts = {}
    for label, text in PROMPTS:
        print(f"\n[{label}] {text}")
        for _ in range(args.repeat):
            body = json.dumps({"text": text}).encode("utf-8")
            req = urllib.request.Request(args.url, data=body,
                                        headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=40) as resp:
                    result = json.loads(resp.read())
            except Exception as e:
                print(f"    ERROR {e}")
                continue
            dt = time.perf_counter() - t0
            motion = result.get("motion") or "-"
            counts[motion] = counts.get(motion, 0) + 1
            print(f"    -> {result['reply_text']}")
            print(f"       [{motion}]  {dt:.1f}s")

    print("\n" + "=" * 60)
    print("gesture distribution:")
    for motion, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {motion:15} {n}")


if __name__ == "__main__":
    main()
