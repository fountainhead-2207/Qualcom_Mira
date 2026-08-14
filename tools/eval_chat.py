"""
Scored evaluation for mira_chat_server.py - gesture accuracy plus automatic
persona rule checks.

Reading replies by eye stopped being enough once prompt tweaks started trading
one failure for another: making the persona warmer brought back wrong gestures,
and adding refusal examples made it refuse greetings. This gives a number per
axis so a change can be judged as better or worse rather than just different.

Each prompt carries the gestures that would be defensible for it, so a wrong
pick is scored automatically. The rule checks are deliberately narrow - they
look for failures actually observed in testing rather than trying to judge
quality in general:

  language   non-Vietnamese script or emoji in the reply (it gets read aloud)
  pronoun    "tao"/"mày" - the persona is playful, not rude
  identity   addressing the user as Mira (Mira is the robot)
  ability    claiming to fetch/carry/switch things it has no way to do
  invented   digits in a reply to something it cannot possibly know
  length     over ~35 words, which stops being a short spoken reply

Usage:
    python eval_chat.py [url] [--repeat N]
"""
import argparse
import json
import re
import time
import urllib.request

# (category, prompt, acceptable gestures). "none" is accepted where no gesture
# is clearly better than a wrong one.
CASES = [
    ("greeting", "chào mira", {"wave"}),
    ("greeting", "mira ơi dậy chưa", {"wave", "nod", "curious_tilt"}),
    ("farewell", "thôi mira nhé tôi đi ngủ", {"bow", "wave"}),
    ("farewell", "tạm biệt mira hẹn mai gặp lại", {"bow", "wave"}),
    ("good news", "mira ơi tôi vừa được nhận vào làm intern", {"celebrate", "dance"}),
    ("good news", "tôi vừa thi đậu rồi mira ơi", {"celebrate", "dance"}),
    ("good news", "mira ơi tôi mới được tăng lương", {"celebrate", "dance"}),
    ("unknowable", "mira có biết ngày mai trời mưa không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira biết giá bitcoin bao nhiêu không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira ơi tôi nên chọn công ty nào", {"shrug", "curious_tilt", "nod", "none"}),
    ("question back", "mira biết cái này là gì không", {"curious_tilt", "shrug", "point"}),
    ("question back", "mira thấy cái kia lạ không", {"curious_tilt", "point"}),
    ("about itself", "mira ơi bạn tên gì", {"wave", "bow", "nod", "none"}),
    # shrug belongs here: it answers this by making light of how little it can
    # do ("chỉ quơ quơ được thôi"), which is exactly a shrug.
    ("about itself", "mira bạn làm được gì", {"wave", "dance", "nod", "scan", "shrug", "none"}),
    ("about itself", "mira ơi bạn có buồn không", {"shake", "nod", "curious_tilt"}),
    ("comfort", "mira ơi hôm nay tôi mệt quá", {"nod", "bow", "curious_tilt"}),
    ("comfort", "tôi buồn quá mira à", {"nod", "bow", "curious_tilt"}),
    ("complex", "tôi mới được nhận vào làm intern tôi vui quá bạn có thể nhảy với tôi không mira",
     {"dance", "celebrate"}),
    # wave is fair game when the reply itself offers a wave to cheer you up.
    ("complex", "mira ơi hôm nay trời đẹp mà tôi phải làm việc cả ngày mệt lắm",
     {"nod", "shrug", "curious_tilt", "wave"}),
    ("complex", "mira này nếu tôi cho bạn một cánh tay nữa thì bạn sẽ làm gì",
     {"celebrate", "dance", "curious_tilt", "none"}),
    ("cannot do", "mira lấy giúp tôi cái ly nước", {"shrug"}),
    ("cannot do", "mira ơi bật đèn lên đi", {"shrug"}),
    ("small talk", "mira ơi hôm nay trời đẹp không", {"curious_tilt", "shrug", "nod", "scan"}),
    ("small talk", "mira ăn cơm chưa", {"shake", "curious_tilt", "shrug", "none"}),

    ("greeting", "hê lô mira", {"wave"}),
    ("greeting", "mira ơi mình về rồi đây", {"wave", "celebrate", "bow"}),
    ("farewell", "mira ngủ ngon nha", {"bow", "wave", "nod"}),
    ("good news", "mira ơi dự án của tôi chạy được rồi", {"celebrate", "dance", "nod"}),
    ("good news", "hôm nay tôi được khen trước cả lớp mira ơi", {"celebrate", "dance"}),
    ("unknowable", "mira biết mấy giờ rồi không", {"shrug", "curious_tilt"}),
    ("unknowable", "mira ơi mai tôi có nên đi làm không", {"shrug", "curious_tilt", "nod", "none"}),
    ("unknowable", "mira biết đội nào thắng tối nay không", {"shrug", "curious_tilt"}),
    ("question back", "mira đoán xem tôi đang cầm gì", {"curious_tilt", "shrug", "point"}),
    ("comfort", "mira ơi tôi vừa bị mắng", {"nod", "bow", "curious_tilt", "shrug"}),
    ("comfort", "tôi thấy chán quá mira", {"nod", "dance", "curious_tilt", "celebrate"}),
    ("cannot do", "mira mở cửa giúp tôi với", {"shrug"}),
    ("cannot do", "mira ơi gọi điện cho mẹ tôi đi", {"shrug"}),
    ("cannot do", "mira đi mua cà phê cho tôi nha", {"shrug"}),
    ("complex", "mira ơi tôi vừa cãi nhau với bạn tôi nhưng mà tôi thấy tôi sai rồi",
     {"nod", "bow", "shrug", "curious_tilt"}),
    ("complex", "mira nếu bạn được làm người thì bạn muốn làm gì đầu tiên",
     {"curious_tilt", "dance", "celebrate", "none"}),
    ("about itself", "mira ơi bạn bao nhiêu tuổi rồi", {"shrug", "curious_tilt", "none", "nod"}),
    ("about itself", "mira có thích tôi không", {"nod", "celebrate", "curious_tilt", "dance"}),
    ("small talk", "mira ơi kể chuyện vui đi", {"dance", "celebrate", "curious_tilt", "nod", "none"}),
]

# Prompts where any concrete figure has to be invented, since it has no sensor,
# clock or network to get one from.
UNKNOWABLE = {"unknowable"}
CANNOT_DO = {"cannot do"}

NON_VIETNAMESE = re.compile(
    r"[^\s0-9A-Za-zÀ-ỹ.,!?:;'\"()\-]"      # anything outside latin + Vietnamese + basic punctuation
)
RUDE_PRONOUN = re.compile(r"\b(tao|mày|mầy)\b", re.IGNORECASE)
# "Mira" used as a vocative or with a second-person marker = it thinks the user
# is Mira. "mình tên Mira" / "Mira đây" are correct self-reference, so only the
# addressing patterns count.
USER_CALLED_MIRA = re.compile(r"(bạn\s+mira|mira\s+(à|ơi|nhé|nha|nhá)\b|đó\s+mira|đấy\s+mira)",
                              re.IGNORECASE)
CLAIMS_ABILITY = re.compile(
    r"(sẽ\s+(lấy|mang|bưng|đưa|bật|tắt)|đã\s+(lấy|bật|tắt|mang)|để\s+mình\s+(lấy|bật|mang)"
    r"|mình\s+(lấy|bật|tắt|mang)\s+(giúp|cho))", re.IGNORECASE)
REFUSAL_CUE = re.compile(
    r"(không\s+(thể|làm được|biết|với tới)|chịu\s+(rồi|thôi)|chỉ\s+có\s+(một|1)\s+cánh tay"
    r"|mình\s+chỉ|không\s+có\s+(tay|chân|khả năng))", re.IGNORECASE)


def check_rules(category, reply, motion):
    """Returns a list of rule names the reply broke."""
    broken = []
    if NON_VIETNAMESE.search(reply):
        broken.append("language")
    if RUDE_PRONOUN.search(reply):
        broken.append("pronoun")
    if USER_CALLED_MIRA.search(reply):
        broken.append("identity")
    if CLAIMS_ABILITY.search(reply):
        broken.append("ability")
    if category in UNKNOWABLE and re.search(r"\d", reply):
        broken.append("invented")
    if category in CANNOT_DO and not REFUSAL_CUE.search(reply) and motion != "shrug":
        broken.append("ability")
    if len(reply.split()) > 35:
        broken.append("length")
    return broken


def ask(url, text):
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8766/chat")
    ap.add_argument("--repeat", type=int, default=2,
                    help="runs per prompt; the model samples, so one run per "
                         "prompt measures luck as much as behaviour")
    args = ap.parse_args()

    gesture_hits = 0
    total = 0
    latencies = []
    violations = {}
    by_category = {}
    wrong_gestures = []

    for category, prompt, allowed in CASES:
        for _ in range(args.repeat):
            try:
                result, dt = ask(args.url, prompt)
            except Exception as e:
                print(f"ERROR {prompt!r}: {e}")
                continue
            reply = result["reply_text"]
            motion = result.get("motion") or "none"
            total += 1
            latencies.append(dt)

            ok = motion in allowed
            gesture_hits += ok
            hits, seen = by_category.get(category, (0, 0))
            by_category[category] = (hits + ok, seen + 1)
            if not ok:
                wrong_gestures.append((category, prompt, motion, sorted(allowed), reply))

            for rule in check_rules(category, reply, motion):
                violations[rule] = violations.get(rule, 0) + 1

    print(f"\n{'=' * 68}")
    print(f"gesture accuracy : {gesture_hits}/{total} = {100 * gesture_hits / max(total, 1):.0f}%")
    print(f"latency          : median {sorted(latencies)[len(latencies) // 2]:.1f}s, "
          f"max {max(latencies):.1f}s")
    print(f"rule violations  : {sum(violations.values())} across {total} replies"
          + (f"  -> {violations}" if violations else "  (none)"))

    print("\nper category:")
    for category, (hits, seen) in sorted(by_category.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {category:14} {hits}/{seen}")

    if wrong_gestures:
        print(f"\nwrong gesture picks ({len(wrong_gestures)}):")
        for category, prompt, got, allowed, reply in wrong_gestures[:15]:
            print(f"  [{category}] {prompt}")
            print(f"      got {got}, wanted one of {allowed}")
            print(f"      reply: {reply}")


if __name__ == "__main__":
    main()
