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

from eval_cases import CASES, UNKNOWABLE, CANNOT_DO, MANIPULATION  # noqa: E402

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
    task_expected = 0
    task_emitted = 0
    task_leaked = 0      # a task on a prompt that needs no manipulation
    task_examples = []

    for category, prompt, allowed in CASES:
        for _ in range(args.repeat):
            try:
                result, dt = ask(args.url, prompt)
            except Exception as e:
                print(f"ERROR {prompt!r}: {e}")
                continue
            reply = result["reply_text"]
            motion = result.get("motion") or "none"
            task = result.get("task")
            total += 1
            latencies.append(dt)

            if category == "manipulation":
                task_expected += 1
                if task:
                    task_emitted += 1
                    task_examples.append((prompt, reply, task))
            elif task:
                task_leaked += 1
                task_examples.append((prompt + "  [LEAKED]", reply, task))

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

    if task_expected:
        print(f"MolmoAct2 routing : {task_emitted}/{task_expected} manipulation prompts "
              f"produced a task"
              + (f", {task_leaked} leaked onto non-manipulation prompts" if task_leaked
                 else ", none leaked elsewhere"))

    print("\nper category:")
    for category, (hits, seen) in sorted(by_category.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {category:14} {hits}/{seen}")

    if task_examples:
        print("\ntasks routed to MolmoAct2:")
        for prompt, reply, task in task_examples[:10]:
            print(f"  {prompt}")
            print(f"      reply: {reply}")
            print(f"      task:  {task}")

    if wrong_gestures:
        print(f"\nwrong gesture picks ({len(wrong_gestures)}):")
        for category, prompt, got, allowed, reply in wrong_gestures[:15]:
            print(f"  [{category}] {prompt}")
            print(f"      got {got}, wanted one of {allowed}")
            print(f"      reply: {reply}")


if __name__ == "__main__":
    main()
