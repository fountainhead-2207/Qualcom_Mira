# Proposed new gestures for conversational replies

Existing canned motions: `wave`, `dance`, `nod` (yes), `shake` (no), `play-dead`,
`clean`, `scan`. These new ones are picked specifically to pair with LLM-generated
conversational replies, not to duplicate what's already covered.

| Name | What it looks like | When the LLM would pick it |
|---|---|---|
| `thinking` | Slow, small idle movement - e.g. tilt/settle at a middle pose, no big travel | Filler while the LLM is still generating a reply, so the arm isn't just frozen during the ~1-2s round trip |
| `shrug` | Both "shoulders" (shoulder_lift) rise briefly then settle | Reply expresses uncertainty ("tôi không chắc", "không biết nữa") |
| `point` | Extend and aim the wrist/gripper toward camera or a fixed direction | Reply references something specific ("cái đó kìa", pointing out an object once vision is wired in) |
| `bow` | Lean forward at the base, pause, return | Greeting or farewell in a more formal register than `wave` |
| `celebrate` | Quick double raise + small wiggle, more energetic than `dance` | Reply is enthusiastic/positive ("tuyệt vời!", congratulating the user) |
| `curious_tilt` | Slow head-equivalent (wrist) tilt side to side, holds a beat | Reply is a question back to the user, or reacting to something surprising |

Suggested minimum for a first pass: `thinking` and `shrug` are the two that matter
most functionally - `thinking` covers the LLM's generation latency so the arm
doesn't look inert while "listening", and `shrug` is the natural fallback body
language whenever the LLM itself is uncertain. The rest are nice-to-have polish.

Recording process (once you're back): same as the existing motions - teleop the
leader arm through the gesture with LeLab, save as a LeRobot dataset episode
under `hf_lerobot/local/motion_<name>/`, matching the existing bundled motions'
structure so `mira-robot replay <name>` picks it up with no code changes.
