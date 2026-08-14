import time
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

CKPT = "/data/qualcom-robotic/mira_molmoact2_step2000"

print("Loading policy config + pre/post processors...", flush=True)
cfg = PreTrainedConfig.from_pretrained(CKPT)
preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT)

print("Loading policy (weights)...", flush=True)
t0 = time.perf_counter()
policy = MolmoAct2Policy.from_pretrained(CKPT)
policy.to("cuda")
policy.eval()
print(f"Loaded in {time.perf_counter() - t0:.1f}s", flush=True)
print("VRAM after load (MiB):", torch.cuda.memory_allocated() / 1024**2, flush=True)

batch = {
    "observation.images.camera1": torch.rand(1, 3, 240, 320),
    "observation.images.camera2": torch.rand(1, 3, 240, 320),
    "observation.state": torch.zeros(1, 6),
    "task": ["pick up the screwdriver and put it on the black workspace"],
}

print("Running forward pass (synthetic input, plumbing test only)...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    processed = preprocessor(batch)
    action = policy.predict_action_chunk(processed, inference_action_mode="continuous")
    action = postprocessor(action)
elapsed = time.perf_counter() - t0
print(f"Inference latency: {elapsed*1000:.1f} ms", flush=True)
print("VRAM peak (MiB):", torch.cuda.max_memory_allocated() / 1024**2, flush=True)
print("Action type:", type(action), flush=True)
if torch.is_tensor(action):
    print("Action shape:", action.shape, flush=True)
    print("Action has NaN:", torch.isnan(action).any().item(), flush=True)
    print("Action has Inf:", torch.isinf(action).any().item(), flush=True)
    print("Action values (first row):", action.flatten()[:6].tolist(), flush=True)
else:
    print("Action:", action, flush=True)
