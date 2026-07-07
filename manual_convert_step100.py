"""Manual DCP->HF for step-100 (torchtitan Qwen3.5-9B GDN). Same approach as
manual_convert_step40.py: load the DCP, apply the tt->hf state-dict adapter, and
save with safetensors.save_file directly (bypasses the buggy HuggingFaceStorageWriter
consolidation that truncated shard headers)."""
import importlib, os, torch
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint import ModelWrapper
from safetensors.torch import save_file

INPUT = "/home/yichuan/ckpts/q35_9b_tmax_step100/step-100"
OUT_DIR = "/home/yichuan/ckpt_step100_hf"
OUT = os.path.join(OUT_DIR, "model.safetensors")
HF_ASSETS = "/home/yichuan/.cache/huggingface/hub/models--hamishivi--Qwen3.5-9B/snapshots/be36edae3fd57c6cd66556fbce88b6c896ec3f0a"

os.makedirs(OUT_DIR, exist_ok=True)

mm = importlib.import_module("torchtitan.models.qwen3_5")
spec = mm.model_registry("9B")
cfg = spec.model
with torch.device("cpu"):
    model = cfg.build()
model = ModelWrapper(model)
adapter = spec.state_dict_adapter(cfg, HF_ASSETS)

sd = model._get_state_dict()
print("loading DCP...", flush=True)
dcp.load(sd, checkpoint_id=INPUT)
print("mapping tt->hf...", flush=True)
hf = adapter.to_hf(sd)

# bf16 + contiguous + clone (break any shared storage) + tensors only
clean = {}
for k, v in hf.items():
    if isinstance(v, torch.Tensor):
        clean[k] = v.detach().to(torch.bfloat16).contiguous().clone()
print(f"saving {len(clean)} tensors to {OUT} ...", flush=True)
save_file(clean, OUT, metadata={"format": "pt"})
print("SAVED_OK", len(clean), flush=True)
