#!/usr/bin/env python3
"""Patch See-through's inference_utils to use diffusers sequential CPU offload on 4 GB VRAM GPUs.

Problem
-------
RTX 2050 has 4 GB VRAM. LayerDiff + Marigold need ~8-10 GB VRAM total. The upstream code calls
`enable_group_offload('cuda', num_blocks_per_group=1)` which is a See-through custom offload that
expects weights to already be on CPU — but then `.to('cuda')` is called before it, causing OOM.
Even if the order were fixed, group_offload moves blocks CPU↔GPU but leaves VAE on CUDA, causing
dtype/device mismatches.

Solution
--------
Replace the entire group_offload block with diffusers' built-in
`enable_sequential_cpu_offload()` — the standard, well-tested API for running SDXL-class models
on 4-6 GB VRAM. It:
  1. Hooks every sub-module's forward() to move it to GPU just-in-time
  2. Moves it back to CPU after each forward pass
  3. Works correctly with VAE/trans_vae on CUDA because it handles device routing internally
  4. Is faster than group_offload for this use case (no manual block slicing)

The `.to(dtype=..., device='cuda')` calls are kept as-is so everything loads onto GPU first,
then sequential_cpu_offload re-registers the hooks.

Design notes
------------
- Replaces the `if group_offload: enable_group_offload(...)` block in apply_layerdiff AND apply_marigold
- Also replaces the `marigold_pipeline.to(device='cuda', ...)` with `to(dtype=...)` only (device
  will be managed by sequential_cpu_offload)
- Idempotent: second run is a no-op
- Non-fatal: if anchors not found, exits non-zero without touching the file
- Self-verifying: ast.parse before writing

Usage: python low_vram_cpu_load.py /opt/see-through/common/utils/inference_utils.py
"""
from __future__ import annotations

import ast
import sys

MARKER = "image2live2d: sequential-cpu-offload"

# Exact stripped lines to replace
REPLACEMENTS = {
    # LayerDiff: replace group_offload block
    "layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)":
        f"layerdiff_pipeline.enable_sequential_cpu_offload()  # {MARKER}",
    # Marigold: replace group_offload block
    "marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)":
        f"marigold_pipeline.enable_sequential_cpu_offload()  # {MARKER}",
    # Marigold: .to(device='cuda', dtype=bfloat16) -> .to(dtype=bfloat16) only, device managed by offload
    "marigold_pipeline.to(device='cuda', dtype=torch.bfloat16)":
        f"marigold_pipeline.to(dtype=torch.bfloat16)  # {MARKER}: device managed by sequential_cpu_offload",
}


def patch_text(text: str) -> tuple[str, str]:
    if MARKER in text:
        return text, "already-patched"

    lines = text.splitlines(keepends=True)
    new_lines = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        if stripped in REPLACEMENTS:
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(indent + REPLACEMENTS[stripped] + "\n")
            changed += 1
        else:
            new_lines.append(line)

    if changed == 0:
        return text, "anchor-not-found"

    new_text = "".join(new_lines)
    ast.parse(new_text)
    return new_text, f"patched ({changed} replacements)"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out, status = patch_text(text)
    if status.startswith("patched"):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"low_vram_cpu_load: {status} -> {path}")
        return 0
    if status == "already-patched":
        print(f"low_vram_cpu_load: {path} already patched (no-op)")
        return 0
    print(f"low_vram_cpu_load: {status} in {path} — leaving unpatched", file=sys.stderr)
    return 3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: low_vram_cpu_load.py <inference_utils.py>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
