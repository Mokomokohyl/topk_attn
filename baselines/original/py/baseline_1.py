#!/usr/bin/env python3
"""PyTorch baseline pipeline matching baseline_1.cu.

Steps:
1. Generate synthetic data (centers, kv-cache, query, value).
2. Coarse selection: per-head top-5 clusters via dot products with centers.
3. Fine selection: per selected cluster top-512 tokens -> 2560 indices per head.
4. Attention decode: compute attention output over the selected key/value tokens.
5. Annotate the single execution with NVTX ranges for nsys profiling.
"""
from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from typing import Iterator, Tuple

import torch
from torch import Tensor

# Problem constants (keep in sync with baseline_1.cu)
HD = 128
HN = 32
CSZ = 20
CLEN = 2048
TOPC = 5
TOPK_PER_CLUSTER = 512
OUT_PER_HEAD = TOPC * TOPK_PER_CLUSTER
FSL = CSZ * CLEN


@contextmanager
def nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    """Context manager that pushes/pops an NVTX range when enabled."""
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def generate_data(device: torch.device, dtype: torch.dtype = torch.float16) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate centers, kv-cache, value-cache, and query vectors.

    Returns:
        q: (HN, HD)
        k: (HN, CSZ, CLEN, HD)
        v: (HN, CSZ, CLEN, HD)
        centers: (HN, CSZ, HD)
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(12345)

    centers = (torch.empty(HN, CSZ, HD, device=device, dtype=dtype)
               .uniform_(-0.5, 0.5, generator=rng))

    noise = torch.empty(HN, CSZ, CLEN, HD, device=device, dtype=dtype)
    noise.normal_(mean=0.0, std=0.05, generator=rng)

    k = centers.unsqueeze(2) + noise  # broadcast over CLEN
    v = k.clone()  # reuse keys as values for simplicity

    q = (torch.empty(HN, HD, device=device, dtype=dtype)
         .uniform_(-0.5, 0.5, generator=rng))

    return q, k, v, centers


def coarse_topk(q: Tensor, centers: Tensor) -> Tensor:
    """Select top-5 clusters per head based on dot(q, center)."""
    q_f = q.to(torch.float32)
    centers_f = centers.to(torch.float32)
    # center_scores: (HN, CSZ)
    center_scores = torch.einsum("hcd,hd->hc", centers_f, q_f)
    top_scores, top_idx = torch.topk(center_scores, TOPC, dim=1, largest=True, sorted=True)
    return top_idx.to(torch.long)


def fine_topk(q: Tensor, k_tokens: Tensor, top_clusters: Tensor) -> Tensor:
    """Per selected cluster, take top-512 tokens -> 2560 indices per head."""
    q_f = q.to(torch.float32)
    k_f = k_tokens.to(torch.float32)
    # Gather selected clusters: shapes -> (HN, TOPC, CLEN, HD)
    gather_idx = top_clusters.view(HN, TOPC, 1, 1).expand(-1, -1, CLEN, HD)
    selected_k = torch.gather(k_f, 1, gather_idx)

    token_scores = (selected_k * q_f.view(HN, 1, 1, HD)).sum(-1)
    token_vals, token_idx = torch.topk(token_scores, TOPK_PER_CLUSTER, dim=-1, largest=True, sorted=True)

    # Convert to global indices (relative to flattened per-head tokens)
    global_idx = top_clusters.view(HN, TOPC, 1) * CLEN + token_idx
    return global_idx.view(HN, OUT_PER_HEAD)


def compute_attention(q: Tensor, k_tokens: Tensor, v_tokens: Tensor, kv_indices: Tensor) -> Tensor:
    """Compute single-query attention outputs using selected indices."""
    q_f = q.to(torch.float32)
    k_flat = k_tokens.reshape(HN, FSL, HD).to(torch.float32)
    v_flat = v_tokens.reshape(HN, FSL, HD).to(torch.float32)

    gather_idx = kv_indices.view(HN, OUT_PER_HEAD, 1).expand(-1, -1, HD)
    selected_k = torch.gather(k_flat, 1, gather_idx)
    selected_v = torch.gather(v_flat, 1, gather_idx)

    logits = (selected_k * q_f.view(HN, 1, HD)).sum(-1) / math.sqrt(HD)
    attn_weights = torch.softmax(logits, dim=-1)
    output = torch.sum(attn_weights.unsqueeze(-1) * selected_v, dim=1)
    return output.to(q.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch baseline matching baseline_1.cu")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda", help="Execution device")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Base tensor dtype")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(2025)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2025)

    nvtx_enabled = device.type == "cuda"

    with nvtx_range("generate_data", nvtx_enabled):
        q, k_tokens, v_tokens, centers = generate_data(device, dtype=dtype)

    with nvtx_range("coarse_topk", nvtx_enabled):
        top_clusters = coarse_topk(q, centers)

    with nvtx_range("fine_topk", nvtx_enabled):
        kv_indices = fine_topk(q, k_tokens, top_clusters)

    with nvtx_range("attention", nvtx_enabled):
        output = compute_attention(q, k_tokens, v_tokens, kv_indices)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Sample head0 indices:", kv_indices[0, :16].tolist())
    print("Attention output[0, :8]:", output[0, :8].tolist())


if __name__ == "__main__":
    main()
