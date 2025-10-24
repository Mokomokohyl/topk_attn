#!/usr/bin/env python3
"""Emit NVTX ranges around several torch.topk calls with varying problem sizes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.cuda.nvtx as nvtx


@dataclass(frozen=True)
class TopKConfig:
    numel: int
    k: int

    @staticmethod
    def parse(spec: str) -> "TopKConfig":
        try:
            n_str, k_str = spec.split(":", maxsplit=1)
            numel = int(n_str)
            k = int(k_str)
        except ValueError as exc:  # noqa: PERF203 - clarity
            raise argparse.ArgumentTypeError(
                f"Invalid config '{spec}'. Expected the form <numel>:<k>."
            ) from exc
        if numel <= 0 or k <= 0:
            raise argparse.ArgumentTypeError("Both numel and k must be positive.")
        if k > numel:
            raise argparse.ArgumentTypeError("k must not exceed numel.")
        return TopKConfig(numel=numel, k=k)


DEFAULT_CONFIGS: Tuple[TopKConfig, ...] = (
    TopKConfig(1024, 32),
    TopKConfig(4096, 64),
    TopKConfig(65536, 128),
    TopKConfig(262144, 512),
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run torch.topk on several input sizes with NVTX markers to inspect kernel selection."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda). Must be a CUDA device to emit NVTX ranges.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Input dtype for the generated tensors.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to repeat each topk call within its NVTX range.",
    )
    parser.add_argument(
        "--configs",
        type=TopKConfig.parse,
        nargs="*",
        default=list(DEFAULT_CONFIGS),
        help="Problem sizes as <numel>:<k>. Defaults cover four representative pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the first value/index from each run for quick sanity checking.",
    )
    return parser.parse_args(argv)


def to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def ensure_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            f"NVTX ranges require a CUDA device, but got '{device}'. Use --device=cuda or cuda:<index>."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ensure torch with CUDA support is installed.")


def run_topk(configs: Iterable[TopKConfig], repeats: int, device: torch.device, dtype: torch.dtype, print_summary: bool) -> None:
    gen = torch.Generator(device=device)
    for cfg in configs:
        label = f"topk_N{cfg.numel}_K{cfg.k}"
        data = torch.randn(cfg.numel, generator=gen, device=device, dtype=dtype)

        nvtx.range_push(label)
        for rep in range(repeats):
            iter_label = f"{label}_iter{rep}"
            nvtx.range_push(iter_label)
            torch.cuda.synchronize()
            values, indices = torch.topk(data, cfg.k, dim=0, largest=True, sorted=True)
            torch.cuda.synchronize()
            nvtx.range_pop()

        nvtx.range_pop()

        if print_summary:
            print(
                f"{label}: value[0]={values[0].item():.4f}, index[0]={int(indices[0])}",
                flush=True,
            )


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)

    device = torch.device(args.device)
    ensure_cuda_device(device)
    dtype = to_dtype(args.dtype)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    configs: List[TopKConfig] = list(args.configs)
    if not configs:
        raise ValueError("At least one <numel>:<k> configuration must be provided.")

    torch.cuda.synchronize()
    run_topk(configs, args.repeats, device, dtype, args.print_summary)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main(sys.argv[1:])
