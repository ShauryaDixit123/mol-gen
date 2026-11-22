"""Training utilities shared by the Kaggle notebooks.

The Kaggle-specific pieces matter as much as the model code: a session is
killed at a hard wall clock limit, so a run that only saves at the end loses
everything.  :class:`TimeBudget` plus checkpoint-every-epoch makes a run
resumable across sessions, which is how you actually train a diffusion model
on a free GPU quota.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def device_info() -> dict:
    info = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


class EMA:
    """Exponential moving average of weights.

    Diffusion sample quality is notoriously sensitive to this: the EMA weights
    routinely generate far cleaner geometry than the raw ones at the same step,
    so sampling always uses the EMA copy.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap EMA weights in, keeping the live ones for :meth:`restore`."""
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(
            {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
             for k, v in model.state_dict().items()},
            strict=False,
        )

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


def warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_factor: float = 0.05):
    """LR schedule: linear warm-up then cosine decay to ``min_factor``."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


@dataclass
class TimeBudget:
    """Stop training before the Kaggle session is killed.

    Kaggle gives 9 h (CPU/GPU) or 12 h (TPU) per session and kills the kernel
    without running the rest of the notebook, so the default leaves 25 min of
    headroom for the final checkpoint, sampling demo and dataset commit.
    """

    limit_s: float = 8.5 * 3600
    reserve_s: float = 1500.0
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float:
        return self.limit_s - self.reserve_s - self.elapsed

    def expired(self) -> bool:
        return self.remaining <= 0

    def __str__(self) -> str:
        return f"elapsed {self.elapsed / 60:.1f} min, remaining {max(0.0, self.remaining) / 60:.1f} min"


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    ema: EMA | None = None,
    epoch: int = 0,
    step: int = 0,
    config: dict | None = None,
    history: list | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config or {},
        "history": history or [],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if extra:
        payload["extra"] = extra
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a half-written checkpoint is worse than none


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer=None,
    scheduler=None,
    ema: EMA | None = None,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


class History:
    """Append-only metric log that survives a kernel restart."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.rows: list[dict] = []
        if self.path and self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def log(self, **kwargs) -> dict:
        row = {"wall": round(time.time(), 2), **kwargs}
        self.rows.append(row)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def plot(self, keys: list[str], ax=None, logy: bool = True):
        import matplotlib.pyplot as plt

        df = self.to_frame()
        ax = ax or plt.gca()
        for k in keys:
            if k in df.columns:
                sub = df[["step", k]].dropna() if "step" in df.columns else df[[k]].dropna()
                ax.plot(sub["step"] if "step" in sub else range(len(sub)), sub[k], label=k)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return math.sqrt(total)
