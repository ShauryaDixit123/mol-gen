"""Shard-based graph dataset: build once on Kaggle, reuse as a Kaggle dataset.

Preprocessing 150 k complexes takes ~1-2 h of CPU.  Doing that inside the
training notebook would burn most of the 9/12 h session budget on every run, so
notebook 01 writes ``.pt`` shards plus a ``manifest.json`` and saves them as its
output; notebooks 02-04 attach that output as an input dataset and read it
lazily.

One shard = a list of *encoded* complexes (:func:`molgen.graphs.encode_complex`
dicts of small numpy arrays, ~12 kB each), not featurised graphs.  Storing the
featurised form would cost ~350 kB per complex — 60 GB for the full BioLiP set,
against Kaggle's 20 GB output limit — because the dense ``(E, 17)`` pocket edge
attributes dwarf everything else.  :func:`molgen.graphs.featurize` rebuilds them
in the DataLoader workers, overlapped with the GPU.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import chem, graphs
from .sources import ComplexSpec

MANIFEST_NAME = "manifest.json"


@dataclass
class BuildConfig:
    """Graph-construction hyperparameters.

    ``pocket_radius`` / ``max_pocket_atoms`` / ``center_mode`` are baked into
    the shards; ``pocket_cutoff`` / ``inter_cutoff`` are applied at load time by
    :func:`molgen.graphs.featurize` and can still be changed afterwards.  The
    manifest records all of them so the training notebooks can assert the model
    they load was trained on matching graphs.
    """

    pocket_radius: float = 8.0
    max_pocket_atoms: int = 600
    pocket_cutoff: float = 4.5
    inter_cutoff: float = 6.0
    center_mode: str = "pocket"
    min_ligand_atoms: int = 6
    max_ligand_atoms: int = 48
    shard_size: int = 2000


def _process_one(spec: ComplexSpec, cfg: BuildConfig):
    """Parse one complex into an encoded sample.  Returns ``(sample, reason)``."""
    try:
        receptor, ligand, bonds = spec.load()
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the build
        return None, f"load_error:{type(exc).__name__}"

    lig = ligand.heavy()
    keep, reason = chem.is_interesting_ligand(
        spec.ligand_name or "LIG",
        len(lig),
        lig.element,
        min_atoms=cfg.min_ligand_atoms,
        max_atoms=cfg.max_ligand_atoms,
    )
    if not keep:
        return None, reason

    try:
        sample = graphs.encode_complex(
            receptor,
            lig,
            complex_id=spec.complex_id,
            lig_name=spec.ligand_name,
            affinity=spec.affinity,
            pocket_radius=cfg.pocket_radius,
            max_pocket_atoms=cfg.max_pocket_atoms,
            center_mode=cfg.center_mode,
            lig_bonds=bonds,
        )
    except Exception:  # noqa: BLE001
        return None, "graph_error"

    if sample is None:
        return None, "empty_pocket"
    return sample, "ok"


def _worker(payload):
    specs, cfg = payload
    out, reasons = [], Counter()
    for spec in specs:
        data, reason = _process_one(spec, cfg)
        reasons[reason] += 1
        if data is not None:
            out.append(data)
    return out, reasons


def _chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_shards(
    specs: Sequence[ComplexSpec],
    out_dir: str | Path,
    split: str = "train",
    cfg: BuildConfig | None = None,
    num_workers: int | None = None,
    chunk_size: int = 128,
    time_budget_s: float | None = None,
    verbose: bool = True,
) -> dict:
    """Parse ``specs`` in parallel and write ``{split}_{k:05d}.pt`` shards.

    ``time_budget_s`` stops cleanly (keeping what was built) before Kaggle kills
    the session, so a partial dataset is still usable.
    """
    cfg = cfg or BuildConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_workers = num_workers or max(1, (os.cpu_count() or 2))

    buffer: list = []
    shard_paths: list[str] = []
    shard_sizes: list[int] = []
    reasons: Counter = Counter()
    n_written = 0
    nbytes = 0
    started = time.time()
    stopped_early = False

    def flush() -> None:
        nonlocal buffer, n_written, nbytes
        if not buffer:
            return
        path = out_dir / f"{split}_{len(shard_paths):05d}.pt"
        torch.save(buffer, path)
        shard_paths.append(path.name)
        shard_sizes.append(len(buffer))
        n_written += len(buffer)
        nbytes += path.stat().st_size
        buffer = []

    payloads = [(chunk, cfg) for chunk in _chunks(list(specs), chunk_size)]

    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i, (datas, rs) in enumerate(pool.map(_worker, payloads)):
                buffer.extend(datas)
                reasons.update(rs)
                if len(buffer) >= cfg.shard_size:
                    flush()
                if verbose and i % 25 == 0:
                    done = min((i + 1) * chunk_size, len(specs))
                    rate = done / max(time.time() - started, 1e-6)
                    print(
                        f"[{split}] {done}/{len(specs)} specs | kept {n_written + len(buffer)}"
                        f" | {rate:.0f}/s | {time.time() - started:.0f}s",
                        flush=True,
                    )
                if time_budget_s and time.time() - started > time_budget_s:
                    print(f"[{split}] time budget reached, stopping early", flush=True)
                    stopped_early = True
                    break
    else:
        for i, payload in enumerate(payloads):
            datas, rs = _worker(payload)
            buffer.extend(datas)
            reasons.update(rs)
            if len(buffer) >= cfg.shard_size:
                flush()
            if time_budget_s and time.time() - started > time_budget_s:
                stopped_early = True
                break
    flush()

    manifest = {
        "split": split,
        "shards": shard_paths,
        "shard_sizes": shard_sizes,
        "num_graphs": n_written,
        "num_specs": len(specs),
        "stopped_early": stopped_early,
        "reject_reasons": dict(reasons),
        "config": asdict(cfg),
        "feature_dims": {
            "ligand_type_dim": graphs.LIGAND_TYPE_DIM,
            "pocket_feature_dim": graphs.POCKET_FEATURE_DIM,
            "ligand_extra_dim": graphs.LIGAND_EXTRA_DIM,
            "ligand_bond_edge_dim": graphs.LIGAND_BOND_EDGE_DIM,
            "pocket_edge_dim": graphs.POCKET_EDGE_DIM,
            "inter_edge_dim": graphs.INTER_EDGE_DIM,
            "ligand_elements": chem.LIGAND_ELEMENTS,
        },
        "build_seconds": round(time.time() - started, 1),
        "bytes_on_disk": nbytes,
        "bytes_per_graph": int(nbytes / max(n_written, 1)),
    }
    with open(out_dir / f"{split}_{MANIFEST_NAME}", "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(
            f"[{split}] wrote {n_written} complexes in {len(shard_paths)} shards "
            f"({nbytes / 1e6:.0f} MB, {nbytes / max(n_written, 1) / 1e3:.1f} kB each) -> {out_dir}"
        )
        print(f"[{split}] rejections: {dict(reasons)}")
    return manifest


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
class ShardedComplexDataset(Dataset):
    """Lazily-loaded dataset over ``*.pt`` shards, featurising on access.

    Keeps at most ``cache_shards`` shards in memory.  With ``shuffle=True`` in
    the DataLoader, access is random across shards, so use several DataLoader
    workers — each holds its own small cache and does its own featurisation, so
    the CPU work overlaps the GPU step.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        cache_shards: int = 2,
        max_graphs: int | None = None,
        pocket_cutoff: float = 4.5,
        inter_cutoff: float = 6.0,
    ):
        self.root = Path(root)
        self.split = split
        self.pocket_cutoff = pocket_cutoff
        self.inter_cutoff = inter_cutoff
        manifest_path = self.root / f"{split}_{MANIFEST_NAME}"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            shard_names = self.manifest["shards"]
        else:
            self.manifest = {}
            shard_names = sorted(p.name for p in self.root.glob(f"{split}_*.pt"))
        if not shard_names:
            raise FileNotFoundError(f"no shards for split '{split}' under {self.root}")

        self.shard_paths = [self.root / n for n in shard_names]
        self._cache: dict[int, list] = {}
        self._cache_order: list[int] = []
        self.cache_shards = max(1, cache_shards)

        # Index: (shard_idx, offset_in_shard) for every graph.
        sizes = self.manifest.get("shard_sizes")
        if sizes is None:
            sizes = [len(self._load_shard(i)) for i in range(len(self.shard_paths))]
        self.shard_sizes = sizes
        self.index: list[tuple[int, int]] = [
            (s, k) for s, size in enumerate(sizes) for k in range(size)
        ]
        if max_graphs is not None:
            self.index = self.index[:max_graphs]

    def _load_shard(self, i: int) -> list:
        if i in self._cache:
            return self._cache[i]
        # weights_only=False is required: shards hold dicts of numpy arrays.
        shard = torch.load(self.shard_paths[i], map_location="cpu", weights_only=False)
        self._cache[i] = shard
        self._cache_order.append(i)
        while len(self._cache_order) > self.cache_shards:
            drop = self._cache_order.pop(0)
            self._cache.pop(drop, None)
        return shard

    def __len__(self) -> int:
        return len(self.index)

    def get_sample(self, idx: int) -> dict:
        """Raw encoded sample — use for statistics, to skip featurisation."""
        shard_idx, offset = self.index[idx]
        return self._load_shard(shard_idx)[offset]

    def __getitem__(self, idx: int):
        return graphs.featurize(
            self.get_sample(idx),
            pocket_cutoff=self.pocket_cutoff,
            inter_cutoff=self.inter_cutoff,
        )


def load_all(root: str | Path, split: str, featurized: bool = True) -> list:
    """Read every shard into RAM.  Fine for the affinity set (~20 k complexes)."""
    root = Path(root)
    manifest_path = root / f"{split}_{MANIFEST_NAME}"
    names = (
        json.loads(manifest_path.read_text())["shards"]
        if manifest_path.exists()
        else sorted(p.name for p in root.glob(f"{split}_*.pt"))
    )
    out: list = []
    for n in names:
        out.extend(torch.load(root / n, map_location="cpu", weights_only=False))
    return [graphs.featurize(s) for s in out] if featurized else out


def make_loader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    drop_last: bool = False,
):
    """PyG DataLoader wired with the ``follow_batch`` keys the models expect."""
    from torch_geometric.loader import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=graphs.FOLLOW_BATCH,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------
# ligand-size prior p(n_ligand_atoms | pocket size)
# --------------------------------------------------------------------------
@dataclass
class SizePrior:
    """Conditional histogram of ligand size given pocket size.

    A pocket-conditioned generator must decide *how many atoms* to place before
    denoising starts.  Sampling that from the empirical conditional keeps the
    generated molecules commensurate with the cavity instead of always emitting
    the dataset's mean size.
    """

    pocket_bins: list[int] = field(default_factory=list)   # upper edges, residue counts
    counts: list[list[int]] = field(default_factory=list)  # [bin][n_atoms]
    max_atoms: int = 64

    @classmethod
    def fit(
        cls,
        dataset,
        pocket_bins: Sequence[int] = (20, 30, 40, 50, 60, 10_000),
        max_atoms: int = 64,
        max_samples: int | None = None,
    ) -> "SizePrior":
        bins = list(pocket_bins)
        counts = np.zeros((len(bins), max_atoms + 1), dtype=np.int64)
        n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for i in range(n):
            n_res, n_lig = _sizes_of(dataset, i)
            if n_lig > max_atoms:
                continue
            b = int(np.searchsorted(bins, n_res, side="left"))
            counts[min(b, len(bins) - 1), n_lig] += 1
        return cls(pocket_bins=bins, counts=counts.tolist(), max_atoms=max_atoms)

    def sample(self, n_pocket_residues: int, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        counts = np.asarray(self.counts, dtype=np.float64)
        b = min(int(np.searchsorted(self.pocket_bins, n_pocket_residues, side="left")), len(self.pocket_bins) - 1)
        row = counts[b]
        if row.sum() < 10:                      # sparse bin -> fall back to the marginal
            row = counts.sum(0)
        if row.sum() == 0:
            return 24
        return int(rng.choice(len(row), p=row / row.sum()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "SizePrior":
        return cls(**json.loads(Path(path).read_text()))


def _raw(dataset, i: int):
    """Encoded sample when the dataset can provide one, else the graph."""
    getter = getattr(dataset, "get_sample", None)
    return getter(i) if getter is not None else dataset[i]


def _sizes_of(dataset, i: int) -> tuple[int, int]:
    d = _raw(dataset, i)
    if isinstance(d, dict):
        return int(d.get("n_poc_res", 0)), int(len(d["lig_type"]))
    n_res = int(d.n_poc_res.item()) if hasattr(d, "n_poc_res") else int(d.n_poc.item()) // 8
    return n_res, int(d.lig_pos.size(0))


def summarize(dataset, max_samples: int = 2000) -> dict:
    """Quick dataset statistics — sanity-check before committing GPU hours."""
    n = min(len(dataset), max_samples)
    n_lig, n_poc, n_res, y = [], [], [], []
    elem_counts: Counter = Counter()
    for i in range(n):
        d = _raw(dataset, i)
        if isinstance(d, dict):
            n_lig.append(int(len(d["lig_type"])))
            n_poc.append(int(len(d["poc_pos"])))
            n_res.append(int(d.get("n_poc_res", -1)))
            if np.isfinite(d.get("y", np.nan)):
                y.append(float(d["y"]))
            types = np.asarray(d["lig_type"]).tolist()
        else:
            n_lig.append(int(d.lig_pos.size(0)))
            n_poc.append(int(d.poc_pos.size(0)))
            n_res.append(int(d.n_poc_res.item()) if hasattr(d, "n_poc_res") else -1)
            if hasattr(d, "y") and torch.isfinite(d.y).all():
                y.append(float(d.y.item()))
            types = d.lig_type.tolist()
        for t in types:
            elem_counts[chem.LIGAND_ELEMENTS[int(t)]] += 1
    out = {
        "n_sampled": n,
        "ligand_atoms": {"mean": float(np.mean(n_lig)), "min": int(np.min(n_lig)), "max": int(np.max(n_lig))},
        "pocket_atoms": {"mean": float(np.mean(n_poc)), "min": int(np.min(n_poc)), "max": int(np.max(n_poc))},
        "pocket_residues": {"mean": float(np.mean(n_res))},
        "element_fraction": {k: round(v / max(sum(elem_counts.values()), 1), 4) for k, v in elem_counts.most_common()},
    }
    if y:
        out["affinity"] = {"n": len(y), "mean": float(np.mean(y)), "std": float(np.std(y)),
                           "min": float(np.min(y)), "max": float(np.max(y))}
    return out
