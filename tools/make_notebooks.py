#!/usr/bin/env python3
"""Generate the Kaggle notebooks from source.

Notebook JSON is miserable to edit by hand and impossible to review in a diff,
so the notebooks are generated.  Edit this file, re-run it, and the four
notebooks in ``notebooks/`` are rebuilt:

    python tools/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"


# ==========================================================================
# shared cells
# ==========================================================================
BOOTSTRAP = r'''
# --- locate the molgen package ---------------------------------------------
# On Kaggle, make `molgen` importable in one of these ways:
#   1. Add the "molgen-src" Dataset (upload this repo's `src/` folder), or
#   2. Add the output of notebook 01 (it ships a copy of `src/molgen`), or
#   3. Run the notebook from a clone of the repo.
import glob
import subprocess
import sys
from pathlib import Path


def find_molgen():
    patterns = [
        "/kaggle/input/*/src", "/kaggle/input/*", "/kaggle/input/*/*",
        "/kaggle/working/src", "./src", "../src", ".", "..",
    ]
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if (Path(candidate) / "molgen" / "__init__.py").is_file():
                return str(Path(candidate).resolve())
    return None


MOLGEN_SRC = find_molgen()
if MOLGEN_SRC is None:
    raise FileNotFoundError(
        "Could not find the `molgen` package.\n"
        "Attach the molgen-src dataset (the repo's src/ folder, containing molgen/), "
        "or the output of notebook 01, then re-run this cell."
    )
if MOLGEN_SRC not in sys.path:
    sys.path.insert(0, MOLGEN_SRC)
print("molgen source:", MOLGEN_SRC)
'''.strip()

DEPS = r'''
# --- dependencies -----------------------------------------------------------
# torch_geometric is required. We deliberately avoid torch_scatter/torch_cluster
# (they need wheels matched to the exact torch build and are the usual cause of
# Kaggle install failures) -- molgen uses scipy KD-trees and native torch ops.
# rdkit is optional: without it you lose SMILES/QED/SA but nothing else.
def ensure(module: str, package: str | None = None) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        pkg = package or module
        print(f"installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    try:
        __import__(module)
        return True
    except ImportError:
        print(f"  !! {module} unavailable")
        return False


HAS_PYG = ensure("torch_geometric")
HAS_RDKIT = ensure("rdkit")
assert HAS_PYG, "torch_geometric is required"
print("rdkit:", HAS_RDKIT)
'''.strip()

IMPORTS = r'''
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import molgen
from molgen import (
    affinity, chem, dataset, diffusion, egnn, evaluate, graphs, pdbio, pockets, sources,
)
from molgen.train_utils import (
    EMA, History, TimeBudget, count_parameters, device_info, grad_norm,
    load_checkpoint, save_checkpoint, set_seed, warmup_cosine_lambda,
)

warnings.filterwarnings("ignore", category=UserWarning)
set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("molgen", molgen.__version__, "| device:", device)
print(device_info())

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
WORK.mkdir(parents=True, exist_ok=True)
print("working dir:", WORK)
'''.strip()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    }


def notebook(cells: list[dict], name: str) -> None:
    for cell in cells:
        src = cell["source"]
        lines = src.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"isGpuEnabled": True, "isInternetEnabled": True, "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")


# ==========================================================================
# 01 — dataset construction
# ==========================================================================
def nb01() -> None:
    cells = [
        md(r"""
# 01 · Build pocket–ligand graph shards

Converts raw PDB files into the compact graph dataset every other notebook consumes.

**Inputs** (attach as Kaggle Datasets)
- `protein-ligand-binding-site` — the BioLiP-style export: `nonredund_rec/`, `nonredund_lig/`,
  `PL_nonredund_annotation.csv`. Trains the **generator** (no affinity labels).
- *optional* PDBBind / PDBBind+ `refined-set` + `INDEX_*` — trains the **affinity ranker**.

**Output** → `/kaggle/working/graphs/` : `{split}_*.pt` shards + `{split}_manifest.json`,
the ligand-size prior, and a copy of `src/molgen` so later notebooks need only this one input.

**Run this with the accelerator set to *None***. It is pure CPU work, so a GPU session
wastes your GPU quota and (on P100 sessions) gives you fewer CPU cores.

### What one sample contains
A complex is stored as positions plus small integer codes — ~10 kB. Featurising it into the
dense graph the model eats costs **0.8 ms** and inflates it **58×** to ~590 kB, so the dense
form is built on the fly in DataLoader workers. Storing it instead would need **88 GB** for the
full set against Kaggle's 20 GB output limit.
"""),
        code(BOOTSTRAP),
        code(DEPS),
        code(IMPORTS),
        md(r"""
## 1 · Locate the raw data

`sources.BioLiPSource` searches `/kaggle/input` for the three pieces it needs. Pass the paths
explicitly if your dataset slug nests them differently.
"""),
        code(r'''
biolip = sources.BioLiPSource()      # or BioLiPSource(annotation_csv=..., receptor_dir=..., ligand_dir=...)
print("annotation:", biolip.annotation_csv)
print("receptors :", biolip.receptor_dir)
print("ligands   :", biolip.ligand_dir)

n_rec = sum(1 for _ in biolip.receptor_dir.glob("*.pdb"))
n_lig = sum(1 for _ in biolip.ligand_dir.glob("*.pdb"))
print(f"\n{n_rec:,} receptor files | {n_lig:,} ligand files")
'''),
        md(r"""
## 2 · Configuration

`pocket_radius` and `max_pocket_atoms` are **baked into the shards** — changing them later
means rebuilding. The load-time cutoffs (`pocket_cutoff`, `inter_cutoff`) are not.

Set `MAX_COMPLEXES = None` for the full set. The full build is ~40 min on 4 cores; the
`TIME_BUDGET` stops cleanly and keeps whatever finished, so a partial dataset is still usable.
"""),
        code(r'''
CFG = dataset.BuildConfig(
    pocket_radius=8.0,       # A around any ligand heavy atom, whole residues kept
    max_pocket_atoms=600,    # caps memory per graph; ~600 covers a druggable site
    pocket_cutoff=4.5,       # pocket contact graph radius (load time)
    inter_cutoff=6.0,        # ligand-pocket cross edges (load time)
    center_mode="pocket",    # origin = pocket centroid, known at inference too
    min_ligand_atoms=6,
    max_ligand_atoms=48,     # must be >= the largest ligand you want to generate
    shard_size=4000,
)

MAX_COMPLEXES = None          # None = all; set e.g. 20_000 for a fast first pass
DEDUP = True                  # one entry per (assembly, ligand id)
SPLITS = (0.94, 0.03, 0.03)
TIME_BUDGET = {"train": 4.0 * 3600, "val": 900, "test": 900}

GRAPH_DIR = WORK / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
print(CFG)
'''),
        md(r"""
## 3 · Index and filter

Rejections are counted by reason, so you can see *why* the set shrank rather than guessing.
Ions, buffers, cryoprotectants, bare amino acids and nucleotides are dropped — BioLiP calls
them ligands, but they are not generative targets.
"""),
        code(r'''
t0 = time.time()
specs = biolip.specs(dedup=DEDUP, limit=MAX_COMPLEXES)
print(f"{len(specs):,} candidate complexes indexed in {time.time() - t0:.0f}s")
print(f"{len({s.receptor_key for s in specs}):,} distinct PDB entries")

ligand_counts = pd.Series([s.ligand_name for s in specs]).value_counts()
print("\nmost common ligands:")
print(ligand_counts.head(12).to_string())
'''),
        md(r"""
## 4 · Split by PDB entry, not at random

A random split puts `1b7z_1_OXL_D` in train and `1b7z_1_OXL_E` in test — the same protein with
the same ligand in a second copy of the site. Every metric would then be measuring memorisation.
Grouping on the 4-character PDB code is the minimum defensible unit.
"""),
        code(r'''
split_specs = sources.split_by_receptor(specs, fractions=SPLITS, seed=0)
for name, items in split_specs.items():
    print(f"{name:>5}: {len(items):>7,} complexes  {len({s.receptor_key for s in items}):>6,} PDB entries")

overlap = (
    {s.receptor_key for s in split_specs["train"]}
    & {s.receptor_key for s in split_specs["test"]}
)
assert not overlap, f"leakage: {len(overlap)} PDB entries in both train and test"
print("\nno PDB entry appears in two splits")
'''),
        md(r"""
## 5 · Build the shards

Parsing runs in a process pool. Failures are per-complex and never abort the build.
"""),
        code(r'''
import os

manifests = {}
for name in ("val", "test", "train"):      # small splits first: fail fast
    manifests[name] = dataset.build_shards(
        split_specs[name],
        GRAPH_DIR,
        split=name,
        cfg=CFG,
        num_workers=max(1, os.cpu_count() or 2),
        chunk_size=128,
        time_budget_s=TIME_BUDGET[name],
    )
    print()

total = sum(m["num_graphs"] for m in manifests.values())
size_gb = sum(m["bytes_on_disk"] for m in manifests.values()) / 1e9
print(f"TOTAL {total:,} complexes, {size_gb:.2f} GB on disk")
'''),
        code(r'''
# why complexes were dropped, pooled over splits
rej = pd.Series(
    pd.DataFrame([m["reject_reasons"] for m in manifests.values()]).sum().sort_values(ascending=False)
)
rej.name = "count"
display(rej.to_frame().assign(pct=lambda d: (100 * d["count"] / d["count"].sum()).round(1)))
'''),
        md(r"""
## 6 · Fit the ligand-size prior and the coordinate scale

Two numbers the generator cannot work without:

- **`SizePrior`** — a pocket-conditioned histogram of ligand size. A pocket-conditioned model
  must choose *how many atoms* to place before denoising starts; sampling that from
  `p(n_atoms | pocket residues)` keeps molecules commensurate with the cavity instead of
  always emitting the dataset mean.
- **`coord_scale`** — the std of ligand coordinates about the pocket centroid. A DDPM's
  terminal prior is unit variance, so this is the factor that maps it onto Ångström. Get it
  wrong by 2× and you either generate a tight blob at the centre or spray atoms into solvent.
"""),
        code(r'''
train_ds = dataset.ShardedComplexDataset(GRAPH_DIR, "train",
                                         pocket_cutoff=CFG.pocket_cutoff,
                                         inter_cutoff=CFG.inter_cutoff)
print(f"train dataset: {len(train_ds):,} complexes")

prior = dataset.SizePrior.fit(train_ds, max_samples=30_000, max_atoms=CFG.max_ligand_atoms)
prior.save(GRAPH_DIR / "size_prior.json")

coord_scale = diffusion.estimate_coord_scale(train_ds, max_samples=5_000)

meta = {
    "build_config": manifests["train"]["config"],
    "feature_dims": manifests["train"]["feature_dims"],
    "coord_scale": round(coord_scale, 4),
    "counts": {k: m["num_graphs"] for k, m in manifests.items()},
}
(GRAPH_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

print(f"coord_scale = {coord_scale:.3f} A")
print("size prior draws for a 40-residue pocket:",
      [prior.sample(40) for _ in range(10)])
print(json.dumps(meta["feature_dims"], indent=2))
'''),
        md(r"""
## 7 · Sanity check the dataset

Look at this before spending GPU hours. The things that go wrong here are a pocket-size
distribution pinned against `max_pocket_atoms` (raise it), an element distribution with a
surprise element (check the filter), and ligand sizes clipped at `max_ligand_atoms`.
"""),
        code(r'''
stats = dataset.summarize(train_ds, max_samples=5_000)
print(json.dumps(stats, indent=2))
'''),
        code(r'''
n_show = min(len(train_ds), 5_000)
rows = [dataset._sizes_of(train_ds, i) for i in range(n_show)]
n_res = np.array([r[0] for r in rows]); n_lig = np.array([r[1] for r in rows])
n_poc = np.array([len(train_ds.get_sample(i)["poc_pos"]) for i in range(n_show)])

fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
ax[0].hist(n_lig, bins=range(0, CFG.max_ligand_atoms + 2), color="#3b82f6")
ax[0].set_title("ligand heavy atoms"); ax[0].set_xlabel("atoms")
ax[1].hist(n_poc, bins=40, color="#10b981")
ax[1].axvline(CFG.max_pocket_atoms, color="crimson", ls="--", label="cap")
ax[1].set_title("pocket atoms"); ax[1].legend()
ax[2].hist(n_res, bins=40, color="#f59e0b"); ax[2].set_title("pocket residues")
frac = pd.Series(stats["element_fraction"])
ax[3].bar(frac.index, frac.values, color="#8b5cf6"); ax[3].set_title("ligand element fraction")
ax[3].set_yscale("log")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"ligands at the {CFG.max_ligand_atoms}-atom cap: {(n_lig >= CFG.max_ligand_atoms).mean():.2%}")
print(f"pockets at the {CFG.max_pocket_atoms}-atom cap: {(n_poc >= CFG.max_pocket_atoms).mean():.2%}")
'''),
        code(r'''
# one complex end to end: the shapes the models will see
d = train_ds[0]
print(d)
print(f"\n{d.complex_id}  ligand={d.lig_name}")
print(f"  ligand atoms      {d.lig_pos.shape[0]:>5}   bonds (directed) {d.lig_bond_index.shape[1]:>6}")
print(f"  pocket atoms      {d.poc_pos.shape[0]:>5}   contacts         {d.poc_edge_index.shape[1]:>6}")
print(f"  cross edges       {d.inter_edge_index.shape[1]:>5}")
print(f"  covalent fraction of pocket edges: {d.poc_edge_attr[:, 0].mean():.3f}")
print(f"  ligand elements: {[chem.LIGAND_ELEMENTS[t] for t in d.lig_type.tolist()]}")
'''),
        md(r"""
## 8 · Optional · PDBBind shards with affinity labels

The generator trains on BioLiP; the **ranker needs labels**, which only PDBBind/PDBBind+
provides. Download `refined-set` (or the general set) plus the `INDEX_*` files from
<https://www.pdbbind-plus.org.cn/download>, attach it as a Dataset, and run this section.

The CASF-2016 core set is forced into the test split: it is the standard held-out benchmark
and is a subset of the refined set, so leaving it in train would make the reported
correlation meaningless.
"""),
        code(r'''
AFFINITY_DIR = WORK / "graphs_affinity"

try:
    pdbbind = sources.PDBBindSource()
    print("set dir   :", pdbbind.set_dir)
    print("index file:", pdbbind.index_file)
    print(f"{len(pdbbind.labels):,} labelled entries")

    aff_specs = pdbbind.specs(require_affinity=True, prefer_pocket_file=True)
    print(f"{len(aff_specs):,} complexes with both files and a label")

    # CASF-2016 core set, if its index is present alongside
    core = sources.find_file(r"INDEX_core_data\.2016")
    holdout = set(sources.parse_pdbbind_index(core)) if core else None
    print("CASF-2016 core entries forced into test:", len(holdout or []))

    aff_split = sources.split_by_receptor(aff_specs, (0.8, 0.1, 0.1), seed=0, holdout_ids=holdout)
    AFFINITY_DIR.mkdir(parents=True, exist_ok=True)
    aff_manifests = {}
    for name in ("val", "test", "train"):
        aff_manifests[name] = dataset.build_shards(
            aff_split[name], AFFINITY_DIR, split=name, cfg=CFG,
            num_workers=max(1, os.cpu_count() or 2), chunk_size=64,
        )
    aff_train = dataset.ShardedComplexDataset(AFFINITY_DIR, "train")
    print("\naffinity label stats:", json.dumps(dataset.summarize(aff_train).get("affinity", {}), indent=2))
    HAS_AFFINITY_DATA = True
except FileNotFoundError as exc:
    print("PDBBind not found -> skipping affinity shards.")
    print(" ", exc)
    print("\nNotebook 03 needs these. Attach PDBBind and re-run this cell.")
    HAS_AFFINITY_DATA = False
'''),
        md(r"""
## 9 · Ship the library with the output

Copying `src/molgen` into the output means notebooks 02–04 need **one** input dataset: this
notebook's output. Save the version, click **Save Version → Save & Run All**, then add the
output as a Dataset input to the next notebook.
"""),
        code(r'''
import shutil

dst = WORK / "src" / "molgen"
if Path(MOLGEN_SRC, "molgen").resolve() != dst.resolve():
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(Path(MOLGEN_SRC) / "molgen", dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print("library copied to:", dst)

print("\noutput tree:")
for p in sorted(WORK.rglob("*")):
    if p.is_file() and "__pycache__" not in str(p):
        rel = p.relative_to(WORK)
        print(f"  {str(rel):<48} {p.stat().st_size / 1e6:>9.2f} MB")
print(f"\ntotal: {sum(f.stat().st_size for f in WORK.rglob('*') if f.is_file()) / 1e9:.2f} GB")
'''),
        md(r"""
---
**Next:** `02_train_pocket_conditioned_diffusion.ipynb` — attach this output and train the generator.
"""),
    ]
    notebook(cells, "01_build_pocket_ligand_graphs.ipynb")


# ==========================================================================
# 02 — diffusion training
# ==========================================================================
def nb02() -> None:
    cells = [
        md(r"""
# 02 · Train the pocket-conditioned equivariant diffusion model

Learns `p(ligand | pocket)` — joint diffusion over ligand **coordinates** and **atom types**,
denoised by an E(n)-equivariant GNN with the pocket as frozen context.

**Input** → the output of notebook 01 (`graphs/` + `src/molgen`). **Output** → checkpoints in
`/kaggle/working/`, resumable across sessions.

### The model
Forward process, per ligand atom, on `z = x / coord_scale` and one-hot types `h / type_scale`:

$$q(z_t \mid z_0) = \mathcal{N}\!\left(\alpha_t z_0,\; \sigma_t^2 I\right)$$

The network predicts the **denoised** $\hat x_0$ and training minimises
$\lVert \hat x_0 - x_0 \rVert^2$ over coordinates and types, plus a cross-entropy term on
element identity.

**Why $x_0$ and not $\epsilon$.** With noise prediction the reverse mean is
$z_t/\alpha_{t|s} - (\sigma^2_{t|s}/\alpha_{t|s}\sigma_t)\,\hat\epsilon$ — it **divides** $z$
by $\alpha_{t|s} < 1$ every step, amplifying it by $\alpha_0/\alpha_T \approx 316\times$ over
the trajectory, and only a perfect $\hat\epsilon$ cancels that. Here the error does not merely
degrade samples, it runs away: once an atom drifts past the 8 Å cross-edge cutoff the pocket
context vanishes from its neighbourhood, the network predicts almost nothing, and the rest of
the trajectory is pure amplification. *Measured:* a model overfit to 8 complexes put its
samples **331 Å** from the pocket (untrained: 472 Å) with the loss already down 78 %.

Predicting $x_0$ gives $\mu = c_t z_t + c_0 \hat x_0$ with $c_t, c_0 \ge 0$ and
$c_t + c_0 \approx 1$ — an interpolation, so nothing is amplified, and a bad prediction
collapses samples toward the pocket centroid where the context still reaches them. Same
posterior, numerically stable half. This is what TargetDiff does for the same task.

**Why equivariance.** Coordinate updates are built only from *relative* positions scaled by
scalar messages:

$$x_i \leftarrow x_i + \sum_j \frac{x_i - x_j}{d_{ij} + 1}\, \phi_x(m_{ij})$$

so rotating the input rotates the output exactly, by construction. A model on raw coordinates
would have to learn that symmetry from data — unaffordable at this dataset size.

**Why the pocket is frozen.** Pocket atoms join message passing but their coordinates never
move, and the complex was centred on the pocket centroid during preprocessing. That handles
translation, so no zero-centre-of-mass subspace projection is needed and every radius cutoff
keeps its meaning in Ångström.
"""),
        code(BOOTSTRAP),
        code(DEPS),
        code(IMPORTS),
        md(r"""
## 1 · Load the dataset and its metadata

`coord_scale` comes from the dataset, not a guess — see notebook 01 §6.
"""),
        code(r'''
GRAPH_DIR = next(
    (p for p in [Path(g) for g in glob.glob("/kaggle/input/*/graphs")] + [WORK / "graphs"]
     if (p / "train_manifest.json").exists()),
    None,
)
assert GRAPH_DIR is not None, "could not find graphs/ — attach the notebook-01 output"
print("graphs:", GRAPH_DIR)

meta = json.loads((GRAPH_DIR / "dataset_meta.json").read_text())
build = meta["build_config"]
print(json.dumps(meta, indent=2))

train_ds = dataset.ShardedComplexDataset(GRAPH_DIR, "train", cache_shards=2,
                                         pocket_cutoff=build["pocket_cutoff"],
                                         inter_cutoff=build["inter_cutoff"])
val_ds = dataset.ShardedComplexDataset(GRAPH_DIR, "val", cache_shards=1,
                                       pocket_cutoff=build["pocket_cutoff"],
                                       inter_cutoff=build["inter_cutoff"])
size_prior = dataset.SizePrior.load(GRAPH_DIR / "size_prior.json")
print(f"\ntrain {len(train_ds):,} | val {len(val_ds):,}")
'''),
        md(r"""
## 2 · Configuration

Defaults are sized for one Kaggle P100/T4 session. The knobs that matter most:

| knob | effect |
|---|---|
| `timesteps` | 500 is plenty for molecules; 1000 doubles sampling cost for little gain |
| `hidden` / `num_layers` | 128×6 ≈ 1.4 M params, fits comfortably; 256×9 if you have the hours |
| `cross_cutoff` | how far ligand atoms see the pocket. 8 Å covers the first two shells |
| `type_loss_weight` / `type_ce_weight` | ↑ favours correct elements, ↓ favours correct geometry |
| `parameterization` | leave at `"x0"`. `"eps"` is there for comparison and is unstable here |
| `batch_size` | pocket graphs are the memory cost — 16 is safe at 600 pocket atoms |
"""),
        code(r'''
DCFG = diffusion.DiffusionConfig(
    timesteps=500,
    schedule="polynomial",          # decays slowly at small t: fine geometry is set late
    coord_scale=meta["coord_scale"],
    type_scale=4.0,                 # EDM convention; keeps type signal alive under noise
    type_loss_weight=1.0,
    type_ce_weight=0.3,             # auxiliary CE sharpens element identity
    parameterization="x0",          # see the note above — do not switch to "eps"
    clip_x0=4.0,                    # normalised units from the pocket centroid
    hidden=128,
    num_layers=6,
    cross_cutoff=8.0,
    cross_max_neighbors=32,
    attention=True,
    ligand_type_dim=meta["feature_dims"]["ligand_type_dim"],
    pocket_feature_dim=meta["feature_dims"]["pocket_feature_dim"],
)

TRAIN = dict(
    epochs=200,
    batch_size=16,
    lr=2e-4,
    weight_decay=1e-12,
    grad_clip=1.0,
    warmup_steps=1000,
    ema_decay=0.999,
    num_workers=2,
    val_every=1,
    log_every=100,
    max_steps_per_epoch=None,       # cap for quick experiments
    session_hours=8.5,              # Kaggle kills at 9 h
)

CKPT = WORK / "diffusion_last.pt"
BEST = WORK / "diffusion_best.pt"
RESUME = CKPT if CKPT.exists() else None     # set to a previous session's checkpoint
print(DCFG)
'''),
        md(r"""
## 3 · Build the model and **verify the symmetry numerically**

Run this every time. A broken equivariance is completely silent — the model still trains, just
badly — so it has to be asserted, not assumed. Rotating the complex must rotate `eps_x` by the
same matrix and leave the type prediction unchanged.
"""),
        code(r'''
model = diffusion.PocketLigandDiffusion(DCFG).to(device)
print(f"{count_parameters(model):,} trainable parameters")

check_batch = diffusion.build_pocket_batch([train_ds[i] for i in range(4)], keep_ligand=True)
report = egnn.check_equivariance(model.net, check_batch, device="cpu")
print(json.dumps(report, indent=2))
assert report["passed"], "E(3) equivariance broken — do not train this"
print(f"\nall four symmetry checks hold to {report['worst_error']:.1e}")
model = model.to(device)
'''),
        code(r'''
train_loader = dataset.make_loader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True,
                                   num_workers=TRAIN["num_workers"], drop_last=True)
val_loader = dataset.make_loader(val_ds, batch_size=TRAIN["batch_size"], shuffle=False,
                                 num_workers=TRAIN["num_workers"])

opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
steps_per_epoch = TRAIN["max_steps_per_epoch"] or len(train_loader)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, warmup_cosine_lambda(TRAIN["warmup_steps"], TRAIN["epochs"] * steps_per_epoch)
)
ema = EMA(model, decay=TRAIN["ema_decay"])
history = History(WORK / "diffusion_history.jsonl")
budget = TimeBudget(limit_s=TRAIN["session_hours"] * 3600)

start_epoch, global_step, best_val = 0, 0, float("inf")
if RESUME is not None:
    ck = load_checkpoint(RESUME, model, opt, sched, ema, map_location=device)
    start_epoch, global_step = ck["epoch"] + 1, ck["step"]
    best_val = ck.get("extra", {}).get("best_val", float("inf"))
    print(f"resumed from {RESUME} at epoch {start_epoch}, step {global_step:,}")

print(f"{steps_per_epoch:,} steps/epoch | {budget}")
'''),
        md(r"""
## 4 · Validation loss, made comparable

The training loss samples `t` at random, so epoch-to-epoch differences are mostly noise in the
`t` draw. Validation therefore uses a **fixed seed**, which makes the curve actually readable.
"""),
        code(r'''
@torch.no_grad()
def validate(net, loader, max_batches=50, seed=1234):
    net.eval()
    totals, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        torch.manual_seed(seed + i)          # same noise draw every evaluation
        out = net(batch.to(device))
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in totals.items()}
'''),
        md("## 5 · Train"),
        code(r'''
model.train()
stop = False
for epoch in range(start_epoch, TRAIN["epochs"]):
    t_epoch = time.time()
    run, seen = {}, 0

    for i, batch in enumerate(train_loader):
        if TRAIN["max_steps_per_epoch"] and i >= TRAIN["max_steps_per_epoch"]:
            break
        out = model(batch.to(device))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
        opt.step(); sched.step(); ema.update(model)

        global_step += 1
        seen += 1
        for k, v in out.items():
            run[k] = run.get(k, 0.0) + float(v)

        if global_step % TRAIN["log_every"] == 0:
            avg = {k: v / seen for k, v in run.items()}
            history.log(step=global_step, epoch=epoch, split="train", **avg,
                        lr=sched.get_last_lr()[0], grad_norm=float(gnorm))
            print(f"e{epoch:>3} s{global_step:>7,} | loss {avg['loss']:.4f} "
                  f"(x {avg['loss_x']:.4f}  h {avg['loss_h']:.4f}"
                  f"{'  acc %.2f' % avg['type_acc'] if 'type_acc' in avg else ''}) | "
                  f"lr {sched.get_last_lr()[0]:.2e} | gn {float(gnorm):.2f}", flush=True)

        if budget.expired():
            print(f"\ntime budget reached ({budget}) — checkpointing and stopping")
            stop = True
            break

    if epoch % TRAIN["val_every"] == 0 or stop:
        ema.apply_to(model)                   # EMA weights are what we will sample with
        val = validate(model, val_loader)
        ema.restore(model)
        history.log(step=global_step, epoch=epoch, split="val", **{f"val_{k}": v for k, v in val.items()})
        print(f"  -> epoch {epoch} val loss {val['loss']:.4f} "
              f"(x {val['loss_x']:.4f}  h {val['loss_h']:.4f}"
              f"{'  acc %.2f' % val['type_acc'] if 'type_acc' in val else ''})"
              f" | {time.time()-t_epoch:.0f}s | {budget}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(BEST, model, opt, sched, ema, epoch, global_step,
                            config=model.config_dict(), extra={"best_val": best_val, "meta": meta})
            print(f"     new best ({best_val:.4f}) -> {BEST.name}")

    save_checkpoint(CKPT, model, opt, sched, ema, epoch, global_step,
                    config=model.config_dict(), extra={"best_val": best_val, "meta": meta})
    if stop:
        break

print(f"\nfinished at epoch {epoch}, step {global_step:,}; best val {best_val:.4f}")
'''),
        code(r'''
df = history.to_frame()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
tr = df[df.split == "train"]; va = df[df.split == "val"]
ax[0].plot(tr.step, tr.loss, lw=0.8, label="train")
if len(va): ax[0].plot(va.step, va.val_loss, "o-", label="val")
ax[0].set_title("total loss"); ax[0].set_yscale("log")
ax[1].plot(tr.step, tr.loss_x, lw=0.8, label="coords (x0 MSE)")
ax[1].plot(tr.step, tr.loss_h, lw=0.8, label="types (MSE)")
if "loss_ce" in tr: ax[1].plot(tr.step, tr.loss_ce, lw=0.8, label="types (CE)")
ax[1].set_title("loss terms"); ax[1].set_yscale("log")
if "type_acc" in tr:
    ax[2].plot(tr.step, tr.type_acc, lw=0.8, label="train")
    if len(va) and "val_type_acc" in va: ax[2].plot(va.step, va.val_type_acc, "o-", label="val")
    ax[2].set_title("element accuracy at random t"); ax[2].set_ylim(0, 1)
else:
    ax[2].plot(tr.step, tr.grad_norm, lw=0.6); ax[2].set_title("grad norm"); ax[2].set_yscale("log")
for a in ax: a.set_xlabel("step"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),
        md(r"""
## 6 · Sample from held-out pockets

The real test of a generator is not its loss. `build_pocket_batch` **strips the crystal
ligand** — generating with the true ligand still in the batch is the easiest way to report
leaked results, so it is made structurally impossible rather than promised.

Expect from a short run: coordinates in the right region but poor validity. Validity above
~80 % with mostly single fragments needs a few hundred epochs.
"""),
        code(r'''
N_POCKETS, PER_POCKET, SAMPLE_STEPS = 8, 5, 250

ema.apply_to(model)
model.eval()

rng = np.random.default_rng(0)
pick = rng.choice(len(val_ds), size=min(N_POCKETS, len(val_ds)), replace=False)
records, generated = [], []

for idx in pick:
    truth = val_ds[int(idx)]
    repeats = [truth] * PER_POCKET
    batch = diffusion.build_pocket_batch(repeats, device=device)   # ligand stripped here
    n_atoms = [size_prior.sample(int(truth.n_poc_res), rng) for _ in range(PER_POCKET)]

    out = model.sample(batch, n_atoms=n_atoms, num_steps=SAMPLE_STEPS, eta=1.0)
    poc_pos = truth.poc_pos.numpy()
    # real vdW radii per pocket atom: defaulting everything to carbon
    # under-counts clashes against sulfur and over-counts against oxygen
    poc_el = graphs.pocket_elements_from_features(truth.poc_x)
    for mol in model.to_molecules(out):
        rec = evaluate.evaluate_molecule(mol["elements"], mol["pos"], poc_pos=poc_pos,
                                         poc_elements=poc_el,
                                         index=len(records), pocket_id=truth.complex_id)
        records.append(rec)
        generated.append((truth, mol))

ema.restore(model)
metrics = evaluate.set_metrics(records)
print(json.dumps(metrics, indent=2))

# The check that matters first, before any chemistry: did the molecules land in
# the pocket at all? Compare the generated centroid to the crystal ligand's.
# A number in the tens or hundreds of Angstrom means the reverse process is
# diverging, not that the chemistry needs more epochs.
errs = [float(np.linalg.norm(mol["pos"].mean(0) - truth.lig_pos.numpy().mean(0)))
        for truth, mol in generated]
rg_gen = [float(np.sqrt(((m["pos"] - m["pos"].mean(0)) ** 2).sum(1).mean())) for _, m in generated]
rg_true = [float(np.sqrt(((t.lig_pos.numpy() - t.lig_pos.numpy().mean(0)) ** 2).sum(1).mean()))
           for t, _ in generated]
print(f"\ncentroid offset from the crystal ligand: mean {np.mean(errs):.2f} A, max {np.max(errs):.2f} A")
print(f"radius of gyration: generated {np.mean(rg_gen):.2f} A vs crystal {np.mean(rg_true):.2f} A")
assert np.mean(errs) < 25.0, (
    f"samples are {np.mean(errs):.0f} A from the pocket — the reverse process is diverging, "
    "not under-trained. Check DCFG.parameterization == 'x0' and clip_x0."
)
history.log(step=global_step, epoch=epoch, split="sample", **{f"gen_{k}": v for k, v in metrics.items()
                                                              if isinstance(v, (int, float))})
'''),
        code(r'''
tbl = pd.DataFrame([r.row() for r in records])
cols = [c for c in ["pocket_id", "n_atoms", "valid", "n_fragments", "min_dist",
                    "n_clash", "buriedness", "qed", "sa", "smiles"] if c in tbl.columns]
display(tbl[cols].head(20))

# reference: the same statistics for the real ligands of those pockets
real = [evaluate.evaluate_molecule(
            [chem.LIGAND_ELEMENTS[t] for t in val_ds[int(i)].lig_type.tolist()],
            val_ds[int(i)].lig_pos.numpy(), poc_pos=val_ds[int(i)].poc_pos.numpy(),
            poc_elements=graphs.pocket_elements_from_features(val_ds[int(i)].poc_x))
        for i in pick]
print("\ngenerated:", {k: round(v, 3) for k, v in evaluate.set_metrics(records).items() if isinstance(v, float)})
print("crystal  :", {k: round(v, 3) for k, v in evaluate.set_metrics(real).items() if isinstance(v, float)})
'''),
        code(r'''
# write the best molecules per pocket back into the crystal frame for PyMOL
best_by_pocket = {}
for rec, (truth, _) in zip(records, generated):
    rec.score = evaluate.composite_score(rec)
    if rec.score > best_by_pocket.get(rec.pocket_id, (None, -1e9))[1]:
        best_by_pocket[rec.pocket_id] = (rec, rec.score, truth)

blocks = 0
for pid, (rec, _, truth) in best_by_pocket.items():
    blocks += evaluate.records_to_sdf([rec], WORK / f"sample_{pid}.sdf",
                                      offset=truth.center.numpy().reshape(3))
print(f"wrote {blocks} molecules as sample_*.sdf (crystal frame — open next to the receptor PDB)")

save_checkpoint(WORK / "diffusion_final.pt", model, ema=ema, epoch=epoch, step=global_step,
                config=model.config_dict(),
                extra={"best_val": best_val, "meta": meta, "sample_metrics": metrics})
print("saved diffusion_final.pt")
'''),
        md(r"""
---
**Next:** `03_train_binding_affinity.ipynb` for the ranker, then
`04_generate_and_rank.ipynb` for the full protein → ranked molecules pipeline.

**To continue training in a new session:** attach this notebook's output, point `RESUME` at
`diffusion_last.pt`, and re-run. Optimiser, scheduler and EMA state all restore.
"""),
    ]
    notebook(cells, "02_train_pocket_conditioned_diffusion.ipynb")


# ==========================================================================
# 03 — affinity model
# ==========================================================================
def nb03() -> None:
    cells = [
        md(r"""
# 03 · Train the binding-affinity ranker

Regresses $\mathrm{pK} = -\log_{10} K_d/K_i$ on the protein–ligand interaction graph. This is
the model that **ranks** generated molecules in notebook 04.

**Input** → `graphs_affinity/` from notebook 01 §8 (PDBBind / PDBBind+ labels).
**Output** → `affinity_best.pt`.

### Why a separate network from the generator
Different data and different inductive bias. The generator trains on ~10⁵ unlabelled BioLiP
complexes and only needs plausible chemistry in a cavity. Ranking needs the ~2·10⁴ labelled
PDBBind complexes and a scalar output, so the right symmetry is **invariance**, not
equivariance — the same `EGCL` layer with coordinate updates switched off, since it consumes
only interatomic distances.

Four edge families carry different physics and get separate projections: covalent ligand bonds,
through-space ligand contacts, pocket contacts, and ligand–pocket contacts (the interaction
term the label actually depends on).

### Read the numbers honestly
A CASF-2016 core-set Pearson *r* of 0.7–0.8 is where good published scoring functions sit.
Much above that on a random split usually means leakage, which is why notebook 01 splits by
PDB entry and forces the core set into test.
"""),
        code(BOOTSTRAP),
        code(DEPS),
        code(IMPORTS),
        code(r'''
AFF_DIR = next(
    (p for p in [Path(g) for g in glob.glob("/kaggle/input/*/graphs_affinity")] + [WORK / "graphs_affinity"]
     if (p / "train_manifest.json").exists()),
    None,
)
assert AFF_DIR is not None, (
    "No graphs_affinity/ found. Run notebook 01 section 8 with PDBBind attached — "
    "the BioLiP set has no affinity labels."
)
print("affinity graphs:", AFF_DIR)

# These sets are small enough to hold in RAM, which makes the epochs much faster.
train_list = dataset.load_all(AFF_DIR, "train")
val_list = dataset.load_all(AFF_DIR, "val")
test_list = dataset.load_all(AFF_DIR, "test")


def labelled(items):
    return [d for d in items if torch.isfinite(d.y).all()]


train_list, val_list, test_list = labelled(train_list), labelled(val_list), labelled(test_list)
y_train = np.array([float(d.y) for d in train_list])
print(f"train {len(train_list):,} | val {len(val_list):,} | test {len(test_list):,}")
print(f"pK: mean {y_train.mean():.2f}  std {y_train.std():.2f}  range [{y_train.min():.2f}, {y_train.max():.2f}]")
'''),
        code(r'''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(y_train, bins=50, color="#3b82f6"); ax[0].set_title("train pK"); ax[0].set_xlabel("pK")
ax[1].scatter([d.lig_pos.shape[0] for d in train_list], y_train, s=3, alpha=0.25)
ax[1].set_xlabel("ligand heavy atoms"); ax[1].set_ylabel("pK")
ax[1].set_title("affinity vs ligand size (the trivial baseline)")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

from scipy.stats import pearsonr
r_size = pearsonr([d.lig_pos.shape[0] for d in train_list], y_train)[0]
print(f"size-only Pearson r = {r_size:.3f}  <- the model must clearly beat this")
'''),
        md(r"""
## Configuration

Labels are standardised by the training mean/std inside the model, which is worth a few epochs
of convergence on its own.
"""),
        code(r'''
ACFG = affinity.AffinityConfig(
    hidden=128,
    num_layers=4,
    edge_embed_dim=32,
    ligand_contact_cutoff=5.0,
    rbf_cutoff=8.0,
    dropout=0.1,
    label_mean=float(y_train.mean()),
    label_std=float(y_train.std()),
)

ATRAIN = dict(
    epochs=150, batch_size=24, lr=3e-4, weight_decay=1e-6,
    grad_clip=5.0, warmup_steps=300, huber_delta=1.0,
    patience=25, num_workers=2, session_hours=8.5,
)

ABEST = WORK / "affinity_best.pt"
ALAST = WORK / "affinity_last.pt"
print(ACFG)
'''),
        code(r'''
amodel = affinity.AffinityNet(ACFG).to(device)
print(f"{count_parameters(amodel):,} trainable parameters")

atrain_loader = dataset.make_loader(train_list, batch_size=ATRAIN["batch_size"], shuffle=True,
                                    num_workers=ATRAIN["num_workers"], drop_last=True)
aval_loader = dataset.make_loader(val_list, batch_size=ATRAIN["batch_size"], shuffle=False,
                                  num_workers=ATRAIN["num_workers"])

aopt = torch.optim.AdamW(amodel.parameters(), lr=ATRAIN["lr"], weight_decay=ATRAIN["weight_decay"])
asched = torch.optim.lr_scheduler.LambdaLR(
    aopt, warmup_cosine_lambda(ATRAIN["warmup_steps"], ATRAIN["epochs"] * max(len(atrain_loader), 1))
)
aema = EMA(amodel, decay=0.995)
ahistory = History(WORK / "affinity_history.jsonl")
abudget = TimeBudget(limit_s=ATRAIN["session_hours"] * 3600)


@torch.no_grad()
def evaluate_affinity(net, items, batch_size=32):
    pred = affinity.predict_batched(net, items, device=str(device), batch_size=batch_size).numpy()
    target = np.array([float(d.y) for d in items])
    return affinity.regression_metrics(pred, target), pred, target
'''),
        md("## Train"),
        code(r'''
best_rmse, bad_epochs = float("inf"), 0
for epoch in range(ATRAIN["epochs"]):
    amodel.train()
    total, nb = 0.0, 0
    for batch in atrain_loader:
        out = amodel.loss(batch.to(device), huber_delta=ATRAIN["huber_delta"])
        aopt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(amodel.parameters(), ATRAIN["grad_clip"])
        aopt.step(); asched.step(); aema.update(amodel)
        total += float(out["loss"]); nb += 1

    aema.apply_to(amodel)
    vm, _, _ = evaluate_affinity(amodel, val_list)
    aema.restore(amodel)
    ahistory.log(step=epoch, epoch=epoch, train_loss=total / max(nb, 1), **{f"val_{k}": v for k, v in vm.items()})

    flag = ""
    if vm["rmse"] < best_rmse:
        best_rmse, bad_epochs = vm["rmse"], 0
        save_checkpoint(ABEST, amodel, aopt, asched, aema, epoch, epoch,
                        config=amodel.config_dict(), extra={"val": vm})
        flag = "  <- best"
    else:
        bad_epochs += 1
    save_checkpoint(ALAST, amodel, aopt, asched, aema, epoch, epoch, config=amodel.config_dict())

    if epoch % 5 == 0 or flag:
        print(f"e{epoch:>3} train {total/max(nb,1):.4f} | val RMSE {vm['rmse']:.3f} "
              f"MAE {vm['mae']:.3f} r {vm['pearson']:.3f} rho {vm['spearman']:.3f}{flag}", flush=True)

    if bad_epochs >= ATRAIN["patience"]:
        print(f"\nearly stop: no val improvement for {ATRAIN['patience']} epochs")
        break
    if abudget.expired():
        print(f"\ntime budget reached ({abudget})")
        break

print(f"\nbest val RMSE {best_rmse:.3f}")
'''),
        code(r'''
adf = ahistory.to_frame()
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].plot(adf.epoch, adf.train_loss, label="train (huber)")
ax[0].plot(adf.epoch, adf.val_rmse, label="val RMSE"); ax[0].set_title("loss")
ax[1].plot(adf.epoch, adf.val_pearson, label="Pearson r")
ax[1].plot(adf.epoch, adf.val_spearman, label="Spearman rho")
ax[1].axhline(r_size, color="crimson", ls="--", label="size-only baseline")
ax[1].set_title("val correlation")
ax[2].plot(adf.epoch, adf.val_mae); ax[2].set_title("val MAE (pK units)")
for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),
        md(r"""
## Test-set evaluation

`sd` is the standard deviation of the residuals, reported alongside RMSE in the PDBBind/CASF
literature. If the CASF-2016 core set was available it is inside this test split.
"""),
        code(r'''
load_checkpoint(ABEST, amodel, map_location=device)
amodel = amodel.to(device)

for name, items in [("val", val_list), ("test", test_list)]:
    m, pred, target = evaluate_affinity(amodel, items)
    print(f"{name:>5}: " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in m.items()))
    if name == "test":
        test_pred, test_target = pred, target

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
lim = [min(test_target.min(), test_pred.min()) - 0.5, max(test_target.max(), test_pred.max()) + 0.5]
ax[0].scatter(test_target, test_pred, s=10, alpha=0.4)
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("experimental pK"); ax[0].set_ylabel("predicted pK")
mt = affinity.regression_metrics(test_pred, test_target)
ax[0].set_title(f"test: r={mt['pearson']:.3f}  RMSE={mt['rmse']:.2f}")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[1].hist(test_pred - test_target, bins=40, color="#f59e0b")
ax[1].set_xlabel("predicted - experimental"); ax[1].set_title("residuals")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

(WORK / "affinity_test_metrics.json").write_text(json.dumps(mt, indent=2))
print(json.dumps(mt, indent=2))
'''),
        md(r"""
---
**Next:** `04_generate_and_rank.ipynb` — attach this notebook's output together with
notebook 02's for the full pipeline.
"""),
    ]
    notebook(cells, "03_train_binding_affinity.ipynb")


# ==========================================================================
# 04 — generate and rank
# ==========================================================================
def nb04() -> None:
    cells = [
        md(r"""
# 04 · Protein in, ranked molecules out

The full inference pipeline:

```
protein PDB
   └─ detect binding sites        LIGSITE (builtin) / fpocket / P2Rank
        └─ pocket graph           whole residues within 8 Å, pocket-centred
             └─ diffusion sample  N molecules per pocket, size from the prior
                  └─ bond perception   RDKit DetermineBonds, geometric fallback
                       └─ score        affinity pK + QED + SA + clash/buriedness
                            └─ rank    CSV + SDF in the crystal frame
```

**Inputs** → notebook 02's output (generator), notebook 03's (ranker), notebook 01's
(graphs + library).

### On binding-site detectors
`method="auto"` tries **P2Rank**, then **fpocket**, then the builtin **LIGSITE**, and reports
which one ran in `Pocket.source`. The builtin needs no internet and no JVM — which matters,
because a Kaggle notebook that writes a dataset output usually has networking off. Measured on
30 held-out BioLiP complexes (a hit = the cavity covers ≥30 % of the true ligand's atoms
within 3 Å): **76.7 % top-1, 86.7 % top-3** at ~1 s per protein.

**DeepSite** is deliberately not wrapped: it is a hosted PlayMolecule web service with no
redistributable local model, so it cannot be a dependency of an offline pipeline. Feed its
predictions in as explicit centres if you want to use it.
"""),
        code(BOOTSTRAP),
        code(DEPS),
        code(IMPORTS),
        md(r"""
## 1 · Optional · install a stronger pocket finder

Skip this and the builtin detector is used. Both need internet enabled in the session.
"""),
        code(r'''
# fpocket (fast, ~2 min to install)
# !apt-get -qq update && apt-get -qq install -y fpocket && fpocket --version

# P2Rank (most accurate; needs a JRE)
# !apt-get -qq update && apt-get -qq install -y default-jre > /dev/null
# !wget -q https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz
# !tar -xzf p2rank_2.4.2.tar.gz -C /kaggle/working
# P2RANK = "/kaggle/working/p2rank_2.4.2/prank"

import shutil
P2RANK = globals().get("P2RANK")
print("fpocket:", shutil.which("fpocket") or "not installed")
print("p2rank :", P2RANK or "not installed")
POCKET_METHOD = "auto"     # "auto" | "builtin" | "fpocket" | "p2rank"
'''),
        md("## 2 · Load the trained models"),
        code(r'''
def locate(filename, extra=()):
    patterns = [
        f"/kaggle/input/*/{filename}", f"/kaggle/input/*/*/{filename}",
        f"/kaggle/input/**/{filename}", str(WORK / "**" / filename),
        str(WORK / filename), *extra,
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0])
    return None


diff_path = locate("diffusion_best.pt") or locate("diffusion_final.pt") or locate("diffusion_last.pt")
aff_path = locate("affinity_best.pt") or locate("affinity_last.pt")
assert diff_path, "no diffusion checkpoint found — attach notebook 02's output"
print("generator:", diff_path)
print("ranker   :", aff_path or "NONE (molecules will be ranked without predicted affinity)")

dck = load_checkpoint(diff_path, map_location="cpu")
gen = diffusion.PocketLigandDiffusion(diffusion.DiffusionConfig(**dck["config"]))
# EMA weights generate visibly cleaner geometry than the raw ones at the same step.
if "ema" in dck:
    state = {k: v for k, v in dck["ema"]["shadow"].items()}
    gen.load_state_dict({**dck["model"], **state}, strict=False)
    print("loaded EMA weights")
else:
    gen.load_state_dict(dck["model"])
gen = gen.to(device).eval()

ranker = None
if aff_path:
    ack = load_checkpoint(aff_path, map_location="cpu")
    ranker = affinity.AffinityNet(affinity.AffinityConfig(**ack["config"]))
    ranker.load_state_dict(ack["model"])
    ranker = ranker.to(device).eval()

build = dck.get("extra", {}).get("meta", {}).get("build_config", {})
POCKET_RADIUS = build.get("pocket_radius", 8.0)
MAX_POCKET_ATOMS = build.get("max_pocket_atoms", 600)
POCKET_CUTOFF = build.get("pocket_cutoff", 4.5)
print(f"\ngenerator trained with pocket_radius={POCKET_RADIUS} max_pocket_atoms={MAX_POCKET_ATOMS}")
print(f"coord_scale={gen.cfg.coord_scale:.3f}  timesteps={gen.cfg.timesteps}")

prior_path = locate("size_prior.json", extra=["/kaggle/input/*/graphs/size_prior.json"])
size_prior = dataset.SizePrior.load(prior_path) if prior_path else dataset.SizePrior()
print("size prior:", prior_path or "default")
'''),
        md(r"""
## 3 · Pick a target protein

Either point at a local/attached PDB, or fetch one from RCSB (needs internet).
"""),
        code(r'''
TARGET = "1b7z"          # RCSB id, or set TARGET_PDB to a path
TARGET_PDB = None

if TARGET_PDB is None:
    candidates = sorted(glob.glob(f"/kaggle/input/**/{TARGET}*.pdb", recursive=True))
    if candidates:
        TARGET_PDB = candidates[0]
    else:
        import urllib.request
        TARGET_PDB = str(WORK / f"{TARGET}.pdb")
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{TARGET.upper()}.pdb", TARGET_PDB)
        print("downloaded from RCSB")

TARGET_PDB = str(TARGET_PDB)
receptor = pdbio.parse_pdb(TARGET_PDB)
print(f"{TARGET_PDB}\n{len(receptor):,} heavy atoms | "
      f"{len(np.unique(receptor.residue_group_ids())):,} residues | "
      f"chains {sorted(set(receptor.chain))}")

# A crystallised ligand, when present, gives a free check on the detector.
het = receptor[receptor.is_hetatm]
known_ligand = None
for resname in pd.Series(het.resname).value_counts().index:
    sub = het[het.resname == resname]
    keep, _ = chem.is_interesting_ligand(resname, len(sub), sub.element, 6, 64)
    if keep:
        known_ligand = sub
        print(f"reference ligand in the file: {resname} ({len(sub)} atoms)")
        break
'''),
        md("## 4 · Detect binding sites"),
        code(r'''
t0 = time.time()
found = pockets.detect_pockets(
    TARGET_PDB, method=POCKET_METHOD, receptor=receptor, max_pockets=8,
    **({"prank": P2RANK} if POCKET_METHOD == "p2rank" and P2RANK else {}),
)
print(f"{len(found)} pockets in {time.time() - t0:.1f}s via '{found[0].source if found else '-'}'\n")
display(pd.DataFrame([p.summary() for p in found]))

if known_ligand is not None:
    det = pockets.score_detection(found, known_ligand.pos.mean(0), ligand_pos=known_ligand.pos)
    print("\ncheck against the crystal ligand:")
    print(f"  top-1 covers the ligand : {det['overlap_top1']}")
    print(f"  top-3 covers the ligand : {det['overlap_top3']}  (first hit at rank {det['overlap_rank']})")
    print(f"  centre distance, best of top 3: {det['dcc']:.2f} A")
'''),
        md(r"""
## 5 · Build pocket graphs

The detector's **point cloud** is handed to the encoder, not just the centre — an elongated
cleft then selects the residues that actually line it. The encoding is identical to training,
which is what keeps the model in distribution.
"""),
        code(r'''
TOP_POCKETS = 3

pocket_samples = []
for p in found[:TOP_POCKETS]:
    ref = p.points if p.points is not None else p.center.reshape(1, 3)
    sample = graphs.encode_pocket_only(
        receptor, reference_pos=ref, complex_id=f"{TARGET}_p{p.rank}",
        pocket_radius=POCKET_RADIUS, max_pocket_atoms=MAX_POCKET_ATOMS,
    )
    if sample is None:
        print(f"pocket {p.rank}: too few atoms, skipped")
        continue
    pocket_samples.append((p, sample))
    print(f"pocket {p.rank}: {len(sample['poc_pos']):>4} atoms, "
          f"{sample['n_poc_res']:>3} residues, "
          f"size prior -> {[size_prior.sample(sample['n_poc_res']) for _ in range(5)]}")

assert pocket_samples, "no usable pocket"
'''),
        md(r"""
## 6 · Generate

`eta=1.0` is ancestral DDPM sampling; `eta=0.0` is deterministic DDIM, which is sharper but
loses diversity — and diversity is the point when you are generating a candidate library.
`num_steps` below `cfg.timesteps` strides the schedule and trades quality for speed.
"""),
        code(r'''
N_PER_POCKET = 40
SAMPLE_STEPS = 300
ETA = 1.0
CHUNK = 10                 # molecules per forward batch

rng = np.random.default_rng(0)
all_records, provenance = [], []

for p, sample in pocket_samples:
    graph = graphs.featurize(sample, pocket_cutoff=POCKET_CUTOFF)
    poc_pos = graph.poc_pos.numpy()
    poc_elements = sample.get("poc_elements_str")
    made = 0
    t0 = time.time()

    while made < N_PER_POCKET:
        k = min(CHUNK, N_PER_POCKET - made)
        batch = diffusion.build_pocket_batch([graph] * k, device=device)
        n_atoms = [size_prior.sample(sample["n_poc_res"], rng) for _ in range(k)]
        out = gen.sample(batch, n_atoms=n_atoms, num_steps=SAMPLE_STEPS, eta=ETA)

        for mol in gen.to_molecules(out):
            rec = evaluate.evaluate_molecule(
                mol["elements"], mol["pos"], poc_pos=poc_pos, poc_elements=poc_elements,
                index=len(all_records), pocket_id=f"p{p.rank}",
            )
            all_records.append(rec)
            provenance.append((p, sample, mol))
        made += k

    print(f"pocket {p.rank}: {made} molecules in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / made:.1f}s each)", flush=True)

print(f"\n{len(all_records)} molecules generated")
'''),
        md(r"""
## 7 · Score with the affinity model

Each generated molecule is inserted back into its pocket graph (`attach_ligand`) and scored.
Bond perception runs on the generated geometry, so a molecule that fails to sanitise still
gets geometry and clash numbers — it simply carries the invalid penalty into the ranking.
"""),
        code(r'''
if ranker is not None:
    complexes, keep_idx = [], []
    for i, (rec, (p, sample, mol)) in enumerate(zip(all_records, provenance)):
        c = graphs.attach_ligand(sample, mol["elements"], mol["pos"], pocket_cutoff=POCKET_CUTOFF)
        if c is not None:
            complexes.append(c)
            keep_idx.append(i)
    preds = affinity.predict_batched(ranker, complexes, device=str(device), batch_size=16).numpy()
    for i, pk in zip(keep_idx, preds):
        all_records[i].pred_affinity = float(pk)
    print(f"scored {len(preds)} molecules | predicted pK "
          f"mean {preds.mean():.2f} range [{preds.min():.2f}, {preds.max():.2f}]")
else:
    print("no ranker loaded — ranking on chemistry and geometry only")
'''),
        md(r"""
## 8 · Rank

The composite score is affinity-dominated with QED and synthetic accessibility as
regularisers and hard geometry failures as penalties. Maximising predicted affinity alone
drifts towards large greasy molecules that the affinity model happens to like — which is the
standard failure mode of this kind of pipeline, not a hypothetical one. Adjust `RankWeights`
to match what you are actually screening for.
"""),
        code(r'''
weights = evaluate.RankWeights(
    affinity=1.0, qed=0.4, sa=0.3, buriedness=0.3,
    clash_penalty=0.5, fragment_penalty=1.0, invalid_penalty=3.0,
)
ranked = evaluate.rank_molecules(all_records, weights)

cols = [c for c in ["pocket_id", "score", "pred_affinity", "qed", "sa", "n_atoms",
                    "n_fragments", "n_clash", "buriedness", "mw", "logp", "smiles"]
        if c in ranked.columns]
display(ranked[cols].head(25))

print("\nset-level metrics:")
print(json.dumps(evaluate.set_metrics(all_records), indent=2))
print("\nper pocket:")
display(ranked.groupby("pocket_id").agg(
    n=("score", "size"), best_score=("score", "max"),
    mean_affinity=("pred_affinity", "mean"), validity=("valid", "mean"),
    clash_free=("n_clash", lambda s: (s == 0).mean()),
).round(3))
'''),
        code(r'''
fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
ok = ranked[ranked.valid] if "valid" in ranked else ranked
ax[0].hist(ranked.score, bins=30, color="#3b82f6"); ax[0].set_title("composite score")
if "pred_affinity" in ranked:
    ax[1].hist(ranked.pred_affinity.dropna(), bins=30, color="#10b981")
    ax[1].set_title("predicted pK")
if len(ok) and "qed" in ok:
    ax[2].scatter(ok.qed, ok.pred_affinity, s=12, alpha=0.5)
    ax[2].set_xlabel("QED"); ax[2].set_ylabel("predicted pK")
    ax[2].set_title("drug-likeness vs affinity")
ax[3].hist(ranked.n_atoms, bins=20, color="#8b5cf6"); ax[3].set_title("heavy atoms")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),
        md(r"""
## 9 · Export

SDF poses are shifted back into the receptor's crystal frame, so they open directly next to
the original PDB in PyMOL or ChimeraX.
"""),
        code(r'''
ranked.to_csv(WORK / f"{TARGET}_ranked.csv", index=False)

by_index = {r.index: r for r in all_records}
prov_by_index = {r.index: pr for r, pr in zip(all_records, provenance)}

written = 0
for pid, group in ranked.groupby("pocket_id"):
    recs, offset = [], None
    for idx in group.head(20)["index"]:
        rec = by_index[int(idx)]
        recs.append(rec)
        offset = prov_by_index[int(idx)][1]["center"]
    if recs:
        written += evaluate.records_to_sdf(
            recs, WORK / f"{TARGET}_{pid}_top.sdf", offset=np.asarray(offset).reshape(3)
        )

print(f"wrote {TARGET}_ranked.csv and {written} poses across {ranked.pocket_id.nunique()} SDF files")
print("\nin PyMOL:")
print(f"  load {Path(TARGET_PDB).name}")
print(f"  load {TARGET}_p1_top.sdf")
'''),
        md(r"""
## 10 · Optional · benchmark the detector on held-out complexes

Measures the *first* stage of the pipeline on the notebook-01 test split, where the true site
is known. Worth running once: if detection misses the site, nothing downstream can recover.
"""),
        code(r'''
BENCH_N = 25

try:
    biolip = sources.BioLiPSource()
    bench = biolip.specs(dedup=True, limit=3000)
    rng_b = np.random.default_rng(0)
    rows = []
    for spec in [bench[i] for i in rng_b.choice(len(bench), size=min(BENCH_N * 4, len(bench)), replace=False)]:
        rec = pdbio.parse_pdb(spec.receptor_path).heavy()
        lig = pdbio.parse_pdb(spec.ligand_path).heavy()
        keep, _ = chem.is_interesting_ligand(spec.ligand_name, len(lig), lig.element, 10, 50)
        if not keep or not (300 < len(rec) < 3000):
            continue
        pk = pockets.detect_pockets(str(spec.receptor_path), method=POCKET_METHOD,
                                    receptor=rec, max_pockets=10)
        s = pockets.score_detection(pk, lig.pos.mean(0), ligand_pos=lig.pos)
        rows.append({"id": spec.complex_id, "n_pockets": len(pk),
                     "top1": s["overlap_top1"], "top3": s["overlap_top3"], "dcc3": s["dcc"]})
        if len(rows) >= BENCH_N:
            break

    bdf = pd.DataFrame(rows)
    print(f"n={len(bdf)}  top-1 {bdf.top1.mean():.1%}  top-3 {bdf.top3.mean():.1%}  "
          f"median DCC(top3) {bdf.dcc3.median():.2f} A")
    display(bdf.head(15))
except FileNotFoundError:
    print("BioLiP source not attached — skipping the detector benchmark")
'''),
        md(r"""
---
## Where to push this next

- **Train longer.** Validity is the metric that responds most to epochs. Resume notebook 02 in
  a fresh session via `RESUME`.
- **Scale the generator.** `hidden=256, num_layers=9` is the usual next step once the pipeline
  is known-good.
- **Close the loop.** `plan.md`'s "evaluate → improve → loop" is inpainting: keep the
  best-scoring fragment fixed and re-noise the rest. `sample()` already takes the pocket as
  frozen context, so fixing a ligand sub-graph is the same mechanism applied to ligand atoms.
- **Harder validation.** Re-dock the generated poses (smina/gnina) and compare the docking
  score against the ranker's prediction; a learned scorer that disagrees with docking on its
  own top picks is not ready to screen with.
- **Categorical types.** Element types use the EDM continuous relaxation here. A D3PM-style
  discrete diffusion on types usually improves validity.
"""),
    ]
    notebook(cells, "04_generate_and_rank.ipynb")


if __name__ == "__main__":
    nb01()
    nb02()
    nb03()
    nb04()
