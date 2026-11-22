# Runbook — running this from scratch

Follow top to bottom. There are 25 files in the repo but you only ever **touch two things**:
one zip you upload once, and four notebooks you run in order.

```
  LOCAL (10 min, once)          KAGGLE (the actual work)
  ────────────────────          ────────────────────────
  selftest.py       ✔          01  build graph shards      CPU   ~40 min
  pack_for_kaggle.py ─zip─►     02  train the generator     GPU   9 h × N
                                03  train the ranker        GPU   ~1 h
                                04  generate and rank       GPU   ~5 min
```

---

## Stage 0 · Clear the decks

The `work/` directory in this repo holds **toy artifacts from verification runs** — a few
hundred complexes, a handful of epochs. `work/diffusion_best.pt` is *not* a trained model and
`work/1b7z_ranked.csv` is *not* a real result. They exist only to prove the chain executes.

```bash
rm -rf work
```

Nothing depends on it. Kaggle writes its own `/kaggle/working`.

---

## Stage 1 · Local, 10 minutes

You are not training locally. This step only confirms the library is sound before you spend
GPU quota.

```bash
cd "/Users/shauryadixit007/Desktop/untitled folder/untitled folder 2"
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

```bash
python tools/selftest.py
```

Expect `38 passed` in about 4 seconds. It builds a synthetic protein and ligand and exercises
every stage — no datasets needed. If this fails, stop here; nothing downstream will work.

```bash
python tools/pack_for_kaggle.py
```

Produces `dist/molgen-src.zip` (~57 kB, 13 modules). **Re-run this any time you edit
`src/molgen/`** and re-upload, or Kaggle keeps running the old code.

> Already done — the zip in `dist/` is current as of the last `src/` edit.

---

## Stage 2 · Kaggle account prerequisites

Two things that block everything and are easy to miss:

1. **Phone-verify your account.** Settings → Phone Verification. Without it you cannot enable
   notebook internet, and without internet the `ensure()` cell cannot `pip install`
   `torch_geometric` and `rdkit`. This is the single most common first failure.
2. **Check your GPU quota.** 30 h/week, resets Saturday. Notebook 02 is the only hungry one.

---

## Stage 3 · Create the datasets

### 3a · `molgen-src` — required

Kaggle → Datasets → **New Dataset** → upload `dist/molgen-src.zip` → title it exactly
`molgen-src` → Create.

Kaggle auto-extracts, so it lands at `/kaggle/input/molgen-src/src/molgen/`. Each notebook's
first cell walks `/kaggle/input/*` looking for `molgen/__init__.py` and finds it — you never
hardcode a path.

### 3b · `protein-ligand-binding-site` — required

Already yours. You'll attach it, not create it.

### 3c · PDBBind — required for notebooks 03 and 04

This one is manual and cannot be automated: [PDBBind+](https://www.pdbbind-plus.org.cn/download)
requires a login and accepting their terms.

Download and upload as a new Kaggle Dataset (call it `pdbbind-refined`):

```
refined-set/            one folder per complex: 1a1e/1a1e_protein.pdb, 1a1e_ligand.mol2 …
index/INDEX_refined_data.2020    the affinity labels
```

`sources.PDBBindSource` auto-detects this layout — keep the folder names as they ship.

**You can skip 3c for now.** Notebooks 01 and 02 don't need it, and that's the whole
generative half. Without it, notebook 04 ranks on drug-likeness and geometry only.

---

## Stage 4 · Run the notebooks

For each one: **New Notebook → File → Import Notebook →** upload the `.ipynb` → set inputs and
accelerator in the right-hand panel → **Save Version → Save & Run All (Commit)**.

Interactive "Run All" output is **not persisted**. Only a committed version produces an output
you can attach to the next notebook. This is the second most common first failure.

### 4.1 — `01_build_pocket_ligand_graphs.ipynb`

| setting | value |
|---|---|
| Accelerator | **None** |
| Internet | On |
| Inputs | `protein-ligand-binding-site`, `molgen-src`, *(optional)* `pdbbind-refined` |

**Set `MAX_COMPLEXES = 2000` for your first run.** Takes ~3 min instead of ~40 and proves the
whole chain before you commit real time. Set it back to `None` later.

CPU, not GPU, deliberately: this is PDB parsing across cores, a GPU session gives you *fewer*
cores and burns quota.

What to check in the output:
- the reject table — `ok` should be ~75–80%
- the leakage assertion passes (no PDB code in two splits)
- `TOTAL n complexes, X GB` — must be comfortably under 20 GB
- if you attached PDBBind, §8 built `graphs_affinity/` too

Output contains `graphs/`, `graphs_affinity/`, `dataset_meta.json`, `size_prior.json`, and a
copy of `src/molgen`. **Because of that copy, notebooks 02–04 need only this one input.**

### 4.2 — `02_train_pocket_conditioned_diffusion.ipynb`

| setting | value |
|---|---|
| Accelerator | **GPU P100** (or T4 ×2) |
| Internet | On |
| Inputs | output of notebook 01 |

Defaults: 500 timesteps, hidden 128, 6 layers, `epochs=200`, `batch_size=16`, `lr=2e-4`.
For a first pass drop `epochs` to 3 — you want to see the chain finish, not a good model.

Three things in the output are assertions, not decoration:
- **equivariance < 1e-3** before training starts. Failure here means a broken denoiser.
- the loss curve falls
- **sampled molecules land < 25 Å from the pocket.** This is the guard against the ε-style
  runaway. The loss curve will *not* tell you.

**Hitting the 9-hour wall is expected and fine.** Checkpoints are atomic and per-epoch. To
continue: attach *this notebook's own output* as an extra input alongside notebook 01's, and
re-commit. `RESUME` picks up `diffusion_last.pt` automatically. Repeat as many sessions as you
want; this is where almost all of the training time goes.

Sampling always uses the EMA weights — don't "fix" that.

### 4.3 — `03_train_binding_affinity.ipynb`

| setting | value |
|---|---|
| Accelerator | GPU |
| Internet | On |
| Inputs | output of notebook 01 (must include `graphs_affinity/`) |

Needs PDBBind from 3c. Early stopping with `patience=25`, so ~1 h and it stops on its own.

The number that matters: your test Pearson *r* against the **red size-only baseline** drawn on
the plot. Not above the line = the model learned "bigger binds harder" and nothing else.
Expect 0.6–0.8. Much above 0.85 on a random split means leakage, not success.

### 4.4 — `04_generate_and_rank.ipynb`

| setting | value |
|---|---|
| Accelerator | GPU |
| Internet | On (to fetch the target PDB from RCSB) |
| Inputs | outputs of 01, 02, 03 |

Set your target:

```python
TARGET = "1b7z"        # any RCSB id
TARGET_PDB = None      # or a path to your own .pdb, which overrides TARGET
```

Other knobs: `TOP_POCKETS=3`, `N_PER_POCKET=40`, `SAMPLE_STEPS=300`, `POCKET_METHOD="auto"`.

Optional first cells install fpocket/P2Rank. Skip them and the builtin LIGSITE runs — 76.7%
top-1, no install. Install P2Rank if you want better detection.

Outputs: `<target>_ranked.csv` (every metric per molecule) and `<target>_p1_top.sdf` (poses in
the original crystal frame — open directly on top of your input PDB in PyMOL).

---

## The order that actually saves you time

Do a **full dry run at toy scale first.** All four notebooks, `MAX_COMPLEXES=2000`,
`epochs=3`. Perhaps 90 minutes of wall time, almost no GPU quota, and it surfaces every
plumbing problem — a missing input, an unverified phone, a stale zip — while they're cheap to
fix.

Then scale up: `MAX_COMPLEXES=None`, re-run 01, and give notebook 02 as many 9-hour sessions
as you're willing to spend. Generation quality is almost entirely a function of that number.

```
dry run  ──►  01 full  ──►  02 × N sessions  ──►  03  ──►  04
 90 min       40 min         the real cost         1 h     5 min
```

---

## Troubleshooting

| symptom | cause |
|---|---|
| `molgen not found` in cell 1 | `molgen-src` not attached, or zip uploaded without the `src/` prefix — re-run `pack_for_kaggle.py` |
| `pip install` fails | internet off; needs phone verification |
| notebook 02 can't find `GRAPH_DIR` | you ran 01 interactively instead of committing a version |
| notebook 03 has no data | PDBBind wasn't attached when 01 ran, so §8 never built `graphs_affinity/` |
| samples > 25 Å, assertion trips | `DCFG.parameterization` was changed to `"eps"` — put it back to `"x0"` |
| `torch.load` complains | shards hold numpy dicts; `weights_only=False` is required and already set |
| output > 20 GB | `max_ligand_atoms` or `max_pocket_atoms` raised above the defaults |

Edit notebooks via `tools/make_notebooks.py`, not the `.ipynb` JSON, then re-run it.
