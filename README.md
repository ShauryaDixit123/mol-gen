# molgen — structure-based molecule generation, end to end

Protein in, ranked candidate molecules out. Graph representations throughout, trained on Kaggle.

```
protein PDB
   │
   ├─ 1. binding-site detection ──────── LIGSITE (builtin) · fpocket · P2Rank
   │        └─ pocket point cloud
   │
   ├─ 2. pocket graph ────────────────── whole residues within 8 Å, pocket-centred
   │        └─ atoms + integer codes → vectorised features + 3 edge families
   │
   ├─ 3. molecule generation ─────────── E(3)-equivariant diffusion (EGNN denoiser)
   │        └─ joint diffusion of coordinates + atom types, pocket as frozen context
   │
   ├─ 4. bond perception ─────────────── RDKit DetermineBonds, geometric fallback
   │        └─ SMILES, QED, SA, geometry + clash checks
   │
   └─ 5. ranking ─────────────────────── affinity GNN (pK) + drug-likeness − penalties
            └─ ranked CSV + SDF poses in the crystal frame
```

## Layout

```
src/molgen/            the library — everything below is thin orchestration
  chem.py              element/residue vocabularies, covalent templates, bond perception
  pdbio.py             fast columnar PDB reader (~1 ms/structure), SDF/XYZ writers
  molfiles.py           SDF / MOL2 readers (real bond blocks from PDBBind)
  graphs.py            encode (compact storage) ↔ featurize (model-ready ComplexData)
  sources.py           BioLiP + PDBBind dataset indexing, leakage-free splits
  dataset.py           shard writer, lazy dataset, ligand-size prior
  egnn.py              E(n)-equivariant layer + pocket-conditioned denoiser
  diffusion.py         noise schedules, training loss, DDPM/DDIM sampling
  affinity.py          interaction-graph pK regressor
  pockets.py           LIGSITE / fpocket / P2Rank detection + detection metrics
  evaluate.py          validity, chemistry, clash, composite ranking
  train_utils.py       EMA, schedules, Kaggle time budget, atomic checkpoints

notebooks/             run these on Kaggle, in order
  01_build_pocket_ligand_graphs.ipynb
  02_train_pocket_conditioned_diffusion.ipynb
  03_train_binding_affinity.ipynb
  04_generate_and_rank.ipynb

tools/
  selftest.py          data-free end-to-end check (~4 s, 38 assertions)
  make_notebooks.py    regenerates the notebooks (edit here, not the JSON)
  pack_for_kaggle.py   zips src/molgen for upload as a Kaggle Dataset
docs/RUNBOOK.md        how to run it from scratch, step by step
docs/HOW_IT_WORKS.md   progressive walkthrough — start here
docs/PIPELINE.md       the design decisions and why they are what they are
```

## Running it on Kaggle

**One-time setup.** `python tools/pack_for_kaggle.py` → upload `dist/molgen-src.zip`
as a Kaggle Dataset named `molgen-src`. Each notebook's first cell finds it automatically.

| # | notebook | accelerator | inputs | time |
|---|---|---|---|---|
| 01 | build graph shards | **None** (CPU work — a GPU session wastes quota and gives fewer cores) | `protein-ligand-binding-site`, `molgen-src`, *optional* PDBBind | ~40 min |
| 02 | train the generator | GPU P100/T4 | output of 01 | 1–N sessions, resumable |
| 03 | train the ranker | GPU | output of 01 (needs PDBBind labels) | ~1 h |
| 04 | generate and rank | GPU | outputs of 01, 02, 03 | minutes |

Notebook 01 copies `src/molgen` into its own output, so notebooks 02–04 need only
*that* output as an input. Each training notebook resumes from its last checkpoint,
so hitting Kaggle's 9-hour wall is not a loss — point `RESUME` at `*_last.pt` and re-run.

## Data

| source | what for | labels |
|---|---|---|
| [protein-ligand-binding-site](https://www.kaggle.com/datasets/shauryadixit/protein-ligand-binding-site) (BioLiP-style, 33 k receptors / 148 k ligands) | trains the **generator** | binding residues only |
| [PDBBind+](https://www.pdbbind-plus.org.cn/download) `refined-set` + `INDEX_*` | trains the **ranker** | pK = −log₁₀(K_d/K_i) |

Both layouts are auto-detected (`sources.BioLiPSource`, `sources.PDBBindSource`).

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python tools/selftest.py              # verify the install — no datasets needed
python tools/make_notebooks.py        # regenerate notebooks after editing
```

`selftest.py` builds a synthetic protein and ligand and exercises every stage in ~4 s. It
checks *behaviour*, not imports — in particular the batch-offset correctness of
`ComplexData.__inc__` and the E(3) symmetry of the denoiser, both of which fail silently
during training rather than loudly.

`torch_scatter` / `torch_cluster` are deliberately **not** dependencies — they need wheels
matched to the exact torch build and are the usual cause of Kaggle install failures. Graph
construction uses scipy KD-trees, train-time edges use native torch ops. RDKit is optional:
without it you lose SMILES/QED/SA and nothing else.

## Measured so far

- **Storage**: a complex is stored as positions + integer codes at ~10 kB. Featurising costs
  0.8 ms and inflates it 58× to ~590 kB, so the full set is **1.5 GB** stored rather than
  88 GB — the difference between fitting Kaggle's 20 GB output limit and not.
- **Pocket detection** (builtin LIGSITE, 30 held-out BioLiP complexes, hit = cavity covers
  ≥30 % of the true ligand's atoms within 3 Å): **76.7 % top-1, 86.7 % top-3**, ~1 s/protein.
  In line with published LIGSITE numbers. P2Rank does better if you can install it.
- **Equivariance**: rotation and translation errors ~1 × 10⁻⁶ (asserted in notebook 02).
- **Learning**: overfitting 8 complexes and sampling back from the memorised pockets puts the
  generated molecules **1.59 Å** from the crystal ligand's centroid, at a 2.6 Å closest
  approach to the protein with half the atoms buried in the cavity.
- **x₀-prediction, not ε-prediction** — the single most consequential choice here. The ε form
  divides by `α_{t|s}` every step: a cumulative 316× amplification that only a *perfect* noise
  prediction cancels. Once an atom drifts past the 8 Å cross-edge cutoff the pocket context
  vanishes, the network predicts ~nothing, and the rest of the trajectory is pure
  amplification. Same experiment, same architecture, only the parameterisation changed:

  | | ε | x₀ |
  |---|---|---|
  | centroid offset, trained | 331 Å | **1.59 Å** |
  | centroid offset, untrained | 472 Å | 3.5 Å |
  | buriedness | 0.00 | 0.50 |

  **The loss curve does not show this** — it had already fallen 78 % in the ε run. So
  notebook 02 asserts on the generated centroid instead. [docs/PIPELINE.md §4](docs/PIPELINE.md).

Generative quality numbers (validity, uniqueness, novelty) depend entirely on how long you
train — notebook 02 reports them against the crystal ligands of the same pockets as a
reference, which is the comparison that means something.

## Reading the metrics

**Validity alone is close to meaningless.** RDKit happily sanitises a bag of disconnected
atoms, so a barely-trained model can report 100 % validity with `connected_fraction = 0`.
Always read validity, connectivity, uniqueness and clash-free fraction together —
`evaluate.set_metrics` returns them as a set for that reason.
