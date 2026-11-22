# Design notes

Why the pipeline is built the way it is. The reasoning matters more than the code —
most of these decisions have a plausible-looking alternative that quietly fails.

---

## 0 · EGNN, not DiffDock

Both get mentioned in the same breath, but they solve different problems:

| | DiffDock | this pipeline |
|---|---|---|
| input | a protein **and a known ligand** | a protein alone |
| output | the ligand's **pose** (translation, rotation, torsions) | a **new molecule**: atoms, elements, coordinates |
| search space | ~6 + *n*<sub>torsions</sub> dimensions | 4·*n*<sub>atoms</sub> dimensions |

DiffDock diffuses over a small, fixed set of pose parameters for a molecule that already
exists. It cannot invent one — there is no "which atoms" variable in its state space. For
*de novo* generation the diffusion has to act on the atoms themselves, which is what an
E(n)-equivariant GNN over a point cloud gives you. Hence EGNN.

DiffDock is still useful here, just further down the pipeline: re-dock the generated molecules
and check that its poses agree with the generated ones, and that its confidence agrees with
the affinity model's ranking. That is in §11 as validation, not generation.

---

## 1 · Two models, not one

| | generator | ranker |
|---|---|---|
| learns | `p(ligand \| pocket)` | `pK = f(complex)` |
| data | BioLiP, ~10⁵ complexes, **unlabelled** | PDBBind, ~2·10⁴, **pK labels** |
| symmetry | **equivariant** — output is a vector field on atoms | **invariant** — output is a scalar |
| notebook | 02 | 03 |

A single network cannot have both. Affinity is a scalar, so it must be invariant under
rotation; a denoiser predicts per-atom displacements, so it must be equivariant. And the
useful data differs by an order of magnitude in size: the generator can use every BioLiP
complex, the ranker only the labelled ones.

---

## 2 · The graph representation

One complex → **two node sets**, **three edge sets**:

| | nodes | features |
|---|---|---|
| ligand | heavy atoms | element one-hot (11) — *that is all the generator may see* |
| pocket | heavy atoms of whole residues within 8 Å | 40-d: element, residue type, backbone role, charge/hydropathy/mass/polarity/aromaticity/H-bond counts |

| edges | built from | attributes |
|---|---|---|
| ligand bonds | perceived geometry, or the SDF/MOL2 bond block when available | bond-order one-hot + distance RBF |
| pocket contacts | KD-tree radius graph, 4.5 Å | covalent flag + distance RBF |
| ligand ↔ pocket | radius graph, 6 Å (stored) / 8 Å (rebuilt at sample time) | distance RBF |

### What each model may look at

This split is load-bearing, not bookkeeping. **At sampling time the ligand does not exist
yet.** Anything bond-derived — degree, ring membership, bond-order sums — is unavailable, so
feeding it during training would teach the model to rely on information it will never have.
Hence:

- the **generator** sees only noisy coordinates and noisy element types, and rebuilds its
  ligand–ligand and ligand–pocket edges from the *current noisy* coordinates at every
  denoising step;
- the **ranker** gets the ligand as a given, so bonds, degrees and the stored contact edges
  are all legitimate inputs (`lig_extra`).

### Why whole residues

Clipping a side chain mid-way leaves a residue whose chemistry no longer matches its residue
type, and the residue-level features (charge, hydropathy) then describe something that is not
there. Selecting whole residues costs a few more atoms and keeps the features honest.

---

## 3 · Equivariance, and why it is asserted rather than assumed

The E(n)-GNN layer (Satorras et al. 2021) updates coordinates as

$$x_i \leftarrow x_i + \sum_{j \in \mathcal N(i)} \frac{x_i - x_j}{d_{ij} + 1}\, \phi_x(m_{ij}),
\qquad m_{ij} = \phi_e\big(h_i, h_j, \mathrm{rbf}(d_{ij}), a_{ij}\big)$$

Every coordinate update is a sum of *relative* position vectors weighted by *scalar* messages,
and the messages depend on coordinates only through distances. So rotating the input rotates
the output by the same matrix, exactly, and translating it changes nothing. The network
returns the **displacement** `x_final − x_input`, which is the equivariant quantity; absolute
positions would also carry the pocket's frame.

A broken symmetry is completely silent — the model still trains, just badly. So
`egnn.check_equivariance` is an `assert` in notebook 02, not a comment. It rotates a real
batch and checks that `eps_x` rotates with it (~10⁻⁶) and that the type prediction does not
move at all.

Two implementation details that are not optional:

- **`gain=0.001` on the final coordinate layer.** Summed over ~30 neighbours and 6 layers, a
  unit-gain initialisation produces displacements of tens of Ångström on step one.
- **the `1/(d+1)` normalisation.** Bounds each edge's contribution; without it close contacts
  dominate and training diverges.

---

## 4 · Pocket conditioning, and the scaling trap

Pocket atoms join message passing but their **coordinates are frozen**, and preprocessing
centres the complex on the pocket centroid (DiffSBDD/TargetDiff's "fixed context" scheme).
Translation is therefore handled by the centring, so no zero-centre-of-mass subspace
projection is needed and every radius cutoff keeps its physical meaning in Ångström.

### The scaling trap

A DDPM's terminal prior is `N(0, I)` — **unit** variance. Ligand coordinates in Ångström
relative to the pocket centroid have a standard deviation of ~3.5 Å (measured: 3.48 on this
dataset). Sampling `z_T ~ N(0, I)` in raw Ångström would start every molecule as a 1 Å blob at
the pocket centre, and the reverse process would have to transport mass it was never trained
to transport.

So the diffusion variable is `z = x / coord_scale`, while the **network is still fed
`z_t · coord_scale` in Ångström**. Multiplying by a scalar is equivariant, so the symmetry
argument is untouched, and the pocket plus all cutoffs stay in physical units. `coord_scale`
is measured from the data in notebook 01, not guessed. Get it wrong by 2× and you generate
either a tight blob at the centre or a spray of atoms in solvent.

Atom types follow EDM: one-hot `/ type_scale` (4.0), which keeps the categorical signal alive
deeper into the noise schedule.

### x₀-prediction, not ε-prediction

This one cost a debugging session and is the single most important choice in the file.

With noise prediction the reverse mean is

$$\mu = \frac{z_t}{\alpha_{t|s}} - \frac{\sigma^2_{t|s}}{\alpha_{t|s}\sigma_t}\hat\epsilon$$

`z` is **divided** by `α_{t|s} < 1` at every step. Over the trajectory that is a cumulative
`α₀/α_T ≈ 316×` amplification, and *only a perfect* `ε̂` cancels it. In a pocket-conditioned
model the residual error does not merely degrade samples, it runs away: once an atom drifts
past the 8 Å cross-edge cutoff the pocket context disappears from its neighbourhood, the
network has nothing to condition on and predicts almost nothing, and every remaining step is
pure amplification.

Measured, on a model overfit to 8 complexes until the loss had fallen 78 % (2.02 → 0.45):

| | centroid offset from the true ligand |
|---|---|
| untrained | 472 Å |
| ε-parameterised, trained | **331 Å** |

The loss curve looked fine. The samples were in a different part of the universe.

Predicting `x₀` uses the algebraically equivalent but numerically stable half of the same
posterior:

$$\mu = c_t z_t + c_0 \hat x_0, \qquad
c_t = \frac{\alpha_{t|s}\sigma_s^2}{\sigma_t^2},\quad
c_0 = \frac{\alpha_s\sigma^2_{t|s}}{\sigma_t^2},\qquad c_t + c_0 \approx 1$$

Both coefficients are non-negative and sum to ~1, so the update is an **interpolation** —
nothing is amplified, and a useless prediction collapses samples toward the pocket centroid,
where the context still reaches them and the model can recover. This is what TargetDiff does
for this task. `parameterization="eps"` remains available for comparison.

`clip_x0` (4.0 normalised units ≈ 14 Å) bounds the predicted `x₀`; ligand atoms sit within
~2–3 units of the pocket centroid, so it only ever truncates a diverging prediction. The older
`sample(clamp_norm=...)` rail is now off by default and only matters under `"eps"`.

### The same experiment after the switch

Identical setup — 8 complexes, 800 steps, sample 3 molecules per memorised pocket:

| | ε-parameterised | x₀-parameterised |
|---|---|---|
| centroid offset, trained | 331 Å | **1.59 Å** |
| centroid offset, untrained | 472 Å | 3.5 Å |
| radius of gyration (true 3.59 Å) | — | 2.71 Å |
| closest approach to protein | 307 Å | 2.62 Å |
| buriedness | 0.00 | 0.50 |

A 200-fold improvement from a change of parameterisation, with the architecture, schedule and
data untouched. The trained model places molecules **1.6 Å** from the crystal ligand's centroid
at a physically sensible 2.6 Å contact distance, half the atoms buried in the cavity. The
remaining gaps — slightly compact radius of gyration, ~1.6 mild clashes per molecule — are what
800 steps on a 323 k-parameter model looks like, and are what training longer fixes.

**Notebook 02 asserts this**, comparing generated centroids against the crystal ligands' and
failing above 25 Å — because the loss curve will not tell you.

---

## 5 · Deciding how many atoms to place

A pocket-conditioned generator must choose the ligand size *before* denoising starts. Using
the dataset mean gives every cavity the same molecule size, which is wrong in both directions.
`dataset.SizePrior` is a histogram of `p(n_atoms | pocket residues)`, sampled per molecule, so
sizes stay commensurate with the cavity. It falls back to the marginal for sparse bins.

---

## 6 · Binding-site detection

Three backends, one `list[Pocket]` interface, tried in order by `method="auto"`:

| backend | needs | measured (30 held-out BioLiP complexes) |
|---|---|---|
| **P2Rank** | JRE + download | best in published benchmarks |
| **fpocket** | apt/conda | good |
| **builtin LIGSITE** | nothing | **76.7 % top-1, 86.7 % top-3**, ~1 s/protein |

The builtin matters because **a Kaggle notebook that writes a dataset output usually has
networking off**, and a pipeline whose first stage needs the internet is not an offline
pipeline. It is a vectorised LIGSITE: build an occupancy grid, count protein–solvent–protein
events along 7 lattice directions per free grid point (shift-doubling, so `log₂` passes rather
than one per step), connected-component the high-PSP points, rank the clusters.

Defaults came from a grid search, not from the paper:

| setting | top-1 | top-3 |
|---|---|---|
| spacing 0.8, PSP ≥ 4, rank by fully-enclosed points | **0.767** | **0.867** |
| spacing 1.0, PSP ≥ 4 | 0.700 | 0.867 |
| spacing 0.8, PSP ≥ 5 | 0.667 | 0.767 |

Ranking by the count of *fully* enclosed points (PSP = 7/7) beats ranking by cluster volume,
because volume's failure mode is preferring large shallow surface grooves to the real cavity.
`spacing=1.0` halves the runtime for ~7 points of top-1 — the right trade when screening
thousands of structures.

**DeepSite is not wrapped.** It is a hosted PlayMolecule web service with no redistributable
local model, so it cannot be a dependency here. Feed its predictions in as explicit centres.

### Evaluating detection

Centre-to-centre distance (DCC) is the conventional number but it is unfair to elongated
cavities, which can cover a ligand perfectly with a centroid several Ångström away. So
`pockets.overlap_hit` is the primary criterion: does the cavity cover ≥30 % of the true
ligand's atoms within 3 Å? Both are reported.

---

## 7 · Splitting, or how to avoid fooling yourself

BioLiP lists the same chemical ligand once per copy in the asymmetric unit: `1b7z_1_OXL_D` and
`1b7z_1_OXL_E` are the same molecule in two copies of the same site. A random split puts one
in train and one in test and every metric becomes a measure of memorisation.

`sources.split_by_receptor` groups on the 4-character PDB code — the minimum defensible unit —
and notebook 01 **asserts** that no code appears in two splits. `dedup=True` additionally keeps
one entry per `(assembly, ligand)`. For the affinity model, the CASF-2016 core set is forced
into test via `holdout_ids`: it is a subset of the refined set, so leaving it in train makes the
headline correlation meaningless.

A sequence-identity split (e.g. MMseqs2 at 30 %) is stricter still and is the right next step
if you publish numbers.

---

## 8 · Storage: the constraint that shaped the data layer

| form | per complex | 150 k complexes |
|---|---|---|
| featurised `ComplexData` | ~590 kB | **88 GB** |
| encoded (positions + int codes) | ~10 kB | **1.5 GB** |

Kaggle's output limit is 20 GB, so the dense form is not an option. The cost is dominated by
the `(E, 17)` float attributes over ~4 k pocket contact edges — all of it recomputable.

So `graphs.encode_complex` stores positions plus `int8` codes (element, residue, atom name,
chain, residue group) and `graphs.featurize` rebuilds the dense form in **0.8 ms**, inside the
DataLoader workers, overlapped with the GPU step. Making that fast required replacing the
per-edge python in the covalent-flag computation with a precomputed
`(residue, name, name)` boolean table — ~34 kB, built once at import.

A consequence worth knowing: `pocket_radius`, `max_pocket_atoms` and `center_mode` are baked
into the shards, while `pocket_cutoff` and `inter_cutoff` are applied at load time and can
still be changed. The manifest records all of them.

---

## 9 · Kaggle practicalities

- **Notebook 01 with accelerator = None.** Pure CPU work; a GPU session wastes quota and P100
  sessions give fewer cores.
- **Everything checkpoints every epoch, atomically** (`save_checkpoint` writes to `.tmp` then
  renames — a half-written checkpoint is worse than none). Optimiser, scheduler and EMA state
  all restore, so hitting the 9-hour wall costs one epoch, not a run.
- **`TimeBudget`** stops cleanly with ~25 min of headroom for the final checkpoint and the
  sampling demo. `build_shards` honours the same idea and keeps a partial dataset.
- **`torch.load(..., weights_only=False)`** is required for the shards — torch ≥ 2.6 defaults
  to `True`, which cannot deserialise numpy-array dicts.
- **No `torch_scatter` / `torch_cluster`.** They need wheels matched to the exact torch build
  and are the usual cause of Kaggle install failures. `torch_geometric.utils.scatter` wraps
  native `scatter_reduce`; radius graphs use scipy KD-trees at preprocessing and per-graph
  `cdist` at train time (looping over the ~16 graphs in a batch rather than building one dense
  `N_lig × N_poc` matrix, which is what keeps large pockets in memory).
- **EMA weights are what you sample with.** Diffusion sample quality is notoriously sensitive
  to this; the raw weights at the same step produce visibly worse geometry.

---

## 10 · Judging the output

`evaluate.set_metrics` returns validity, connectivity, uniqueness, novelty and clash-free
fraction **together**, because each is misleading alone:

- **Validity alone is close to meaningless.** RDKit sanitises a bag of disconnected atoms
  happily, so a barely-trained model reports 100 % validity with `connected_fraction = 0`.
  (Observed in the toy run: `Br.C.I.N.N.S` — six fragments, "valid".)
- **A model that emits one molecule every time** scores 100 % validity and ~0 % uniqueness.
- **Good SMILES is not a binder.** Atoms overlapping the protein, or floating in solvent
  outside the cavity, are checked separately (`n_clash`, `buriedness`).

Notebook 02 prints the same statistics for the **crystal ligands of the same pockets** as a
reference line. That is the comparison worth making.

### The ranking objective

```
score = 1.0·affinity + 0.4·QED + 0.3·SA + 0.3·buriedness
        − 0.5·clashes − 1.0·fragments − 3.0·invalid
```

Affinity dominates because that is the design goal, but QED and synthetic accessibility are
present as regularisers: a pipeline that maximises predicted affinity alone drifts toward
large greasy molecules that the affinity model happens to score well. That is the standard
failure mode of learned-scorer-driven generation, not a hypothetical. Tune `RankWeights` to
whatever you are actually screening for.

---

## 11 · Known limitations

- **Atom types use the EDM continuous relaxation** (one-hot diffused as a continuous vector),
  with an auxiliary cross-entropy on the predicted types to sharpen element identity. A
  D3PM-style genuinely discrete diffusion over types usually improves validity further.
- **Bond orders are perceived, never generated.** The model places atoms; connectivity comes
  from `rdDetermineBonds` or geometry. Generating bonds jointly (as MiDi/JODO do) removes a
  failure mode.
- **The pocket is rigid.** No side-chain flexibility, so induced fit is out of scope.
- **The affinity model is a scoring function, not a free-energy calculation.** CASF-2016
  Pearson *r* in the 0.7–0.8 range is where good published scoring functions sit; anything much
  higher on a random split is leakage, which is what §7 is about.
- **PDB-code splits are weaker than sequence-identity splits.** Homologous proteins with
  different codes can still straddle the split.
- **No cross-validation against docking.** Re-docking generated poses (smina/gnina) and
  comparing to the ranker's prediction is the cheapest real check; a learned scorer that
  disagrees with docking on its own top picks is not ready to screen with.

---

## 12 · Closing the loop

`plan.md`'s *evaluate → improve → loop* is **inpainting**: keep the best-scoring fragment
fixed and re-noise the rest. The mechanism already exists — `sample()` treats the pocket as
frozen context via a coordinate mask, so fixing a ligand sub-graph is the same mask applied to
ligand atoms instead. Concretely: extend `PocketLigandDiffusion.sample` with a
`fixed_mask` over ligand atoms, and at each step overwrite the fixed entries of `z_t` with the
forward-noised ground truth `α_t z_0 + σ_t ε`. That turns the generator into a scaffold
decorator and makes the improve-loop a few lines rather than a new model.
