# How it works

A progressive walkthrough of the whole pipeline — what the problem is, why it is built
the way it is, and what happens when you actually run it. For the terse version of the
design decisions see [PIPELINE.md](PIPELINE.md); for the operational checklist see the
[README](../README.md).

---

## 1 · What we're actually trying to do

You have a protein. Somewhere on its surface is a dent — a pocket — where a small molecule
could sit. You want a molecule that fits that specific dent, snugly, and that is a real,
synthesizable, drug-like molecule.

Classically you'd screen a library: take ten million known compounds, dock each one, keep
the best. That's *search* — you can only find what someone already made.

What this generates. It starts from noise and constructs a molecule shaped to that
particular cavity. The search space isn't a catalogue; it's all of chemistry.

---

## 2 · Why this needs three separate pieces

The instinct is "train one big model." It doesn't work, for a reason worth internalising:

```
  find the pocket        make a molecule          judge it
  ──────────────        ───────────────          ─────────
  geometry problem      generative problem       regression problem
  no learning needed    needs 10⁵ structures     needs labels (only 2·10⁴ exist)
  output: a location    output: atoms + coords    output: one number
```

The third column is the killer. There are ~150,000 known protein–ligand structures but only
~20,000 with *measured binding affinity*. If generation and scoring were one model, you'd be
stuck training on 20,000 examples. Splitting them lets the generator learn
chemistry-in-a-cavity from all 150,000, and the scorer learn affinity from the labelled
subset.

There's also a symmetry argument. The generator outputs a *vector* per atom (where to move
it), so it must be **equivariant** — rotate the protein, the arrows rotate too. The scorer
outputs a *scalar*, so it must be **invariant** — rotate the protein, the affinity doesn't
change. Those are mathematically different requirements. One network can't satisfy both.

---

## 3 · Stage zero: what the raw material actually is

The Kaggle dataset is 33,000 receptor PDB files and 148,000 ligand PDB files. Each ligand
file is bare coordinates:

```
HETATM    1  N1G U2G A 130       7.874  52.898  71.696   ...   N
HETATM    2  C2G U2G A 130       6.993  53.856  72.088   ...   C
```

Note what's missing: **no bonds.** The file says "there is a nitrogen here." It does not say
what it's attached to. So the very first thing the library does is *perceive* bonds from
geometry — two atoms are bonded if they're closer than the sum of their covalent radii, with
a valence check so carbon never ends up with six neighbours. That's `chem.infer_bonds`.

Second: most of those 148,000 "ligands" aren't drug candidates. BioLiP counts sulfate ions,
glycerol, water, lone amino acids, cryoprotectants. Generating those would be pointless. The
filter drops them by name, element and size — about 22% get rejected, and notebook 01 prints
*why*, so you can see it rather than guess:

```
ok            77.5%
too_large     12.5%   (HEM, FAD, NAD — bigger than 48 atoms)
bad_element    9.2%   (iron, magnesium — outside the vocabulary)
too_small      0.8%
```

---

## 4 · Stage one: a structure becomes a graph

This is where the design starts mattering. One complex becomes **two sets of nodes and three
sets of edges**:

```
       LIGAND NODES                        POCKET NODES
       ~27 heavy atoms                     ~360 atoms (whole residues within 8 Å)
       feature: element only               feature: 40-d — element, residue type,
                (C/N/O/F/P/S/Cl/...)                 backbone role, charge,
                                                     hydropathy, H-bond counts
            │                                        │
            ├── bonds ──────┐                        ├── contacts (4.5 Å,
            │  (order +     │                        │   covalent flag + distance)
            │   distance)   │                        │
            └── cross edges ┴────────────────────────┘
                (6 Å, ligand atom ↔ pocket atom)
```

**Two decisions in there are load-bearing.**

*Whole residues, not a sphere of atoms.* If you clip a sphere you cut side chains in half,
and then the residue-level features (this is a charged arginine, this is a greasy leucine)
describe something that isn't there any more. Costs a few more atoms, keeps the features
honest.

*The ligand gets element identity and nothing else.* No bond count, no ring membership, no
degree. This looks like throwing away free information — and it's the most important
restriction in the codebase. **At generation time the ligand doesn't exist yet.** There are
no bonds to count. If you trained the generator on bond-derived features it would learn to
depend on information it can never have at inference. The scorer *does* get all of it,
because by then the molecule is real.

And everything is centred on the **pocket centroid**, not the ligand centroid. Both are
available during training; only the pocket one exists at inference. Train on the convention
you can actually use.

---

## 5 · Stage two: teaching a model to generate

### The idea, in plain terms

Take a real molecule sitting in its real pocket. Add a little random noise to every atom's
position and identity. Add more. Keep going until it's indistinguishable from random static.

Now train a network to undo one step of that. Show it a noisy molecule and ask: *what did
this look like before the noise?*

```
 t=500          t=375          t=250          t=125          t=0
  ·    ·         ·   ·          · ·· ·         ·-·-·        O=C-C-N
 ·  ·    ·      ·  ·  ·        ·· · ··        ·· ·· ·       │     │
   ·   ·         ·   ·  ·       · ·· ·        ·-· ·-·       C─────C
 pure noise     vague blobs    clusters      connectivity   molecule
 ◄──────────────────────────────────────────────────────────────────
                        the model runs this direction
```

Once it can undo one step, you can run it 500 times starting from *actual* random static,
and a molecule condenses out. That's the whole trick. The noise is different every time, so
you get a different molecule every time.

**The pocket is the prompt.** It never gets noised. It sits in the graph as fixed context —
the ligand atoms are passing messages to it at every step, so the thing condensing out of
the noise is being shaped, continuously, by the walls around it.

### Why the network has to be equivariant

A protein has no preferred orientation. If you rotate the whole complex 40° about some axis,
every physical answer rotates with it and nothing else changes.

A plain neural network on raw `(x, y, z)` doesn't know that. It would have to *learn* it from
data — effectively memorising every molecule in every orientation. At 150,000 examples you
can't afford that.

So the layer is built so the symmetry is impossible to violate. Every coordinate update looks
like:

```
                      x_i − x_j
  x_i  ←  x_i  +  Σ  ───────────  ·  φ(message_ij)
                   j   d_ij + 1
                      ↑              ↑
            a relative direction    a plain number
```

The only vectors are *differences between positions*, scaled by *scalars*. Rotate the input
and every difference rotates identically, so the output rotates identically. Translate it and
the differences don't change at all. The symmetry is structural, not learned.

That's not a claim to take on faith — notebook 02 rotates a real batch and asserts the output
rotated to within 10⁻⁶ before it will train. A broken symmetry is *silent*: the model still
trains, the loss still falls, it's just quietly much worse. So it's an assertion, not a
comment.

### The thing that nearly sank it

Textbook diffusion trains the network to predict **the noise** it should remove. The reverse
step is then:

```
  μ  =  z_t / α  −  (something) · predicted_noise
            ↑
      divides by a number less than 1 — amplifies — every single step
```

Over 500 steps that's a cumulative **316× amplification**, held in check only by the
predicted noise cancelling it exactly. For images that's fine. Here it isn't, and the failure
mode is vicious:

```
  small error → atom drifts past the 8 Å cross-edge cutoff
              → that atom can no longer see the pocket
              → network has nothing to condition on, predicts ≈ nothing
              → nothing cancels the amplification
              → remaining steps are pure ×316 runaway
```

Measured. Overfitting 8 complexes until the loss had dropped **78%**, the model placed its
molecules **331 Å** from the pocket. The loss curve looked textbook. The molecules were in a
different postcode.

The fix: train the network to predict **the clean molecule** instead of the noise. Same
posterior distribution — just the other half of the algebra:

```
  μ  =  c_t · z_t  +  c_0 · predicted_clean_molecule
         ↑              ↑
        both ≥ 0, and they sum to ≈ 1  →  an interpolation. Nothing amplifies.
```

Now a *useless* prediction collapses molecules toward the pocket centre — where the context
still reaches them and the model can recover — instead of flinging them into solvent.
Identical architecture, identical data, only this changed:

| | predict noise (ε) | predict molecule (x₀) |
|---|---|---|
| offset from the true ligand | 331 Å | **1.59 Å** |
| buriedness | 0.00 | 0.50 |

200×, from choosing which side of an equation to implement.

### One detail that's easy to miss

Before you can denoise anything you have to decide **how many atoms the molecule has**. The
noise has to be *some* shape.

Using the dataset average gives every cavity the same size molecule, which is wrong in both
directions — tiny molecules rattling around in big cavities, big ones jammed into small ones.
So the library fits `p(atoms | pocket residues)` from the training data and draws from it per
molecule. A 25-residue pocket gets a different size distribution than a 60-residue one.

---

## 6 · Stage three: teaching a model to judge

Generation gives you a molecule shaped like it belongs. It says nothing about *how tightly it
binds*.

That's the second model, trained on PDBBind's measured affinities (pK = −log₁₀ of the
dissociation constant; higher is tighter). Same equivariant layer, but with **coordinate
updates switched off** — it consumes only interatomic distances, so its output is a
rotation-invariant scalar. Exactly right for a number that shouldn't care which way you
turned the molecule.

It reads four physically distinct kinds of edge, each with its own learned projection:

```
  ligand covalent bonds      ─ what the molecule is
  ligand through-space       ─ its 3-D shape
  pocket contacts            ─ what the cavity is
  ligand ↔ pocket contacts   ─ the interaction. this is what the label depends on.
```

There's a baseline you have to beat here, and notebook 03 plots it: **ligand size alone
correlates with affinity.** Bigger molecules touch more protein and bind harder. A model
that's only learned "bigger = better" is worthless for ranking, so the size-only correlation
is drawn as a red line on the metric plot. Beat the line or you have nothing.

---

## 7 · Now the actual run: protein in, molecules out

Here's the whole thing executing, concretely.

**You hand it a PDB file.** Say 1b7z — 5,281 atoms, 689 residues.

**Find the cavities.** A grid is laid over the protein at 0.8 Å spacing. For each empty grid
point, cast rays along 7 lattice directions and count how many get blocked by protein in
*both* directions. A point deep in a cleft is enclosed along most of them; a point on a bulge
is enclosed along few. Cluster the enclosed points, rank the clusters by how many are *fully*
enclosed.

That's LIGSITE, vectorised. ~1 second per protein, no internet, no Java. **76.7% top-1, 86.7%
top-3** on a 30-complex benchmark. If you install P2Rank it uses that instead and does better
— `method="auto"` prefers it and falls back silently.

Ranking by *fully-enclosed count* rather than cluster volume was worth 13 points of top-1,
because volume's failure mode is picking a big shallow surface groove over the real buried
pocket.

**Build the pocket graph.** The detector's whole point cloud is handed over — not just the
centre — so an elongated cleft selects the residues that genuinely line it. Encoded
byte-identically to training: same radius, same features, same centring. Train and inference
must agree or the model is off-distribution and you'll never know.

**Generate.** For the top 3 pockets, 40 molecules each:

```
  draw n_atoms from the size prior         →  e.g. 31 atoms
  initialise 31 atoms as pure noise
  for t = 500 → 0:
       rebuild ligand↔ligand and ligand↔pocket edges from current noisy positions
       network predicts the clean molecule
       take one interpolation step toward it, add a little noise back
  read off final positions and elements
```

The edges being **rebuilt every single step** is the point — the molecule's neighbourhood
changes as it condenses, so the graph has to change with it.

**Perceive bonds.** You now have atoms with coordinates, which is not yet a molecule.
RDKit's `DetermineBonds` infers connectivity and bond orders; geometric perception is the
fallback. Whether this *succeeds* is itself a headline metric — that's "validity."

**Score and rank.** Each molecule is slotted back into its pocket graph and scored:

```
  score = 1.0 · predicted affinity     ← the objective
        + 0.4 · QED                    ← drug-likeness
        + 0.3 · synthetic accessibility ← can a chemist actually make it
        + 0.3 · buriedness             ← is it inside the cavity
        − 0.5 · clashes with protein
        − 1.0 · disconnected fragments
        − 3.0 · failed to be a molecule at all
```

QED and synthesizability aren't decoration. Maximising predicted affinity *alone* reliably
drifts toward big greasy molecules the scorer happens to overrate. That's the standard
failure of this whole class of pipeline, so they're in there as regularisers from the start.

**Out comes** a ranked CSV and SDF poses shifted back into the original crystal frame, so
they open directly next to your PDB in PyMOL.

---

## 8 · How it runs on Kaggle

Four notebooks, chained by Kaggle's own dataset mechanism:

```
  01  build graph shards          CPU, ~40 min  →  1.5 GB of shards + library copy
       │                                             │
       ├─────────────────────────────────────────────┤
       ▼                                             ▼
  02  train the generator         GPU, resumable 03  train the scorer      GPU, ~1 h
       │                                             │
       └──────────────────┬──────────────────────────┘
                          ▼
  04  generate and rank          GPU, minutes
```

Notebook 01 copies the library into its own output, so 02–04 need exactly one input: that
output.

Two Kaggle realities shaped this more than anything else.

**The 20 GB output limit.** A featurised complex is ~590 kB — 88 GB for the full dataset. So
shards store positions plus `int8` codes at ~10 kB, and the dense form is rebuilt in 0.8 ms
inside DataLoader workers, overlapped with the GPU. **1.5 GB instead of 88 GB.** Making that
fast meant replacing per-edge Python in the covalent-bond logic with a precomputed
`(residue, atom name, atom name)` lookup table.

**The 9-hour wall.** Kaggle kills the kernel mid-run. So every notebook checkpoints every
epoch, *atomically* (write to `.tmp`, then rename — a half-written checkpoint is worse than
none), saving optimiser, scheduler and EMA state. Hitting the wall costs one epoch, not a
run. Point `RESUME` at the last checkpoint and continue in a fresh session. A `TimeBudget`
stops cleanly ~25 min early so the final save and the sampling demo actually happen.

One more: sampling always uses the **EMA** weights, never the raw ones. Diffusion sample
quality is notoriously sensitive to this — same training step, visibly cleaner geometry.

---

## 9 · Where it honestly stands

**Verified:** all four notebooks execute end-to-end. 38-assertion self-test passes in 4 s
with no data. Equivariance holds to 5×10⁻⁷. Batch-offset correctness proven by checking edge
*lengths* survive collation. Pocket detection measured at 76.7%/86.7%. The objective learns —
1.59 Å placement on memorised pockets.

**Not done:** nothing is trained at scale. That's GPU sessions, and validity is the metric
that responds most to epochs.

**Known limits, stated plainly:** the pocket is rigid (no induced fit). Bond orders are
perceived, never generated. The scorer is a scoring function, not a free-energy calculation —
CASF Pearson *r* of 0.7–0.8 is where good published ones sit, and anything much higher on a
random split means leakage. PDB-code splits are weaker than sequence-identity splits;
homologues with different codes can still straddle them.

**On that last point** — the most dangerous thing in a project like this isn't a bug, it's a
metric that lies. The dataset lists the same ligand once per copy in the asymmetric unit:
`1b7z_1_OXL_D` and `1b7z_1_OXL_E` are the same molecule in two copies of the same site. A
random split puts one in train and one in test, and every number you report becomes a measure
of memorisation. The splitter groups by PDB code and notebook 01 **asserts** no code appears
in two splits. Same reason the CASF-2016 core set is forced into test, and the same reason
`build_pocket_batch` physically deletes the crystal ligand rather than merely promising not
to look at it.

---

## 10 · Closing the loop

The loop sketched in `plan.md` — *generate → evaluate → improve → loop* — is inpainting, and
the mechanism already exists. `sample()` freezes the pocket via a coordinate mask. Keep the
best-scoring fragment of a molecule, apply that same mask to those ligand atoms, re-noise the
rest, and you have a scaffold decorator: it keeps what worked and regrows the rest. That's a
few lines, not a new model — the recipe is in [PIPELINE.md §12](PIPELINE.md).
