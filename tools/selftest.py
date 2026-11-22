#!/usr/bin/env python3
"""Data-free end-to-end self-test.

    python tools/selftest.py

Builds a synthetic protein and ligand, then exercises every stage: encoding,
featurisation, batch offsetting, the diffusion objective, E(3) symmetry,
sampling, pocket detection, affinity scoring, bond perception and ranking.

Runs in well under a minute and needs no datasets, so it is the first thing to
run after changing the library or installing on a new machine.  It checks
behaviour, not just imports — in particular the batch-offset correctness of
``ComplexData.__inc__`` and the symmetry of the denoiser, both of which fail
silently in training rather than loudly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from molgen import affinity, chem, diffusion, egnn, evaluate, graphs, pdbio, pockets

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------
def synth_protein(n_res: int = 90, seed: int = 0) -> pdbio.Atoms:
    """A toy protein: residues on a compact 3-D lattice with N/CA/C/O/CB atoms.

    Not physically realistic, but it has the structural properties every stage
    depends on: residue grouping, backbone naming, a buried interior and a
    solvent-exposed surface.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n_res ** (1 / 3)))
    rec, ser, nam, rsn, chn, rsq, ele, pos = [], [], [], [], [], [], [], []
    names = ["N", "CA", "C", "O", "CB"]
    elements = ["N", "C", "C", "O", "C"]
    k = 0
    for i in range(side):
        for j in range(side):
            for l in range(side):
                if k >= n_res:
                    break
                centre = np.array([i, j, l], dtype=float) * 5.5
                resname = chem.AMINO_ACIDS[rng.integers(0, 20)]
                for nm, el, off in zip(names, elements, rng.normal(0, 1.2, (5, 3))):
                    rec.append("ATOM"); ser.append(len(ser) + 1)
                    nam.append(nm); rsn.append(resname); chn.append("A")
                    rsq.append(k + 1); ele.append(el)
                    pos.append(centre + off)
                k += 1
    n = len(ser)
    return pdbio.Atoms(
        record=np.array(rec, "<U6"), serial=np.array(ser, np.int64),
        name=np.array(nam, "<U4"), altloc=np.array([""] * n, "<U1"),
        resname=np.array(rsn, "<U3"), chain=np.array(chn, "<U2"),
        resseq=np.array(rsq, np.int64), icode=np.array([""] * n, "<U1"),
        pos=np.array(pos, np.float32), occupancy=np.ones(n, np.float32),
        bfactor=np.zeros(n, np.float32), element=np.array(ele, "<U2"),
    )


def synth_ligand(center: np.ndarray, n_atoms: int = 18, seed: int = 0) -> pdbio.Atoms:
    """A toy ligand: a self-avoiding chain of C/N/O at ~1.5 A bond length."""
    rng = np.random.default_rng(seed)
    pos = [np.asarray(center, float)]
    while len(pos) < n_atoms:
        step = rng.normal(0, 1, 3)
        cand = pos[-1] + 1.5 * step / np.linalg.norm(step)
        if all(np.linalg.norm(cand - p) > 1.3 for p in pos):
            pos.append(cand)
    els = [["C", "C", "C", "N", "O"][rng.integers(0, 5)] for _ in range(n_atoms)]
    return pdbio.Atoms(
        record=np.array(["HETATM"] * n_atoms, "<U6"),
        serial=np.arange(1, n_atoms + 1, dtype=np.int64),
        name=np.array([f"{e}{i}" for i, e in enumerate(els)], "<U4"),
        altloc=np.array([""] * n_atoms, "<U1"),
        resname=np.array(["LIG"] * n_atoms, "<U3"),
        chain=np.array(["L"] * n_atoms, "<U2"),
        resseq=np.ones(n_atoms, np.int64), icode=np.array([""] * n_atoms, "<U1"),
        pos=np.array(pos, np.float32), occupancy=np.ones(n_atoms, np.float32),
        bfactor=np.zeros(n_atoms, np.float32), element=np.array(els, "<U2"),
    )


# --------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    torch.manual_seed(0)

    print("\n1 · chemistry and I/O")
    rec = synth_protein()
    centre = rec.pos.mean(0)
    lig = synth_ligand(centre, 18)
    check("synthetic structures", len(rec) == 450 and len(lig) == 18,
          f"{len(rec)} protein atoms, {len(lig)} ligand atoms")

    bi, bo = chem.infer_bonds(lig.pos, lig.element)
    check("bond perception", bi.shape[1] >= 15, f"{bi.shape[1]} bonds for an 18-atom chain")

    degree = np.bincount(bi.reshape(-1), minlength=len(lig))
    caps = np.array([chem.MAX_VALENCE.get(str(e), 4) for e in lig.element])
    check("valence respected", bool((degree <= caps).all()), f"max degree {degree.max()}")

    keep, reason = chem.is_interesting_ligand("LIG", len(lig), lig.element)
    check("ligand filter accepts a druglike ligand", keep, reason)
    check("ligand filter rejects sulfate",
          not chem.is_interesting_ligand("SO4", 5, ["S", "O", "O", "O", "O"])[0])

    print("\n2 · graph encoding and featurisation")
    sample = graphs.encode_complex(rec, lig, complex_id="synth", lig_name="LIG", affinity=7.0)
    check("encode_complex", sample is not None)
    nbytes = graphs.sample_nbytes(sample)

    data = graphs.featurize(sample)
    check("pocket feature width", data.poc_x.shape[1] == graphs.POCKET_FEATURE_DIM,
          f"{data.poc_x.shape[1]} == {graphs.POCKET_FEATURE_DIM}")
    check("pocket contacts found", data.poc_edge_index.shape[1] > 0,
          f"{data.poc_edge_index.shape[1]} directed edges")
    check("cross edges found", data.inter_edge_index.shape[1] > 0,
          f"{data.inter_edge_index.shape[1]} ligand-pocket edges")
    check("covalent flags set", float(data.poc_edge_attr[:, 0].sum()) > 0,
          f"{data.poc_edge_attr[:, 0].mean():.3f} of pocket edges are covalent")
    check("features are finite",
          all(torch.isfinite(t).all() for t in [data.poc_x, data.lig_pos, data.poc_pos,
                                                data.poc_edge_attr, data.inter_edge_attr]))

    t0 = time.time()
    for _ in range(20):
        graphs.featurize(sample)
    per = (time.time() - t0) / 20
    dense = sum(t.numel() * t.element_size() for t in
                [data.poc_x, data.poc_edge_attr, data.poc_edge_index, data.poc_pos,
                 data.inter_edge_attr, data.inter_edge_index])
    check("storage stays compact", dense > 5 * nbytes,
          f"{nbytes/1e3:.1f} kB encoded vs {dense/1e3:.1f} kB dense ({dense/nbytes:.0f}x), "
          f"featurize {per*1000:.2f} ms")

    print("\n3 · batching (ComplexData.__inc__)")
    items = [graphs.featurize(graphs.encode_complex(
        rec, synth_ligand(centre + np.array([float(i), 0.0, 0.0]), 14 + i, seed=i),
        complex_id=f"s{i}", lig_name="LIG", affinity=5.0 + i)) for i in range(4)]
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(items, follow_batch=graphs.FOLLOW_BATCH)

    check("bond index in range", int(batch.lig_bond_index.max()) < batch.lig_pos.size(0))
    check("pocket edge index in range", int(batch.poc_edge_index.max()) < batch.poc_pos.size(0))
    check("cross edge rows in range",
          int(batch.inter_edge_index[0].max()) < batch.lig_pos.size(0)
          and int(batch.inter_edge_index[1].max()) < batch.poc_pos.size(0))

    # The offsets are only *correct* if the edge lengths survive collation.
    one = items[2]
    off = sum(items[i].inter_edge_index.size(1) for i in range(2))
    sl = slice(off, off + one.inter_edge_index.size(1))
    d_single = (one.lig_pos[one.inter_edge_index[0]] - one.poc_pos[one.inter_edge_index[1]]).norm(dim=-1)
    d_batch = (batch.lig_pos[batch.inter_edge_index[0, sl]]
               - batch.poc_pos[batch.inter_edge_index[1, sl]]).norm(dim=-1)
    check("cross-edge geometry preserved by collation",
          torch.allclose(d_single, d_batch, atol=1e-5),
          f"max diff {float((d_single - d_batch).abs().max()):.2e}")

    ll = graphs.intra_radius_edges(batch.lig_pos, batch.lig_pos_batch)
    cr = graphs.cross_radius_edges(batch.lig_pos, batch.lig_pos_batch,
                                   batch.poc_pos, batch.poc_pos_batch, cutoff=8.0)
    check("no ligand edge crosses graphs",
          bool((batch.lig_pos_batch[ll[0]] == batch.lig_pos_batch[ll[1]]).all()))
    check("no cross edge crosses graphs",
          bool((batch.lig_pos_batch[cr[0]] == batch.poc_pos_batch[cr[1]]).all()))

    print("\n4 · diffusion model")
    cfg = diffusion.DiffusionConfig(timesteps=50, hidden=32, num_layers=2, coord_scale=3.5)
    model = diffusion.PocketLigandDiffusion(cfg)
    out = model(batch)
    check("loss is finite", bool(torch.isfinite(out["loss"])),
          "  ".join(f"{k}={float(v):.4f}" for k, v in out.items()))
    out["loss"].backward()
    gn = sum(float(p.grad.norm() ** 2) for p in model.parameters() if p.grad is not None) ** 0.5
    check("gradients flow", np.isfinite(gn) and gn > 0, f"grad norm {gn:.3f}")

    print("\n5 · E(3) symmetry of the denoiser")
    rep = egnn.check_equivariance(model.net, Batch.from_data_list(items, follow_batch=graphs.FOLLOW_BATCH))
    for key in ["rotation_equivariance_error", "translation_equivariance_error",
                "displacement_translation_invariance_error", "type_invariance_error"]:
        check(key.replace("_", " "), rep[key] < 1e-3, f"{rep[key]:.2e}")

    print("\n6 · sampling")
    pbatch = diffusion.build_pocket_batch(items[:2])
    check("ligand stripped for sampling", pbatch.lig_pos.size(0) == 0)
    smp = model.sample(pbatch, n_atoms=[12, 15], num_steps=50, eta=1.0)
    mols = model.to_molecules(smp)
    check("sampled the requested sizes", [len(m["elements"]) for m in mols] == [12, 15])
    radii = [float(np.linalg.norm(m["pos"], axis=1).max()) for m in mols]
    # An untrained model cannot place atoms well, but the x0 parameterisation
    # must still keep them near the pocket instead of expelling them.
    check("samples stay near the pocket", max(radii) < 60.0,
          f"max atom radius {max(radii):.1f} A from the pocket centroid")
    check("sampled coordinates finite", bool(np.isfinite(mols[0]["pos"]).all()))

    print("\n7 · pocket-only path (inference)")
    po = graphs.encode_pocket_only(rec, reference_pos=lig.pos, complex_id="synth")
    check("encode_pocket_only", po is not None and len(po["lig_type"]) == 0)
    check("pocket codes identical to training encoding",
          bool(np.array_equal(po["poc_name"], sample["poc_name"])),
          "same residues selected and encoded the same way")
    att = graphs.attach_ligand(po, mols[0]["elements"], mols[0]["pos"] * 0.1)
    check("attach_ligand", att is not None and att.lig_pos.size(0) == 12)

    print("\n8 · affinity model")
    anet = affinity.AffinityNet(affinity.AffinityConfig(hidden=32, num_layers=2))
    pred = anet(batch)
    check("predicts one value per graph", tuple(pred.shape) == (4,),
          f"pK {pred.detach().numpy().round(2).tolist()}")
    aloss = anet.loss(batch)
    aloss["loss"].backward()
    check("affinity loss finite and differentiable", bool(torch.isfinite(aloss["loss"])),
          f"loss {float(aloss['loss']):.4f}  n={aloss['n']}")
    scored = affinity.predict_batched(anet, [att, att])
    check("scores generated molecules", tuple(scored.shape) == (2,),
          f"{scored.numpy().round(2).tolist()}")
    m = affinity.regression_metrics([1.0, 2.0, 3.0, 4.0], [1.2, 1.9, 3.3, 3.8])
    check("regression metrics", m["pearson"] > 0.95 and m["rmse"] < 0.3,
          f"r={m['pearson']:.3f}  rmse={m['rmse']:.3f}")

    print("\n9 · pocket detection")
    t0 = time.time()
    found = pockets.detect_pockets_builtin(rec, spacing=1.0, max_pockets=5)
    check("finds cavities", len(found) > 0, f"{len(found)} in {time.time()-t0:.2f}s")
    if found:
        check("pockets are ranked and annotated",
              found[0].rank == 1 and found[0].score >= found[-1].score and bool(found[0].residues),
              f"top score {found[0].score:.1f}, {len(found[0].residues)} lining residues")
        det = pockets.score_detection(found, lig.pos.mean(0), ligand_pos=lig.pos)
        check("detection metrics computed", "overlap_top1" in det and "dcc" in det,
              f"dcc {det['dcc']:.1f} A, top-3 overlap {det['overlap_top3']}")
    check("auto dispatch falls back to builtin",
          len(pockets.detect_pockets(Path("/nonexistent.pdb"), method="auto",
                                     receptor=rec, max_pockets=2)) > 0)

    print("\n10 · evaluation and ranking")
    poc_el = graphs.pocket_elements_from_features(items[0].poc_x)
    check("elements recovered from features", set(poc_el) <= set(chem.POCKET_ELEMENTS),
          f"{sorted(set(poc_el))}")

    true_el = [chem.LIGAND_ELEMENTS[t] for t in items[0].lig_type.tolist()]
    real = evaluate.evaluate_molecule(true_el, items[0].lig_pos.numpy(),
                                      poc_pos=items[0].poc_pos.numpy(), poc_elements=poc_el,
                                      index=0, pocket_id="synth")
    check("real ligand evaluates as one fragment", real.n_fragments == 1,
          f"{real.n_fragments} fragments, min dist {real.min_dist:.2f} A"
          + (f", SMILES {real.smiles}" if real.smiles else " (no rdkit)"))

    recs = [real]
    for i, mol in enumerate(mols):
        recs.append(evaluate.evaluate_molecule(mol["elements"], mol["pos"],
                                               poc_pos=items[0].poc_pos.numpy(),
                                               poc_elements=poc_el, index=i + 1, pocket_id="synth"))
    recs[0].pred_affinity = 9.0
    sm = evaluate.set_metrics(recs)
    check("set metrics computed", {"validity", "uniqueness", "connected_fraction"} <= set(sm),
          f"validity {sm['validity']:.2f}, connected {sm['connected_fraction']:.2f}")

    df = evaluate.rank_molecules(recs)
    check("ranking is ordered", bool((df["score"].diff().dropna() <= 1e-9).all()),
          f"top score {df['score'].iloc[0]:.3f}")
    check("the real ligand outranks untrained samples", int(df["index"].iloc[0]) == 0,
          "a well-formed molecule with good predicted affinity wins")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        n = evaluate.records_to_sdf(recs, str(Path(tmp) / "g.sdf"),
                                    offset=np.asarray(sample["center"]))
        text = (Path(tmp) / "g.sdf").read_text()
    check("SDF export", n == len(recs) and text.count("$$$$") == len(recs),
          f"{n} records, {len(text)} bytes")

    print(f"\n{'='*70}")
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s): " + ", ".join(FAILS))
    else:
        print(f"all checks passed in {time.time() - t_start:.1f}s")
    print("=" * 70)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
