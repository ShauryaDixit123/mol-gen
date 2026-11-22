"""Turn generated point clouds into molecules, judge them, rank them.

A diffusion model emits *atoms with coordinates*, not a molecule.  Everything
downstream — SMILES, QED, synthetic accessibility, the affinity model — needs
connectivity, so bond perception is a load-bearing step and its failure rate is
itself a headline metric ("validity").

Two things are checked that are easy to skip and that make generative numbers
meaningless when skipped:

* **geometry sanity** — a model can minimise the diffusion loss while emitting
  atoms 0.5 A apart.  Such a "molecule" often still sanitises after bond
  perception drops the overlapping pair.
* **pocket compatibility** — atoms overlapping the protein, or floating in
  solvent outside the cavity, are not binders no matter how good the SMILES is.

RDKit is optional: without it you still get geometry, clash, size and element
statistics; with it you additionally get SMILES, QED, SA, logP and ring counts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

from . import chem
from .pdbio import write_sdf_block

# --------------------------------------------------------------------------
# optional RDKit
# --------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:  # pragma: no cover
    HAS_RDKIT = False

_SASCORER = None


def _sascorer():
    """Lazily import RDKit's contrib SA_Score (not importable by default)."""
    global _SASCORER
    if _SASCORER is not None or not HAS_RDKIT:
        return _SASCORER
    try:
        from rdkit.Chem import RDConfig

        path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if path not in sys.path:
            sys.path.append(path)
        import sascorer  # type: ignore

        _SASCORER = sascorer
    except Exception:  # noqa: BLE001
        _SASCORER = None
    return _SASCORER


# --------------------------------------------------------------------------
# point cloud -> molecule
# --------------------------------------------------------------------------
def to_rdkit_mol(elements, pos, use_determine_bonds: bool = True):
    """Build an RDKit mol from elements + coordinates.

    ``rdDetermineBonds`` (the xyz2mol port) also assigns bond orders and formal
    charges, which is why it is tried first; it is strict and raises often, so
    the geometric perception in :func:`molgen.chem.infer_bonds` is the fallback.
    Returns ``(mol, method)`` with ``mol=None`` on failure.
    """
    if not HAS_RDKIT:
        return None, "no_rdkit"

    pos = np.asarray(pos, dtype=float)
    if use_determine_bonds:
        try:
            from rdkit.Chem import rdDetermineBonds

            xyz = f"{len(pos)}\n\n" + "\n".join(
                f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(elements, pos)
            )
            mol = Chem.MolFromXYZBlock(xyz)
            if mol is not None:
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                Chem.SanitizeMol(mol)
                return mol, "determine_bonds"
        except Exception:  # noqa: BLE001
            pass

    try:
        bond_index, bond_order = chem.infer_bonds(pos, elements)
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(pos))
        for i, e in enumerate(elements):
            rw.AddAtom(Chem.Atom(str(e)))
            conf.SetAtomPosition(i, tuple(float(v) for v in pos[i]))
        order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for k in range(bond_index.shape[1]):
            rw.AddBond(int(bond_index[0, k]), int(bond_index[1, k]), order_map.get(int(bond_order[k]), Chem.BondType.SINGLE))
        mol = rw.GetMol()
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        return mol, "infer_bonds"
    except Exception:  # noqa: BLE001
        return None, "failed"


# --------------------------------------------------------------------------
# geometry / clash checks
# --------------------------------------------------------------------------
def geometry_report(elements, pos, min_allowed: float = 0.9) -> dict:
    """Internal geometry sanity, independent of RDKit."""
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n < 2:
        return {"n_atoms": n, "min_dist": float("inf"), "n_self_clash": 0,
                "n_bonds": 0, "n_fragments": 1 if n else 0, "radius_of_gyration": 0.0}

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    bond_index, _ = chem.infer_bonds(pos, elements)

    # connected components over the perceived bond graph
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in range(bond_index.shape[1]):
        a, b = find(int(bond_index[0, k])), find(int(bond_index[1, k]))
        if a != b:
            parent[a] = b
    n_frag = len({find(i) for i in range(n)})

    centroid = pos.mean(0)
    rg = float(np.sqrt(((pos - centroid) ** 2).sum(1).mean()))
    return {
        "n_atoms": n,
        "min_dist": float(d.min()),
        "n_self_clash": int((d < min_allowed).sum() // 2),
        "n_bonds": int(bond_index.shape[1]),
        "n_fragments": n_frag,
        "radius_of_gyration": rg,
    }


def pocket_report(
    lig_pos,
    poc_pos,
    poc_elements=None,
    lig_elements=None,
    clash_slack: float = 0.5,
    contact_cutoff: float = 4.5,
) -> dict:
    """Steric fit against the receptor.

    A clash is a heavy-atom pair closer than ``r_vdw(i) + r_vdw(j) - slack``;
    0.5 A of slack is roughly what crystal structures show for genuine close
    contacts (h-bonds, metal coordination).
    """
    from .pockets import DEFAULT_VDW, VDW

    lig_pos = np.asarray(lig_pos, dtype=float)
    poc_pos = np.asarray(poc_pos, dtype=float)
    if len(lig_pos) == 0 or len(poc_pos) == 0:
        return {"n_clash": 0, "min_prot_dist": float("inf"), "n_contacts": 0, "buriedness": 0.0}

    d = np.linalg.norm(lig_pos[:, None, :] - poc_pos[None, :, :], axis=-1)
    r_l = np.array([VDW.get(str(e).capitalize(), DEFAULT_VDW) for e in (lig_elements or ["C"] * len(lig_pos))])
    r_p = np.array([VDW.get(str(e).capitalize(), DEFAULT_VDW) for e in (poc_elements if poc_elements is not None else ["C"] * len(poc_pos))])
    limit = r_l[:, None] + r_p[None, :] - clash_slack

    contacts_per_atom = (d < contact_cutoff).sum(1)
    return {
        "n_clash": int((d < limit).sum()),
        "min_prot_dist": float(d.min()),
        "n_contacts": int(contacts_per_atom.sum()),
        # fraction of ligand atoms with at least 4 protein neighbours: a cheap
        # proxy for "inside the cavity" rather than "floating in solvent"
        "buriedness": float((contacts_per_atom >= 4).mean()),
    }


# --------------------------------------------------------------------------
# chemistry metrics
# --------------------------------------------------------------------------
@dataclass
class MoleculeRecord:
    index: int = -1
    pocket_id: str = ""
    elements: list[str] = field(default_factory=list)
    pos: np.ndarray | None = None
    smiles: str = ""
    valid: bool = False
    bond_method: str = ""
    n_atoms: int = 0
    n_fragments: int = 0
    min_dist: float = 0.0
    n_self_clash: int = 0
    radius_of_gyration: float = 0.0
    n_clash: int = 0
    min_prot_dist: float = 0.0
    n_contacts: int = 0
    buriedness: float = 0.0
    qed: float = float("nan")
    sa: float = float("nan")
    logp: float = float("nan")
    mw: float = float("nan")
    n_rings: int = 0
    n_rotatable: int = 0
    tpsa: float = float("nan")
    lipinski_violations: int = 0
    pred_affinity: float = float("nan")
    score: float = float("nan")

    def row(self) -> dict:
        d = asdict(self)
        d.pop("pos", None)
        d["elements"] = "".join(self.elements) if self.elements else ""
        return d


def chemistry_report(mol) -> dict:
    """RDKit descriptors; empty dict when RDKit is unavailable or mol is None."""
    if not HAS_RDKIT or mol is None:
        return {}
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        return {}
    out = {"smiles": smiles}
    try:
        out["qed"] = float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        pass
    sascorer = _sascorer()
    if sascorer is not None:
        try:
            out["sa"] = float(sascorer.calculateScore(mol))
        except Exception:  # noqa: BLE001
            pass
    try:
        out["logp"] = float(Crippen.MolLogP(mol))
        out["mw"] = float(Descriptors.MolWt(mol))
        out["tpsa"] = float(Descriptors.TPSA(mol))
        out["n_rings"] = int(mol.GetRingInfo().NumRings())
        out["n_rotatable"] = int(Descriptors.NumRotatableBonds(mol))
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        out["lipinski_violations"] = int(
            (out["mw"] > 500) + (out["logp"] > 5) + (hbd > 5) + (hba > 10)
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def evaluate_molecule(
    elements,
    pos,
    poc_pos=None,
    poc_elements=None,
    index: int = -1,
    pocket_id: str = "",
) -> MoleculeRecord:
    """Full per-molecule report: geometry + chemistry + pocket fit."""
    rec = MoleculeRecord(index=index, pocket_id=pocket_id, elements=list(elements), pos=np.asarray(pos))
    for k, v in geometry_report(elements, pos).items():
        setattr(rec, k, v)
    if poc_pos is not None:
        for k, v in pocket_report(pos, poc_pos, poc_elements, elements).items():
            setattr(rec, k, v)

    mol, method = to_rdkit_mol(elements, pos)
    rec.bond_method = method
    rec.valid = mol is not None
    for k, v in chemistry_report(mol).items():
        setattr(rec, k, v)
    return rec


# --------------------------------------------------------------------------
# set-level metrics
# --------------------------------------------------------------------------
def set_metrics(records: list[MoleculeRecord], reference_smiles: set[str] | None = None) -> dict:
    """Validity / uniqueness / novelty — reported together, as they trade off.

    A model that emits one valid molecule every time scores 100 % validity and
    ~0 % uniqueness, so neither number means anything alone.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    valid = [r for r in records if r.valid and r.smiles]
    smiles = [r.smiles for r in valid]
    unique = set(smiles)
    connected = [r for r in records if r.n_fragments == 1]
    out = {
        "n": n,
        "validity": len(valid) / n,
        "connected_fraction": len(connected) / n,
        "uniqueness": (len(unique) / len(smiles)) if smiles else 0.0,
        "mean_atoms": float(np.mean([r.n_atoms for r in records])),
        "clash_free_fraction": float(np.mean([r.n_clash == 0 for r in records])),
        "mean_buriedness": float(np.mean([r.buriedness for r in records])),
    }
    if reference_smiles:
        out["novelty"] = float(np.mean([s not in reference_smiles for s in smiles])) if smiles else 0.0
    for key in ("qed", "sa", "logp", "mw", "pred_affinity"):
        vals = [getattr(r, key) for r in valid if np.isfinite(getattr(r, key))]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
@dataclass
class RankWeights:
    """Weights of the composite objective.

    Affinity dominates (that is the design goal), but a generator left to
    maximise predicted affinity alone drifts towards large greasy molecules the
    affinity model happens to score well — so QED and synthetic accessibility
    enter as regularisers and hard geometry failures as penalties.
    """

    affinity: float = 1.0
    qed: float = 0.4
    sa: float = 0.3
    buriedness: float = 0.3
    clash_penalty: float = 0.5
    fragment_penalty: float = 1.0
    invalid_penalty: float = 3.0


def composite_score(rec: MoleculeRecord, w: RankWeights = RankWeights()) -> float:
    """Single number for ranking, all terms normalised to roughly [0, 1]."""
    aff = 0.0 if not np.isfinite(rec.pred_affinity) else np.clip((rec.pred_affinity - 2.0) / 10.0, 0.0, 1.5)
    qed = 0.0 if not np.isfinite(rec.qed) else rec.qed
    sa = 0.0 if not np.isfinite(rec.sa) else np.clip((10.0 - rec.sa) / 9.0, 0.0, 1.0)

    score = (
        w.affinity * aff
        + w.qed * qed
        + w.sa * sa
        + w.buriedness * rec.buriedness
        - w.clash_penalty * np.clip(rec.n_clash / 5.0, 0.0, 2.0)
        - w.fragment_penalty * max(0, rec.n_fragments - 1) / 3.0
        - (0.0 if rec.valid else w.invalid_penalty)
    )
    return float(score)


def rank_molecules(records: list[MoleculeRecord], weights: RankWeights = RankWeights()):
    """Score, sort and return a pandas DataFrame (best first)."""
    import pandas as pd

    for r in records:
        r.score = composite_score(r, weights)
    df = pd.DataFrame([r.row() for r in records])
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def records_to_sdf(
    records: list[MoleculeRecord],
    path: str,
    offset: np.ndarray | None = None,
) -> int:
    """Write molecules to one SDF, with scores as data fields.

    ``offset`` (the pocket centroid) puts the poses back into the receptor's
    crystal frame so they can be opened next to the original PDB in PyMOL.
    """
    blocks = []
    for r in records:
        if r.pos is None:
            continue
        pos = r.pos if offset is None else r.pos + np.asarray(offset).reshape(1, 3)
        bond_index, bond_order = chem.infer_bonds(pos, r.elements)
        blocks.append(
            write_sdf_block(
                r.elements, pos, bond_index, bond_order,
                name=f"{r.pocket_id}_gen{r.index}",
                props={
                    "score": round(r.score, 4),
                    "pred_affinity_pK": round(r.pred_affinity, 3),
                    "qed": round(r.qed, 3) if np.isfinite(r.qed) else "",
                    "sa": round(r.sa, 3) if np.isfinite(r.sa) else "",
                    "n_clash": r.n_clash,
                    "n_fragments": r.n_fragments,
                    "smiles": r.smiles,
                },
            )
        )
    with open(path, "w") as fh:
        fh.writelines(blocks)
    return len(blocks)
