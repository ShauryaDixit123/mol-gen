"""Graph construction: protein pocket + ligand -> a single PyG object.

Design notes
------------
The whole pipeline is graph-native.  One complex becomes one
:class:`ComplexData` holding **two node sets** and **three edge sets**:

    ligand atoms   lig_pos / lig_type        bonds     lig_bond_index
    pocket atoms   poc_pos / poc_x           contacts  poc_edge_index
                                             cross     inter_edge_index

Storage is split from featurisation (``encode_*`` vs :func:`featurize`), which
is what makes the dataset fit on Kaggle at all.  A featurised complex is
~350 kB — dominated by ``(E, 17)`` float edge attributes over ~4 k pocket
contacts — so 150 k of them would be ~60 GB against a 20 GB output limit.  The
encoded form keeps only positions and small integer codes (~12 kB), and
:func:`featurize` rebuilds the KD-tree edges and float features in a few
milliseconds inside the DataLoader workers, overlapped with the GPU.

Which features the two models may look at differs, and this matters:

* **Diffusion model** — at sampling time the ligand does not exist yet, so the
  only ligand input is the (noisy) coordinate and the (noisy) element type.
  Anything bond-derived would leak.  Cross and intra-ligand edges are therefore
  *rebuilt every denoising step* from the current noisy coordinates
  (:func:`cross_radius_edges`, :func:`intra_radius_edges`); the stored
  ``inter_edge_index`` is not used there.
* **Affinity model** — the ligand is given, so bonds, degrees and the stored
  contact edges are all legitimate inputs.

``torch_cluster`` is never required: preprocessing uses scipy KD-trees and
train-time edges use plain torch, which avoids the most common Kaggle wheel
mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from . import chem
from .pdbio import Atoms

# --------------------------------------------------------------------------
# feature dimensions (import these instead of hardcoding model input sizes)
# --------------------------------------------------------------------------
LIGAND_TYPE_DIM = chem.NUM_LIGAND_ELEMENTS          # diffusion categorical dim
POCKET_FEATURE_DIM = (
    chem.NUM_POCKET_ELEMENTS                        # element one-hot + other
    + chem.NUM_AA                                   # residue one-hot + UNK
    + 4                                             # backbone role: N/CA/C/O
    + 1                                             # is_backbone
    + 1                                             # is_hetero (HETATM/metal)
    + len(chem.AA_PROPERTY_KEYS)                    # residue descriptors
)
LIGAND_EXTRA_DIM = 4                                # degree, en, mass, bond-order sum
NUM_RBF = 16
BOND_ORDER_DIM = 3
LIGAND_BOND_EDGE_DIM = BOND_ORDER_DIM + NUM_RBF
POCKET_EDGE_DIM = 1 + NUM_RBF                       # covalent flag + distance rbf
INTER_EDGE_DIM = NUM_RBF


# --------------------------------------------------------------------------
# distance featurisation
# --------------------------------------------------------------------------
def rbf_centers(cutoff: float, num: int = NUM_RBF) -> torch.Tensor:
    return torch.linspace(0.0, cutoff, num)


def rbf_expand(dist: torch.Tensor, cutoff: float = 10.0, num: int = NUM_RBF) -> torch.Tensor:
    """Gaussian smearing of distances -> (E, num) soft one-hot."""
    centers = rbf_centers(cutoff, num).to(dist.device, dist.dtype)
    gamma = num / cutoff
    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)


def _rbf_np(dist: np.ndarray, cutoff: float, num: int = NUM_RBF) -> np.ndarray:
    centers = np.linspace(0.0, cutoff, num, dtype=np.float32)
    gamma = num / cutoff
    return np.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2).astype(np.float32)


# --------------------------------------------------------------------------
# the data object
# --------------------------------------------------------------------------
class ComplexData(Data):
    """A pocket-ligand complex.

    Custom ``__inc__`` so PyG's collater offsets each edge set against the
    right node set.  Use ``follow_batch=['lig_pos', 'poc_pos']`` in the
    DataLoader to get ``lig_pos_batch`` / ``poc_pos_batch`` assignment vectors.
    """

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "lig_bond_index":
            return self.lig_pos.size(0)
        if key == "poc_edge_index":
            return self.poc_pos.size(0)
        if key == "inter_edge_index":
            # row 0 indexes ligand atoms, row 1 indexes pocket atoms
            return torch.tensor([[self.lig_pos.size(0)], [self.poc_pos.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ("lig_bond_index", "poc_edge_index", "inter_edge_index"):
            return 1
        if key in ("complex_id", "lig_name"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


FOLLOW_BATCH = ["lig_pos", "poc_pos"]


# --------------------------------------------------------------------------
# pocket extraction
# --------------------------------------------------------------------------
def pocket_atoms_near(
    receptor: Atoms,
    reference_pos: np.ndarray,
    radius: float = 8.0,
    max_atoms: int = 600,
    whole_residues: bool = True,
) -> Atoms:
    """Select receptor atoms forming the pocket around ``reference_pos``.

    ``reference_pos`` is normally the ligand's heavy-atom coordinates (training)
    or a predicted pocket point cloud (inference).  Selecting *whole residues*
    keeps side chains chemically intact, which the residue-level features need.
    """
    if len(receptor) == 0 or len(reference_pos) == 0:
        return receptor[np.zeros(len(receptor), dtype=bool)]

    tree = cKDTree(receptor.pos)
    hit: set[int] = set()
    for idx in tree.query_ball_point(np.asarray(reference_pos, dtype=np.float64), r=radius):
        hit.update(idx)
    if not hit:
        return receptor[np.zeros(len(receptor), dtype=bool)]

    mask = np.zeros(len(receptor), dtype=bool)
    mask[sorted(hit)] = True

    groups = receptor.residue_group_ids()
    if whole_residues:
        mask = np.isin(groups, np.unique(groups[mask]))

    if mask.sum() > max_atoms:
        # keep whole residues, nearest first, until the budget is spent
        centre = np.asarray(reference_pos, dtype=np.float64).mean(0)
        sel = np.where(mask)[0]
        gid = groups[sel]
        d = np.linalg.norm(receptor.pos[sel] - centre, axis=1)
        uniq = np.unique(gid)
        order = np.argsort([d[gid == g].min() for g in uniq])
        mask = np.zeros(len(receptor), dtype=bool)
        total = 0
        for g in uniq[order]:
            g_mask = groups == g
            n = int(g_mask.sum())
            if total + n > max_atoms:
                break
            mask |= g_mask
            total += n
    return receptor[mask]


# --------------------------------------------------------------------------
# encoding (compact, on-disk form)
# --------------------------------------------------------------------------
def encode_pocket(pocket: Atoms) -> dict:
    """Pocket atoms -> compact integer codes + float32 positions."""
    _, chain_code = np.unique(pocket.chain, return_inverse=True)
    return {
        "poc_pos": pocket.pos.astype(np.float32),
        "poc_elem": chem.encode_elements(pocket.element, chem.POCKET_ELEMENTS),
        "poc_aa": chem.encode_residues(pocket.resname),
        "poc_name": chem.encode_atom_names(pocket.name),
        "poc_res_group": pocket.residue_group_ids().astype(np.int16),
        "poc_chain": chain_code.astype(np.int8),
        "poc_resseq": pocket.resseq.astype(np.int32),
        "poc_is_het": (pocket.record == "HETATM"),
    }


def encode_complex(
    receptor: Atoms,
    ligand: Atoms,
    complex_id: str = "",
    lig_name: str = "",
    affinity: float | None = None,
    pocket_radius: float = 8.0,
    max_pocket_atoms: int = 600,
    center_mode: str = "pocket",
    lig_bonds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    """Receptor/ligand pair -> the compact dict that gets written to a shard.

    Coordinates are translated so the chosen centre sits at the origin, which
    removes the arbitrary crystal frame.  ``center_mode='pocket'`` (the default)
    uses the pocket centroid — available at inference time too, unlike the
    ligand centroid, so train and sample see the same convention.
    """
    lig = ligand.heavy()
    if len(lig) == 0:
        return None

    types = chem.encode_elements(lig.element, chem.LIGAND_ELEMENTS)
    if (types >= len(chem.LIGAND_ELEMENTS)).any():
        return None

    pocket = pocket_atoms_near(
        receptor.heavy(), lig.pos, radius=pocket_radius, max_atoms=max_pocket_atoms
    )
    if len(pocket) < 8:
        return None

    if lig_bonds is not None:
        bond_index = np.asarray(lig_bonds[0], dtype=np.int32).reshape(2, -1)
        bond_order = np.asarray(lig_bonds[1], dtype=np.int8).reshape(-1)
    else:
        bi, bo = chem.infer_bonds(lig.pos, lig.element)
        bond_index, bond_order = bi.astype(np.int32), bo.astype(np.int8)

    center = (pocket.pos.mean(0) if center_mode == "pocket" else lig.pos.mean(0)).astype(np.float32)
    sample = encode_pocket(pocket)
    sample["poc_pos"] = sample["poc_pos"] - center
    sample.update({
        "lig_pos": (lig.pos - center).astype(np.float32),
        "lig_type": types,
        "lig_bond_index": bond_index,
        "lig_bond_order": bond_order,
        "center": center,
        "y": np.float32(np.nan if affinity is None else affinity),
        "n_poc_res": int(len(np.unique(sample["poc_res_group"]))),
        "complex_id": complex_id,
        "lig_name": lig_name,
    })
    return sample


def encode_pocket_only(
    receptor: Atoms,
    reference_pos: np.ndarray,
    complex_id: str = "",
    pocket_radius: float = 8.0,
    max_pocket_atoms: int = 600,
) -> dict | None:
    """Inference-time encoding: a pocket with **no ligand**.

    Produces byte-identical pocket codes to :func:`encode_complex`, so the
    model sees the same input distribution at sampling time as in training.
    ``reference_pos`` may be a detector's pocket point cloud rather than a
    single centre, which picks a better-shaped residue set for non-spherical
    cavities.
    """
    ref = np.asarray(reference_pos, dtype=np.float64).reshape(-1, 3)
    pocket = pocket_atoms_near(
        receptor.heavy(), ref, radius=pocket_radius, max_atoms=max_pocket_atoms
    )
    if len(pocket) < 8:
        return None

    center = pocket.pos.mean(0).astype(np.float32)
    sample = encode_pocket(pocket)
    sample["poc_pos"] = sample["poc_pos"] - center
    sample.update({
        "lig_pos": np.zeros((0, 3), dtype=np.float32),
        "lig_type": np.zeros((0,), dtype=np.int8),
        "lig_bond_index": np.zeros((2, 0), dtype=np.int32),
        "lig_bond_order": np.zeros((0,), dtype=np.int8),
        "center": center,
        "y": np.float32(np.nan),
        "n_poc_res": int(len(np.unique(sample["poc_res_group"]))),
        "complex_id": complex_id,
        "lig_name": "",
        # kept so generated poses can be written back into the crystal frame
        # and clash-checked against the real receptor
        "poc_elements_str": np.array([str(e) for e in pocket.element], dtype="<U2"),
    })
    return sample


def sample_nbytes(sample: dict) -> int:
    return sum(v.nbytes for v in sample.values() if isinstance(v, np.ndarray))


# --------------------------------------------------------------------------
# featurisation (load time)
# --------------------------------------------------------------------------
def pocket_features(sample: dict) -> np.ndarray:
    """(N, POCKET_FEATURE_DIM) float32 node features, fully vectorised."""
    elem = np.asarray(sample["poc_elem"], dtype=np.intp)
    aa = np.asarray(sample["poc_aa"], dtype=np.intp)
    name = np.asarray(sample["poc_name"], dtype=np.intp)
    n = len(elem)

    elem_oh = np.zeros((n, chem.NUM_POCKET_ELEMENTS), dtype=np.float32)
    elem_oh[np.arange(n), np.clip(elem, 0, chem.NUM_POCKET_ELEMENTS - 1)] = 1.0

    aa_oh = np.zeros((n, chem.NUM_AA), dtype=np.float32)
    aa_oh[np.arange(n), np.clip(aa, 0, chem.NUM_AA - 1)] = 1.0

    role = np.zeros((n, 4), dtype=np.float32)
    role_idx = chem.BACKBONE_ROLE_TABLE[name]
    has_role = role_idx >= 0
    role[np.arange(n)[has_role], role_idx[has_role]] = 1.0

    is_bb = chem.IS_BACKBONE_TABLE[name].astype(np.float32)[:, None]
    is_het = np.asarray(sample["poc_is_het"], dtype=np.float32)[:, None]
    props = chem.AA_PROPERTY_TABLE[np.clip(aa, 0, chem.NUM_AA - 1)]
    return np.concatenate([elem_oh, aa_oh, role, is_bb, is_het, props], axis=1)


def pocket_elements_from_features(poc_x: "torch.Tensor | np.ndarray") -> list[str]:
    """Recover pocket element symbols from the element one-hot block of ``poc_x``.

    A featurised :class:`ComplexData` keeps no element strings, but clash checks
    need real van der Waals radii — defaulting every pocket atom to carbon
    under-counts clashes against sulfur and over-counts against oxygen.
    """
    x = poc_x.numpy() if hasattr(poc_x, "numpy") else np.asarray(poc_x)
    idx = x[:, : chem.NUM_POCKET_ELEMENTS].argmax(axis=1)
    vocab = chem.POCKET_ELEMENTS + ["C"]      # last bucket is 'other'
    return [vocab[min(int(i), len(vocab) - 1)] for i in idx]


def pocket_edges(sample: dict, cutoff: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    """Radius graph over pocket atoms with a covalent-bond flag, vectorised.

    The covalent flag comes from a precomputed ``(residue, name, name)`` table
    plus the peptide-bond special case, so no python runs per edge.
    """
    pos = sample["poc_pos"]
    n = len(pos)
    if n < 2:
        return np.zeros((2, 0), np.int64), np.zeros((0, POCKET_EDGE_DIM), np.float32)

    pairs = cKDTree(pos).query_pairs(r=cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((2, 0), np.int64), np.zeros((0, POCKET_EDGE_DIM), np.float32)

    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[i] - pos[j], axis=1).astype(np.float32)

    aa = np.asarray(sample["poc_aa"], dtype=np.intp)
    name = np.asarray(sample["poc_name"], dtype=np.intp)
    group = np.asarray(sample["poc_res_group"])
    chain = np.asarray(sample["poc_chain"])
    resseq = np.asarray(sample["poc_resseq"], dtype=np.int64)

    covalent = np.zeros(len(i), dtype=np.float32)
    same = group[i] == group[j]
    if same.any():
        covalent[same] = chem.COVALENT_TEMPLATE[
            np.clip(aa[i[same]], 0, chem.NUM_AA - 1), name[i[same]], name[j[same]]
        ].astype(np.float32)

    adjacent = (chain[i] == chain[j]) & (np.abs(resseq[i] - resseq[j]) == 1)
    peptide = (
        ((name[i] == chem.C_NAME_IDX) & (name[j] == chem.N_NAME_IDX))
        | ((name[i] == chem.N_NAME_IDX) & (name[j] == chem.C_NAME_IDX))
    )
    covalent = np.maximum(covalent, (adjacent & peptide).astype(np.float32))

    attr = np.concatenate([covalent[:, None], _rbf_np(d, cutoff)], axis=1)
    edge_index = np.concatenate([pairs.T, pairs.T[::-1]], axis=1).astype(np.int64)
    return edge_index, np.concatenate([attr, attr], axis=0)


def ligand_bond_edges(
    pos: np.ndarray,
    bond_index: np.ndarray,
    bond_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bidirectional bond graph with one-hot order + distance RBF attributes."""
    bond_index = np.asarray(bond_index, dtype=np.int64).reshape(2, -1)
    bond_order = np.asarray(bond_order, dtype=np.int64).reshape(-1)
    if bond_index.shape[1] == 0:
        return np.zeros((2, 0), np.int64), np.zeros((0, LIGAND_BOND_EDGE_DIM), np.float32)

    d = np.linalg.norm(pos[bond_index[0]] - pos[bond_index[1]], axis=1).astype(np.float32)
    order_oh = np.zeros((len(bond_order), BOND_ORDER_DIM), dtype=np.float32)
    order_oh[np.arange(len(bond_order)), np.clip(bond_order - 1, 0, BOND_ORDER_DIM - 1)] = 1.0
    attr = np.concatenate([order_oh, _rbf_np(d, 3.0)], axis=1)
    return (
        np.concatenate([bond_index, bond_index[::-1]], axis=1),
        np.concatenate([attr, attr], axis=0),
    )


def ligand_extra_features(
    types: np.ndarray,
    n_atoms: int,
    bond_index: np.ndarray,
    bond_order: np.ndarray,
) -> np.ndarray:
    """Bond-derived ligand features — **affinity model only** (leaks for diffusion)."""
    degree = np.zeros(n_atoms, dtype=np.float32)
    order_sum = np.zeros(n_atoms, dtype=np.float32)
    bi = np.asarray(bond_index, dtype=np.int64).reshape(2, -1)
    bo = np.asarray(bond_order, dtype=np.float32).reshape(-1)
    if bi.shape[1]:
        np.add.at(degree, bi[0], 1.0)
        np.add.at(degree, bi[1], 1.0)
        np.add.at(order_sum, bi[0], bo)
        np.add.at(order_sum, bi[1], bo)
    els = [chem.LIGAND_ELEMENTS[min(int(t), len(chem.LIGAND_ELEMENTS) - 1)] for t in types]
    en = np.array([chem.ELECTRONEGATIVITY.get(e, 2.5) for e in els], dtype=np.float32)
    mass = np.array([chem.ATOMIC_WEIGHT.get(e, 12.0) for e in els], dtype=np.float32)
    return np.stack([degree / 4.0, en / 4.0, mass / 127.0, order_sum / 6.0], axis=1)


def interaction_edges(
    lig_pos: np.ndarray,
    poc_pos: np.ndarray,
    cutoff: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross edges ligand-atom -> pocket-atom within ``cutoff``."""
    if len(lig_pos) == 0 or len(poc_pos) == 0:
        return np.zeros((2, 0), np.int64), np.zeros((0, INTER_EDGE_DIM), np.float32)
    tree = cKDTree(poc_pos)
    rows, cols = [], []
    for i, neigh in enumerate(tree.query_ball_point(lig_pos.astype(np.float64), r=cutoff)):
        rows.extend([i] * len(neigh))
        cols.extend(neigh)
    if not rows:
        return np.zeros((2, 0), np.int64), np.zeros((0, INTER_EDGE_DIM), np.float32)
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    d = np.linalg.norm(lig_pos[rows] - poc_pos[cols], axis=1).astype(np.float32)
    return np.stack([rows, cols]), _rbf_np(d, cutoff)


def featurize(
    sample: dict,
    pocket_cutoff: float = 4.5,
    inter_cutoff: float = 6.0,
) -> ComplexData:
    """Compact stored sample -> model-ready :class:`ComplexData`.

    Called in the DataLoader worker on every access, so it stays vectorised:
    two KD-tree queries plus table lookups, a few milliseconds per complex.
    """
    lig_pos = np.asarray(sample["lig_pos"], dtype=np.float32)
    types = np.asarray(sample["lig_type"], dtype=np.int64)
    poc_pos = np.asarray(sample["poc_pos"], dtype=np.float32)

    poc_x = pocket_features(sample)
    poc_ei, poc_ea = pocket_edges(sample, cutoff=pocket_cutoff)
    bond_ei, bond_ea = ligand_bond_edges(lig_pos, sample["lig_bond_index"], sample["lig_bond_order"])
    extra = ligand_extra_features(types, len(types), sample["lig_bond_index"], sample["lig_bond_order"])
    inter_ei, inter_ea = interaction_edges(lig_pos, poc_pos, cutoff=inter_cutoff)

    data = ComplexData(
        lig_pos=torch.from_numpy(lig_pos),
        lig_type=torch.from_numpy(types),
        lig_extra=torch.from_numpy(extra),
        lig_bond_index=torch.from_numpy(bond_ei),
        lig_bond_attr=torch.from_numpy(bond_ea),
        poc_pos=torch.from_numpy(poc_pos),
        poc_x=torch.from_numpy(poc_x),
        poc_edge_index=torch.from_numpy(poc_ei),
        poc_edge_attr=torch.from_numpy(poc_ea),
        inter_edge_index=torch.from_numpy(inter_ei),
        inter_edge_attr=torch.from_numpy(inter_ea),
        center=torch.from_numpy(np.asarray(sample["center"], dtype=np.float32)).view(1, 3),
        n_lig=torch.tensor([len(types)], dtype=torch.long),
        n_poc=torch.tensor([len(poc_pos)], dtype=torch.long),
        n_poc_res=torch.tensor([int(sample.get("n_poc_res", 0))], dtype=torch.long),
        complex_id=str(sample.get("complex_id", "")),
        lig_name=str(sample.get("lig_name", "")),
    )
    y = float(sample.get("y", np.nan))
    data.y = torch.tensor([y], dtype=torch.float)
    return data


def build_complex_graph(receptor: Atoms, ligand: Atoms, **kwargs) -> ComplexData | None:
    """Convenience: encode then featurise in one call (tests, small scripts)."""
    pocket_cutoff = kwargs.pop("pocket_cutoff", 4.5)
    inter_cutoff = kwargs.pop("inter_cutoff", 6.0)
    sample = encode_complex(receptor, ligand, **kwargs)
    if sample is None:
        return None
    return featurize(sample, pocket_cutoff=pocket_cutoff, inter_cutoff=inter_cutoff)


def attach_ligand(
    pocket_sample: dict,
    elements: list[str],
    pos: np.ndarray,
    inter_cutoff: float = 6.0,
    pocket_cutoff: float = 4.5,
) -> ComplexData | None:
    """Insert a generated ligand into a pocket-only sample (for affinity scoring).

    ``pos`` must be in the same centred frame the pocket uses, i.e. the frame
    the diffusion model emits.
    """
    types = chem.encode_elements(elements, chem.LIGAND_ELEMENTS)
    if len(types) == 0 or (types >= len(chem.LIGAND_ELEMENTS)).any():
        return None

    pos = np.asarray(pos, dtype=np.float32)
    bi, bo = chem.infer_bonds(pos, elements)
    merged = dict(pocket_sample)
    merged.update({
        "lig_pos": pos,
        "lig_type": types,
        "lig_bond_index": bi.astype(np.int32),
        "lig_bond_order": bo.astype(np.int8),
        "lig_name": "GEN",
        "y": np.float32(np.nan),
    })
    return featurize(merged, pocket_cutoff=pocket_cutoff, inter_cutoff=inter_cutoff)


# --------------------------------------------------------------------------
# dynamic edges (train/sample time, torch)
# --------------------------------------------------------------------------
def _ptr_from_batch(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    counts = torch.bincount(batch, minlength=num_graphs)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def intra_radius_edges(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 1e9,
    max_nodes_fully_connected: int = 64,
) -> torch.Tensor:
    """Edges among one node set, per graph.

    Ligands are small, so below ``max_nodes_fully_connected`` atoms the complete
    graph is used — what EDM/DiffSBDD do, and it lets the model move atoms that
    are currently far apart, which is essential early in denoising.
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    ptr = _ptr_from_batch(batch, num_graphs)
    rows, cols = [], []
    for g in range(num_graphs):
        s, e = int(ptr[g]), int(ptr[g + 1])
        n = e - s
        if n < 2:
            continue
        idx = torch.arange(s, e, device=pos.device)
        if n <= max_nodes_fully_connected and cutoff >= 1e8:
            i = idx.repeat_interleave(n)
            j = idx.repeat(n)
            keep = i != j
            rows.append(i[keep])
            cols.append(j[keep])
        else:
            d = torch.cdist(pos[s:e], pos[s:e])
            mask = (d < cutoff) & (d > 0)
            i, j = mask.nonzero(as_tuple=True)
            rows.append(i + s)
            cols.append(j + s)
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def cross_radius_edges(
    pos_a: torch.Tensor,
    batch_a: torch.Tensor,
    pos_b: torch.Tensor,
    batch_b: torch.Tensor,
    cutoff: float = 8.0,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Edges from node set A to node set B within ``cutoff``, per graph.

    Loops over graphs (tens of iterations) rather than building one dense
    ``N_a x N_b`` matrix, so batches with large pockets stay within memory.
    Returns a ``(2, E)`` index where row 0 indexes A and row 1 indexes B.
    """
    num_graphs = max(
        int(batch_a.max().item()) + 1 if batch_a.numel() else 0,
        int(batch_b.max().item()) + 1 if batch_b.numel() else 0,
    )
    ptr_a = _ptr_from_batch(batch_a, num_graphs)
    ptr_b = _ptr_from_batch(batch_b, num_graphs)
    rows, cols = [], []
    for g in range(num_graphs):
        sa, ea = int(ptr_a[g]), int(ptr_a[g + 1])
        sb, eb = int(ptr_b[g]), int(ptr_b[g + 1])
        if ea <= sa or eb <= sb:
            continue
        d = torch.cdist(pos_a[sa:ea], pos_b[sb:eb])
        if max_neighbors > 0 and d.size(1) > max_neighbors:
            topk = d.topk(max_neighbors, dim=1, largest=False)
            mask = torch.zeros_like(d, dtype=torch.bool)
            mask.scatter_(1, topk.indices, True)
            mask &= d < cutoff
        else:
            mask = d < cutoff
        i, j = mask.nonzero(as_tuple=True)
        rows.append(i + sa)
        cols.append(j + sb)
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long, device=pos_a.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)])
