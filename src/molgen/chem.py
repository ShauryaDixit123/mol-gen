"""Chemistry constants, vocabularies and geometry-based bond inference.

Everything here is pure-python/numpy so it works on a Kaggle image with no
RDKit / OpenBabel installed.  RDKit is used *opportunistically* elsewhere
(``molgen.evaluate``) but is never required to build the graph dataset.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# element vocabularies
# --------------------------------------------------------------------------
# Ligand atom types the generative model is allowed to emit.  Keep this list
# frozen once you start training: it defines the categorical dimension of the
# diffusion model.
LIGAND_ELEMENTS: list[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Se"]
LIGAND_ELEMENT_INDEX: dict[str, int] = {e: i for i, e in enumerate(LIGAND_ELEMENTS)}
NUM_LIGAND_ELEMENTS = len(LIGAND_ELEMENTS)

# Protein heavy atoms.  'other' absorbs metals/modified residues.
POCKET_ELEMENTS: list[str] = ["C", "N", "O", "S", "Se"]
POCKET_ELEMENT_INDEX: dict[str, int] = {e: i for i, e in enumerate(POCKET_ELEMENTS)}
NUM_POCKET_ELEMENTS = len(POCKET_ELEMENTS) + 1  # + 'other' bucket

# Covalent radii (Angstrom), Cordero et al. 2008.  Used for bond perception.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20,
    "I": 1.39, "Na": 1.66, "Mg": 1.41, "K": 2.03, "Ca": 1.76, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
}
DEFAULT_RADIUS = 1.5

# Max number of covalent neighbours tolerated per element during perception.
MAX_VALENCE: dict[str, int] = {
    "H": 1, "B": 4, "C": 4, "N": 4, "O": 2, "F": 1, "P": 5, "S": 6,
    "Cl": 1, "Se": 4, "Br": 1, "I": 1,
}

ATOMIC_WEIGHT: dict[str, float] = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Se": 78.97,
    "Br": 79.904, "I": 126.90,
}

# Pauling electronegativity — a cheap, useful continuous node feature.
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Se": 2.55, "Br": 2.96, "I": 2.66,
}

# --------------------------------------------------------------------------
# amino acids
# --------------------------------------------------------------------------
AMINO_ACIDS: list[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_INDEX: dict[str, int] = {a: i for i, a in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS) + 1  # + UNK

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE: dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}

# Residue-level physico-chemical descriptors.
# charge @pH7, Kyte-Doolittle hydropathy, residue mass, polar, aromatic,
# h-bond donors, h-bond acceptors.
AA_PROPERTIES: dict[str, dict[str, float]] = {
    "ALA": {"charge": 0.0, "hydropathy": 1.8, "weight": 71.08, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "ARG": {"charge": 1.0, "hydropathy": -4.5, "weight": 156.19, "polar": 1, "aromatic": 0, "donor": 3, "acceptor": 0},
    "ASN": {"charge": 0.0, "hydropathy": -3.5, "weight": 114.10, "polar": 1, "aromatic": 0, "donor": 1, "acceptor": 1},
    "ASP": {"charge": -1.0, "hydropathy": -3.5, "weight": 115.09, "polar": 1, "aromatic": 0, "donor": 0, "acceptor": 2},
    "CYS": {"charge": 0.0, "hydropathy": 2.5, "weight": 103.14, "polar": 0, "aromatic": 0, "donor": 1, "acceptor": 0},
    "GLN": {"charge": 0.0, "hydropathy": -3.5, "weight": 128.13, "polar": 1, "aromatic": 0, "donor": 1, "acceptor": 1},
    "GLU": {"charge": -1.0, "hydropathy": -3.5, "weight": 129.12, "polar": 1, "aromatic": 0, "donor": 0, "acceptor": 2},
    "GLY": {"charge": 0.0, "hydropathy": -0.4, "weight": 57.05, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "HIS": {"charge": 0.1, "hydropathy": -3.2, "weight": 137.14, "polar": 1, "aromatic": 1, "donor": 1, "acceptor": 1},
    "ILE": {"charge": 0.0, "hydropathy": 4.5, "weight": 113.16, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "LEU": {"charge": 0.0, "hydropathy": 3.8, "weight": 113.16, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "LYS": {"charge": 1.0, "hydropathy": -3.9, "weight": 128.17, "polar": 1, "aromatic": 0, "donor": 1, "acceptor": 0},
    "MET": {"charge": 0.0, "hydropathy": 1.9, "weight": 131.20, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "PHE": {"charge": 0.0, "hydropathy": 2.8, "weight": 147.18, "polar": 0, "aromatic": 1, "donor": 0, "acceptor": 0},
    "PRO": {"charge": 0.0, "hydropathy": -1.6, "weight": 97.12, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "SER": {"charge": 0.0, "hydropathy": -0.8, "weight": 87.08, "polar": 1, "aromatic": 0, "donor": 1, "acceptor": 1},
    "THR": {"charge": 0.0, "hydropathy": -0.7, "weight": 101.11, "polar": 1, "aromatic": 0, "donor": 1, "acceptor": 1},
    "TRP": {"charge": 0.0, "hydropathy": -0.9, "weight": 186.21, "polar": 0, "aromatic": 1, "donor": 1, "acceptor": 0},
    "TYR": {"charge": 0.0, "hydropathy": -1.3, "weight": 163.18, "polar": 1, "aromatic": 1, "donor": 1, "acceptor": 1},
    "VAL": {"charge": 0.0, "hydropathy": 4.2, "weight": 117.15, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
    "UNK": {"charge": 0.0, "hydropathy": 0.0, "weight": 110.0, "polar": 0, "aromatic": 0, "donor": 0, "acceptor": 0},
}
AA_PROPERTY_KEYS = ["charge", "hydropathy", "weight", "polar", "aromatic", "donor", "acceptor"]
# Normalisation constants so the numeric block is roughly unit-scale.
AA_PROPERTY_SCALE = np.array([1.0, 4.5, 186.21, 1.0, 1.0, 3.0, 2.0], dtype=np.float32)

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# Heavy-atom covalent bond templates for the standard residues.  Used to tag
# pocket edges as covalent vs. non-bonded contacts.
RESIDUE_BONDS: dict[str, list[tuple[str, str]]] = {
    "_BACKBONE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT")],
    "ALA": [("CA", "CB")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"),
            ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"),
            ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"),
            ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "NE1"),
            ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CG"), ("CE2", "CZ2"),
            ("CZ2", "CH2"), ("CH2", "CZ3"), ("CZ3", "CE3"), ("CE3", "CD2")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"),
            ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG"),
            ("CZ", "OH")],
    "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
}


def residue_bond_set(resname: str) -> set[frozenset[str]]:
    """Covalent heavy-atom bonds (as unordered atom-name pairs) of a residue."""
    bonds = RESIDUE_BONDS["_BACKBONE"] + RESIDUE_BONDS.get(resname.upper(), [])
    return {frozenset(b) for b in bonds}


# Heavy-atom names occurring in the standard residues.  Fixing the vocabulary
# lets every per-atom property become an integer index, so features and the
# covalent-bond flag are table lookups instead of per-atom python.
POCKET_ATOM_NAMES: list[str] = [
    "N", "CA", "C", "O", "OXT", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1", "ND2", "NE",
    "NE1", "NE2", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2", "OG", "OG1",
    "OH", "SD", "SG", "SE",
]
ATOM_NAME_INDEX: dict[str, int] = {n: i for i, n in enumerate(POCKET_ATOM_NAMES)}
NUM_ATOM_NAMES = len(POCKET_ATOM_NAMES) + 1  # + 'other'
OTHER_NAME_IDX = NUM_ATOM_NAMES - 1

BACKBONE_ROLE: dict[str, int] = {"N": 0, "CA": 1, "C": 2, "O": 3}


def _build_covalent_template() -> np.ndarray:
    """``(NUM_AA, NUM_ATOM_NAMES, NUM_ATOM_NAMES)`` bool: intra-residue bonds.

    ~34 k booleans, built once at import.  Replaces a per-edge python set
    lookup that dominated pocket-graph construction.
    """
    table = np.zeros((NUM_AA, NUM_ATOM_NAMES, NUM_ATOM_NAMES), dtype=bool)
    for res_idx, res in enumerate(AMINO_ACIDS):
        for a, b in RESIDUE_BONDS["_BACKBONE"] + RESIDUE_BONDS.get(res, []):
            ia, ib = ATOM_NAME_INDEX.get(a), ATOM_NAME_INDEX.get(b)
            if ia is None or ib is None:
                continue
            table[res_idx, ia, ib] = True
            table[res_idx, ib, ia] = True
    # UNK residues: allow the backbone only
    for a, b in RESIDUE_BONDS["_BACKBONE"]:
        ia, ib = ATOM_NAME_INDEX.get(a), ATOM_NAME_INDEX.get(b)
        if ia is not None and ib is not None:
            table[NUM_AA - 1, ia, ib] = True
            table[NUM_AA - 1, ib, ia] = True
    return table


COVALENT_TEMPLATE = _build_covalent_template()

# Lookup tables indexed by the integer codes above.
AA_PROPERTY_TABLE = np.array(
    [[AA_PROPERTIES.get(a, AA_PROPERTIES["UNK"])[k] for k in AA_PROPERTY_KEYS] for a in AMINO_ACIDS]
    + [[AA_PROPERTIES["UNK"][k] for k in AA_PROPERTY_KEYS]],
    dtype=np.float32,
) / AA_PROPERTY_SCALE[None, :]

BACKBONE_ROLE_TABLE = np.full(NUM_ATOM_NAMES, -1, dtype=np.int8)
for _n, _r in BACKBONE_ROLE.items():
    BACKBONE_ROLE_TABLE[ATOM_NAME_INDEX[_n]] = _r

IS_BACKBONE_TABLE = np.zeros(NUM_ATOM_NAMES, dtype=bool)
for _n in BACKBONE_ATOMS:
    if _n in ATOM_NAME_INDEX:
        IS_BACKBONE_TABLE[ATOM_NAME_INDEX[_n]] = True

C_NAME_IDX = ATOM_NAME_INDEX["C"]
N_NAME_IDX = ATOM_NAME_INDEX["N"]


def encode_atom_names(names) -> np.ndarray:
    return np.array(
        [ATOM_NAME_INDEX.get(str(n).strip().upper(), OTHER_NAME_IDX) for n in names],
        dtype=np.int8,
    )


def encode_residues(resnames) -> np.ndarray:
    return np.array(
        [AA_INDEX.get(str(r).strip().upper(), NUM_AA - 1) for r in resnames], dtype=np.int8
    )


def encode_elements(elements, vocab: list[str]) -> np.ndarray:
    index = {e: i for i, e in enumerate(vocab)}
    return np.array(
        [index.get(str(e).strip().capitalize(), len(vocab)) for e in elements], dtype=np.int8
    )


# --------------------------------------------------------------------------
# ligand filtering
# --------------------------------------------------------------------------
# Ions, cryo-protectants, buffers and crystallisation additives.  BioLiP keeps
# many of these as "ligands"; they are useless as generative targets.
LIGAND_BLACKLIST: set[str] = {
    # waters / ions
    "HOH", "DOD", "NA", "K", "LI", "CS", "RB", "MG", "CA", "ZN", "MN", "FE",
    "FE2", "CO", "NI", "CU", "CU1", "CD", "HG", "BA", "SR", "AL", "AU", "AG",
    "PT", "PB", "TL", "CL", "BR", "IOD", "F", "FLC", "OH", "O", "OXY", "NH4",
    # common buffers / additives / cryo
    "SO4", "PO4", "PI", "2PE", "ACT", "ACY", "EDO", "GOL", "PEG", "PG4", "PGE",
    "1PE", "P6G", "DMS", "DMF", "MES", "TRS", "EPE", "IMD", "MPD", "BME",
    "NO3", "CO3", "CO2", "AZI", "CN", "SCN", "WO4", "MOO", "VO4", "BEF", "ALF",
    "FMT", "EOH", "IPA", "MOH", "ACE", "NH2", "SIN", "TAR", "CIT", "MLA",
    "BOG", "LDA", "C8E", "LMT", "BNG", "SUC", "TRE", "BCT", "UNX", "UNL",
    "ARS", "SE4", "CAC", "NCO", "HEZ", "OGA", "12P", "15P", "33O", "XE", "KR",
    "NAG",  # glycosylation, usually not a druglike target
}

NUCLEOTIDE_RESIDUES: set[str] = {
    "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DI", "DU", "T",
    "5MC", "PSU", "7MG", "1MA", "2MG", "H2U", "OMC", "OMG",
}

STANDARD_AA_SET: set[str] = set(AMINO_ACIDS) | {
    "MSE", "SEC", "PYL", "ASX", "GLX", "UNK", "XAA", "HSD", "HSE", "HSP",
}


def is_interesting_ligand(
    resname: str,
    n_heavy: int,
    elements: list[str] | np.ndarray,
    min_atoms: int = 6,
    max_atoms: int = 64,
) -> tuple[bool, str]:
    """Decide whether a BioLiP/PDBBind ligand is worth learning to generate.

    Returns ``(keep, reason)``; ``reason`` names the rejection rule so the
    dataset builder can report a histogram of why things were dropped.
    """
    rn = resname.strip().upper()
    if rn in LIGAND_BLACKLIST:
        return False, "blacklist"
    if rn in STANDARD_AA_SET:
        return False, "amino_acid"
    if rn in NUCLEOTIDE_RESIDUES:
        return False, "nucleotide"
    if n_heavy < min_atoms:
        return False, "too_small"
    if n_heavy > max_atoms:
        return False, "too_large"
    els = {str(e).strip().capitalize() for e in elements}
    unknown = els - set(LIGAND_ELEMENTS)
    if unknown:
        return False, "bad_element"
    if "C" not in els:
        return False, "inorganic"
    return True, "ok"


# --------------------------------------------------------------------------
# bond perception from 3D coordinates
# --------------------------------------------------------------------------
def infer_bonds(
    pos: np.ndarray,
    elements: list[str] | np.ndarray,
    tolerance: float = 0.45,
    max_dist: float = 2.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Perceive covalent bonds from heavy-atom geometry.

    Two atoms are bonded when ``d < r_i + r_j + tolerance``.  Candidates are
    then pruned shortest-first so no atom exceeds its maximum valence, which
    removes the spurious bonds that a pure distance cut produces in crowded
    regions (fused rings, phosphates).

    Returns
    -------
    bond_index : (2, E) int64, upper-triangular (i < j), one entry per bond
    bond_order : (E,) int64 in {1, 2, 3} — a *geometric guess* from bond length
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    els = [str(e).strip().capitalize() for e in elements]
    radii = np.array([COVALENT_RADII.get(e, DEFAULT_RADIUS) for e in els])
    valence_cap = np.array([MAX_VALENCE.get(e, 4) for e in els])

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    cutoff = radii[:, None] + radii[None, :] + tolerance
    np.fill_diagonal(dist, np.inf)

    cand_i, cand_j = np.where((dist < np.minimum(cutoff, max_dist)))
    keep = cand_i < cand_j
    cand_i, cand_j = cand_i[keep], cand_j[keep]
    if len(cand_i) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    # shortest bonds win when valence is contested
    order = np.argsort(dist[cand_i, cand_j])
    degree = np.zeros(n, dtype=np.int64)
    bonds: list[tuple[int, int]] = []
    for k in order:
        i, j = int(cand_i[k]), int(cand_j[k])
        if degree[i] >= valence_cap[i] or degree[j] >= valence_cap[j]:
            continue
        bonds.append((i, j))
        degree[i] += 1
        degree[j] += 1

    if not bonds:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    bond_index = np.array(bonds, dtype=np.int64).T
    d = dist[bond_index[0], bond_index[1]]
    ref = radii[bond_index[0]] + radii[bond_index[1]]
    # length ratio heuristic: triple ~0.78, double ~0.87, single ~1.0 of r_i+r_j
    ratio = d / ref
    bond_order = np.ones(len(d), dtype=np.int64)
    bond_order[ratio < 0.91] = 2
    bond_order[ratio < 0.80] = 3
    return bond_index, bond_order


def element_one_hot(elements: list[str] | np.ndarray, vocab: list[str], with_other: bool = True) -> np.ndarray:
    """One-hot encode elements against ``vocab``, optionally with an 'other' bin."""
    width = len(vocab) + (1 if with_other else 0)
    out = np.zeros((len(elements), width), dtype=np.float32)
    index = {e: i for i, e in enumerate(vocab)}
    for k, e in enumerate(elements):
        i = index.get(str(e).strip().capitalize(), -1)
        if i >= 0:
            out[k, i] = 1.0
        elif with_other:
            out[k, -1] = 1.0
    return out
