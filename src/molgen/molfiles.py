"""Minimal SDF / MOL2 readers.

PDBBind ships ligands as ``.sdf`` and ``.mol2``, both of which carry a real
bond block — far better than perceiving bonds from geometry.  These readers are
deliberately dependency-free so the dataset builder never needs RDKit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pdbio import Atoms

_MOL2_ORDER = {"1": 1, "2": 2, "3": 3, "ar": 2, "am": 1, "du": 1, "un": 1, "nc": 1}


def _as_atoms(elements: list[str], pos: list[list[float]], resname: str) -> Atoms:
    n = len(elements)
    return Atoms(
        record=np.array(["HETATM"] * n, dtype="<U6"),
        serial=np.arange(1, n + 1, dtype=np.int64),
        name=np.array([f"{e}{i + 1}" for i, e in enumerate(elements)], dtype="<U4"),
        altloc=np.array([""] * n, dtype="<U1"),
        resname=np.array([resname[:3]] * n, dtype="<U3"),
        chain=np.array(["L"] * n, dtype="<U2"),
        resseq=np.ones(n, dtype=np.int64),
        icode=np.array([""] * n, dtype="<U1"),
        pos=np.asarray(pos, dtype=np.float32).reshape(n, 3),
        occupancy=np.ones(n, dtype=np.float32),
        bfactor=np.zeros(n, dtype=np.float32),
        element=np.array(elements, dtype="<U2"),
    )


def read_sdf(path: str | Path, resname: str = "LIG") -> tuple[Atoms, np.ndarray, np.ndarray]:
    """Read the first record of a V2000 SDF.

    Returns ``(atoms, bond_index, bond_order)`` with hydrogens removed and
    bond indices remapped onto the heavy-atom numbering.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: truncated SDF")
    counts = lines[3]
    n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])

    elements: list[str] = []
    pos: list[list[float]] = []
    for line in lines[4 : 4 + n_atoms]:
        pos.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        elements.append(line[31:34].strip().capitalize())

    bi: list[tuple[int, int]] = []
    bo: list[int] = []
    for line in lines[4 + n_atoms : 4 + n_atoms + n_bonds]:
        a, b = int(line[0:3]) - 1, int(line[3:6]) - 1
        order = int(line[6:9]) if line[6:9].strip() else 1
        bi.append((a, b))
        bo.append(4 if order == 4 else order)  # 4 = aromatic in SDF

    return _strip_hydrogens(elements, pos, bi, bo, resname)


def read_mol2(path: str | Path, resname: str = "LIG") -> tuple[Atoms, np.ndarray, np.ndarray]:
    """Read the first molecule of a MOL2 file."""
    elements: list[str] = []
    pos: list[list[float]] = []
    bi: list[tuple[int, int]] = []
    bo: list[int] = []
    section = None
    serial_to_idx: dict[int, int] = {}

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("@<TRIPOS>"):
            tag = line[len("@<TRIPOS>") :].upper()
            if section == "BOND" and tag == "MOLECULE":
                break  # second record — stop
            section = tag
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if section == "ATOM" and len(parts) >= 6:
            serial_to_idx[int(parts[0])] = len(elements)
            pos.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elements.append(parts[5].split(".")[0].strip().capitalize())
        elif section == "BOND" and len(parts) >= 4:
            a = serial_to_idx.get(int(parts[1]))
            b = serial_to_idx.get(int(parts[2]))
            if a is None or b is None:
                continue
            bi.append((a, b))
            kind = parts[3].lower()
            bo.append(4 if kind == "ar" else _MOL2_ORDER.get(kind, 1))

    return _strip_hydrogens(elements, pos, bi, bo, resname)


def _strip_hydrogens(
    elements: list[str],
    pos: list[list[float]],
    bi: list[tuple[int, int]],
    bo: list[int],
    resname: str,
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    keep = [i for i, e in enumerate(elements) if e not in ("H", "D")]
    remap = {old: new for new, old in enumerate(keep)}
    hv_elements = [elements[i] for i in keep]
    hv_pos = [pos[i] for i in keep]

    pairs, orders = [], []
    for (a, b), o in zip(bi, bo):
        if a in remap and b in remap:
            pairs.append((remap[a], remap[b]))
            # aromatic (4) is folded to "1.5 -> 2" for the 3-way one-hot; the
            # exact Kekule order is irrelevant to the geometric models here.
            orders.append(2 if o == 4 else min(o, 3))

    bond_index = np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), np.int64)
    bond_order = np.array(orders, dtype=np.int64) if orders else np.zeros((0,), np.int64)
    return _as_atoms(hv_elements, hv_pos, resname), bond_index, bond_order


def read_ligand_any(path: str | Path, resname: str = "LIG"):
    """Dispatch on extension; PDB ligands have no bond block so bonds are None."""
    from .pdbio import parse_pdb

    suffix = Path(path).suffix.lower()
    if suffix == ".sdf" or suffix == ".mol":
        return read_sdf(path, resname)
    if suffix == ".mol2":
        return read_mol2(path, resname)
    if suffix in (".pdb", ".ent", ".gz"):
        return parse_pdb(path), None, None
    raise ValueError(f"unsupported ligand format: {path}")
