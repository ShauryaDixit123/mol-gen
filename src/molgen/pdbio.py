"""Fast, dependency-free PDB/SDF I/O.

Biopython's ``PDBParser`` is convenient but costs ~30-60 ms per structure and
emits a storm of ``PDBConstructionWarning``s.  At 150 k complexes that alone is
hours of Kaggle runtime, so this module reads the fixed-column format directly
into numpy arrays.

Column layout follows the PDB v3.3 spec (0-indexed slices)::

    record  0:6    serial  6:11   name   12:16  altLoc 16
    resName 17:20  chain   21     resSeq 22:26  iCode  26
    x       30:38  y       38:46  z      46:54
    occ     54:60  bfac    60:66  elem   76:78  charge 78:80
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TWO_LETTER = {
    "CL", "BR", "SE", "ZN", "MG", "NA", "CA", "FE", "MN", "CU", "NI", "CO",
    "CD", "HG", "PT", "AU", "AG", "LI", "SI", "AL", "BA", "SR", "RB", "CS",
    "PB", "TL", "AS", "SB", "TE", "XE", "KR", "MO", "WO", "VA",
}


def guess_element(atom_name: str, resname: str = "") -> str:
    """Recover the element when the element column is blank or wrong.

    PDB atom names are right-justified in columns 13-16 with the element in
    13-14, so a leading space means a one-letter element.  Files written by
    older tools break that rule, hence the fallbacks.
    """
    raw = atom_name
    name = atom_name.strip()
    if not name:
        return "C"
    # Properly formatted: element occupies the first two columns.
    if len(raw) >= 2 and raw[0] != " ":
        two = raw[:2].strip().upper()
        if two in _TWO_LETTER:
            return two.capitalize()
    # Digits and primes are never elements.
    stripped = name.lstrip("0123456789")
    if not stripped:
        return "C"
    two = stripped[:2].upper()
    if two in _TWO_LETTER and resname.strip().upper() not in {"HOH", "DOD"}:
        return two.capitalize()
    first = stripped[0].upper()
    if first == "D":  # deuterium
        return "H"
    return first


@dataclass
class Atoms:
    """Columnar atom table.  All arrays share the same length."""

    record: np.ndarray      # '<U6'   ATOM / HETATM
    serial: np.ndarray      # int64
    name: np.ndarray        # '<U4'
    altloc: np.ndarray      # '<U1'
    resname: np.ndarray     # '<U3'
    chain: np.ndarray       # '<U2'
    resseq: np.ndarray      # int64
    icode: np.ndarray       # '<U1'
    pos: np.ndarray         # (N, 3) float32
    occupancy: np.ndarray   # float32
    bfactor: np.ndarray     # float32
    element: np.ndarray     # '<U2'

    def __len__(self) -> int:
        return len(self.serial)

    def __getitem__(self, mask) -> "Atoms":
        return Atoms(
            record=self.record[mask], serial=self.serial[mask], name=self.name[mask],
            altloc=self.altloc[mask], resname=self.resname[mask], chain=self.chain[mask],
            resseq=self.resseq[mask], icode=self.icode[mask], pos=self.pos[mask],
            occupancy=self.occupancy[mask], bfactor=self.bfactor[mask],
            element=self.element[mask],
        )

    @property
    def is_hetatm(self) -> np.ndarray:
        return self.record == "HETATM"

    def heavy(self) -> "Atoms":
        return self[(self.element != "H") & (self.element != "D")]

    def residue_keys(self) -> np.ndarray:
        """``(chain, resseq, icode)`` tuples as a structured array for grouping."""
        return np.array(
            list(zip(self.chain, self.resseq, self.icode)),
            dtype=[("chain", "<U2"), ("resseq", "i8"), ("icode", "<U1")],
        )

    def residue_group_ids(self) -> np.ndarray:
        """Dense 0..R-1 id per atom identifying its residue (order preserved)."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        keys = self.residue_keys()
        ids = np.zeros(len(self), dtype=np.int64)
        current = 0
        for i in range(1, len(self)):
            if keys[i] != keys[i - 1]:
                current += 1
            ids[i] = current
        return ids


def _open_text(path: str | Path) -> io.TextIOBase:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def parse_pdb(
    path: str | Path,
    keep_hydrogens: bool = False,
    first_model_only: bool = True,
    keep_altloc: str = "A",
    records: tuple[str, ...] = ("ATOM", "HETATM"),
) -> Atoms:
    """Read a PDB file into an :class:`Atoms` table.

    Parameters
    ----------
    keep_altloc
        Alternate-location indicator to retain alongside the blank one.  Keeping
        every altloc would duplicate atoms and corrupt the geometry.
    """
    rec, ser, nam, alt, rsn, chn, rsq, ico = [], [], [], [], [], [], [], []
    xs, ys, zs, occ, bfc, ele = [], [], [], [], [], []

    with _open_text(path) as fh:
        for line in fh:
            tag = line[:6]
            if tag.startswith("ENDMDL") and first_model_only:
                break
            if not (tag.startswith("ATOM") or tag.startswith("HETATM")):
                continue
            record = "HETATM" if tag.startswith("HETATM") else "ATOM"
            if record not in records:
                continue
            altloc = line[16:17].strip()
            if altloc and altloc != keep_altloc:
                continue
            atom_name = line[12:16]
            resname = line[17:20]
            elem = line[76:78].strip().capitalize() if len(line) >= 78 else ""
            if not elem or elem.isdigit():
                elem = guess_element(atom_name, resname)
            if not keep_hydrogens and elem in ("H", "D"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                resnum = 0
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = len(ser) + 1

            rec.append(record); ser.append(serial); nam.append(atom_name.strip())
            alt.append(altloc); rsn.append(resname.strip()); chn.append(line[21:22].strip())
            rsq.append(resnum); ico.append(line[26:27].strip())
            xs.append(x); ys.append(y); zs.append(z)
            occ.append(_safe_float(line[54:60], 1.0))
            bfc.append(_safe_float(line[60:66], 0.0))
            ele.append(elem)

    return Atoms(
        record=np.array(rec, dtype="<U6"),
        serial=np.array(ser, dtype=np.int64),
        name=np.array(nam, dtype="<U4"),
        altloc=np.array(alt, dtype="<U1"),
        resname=np.array(rsn, dtype="<U3"),
        chain=np.array(chn, dtype="<U2"),
        resseq=np.array(rsq, dtype=np.int64),
        icode=np.array(ico, dtype="<U1"),
        pos=np.array([xs, ys, zs], dtype=np.float32).T.copy() if xs else np.zeros((0, 3), np.float32),
        occupancy=np.array(occ, dtype=np.float32),
        bfactor=np.array(bfc, dtype=np.float32),
        element=np.array(ele, dtype="<U2"),
    )


def _safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_xyz(path: str | Path, elements, pos, comment: str = "") -> None:
    pos = np.asarray(pos)
    with open(path, "w") as fh:
        fh.write(f"{len(pos)}\n{comment}\n")
        for e, (x, y, z) in zip(elements, pos):
            fh.write(f"{e:<3s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def write_sdf_block(
    elements,
    pos,
    bond_index: np.ndarray,
    bond_order: np.ndarray | None = None,
    name: str = "mol",
    props: dict | None = None,
) -> str:
    """Build a V2000 SDF record.  Works without RDKit."""
    pos = np.asarray(pos, dtype=float)
    n_atoms = len(pos)
    n_bonds = bond_index.shape[1] if bond_index.size else 0
    if bond_order is None:
        bond_order = np.ones(n_bonds, dtype=int)

    lines = [name, "  molgen", ""]
    lines.append(f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000")
    for (x, y, z), e in zip(pos, elements):
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {str(e):<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for k in range(n_bonds):
        i = int(bond_index[0, k]) + 1
        j = int(bond_index[1, k]) + 1
        o = int(bond_order[k])
        lines.append(f"{i:>3d}{j:>3d}{o:>3d}  0  0  0  0")
    lines.append("M  END")
    for key, value in (props or {}).items():
        lines.append(f">  <{key}>")
        lines.append(str(value))
        lines.append("")
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


def write_sdf(path: str | Path, records: list[str]) -> None:
    with open(path, "w") as fh:
        fh.writelines(records)


def write_pdb(path: str | Path, atoms: Atoms) -> None:
    with open(path, "w") as fh:
        for k in range(len(atoms)):
            name = atoms.name[k]
            name_field = f" {name:<3s}" if len(name) < 4 else name[:4]
            fh.write(
                f"{atoms.record[k]:<6s}{k + 1:>5d} {name_field}{'':1s}"
                f"{atoms.resname[k]:>3s} {atoms.chain[k] or 'A':<1s}"
                f"{atoms.resseq[k]:>4d}{atoms.icode[k] or '':<1s}   "
                f"{atoms.pos[k, 0]:>8.3f}{atoms.pos[k, 1]:>8.3f}{atoms.pos[k, 2]:>8.3f}"
                f"{atoms.occupancy[k]:>6.2f}{atoms.bfactor[k]:>6.2f}"
                f"          {atoms.element[k]:>2s}\n"
            )
        fh.write("END\n")
