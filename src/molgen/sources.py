"""Dataset sources: locate complexes on disk and attach labels.

Two layouts are supported, because the project needs both:

``BioLiPSource``
    The Kaggle *protein-ligand-binding-site* dataset — a non-redundant BioLiP
    export: one receptor PDB per assembly, one PDB per bound ligand, and a CSV
    of binding residues.  Large (~150 k entries) but carries **no affinity
    labels**, so it trains the generative model.

``PDBBindSource``
    PDBBind / PDBBind+ ``refined-set`` style directories with an ``INDEX_*``
    file.  Smaller but carries **pKd/pKi labels**, so it trains the affinity
    model and provides the CASF core set for held-out evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import chem
from .molfiles import read_ligand_any
from .pdbio import Atoms, parse_pdb

# --------------------------------------------------------------------------


@dataclass
class ComplexSpec:
    """One trainable example, before any parsing."""

    complex_id: str                      # e.g. '1b7z_1_OXL_D' or '3zzf'
    receptor_path: Path
    ligand_path: Path
    ligand_name: str = "LIG"             # CCD code
    receptor_key: str = ""               # grouping key for leakage-free splits
    affinity: float | None = None        # -log10(Kd/Ki), i.e. pK
    binding_residues: list[tuple[str, int]] = field(default_factory=list)

    def load(self) -> tuple[Atoms, Atoms, tuple[np.ndarray, np.ndarray] | None]:
        """Parse receptor + ligand.  Returns ``(receptor, ligand, bonds|None)``."""
        receptor = parse_pdb(self.receptor_path)
        ligand, bond_index, bond_order = read_ligand_any(self.ligand_path, self.ligand_name)
        bonds = None if bond_index is None else (bond_index, bond_order)
        return receptor, ligand, bonds


# --------------------------------------------------------------------------
# path discovery (Kaggle input roots move around between datasets)
# --------------------------------------------------------------------------
DEFAULT_SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/working",
    ".",
    "..",
    str(Path.home() / "Desktop"),
]


def find_dir(name: str, roots: list[str] | None = None, max_depth: int = 4) -> Path | None:
    """Breadth-first search for a directory called ``name`` under ``roots``."""
    roots = roots or DEFAULT_SEARCH_ROOTS
    frontier = [Path(r) for r in roots if Path(r).exists()]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for d in frontier:
            try:
                for child in sorted(d.iterdir()):
                    if not child.is_dir():
                        continue
                    if child.name == name:
                        return child
                    nxt.append(child)
            except (PermissionError, OSError):
                continue
        frontier = nxt
    return None


def find_file(pattern: str, roots: list[str] | None = None, max_depth: int = 4) -> Path | None:
    roots = roots or DEFAULT_SEARCH_ROOTS
    frontier = [Path(r) for r in roots if Path(r).exists()]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for d in frontier:
            try:
                for child in sorted(d.iterdir()):
                    if child.is_dir():
                        nxt.append(child)
                    elif re.fullmatch(pattern, child.name):
                        return child
            except (PermissionError, OSError):
                continue
        frontier = nxt
    return None


# --------------------------------------------------------------------------
# BioLiP-style source
# --------------------------------------------------------------------------
_RESIDUE_RE = re.compile(r"([A-Z])(-?\d+)([A-Z]?)")


def parse_binding_residues(field_value: str) -> list[tuple[str, int]]:
    """Parse BioLiP's ``'A:D60 Y92 T117'`` into ``[('A', 60), ('A', 92), ...]``.

    Multiple chains appear as space-separated ``chain:res`` groups, so the
    current chain is tracked while scanning left to right.
    """
    out: list[tuple[str, int]] = []
    if not isinstance(field_value, str):
        return out
    chain = "A"
    for token in field_value.replace(",", " ").split():
        if ":" in token:
            chain, _, token = token.partition(":")
            chain = chain.strip() or "A"
            if not token:
                continue
        m = _RESIDUE_RE.fullmatch(token.strip())
        if m:
            out.append((chain, int(m.group(2))))
    return out


class BioLiPSource:
    """Index the BioLiP-style Kaggle dataset.

    Parameters
    ----------
    dedup
        BioLiP lists the same chemical ligand once per copy in the asymmetric
        unit (``1b7z_1_OXL_D`` and ``1b7z_1_OXL_E`` are the same molecule in two
        sites).  With ``dedup=True`` only the first copy per
        ``(assembly, ligand_id)`` is kept, which roughly halves the set and
        removes the dominant source of train/test near-duplication.
    """

    def __init__(
        self,
        annotation_csv: str | Path | None = None,
        receptor_dir: str | Path | None = None,
        ligand_dir: str | Path | None = None,
        roots: list[str] | None = None,
    ):
        self.annotation_csv = Path(annotation_csv) if annotation_csv else find_file(
            r"PL_nonredund_annotation\.csv", roots
        )
        self.receptor_dir = Path(receptor_dir) if receptor_dir else find_dir("nonredund_rec", roots)
        self.ligand_dir = Path(ligand_dir) if ligand_dir else find_dir("nonredund_lig", roots)
        missing = [
            n for n, v in [
                ("annotation csv", self.annotation_csv),
                ("nonredund_rec", self.receptor_dir),
                ("nonredund_lig", self.ligand_dir),
            ] if v is None
        ]
        if missing:
            raise FileNotFoundError(
                "BioLiP source incomplete, could not locate: "
                + ", ".join(missing)
                + ". Pass the paths explicitly."
            )

    def specs(
        self,
        dedup: bool = True,
        limit: int | None = None,
        require_files: bool = True,
    ) -> list[ComplexSpec]:
        import pandas as pd

        df = pd.read_csv(self.annotation_csv)
        needed = {"Assembly_ID", "Ligand_file", "Ligand_ID"}
        if not needed.issubset(df.columns):
            raise ValueError(f"annotation csv missing columns: {needed - set(df.columns)}")

        if dedup:
            df = df.drop_duplicates(subset=["Assembly_ID", "Ligand_ID"], keep="first")

        # Cheap pre-filter on the CCD code before touching the filesystem.
        bad = chem.LIGAND_BLACKLIST | chem.STANDARD_AA_SET | chem.NUCLEOTIDE_RESIDUES
        df = df[~df["Ligand_ID"].astype(str).str.strip().str.upper().isin(bad)]

        rec_cache = {p.stem for p in self.receptor_dir.glob("*.pdb")} if require_files else None
        lig_cache = {p.stem for p in self.ligand_dir.glob("*.pdb")} if require_files else None

        specs: list[ComplexSpec] = []
        for row in df.itertuples(index=False):
            assembly = str(row.Assembly_ID).strip()
            lig_file = str(row.Ligand_file).strip()
            if require_files and (assembly not in rec_cache or lig_file not in lig_cache):
                continue
            specs.append(
                ComplexSpec(
                    complex_id=lig_file,
                    receptor_path=self.receptor_dir / f"{assembly}.pdb",
                    ligand_path=self.ligand_dir / f"{lig_file}.pdb",
                    ligand_name=str(row.Ligand_ID).strip().upper(),
                    receptor_key=assembly.split("_")[0].lower(),
                    binding_residues=parse_binding_residues(getattr(row, "Binding_residue", "")),
                )
            )
            if limit and len(specs) >= limit:
                break
        return specs


# --------------------------------------------------------------------------
# PDBBind source
# --------------------------------------------------------------------------
_INDEX_RE = re.compile(
    r"^(?P<code>\w{4})\s+(?P<res>[\d.]+|NMR)\s+(?P<year>\d{4})\s+(?P<pk>[-\d.]+)\s+(?P<meas>\S+)"
)


def parse_pdbbind_index(path: str | Path) -> dict[str, dict]:
    """Parse ``INDEX_refined_data.*`` / ``INDEX_general_PL_data.*``.

    Each data line looks like::

        3zzf  2.20  2012   0.40  Ki=400mM      // 3zzf.pdf (NLG)

    where column 4 is already ``-log10(Kd/Ki)``.  Returns ``{code: {...}}``.
    """
    out: dict[str, dict] = {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _INDEX_RE.match(raw.strip())
        if not m:
            continue
        lig = re.search(r"\(([^)]+)\)\s*$", raw.strip())
        out[m.group("code").lower()] = {
            "pk": float(m.group("pk")),
            "resolution": m.group("res"),
            "year": int(m.group("year")),
            "measure": m.group("meas"),
            "ligand_name": (lig.group(1) if lig else "LIG")[:3].upper(),
        }
    return out


class PDBBindSource:
    """Index a PDBBind-style ``<set>/<code>/<code>_{protein,ligand,pocket}.*`` tree."""

    def __init__(
        self,
        set_dir: str | Path | None = None,
        index_file: str | Path | None = None,
        roots: list[str] | None = None,
    ):
        self.set_dir = Path(set_dir) if set_dir else (
            find_dir("refined-set", roots) or find_dir("general-set", roots)
        )
        if self.set_dir is None:
            raise FileNotFoundError("could not locate a PDBBind set directory")
        self.index_file = Path(index_file) if index_file else find_file(
            r"INDEX_(refined|general_PL)_data\..*", roots or [str(self.set_dir.parent)]
        )
        self.labels = parse_pdbbind_index(self.index_file) if self.index_file else {}

    def specs(
        self,
        require_affinity: bool = True,
        prefer_pocket_file: bool = True,
        limit: int | None = None,
    ) -> list[ComplexSpec]:
        """Build specs.

        ``prefer_pocket_file`` uses PDBBind's own ``*_pocket.pdb`` (a 10 A cut
        around the ligand) as the receptor when present — a large parsing speed
        win over the full ``*_protein.pdb``, with identical pocket graphs at the
        default 8 A radius.
        """
        specs: list[ComplexSpec] = []
        for entry in sorted(self.set_dir.iterdir()):
            if not entry.is_dir():
                continue
            code = entry.name.lower()
            if len(code) != 4:
                continue
            label = self.labels.get(code)
            if require_affinity and label is None:
                continue

            ligand = None
            for ext in ("sdf", "mol2"):
                cand = entry / f"{code}_ligand.{ext}"
                if cand.exists():
                    ligand = cand
                    break
            if ligand is None:
                continue

            receptor = entry / f"{code}_pocket.pdb"
            if not (prefer_pocket_file and receptor.exists()):
                receptor = entry / f"{code}_protein.pdb"
            if not receptor.exists():
                continue

            specs.append(
                ComplexSpec(
                    complex_id=code,
                    receptor_path=receptor,
                    ligand_path=ligand,
                    ligand_name=(label or {}).get("ligand_name", "LIG"),
                    receptor_key=code,
                    affinity=(label or {}).get("pk"),
                )
            )
            if limit and len(specs) >= limit:
                break
        return specs


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def split_by_receptor(
    specs: list[ComplexSpec],
    fractions: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 0,
    holdout_ids: set[str] | None = None,
) -> dict[str, list[ComplexSpec]]:
    """Group-aware split so no PDB entry appears in two splits.

    Splitting examples at random would put ``1b7z_1_OXL_D`` in train and
    ``1b7z_1_OXL_E`` in test — the same protein and the same ligand — and the
    reported metrics would be meaningless.  Grouping on the 4-character PDB
    code is the minimum defensible unit; pass ``holdout_ids`` (e.g. the CASF
    core set) to force specific codes into the test split.
    """
    groups: dict[str, list[ComplexSpec]] = {}
    for s in specs:
        groups.setdefault(s.receptor_key or s.complex_id, []).append(s)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    forced = {k for k in keys if holdout_ids and k in holdout_ids}
    keys = [k for k in keys if k not in forced]

    n = len(keys)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    buckets = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :] + sorted(forced),
    }
    return {name: [s for k in ks for s in groups[k]] for name, ks in buckets.items()}
