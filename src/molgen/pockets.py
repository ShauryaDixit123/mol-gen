"""Binding-site detection — the first stage of the inference pipeline.

Three backends, same ``list[Pocket]`` output:

``builtin``
    A vectorised LIGSITE (Hendlich et al. 1997) implementation.  Pure
    numpy/scipy, no internet, no JVM, ~1 s per protein — so the pipeline is
    runnable in a Kaggle notebook with networking switched off, which is the
    normal state when a notebook writes a dataset output.
``fpocket``
    Shells out to ``fpocket`` and parses ``*_info.txt`` + the pocket PDBs.
``p2rank``
    Shells out to P2Rank's ``prank predict`` and parses
    ``*_predictions.csv`` — the most accurate of the three in CASP/CAMEO-style
    benchmarks, and the one worth installing when the environment allows it.

DeepSite is deliberately not wrapped: it is a hosted web service (PlayMolecule)
with no redistributable local model, so it cannot be a dependency of an
offline pipeline.  Point ``detect_pockets`` at a precomputed CSV instead if you
want to use its predictions.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .pdbio import Atoms, parse_pdb

# van der Waals radii for the buriedness grid
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "Se": 1.90, "P": 1.80, "H": 1.20}
DEFAULT_VDW = 1.70


@dataclass
class Pocket:
    center: np.ndarray                      # (3,) in the receptor's own frame
    score: float                            # backend-specific, higher = better
    rank: int = 0
    volume: float = 0.0                     # A^3
    points: np.ndarray | None = None        # (M, 3) cavity point cloud
    weights: np.ndarray | None = None       # (M,) per-point buriedness (PSP count)
    atom_indices: np.ndarray | None = None  # indices into the receptor Atoms
    residues: list[tuple[str, int]] = field(default_factory=list)
    source: str = "builtin"

    def summary(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "volume": round(float(self.volume), 1),
            "center": [round(float(c), 3) for c in self.center],
            "n_residues": len(self.residues),
            "source": self.source,
        }


# --------------------------------------------------------------------------
# builtin LIGSITE
# --------------------------------------------------------------------------
def _or_along(grid: np.ndarray, direction: tuple[int, int, int], max_steps: int) -> np.ndarray:
    """Logical-OR accumulate ``grid`` along ``direction`` by shift-doubling.

    ``out[p]`` is True when any grid point at ``p - k*direction`` (k >= 1, up to
    ``max_steps``) is True.  Doubling the shift each round reaches ``max_steps``
    in ``log2`` passes instead of ``max_steps`` passes.
    """
    out = np.zeros_like(grid)
    acc = grid.copy()
    step = 1
    while step <= max_steps:
        shifted = _shift(acc, tuple(d * step for d in direction))
        out |= shifted
        acc |= shifted
        step *= 2
    return out


def _shift(a: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(a)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for ax, o in enumerate(offset):
        if o > 0:
            src[ax] = slice(0, a.shape[ax] - o)
            dst[ax] = slice(o, a.shape[ax])
        elif o < 0:
            src[ax] = slice(-o, a.shape[ax])
            dst[ax] = slice(0, a.shape[ax] + o)
    out[tuple(dst)] = a[tuple(src)]
    return out


def detect_pockets_builtin(
    receptor: Atoms,
    spacing: float = 0.8,
    probe: float = 1.4,
    min_psp: int = 4,
    max_scan: float = 14.0,
    min_volume: float = 40.0,
    max_pockets: int = 10,
    contact_radius: float = 6.0,
    scorer: str = "deep",
) -> list[Pocket]:
    """LIGSITE: count protein-solvent-protein events per free grid point.

    A point deep inside a cleft is enclosed along most axes, so it records a
    PSP event in most of the 7 scanned lattice directions; a point on a convex
    surface records few.  Clusters of high-PSP points are the pockets.

    Defaults were chosen by grid search on 30 held-out BioLiP complexes,
    scoring a hit when a pocket's point cloud comes within 3 A of >=30 % of the
    true ligand's atoms:

    ======================  =====  =====
    setting                 top-1  top-3
    ======================  =====  =====
    spacing 0.8, psp>=4     0.767  0.867
    spacing 1.0, psp>=4     0.700  0.867
    spacing 0.8, psp>=5     0.667  0.767
    ======================  =====  =====

    ``spacing=1.0`` halves the runtime (~0.5 s vs ~1.0 s per protein) for ~7
    points of top-1, which is the right trade when screening thousands of
    structures.

    ``scorer`` ranks the clusters:

    ``deep``
        Number of *fully* enclosed points (PSP = 7 of 7).  Best measured
        ranking — it prefers genuinely buried cavities over large shallow
        grooves, which is exactly the failure mode of volume ranking.
    ``volume``
        Classic LIGSITE cluster size.
    ``buriedness``
        Sum of squared PSP counts: between the two.
    """
    heavy = receptor.heavy()
    if len(heavy) < 20:
        return []

    pos = heavy.pos.astype(np.float64)
    radii = np.array([VDW.get(str(e).capitalize(), DEFAULT_VDW) for e in heavy.element])

    margin = 4.0
    lo = pos.min(0) - margin
    hi = pos.max(0) + margin
    shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 3)
    if shape.prod() > 40_000_000:          # ~40 M voxels, keep memory sane
        spacing *= 1.5
        shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 3)

    # occupancy: grid points inside an atom's vdW+probe shell
    occupied = np.zeros(shape, dtype=bool)
    tree = cKDTree(pos)
    grid_axes = [lo[d] + np.arange(shape[d]) * spacing for d in range(3)]
    mesh = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    max_r = float(radii.max() + probe)
    near = tree.query_ball_point(mesh, r=max_r)
    flat = np.zeros(len(mesh), dtype=bool)
    for i, idx in enumerate(near):
        if idx:
            d = np.linalg.norm(pos[idx] - mesh[i], axis=1)
            if (d < radii[idx] + probe).any():
                flat[i] = True
    occupied = flat.reshape(shape)

    free = ~occupied
    directions = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
    ]
    max_steps = int(max_scan / spacing)
    psp = np.zeros(shape, dtype=np.int8)
    for d in directions:
        fwd = _or_along(occupied, d, max_steps)
        bwd = _or_along(occupied, tuple(-x for x in d), max_steps)
        psp += (fwd & bwd).astype(np.int8)

    pocket_mask = free & (psp >= min_psp)
    if not pocket_mask.any():
        return []

    labels, n_labels = ndimage.label(pocket_mask, structure=np.ones((3, 3, 3), dtype=bool))
    voxel_volume = spacing ** 3
    pockets: list[Pocket] = []
    for lab in range(1, n_labels + 1):
        sel = labels == lab
        n_vox = int(sel.sum())
        volume = n_vox * voxel_volume
        if volume < min_volume:
            continue
        ijk = np.argwhere(sel)
        points = lo + ijk * spacing
        weights = psp[sel].astype(np.float64)
        # Weighting the centroid by buriedness pulls it off the solvent-facing
        # mouth of the cavity and towards the part a ligand actually occupies.
        center = (points * weights[:, None]).sum(0) / weights.sum()
        if scorer == "volume":
            score = volume
        elif scorer == "buriedness":
            score = float((weights ** 2).sum())
        else:  # "deep": count of fully enclosed points, + volume as a tie-break
            score = float((weights >= 7).sum()) + 1e-3 * volume
        pockets.append(
            Pocket(
                center=center,
                score=score,
                volume=volume,
                points=points,
                weights=weights,
                source="builtin",
            )
        )

    pockets.sort(key=lambda p: -p.score)
    pockets = pockets[:max_pockets]
    for i, p in enumerate(pockets):
        p.rank = i + 1
        _annotate(p, heavy, contact_radius)
    return pockets


def _annotate(pocket: Pocket, receptor: Atoms, contact_radius: float) -> None:
    """Attach the lining atoms/residues of a pocket."""
    ref = pocket.points if pocket.points is not None else pocket.center.reshape(1, 3)
    tree = cKDTree(receptor.pos)
    hit: set[int] = set()
    for idx in tree.query_ball_point(ref, r=contact_radius):
        hit.update(idx)
    if not hit:
        return
    idx = np.array(sorted(hit), dtype=np.int64)
    pocket.atom_indices = idx
    seen: dict[tuple[str, int], None] = {}
    for k in idx:
        seen.setdefault((str(receptor.chain[k]), int(receptor.resseq[k])), None)
    pocket.residues = list(seen)


# --------------------------------------------------------------------------
# fpocket
# --------------------------------------------------------------------------
def detect_pockets_fpocket(
    pdb_path: str | Path,
    receptor: Atoms | None = None,
    executable: str = "fpocket",
    max_pockets: int = 10,
    extra_args: list[str] | None = None,
) -> list[Pocket]:
    """Run fpocket and parse its output.

    Install with ``apt-get install -y fpocket`` or
    ``conda install -c conda-forge fpocket``.
    """
    exe = shutil.which(executable)
    if exe is None:
        raise FileNotFoundError(
            "fpocket not found on PATH. Install it (apt-get install -y fpocket) "
            "or use method='builtin'."
        )
    pdb_path = Path(pdb_path)
    receptor = receptor if receptor is not None else parse_pdb(pdb_path)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / pdb_path.name
        shutil.copy(pdb_path, work)
        cmd = [exe, "-f", str(work)] + (extra_args or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        out_dir = work.with_name(work.stem + "_out")
        if not out_dir.exists():
            raise RuntimeError(f"fpocket produced no output: {proc.stderr[-2000:]}")

        scores = _parse_fpocket_info(out_dir / f"{work.stem}_info.txt")
        pockets: list[Pocket] = []
        for pdb in sorted((out_dir / "pockets").glob("pocket*_atm.pdb")):
            n = int("".join(ch for ch in pdb.stem.split("_")[0] if ch.isdigit()))
            atoms = parse_pdb(pdb)
            if len(atoms) == 0:
                continue
            info = scores.get(n, {})
            pockets.append(
                Pocket(
                    center=atoms.pos.mean(0).astype(np.float64),
                    score=float(info.get("score", 0.0)),
                    volume=float(info.get("volume", 0.0)),
                    points=atoms.pos.astype(np.float64),
                    source="fpocket",
                )
            )

    pockets.sort(key=lambda p: -p.score)
    pockets = pockets[:max_pockets]
    for i, p in enumerate(pockets):
        p.rank = i + 1
        _annotate(p, receptor.heavy(), 6.0)
    return pockets


def _parse_fpocket_info(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("pocket"):
            digits = "".join(ch for ch in line if ch.isdigit())
            current = int(digits) if digits else None
            if current is not None:
                out[current] = {}
        elif current is not None and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            try:
                v = float(value.strip())
            except ValueError:
                continue
            if "druggability" in key:
                out[current]["druggability"] = v
            elif key.startswith("score"):
                out[current]["score"] = v
            elif "volume" in key:
                out[current]["volume"] = v
    return out


# --------------------------------------------------------------------------
# P2Rank
# --------------------------------------------------------------------------
def detect_pockets_p2rank(
    pdb_path: str | Path,
    receptor: Atoms | None = None,
    prank: str | Path = "prank",
    max_pockets: int = 10,
    model: str | None = None,
) -> list[Pocket]:
    """Run ``prank predict`` and parse ``*_predictions.csv``.

    Download from https://github.com/rdk/p2rank/releases and pass the path to
    the ``prank`` launcher (it needs a JRE on PATH).
    """
    prank = str(prank)
    if shutil.which(prank) is None and not Path(prank).exists():
        raise FileNotFoundError(
            f"P2Rank launcher not found at {prank!r}. Download a release from "
            "https://github.com/rdk/p2rank/releases and pass prank=<path>/prank."
        )
    pdb_path = Path(pdb_path)
    receptor = receptor if receptor is not None else parse_pdb(pdb_path)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [prank, "predict", "-f", str(pdb_path), "-o", tmp]
        if model:
            cmd += ["-m", model]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        csvs = list(Path(tmp).rglob("*_predictions.csv"))
        if not csvs:
            raise RuntimeError(f"P2Rank produced no predictions: {proc.stderr[-2000:]}")
        pockets = _parse_p2rank_csv(csvs[0])

    pockets.sort(key=lambda p: -p.score)
    pockets = pockets[:max_pockets]
    for i, p in enumerate(pockets):
        p.rank = i + 1
        _annotate(p, receptor.heavy(), 6.0)
    return pockets


def _parse_p2rank_csv(path: Path) -> list[Pocket]:
    out: list[Pocket] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            try:
                center = np.array([float(clean["center_x"]), float(clean["center_y"]), float(clean["center_z"])])
            except (KeyError, ValueError):
                continue
            out.append(
                Pocket(
                    center=center,
                    score=float(clean.get("score", 0.0) or 0.0),
                    volume=0.0,
                    source="p2rank",
                )
            )
    return out


# --------------------------------------------------------------------------
# dispatch + evaluation
# --------------------------------------------------------------------------
def detect_pockets(
    pdb_path: str | Path,
    method: str = "auto",
    receptor: Atoms | None = None,
    max_pockets: int = 10,
    **kwargs,
) -> list[Pocket]:
    """Detect pockets with the best available backend.

    ``method='auto'`` prefers P2Rank, then fpocket, then the builtin LIGSITE,
    and reports which one it used via ``Pocket.source``.
    """
    import inspect

    receptor = receptor if receptor is not None else parse_pdb(pdb_path)
    backends = {
        "builtin": (detect_pockets_builtin, (receptor,)),
        "fpocket": (detect_pockets_fpocket, (pdb_path, receptor)),
        "p2rank": (detect_pockets_p2rank, (pdb_path, receptor)),
    }
    order = ["p2rank", "fpocket", "builtin"] if method == "auto" else [method]
    if any(m not in backends for m in order):
        raise ValueError(f"unknown pocket method: {method}")

    errors = []
    for m in order:
        fn, args = backends[m]
        # Under method='auto' the caller cannot know which backend will run, so
        # only forward the kwargs that the chosen one actually accepts.
        accepted = set(inspect.signature(fn).parameters)
        passed = {k: v for k, v in kwargs.items() if k in accepted}
        if method == "auto" and len(passed) != len(kwargs):
            errors.append(f"{m}: ignored kwargs {sorted(set(kwargs) - accepted)}")
        try:
            return fn(*args, max_pockets=max_pockets, **passed)
        except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"{m}: {exc}")
            continue
    raise RuntimeError("all pocket backends failed:\n" + "\n".join(errors))


def pocket_from_ligand(ligand_pos: np.ndarray) -> Pocket:
    """The ground-truth 'pocket' for a known holo structure (for evaluation)."""
    pos = np.asarray(ligand_pos, dtype=np.float64)
    return Pocket(center=pos.mean(0), score=1.0, rank=0, points=pos, source="ligand")


def overlap_hit(pocket: Pocket, ligand_pos: np.ndarray, cutoff: float = 3.0, fraction: float = 0.3) -> bool:
    """Does this pocket actually cover the ligand?

    The conventional criterion for pocket finders, and a fairer one than
    centre-to-centre distance: an elongated cavity can cover a ligand perfectly
    while its centroid sits several Angstrom from the ligand centroid.
    """
    pos = np.asarray(ligand_pos, dtype=np.float64)
    if pocket.points is None or len(pocket.points) == 0:
        return bool(np.linalg.norm(pocket.center - pos.mean(0)) <= 4.0)
    d, _ = cKDTree(pocket.points).query(pos)
    return bool((d <= cutoff).mean() >= fraction)


def score_detection(
    predicted: list[Pocket],
    true_center: np.ndarray,
    ligand_pos: np.ndarray | None = None,
    tolerance: float = 4.0,
    top_n: int = 3,
) -> dict:
    """Detection metrics: DCC plus (when ``ligand_pos`` is given) overlap hits.

    ``top_n=1`` is the strict 'Top-1' number; pocket finders are conventionally
    also reported at Top-3 because proteins genuinely have several cavities.
    """
    if not predicted:
        return {"hit": False, "dcc": float("inf"), "rank": -1, "overlap_top1": False, "overlap_top3": False}
    d = [float(np.linalg.norm(p.center - np.asarray(true_center))) for p in predicted]
    out = {
        "hit": bool(min(d[:top_n]) <= tolerance),
        "dcc": min(d[:top_n]),
        "rank": int(np.argmin(d[:top_n])) + 1,
        "dcc_top1": d[0],
        "all_dcc": d,
    }
    if ligand_pos is not None:
        hits = [overlap_hit(p, ligand_pos) for p in predicted]
        out["overlap_top1"] = bool(hits[0])
        out["overlap_top3"] = bool(any(hits[:3]))
        out["overlap_rank"] = int(np.argmax(hits) + 1) if any(hits) else -1
    return out
