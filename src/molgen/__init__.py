"""molgen — structure-based molecule generation, end to end.

Pipeline
--------
::

    protein PDB
        |  pockets.detect_pockets        (LIGSITE / fpocket / P2Rank)
        v
    binding site
        |  graphs.build_pocket_only_graph
        v
    pocket graph  ---- diffusion.PocketLigandDiffusion.sample ---->  atoms + coords
                                                                        |
                                            evaluate.to_rdkit_mol       |  bond perception
                                                                        v
                                                                    molecules
                                                                        |
                                           affinity.AffinityNet         |  pK prediction
                                                                        v
                                                              evaluate.rank_molecules

Training data comes from two sources (``molgen.sources``): a large unlabelled
BioLiP export for the generator, and PDBBind/PDBBind+ with pK labels for the
ranker.  Everything in between is a graph.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (  # noqa: F401
    affinity,
    chem,
    dataset,
    diffusion,
    egnn,
    evaluate,
    graphs,
    molfiles,
    pdbio,
    pockets,
    sources,
    train_utils,
)

__all__ = [
    "affinity",
    "chem",
    "dataset",
    "diffusion",
    "egnn",
    "evaluate",
    "graphs",
    "molfiles",
    "pdbio",
    "pockets",
    "sources",
    "train_utils",
]
