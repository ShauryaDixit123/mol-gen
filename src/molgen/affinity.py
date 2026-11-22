"""Binding-affinity regression over the protein-ligand interaction graph.

Trained on PDBBind/PDBBind+ ``-log10(Kd/Ki)`` labels, this is the model that
*ranks* generated molecules.  It is a separate network from the generator on
purpose: the generator is trained on a large unlabelled set (BioLiP) and only
has to produce plausible chemistry in a cavity, while ranking needs the small
labelled set and a different inductive bias.

Architecture: the same :class:`~molgen.egnn.EGCL` layer with coordinate updates
switched **off**.  Affinity is a scalar, so the right symmetry is *invariance*,
not equivariance — the layer already achieves that by consuming only
interatomic distances.

Four edge families carry distinct physics and get distinct projections:
covalent ligand bonds, through-space ligand contacts, pocket contacts, and
ligand-pocket contacts (the interaction term the label actually depends on).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from .egnn import EGCL, mlp
from .graphs import (
    INTER_EDGE_DIM,
    LIGAND_BOND_EDGE_DIM,
    LIGAND_EXTRA_DIM,
    LIGAND_TYPE_DIM,
    NUM_RBF,
    POCKET_EDGE_DIM,
    POCKET_FEATURE_DIM,
    intra_radius_edges,
    rbf_expand,
)

EDGE_FAMILIES = 4  # bond, ligand-contact, pocket-contact, cross


@dataclass
class AffinityConfig:
    hidden: int = 128
    num_layers: int = 4
    edge_embed_dim: int = 32
    ligand_contact_cutoff: float = 5.0
    rbf_cutoff: float = 8.0
    dropout: float = 0.1
    ligand_type_dim: int = LIGAND_TYPE_DIM
    ligand_extra_dim: int = LIGAND_EXTRA_DIM
    pocket_feature_dim: int = POCKET_FEATURE_DIM
    # Labels are pK in roughly [0, 16]; centring them speeds up convergence a lot.
    label_mean: float = 6.5
    label_std: float = 2.0


class AffinityNet(nn.Module):
    def __init__(self, cfg: AffinityConfig | None = None):
        super().__init__()
        self.cfg = cfg or AffinityConfig()
        c = self.cfg
        h = c.hidden

        self.lig_embed = mlp([c.ligand_type_dim + c.ligand_extra_dim, h, h])
        self.poc_embed = mlp([c.pocket_feature_dim, h, h])

        self.bond_proj = nn.Linear(LIGAND_BOND_EDGE_DIM, c.edge_embed_dim)
        self.lig_contact_proj = nn.Linear(NUM_RBF, c.edge_embed_dim)
        self.poc_proj = nn.Linear(POCKET_EDGE_DIM, c.edge_embed_dim)
        self.cross_proj = nn.Linear(INTER_EDGE_DIM, c.edge_embed_dim)
        edge_dim = c.edge_embed_dim + EDGE_FAMILIES

        self.layers = nn.ModuleList(
            EGCL(h, edge_dim, rbf_cutoff=c.rbf_cutoff, update_coords=False, attention=True)
            for _ in range(c.num_layers)
        )
        # ligand mean+max, interface-pocket mean, pocket mean, 4 scalar descriptors
        self.head = nn.Sequential(
            nn.Linear(4 * h + 4, h), nn.SiLU(), nn.Dropout(c.dropout),
            nn.Linear(h, h // 2), nn.SiLU(), nn.Dropout(c.dropout),
            nn.Linear(h // 2, 1),
        )

    # ------------------------------------------------------------------
    def _build_graph(self, batch):
        n_lig = batch.lig_pos.size(0)
        device = batch.lig_pos.device
        c = self.cfg

        bond = batch.lig_bond_index
        bond_attr = self.bond_proj(batch.lig_bond_attr)

        lc = intra_radius_edges(
            batch.lig_pos, batch.lig_pos_batch,
            cutoff=c.ligand_contact_cutoff, max_nodes_fully_connected=0,
        )
        lc_d = (batch.lig_pos[lc[0]] - batch.lig_pos[lc[1]]).norm(dim=-1) if lc.numel() else torch.zeros(0, device=device)
        lc_attr = self.lig_contact_proj(rbf_expand(lc_d, cutoff=c.ligand_contact_cutoff))

        pp = batch.poc_edge_index + n_lig
        pp_attr = self.poc_proj(batch.poc_edge_attr)

        cross = batch.inter_edge_index
        cross_attr = self.cross_proj(batch.inter_edge_attr)
        cross_lp = torch.stack([cross[0], cross[1] + n_lig]) if cross.numel() else cross.new_zeros((2, 0))
        cross_pl = torch.stack([cross[1] + n_lig, cross[0]]) if cross.numel() else cross.new_zeros((2, 0))

        edge_index = torch.cat([bond, lc, pp, cross_lp, cross_pl], dim=1)
        attrs = torch.cat([bond_attr, lc_attr, pp_attr, cross_attr, cross_attr], dim=0)
        family = torch.cat([
            torch.zeros(bond.size(1), dtype=torch.long, device=device),
            torch.full((lc.size(1),), 1, dtype=torch.long, device=device),
            torch.full((pp.size(1),), 2, dtype=torch.long, device=device),
            torch.full((2 * cross_lp.size(1),), 3, dtype=torch.long, device=device),
        ])
        attrs = torch.cat([attrs, F.one_hot(family, EDGE_FAMILIES).float()], dim=-1)
        return edge_index, attrs

    # ------------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        """Predicted pK, shape ``(num_graphs,)``."""
        n_lig = batch.lig_pos.size(0)
        num_graphs = int(batch.lig_pos_batch.max().item()) + 1 if n_lig else int(batch.num_graphs)

        h_l = self.lig_embed(torch.cat([
            F.one_hot(batch.lig_type, self.cfg.ligand_type_dim).float(), batch.lig_extra
        ], dim=-1))
        h_p = self.poc_embed(batch.poc_x)
        h = torch.cat([h_l, h_p], dim=0)
        x = torch.cat([batch.lig_pos, batch.poc_pos], dim=0)

        edge_index, edge_attr = self._build_graph(batch)
        for layer in self.layers:
            h, _ = layer(h, x, edge_index, edge_attr, coord_mask=None)

        h_lig, h_poc = h[:n_lig], h[n_lig:]
        lb, pb = batch.lig_pos_batch, batch.poc_pos_batch

        lig_mean = scatter(h_lig, lb, dim=0, dim_size=num_graphs, reduce="mean")
        lig_max = scatter(h_lig, lb, dim=0, dim_size=num_graphs, reduce="max")
        poc_mean = scatter(h_poc, pb, dim=0, dim_size=num_graphs, reduce="mean")

        # Interface pocket atoms only: those with at least one cross edge.
        cross = batch.inter_edge_index
        if cross.numel():
            weight = scatter(
                torch.ones(cross.size(1), device=h.device), cross[1],
                dim=0, dim_size=h_poc.size(0), reduce="sum",
            ).unsqueeze(-1)
        else:
            weight = torch.zeros(h_poc.size(0), 1, device=h.device)
        num = scatter(h_poc * weight, pb, dim=0, dim_size=num_graphs, reduce="sum")
        den = scatter(weight, pb, dim=0, dim_size=num_graphs, reduce="sum").clamp(min=1e-6)
        iface_mean = num / den

        # Scalar descriptors: size and contact density are strong, cheap priors
        # (affinity grows with buried contact surface).
        n_l = scatter(torch.ones(n_lig, device=h.device), lb, dim=0, dim_size=num_graphs, reduce="sum")
        n_c = (
            scatter(torch.ones(cross.size(1), device=h.device), lb[cross[0]], dim=0,
                    dim_size=num_graphs, reduce="sum")
            if cross.numel() else torch.zeros(num_graphs, device=h.device)
        )
        desc = torch.stack([
            n_l / 30.0,
            n_c / 200.0,
            (n_c / n_l.clamp(min=1)) / 10.0,
            scatter(batch.lig_pos.norm(dim=-1), lb, dim=0, dim_size=num_graphs, reduce="mean") / 10.0,
        ], dim=-1)

        feats = torch.cat([lig_mean, lig_max, poc_mean, iface_mean, desc], dim=-1)
        z = self.head(feats).squeeze(-1)
        return z * self.cfg.label_std + self.cfg.label_mean

    # ------------------------------------------------------------------
    def loss(self, batch, huber_delta: float = 1.0) -> dict[str, torch.Tensor]:
        pred = self(batch)
        target = batch.y.view(-1)
        mask = torch.isfinite(target)
        if mask.sum() == 0:
            zero = pred.sum() * 0.0
            return {"loss": zero, "mse": zero.detach(), "n": 0}
        p, t = pred[mask], target[mask]
        # Huber: PDBBind labels include a long tail of weak binders and a few
        # mis-annotated outliers that pure MSE chases.
        loss = F.huber_loss(p, t, delta=huber_delta)
        return {
            "loss": loss,
            "mse": F.mse_loss(p, t).detach(),
            "mae": (p - t).abs().mean().detach(),
            "n": int(mask.sum()),
        }

    def config_dict(self) -> dict:
        return asdict(self.cfg)


@torch.no_grad()
def predict_batched(model: AffinityNet, data_list, device: str = "cpu", batch_size: int = 32) -> torch.Tensor:
    """Score a list of complexes (e.g. generated molecules) in batches."""
    from torch_geometric.data import Batch

    from .graphs import FOLLOW_BATCH

    model = model.to(device).eval()
    out = []
    for i in range(0, len(data_list), batch_size):
        chunk = list(data_list[i : i + batch_size])
        batch = Batch.from_data_list(chunk, follow_batch=FOLLOW_BATCH).to(device)
        out.append(model(batch).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def regression_metrics(pred, target) -> dict:
    """RMSE / MAE / Pearson / Spearman — the standard PDBBind report."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    if len(p) < 2:
        return {"n": int(len(p))}
    return {
        "n": int(len(p)),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
        "mae": float(np.abs(p - t).mean()),
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
        "sd": float(np.std(p - t)),
    }
