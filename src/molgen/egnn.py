"""E(n)-equivariant graph neural network and the pocket-conditioned dynamics net.

The layer follows Satorras, Hoogeboom & Welling (2021), *E(n) Equivariant Graph
Neural Networks*::

    m_ij  = phi_e(h_i, h_j, rbf(d_ij), a_ij)
    x_i  <- x_i + sum_j (x_i - x_j) / (d_ij + 1) * phi_x(m_ij)
    h_i  <- h_i + phi_h(h_i, sum_j m_ij)

Why this is the right primitive here: coordinate updates are built only from
*relative* positions scaled by scalar messages, so rotating the input rotates
the output and translating it leaves the output unchanged.  A plain MLP on raw
coordinates would have to learn that symmetry from data, which is exactly the
sample-efficiency problem on a 100 k-complex dataset.

Pocket conditioning follows the "fixed context" scheme of DiffSBDD / TargetDiff:
pocket atoms join the message-passing graph but their coordinates are frozen,
and the whole complex was already centred on the pocket centroid at
preprocessing time.  That handles translation, so no zero-CoM subspace
projection is needed — rotation equivariance about the pocket centre is
provided by the layer itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from .graphs import (
    INTER_EDGE_DIM,
    LIGAND_TYPE_DIM,
    NUM_RBF,
    POCKET_EDGE_DIM,
    POCKET_FEATURE_DIM,
    cross_radius_edges,
    intra_radius_edges,
    rbf_expand,
)

EDGE_TYPE_DIM = 3  # ligand-ligand, pocket-pocket, ligand-pocket


def mlp(sizes: list[int], act=nn.SiLU, out_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


class EGCL(nn.Module):
    """One E(n)-equivariant convolution."""

    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        num_rbf: int = NUM_RBF,
        rbf_cutoff: float = 10.0,
        update_coords: bool = True,
        attention: bool = True,
        coord_clamp: float = 10.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_rbf = num_rbf
        self.rbf_cutoff = rbf_cutoff
        self.update_coords = update_coords
        self.attention = attention
        self.coord_clamp = coord_clamp

        self.edge_mlp = mlp([2 * hidden + num_rbf + edge_dim, hidden, hidden], out_act=True)
        self.node_mlp = mlp([2 * hidden, hidden, hidden])
        self.node_norm = nn.LayerNorm(hidden)
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        if update_coords:
            # Small final weights keep the first denoising steps from exploding:
            # with gain 1 the summed displacement over ~30 neighbours and 6
            # layers easily reaches tens of Angstrom.
            last = nn.Linear(hidden, 1, bias=False)
            nn.init.xavier_uniform_(last.weight, gain=0.001)
            self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), last)

    def forward(
        self,
        h: torch.Tensor,            # (N, hidden)
        x: torch.Tensor,            # (N, 3)
        edge_index: torch.Tensor,   # (2, E) target <- source  (row, col)
        edge_attr: torch.Tensor,    # (E, edge_dim)
        coord_mask: torch.Tensor | None = None,   # (N, 1) 1 where x may move
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        if row.numel() == 0:
            return h, x

        diff = x[row] - x[col]
        dist = diff.norm(dim=-1, keepdim=True)
        feats = rbf_expand(dist.squeeze(-1), cutoff=self.rbf_cutoff, num=self.num_rbf)

        m = self.edge_mlp(torch.cat([h[row], h[col], feats, edge_attr], dim=-1))
        if self.attention:
            m = m * self.att_mlp(m)

        agg = scatter(m, row, dim=0, dim_size=h.size(0), reduce="sum")
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.node_norm(h)

        if self.update_coords:
            # normalising by (d + 1) bounds the per-edge contribution, as in EDM
            weight = self.coord_mlp(m)
            delta = scatter(
                diff / (dist + 1.0) * weight, row, dim=0, dim_size=x.size(0), reduce="sum"
            )
            delta = delta.clamp(-self.coord_clamp, self.coord_clamp)
            if coord_mask is not None:
                delta = delta * coord_mask
            x = x + delta
        return h, x


class SinusoidalTime(nn.Module):
    """Standard transformer-style timestep embedding."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.view(-1, 1).float() * freqs.view(1, -1)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PocketConditionedEGNN(nn.Module):
    """Denoiser ``f(x_t, h_t, t | pocket)`` over ligand atoms.

    Returns **absolute updated ligand coordinates** (in Angstrom) and a ligand
    type vector.  :class:`~molgen.diffusion.PocketLigandDiffusion` interprets
    them as either the denoised ``x_0`` or, by subtracting the input, the noise
    ``eps`` — see its ``parameterization`` option.  The pocket enters as frozen
    context nodes whose coordinates never move.
    """

    def __init__(
        self,
        ligand_type_dim: int = LIGAND_TYPE_DIM,
        pocket_feature_dim: int = POCKET_FEATURE_DIM,
        hidden: int = 128,
        num_layers: int = 6,
        time_dim: int = 32,
        edge_embed_dim: int = 32,
        rbf_cutoff: float = 10.0,
        cross_cutoff: float = 8.0,
        cross_max_neighbors: int = 32,
        attention: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.cross_cutoff = cross_cutoff
        self.cross_max_neighbors = cross_max_neighbors
        self.ligand_type_dim = ligand_type_dim

        self.time_embed = SinusoidalTime(time_dim)
        self.lig_embed = mlp([ligand_type_dim + time_dim + 1, hidden, hidden])
        self.poc_embed = mlp([pocket_feature_dim + time_dim + 1, hidden, hidden])

        # Each edge family has its own attribute layout, so project all of them
        # into one shared space and tag the family with a one-hot.
        self.lig_edge_proj = nn.Linear(NUM_RBF, edge_embed_dim)
        self.poc_edge_proj = nn.Linear(POCKET_EDGE_DIM, edge_embed_dim)
        self.cross_edge_proj = nn.Linear(INTER_EDGE_DIM, edge_embed_dim)
        edge_dim = edge_embed_dim + EDGE_TYPE_DIM

        self.layers = nn.ModuleList(
            EGCL(hidden, edge_dim, rbf_cutoff=rbf_cutoff, attention=attention)
            for _ in range(num_layers)
        )
        self.type_head = mlp([hidden, hidden, ligand_type_dim])

    # ------------------------------------------------------------------
    def _build_graph(
        self,
        x_lig: torch.Tensor,
        lig_batch: torch.Tensor,
        x_poc: torch.Tensor,
        poc_batch: torch.Tensor,
        poc_edge_index: torch.Tensor,
        poc_edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the joint edge set for the concatenated ``[ligand; pocket]`` nodes."""
        n_lig = x_lig.size(0)
        device = x_lig.device

        # (1) ligand-ligand: complete graph per molecule
        ll = intra_radius_edges(x_lig, lig_batch)
        ll_dist = (x_lig[ll[0]] - x_lig[ll[1]]).norm(dim=-1) if ll.numel() else torch.zeros(0, device=device)
        ll_attr = self.lig_edge_proj(rbf_expand(ll_dist, cutoff=self.cross_cutoff))

        # (2) pocket-pocket: precomputed static contact graph
        pp = poc_edge_index + n_lig
        pp_attr = self.poc_edge_proj(poc_edge_attr)

        # (3) ligand-pocket: rebuilt from the *current noisy* ligand coordinates
        cross = cross_radius_edges(
            x_lig, lig_batch, x_poc, poc_batch,
            cutoff=self.cross_cutoff, max_neighbors=self.cross_max_neighbors,
        )
        c_dist = (
            (x_lig[cross[0]] - x_poc[cross[1]]).norm(dim=-1)
            if cross.numel() else torch.zeros(0, device=device)
        )
        c_attr = self.cross_edge_proj(rbf_expand(c_dist, cutoff=self.cross_cutoff, num=INTER_EDGE_DIM))
        cross_lp = torch.stack([cross[0], cross[1] + n_lig])
        cross_pl = torch.stack([cross[1] + n_lig, cross[0]])  # messages both ways

        edge_index = torch.cat([ll, pp, cross_lp, cross_pl], dim=1)
        attrs = torch.cat([ll_attr, pp_attr, c_attr, c_attr], dim=0)
        type_id = torch.cat([
            torch.zeros(ll.size(1), dtype=torch.long, device=device),
            torch.ones(pp.size(1), dtype=torch.long, device=device),
            torch.full((2 * cross_lp.size(1),), 2, dtype=torch.long, device=device),
        ])
        type_oh = torch.nn.functional.one_hot(type_id, EDGE_TYPE_DIM).float()
        return edge_index, torch.cat([attrs, type_oh], dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_lig: torch.Tensor,        # (N_l, 3)   noisy ligand coordinates
        h_lig: torch.Tensor,        # (N_l, K)   noisy ligand type features
        lig_batch: torch.Tensor,
        x_poc: torch.Tensor,        # (N_p, 3)
        h_poc: torch.Tensor,        # (N_p, F)
        poc_batch: torch.Tensor,
        poc_edge_index: torch.Tensor,
        poc_edge_attr: torch.Tensor,
        t: torch.Tensor,            # (B,) in [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_out, h_out)``: absolute ligand coordinates and type vector."""
        n_lig = x_lig.size(0)
        t_emb = self.time_embed(t)

        h_l = self.lig_embed(torch.cat([
            h_lig, t_emb[lig_batch], torch.ones(n_lig, 1, device=x_lig.device)
        ], dim=-1))
        h_p = self.poc_embed(torch.cat([
            h_poc, t_emb[poc_batch], torch.zeros(x_poc.size(0), 1, device=x_poc.device)
        ], dim=-1))

        h = torch.cat([h_l, h_p], dim=0)
        x = torch.cat([x_lig, x_poc], dim=0)

        coord_mask = torch.cat([
            torch.ones(n_lig, 1, device=x.device),
            torch.zeros(x_poc.size(0), 1, device=x.device),
        ])

        edge_index, edge_attr = self._build_graph(
            x_lig, lig_batch, x_poc, poc_batch, poc_edge_index, poc_edge_attr
        )
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, coord_mask=coord_mask)

        # Every coordinate update was a sum of relative-position vectors, so
        # the output positions carry the input frame exactly: rotating the
        # complex rotates them, translating it translates them.
        return x[:n_lig], self.type_head(h[:n_lig])


@torch.no_grad()
def check_equivariance(
    model: PocketConditionedEGNN,
    batch,
    device: str = "cpu",
    seed: int = 0,
    atol: float = 1e-3,
) -> dict:
    """Numerically verify the net's E(3) symmetry.

    The net returns *absolute* coordinates, so the correct statements are:
    rotating the complex rotates the output by the same matrix, translating it
    translates the output by the same vector, and the type output does not move
    at all.  (Were the net returning a displacement — the ``eps``
    parameterisation — translation would instead leave the output unchanged;
    that case is the difference of the two positions and follows from these.)

    Run this after any change to the dynamics net.  A broken symmetry is
    silent otherwise: the model still trains, just badly.
    """
    model = model.to(device).eval()
    batch = batch.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)

    n_graphs = int(batch.lig_pos_batch.max().item()) + 1
    t = torch.rand(n_graphs, generator=g).to(device)
    h_lig = torch.nn.functional.one_hot(batch.lig_type, model.ligand_type_dim).float()

    def run(x_lig, x_poc):
        return model(
            x_lig, h_lig, batch.lig_pos_batch, x_poc, batch.poc_x, batch.poc_pos_batch,
            batch.poc_edge_index, batch.poc_edge_attr, t,
        )

    x_out, h_out = run(batch.lig_pos, batch.poc_pos)

    # random rotation via QR of a gaussian matrix
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = (q * torch.sign(torch.diagonal(r)).unsqueeze(0)).to(device)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]

    x_rot, h_rot = run(batch.lig_pos @ q.T, batch.poc_pos @ q.T)
    rot_err = (x_rot - x_out @ q.T).abs().max().item()
    type_err = (h_rot - h_out).abs().max().item()

    shift = torch.randn(1, 3, generator=g).to(device) * 5.0
    x_tr, h_tr = run(batch.lig_pos + shift, batch.poc_pos + shift)
    trans_err = (x_tr - (x_out + shift)).abs().max().item()
    # the displacement (eps parameterisation) is translation-INvariant
    disp_err = ((x_tr - (batch.lig_pos + shift)) - (x_out - batch.lig_pos)).abs().max().item()

    worst = max(rot_err, trans_err, type_err, disp_err)
    return {
        "rotation_equivariance_error": rot_err,
        "translation_equivariance_error": trans_err,
        "displacement_translation_invariance_error": disp_err,
        "type_invariance_error": type_err,
        "passed": bool(worst < atol),
        "worst_error": worst,
    }
