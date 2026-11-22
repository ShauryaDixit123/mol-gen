"""Pocket-conditioned equivariant diffusion over ligand atoms.

Joint diffusion of continuous coordinates and (continuously relaxed) atom
types, denoised by :class:`~molgen.egnn.PocketConditionedEGNN`.

Scaling — the one detail that quietly breaks this family of models
-----------------------------------------------------------------
A DDPM's terminal prior is ``N(0, I)``: unit variance.  Ligand coordinates
measured in Angstrom relative to the pocket centroid have a standard deviation
of roughly 3-4 A, so sampling ``z_T ~ N(0, I)`` in raw Angstrom would start
every molecule collapsed into a 1 A ball at the pocket centre and the reverse
process would have to transport mass it was never trained to transport.

So the diffusion variable is ``z = x / coord_scale`` (unit-ish), while the
*network* is still fed ``z_t * coord_scale`` in Angstrom — the pocket stays in
Angstrom and every radius cutoff keeps its physical meaning.  Multiplying by a
scalar is equivariant, so nothing about the symmetry argument changes.
Atom types are scaled the EDM way (one-hot / ``type_scale``), which keeps the
categorical signal alive deeper into the noise schedule.

Parameterisation — why ``x0`` and not ``eps``
---------------------------------------------
With noise prediction the reverse mean is
``mu = z_t / alpha_{t|s} - (sigma^2_{t|s} / (alpha_{t|s} sigma_t)) * eps_hat``:
``z`` is **divided** by ``alpha_{t|s} < 1`` at every step, which over the whole
trajectory amplifies it by ``alpha_0 / alpha_T ~ 316x``, and only a perfect
``eps_hat`` cancels that.  Here the error does not merely degrade samples, it
runs away: once a ligand atom drifts past the 8 A cross-edge cutoff the pocket
context disappears from its neighbourhood, the network predicts almost nothing,
and the remaining steps are pure amplification.  Measured on a model overfit to
8 complexes, sampled ligands landed **331 A** from the pocket (untrained: 472 A)
even though the training loss had fallen by 78 %.

Predicting ``x_0`` instead gives
``mu = c_t z_t + c_0 x0_hat`` with ``c_t, c_0 >= 0`` and ``c_t + c_0 ~ 1``:
bounded coefficients, no amplification, and a useless prediction collapses
samples toward the pocket centroid — where the context still reaches them —
rather than expelling them.  This is what TargetDiff does for the same task.
``parameterization="eps"`` remains available for comparison.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from . import chem
from .egnn import PocketConditionedEGNN
from .graphs import LIGAND_TYPE_DIM, POCKET_FEATURE_DIM


# --------------------------------------------------------------------------
# noise schedules
# --------------------------------------------------------------------------
def polynomial_schedule(timesteps: int, s: float = 1e-5, power: float = 2.0) -> np.ndarray:
    """EDM's polynomial schedule: ``alpha_bar = (1 - (t/T)^power)^2``.

    Decays more slowly than cosine at small ``t``, which empirically matters for
    molecular geometry: the fine positional detail is set in the last few steps.
    """
    t = np.linspace(0, timesteps, timesteps + 1)
    alphas2 = (1 - (t / timesteps) ** power) ** 2
    alphas2 = np.clip(alphas2, 0.0, 1.0)
    # precision adjustment so alpha_bar never hits exactly 0 or 1
    alphas2 = (1 - 2 * s) * alphas2 + s
    return alphas2


def cosine_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    t = np.linspace(0, timesteps, timesteps + 1)
    f = np.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas2 = f / f[0]
    return np.clip(alphas2, 1e-5, 1.0)


def linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas2 = np.cumprod(1.0 - betas)
    return np.concatenate([[1.0], alphas2])


SCHEDULES = {
    "polynomial": polynomial_schedule,
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


@dataclass
class DiffusionConfig:
    timesteps: int = 500
    schedule: str = "polynomial"
    coord_scale: float = 4.0
    type_scale: float = 4.0
    type_loss_weight: float = 1.0
    # Auxiliary cross-entropy on the predicted types. The MSE alone treats a
    # one-hot as a regression target; CE sharpens element identity noticeably.
    type_ce_weight: float = 0.3
    parameterization: str = "x0"      # "x0" (stable, default) or "eps"
    # Clip the predicted x0 to this many normalised units from the pocket
    # centroid. Ligand atoms sit within ~2-3 units, so 4.0 only ever truncates
    # a diverging prediction -- the principled version of a sampling rail.
    clip_x0: float | None = 4.0
    hidden: int = 128
    num_layers: int = 6
    cross_cutoff: float = 8.0
    cross_max_neighbors: int = 32
    attention: bool = True
    ligand_type_dim: int = LIGAND_TYPE_DIM
    pocket_feature_dim: int = POCKET_FEATURE_DIM


class PocketLigandDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig | None = None):
        super().__init__()
        self.cfg = cfg or DiffusionConfig()
        c = self.cfg

        alphas2 = SCHEDULES[c.schedule](c.timesteps)
        self.register_buffer("alpha_bar", torch.tensor(alphas2, dtype=torch.float64))
        self.register_buffer("alpha", torch.sqrt(torch.tensor(alphas2, dtype=torch.float64)))
        self.register_buffer("sigma", torch.sqrt(1.0 - torch.tensor(alphas2, dtype=torch.float64)))

        self.net = PocketConditionedEGNN(
            ligand_type_dim=c.ligand_type_dim,
            pocket_feature_dim=c.pocket_feature_dim,
            hidden=c.hidden,
            num_layers=c.num_layers,
            cross_cutoff=c.cross_cutoff,
            cross_max_neighbors=c.cross_max_neighbors,
            attention=c.attention,
        )

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------
    def normalize(self, pos: torch.Tensor, types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_x = pos / self.cfg.coord_scale
        z_h = F.one_hot(types, self.cfg.ligand_type_dim).float() / self.cfg.type_scale
        return z_x, z_h

    def denormalize_pos(self, z_x: torch.Tensor) -> torch.Tensor:
        return z_x * self.cfg.coord_scale

    # ------------------------------------------------------------------
    def _gather(self, buf: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-node view of a per-graph schedule coefficient."""
        return buf[t][batch].to(torch.float32).unsqueeze(-1)

    def predict_x0(
        self,
        z_x: torch.Tensor,
        z_h: torch.Tensor,
        lig_batch: torch.Tensor,
        batch,
        t_int: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the denoised ``(x_0, h_0)`` in normalised units.

        The network is fed Angstrom coordinates and returns Angstrom
        coordinates; the unit conversion lives here, and under
        ``parameterization='eps'`` the network output is first read as a
        displacement and converted through
        ``x_0 = (z_t - sigma_t eps) / alpha_t``.
        """
        t_norm = t_int.float() / self.cfg.timesteps
        x_out, h_out = self.net(
            x_lig=self.denormalize_pos(z_x),
            h_lig=z_h,
            lig_batch=lig_batch,
            x_poc=batch.poc_pos,
            h_poc=batch.poc_x,
            poc_batch=batch.poc_pos_batch,
            poc_edge_index=batch.poc_edge_index,
            poc_edge_attr=batch.poc_edge_attr,
            t=t_norm,
        )
        x_out = x_out / self.cfg.coord_scale

        if self.cfg.parameterization == "eps":
            a = self._gather(self.alpha, t_int, lig_batch)
            s = self._gather(self.sigma, t_int, lig_batch)
            eps_x, eps_h = x_out - z_x, h_out
            x0_x = (z_x - s * eps_x) / a.clamp(min=1e-8)
            x0_h = (z_h - s * eps_h) / a.clamp(min=1e-8)
        else:
            x0_x, x0_h = x_out, h_out

        if self.cfg.clip_x0 is not None:
            c = self.cfg.clip_x0
            x0_x = x0_x.clamp(-c, c)
            x0_h = x0_h.clamp(-c, c)
        return x0_x, x0_h

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def forward(self, batch) -> dict[str, torch.Tensor]:
        """Training step on one batch: MSE on the denoised target, plus type CE."""
        lig_batch = batch.lig_pos_batch
        num_graphs = int(lig_batch.max().item()) + 1
        device = batch.lig_pos.device

        z_x0, z_h0 = self.normalize(batch.lig_pos, batch.lig_type)

        # t = 0 carries no signal to learn from, so sample t in [1, T]
        t_int = torch.randint(1, self.cfg.timesteps + 1, (num_graphs,), device=device)
        a = self._gather(self.alpha, t_int, lig_batch)
        s = self._gather(self.sigma, t_int, lig_batch)

        z_xt = a * z_x0 + s * torch.randn_like(z_x0)
        z_ht = a * z_h0 + s * torch.randn_like(z_h0)

        pred_x, pred_h = self.predict_x0(z_xt, z_ht, lig_batch, batch, t_int)

        # Mean over atoms *within* a graph then over graphs, so large ligands
        # do not dominate the gradient.
        counts = torch.bincount(lig_batch, minlength=num_graphs).clamp(min=1).float()

        def graph_mean(per_node: torch.Tensor, dim: int) -> torch.Tensor:
            summed = scatter(per_node, lig_batch, dim=0, dim_size=num_graphs, reduce="sum")
            return (summed / counts).mean() / dim

        loss_x = graph_mean(((pred_x - z_x0) ** 2).sum(-1), 3)
        loss_h = graph_mean(((pred_h - z_h0) ** 2).sum(-1), self.cfg.ligand_type_dim)

        out = {"loss_x": loss_x.detach(), "loss_h": loss_h.detach()}
        loss = loss_x + self.cfg.type_loss_weight * loss_h

        if self.cfg.type_ce_weight > 0 and batch.lig_type.numel():
            logits = pred_h * self.cfg.type_scale
            ce_per_node = F.cross_entropy(logits, batch.lig_type, reduction="none")
            loss_ce = graph_mean(ce_per_node, 1)
            loss = loss + self.cfg.type_ce_weight * loss_ce
            out["loss_ce"] = loss_ce.detach()
            out["type_acc"] = (logits.argmax(-1) == batch.lig_type).float().mean().detach()

        out["loss"] = loss
        return out

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch,
        n_atoms: list[int] | torch.Tensor,
        num_steps: int | None = None,
        eta: float = 1.0,
        return_trajectory: bool = False,
        progress: bool = False,
        clamp_norm: float | None = None,
    ) -> dict:
        """Generate ligands inside the pockets of ``batch``.

        Parameters
        ----------
        n_atoms
            Ligand size per graph — draw it from
            :class:`~molgen.dataset.SizePrior` so molecules match the cavity.
        num_steps
            Fewer than ``cfg.timesteps`` enables strided (DDIM-style) sampling.
        eta
            1.0 = ancestral DDPM sampling, 0.0 = deterministic DDIM.
        clamp_norm
            Extra rail on ``|z|`` in normalised units, normally unnecessary:
            under the default ``x0`` parameterisation the reverse step is an
            interpolation and ``cfg.clip_x0`` already bounds the trajectory.
            It exists for ``parameterization='eps'``, which amplifies ``z`` by
            ``1/alpha_T`` (~300x) over the trajectory and can expel atoms from
            the pocket entirely.  Rescaling the per-atom radius is
            rotation-equivariant, so it does not break the symmetry.
        """
        device = batch.poc_pos.device
        self.eval()

        if isinstance(n_atoms, torch.Tensor):
            n_atoms = n_atoms.tolist()
        num_graphs = len(n_atoms)
        lig_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=device), torch.tensor(n_atoms, device=device)
        )
        total = int(sum(n_atoms))

        z_x = torch.randn(total, 3, device=device)
        z_h = torch.randn(total, self.cfg.ligand_type_dim, device=device)

        T = self.cfg.timesteps
        steps = T if num_steps is None else min(num_steps, T)
        ts = torch.linspace(T, 0, steps + 1, device=device).round().long().clamp(0, T)
        traj: list[tuple[np.ndarray, np.ndarray]] = []

        iterator = range(steps)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="denoise", leave=False)
            except ImportError:
                pass

        last_h = z_h
        for k in iterator:
            t_int = ts[k].expand(num_graphs)
            s_int = ts[k + 1].expand(num_graphs)
            x0_x, x0_h = self.predict_x0(z_x, z_h, lig_batch, batch, t_int)
            last_h = x0_h
            z_x, z_h = self._step(z_x, z_h, x0_x, x0_h, t_int, s_int, lig_batch, eta)
            if clamp_norm is not None:
                radius = z_x.norm(dim=-1, keepdim=True)
                z_x = z_x * (clamp_norm / radius.clamp(min=1e-8)).clamp(max=1.0)
            if return_trajectory and (k % max(1, steps // 20) == 0 or k == steps - 1):
                traj.append((self.denormalize_pos(z_x).cpu().numpy(), z_h.cpu().numpy()))

        pos = self.denormalize_pos(z_x)
        # Read the element off the final *prediction*, not off z_0: the last
        # reverse step still injects noise, which flips borderline types.
        logits = last_h * self.cfg.type_scale
        types = logits.argmax(dim=-1)
        out = {
            "pos": pos,
            "types": types,
            "lig_batch": lig_batch,
            "type_logits": logits,
        }
        if return_trajectory:
            out["trajectory"] = traj
        return out

    def _step(
        self,
        z_x: torch.Tensor,
        z_h: torch.Tensor,
        x0_x: torch.Tensor,
        x0_h: torch.Tensor,
        t_int: torch.Tensor,
        s_int: torch.Tensor,
        lig_batch: torch.Tensor,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One reverse step ``z_t -> z_s`` (``s < t``) from the true posterior.

        With ``alpha_{t|s} = alpha_t / alpha_s`` and
        ``sigma^2_{t|s} = sigma_t^2 - alpha_{t|s}^2 sigma_s^2``,

        ``q(z_s | z_t, x_0) = N( c_t z_t + c_0 x_0,  (sigma_{t|s} sigma_s / sigma_t)^2 )``

        with ``c_t = alpha_{t|s} sigma_s^2 / sigma_t^2`` and
        ``c_0 = alpha_s sigma^2_{t|s} / sigma_t^2``.  Both are non-negative and
        sum to about 1, so the update is an interpolation and nothing is
        amplified — unlike the algebraically equivalent noise form, which
        divides by ``alpha_{t|s}``.  ``eta=0`` drops the injected noise and
        gives the deterministic DDIM update.
        """
        a_t = self._gather(self.alpha, t_int, lig_batch)
        a_s = self._gather(self.alpha, s_int, lig_batch)
        s_t = self._gather(self.sigma, t_int, lig_batch)
        s_s = self._gather(self.sigma, s_int, lig_batch)

        a_ts = a_t / a_s.clamp(min=1e-8)
        var_ts = (s_t ** 2 - (a_ts ** 2) * s_s ** 2).clamp(min=0.0)
        inv_var_t = 1.0 / (s_t ** 2).clamp(min=1e-12)

        c_t = a_ts * (s_s ** 2) * inv_var_t
        c_0 = a_s * var_ts * inv_var_t

        mu_x = c_t * z_x + c_0 * x0_x
        mu_h = c_t * z_h + c_0 * x0_h
        if eta <= 0:
            return mu_x, mu_h
        std = eta * torch.sqrt(var_ts) * s_s / s_t.clamp(min=1e-8)
        return mu_x + std * torch.randn_like(mu_x), mu_h + std * torch.randn_like(mu_h)

    # ------------------------------------------------------------------
    def to_molecules(self, sample: dict) -> list[dict]:
        """Split a batched sample into per-molecule element/coordinate records."""
        pos = sample["pos"].detach().cpu().numpy()
        types = sample["types"].detach().cpu().numpy()
        lig_batch = sample["lig_batch"].detach().cpu().numpy()
        out = []
        for g in range(int(lig_batch.max()) + 1 if len(lig_batch) else 0):
            m = lig_batch == g
            out.append({
                "elements": [chem.LIGAND_ELEMENTS[t] for t in types[m]],
                "pos": pos[m],
                "graph_index": g,
            })
        return out

    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return asdict(self.cfg)


def estimate_coord_scale(dataset, max_samples: int = 2000) -> float:
    """Std of ligand coordinates about the pocket centroid — set ``coord_scale`` to this.

    Getting this wrong by 2x is the difference between a model that generates
    molecules filling the pocket and one that generates a tight blob at the
    centre (too small) or sprays atoms into solvent (too large).
    """
    getter = getattr(dataset, "get_sample", None)
    vals = []
    for i in range(min(len(dataset), max_samples)):
        d = getter(i) if getter is not None else dataset[i]
        pos = d["lig_pos"] if isinstance(d, dict) else d.lig_pos.numpy()
        if len(pos):
            vals.append(np.asarray(pos).reshape(-1))
    return float(np.std(np.concatenate(vals))) if vals else 4.0


def strip_ligand(data):
    """Return a copy of ``data`` with every ligand tensor emptied.

    :meth:`PocketLigandDiffusion.sample` never reads the ligand fields, but
    stripping them makes that guarantee structural rather than a promise —
    leaving the crystal ligand in the batch while "generating" is the easiest
    way to accidentally report leaked results.
    """
    out = data.clone()
    out.lig_pos = torch.zeros((0, 3), dtype=torch.float)
    out.lig_type = torch.zeros((0,), dtype=torch.long)
    out.lig_extra = torch.zeros((0, data.lig_extra.size(1)), dtype=torch.float)
    out.lig_bond_index = torch.zeros((2, 0), dtype=torch.long)
    out.lig_bond_attr = torch.zeros((0, data.lig_bond_attr.size(1)), dtype=torch.float)
    out.inter_edge_index = torch.zeros((2, 0), dtype=torch.long)
    out.inter_edge_attr = torch.zeros((0, data.inter_edge_attr.size(1)), dtype=torch.float)
    out.n_lig = torch.tensor([0], dtype=torch.long)
    return out


def build_pocket_batch(data_list, device: str = "cpu", keep_ligand: bool = False):
    """Collate complexes for sampling, dropping the ground-truth ligand."""
    from torch_geometric.data import Batch

    from .graphs import FOLLOW_BATCH

    items = [d if keep_ligand else strip_ligand(d) for d in data_list]
    batch = Batch.from_data_list(items, follow_batch=FOLLOW_BATCH)
    return batch.to(device)
