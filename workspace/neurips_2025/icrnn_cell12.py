from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score


class ICRNNCell(torch.nn.Module):
    """Stabilized convex recurrent update with a deeper convex residual."""

    def __init__(self, latent_dim: int, stim_dim: int, hidden_dim: int = 128, rho: float = 0.7):
        super().__init__()
        self.latent_dim = latent_dim
        self.stim_dim = stim_dim
        self.hidden_dim = int(hidden_dim)
        self.rho = float(rho)

        self.raw_Wz1 = torch.nn.Parameter(0.01 * torch.randn(self.hidden_dim, latent_dim))
        self.raw_Wv1 = torch.nn.Parameter(0.01 * torch.randn(self.hidden_dim, stim_dim))
        self.raw_Wh = torch.nn.Parameter(0.01 * torch.randn(self.hidden_dim, self.hidden_dim))
        self.raw_Wout = torch.nn.Parameter(0.01 * torch.randn(latent_dim, self.hidden_dim))
        self.bias_conv1 = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        self.bias_conv2 = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        self.bias_conv3 = torch.nn.Parameter(torch.zeros(latent_dim))

        self.raw_Az = torch.nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.raw_Av = torch.nn.Parameter(0.01 * torch.randn(latent_dim, stim_dim))
        self.bias_aff = torch.nn.Parameter(torch.zeros(latent_dim))

        self.alpha_raw = torch.nn.Parameter(torch.full((latent_dim,), 1.2))
        self.eta_raw = torch.nn.Parameter(torch.full((latent_dim,), -2.0))

    def _constrained_Wz1(self) -> torch.Tensor:
        Wz_pos = F.softplus(self.raw_Wz1)
        row_sum = Wz_pos.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (Wz_pos / row_sum) * self.rho

    def _constrained_Wv1(self) -> torch.Tensor:
        return F.softplus(self.raw_Wv1)

    def _constrained_Wh(self) -> torch.Tensor:
        Wh_pos = F.softplus(self.raw_Wh)
        row_sum = Wh_pos.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (Wh_pos / row_sum) * self.rho

    def _constrained_Wout(self) -> torch.Tensor:
        Wout_pos = F.softplus(self.raw_Wout)
        row_sum = Wout_pos.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (Wout_pos / row_sum) * self.rho

    def _stable_Az(self) -> torch.Tensor:
        I = torch.eye(self.latent_dim, device=self.raw_Az.device, dtype=self.raw_Az.dtype)
        Az = I + 0.05 * torch.tanh(self.raw_Az)
        op_norm = torch.linalg.matrix_norm(Az, ord=2)
        max_norm = 0.98
        if op_norm > max_norm:
            Az = Az * (max_norm / op_norm)
        return Az

    def _stable_Av(self) -> torch.Tensor:
        return 0.05 * self.raw_Av

    def forward(self, z_prev: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        Wz1 = self._constrained_Wz1()
        Wv1 = self._constrained_Wv1()
        Wh = self._constrained_Wh()
        Wout = self._constrained_Wout()
        Az = self._stable_Az()
        Av = self._stable_Av()

        alpha = 0.92 + 0.07 * torch.sigmoid(self.alpha_raw)
        eta = 0.20 * torch.sigmoid(self.eta_raw)

        pre_conv1 = z_prev @ Wz1.T + v_t @ Wv1.T + self.bias_conv1
        h1 = F.softplus(pre_conv1)
        pre_conv2 = h1 @ Wh.T + self.bias_conv2
        h2 = F.softplus(pre_conv2)
        pre_conv3 = h2 @ Wout.T + self.bias_conv3
        convex_term = F.softplus(pre_conv3)

        affine_term = z_prev @ Az.T + v_t @ Av.T + self.bias_aff
        z_candidate = affine_term + eta[None, :] * convex_term
        z_next = alpha[None, :] * z_prev + (1.0 - alpha[None, :]) * z_candidate
        return z_next


class ICRNNStimForecast(torch.nn.Module):
    def __init__(self, latent_dim: int, stim_dim: int, hidden_dim: int = 128, rho: float = 0.7):
        super().__init__()
        self.cell = ICRNNCell(latent_dim=latent_dim, stim_dim=stim_dim, hidden_dim=hidden_dim, rho=rho)

    def forward(
        self,
        context_z: torch.Tensor,
        context_v: torch.Tensor,
        future_v: torch.Tensor,
        target_z: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
    ) -> torch.Tensor:
        z_prev = context_z[:, -1, :]
        H = future_v.shape[1]
        preds = []

        for h in range(H):
            if h > 0 and target_z is not None and teacher_forcing > 0.0:
                z_prev = teacher_forcing * target_z[:, h - 1, :] + (1.0 - teacher_forcing) * z_prev
            z_prev = self.cell(z_prev, future_v[:, h, :])
            preds.append(z_prev)

        return torch.stack(preds, dim=1)


class ParameterEMA:
    """EMA shadow model for stabler evaluation in stiff dynamics."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k in ssd.keys():
            ssd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


def temporal_smoothness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    return F.mse_loss(dp, dt)


def state_bound_loss(pred: torch.Tensor, bound: float = 4.0) -> torch.Tensor:
    overflow = F.relu(torch.abs(pred) - bound)
    return (overflow ** 2).mean()


@dataclass
class ICRNNCell12Result:
    model_icrnn: ICRNNStimForecast
    optimizer_icrnn: torch.optim.Optimizer
    ema_icrnn: ParameterEMA
    n_epochs_icrnn: int
    loss_history_icrnn: list[float]
    test_loss_history_icrnn: list[float]
    training_time_icrnn: float
    inference_time_icrnn: float
    pred_z_t_icrnn: np.ndarray
    pred_z_icrnn: np.ndarray
    target_z_icrnn: np.ndarray
    weighted_mse_by_h_icrnn: list[float]
    weighted_r2_by_h_icrnn: list[float]
    per_trial_wmse_icrnn: np.ndarray
    best_test_idx_icrnn: int
    trial_id: int
    pred_icrnn_single: np.ndarray
    pred_win_icrnn: np.ndarray
    true_from_onset: np.ndarray
    icrnn_from_onset: np.ndarray


def _require(ns: dict[str, Any], names: list[str]) -> None:
    missing = [name for name in names if name not in ns]
    if missing:
        raise RuntimeError("Missing required variables for ICRNN cell: " + ", ".join(missing))


def _default_apply_ema(z: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if z.ndim == 2:
        out = np.empty_like(z)
        out[0] = z[0]
        for t in range(1, z.shape[0]):
            out[t] = alpha * z[t] + (1 - alpha) * out[t - 1]
        return out
    if z.ndim == 3:
        out = np.empty_like(z)
        out[:, 0, :] = z[:, 0, :]
        for t in range(1, z.shape[1]):
            out[:, t, :] = alpha * z[:, t, :] + (1 - alpha) * out[:, t - 1, :]
        return out
    raise ValueError(f"Expected 2D or 3D array, got {z.ndim}D")


class OnsetForecastDataset(Dataset):
    """Clean dataset for test evaluation."""

    def __init__(
        self,
        z_trial_n: np.ndarray,
        v: np.ndarray,
        trial_indices: np.ndarray,
        context_start: int,
        context_end: int,
        forecast_start: int,
        forecast_end: int,
    ):
        self.z_trial_n = z_trial_n
        self.v = v
        self.trial_indices = np.array(trial_indices)
        self.context_start = context_start
        self.context_end = context_end
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end

    def __len__(self) -> int:
        return len(self.trial_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.trial_indices[idx]
        context_z = self.z_trial_n[tr, self.context_start : self.context_end, :]
        context_v = self.v[tr, self.context_start : self.context_end, :]
        future_v = self.v[tr, self.forecast_start : self.forecast_end, :]
        target_z = self.z_trial_n[tr, self.forecast_start : self.forecast_end, :]
        return (
            torch.from_numpy(context_z),
            torch.from_numpy(context_v),
            torch.from_numpy(future_v),
            torch.from_numpy(target_z),
        )


class AugmentedOnsetForecastDataset(Dataset):
    """Training dataset: augments only stimulated trials with peri-onset noise."""

    def __init__(
        self,
        z_trial_n: np.ndarray,
        v: np.ndarray,
        trial_indices: np.ndarray,
        stim_mask: np.ndarray,
        n_augment: int,
        noise_std: np.ndarray,
        aug_start: int,
        aug_end: int,
        context_start: int,
        context_end: int,
        forecast_start: int,
        forecast_end: int,
    ):
        self.z_trial_n = z_trial_n
        self.v = v
        self.noise_std = noise_std
        self.aug_start = aug_start
        self.aug_end = aug_end
        self.context_start = context_start
        self.context_end = context_end
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end

        self.items: list[tuple[int, int]] = []
        for i, tr in enumerate(trial_indices):
            self.items.append((int(tr), 0))
            if bool(stim_mask[i]):
                for k in range(1, n_augment + 1):
                    self.items.append((int(tr), k))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tr, copy_idx = self.items[idx]

        context_z = self.z_trial_n[tr, self.context_start : self.context_end, :].copy()
        context_v = self.v[tr, self.context_start : self.context_end, :]
        future_v = self.v[tr, self.forecast_start : self.forecast_end, :]
        target_z = self.z_trial_n[tr, self.forecast_start : self.forecast_end, :].copy()

        if copy_idx > 0:
            ctx_noise_start = max(self.aug_start, self.context_start) - self.context_start
            ctx_noise_end = min(self.aug_end, self.context_end) - self.context_start
            if ctx_noise_end > ctx_noise_start:
                n_ctx = ctx_noise_end - ctx_noise_start
                context_z[ctx_noise_start:ctx_noise_end, :] += (
                    np.random.randn(n_ctx, len(self.noise_std)).astype(np.float32) * self.noise_std
                )

            tgt_global_start = self.forecast_start
            tgt_global_end = self.forecast_end
            tgt_noise_start = max(self.aug_start, tgt_global_start) - tgt_global_start
            tgt_noise_end = min(self.aug_end, tgt_global_end) - tgt_global_start
            if tgt_noise_end > tgt_noise_start:
                n_tgt = tgt_noise_end - tgt_noise_start
                target_z[tgt_noise_start:tgt_noise_end, :] += (
                    np.random.randn(n_tgt, len(self.noise_std)).astype(np.float32) * self.noise_std
                )

        return (
            torch.from_numpy(context_z),
            torch.from_numpy(context_v),
            torch.from_numpy(future_v),
            torch.from_numpy(target_z),
        )


def _ensure_cell12_prerequisites(ns: dict[str, Any], verbose: bool = True) -> None:
    required = {
        "aug_train_loader",
        "aug_train_ds",
        "weighted_mse_loss",
        "component_weights_t",
        "component_weights",
        "horizon",
        "horizons",
        "d_z",
        "n_cells",
        "device",
        "z_std",
        "z_mean",
        "z_trial_n",
        "v",
        "context_len",
        "z_trial",
    }
    if required.issubset(ns.keys()):
        return

    base_needed = ["s0", "dpca", "train_indices", "test_indices", "time_axis", "zero_idx"]
    _require(ns, base_needed)

    s0 = ns["s0"]
    neural_data = ns.get("neural_data", s0["behaviour_trials"])
    train_indices = np.array(ns["train_indices"])
    test_indices = np.array(ns["test_indices"])
    zero_idx = int(ns["zero_idx"])

    n_cells = int(s0["n_cells"])
    time_total = int(s0["n_times"])

    pre_event_steps = int(ns.get("pre_event_steps", 50))
    post_event_steps = int(ns.get("post_event_steps", 50))
    context_start = zero_idx - pre_event_steps
    context_end = zero_idx + 1
    forecast_start = zero_idx + 1
    forecast_end = forecast_start + post_event_steps

    if context_start < 0:
        raise RuntimeError("Not enough pre-onset steps for requested context window.")
    if forecast_end > time_total:
        raise RuntimeError("Not enough post-onset steps for requested forecast window.")

    context_len = context_end - context_start
    horizon = forecast_end - forecast_start

    dpca = ns["dpca"]
    if "R" in ns:
        R = ns["R"]
        stim_mean = np.mean(R.reshape((n_cells, -1)), axis=1)[:, None, None]
    else:
        stim_mean = np.mean(neural_data.reshape((n_cells, -1)), axis=1)[:, None, None]

    neural_centered = neural_data - stim_mean
    z_all = dpca.transform(neural_centered)
    z_trial = np.concatenate([z_all["s"], z_all["t"], z_all["st"]], axis=0)
    z_trial = np.transpose(z_trial, (1, 2, 0)).astype(np.float32)

    ema_alpha = float(ns.get("EMA_ALPHA", 0.2))
    apply_ema = ns.get("apply_ema", _default_apply_ema)
    z_trial = apply_ema(z_trial, alpha=ema_alpha)

    v = np.transpose(s0["is_target"], (1, 2, 0)).astype(np.float32)
    d_z = int(z_trial.shape[-1])

    z_train = z_trial[train_indices]
    z_mean = z_train.mean(axis=(0, 1), keepdims=True)
    z_std = z_train.std(axis=(0, 1), keepdims=True) + 1e-6
    z_trial_n = (z_trial - z_mean) / z_std

    component_var = z_train.var(axis=(0, 1))
    component_weights = component_var / (component_var.sum() + 1e-12)

    device = ns.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    component_weights_t = torch.from_numpy(component_weights).float().to(device)

    def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        per_comp_mse = ((pred - target) ** 2).mean(dim=(0, 1))
        return (weights * per_comp_mse).sum()

    n_augment = int(ns.get("n_augment", 20))
    noise_frac = float(ns.get("noise_frac", 0.20))
    aug_half_win = int(ns.get("aug_half_win", 5))
    z_train_n = z_trial_n[train_indices]
    z_comp_std = z_train_n.std(axis=(0, 1))
    noise_std = (noise_frac * z_comp_std).astype(np.float32)
    aug_start = max(zero_idx - aug_half_win, 0)
    aug_end = min(zero_idx + aug_half_win, time_total)

    is_stimulated = v.reshape(v.shape[0], -1).any(axis=1)
    train_stim_mask = is_stimulated[train_indices]

    aug_train_ds = AugmentedOnsetForecastDataset(
        z_trial_n=z_trial_n,
        v=v,
        trial_indices=train_indices,
        stim_mask=train_stim_mask,
        n_augment=n_augment,
        noise_std=noise_std,
        aug_start=aug_start,
        aug_end=aug_end,
        context_start=context_start,
        context_end=context_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )
    batch_size = int(ns.get("batch_size", 32))
    aug_train_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True)

    test_ds = OnsetForecastDataset(
        z_trial_n=z_trial_n,
        v=v,
        trial_indices=test_indices,
        context_start=context_start,
        context_end=context_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    horizons = np.arange(1, horizon + 1)

    ns.update(
        {
            "neural_data": neural_data,
            "z_all": z_all,
            "z_trial": z_trial,
            "z_trial_n": z_trial_n,
            "z_mean": z_mean,
            "z_std": z_std,
            "v": v,
            "d_z": d_z,
            "n_cells": n_cells,
            "device": device,
            "component_weights": component_weights,
            "component_weights_t": component_weights_t,
            "weighted_mse_loss": weighted_mse_loss,
            "pre_event_steps": pre_event_steps,
            "post_event_steps": post_event_steps,
            "context_start": context_start,
            "context_end": context_end,
            "context_len": context_len,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "horizon": horizon,
            "horizons": horizons,
            "n_augment": n_augment,
            "noise_frac": noise_frac,
            "aug_half_win": aug_half_win,
            "aug_start": aug_start,
            "aug_end": aug_end,
            "aug_train_ds": aug_train_ds,
            "aug_train_loader": aug_train_loader,
            "test_ds": test_ds,
            "test_loader": test_loader,
        }
    )

    if verbose:
        print("Built missing ICRNN prerequisites from base variables (s0/dpca/train-test split).")


def run_icrnn_cell12_from_namespace(
    ns: dict[str, Any],
    *,
    plot: bool = True,
    verbose: bool = True,
    n_epochs_icrnn: int = 160,
) -> ICRNNCell12Result:
    """Run the exact ICRNN pipeline from cell 12 using notebook variables in ns.

    Example usage in another notebook:
        from icrnn_cell12 import run_icrnn_cell12_from_namespace
        result = run_icrnn_cell12_from_namespace(globals(), plot=True, verbose=True)
    """

    _ensure_cell12_prerequisites(ns, verbose=verbose)

    required_vars = [
        "aug_train_loader",
        "aug_train_ds",
        "weighted_mse_loss",
        "component_weights_t",
        "component_weights",
        "horizon",
        "horizons",
        "d_z",
        "n_cells",
        "device",
        "z_std",
        "z_mean",
        "test_indices",
        "z_trial_n",
        "v",
        "zero_idx",
        "context_len",
        "z_trial",
        "time_axis",
    ]
    _require(ns, required_vars)

    if "test_loader_aug" in ns and "test_ds_aug" in ns:
        eval_loader, eval_ds = ns["test_loader_aug"], ns["test_ds_aug"]
    elif "test_loader" in ns and "test_ds" in ns:
        eval_loader, eval_ds = ns["test_loader"], ns["test_ds"]
    else:
        raise RuntimeError("Missing test loader/dataset: need (test_loader_aug, test_ds_aug) or (test_loader, test_ds).")

    aug_train_loader = ns["aug_train_loader"]
    aug_train_ds = ns["aug_train_ds"]
    weighted_mse_loss = ns["weighted_mse_loss"]
    component_weights_t = ns["component_weights_t"]
    component_weights = ns["component_weights"]
    horizon = ns["horizon"]
    horizons = ns["horizons"]
    d_z = ns["d_z"]
    n_cells = ns["n_cells"]
    device = ns["device"]
    z_std = ns["z_std"]
    z_mean = ns["z_mean"]
    test_indices = ns["test_indices"]
    z_trial_n = ns["z_trial_n"]
    v = ns["v"]
    zero_idx = ns["zero_idx"]
    context_len = ns["context_len"]
    z_trial = ns["z_trial"]
    time_axis = ns["time_axis"]

    model_icrnn = ICRNNStimForecast(latent_dim=d_z, stim_dim=n_cells, hidden_dim=128, rho=0.7).to(device)
    optimizer_icrnn = torch.optim.AdamW(model_icrnn.parameters(), lr=1e-4, weight_decay=3e-4)
    ema_icrnn = ParameterEMA(model_icrnn, decay=0.99)

    loss_history_icrnn: list[float] = []
    test_loss_history_icrnn: list[float] = []

    lambda_smooth = 0.05
    lambda_bound = 0.01

    train_t0_icrnn = time.perf_counter()
    for epoch in range(1, n_epochs_icrnn + 1):
        model_icrnn.train()
        running_loss = 0.0

        tf_ratio = max(0.0, 0.98 - (0.98 / 80) * (epoch - 1))

        for context_z_b, context_v_b, future_v_b, target_z_b in aug_train_loader:
            context_z_b = context_z_b.to(device)
            context_v_b = context_v_b.to(device)
            future_v_b = future_v_b.to(device)
            target_z_b = target_z_b.to(device)

            optimizer_icrnn.zero_grad(set_to_none=True)
            pred_z_b = model_icrnn(
                context_z_b,
                context_v_b,
                future_v_b,
                target_z=target_z_b,
                teacher_forcing=tf_ratio,
            )

            loss_recon = weighted_mse_loss(pred_z_b, target_z_b, component_weights_t)
            loss_smooth = temporal_smoothness_loss(pred_z_b, target_z_b)
            loss_bound = state_bound_loss(pred_z_b, bound=4.0)
            loss = loss_recon + lambda_smooth * loss_smooth + lambda_bound * loss_bound

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_icrnn.parameters(), 0.3)
            optimizer_icrnn.step()
            ema_icrnn.update(model_icrnn)

            running_loss += loss.item() * context_z_b.shape[0]

        epoch_loss = running_loss / len(aug_train_ds)
        loss_history_icrnn.append(epoch_loss)

        ema_icrnn.shadow.eval()
        with torch.no_grad():
            test_running = 0.0
            for context_z_b, context_v_b, future_v_b, target_z_b in eval_loader:
                context_z_b = context_z_b.to(device)
                context_v_b = context_v_b.to(device)
                future_v_b = future_v_b.to(device)
                target_z_b = target_z_b.to(device)
                pred_z_b = ema_icrnn.shadow(
                    context_z_b,
                    context_v_b,
                    future_v_b,
                    target_z=None,
                    teacher_forcing=0.0,
                )
                test_running += weighted_mse_loss(pred_z_b, target_z_b, component_weights_t).item() * context_z_b.shape[0]
            test_loss_history_icrnn.append(test_running / len(eval_ds))

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"[ICRNN] Epoch {epoch:03d}/{n_epochs_icrnn} | "
                f"loss={epoch_loss:.6f} | test_loss={test_loss_history_icrnn[-1]:.6f} | tf={tf_ratio:.3f}"
            )

    training_time_icrnn = time.perf_counter() - train_t0_icrnn

    ema_icrnn.shadow.eval()
    with torch.no_grad():
        (ctx_z_t, ctx_v_t, fut_v_t, tgt_z_t) = next(iter(eval_loader))
        ctx_z_t = ctx_z_t.to(device)
        ctx_v_t = ctx_v_t.to(device)
        fut_v_t = fut_v_t.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_t0_icrnn = time.perf_counter()
        pred_z_t_icrnn = ema_icrnn.shadow(ctx_z_t, ctx_v_t, fut_v_t, target_z=None, teacher_forcing=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_icrnn = time.perf_counter() - infer_t0_icrnn

    pred_z_t_icrnn = pred_z_t_icrnn.cpu().numpy()
    tgt_z_t = tgt_z_t.numpy()

    pred_z_icrnn = pred_z_t_icrnn * z_std + z_mean
    target_z_icrnn = tgt_z_t * z_std + z_mean

    weighted_mse_by_h_icrnn: list[float] = []
    weighted_r2_by_h_icrnn: list[float] = []

    for h in range(horizon):
        y_true_h = target_z_icrnn[:, h, :]
        y_pred_h = pred_z_icrnn[:, h, :]

        mse_per_comp = np.mean((y_pred_h - y_true_h) ** 2, axis=0)
        weighted_mse = float(np.sum(component_weights * mse_per_comp))

        r2_per_comp = np.zeros(d_z, dtype=np.float64)
        for comp in range(d_z):
            if np.var(y_true_h[:, comp]) < 1e-12:
                r2_per_comp[comp] = 0.0
            else:
                r2_per_comp[comp] = r2_score(y_true_h[:, comp], y_pred_h[:, comp])
        weighted_r2 = float(np.sum(component_weights * r2_per_comp))

        weighted_mse_by_h_icrnn.append(weighted_mse)
        weighted_r2_by_h_icrnn.append(weighted_r2)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(np.arange(1, n_epochs_icrnn + 1), loss_history_icrnn, lw=2, label="ICRNN (train)")
        axes[0].plot(np.arange(1, n_epochs_icrnn + 1), test_loss_history_icrnn, lw=2, ls="--", label="ICRNN (test)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Weighted MSE")
        axes[0].set_title("ICRNN Training & Test Loss")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.3)

        axes[1].plot(horizons, weighted_mse_by_h_icrnn, marker="^", lw=2, label="ICRNN")
        axes[1].set_xlabel("Forecast horizon")
        axes[1].set_ylabel("Weighted MSE")
        axes[1].set_title("ICRNN Test Weighted MSE vs Horizon")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.3)

        axes[2].plot(horizons, weighted_r2_by_h_icrnn, marker="^", lw=2, label="ICRNN")
        axes[2].axhline(0.0, color="k", ls="--", lw=1)
        axes[2].set_xlabel("Forecast horizon")
        axes[2].set_ylabel("Weighted $R^2$")
        axes[2].set_title("ICRNN Test Weighted $R^2$ vs Horizon")
        axes[2].legend(frameon=False)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    wmse_selection_steps = min(50, horizon)
    per_trial_wmse_icrnn = np.zeros(len(test_indices), dtype=np.float64)

    ema_icrnn.shadow.eval()
    with torch.no_grad():
        for i, tr in enumerate(test_indices):
            context_z_single = torch.from_numpy(z_trial_n[tr, :context_len, :]).unsqueeze(0).to(device)
            context_v_single = torch.from_numpy(v[tr, :context_len, :]).unsqueeze(0).to(device)
            future_v_single = torch.from_numpy(v[tr, zero_idx + 1 : zero_idx + 1 + horizon, :]).unsqueeze(0).to(device)

            pred_icrnn_n = ema_icrnn.shadow(
                context_z_single,
                context_v_single,
                future_v_single,
                target_z=None,
                teacher_forcing=0.0,
            )
            pred_icrnn_single_i = pred_icrnn_n.squeeze(0).cpu().numpy() * z_std.squeeze((0, 1)) + z_mean.squeeze((0, 1))

            true_post_i = z_trial[tr, zero_idx + 1 : zero_idx + 1 + horizon, :]
            diff_i = pred_icrnn_single_i[:wmse_selection_steps] - true_post_i[:wmse_selection_steps]
            mse_per_comp_i = np.mean(diff_i ** 2, axis=0)
            per_trial_wmse_icrnn[i] = float(np.sum(component_weights * mse_per_comp_i))

    best_test_idx_icrnn = int(np.argmin(per_trial_wmse_icrnn))
    trial_id = int(test_indices[best_test_idx_icrnn])

    if verbose:
        print(f"\nPer-trial post-stim weighted MSE (first {wmse_selection_steps} steps, test set, ICRNN):")
        for i, tr in enumerate(test_indices):
            marker = " <-- best" if i == best_test_idx_icrnn else ""
            print(f"  Trial {int(tr)}: wMSE = {per_trial_wmse_icrnn[i]:.6f}{marker}")
        print(
            f"\nUsing test trial {trial_id} (lowest post-stim wMSE over first {wmse_selection_steps} "
            f"steps = {per_trial_wmse_icrnn[best_test_idx_icrnn]:.6f}) for trajectory plots."
        )

    pre_steps = 5
    context_z_single = torch.from_numpy(z_trial_n[trial_id, :context_len, :]).unsqueeze(0).to(device)
    context_v_single = torch.from_numpy(v[trial_id, :context_len, :]).unsqueeze(0).to(device)
    future_v_single = torch.from_numpy(v[trial_id, zero_idx + 1 : zero_idx + 1 + horizon, :]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_icrnn_n = ema_icrnn.shadow(
            context_z_single,
            context_v_single,
            future_v_single,
            target_z=None,
            teacher_forcing=0.0,
        )

    pred_icrnn_single = pred_icrnn_n.squeeze(0).cpu().numpy() * z_std.squeeze((0, 1)) + z_mean.squeeze((0, 1))

    start_idx = zero_idx - pre_steps
    end_idx = zero_idx + 1 + horizon
    t_window = time_axis[start_idx:end_idx]
    true_window = z_trial[trial_id, start_idx:end_idx, :]

    pred_win_icrnn = np.full((pre_steps + 1 + horizon, d_z), np.nan, dtype=np.float32)
    pred_win_icrnn[pre_steps, :] = true_window[pre_steps, :]
    pred_win_icrnn[pre_steps + 1 :, :] = pred_icrnn_single

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
        for comp in range(3):
            axes[comp].plot(t_window, true_window[:, comp], lw=2, label="Actual", color="black")
            axes[comp].plot(t_window, pred_win_icrnn[:, comp], lw=2, ls="--", label="ICRNN", color="tab:green")
            axes[comp].axvline(0.0, color="gray", ls=":", lw=1)
            axes[comp].set_title(f"Trial {trial_id} (best test) - Component {comp + 1}")
            axes[comp].set_xlabel("Time from onset (s)")
            axes[comp].grid(alpha=0.3)

        axes[0].set_ylabel("dPCA value")
        axes[0].legend(frameon=False, fontsize=8)
        plt.suptitle(f"ICRNN Trial Trajectories - Best Test Trial {trial_id}", y=1.02, fontsize=13)
        plt.tight_layout()
        plt.show()

    onset_state = z_trial[trial_id, zero_idx, :]
    true_post = z_trial[trial_id, zero_idx + 1 : zero_idx + 1 + horizon, :]
    true_from_onset = np.vstack([onset_state[None, :], true_post])
    icrnn_from_onset = np.vstack([onset_state[None, :], pred_icrnn_single])

    if plot:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            true_from_onset[:, 0],
            true_from_onset[:, 1],
            true_from_onset[:, 2],
            color="black",
            lw=2,
            label="Actual",
        )
        ax.plot(
            icrnn_from_onset[:, 0],
            icrnn_from_onset[:, 1],
            icrnn_from_onset[:, 2],
            color="tab:green",
            lw=2,
            ls="--",
            label="ICRNN",
        )
        ax.scatter(onset_state[0], onset_state[1], onset_state[2], color="gray", s=60, zorder=5, label="Onset")
        ax.set_title(f"Trial {trial_id} (best test) - 3D comparison")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.show()

        fig, ax2d = plt.subplots(figsize=(7, 6))
        ax2d.plot(true_from_onset[:, 0], true_from_onset[:, 1], color="black", lw=2, label="Actual")
        ax2d.plot(icrnn_from_onset[:, 0], icrnn_from_onset[:, 1], color="tab:green", lw=2, ls="--", label="ICRNN")
        ax2d.scatter(onset_state[0], onset_state[1], color="gray", s=60, zorder=5, label="Onset")
        ax2d.set_title(f"Trial {trial_id} (best test) - 2D comparison (comps 1-2)")
        ax2d.set_xlabel("dPCA Component 1")
        ax2d.set_ylabel("dPCA Component 2")
        ax2d.legend(frameon=False, fontsize=8)
        ax2d.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if verbose:
        print(f"Training time (ICRNN): {training_time_icrnn:.2f} s ({training_time_icrnn / n_epochs_icrnn:.4f} s/epoch)")
        print(f"Inference time (ICRNN): {inference_time_icrnn:.6f} s")
        print("\nICRNN weighted metrics (first 5 horizons):")
        print(f"{'h':>4s}  {'ICRNN MSE':>11s}  {'ICRNN R²':>11s}")
        for h in range(min(5, horizon)):
            print(
                f"t+{h + 1:02d}  "
                f"{weighted_mse_by_h_icrnn[h]:11.5f}  "
                f"{weighted_r2_by_h_icrnn[h]:11.5f}"
            )

    return ICRNNCell12Result(
        model_icrnn=model_icrnn,
        optimizer_icrnn=optimizer_icrnn,
        ema_icrnn=ema_icrnn,
        n_epochs_icrnn=n_epochs_icrnn,
        loss_history_icrnn=loss_history_icrnn,
        test_loss_history_icrnn=test_loss_history_icrnn,
        training_time_icrnn=training_time_icrnn,
        inference_time_icrnn=inference_time_icrnn,
        pred_z_t_icrnn=pred_z_t_icrnn,
        pred_z_icrnn=pred_z_icrnn,
        target_z_icrnn=target_z_icrnn,
        weighted_mse_by_h_icrnn=weighted_mse_by_h_icrnn,
        weighted_r2_by_h_icrnn=weighted_r2_by_h_icrnn,
        per_trial_wmse_icrnn=per_trial_wmse_icrnn,
        best_test_idx_icrnn=best_test_idx_icrnn,
        trial_id=trial_id,
        pred_icrnn_single=pred_icrnn_single,
        pred_win_icrnn=pred_win_icrnn,
        true_from_onset=true_from_onset,
        icrnn_from_onset=icrnn_from_onset,
    )


def run_icrnn_cell12_and_export(
    ns: dict[str, Any],
    *,
    plot: bool = True,
    verbose: bool = True,
) -> ICRNNCell12Result:
    """Run and write outputs back to ns using the same names created in cell 12."""
    result = run_icrnn_cell12_from_namespace(ns, plot=plot, verbose=verbose)
    ns.update(
        {
            "model_icrnn": result.model_icrnn,
            "optimizer_icrnn": result.optimizer_icrnn,
            "ema_icrnn": result.ema_icrnn,
            "n_epochs_icrnn": result.n_epochs_icrnn,
            "loss_history_icrnn": result.loss_history_icrnn,
            "test_loss_history_icrnn": result.test_loss_history_icrnn,
            "training_time_icrnn": result.training_time_icrnn,
            "inference_time_icrnn": result.inference_time_icrnn,
            "pred_z_t_icrnn": result.pred_z_t_icrnn,
            "pred_z_icrnn": result.pred_z_icrnn,
            "target_z_icrnn": result.target_z_icrnn,
            "weighted_mse_by_h_icrnn": result.weighted_mse_by_h_icrnn,
            "weighted_r2_by_h_icrnn": result.weighted_r2_by_h_icrnn,
            "per_trial_wmse_icrnn": result.per_trial_wmse_icrnn,
            "best_test_idx_icrnn": result.best_test_idx_icrnn,
            "trial_id": result.trial_id,
            "pred_icrnn_single": result.pred_icrnn_single,
            "pred_win_icrnn": result.pred_win_icrnn,
            "true_from_onset": result.true_from_onset,
            "icrnn_from_onset": result.icrnn_from_onset,
        }
    )
    return result
