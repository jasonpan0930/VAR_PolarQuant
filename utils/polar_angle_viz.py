"""Collect and plot polar angle distributions (pre-quant vs dequant bin centers)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.polar_kv_quant import (
    HEAD_DIM,
    NUM_Q1,
    NUM_Q2,
    PI,
    encode_polar_k64_angles,
    polar_k64_error_breakdown,
)
from utils.angle_quant import (
    dequantize_theta2_numpy,
    get_polar_quant_config,
    get_theta1_scheme,
    get_theta2_scheme,
    set_polar_quant_config,
)

THETA2_LAYER_SIZES = (16, 8, 4, 2, 1)
THETA2_LAYER_NAMES = (
    'L0: 32→16 (16 θ₂)',
    'L1: 16→8 (8 θ₂)',
    'L2: 8→4 (4 θ₂)',
    'L3: 4→2 (2 θ₂)',
    'L4: 2→1 (1 θ₂)',
)

SHOW_THETA1_AFTER_QUANT = True  # set True for θ₁ before/after overlay, scatter, error plots
SHOW_THETA2_TREE_AFTER_QUANT = True  # set True to overlay dequant θ₂ on tree-layer plots
PLOT_K_ERROR_GLOBAL_ONLY = True  # if True, render_all_plots only writes k_error_global.png


def theta2_layer_id(index: int) -> int:
    offset = 0
    for layer, n in enumerate(THETA2_LAYER_SIZES):
        if index < offset + n:
            return layer
        offset += n
    return len(THETA2_LAYER_SIZES) - 1


def _subsample(flat: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if flat.size <= max_n:
        return flat
    idx = rng.choice(flat.size, size=max_n, replace=False)
    return flat[idx]


def _subsample_paired(flat_arrays: Dict[str, np.ndarray], max_n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Same index for all arrays — required for scatter / error plots."""
    n = next(iter(flat_arrays.values())).size
    if n <= max_n:
        return flat_arrays
    idx = rng.choice(n, size=max_n, replace=False)
    return {k: v[idx] for k, v in flat_arrays.items()}


@dataclass
class PolarAngleStatsSession:
    """Accumulate θ before/after quant during inference (hook from SelfAttention)."""

    target_blocks: Optional[Tuple[int, ...]] = None
    max_samples_per_stage: int = 20000
    max_structured_tokens: int = 512
    max_k_vectors: int = 8000
    batch_index: int = 0

    stage_si: int = -1
    stage_pn: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    records: List[Dict] = field(default_factory=list)
    structured_chunks: List[Dict[str, np.ndarray]] = field(default_factory=list)
    k_error_records: List[Dict[str, np.ndarray]] = field(default_factory=list)
    # For K-means θ₂ codebook: FP θ₂ per angle + full-pipeline MSE weight (per K vector).
    theta2_kmeans_theta2: List[np.ndarray] = field(default_factory=list)
    theta2_kmeans_weights: List[np.ndarray] = field(default_factory=list)

    def set_stage(self, si: int, pn: int) -> None:
        self.stage_si, self.stage_pn = si, pn

    def record_k(self, k_blhc: torch.Tensor, block_idx: int) -> None:
        if self.target_blocks is not None and block_idx not in self.target_blocks:
            return
        k = k_blhc[self.batch_index : self.batch_index + 1].float()
        ang = encode_polar_k64_angles(k)
        rec = {
            'stage_si': self.stage_si,
            'stage_pn': self.stage_pn,
            'block_idx': block_idx,
            'theta1': ang['theta1'].reshape(-1).cpu().numpy(),
            'theta1_hat': ang['theta1_hat'].reshape(-1).cpu().numpy(),
            'q1': ang['q1'].reshape(-1).cpu().numpy(),
            'theta2': ang['theta2'].reshape(-1).cpu().numpy(),
            'theta2_hat': ang['theta2_hat'].reshape(-1).cpu().numpy(),
            'q2': ang['q2'].reshape(-1).cpu().numpy(),
        }
        # θ₁ has 32 angles / K, θ₂ has 31 — flatten lengths differ; subsample each group with its own idx.
        g1 = _subsample_paired(
            {k: rec[k] for k in ('theta1', 'theta1_hat', 'q1')},
            self.max_samples_per_stage, self._rng,
        )
        g2 = _subsample_paired(
            {k: rec[k] for k in ('theta2', 'theta2_hat', 'q2')},
            self.max_samples_per_stage, self._rng,
        )
        rec.update(g1)
        rec.update(g2)
        self.records.append(rec)

        t1_flat = ang['theta1'][0].reshape(-1, NUM_Q1)
        n_tok = t1_flat.shape[0]
        n_take = min(n_tok, self.max_structured_tokens)
        if n_tok > n_take:
            idx = self._rng.choice(n_tok, size=n_take, replace=False)
            t1_flat = t1_flat[idx]
            t1h_flat = ang['theta1_hat'][0].reshape(-1, NUM_Q1)[idx]
            t2_flat = ang['theta2'][0].reshape(-1, NUM_Q2)[idx]
            t2h_flat = ang['theta2_hat'][0].reshape(-1, NUM_Q2)[idx]
        else:
            t1h_flat = ang['theta1_hat'][0].reshape(-1, NUM_Q1)
            t2_flat = ang['theta2'][0].reshape(-1, NUM_Q2)
            t2h_flat = ang['theta2_hat'][0].reshape(-1, NUM_Q2)
        self.structured_chunks.append({
            'stage_si': self.stage_si,
            'block_idx': block_idx,
            'theta1': t1_flat.cpu().numpy(),
            'theta1_hat': t1h_flat.cpu().numpy(),
            'theta2': t2_flat.cpu().numpy(),
            'theta2_hat': t2h_flat.cpu().numpy(),
        })

        k_flat = k[0].reshape(-1, HEAD_DIM)
        bd = polar_k64_error_breakdown(k_flat)
        n_vec = k_flat.shape[0]
        if n_vec > self.max_k_vectors:
            vidx = self._rng.choice(n_vec, size=self.max_k_vectors, replace=False)
            bd = {key: val[vidx].cpu().numpy() for key, val in bd.items() if val.dim() > 0}
        else:
            bd = {key: val.cpu().numpy() if val.dim() > 0 else float(val.cpu()) for key, val in bd.items()}
        bd['stage_si'] = np.array([self.stage_si])
        bd['stage_pn'] = np.array([self.stage_pn])
        self.k_error_records.append(bd)

        mse_vec = np.asarray(bd['mse_after_full'], dtype=np.float64).ravel()
        t2_mat = ang['theta2'][0].reshape(-1, NUM_Q2).detach().cpu().numpy()
        n_vec = t2_mat.shape[0]
        if n_vec > self.max_k_vectors:
            vidx = self._rng.choice(n_vec, size=self.max_k_vectors, replace=False)
            t2_mat = t2_mat[vidx]
            mse_vec = mse_vec[vidx]
        self.theta2_kmeans_theta2.append(t2_mat.reshape(-1))
        self.theta2_kmeans_weights.append(np.repeat(mse_vec, NUM_Q2))

    def aggregate_theta2_mse_for_kmeans(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.theta2_kmeans_theta2:
            raise ValueError('no θ₂ samples for K-means (enable angle stats during inference)')
        return (
            np.concatenate(self.theta2_kmeans_theta2),
            np.concatenate(self.theta2_kmeans_weights),
        )

    def aggregate_k_errors(self) -> Dict[str, np.ndarray]:
        if not self.k_error_records:
            raise ValueError('no K error records collected')
        keys = [k for k in self.k_error_records[0] if not k.startswith(('stage_',)) and k != 'dim_mean_abs_full']
        out: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
        dim_sum, dim_cnt = None, 0
        for rec in self.k_error_records:
            for k in keys:
                out[k].append(rec[k])
            d = rec['dim_mean_abs_full']
            dim_sum = d if dim_sum is None else dim_sum + d
            dim_cnt += 1
        result = {k: np.concatenate(v) for k, v in out.items()}
        result['dim_mean_abs_full'] = dim_sum / max(dim_cnt, 1)
        return result

    def aggregate(self) -> Dict[str, np.ndarray]:
        if not self.records:
            raise ValueError('no angle records collected')
        out: Dict[str, List[np.ndarray]] = {k: [] for k in self.records[0] if k.startswith(('theta', 'q'))}
        for rec in self.records:
            for k in out:
                out[k].append(rec[k])
        return {k: np.concatenate(v) for k, v in out.items()}

    def aggregate_structured(self) -> Dict[str, np.ndarray]:
        if not self.structured_chunks:
            raise ValueError('no structured angle records')
        keys = ('theta1', 'theta1_hat', 'theta2', 'theta2_hat')
        return {k: np.concatenate([c[k] for c in self.structured_chunks], axis=0) for k in keys}


def load_angles_from_npz(dump_dir: Path, block_idx: int = 15, max_files: int = 10) -> Dict[str, np.ndarray]:
    """Dequant-only stats from saved npz (no original θ unless re-inferred)."""
    files = sorted(dump_dir.glob(f'stage*_pn*/block{block_idx:02d}_k_polar.npz'))[:max_files]
    if not files:
        raise FileNotFoundError(f'no npz under {dump_dir} for block {block_idx}')
    q1_all, q2_all = [], []
    for f in files:
        data = np.load(f)
        if '_meta_json' in data:
            meta = json.loads(str(data['_meta_json']))
            set_polar_quant_config(meta.get('polar_quant', meta.get('theta2_quant', 'uniform_int4')))
        q1_all.append(data['q1'].reshape(-1))
        q2_all.append(data['q2'].reshape(-1))
    q1 = np.concatenate(q1_all)
    q2 = np.concatenate(q2_all)
    theta1_hat = get_theta1_scheme().codebook_numpy()[q1.astype(np.int64).clip(0, get_theta1_scheme().num_bins - 1)]
    theta2_hat = dequantize_theta2_numpy(q2)
    return {'q1': q1, 'q2': q2, 'theta1_hat': theta1_hat, 'theta2_hat': theta2_hat}


def _bin_centers_1() -> np.ndarray:
    return get_theta1_scheme().codebook_numpy()


def _bin_centers_2() -> np.ndarray:
    return get_theta2_scheme().codebook_numpy()


def _theta2_plot_meta() -> Tuple[str, np.ndarray, float]:
    scheme = get_theta2_scheme()
    centers = scheme.codebook_numpy()
    return scheme.label, scheme.bin_widths(), scheme.typical_step()


def _plot_fp_kde_int_density_compare(
    ax,
    fp: np.ndarray,
    int_deq: np.ndarray,
    delta: float,
    bin_centers: np.ndarray,
    x_lo: float,
    x_hi: float,
    title: str,
    xlabel: str,
    q_codes: Optional[np.ndarray] = None,
    kde_max_points: int = 50000,
    bin_widths: Optional[np.ndarray] = None,
) -> None:
    """
    FP: KDE curve (area ≈ 1). INT dequant: bar height = count / (N * delta) so bar area = count/N.
    Both on comparable density scale — matches standard quant visualization.
    """
    fp = np.asarray(fp, dtype=np.float64).ravel()
    int_deq = np.asarray(int_deq, dtype=np.float64).ravel()
    n_levels = len(bin_centers)
    N = int_deq.size

    if q_codes is not None:
        counts = np.bincount(
            q_codes.astype(np.int64).clip(0, n_levels - 1),
            minlength=n_levels,
        )
    else:
        counts = np.array([np.sum(np.isclose(int_deq, lev)) for lev in bin_centers])

    levels = bin_centers
    widths = bin_widths if bin_widths is not None else np.full(n_levels, delta, dtype=np.float64)
    heights = counts / np.maximum(N * widths, 1e-12)

    rng = np.random.default_rng(0)
    fp_kde = fp if fp.size <= kde_max_points else fp[rng.choice(fp.size, kde_max_points, replace=False)]

    try:
        import seaborn as sns
        sns.kdeplot(
            x=fp_kde, ax=ax, label='FP density', linewidth=2, color='#4C72B0',
            clip=(x_lo, x_hi), warn_singular=False,
        )
    except ImportError:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x_lo, x_hi, 400)
            kde = gaussian_kde(fp_kde)
            ax.plot(xs, kde(xs), label='FP density', color='#4C72B0', linewidth=2)
        except ImportError:
            ax.hist(
                fp_kde, bins=min(64, n_levels * 2), range=(x_lo, x_hi),
                density=True, alpha=0.35, color='#4C72B0', label='FP density (hist fallback)',
            )

    ax.bar(
        levels, heights, width=widths * 0.92, alpha=0.45, align='center',
        color='#DD8452', label='quant density bars', zorder=3,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel('density')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def _plot_angle_cdf_compare(
    ax,
    before: np.ndarray,
    after: np.ndarray,
    x_lo: float,
    x_hi: float,
    title: str,
    xlabel: str,
) -> None:
    """CDF: both curves go 0→1; gap = distributional shift from quant."""
    for arr, label, color in ((before, 'before quant', '#4C72B0'), (after, 'after quant', '#DD8452')):
        xs = np.sort(arr)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, label=label, color=color, lw=1.5, alpha=0.9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.set_ylabel('cumulative fraction')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_theta1_overview(data: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    has_orig = 'theta1' in data
    t1 = data.get('theta1')
    t1h = data.get('theta1_hat')
    err = (t1 - t1h) if has_orig and t1h is not None and SHOW_THETA1_AFTER_QUANT else None
    q1 = data.get('q1')

    if not SHOW_THETA1_AFTER_QUANT:
        if not has_orig:
            print('skip theta1_overview: no before-quant θ₁ in data')
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('θ₁ first polar — before quant (INT6, 32× per K vector)', fontsize=13)
        ax.hist(t1, bins=64, range=(-PI, PI), alpha=0.85, color='#4C72B0', density=True)
        ax.set_xlabel('θ₁ (rad)')
        ax.set_ylabel('density')
        ax.set_title('θ₁ distribution  (−π, π)')
        fig.tight_layout()
        path = out_dir / f'{prefix}theta1_overview.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'saved {path}')
        return

    q1_scheme = get_theta1_scheme()
    q1_step = q1_scheme.typical_step()
    q1_bins = q1_scheme.num_bins
    q1 = q1 if q1 is not None else np.argmin(
        np.abs(t1h[:, None] - q1_scheme.codebook_numpy()[None, :]), axis=1,
    ).astype(np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'θ₁ first polar ({q1_scheme.label}, 32× per K vector)', fontsize=13)

    ax = axes[0, 0]
    if has_orig:
        _plot_fp_kde_int_density_compare(
            ax, t1, t1h, q1_step, _bin_centers_1(), -PI, PI,
            title='θ₁ FP KDE vs INT bars  (−π, π)',
            xlabel='θ₁ (rad)',
            q_codes=q1,
        )
    else:
        ax.hist(t1h, bins=64, range=(-PI, PI), alpha=0.55, color='#DD8452', label='after quant')
        ax.set_xlabel('θ₁ (rad)')
        ax.set_ylabel('fraction of samples')
        ax.set_title('θ₁ after quant')

    ax = axes[0, 1]
    counts = np.bincount(q1.astype(np.int64).clip(0, q1_bins - 1), minlength=q1_bins)
    ax.bar(np.arange(q1_bins), counts / counts.sum(), color='#55A868', width=0.9)
    ax.set_xlabel('q₁ bin index')
    ax.set_ylabel('fraction')
    ax.set_title(f'q₁ bin occupancy ({q1_scheme.label})')

    ax = axes[1, 0]
    if has_orig:
        ax.scatter(t1, t1h, s=2, alpha=0.15, c='#4C72B0', rasterized=True)
        lim = [-PI, PI]
        ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('θ₁ before quant')
        ax.set_ylabel('θ₁ after quant')
        ax.set_title('before vs after (1:1 paired)')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'need inference hook\nfor before-quant θ₁', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1, 1]
    if has_orig and err is not None:
        ax.hist(err, bins=64, range=(-q1_step, q1_step), color='#C44E52', alpha=0.85, density=True)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('θ₁ error (after − before)')
        ax.set_ylabel('density')
        ax.set_title(f'quant error  mean={err.mean():.4f}  std={err.std():.4f}')
    else:
        ax.set_axis_off()

    fig.tight_layout()
    path = out_dir / f'{prefix}theta1_overview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')


def plot_theta1_per_index(records: List[Dict], out_dir: Path, prefix: str = '') -> None:
    """Per pair-index θ₁ error (32 panels). Requires non-flattened record; re-collect in exp."""
    pass  # filled by exp script with per-index arrays


def _plot_angle_kde_pair(
    ax,
    before: np.ndarray,
    after: np.ndarray,
    x_lo: float,
    x_hi: float,
    title: str,
    xlabel: str,
    before_label: str = 'before quant',
    after_label: str = 'after roundtrip',
    kde_max_points: int = 50000,
) -> None:
    rng = np.random.default_rng(0)
    b = before if before.size <= kde_max_points else before[rng.choice(before.size, kde_max_points, replace=False)]
    a = after if after.size <= kde_max_points else after[rng.choice(after.size, kde_max_points, replace=False)]
    try:
        import seaborn as sns
        sns.kdeplot(x=b, ax=ax, label=before_label, color='#4C72B0', clip=(x_lo, x_hi), warn_singular=False)
        sns.kdeplot(x=a, ax=ax, label=after_label, color='#DD8452', clip=(x_lo, x_hi), warn_singular=False)
    except ImportError:
        ax.hist(b, bins=48, range=(x_lo, x_hi), density=True, alpha=0.45, color='#4C72B0', label=before_label)
        ax.hist(a, bins=48, range=(x_lo, x_hi), density=True, alpha=0.45, color='#DD8452', label=after_label)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel('density')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_theta2_overview(data: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    has_orig = 'theta2' in data
    t2 = data.get('theta2')
    t2h = data['theta2_hat']
    err = (t2 - t2h) if has_orig else None
    q2 = data.get('q2')
    scheme_label, _, err_half_range = _theta2_plot_meta()
    if has_orig and err is not None:
        err_half_range = max(err_half_range, float(np.percentile(np.abs(err), 99.5)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f'θ₂ merge-tree polar ({scheme_label}, 31× per K vector)\n'
        f'after = full roundtrip (1×FP z + 63 quant angles → decode → re-polar)',
        fontsize=12,
    )
    hi = PI / 2

    ax = axes[0, 0]
    if has_orig:
        _plot_angle_kde_pair(
            ax, t2, t2h, 0.0, hi,
            title=f'θ₂ before vs full roundtrip  [0, π/2]',
            xlabel='θ₂ (rad)',
            after_label='after roundtrip',
        )
    else:
        ax.hist(t2h, bins=32, range=(0, hi), alpha=0.85, color='#DD8452')
        ax.set_ylabel('count')
        ax.set_xlabel('θ₂ (rad)')
        ax.set_title('θ₂ after roundtrip')

    ax = axes[0, 1]
    q2_scheme = get_theta2_scheme()
    q2_bins = q2_scheme.num_bins
    if q2 is not None:
        counts = np.bincount(q2.astype(np.int64).clip(0, q2_bins - 1), minlength=q2_bins)
        ax.bar(np.arange(q2_bins), counts / counts.sum(), color='#55A868', width=0.85)
        ax.set_xlabel('q₂ bin index')
        ax.set_ylabel('fraction of samples')
        ax.set_title(f'q₂ bin occupancy ({scheme_label})')
    else:
        ax.set_axis_off()

    ax = axes[1, 0]
    if has_orig:
        ax.scatter(t2, t2h, s=2, alpha=0.15, c='#4C72B0', rasterized=True)
        ax.plot([0, hi], [0, hi], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel('θ₂ before quant')
        ax.set_ylabel('θ₂ after roundtrip')
        ax.set_title('before vs after roundtrip (1:1 paired)')
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if has_orig and err is not None:
        ax.hist(
            err, bins=48,
            range=(-err_half_range, err_half_range),
            color='#C44E52', alpha=0.85, density=True,
        )
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('θ₂ error (roundtrip − before)')
        ax.set_title(f'roundtrip error  mean={err.mean():.4f}  std={err.std():.4f}')
    else:
        ax.set_axis_off()

    fig.tight_layout()
    path = out_dir / f'{prefix}theta2_overview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')

    if has_orig:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_angle_cdf_compare(
            ax, t2, t2h, 0.0, hi,
            title='θ₂ CDF — before vs full roundtrip',
            xlabel='θ₂ (rad)',
        )
        fig.tight_layout()
        cdf_path = out_dir / f'{prefix}theta2_cdf.png'
        fig.savefig(cdf_path, dpi=150)
        plt.close(fig)
        print(f'saved {cdf_path}')


def plot_theta2_by_tree_layer(data: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    has_orig = 'theta2' in data
    t2 = data.get('theta2', data['theta2_hat'])
    t2h = data['theta2_hat']
    q2 = data.get('q2')
    # per-sample flat: cannot split by layer without (N,31) shape
    # use q2 mod - for flat mixed index use approximate layer by q2 index position in 31-vector
    # When flat from aggregate(), we lost layer id — plot from last record with structured data
    _ = (has_orig, t2, t2h, q2)
    pass


def plot_theta2_tree_layers_from_struct(
    theta2: np.ndarray,
    theta2_hat: np.ndarray,
    out_dir: Path,
    prefix: str = '',
) -> None:
    """theta2, theta2_hat: (N, 31)"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    title = (
        'θ₂ by merge-tree layer (before quant)'
        if not SHOW_THETA2_TREE_AFTER_QUANT
        else 'θ₂ by merge-tree layer (before vs full roundtrip)'
    )
    fig.suptitle(title, fontsize=13)
    offset = 0
    for li, (n, name) in enumerate(zip(THETA2_LAYER_SIZES, THETA2_LAYER_NAMES)):
        ax = axes[li]
        sl = slice(offset, offset + n)
        t_raw = theta2[:, sl].ravel()
        ax.hist(t_raw, bins=24, range=(0, PI / 2), alpha=0.85, label='before', density=True, color='#4C72B0')
        if SHOW_THETA2_TREE_AFTER_QUANT:
            t_hat = theta2_hat[:, sl].ravel()
            ax.hist(t_hat, bins=24, range=(0, PI / 2), alpha=0.5, label='roundtrip', density=True, color='#DD8452')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('θ₂')
        if li == 0 and SHOW_THETA2_TREE_AFTER_QUANT:
            ax.legend(fontsize=7)
        offset += n
    axes[-1].set_axis_off()
    fig.tight_layout()
    path = out_dir / f'{prefix}theta2_tree_layers.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')


def plot_theta1_per_index_struct(theta1: np.ndarray, theta1_hat: np.ndarray, out_dir: Path, prefix: str = '') -> None:
    """theta1, theta1_hat: (N, 32) — per pair index before quant, or quant error if after mode."""
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    if SHOW_THETA1_AFTER_QUANT:
        err = theta1_hat - theta1
        fig.suptitle('θ₁ quant error per pair index (32 first-polar pairs)', fontsize=12)
        for i in range(NUM_Q1):
            ax = axes[i // 8, i % 8]
            ax.hist(err[:, i], bins=20, range=(-get_theta1_scheme().typical_step(), get_theta1_scheme().typical_step()), color='#4C72B0', alpha=0.85)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f'i={i}', fontsize=7)
            ax.tick_params(labelsize=5)
        fname = f'{prefix}theta1_per_index_error.png'
    else:
        fig.suptitle('θ₁ before quant per pair index (32 first-polar pairs)', fontsize=12)
        for i in range(NUM_Q1):
            ax = axes[i // 8, i % 8]
            ax.hist(theta1[:, i], bins=24, range=(-PI, PI), color='#4C72B0', alpha=0.85, density=True)
            ax.set_title(f'i={i}', fontsize=7)
            ax.tick_params(labelsize=5)
        fname = f'{prefix}theta1_per_index_before.png'
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')


def plot_k_error_global(kerr: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    """Global per-K-vector error histograms: NRMSE by stage + cosine similarity."""
    cfg = get_polar_quant_config()
    k_norm = np.maximum(kerr['k_norm'], 1e-12)
    cos_sim = kerr['cosine_sim_full']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'K vector reconstruction quality (64-dim, per token×head)\n{cfg.label}',
        fontsize=12,
    )

    # ---- left: NRMSE by stage ----
    ax = axes[0]
    stages = (
        ('after_theta1', 'θ₁ quant only\n(exact y)'),
        ('after_theta2', '+ θ₂ tree quant\n(exact θ₁)'),
        ('after_full', 'full pipeline\n(θ₁+θ₂+z quant)'),
    )
    colors = ('#4C72B0', '#DD8452', '#C44E52')

    nrmse_full = np.sqrt(kerr['mse_after_full']) / k_norm * 100  # percent
    nrmse_xmax = 1.5

    for (suf, label), c in zip(stages, colors):
        nrmse = np.sqrt(kerr[f'mse_{suf}']) / k_norm * 100  # percent
        ax.hist(
            nrmse, bins=400, range=(0.0, nrmse_xmax),
            alpha=0.55, label=label, color=c, density=True,
        )
    ax.set_xlim(0.0, nrmse_xmax)
    ax.set_xlabel('NRMSE = √MSE / ‖k‖ × 100  (%)  — lower is better')
    ax.set_ylabel('density')
    ax.set_title('normalized RMSE by stage')
    ax.legend(fontsize=8)

    nrmse_mean = float(np.mean(nrmse_full))
    nrmse_median = float(np.median(nrmse_full))
    nrmse_p95 = float(np.percentile(nrmse_full, 95.0))
    ax.text(
        0.98, 0.98,
        f'NRMSE  (lower = better)\n'
        f'mean   = {nrmse_mean:.3f} %\n'
        f'median = {nrmse_median:.3f} %\n'
        f'p95    = {nrmse_p95:.3f} %',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#C44E52'),
    )

    # ---- right: cosine similarity ----
    ax = axes[1]
    cos_lo = max(0.95, float(np.percentile(cos_sim, 0.5)) - 0.01)
    ax.hist(
        cos_sim, bins=300, range=(cos_lo, 1.0),
        alpha=0.75, color='#55A868', density=True,
    )
    ax.set_xlim(cos_lo, 1.0)
    ax.set_xlabel('cos(k, k̂)  (higher is better)')
    ax.set_ylabel('density')
    ax.set_title('cosine similarity: original vs reconstructed K')

    cos_mean = float(np.mean(cos_sim))
    cos_median = float(np.median(cos_sim))
    cos_p1 = float(np.percentile(cos_sim, 1.0))
    ax.text(
        0.02, 0.98,
        f'cos(k, k̂)  (higher = better)\n'
        f'mean   = {cos_mean:.6f}\n'
        f'median = {cos_median:.6f}\n'
        f'p1     = {cos_p1:.6f} (worst 1%)',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#55A868'),
    )

    fig.tight_layout()
    path = out_dir / f'{prefix}k_error_global.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(
        f'saved {path}  NRMSE mean={nrmse_mean:.3f}% median={nrmse_median:.3f}% '
        f'p95={nrmse_p95:.3f}%  cos mean={cos_mean:.6f} median={cos_median:.6f}'
    )


def plot_k_error_mse(kerr: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    """Global per-K-vector absolute MSE distribution (3 stages overlaid)."""
    cfg = get_polar_quant_config()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f'K vector error propagation (64-dim, per token×head)\n{cfg.label}',
        fontsize=12,
    )

    stages = (
        ('after_theta1', 'θ₁ quant only\n(exact y)'),
        ('after_theta2', '+ θ₂ tree quant\n(exact θ₁)'),
        ('after_full', 'full pipeline\n(θ₁+θ₂+z quant)'),
    )
    colors = ('#4C72B0', '#DD8452', '#C44E52')

    mse_full = kerr['mse_after_full']
    p99 = float(np.percentile(mse_full, 99.0))
    mse_xmax = max(0.00015, p99 * 1.05)
    for (suf, label), c in zip(stages, colors):
        ax.hist(
            kerr[f'mse_{suf}'], bins=300, range=(0.0, mse_xmax),
            alpha=0.5, label=label, color=c, density=True,
        )
    ax.set_xlim(0.0, mse_xmax)
    ax.set_xlabel('MSE per K vector')
    ax.set_ylabel('density')
    ax.set_title('MSE distribution by stage')
    ax.legend(fontsize=8)

    mse_mean = float(np.mean(mse_full))
    mse_median = float(np.median(mse_full))
    mse_std = float(np.std(mse_full))
    ax.text(
        0.98, 0.98,
        f'full pipeline MSE\nmean = {mse_mean:.3e}\nmedian = {mse_median:.3e}\nstd  = {mse_std:.3e}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#C44E52'),
    )

    fig.tight_layout()
    path = out_dir / f'{prefix}k_error_mse.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}  (full MSE mean={mse_mean:.3e} median={mse_median:.3e} std={mse_std:.3e})')


def plot_k_error_cascade(kerr: Dict[str, np.ndarray], out_dir: Path, prefix: str = '') -> None:
    """How much each stage adds on top of the previous (median per K vector)."""
    mse1 = kerr['mse_after_theta1']
    mse2 = kerr['mse_after_theta2']
    msef = kerr['mse_after_full']
    delta_t2 = np.maximum(mse2 - mse1, 0.0)
    delta_t1_full = np.maximum(msef - mse2, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Incremental error added per stage (per K vector)', fontsize=12)

    ax = axes[0]
    data = [mse1, delta_t2, delta_t1_full]
    labels = ['θ₁ quant', 'θ₂ tree added', 'θ₁ re-quant added']
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp['boxes'], ('#4C72B0', '#DD8452', '#C44E52')):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('MSE contribution')
    ax.set_title('error decomposition (global)')

    ax = axes[1]
    n = min(3000, msef.size)
    rng = np.random.default_rng(3)
    idx = rng.choice(msef.size, n, replace=False) if msef.size > n else np.arange(msef.size)
    ax.scatter(mse1[idx], msef[idx], s=4, alpha=0.2, c='#8172B3')
    lim = max(mse1[idx].max(), msef[idx].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
    ax.set_xlabel('MSE after θ₁-only')
    ax.set_ylabel('MSE after full pipeline')
    ax.set_title('per K vector: θ₁-only vs full')

    fig.tight_layout()
    path = out_dir / f'{prefix}k_error_cascade.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')


def plot_quant_landing_strip(theta: np.ndarray, theta_hat: np.ndarray, out_dir: Path, name: str, prefix: str = '') -> None:
    """1D strip: each original angle → dequant target (subsampled)."""
    rng = np.random.default_rng(2)
    n = min(5000, theta.size)
    idx = rng.choice(theta.size, n, replace=False)
    order = np.argsort(theta[idx])
    t = theta[idx][order]
    th = theta_hat[idx][order]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(n)
    ax.scatter(x, t, s=2, c='#4C72B0', alpha=0.4, label='before quant')
    ax.scatter(x, th, s=2, c='#DD8452', alpha=0.4, label='after quant')
    for i in range(n):
        ax.plot([i, i], [t[i], th[i]], c='gray', lw=0.2, alpha=0.3)
    ax.set_xlabel('samples (sorted by θ before quant)')
    ax.set_ylabel('angle (rad)')
    ax.set_title(f'{name}: original → dequant landing')
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    path = out_dir / f'{prefix}{name}_landing_strip.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved {path}')


def render_all_plots(
    data: Dict[str, np.ndarray],
    out_dir: Path,
    structured: Optional[Dict[str, np.ndarray]] = None,
    kerr: Optional[Dict[str, np.ndarray]] = None,
    prefix: str = '',
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if kerr is not None:
        plot_k_error_global(kerr, out_dir, prefix)
        plot_k_error_mse(kerr, out_dir, prefix)
    if PLOT_K_ERROR_GLOBAL_ONLY:
        return
    plot_theta1_overview(data, out_dir, prefix)
    plot_theta2_overview(data, out_dir, prefix)
    if SHOW_THETA1_AFTER_QUANT and 'theta1' in data and 'theta1_hat' in data:
        plot_quant_landing_strip(data['theta1'], data['theta1_hat'], out_dir, 'theta1', prefix)
    if 'theta2' in data and 'theta2_hat' in data:
        plot_quant_landing_strip(data['theta2'], data['theta2_hat'], out_dir, 'theta2', prefix)
    if structured:
        plot_theta1_per_index_struct(structured['theta1'], structured['theta1_hat'], out_dir, prefix)
        plot_theta2_tree_layers_from_struct(structured['theta2'], structured['theta2_hat'], out_dir, prefix)
