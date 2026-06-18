"""MSE-weighted K-means for θ₂ 4-bit (16-level) codebook design."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

PI = math.pi
THETA2_LO, THETA2_HI = 0.0, PI / 2
N_THETA2_LEVELS = 16


def _clip_theta2(theta2: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(theta2, dtype=np.float64).ravel(), THETA2_LO, THETA2_HI)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / w.size
    return w / s


def kmeans_theta2_codebook(
    theta2: np.ndarray,
    weights: np.ndarray,
    n_levels: int = N_THETA2_LEVELS,
    seed: int = 0,
    n_init: int = 20,
) -> Tuple[np.ndarray, dict]:
    """
    MSE-weighted K-means on FP θ₂ samples ∈ [0, π/2].

    weights: per-sample importance (e.g. repeat per-K mse_after_full for each of 31 θ₂).
    Returns sorted centroids (n_levels,) and fit metadata.
    """
    x = _clip_theta2(theta2)
    w = _normalize_weights(weights)
    if x.size != w.size:
        raise ValueError(f'theta2 size {x.size} != weights size {w.size}')
    if x.size < n_levels:
        raise ValueError(f'need at least {n_levels} samples, got {x.size}')

    x_col = x.reshape(-1, 1)
    meta = {'n_samples': int(x.size), 'n_levels': n_levels, 'seed': seed}

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_levels, n_init=n_init, random_state=seed)
        km.fit(x_col, sample_weight=w)
        centers = km.cluster_centers_.ravel()
        labels = km.labels_
        meta['backend'] = 'sklearn'
        meta['inertia'] = float(km.inertia_)
    except ImportError:
        centers, labels, inertia = _kmeans_1d_weighted_lloyd(x, w, n_levels, seed=seed, max_iter=200)
        meta['backend'] = 'numpy_lloyd'
        meta['inertia'] = float(inertia)

    centers = np.sort(centers)
    labels = _assign_to_centers(x, centers)
    meta['weighted_mse_angle'] = float(_weighted_angle_mse(x, labels, centers, w))
    meta['uniform_mse_angle'] = float(
        _weighted_angle_mse(x, _assign_uniform(x, n_levels), _uniform_centers(n_levels), w)
    )
    return centers, meta


def _uniform_centers(n: int) -> np.ndarray:
    w = (THETA2_HI - THETA2_LO) / n
    return np.array([THETA2_LO + (i + 0.5) * w for i in range(n)], dtype=np.float64)


def _assign_uniform(x: np.ndarray, n: int) -> np.ndarray:
    cb = _uniform_centers(n)
    return np.argmin(np.abs(x[:, None] - cb[None, :]), axis=1)


def _assign_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)


def _weighted_angle_mse(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, w: np.ndarray) -> float:
    recon = centers[labels]
    return float(np.sum(w * (x - recon) ** 2))


def _kmeans_1d_weighted_lloyd(
    x: np.ndarray,
    w: np.ndarray,
    k: int,
    seed: int = 0,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted Lloyd on 1D; k-means++ style init."""
    rng = np.random.default_rng(seed)
    # weighted k-means++ init
    centers = np.empty(k, dtype=np.float64)
    idx0 = rng.choice(x.size, p=w)
    centers[0] = x[idx0]
    d2 = np.full(x.size, np.inf)
    for c in range(1, k):
        d2 = np.minimum(d2, (x - centers[c - 1]) ** 2)
        probs = w * d2
        probs /= probs.sum()
        j = rng.choice(x.size, p=probs)
        centers[c] = x[j]
    labels = np.zeros(x.size, dtype=np.int64)
    for _ in range(max_iter):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if not np.any(m):
                new_centers[j] = x[rng.integers(0, x.size)]
                continue
            wj = w[m]
            new_centers[j] = np.average(x[m], weights=wj)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    inertia = float(np.sum(w * (x - centers[labels]) ** 2))
    return np.sort(centers), labels, inertia


def save_theta2_codebook(path: Union[str, Path], centers: np.ndarray, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'centers': centers.tolist(), 'meta': meta or {}}
    if path.suffix == '.npy':
        np.save(path, centers)
        if meta:
            meta_path = path.with_suffix('.meta.json')
            meta_path.write_text(json.dumps(meta, indent=2))
    else:
        path.write_text(json.dumps(payload, indent=2))


def load_theta2_codebook(path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix == '.npy':
        centers = np.load(path)
        meta = {}
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return centers, meta
    data = json.loads(path.read_text())
    return np.asarray(data['centers'], dtype=np.float64), data.get('meta', {})
