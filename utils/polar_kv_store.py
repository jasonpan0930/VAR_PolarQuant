"""Save polar-quantized KV cache dumps for offline research."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.polar_kv_quant import PolarK64Batch, PolarKVCache


class PolarKVDumpSession:
    """
    Collect per-layer polar K caches during autoregressive_infer_cfg.
    Attach via var._polar_dump = session.
    """

    def __init__(self, out_dir: str = 'artifacts/polar_kv_dumps', batch_index: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_index = batch_index
        self.stage_si = -1
        self.stage_pn = 0
        self.records: List[Dict[str, Any]] = []
        self._stage_dir: Optional[Path] = None

    def set_stage(self, si: int, pn: int) -> None:
        self.stage_si, self.stage_pn = si, pn
        self._stage_dir = self.out_dir / f'stage{si:02d}_pn{pn}'
        self._stage_dir.mkdir(parents=True, exist_ok=True)

    def on_k_cache_updated(
        self,
        block_idx: int,
        cache: PolarKVCache,
        k_new: Optional[torch.Tensor] = None,
    ) -> None:
        """Called after each stage forward through a block (full cache snapshot)."""
        if self._stage_dir is None:
            return
        polar = cache.as_batch()
        if self.batch_index != 0:
            b_slice = slice(self.batch_index, self.batch_index + 1)
            polar = PolarK64Batch(
                polar.q1[b_slice],
                polar.q2[b_slice],
                polar.z[b_slice],
            )
        meta = {
            'stage_si': self.stage_si,
            'stage_pn': self.stage_pn,
            'block_idx': block_idx,
            'cache_len': len(cache),
            'batch_index': self.batch_index,
        }
        fname = self._stage_dir / f'block{block_idx:02d}_k_polar.npz'
        polar.save_npz(fname, **meta)
        rec = {**meta, 'path': str(fname)}
        if k_new is not None:
            bi = min(self.batch_index, k_new.shape[0] - 1)
            k_hat_new = PolarK64Batch.from_k(k_new).decode()[bi]
            rec['k_new_mse'] = float(((k_new[bi].float() - k_hat_new.float()) ** 2).mean())
        self.records.append(rec)

    def save_manifest(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        manifest = {'records': self.records}
        if extra:
            manifest.update(extra)
        path = self.out_dir / 'manifest.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f'[PolarKVDump] manifest -> {path} ({len(self.records)} records)')
        return path
