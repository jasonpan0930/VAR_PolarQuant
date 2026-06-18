"""Minimal recorder: one QKT heatmap for a single (stage, block)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class KVRecorder:
    def __init__(
        self,
        out_dir: str = "kv_records",
        target_stage: int = 9,
        target_block: int = 15,
        batch_index: int = 0,
        num_heads: int = 16,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_stage = target_stage
        self.target_block = target_block
        self.batch_index = batch_index
        self.num_heads = num_heads
        self.stage_si = -1
        self.stage_pn = 0
        self._saved = False

    def set_stage(self, si: int, pn: int) -> None:
        self.stage_si, self.stage_pn = si, pn

    def attach(self, var_model) -> None:
        var_model.blocks[self.target_block].attn._kv_recorder = self
        var_model._kv_recorder = self
        print(
            f"[KVRecorder] stage{self.target_stage} block{self.target_block:02d} only -> "
            f"{self.out_dir.resolve()}/stage{self.target_stage:02d}_pn*_block{self.target_block:02d}_QKT.png"
        )

    def detach(self, var_model) -> None:
        attn = var_model.blocks[self.target_block].attn
        if getattr(attn, "_kv_recorder", None) is self:
            attn._kv_recorder = None
        if getattr(var_model, "_kv_recorder", None) is self:
            var_model._kv_recorder = None

    @staticmethod
    def _to_BHLD(t: torch.Tensor, dim_cat: int) -> torch.Tensor:
        if dim_cat == 1:
            return t.permute(0, 2, 1, 3).contiguous()
        return t.contiguous()

    def record_kv(self, attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dim_cat: int) -> None:
        del v
        if self._saved or self.stage_si != self.target_stage or attn_module.block_idx != self.target_block:
            return

        q = self._to_BHLD(q, dim_cat).float()
        k = self._to_BHLD(k, dim_cat).float()
        scores = q @ k.transpose(-2, -1)
        if not attn_module.attn_l2_norm:
            scores = scores * attn_module.scale

        bi = min(self.batch_index, scores.shape[0] - 1)
        arr = scores[bi].abs().mean(0).cpu().numpy()

        stem = f"stage{self.stage_si:02d}_pn{self.stage_pn}_block{attn_module.block_idx:02d}_QKT"
        plt.figure(figsize=(10, 6))
        plt.imshow(arr, aspect="auto", cmap="viridis")
        plt.colorbar(label="mean |Q@K^T| over heads")
        plt.xlabel("key index (Lk)")
        plt.ylabel("query index (Lq)")
        plt.title(stem)
        plt.tight_layout()
        out = self.out_dir / f"{stem}.png"
        plt.savefig(out, dpi=160)
        plt.close()

        self._saved = True
        print(f"[KVRecorder] saved {out}")
