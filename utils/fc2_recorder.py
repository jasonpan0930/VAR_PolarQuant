"""Histogram of FFN fc2 input (= GELU(fc1(x))) for one (stage, block)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class FC2InputRecorder:
    def __init__(
        self,
        out_dir: str = "fc2_records",
        target_stage: int = 9,
        target_block: int = 15,
        batch_index: int = 0,
        bins: int = 120,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_stage = target_stage
        self.target_block = target_block
        self.batch_index = batch_index
        self.bins = bins
        self.stage_si = -1
        self.stage_pn = 0
        self._saved = False

    def set_stage(self, si: int, pn: int) -> None:
        self.stage_si, self.stage_pn = si, pn

    def attach(self, var_model) -> None:
        var_model.blocks[self.target_block].ffn._fc2_recorder = self
        var_model._fc2_recorder = self
        print(
            f"[FC2InputRecorder] stage{self.target_stage} block{self.target_block:02d} -> "
            f"{self.out_dir.resolve()}/stage{self.target_stage:02d}_pn*_block{self.target_block:02d}_fc2_input_hist.png"
        )

    def detach(self, var_model) -> None:
        ffn = var_model.blocks[self.target_block].ffn
        if getattr(ffn, "_fc2_recorder", None) is self:
            ffn._fc2_recorder = None
        if getattr(var_model, "_fc2_recorder", None) is self:
            var_model._fc2_recorder = None

    def record_fc2_input(self, h: torch.Tensor) -> None:
        if self._saved or self.stage_si != self.target_stage:
            return

        bi = min(self.batch_index, h.shape[0] - 1)
        vals = h[bi].detach().float().cpu().numpy().ravel()

        stem = f"stage{self.stage_si:02d}_pn{self.stage_pn}_block{self.target_block:02d}_fc2_input_hist"
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=self.bins, color="steelblue", edgecolor="white", linewidth=0.2)
        plt.xlabel("activation value (fc2 input)")
        plt.ylabel("count")
        plt.title(f"{stem}\n(n={vals.size:,}, mean={vals.mean():.4f}, std={vals.std():.4f})")
        plt.tight_layout()
        out = self.out_dir / f"{stem}.png"
        plt.savefig(out, dpi=160)
        plt.close()

        self._saved = True
        print(f"[FC2InputRecorder] saved {out}")
