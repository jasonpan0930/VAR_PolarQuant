# Artifacts Directory

This directory is intentionally ignored by git.

Use it for generated experiment outputs: plots, metrics JSON files, KV dumps,
FID samples, logs, and temporary diagnostics. Keep reusable experiment inputs
such as theta2 codebooks under `configs/codebooks/` instead.

Current local legacy outputs:

- `angle_plots/`: historical K-error and sample-image plots.
- `cross_block_mse/`: historical forced/free cross-block metrics and figures.
- `codebook_runs/legacy_polar_quant_dumps/`: original theta2 K-means run folders.
- `logs/fid_and_kmeans_logs/`: historical SLURM/evaluator logs.
- `test_imgs/legacy_test_imgs/`: historical sample images used for visual checks.
- `misc/`: one-off diagnostics that do not fit a stable experiment family.
