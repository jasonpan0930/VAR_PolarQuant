# Artifacts Directory

This directory is intentionally ignored by git.

Use it for generated experiment outputs: plots, metrics JSON files, KV dumps,
FID samples, logs, and temporary diagnostics. Keep reusable experiment inputs
such as theta2 codebooks under `configs/codebooks/` instead.

This FPGA/core branch intentionally does not keep historical analysis outputs
in the working tree. Use `../VAR_polarQuant_backup/` if those files are needed.
