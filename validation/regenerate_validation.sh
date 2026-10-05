#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after building the software.
# The command regenerates all seven optimizers for 2-5 thresholds using the
# current release. Current versions also write per-run convergence CSV files.

./build/segm --run_all \
  --batch_dir validation/input \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir validation/convergence/current_release_rerun
