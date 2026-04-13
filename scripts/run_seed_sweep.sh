#!/bin/bash
# Run the seed sweep: 3 seeds on best config per backbone.
# 5 backbones x 1 expansion x 1 K x 3 seeds = 15 runs.
# NOTE: Update configs/sweeps/seed_sweep.yaml with best expansion/k per backbone
#       after analysing main sweep results.

set -e
source /mnt/NAS/home/ds5725/visaebench-internal/.visaebench/bin/activate
cd /mnt/NAS/home/ds5725/visaebench-internal

python3 -m src.training.sweep_runner \
    --sweep_config configs/sweeps/seed_sweep.yaml \
    --mode sequential \
    --skip_existing
