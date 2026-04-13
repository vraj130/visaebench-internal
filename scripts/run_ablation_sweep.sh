#!/bin/bash
# Run the ablation sweep: TopK vs BatchTopK on CLIP + DINOv2 at 16x expansion.
# 2 backbones x 2 architectures x 3 K values = 12 runs.

set -e
source /mnt/NAS/home/ds5725/visaebench-internal/.visaebench/bin/activate
cd /mnt/NAS/home/ds5725/visaebench-internal

python3 -m src.training.sweep_runner \
    --sweep_config configs/sweeps/ablation_sweep.yaml \
    --mode sequential \
    --skip_existing
