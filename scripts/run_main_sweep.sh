#!/bin/bash
# Run the main sweep: 5 backbones x 3 expansions x 4 K values = 60 TopK SAE runs.
# Uses sweep_runner.py in sequential mode with --skip_existing.

set -e
source /mnt/NAS/home/ds5725/visaebench-internal/.visaebench/bin/activate
cd /mnt/NAS/home/ds5725/visaebench-internal

python3 -m src.training.sweep_runner \
    --sweep_config configs/sweeps/main_sweep.yaml \
    --mode sequential \
    --skip_existing
