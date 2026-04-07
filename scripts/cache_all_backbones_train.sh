#!/bin/bash
# Cache train activations for all 5 backbones (100K balanced, train split).
# Skips any backbone that already has shard_000.pt.

set -e

BACKBONES="clip_vitb16 dinov2_vitb14 siglip_vitb16 mae_vitb16 deit_vitb16"
OUTPUT_ROOT="/mnt/NAS/data/ds5725/visaebench/activations"

for backbone in $BACKBONES; do
    outdir="${OUTPUT_ROOT}/${backbone}/layer_11"
    if [ -f "${outdir}/shard_000.pt" ]; then
        echo "=== SKIP ${backbone} — already cached at ${outdir} ==="
        continue
    fi
    echo "=== Caching ${backbone} train activations (100K balanced) ==="
    python -m src.caching.cache_activations --backbone $backbone --use_hf_local --hf_split train --balanced --num_images 100000 --output_dir $outdir --batch_size 512
    echo "=== Done ${backbone} ==="
    echo
done

echo "All backbones cached."
