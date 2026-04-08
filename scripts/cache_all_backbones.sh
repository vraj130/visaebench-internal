#!/bin/bash
# Cache val activations for all 5 backbones (sequential, full 50K val split).
# DINOv2 is already done — skip if shard_000.pt exists.

set -e

BACKBONES="clip_vitb16 siglip_vitb16 mae_vitb16 deit_vitb16"
OUTPUT_ROOT="/mnt/NAS/data/ds5725/visaebench/activations_val"

for backbone in $BACKBONES; do
    outdir="${OUTPUT_ROOT}/${backbone}/layer_11"
    if [ -f "${outdir}/shard_000.pt" ]; then
        echo "=== SKIP ${backbone} — already cached at ${outdir} ==="
        continue
    fi
    echo "=== Caching ${backbone} val activations ==="
    python -m src.caching.cache_activations --backbone $backbone --use_hf_local --hf_split validation --output_dir $outdir --batch_size 512
    echo "=== Done ${backbone} ==="
    echo
done

echo "All backbones cached."
