#!/bin/bash
# Train BatchTopKSAE (16x, k=192) on the 4 remaining backbones.
# DINOv2 is already trained — skip if sae.pt exists.
# Same hyperparameters as the DINOv2 pilot.

set -e

BACKBONES="clip_vitb16 dinov2_vitb14 siglip_vitb16 mae_vitb16 deit_vitb16"
ACT_ROOT="/mnt/NAS/data/ds5725/visaebench/activations"
CKPT_ROOT="/mnt/NAS/data/ds5725/visaebench/checkpoints"

for backbone in $BACKBONES; do
    outdir="${CKPT_ROOT}/${backbone}/batchtopk_16x_k192"
    if [ -f "${outdir}/sae.pt" ]; then
        echo "=== SKIP ${backbone} — already trained at ${outdir} ==="
        continue
    fi
    echo "========================================"
    echo "Training ${backbone} BatchTopK 16x k192"
    echo "========================================"
    python -m src.training.train_sae \
        --backbone $backbone \
        --activation_dir ${ACT_ROOT}/${backbone}/layer_11/ \
        --output_dir $outdir \
        --expansion_factor 16 \
        --k 192 \
        --architecture batchtopk \
        --lr 1e-3 \
        --batch_size 4096 \
        --num_epochs 4 \
        --seed 42
    echo "=== Done ${backbone} ==="
    echo
done

echo "All training complete."
