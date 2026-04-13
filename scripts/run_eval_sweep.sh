#!/bin/bash
# Run all 7 metrics on the default SAE (batchtopk_16x_k192) for all 5 backbones.
#
# Usage:
#   bash scripts/run_eval_sweep.sh
#
# Each backbone is evaluated sequentially. If a checkpoint is missing, the
# backbone is skipped. Results are saved as individual JSON files under
# RESULTS_ROOT, one per (backbone, metric) pair.
set -e

CHECKPOINT_ROOT="/mnt/NAS/data/ds5725/visaebench/checkpoints"
VAL_ROOT="/mnt/NAS/data/ds5725/visaebench/activations_val"
RESULTS_ROOT="/mnt/NAS/data/ds5725/visaebench/results/raw"

# Map backbone -> checkpoint subdir (all use default settings)
declare -A SAE_CONFIGS
SAE_CONFIGS[clip_vitb16]="batchtopk_16x_k192"
SAE_CONFIGS[dinov2_vitb14]="batchtopk_16x_k192"
SAE_CONFIGS[siglip_vitb16]="batchtopk_16x_k192"
SAE_CONFIGS[mae_vitb16]="batchtopk_16x_k192"
SAE_CONFIGS[deit_vitb16]="batchtopk_16x_k192"

mkdir -p "${RESULTS_ROOT}"

for backbone in clip_vitb16 dinov2_vitb14 siglip_vitb16 mae_vitb16 deit_vitb16; do
    config="${SAE_CONFIGS[$backbone]}"
    checkpoint_dir="${CHECKPOINT_ROOT}/${backbone}/${config}"
    val_dir="${VAL_ROOT}/${backbone}/layer_11"
    output_dir="${RESULTS_ROOT}"

    if [ ! -f "${checkpoint_dir}/sae.pt" ]; then
        echo "=== SKIP ${backbone} — no checkpoint at ${checkpoint_dir}/sae.pt ==="
        continue
    fi

    if [ ! -d "${val_dir}" ]; then
        echo "=== SKIP ${backbone} — no val activations at ${val_dir} ==="
        continue
    fi

    echo "=========================================="
    echo "Evaluating ${backbone} / ${config}"
    echo "=========================================="

    python scripts/run_all_metrics.py \
        --sae_checkpoint "${checkpoint_dir}/sae.pt" \
        --sae_config "${checkpoint_dir}/config.yaml" \
        --activation_dir "${val_dir}" \
        --backbone_name "${backbone}" \
        --output_dir "${output_dir}" \
        --device cuda \
        --m5_datasets eurosat

    echo "=== Done ${backbone} ==="
    echo
done

echo "All evaluations complete."
echo "Results in: ${RESULTS_ROOT}"
