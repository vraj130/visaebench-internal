#!/bin/bash
# Cache full ImageNet train activations (~1.28M images) for selected backbones.
#
# Usage:
#   bash scripts/cache_all_backbones_1M.sh                          # all 5 backbones
#   bash scripts/cache_all_backbones_1M.sh dinov2_vitb14             # single backbone
#   bash scripts/cache_all_backbones_1M.sh clip_vitb16 dinov2_vitb14 # subset
#
# Set env vars to override defaults:
#   BATCH_SIZE=256 SHARD_SIZE=5000 HALF=1 bash scripts/cache_all_backbones_1M.sh

set -e

# ── Defaults (override via env) ──────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/NAS/data/ds5725/visaebench/activations_1M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SHARD_SIZE="${SHARD_SIZE:-5000}"
LAYER="${LAYER:-11}"
HALF="${HALF:-1}"          # 1 = float16 (recommended), 0 = float32
NUM_WORKERS="${NUM_WORKERS:-4}"

ALL_BACKBONES="clip_vitb16"

# Use CLI args if provided, otherwise all 5
if [ $# -gt 0 ]; then
    BACKBONES="$@"
else
    BACKBONES="$ALL_BACKBONES"
fi

HALF_FLAG=""
if [ "$HALF" = "1" ]; then
    HALF_FLAG="--half"
    echo "[cache_1M] Saving as float16 (set HALF=0 for float32)"
fi

echo "[cache_1M] Backbones: $BACKBONES"
echo "[cache_1M] Output root: $OUTPUT_ROOT"
echo "[cache_1M] Batch size: $BATCH_SIZE  Shard size: $SHARD_SIZE  Layer: $LAYER"
echo ""

source /mnt/NAS/home/ds5725/visaebench-internal/.venv/bin/activate

for backbone in $BACKBONES; do
    outdir="${OUTPUT_ROOT}/${backbone}/layer_${LAYER}"

    if [ -f "${outdir}/stats.json" ]; then
        echo "=== SKIP ${backbone} — already cached (stats.json exists at ${outdir}) ==="
        echo ""
        continue
    fi

    echo "=== Caching ${backbone} (full train split) ==="
    python3 -m src.caching.cache_activations \
        --backbone "$backbone" \
        --use_hf_local \
        --output_dir "$outdir" \
        --layer "$LAYER" \
        --batch_size "$BATCH_SIZE" \
        --shard_size "$SHARD_SIZE" \
        --num_workers "$NUM_WORKERS" \
        $HALF_FLAG

    echo "=== Done ${backbone} ==="
    echo ""
done

echo "[cache_1M] All requested backbones cached."
