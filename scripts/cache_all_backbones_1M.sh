#!/bin/bash
# Cache full ImageNet train activations (~1.28M images) for selected backbones.
#
# Usage:
#   # Sequential (1 GPU, default):
#   bash scripts/cache_all_backbones_1M.sh
#
#   # Parallel — 5 backbones on 5 GPUs:
#   PARALLEL=1 bash scripts/cache_all_backbones_1M.sh
#
#   # Parallel with custom GPU start offset (e.g., skip GPU 0-2, start at GPU 3):
#   PARALLEL=1 GPU_OFFSET=3 bash scripts/cache_all_backbones_1M.sh
#
#   # Subset:
#   bash scripts/cache_all_backbones_1M.sh clip_vitb16 dinov2_vitb14
#
# Set env vars to override defaults:
#   BATCH_SIZE=1024 SHARD_SIZE=5000 HALF=1 PARALLEL=1 bash scripts/cache_all_backbones_1M.sh

set -e

# ── Defaults (override via env) ──────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/NAS/data/ds5725/visaebench/activations_1M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SHARD_SIZE="${SHARD_SIZE:-5000}"
LAYER="${LAYER:-11}"
HALF="${HALF:-1}"              # 1 = float16 (recommended), 0 = float32
NUM_WORKERS="${NUM_WORKERS:-4}"
PARALLEL="${PARALLEL:-0}"      # 1 = launch all backbones in parallel on separate GPUs
GPU_OFFSET="${GPU_OFFSET:-0}"  # first GPU id to use

ALL_BACKBONES="clip_vitb16 dinov2_vitb14 siglip_vitb16 mae_vitb16 deit_vitb16"

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
echo "[cache_1M] Parallel: $PARALLEL  GPU offset: $GPU_OFFSET"
echo ""

source /mnt/NAS/home/ds5725/visaebench-internal/.venv/bin/activate

PIDS=()
GPU_ID=$GPU_OFFSET

for backbone in $BACKBONES; do
    outdir="${OUTPUT_ROOT}/${backbone}/layer_${LAYER}"

    if [ -f "${outdir}/stats.json" ]; then
        echo "=== SKIP ${backbone} — already cached (stats.json exists at ${outdir}) ==="
        echo ""
        GPU_ID=$((GPU_ID + 1))
        continue
    fi

    if [ "$PARALLEL" = "1" ]; then
        LOGFILE="${OUTPUT_ROOT}/${backbone}_gpu${GPU_ID}.log"
        mkdir -p "$OUTPUT_ROOT"
        echo "=== Launching ${backbone} on GPU ${GPU_ID} → ${LOGFILE} ==="
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.caching.cache_activations \
            --backbone "$backbone" \
            --use_hf_local \
            --output_dir "$outdir" \
            --layer "$LAYER" \
            --batch_size "$BATCH_SIZE" \
            --shard_size "$SHARD_SIZE" \
            --num_workers "$NUM_WORKERS" \
            $HALF_FLAG \
            > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        echo "  PID=$!  GPU=$GPU_ID"
        GPU_ID=$((GPU_ID + 1))
    else
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
    fi
done

# ── Wait for all parallel jobs ───────────────────────────────────────────────
if [ "$PARALLEL" = "1" ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "[cache_1M] Waiting for ${#PIDS[@]} parallel jobs ..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[cache_1M] ERROR: PID $pid failed (check log files above)"
            FAILED=$((FAILED + 1))
        fi
    done
    if [ $FAILED -gt 0 ]; then
        echo "[cache_1M] $FAILED job(s) failed. Check logs in $OUTPUT_ROOT/"
        exit 1
    fi
fi

echo "[cache_1M] All requested backbones cached."
