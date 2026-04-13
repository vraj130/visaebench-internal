#!/bin/bash
# =============================================================================
# test_all_metrics.sh — Run all 7 metrics (M1–M7) across all 5 trained SAEs.
#
# Steps:
#   1. Cache val activations for any backbone missing them
#   2. Run M1 (FVU)              — uses val shards
#   3. Run M2 (Downstream)       — uses val shards + labels
#   4. Run M3 (Sparse Probing)   — uses val shards + labels
#   5. Run M4 (Monosemanticity)  — uses val shards
#   6. Run M5 (Cross-Domain)     — extracts OOD activations live
#   7. Run M6 (Spatial Coherence)— uses val shards
#   8. Run M7 (Absorption)       — uses val shards + labels
#   9. Print comparison summary
#
# Usage:
#   bash scripts/test_all_metrics.sh                    # run everything
#   bash scripts/test_all_metrics.sh --skip-caching     # skip val caching
#   bash scripts/test_all_metrics.sh --backbone dinov2_vitb14  # single backbone
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .visaebench/bin/activate

CKPT_ROOT="/mnt/NAS/data/ds5725/visaebench/checkpoints"
ACT_ROOT="/mnt/NAS/data/ds5725/visaebench/activations"
RESULTS_DIR="results/raw"
SAE_DIR="batchtopk_16x_k192"
DEVICE="${DEVICE:-cuda}"
VAL_IMAGES=10000          # images to cache for val (2 shards)
VAL_BATCH_SIZE=512

ALL_BACKBONES=(dinov2_vitb14 clip_vitb16 siglip_vitb16 mae_vitb16 deit_vitb16)

SKIP_CACHING=false
SINGLE_BACKBONE=""

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-caching) SKIP_CACHING=true; shift ;;
        --backbone)     SINGLE_BACKBONE="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -n "$SINGLE_BACKBONE" ]]; then
    BACKBONES=("$SINGLE_BACKBONE")
else
    BACKBONES=("${ALL_BACKBONES[@]}")
fi

mkdir -p "$RESULTS_DIR"

# Track pass/fail per (backbone, metric)
declare -A STATUS
declare -A TIMING

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo -e "\n\033[1;36m>>> $*\033[0m"; }
ok()   { echo -e "\033[1;32m  [PASS] $*\033[0m"; }
fail() { echo -e "\033[1;31m  [FAIL] $*\033[0m"; }

run_metric() {
    local backbone="$1" metric_name="$2"
    shift 2
    local key="${backbone}::${metric_name}"

    local start_time=$SECONDS
    if "$@" ; then
        STATUS[$key]="PASS"
    else
        STATUS[$key]="FAIL"
    fi
    TIMING[$key]=$(( SECONDS - start_time ))

    if [[ "${STATUS[$key]}" == "PASS" ]]; then
        ok "$backbone / $metric_name  (${TIMING[$key]}s)"
    else
        fail "$backbone / $metric_name  (${TIMING[$key]}s)"
    fi
}

# Common paths per backbone
ckpt()    { echo "$CKPT_ROOT/$1/$SAE_DIR/sae.pt"; }
cfg()     { echo "$CKPT_ROOT/$1/$SAE_DIR/config.yaml"; }
val_dir() { echo "$ACT_ROOT/$1/layer_11_val"; }
out()     { echo "$RESULTS_DIR/${1}_${SAE_DIR}_${2}.json"; }

# =========================================================================
# Step 0: Cache val activations if needed
# =========================================================================
if [[ "$SKIP_CACHING" == false ]]; then
    log "Step 0: Checking / caching val activations"
    for bb in "${BACKBONES[@]}"; do
        vdir="$(val_dir "$bb")"
        if [[ -d "$vdir" ]] && ls "$vdir"/shard_*.pt &>/dev/null; then
            echo "  $bb — val shards already exist at $vdir, skipping"
        else
            log "Caching val activations for $bb ($VAL_IMAGES images) ..."
            python -m src.caching.cache_activations \
                --backbone "$bb" \
                --use_hf_local \
                --hf_split validation \
                --balanced \
                --num_images "$VAL_IMAGES" \
                --output_dir "$vdir" \
                --batch_size "$VAL_BATCH_SIZE"
        fi
    done
else
    log "Step 0: Skipping val caching (--skip-caching)"
fi

# =========================================================================
# Step 1–7: Run metrics per backbone
# =========================================================================
for bb in "${BACKBONES[@]}"; do
    SAE_CKPT="$(ckpt "$bb")"
    SAE_CFG="$(cfg "$bb")"
    VAL_DIR="$(val_dir "$bb")"

    # Sanity check
    if [[ ! -f "$SAE_CKPT" ]]; then
        echo "  SKIP $bb — checkpoint not found: $SAE_CKPT"
        continue
    fi
    if [[ ! -f "$SAE_CFG" ]]; then
        echo "  SKIP $bb — config not found: $SAE_CFG"
        continue
    fi
    if [[ ! -d "$VAL_DIR" ]]; then
        echo "  SKIP $bb — val dir missing: $VAL_DIR"
        continue
    fi

    log "===== $bb ====="

    # --- M1: FVU ---
    log "M1 FVU — $bb"
    run_metric "$bb" "M1_FVU" \
        python scripts/run_fvu.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --output_path "$(out "$bb" fvu)" \
            --device "$DEVICE"

    # --- M2: Downstream Preservation ---
    log "M2 Downstream — $bb"
    run_metric "$bb" "M2_Downstream" \
        python scripts/run_downstream_preservation.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --output_path "$(out "$bb" downstream)" \
            --device "$DEVICE"

    # --- M3: Sparse Probing ---
    log "M3 Sparse Probing — $bb"
    run_metric "$bb" "M3_SparseProbing" \
        python scripts/run_sparse_probing.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --output_path "$(out "$bb" sparse_probing)" \
            --device "$DEVICE"

    # --- M4: Monosemanticity ---
    log "M4 Monosemanticity — $bb"
    run_metric "$bb" "M4_Monosemanticity" \
        python scripts/run_monosemanticity.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --backbone_name "$bb" \
            --output_path "$(out "$bb" monosemanticity)" \
            --device "$DEVICE"

    # --- M5: Cross-Domain ---
    log "M5 Cross-Domain — $bb"
    run_metric "$bb" "M5_CrossDomain" \
        python scripts/run_cross_domain.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --backbone_name "$bb" \
            --datasets eurosat \
            --output_path "$(out "$bb" cross_domain)" \
            --device "$DEVICE"

    # --- M6: Spatial Coherence ---
    log "M6 Spatial Coherence — $bb"
    run_metric "$bb" "M6_SpatialCoherence" \
        python scripts/run_spatial_coherence.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --output_path "$(out "$bb" spatial_coherence)" \
            --device "$DEVICE"

    # --- M7: Absorption ---
    log "M7 Absorption — $bb"
    run_metric "$bb" "M7_Absorption" \
        python scripts/run_absorption.py \
            --sae_checkpoint "$SAE_CKPT" \
            --sae_config "$SAE_CFG" \
            --activation_dir "$VAL_DIR" \
            --output_path "$(out "$bb" absorption)" \
            --device "$DEVICE"

done

# =========================================================================
# Step 8: Pass/Fail summary
# =========================================================================
log "Pass/Fail Summary"
echo ""
printf "%-18s %-10s %-14s %-16s %-16s %-14s %-18s %-12s\n" \
    "Backbone" "M1_FVU" "M2_Downstream" "M3_SparseProbe" "M4_Monosem" "M5_CrossDom" "M6_SpatialCoher" "M7_Absorb"
printf "%-18s %-10s %-14s %-16s %-16s %-14s %-18s %-12s\n" \
    "--------" "------" "-------------" "--------------" "----------" "-----------" "---------------" "---------"

for bb in "${BACKBONES[@]}"; do
    m1="${STATUS[${bb}::M1_FVU]:-SKIP}"
    m2="${STATUS[${bb}::M2_Downstream]:-SKIP}"
    m3="${STATUS[${bb}::M3_SparseProbing]:-SKIP}"
    m4="${STATUS[${bb}::M4_Monosemanticity]:-SKIP}"
    m5="${STATUS[${bb}::M5_CrossDomain]:-SKIP}"
    m6="${STATUS[${bb}::M6_SpatialCoherence]:-SKIP}"
    m7="${STATUS[${bb}::M7_Absorption]:-SKIP}"
    printf "%-18s %-10s %-14s %-16s %-16s %-14s %-18s %-12s\n" \
        "$bb" "$m1" "$m2" "$m3" "$m4" "$m5" "$m6" "$m7"
done

# =========================================================================
# Step 9: Metric comparison table (parse JSONs)
# =========================================================================
log "Metric Comparison (from result JSONs)"

python3 - "$RESULTS_DIR" "$SAE_DIR" "${BACKBONES[@]}" <<'PYEOF'
import json, sys, os

results_dir = sys.argv[1]
sae_dir = sys.argv[2]
backbones = sys.argv[3:]

def load(bb, metric_suffix):
    path = os.path.join(results_dir, f"{bb}_{sae_dir}_{metric_suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt(val, precision=4):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)

# Header
metrics = ["M1:FVU", "M1:L0", "M1:Dead%", "M2:Preserv", "M3:AUC",
           "M4:MS", "M5:EuroSAT", "M6:MoranI", "M7:AbsRate"]

header = f"{'Backbone':<18s}" + "".join(f"{m:>12s}" for m in metrics)
sep = "-" * len(header)
print()
print(header)
print(sep)

for bb in backbones:
    vals = {}

    # M1
    d = load(bb, "fvu")
    if d:
        vals["M1:FVU"]   = fmt(d.get("fvu"), 4)
        vals["M1:L0"]    = fmt(d.get("l0"), 1)
        vals["M1:Dead%"] = fmt(d.get("dead_pct"), 1)
    else:
        vals["M1:FVU"] = vals["M1:L0"] = vals["M1:Dead%"] = "—"

    # M2
    d = load(bb, "downstream")
    if d:
        vals["M2:Preserv"] = fmt(d.get("preservation_ratio"), 4)
    else:
        vals["M2:Preserv"] = "—"

    # M3
    d = load(bb, "sparse_probing")
    if d:
        sp = d.get("sparse_probing", d)
        vals["M3:AUC"] = fmt(sp.get("auc"), 4)
    else:
        vals["M3:AUC"] = "—"

    # M4
    d = load(bb, "monosemanticity")
    if d:
        vals["M4:MS"] = fmt(d.get("monosemanticity_score"), 4)
    else:
        vals["M4:MS"] = "—"

    # M5
    d = load(bb, "cross_domain")
    if d:
        euro = d.get("eurosat", {})
        # Get best SAE probing accuracy across k values
        sae_accs = [v for k, v in euro.items() if k.startswith("sae_k") and isinstance(v, (int, float))]
        if sae_accs:
            vals["M5:EuroSAT"] = fmt(max(sae_accs), 4)
        elif "raw_accuracy" in euro:
            vals["M5:EuroSAT"] = fmt(euro["raw_accuracy"], 4)
        else:
            vals["M5:EuroSAT"] = "—"
    else:
        vals["M5:EuroSAT"] = "—"

    # M6
    d = load(bb, "spatial_coherence")
    if d:
        r = d.get("results", d)
        vals["M6:MoranI"] = fmt(r.get("mean_morans_i"), 4)
    else:
        vals["M6:MoranI"] = "—"

    # M7
    d = load(bb, "absorption")
    if d:
        vals["M7:AbsRate"] = fmt(d.get("absorption_rate"), 4)
    else:
        vals["M7:AbsRate"] = "—"

    row = f"{bb:<18s}" + "".join(f"{vals[m]:>12s}" for m in metrics)
    print(row)

print(sep)
print()
PYEOF

# Count results
TOTAL=0
PASSED=0
FAILED=0
for key in "${!STATUS[@]}"; do
    TOTAL=$((TOTAL + 1))
    if [[ "${STATUS[$key]}" == "PASS" ]]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
log "Done!  $PASSED/$TOTAL passed, $FAILED failed."
echo "Results saved to $RESULTS_DIR/"
