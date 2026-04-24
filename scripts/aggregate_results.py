"""Aggregate all per-metric JSON result files into a single CSV.

Globs all JSON files in results_dir, parses backbone and SAE config from
filenames, extracts key metric values, and produces one row per SAE.

Usage:
    python scripts/aggregate_results.py \
        --results_dir /mnt/NAS/data/ds5725/visaebench/results/raw/ \
        --output_csv /mnt/NAS/data/ds5725/visaebench/results/aggregated/all_results.csv
"""

import argparse
import csv
import glob
import json
import os
import re


# Column definitions: (csv_column_name, extraction_function)
# Each extraction function takes (metric_tag, data_dict) and returns a value.
COLUMNS = [
    "backbone",
    "sae_config",
    # M1: FVU
    "fvu",
    "l0",
    "dead_pct",
    # M2: Downstream Preservation
    "accuracy_orig",
    "accuracy_recon",
    "preservation_ratio",
    # M3: Sparse Probing
    "sparse_probing_auc",
    "sparse_k32",
    "sparse_k128",
    "sparse_k512",
    # M4: Monosemanticity
    "monosemanticity_score",
    "ms_baseline",
    "ms_normalized",
    "ms_cross_model",
    # M5: Cross-Domain (EuroSAT — deprecated, saturates at ~98%)
    "eurosat_raw",
    "eurosat_sae_k128",
    "eurosat_preservation",
    "eurosat_recon_acc",
    "eurosat_preservation_recon",
    # M5: Cross-Domain (iNaturalist 2021 val — replacement metric)
    "inat_raw",
    "inat_sae_k128",
    "inat_preservation",
    "inat_recon_acc",
    "inat_preservation_recon",
    # M6: Localization
    "localization_score",
    # M7: Absorption
    "absorption_rate",
]


def parse_filename(filename: str) -> tuple[str, str, str]:
    """Extract (backbone, sae_config, metric_tag) from a result filename.

    Expected format: {backbone}_{sae_config}_{metric_tag}.json
    where metric_tag is like m1_fvu, m2_downstream, etc.

    Since backbone and sae_config both contain underscores, we identify the
    metric_tag by matching known suffixes first.
    """
    name = os.path.splitext(filename)[0]

    known_suffixes = [
        "m1_fvu",
        "m2_downstream",
        "m3_sparse_probing",
        "m4_monosemanticity",
        "m5_cross_domain_inat",  # iNat-only M5 run (listed before the shorter
        "m5_cross_domain",       # prefix so endswith() matches the longer form first)
        "m6_localization",
        "m7_absorption",
    ]

    for suffix in known_suffixes:
        if name.endswith(f"_{suffix}"):
            prefix = name[: -(len(suffix) + 1)]  # strip _suffix
            # Now split prefix into backbone and sae_config.
            # Try to match known backbone names first.
            for bb in [
                "clip_vitb16", "dinov2_vitb14", "siglip_vitb16",
                "mae_vitb16", "deit_vitb16",
            ]:
                if prefix.startswith(bb + "_"):
                    sae_config = prefix[len(bb) + 1 :]
                    return bb, sae_config, suffix
                elif prefix == bb:
                    # No sae_config in filename — read from JSON instead
                    return bb, "", suffix

            # Fallback: use the JSON's embedded backbone/sae_config fields
            return prefix, "", suffix

    return name, "", "unknown"


def extract_m1(data: dict) -> dict:
    """Extract M1 FVU fields."""
    return {
        "fvu": data.get("fvu"),
        "l0": data.get("l0"),
        "dead_pct": data.get("dead_pct"),
    }


def extract_m2(data: dict) -> dict:
    """Extract M2 downstream preservation fields."""
    return {
        "accuracy_orig": data.get("accuracy_original"),
        "accuracy_recon": data.get("accuracy_reconstructed"),
        "preservation_ratio": data.get("preservation_ratio"),
    }


def extract_m3(data: dict) -> dict:
    """Extract M3 sparse probing fields."""
    # Results may be nested under "sparse_probing" key (from run_sparse_probing.py)
    # or flat (from run_all_metrics.py)
    sp = data.get("sparse_probing", data)
    return {
        "sparse_probing_auc": sp.get("auc"),
        "sparse_k32": sp.get("k_32_accuracy"),
        "sparse_k128": sp.get("k_128_accuracy"),
        "sparse_k512": sp.get("k_512_accuracy"),
    }


def extract_m4(data: dict) -> dict:
    """Extract M4 monosemanticity fields."""
    return {
        "monosemanticity_score": data.get("monosemanticity_score"),
        "ms_baseline": data.get("ms_baseline"),
        "ms_normalized": data.get("ms_normalized"),
        "ms_cross_model": data.get("cross_model"),
    }


def extract_m5(data: dict) -> dict:
    """Extract M5 cross-domain fields for both EuroSAT and iNaturalist.

    A single result JSON may contain either or both dataset blocks depending
    on which loaders were passed to ``--m5_datasets``. Missing blocks leave
    their columns as None, so two separate M5 runs (one per dataset) merge
    cleanly into one row.
    """
    out: dict = {}
    eurosat = data.get("eurosat", {})
    if eurosat:
        out["eurosat_raw"] = eurosat.get("raw_accuracy")
        out["eurosat_sae_k128"] = eurosat.get("sae_k128_accuracy")
        out["eurosat_preservation"] = eurosat.get("preservation_k128")
        # New reconstruction-preservation columns (NaN for legacy EuroSAT
        # result JSONs that predate the recon-probe addition).
        out["eurosat_recon_acc"] = eurosat.get("recon_accuracy")
        out["eurosat_preservation_recon"] = eurosat.get("preservation_recon")
    inat = data.get("inaturalist", {})
    if inat:
        out["inat_raw"] = inat.get("raw_accuracy")
        out["inat_sae_k128"] = inat.get("sae_k128_accuracy")
        out["inat_preservation"] = inat.get("preservation_k128")
        out["inat_recon_acc"] = inat.get("recon_accuracy")
        out["inat_preservation_recon"] = inat.get("preservation_recon")
    return out


def extract_m6(data: dict) -> dict:
    """Extract M6 localization fields."""
    results = data.get("results", {})
    return {
        "localization_score": results.get("mean_morans_i"),
    }


def extract_m7(data: dict) -> dict:
    """Extract M7 absorption fields."""
    return {
        "absorption_rate": data.get("absorption_rate"),
    }


EXTRACTORS = {
    "m1_fvu": extract_m1,
    "m2_downstream": extract_m2,
    "m3_sparse_probing": extract_m3,
    "m4_monosemanticity": extract_m4,
    "m5_cross_domain": extract_m5,
    "m5_cross_domain_inat": extract_m5,  # same extractor; reads "inaturalist" key
    "m6_localization": extract_m6,
    "m7_absorption": extract_m7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate metric JSONs into CSV.")
    p.add_argument("--results_dir", type=str, required=True,
                   help="Directory containing per-metric JSON files.")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Path to output CSV file.")
    p.add_argument("--baseline_json", type=str, default=None,
                   help="Path to MS baselines JSON (from compute_ms_baseline.py). "
                        "Used for post-hoc normalization when ms_baseline is missing "
                        "from result JSONs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.results_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {args.results_dir}")

    # Accumulate data per (backbone, sae_config) key
    rows: dict[tuple[str, str], dict] = {}

    for jf in json_files:
        filename = os.path.basename(jf)
        backbone, sae_config, metric_tag = parse_filename(filename)

        with open(jf) as f:
            data = json.load(f)

        # Use JSON-embedded fields if available (more reliable than filename parsing)
        backbone = data.get("backbone", backbone) or backbone
        sae_config = data.get("sae_config", sae_config) or sae_config

        # Skip error results
        if "error" in data and len(data) <= 4:  # metric, backbone, sae_config, error
            print(f"  Skip {filename} (error: {data['error'][:80]})")
            continue

        key = (backbone, sae_config)
        if key not in rows:
            rows[key] = {"backbone": backbone, "sae_config": sae_config}

        # Extract metric values
        extractor = EXTRACTORS.get(metric_tag)
        if extractor:
            values = extractor(data)
            rows[key].update({k: v for k, v in values.items() if v is not None})
        else:
            print(f"  Skip {filename} (unknown metric tag: {metric_tag})")

    if not rows:
        print("No valid results found.")
        return

    # Post-hoc MS baseline normalization (for results that lack ms_baseline)
    if args.baseline_json:
        with open(args.baseline_json) as f:
            baselines = json.load(f)
        bl_lookup = {sm: info["baseline_ms"]
                     for sm, info in baselines["baselines"].items()}
        # Cross-model map: backbone → scoring model
        from src.evaluation.concept_detection.monosemanticity import CROSS_MODEL_MAP
        patched = 0
        for key, row in rows.items():
            if "ms_baseline" not in row and "monosemanticity_score" in row:
                backbone = row["backbone"]
                scoring_model = CROSS_MODEL_MAP.get(backbone)
                if scoring_model and scoring_model in bl_lookup:
                    bl = bl_lookup[scoring_model]
                    row["ms_baseline"] = bl
                    row["ms_normalized"] = row["monosemanticity_score"] - bl
                    row["ms_cross_model"] = scoring_model
                    patched += 1
        if patched:
            print(f"Applied post-hoc MS baseline normalization to {patched} rows")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])

    print(f"\nWrote {len(rows)} rows to {args.output_csv}")

    # Print summary table
    print()
    has_norm = any("ms_normalized" in r for r in rows.values())
    if has_norm:
        header = f"{'backbone':20s} {'config':22s} {'FVU':>8s} {'L0':>6s} {'Pres':>6s} {'AUC':>6s} {'MS':>6s} {'MS_n':>6s} {'Loc':>6s} {'Abs':>6s}"
    else:
        header = f"{'backbone':20s} {'config':22s} {'FVU':>8s} {'L0':>6s} {'Pres':>6s} {'AUC':>6s} {'MS':>6s} {'Loc':>6s} {'Abs':>6s}"
    print(header)
    print("-" * len(header))
    for key in sorted(rows.keys()):
        r = rows[key]
        fvu = f"{r['fvu']:.4f}" if "fvu" in r else "—"
        l0 = f"{r['l0']:.1f}" if "l0" in r else "—"
        pres = f"{r['preservation_ratio']:.3f}" if "preservation_ratio" in r else "—"
        auc = f"{r['sparse_probing_auc']:.3f}" if "sparse_probing_auc" in r else "—"
        ms = f"{r['monosemanticity_score']:.3f}" if "monosemanticity_score" in r else "—"
        ms_n = f"{r['ms_normalized']:.3f}" if "ms_normalized" in r else "—"
        loc = f"{r['localization_score']:.3f}" if "localization_score" in r else "—"
        abs_rate = f"{r['absorption_rate']:.3f}" if "absorption_rate" in r else "—"
        if has_norm:
            print(f"{r['backbone']:20s} {r['sae_config']:22s} {fvu:>8s} {l0:>6s} {pres:>6s} {auc:>6s} {ms:>6s} {ms_n:>6s} {loc:>6s} {abs_rate:>6s}")
        else:
            print(f"{r['backbone']:20s} {r['sae_config']:22s} {fvu:>8s} {l0:>6s} {pres:>6s} {auc:>6s} {ms:>6s} {loc:>6s} {abs_rate:>6s}")


if __name__ == "__main__":
    main()
