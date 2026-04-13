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
    # M5: Cross-Domain (EuroSAT)
    "eurosat_raw",
    "eurosat_sae_k128",
    "eurosat_preservation",
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
        "m5_cross_domain",
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
    }


def extract_m5(data: dict) -> dict:
    """Extract M5 cross-domain fields (EuroSAT)."""
    eurosat = data.get("eurosat", {})
    return {
        "eurosat_raw": eurosat.get("raw_accuracy"),
        "eurosat_sae_k128": eurosat.get("sae_k128_accuracy"),
        "eurosat_preservation": eurosat.get("preservation_k128"),
    }


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
    "m6_localization": extract_m6,
    "m7_absorption": extract_m7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate metric JSONs into CSV.")
    p.add_argument("--results_dir", type=str, required=True,
                   help="Directory containing per-metric JSON files.")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Path to output CSV file.")
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
        loc = f"{r['localization_score']:.3f}" if "localization_score" in r else "—"
        abs_rate = f"{r['absorption_rate']:.3f}" if "absorption_rate" in r else "—"
        print(f"{r['backbone']:20s} {r['sae_config']:22s} {fvu:>8s} {l0:>6s} {pres:>6s} {auc:>6s} {ms:>6s} {loc:>6s} {abs_rate:>6s}")


if __name__ == "__main__":
    main()
