"""Launch full evaluation sweep across multiple GPUs on a single node.

Reads the same sweep YAML config used for training, discovers all trained
checkpoints, and runs all 7 metrics (M1-M7) on each SAE — distributing
jobs across N GPUs.  When one eval finishes, the next queued job takes
its GPU slot.

Usage:
    # Dry run — see all jobs
    python scripts/launch_eval_sweep.py --dry_run

    # Run on 8 H100s
    python scripts/launch_eval_sweep.py --num_gpus 8

    # Only evaluate specific metrics
    python scripts/launch_eval_sweep.py --num_gpus 8 --metrics m1,m2,m3

    # Skip SAEs that already have all result JSONs
    python scripts/launch_eval_sweep.py --num_gpus 8 --skip_existing
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULTS = {
    "activation_root": "/mnt/NAS/data/ds5725/visaebench/activations",
    "val_root": "/mnt/NAS/data/ds5725/visaebench/activations_val",
    "checkpoint_root": "/mnt/NAS/data/ds5725/visaebench/checkpoints",
    "results_root": "/mnt/NAS/data/ds5725/visaebench/results/raw",
    "sweep_config": "configs/sweeps/main_sweep.yaml",
}

ALL_METRICS = "m1,m2,m3,m4,m5,m6,m7"

# Metric suffixes used in result filenames
METRIC_FILE_SUFFIXES = [
    "m1_fvu", "m2_downstream", "m3_sparse_probing", "m4_monosemanticity",
    "m5_cross_domain", "m6_localization", "m7_absorption",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch evaluation sweep across multiple GPUs on a single node.",
    )
    p.add_argument("--sweep_config", type=str, default=DEFAULTS["sweep_config"],
                   help="Path to sweep YAML config (same one used for training).")
    p.add_argument("--num_gpus", type=int, default=8,
                   help="Number of GPUs to use (default: 8).")
    p.add_argument("--val_root", type=str, default=DEFAULTS["val_root"])
    p.add_argument("--checkpoint_root", type=str, default=DEFAULTS["checkpoint_root"])
    p.add_argument("--results_root", type=str, default=DEFAULTS["results_root"])
    p.add_argument("--metrics", type=str, default=ALL_METRICS,
                   help=f"Comma-separated metrics to run (default: {ALL_METRICS}).")
    p.add_argument("--m5_datasets", type=str, default="eurosat",
                   help="Comma-separated OOD datasets for M5 (default: eurosat).")
    p.add_argument("--max_features_m4", type=int, default=2048,
                   help="Max features to score for M4 (default: 2048).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip SAEs where all requested metric JSONs already exist.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print job list without executing.")
    return p.parse_args()


def build_eval_jobs(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Generate one eval job per trained SAE checkpoint."""
    backbones = cfg["backbones"]
    expansion_factors = cfg["expansion_factors"]
    k_values = cfg["k_values"]

    if "architectures" in cfg:
        architectures = cfg["architectures"]
    elif "architecture" in cfg:
        architectures = [cfg["architecture"]]
    else:
        architectures = ["batchtopk"]

    seeds = cfg.get("seeds", None)
    metrics_list = [m.strip() for m in args.metrics.split(",")]

    jobs = []
    for backbone, arch, exp, k in itertools.product(
        backbones, architectures, expansion_factors, k_values,
    ):
        seed_list = seeds if seeds else [None]
        for seed in seed_list:
            if seed is not None:
                config_name = f"{arch}_{exp}x_k{k}_seed{seed}"
            else:
                config_name = f"{arch}_{exp}x_k{k}"

            checkpoint_dir = os.path.join(args.checkpoint_root, backbone, config_name)
            sae_checkpoint = os.path.join(checkpoint_dir, "sae.pt")
            sae_config = os.path.join(checkpoint_dir, "config.yaml")
            val_dir = os.path.join(args.val_root, backbone, "layer_11")

            # Skip if no trained checkpoint
            if not os.path.isfile(sae_checkpoint):
                continue

            jobs.append({
                "backbone": backbone,
                "config_name": config_name,
                "sae_checkpoint": sae_checkpoint,
                "sae_config": sae_config,
                "val_dir": val_dir,
                "label": f"{backbone}/{config_name}",
                "metrics": args.metrics,
            })

    return jobs


def is_eval_complete(job: dict, results_root: str, metrics: str) -> bool:
    """Check if all requested metric result JSONs already exist for this SAE."""
    requested = set(m.strip() for m in metrics.split(","))
    # Map metric name to file suffix
    metric_to_suffix = {
        "m1": "m1_fvu", "m2": "m2_downstream", "m3": "m3_sparse_probing",
        "m4": "m4_monosemanticity", "m5": "m5_cross_domain",
        "m6": "m6_localization", "m7": "m7_absorption",
    }
    for m in requested:
        suffix = metric_to_suffix.get(m)
        if suffix:
            filename = f"{job['backbone']}_{job['config_name']}_{suffix}.json"
            if not os.path.isfile(os.path.join(results_root, filename)):
                return False
    return True


def build_eval_command(job: dict, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "scripts/run_all_metrics.py",
        "--sae_checkpoint", job["sae_checkpoint"],
        "--sae_config", job["sae_config"],
        "--activation_dir", job["val_dir"],
        "--backbone_name", job["backbone"],
        "--output_dir", args.results_root,
        "--device", "cuda",
        "--metrics", job["metrics"],
        "--max_features_m4", str(args.max_features_m4),
        "--m5_datasets", args.m5_datasets,
    ]


def format_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main() -> None:
    args = parse_args()

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)

    print(f"Evaluation Sweep: {cfg.get('sweep_name', '(unnamed)')}")
    print(f"  Metrics: {args.metrics}")
    print(f"  GPUs: {args.num_gpus}")
    print()

    all_jobs = build_eval_jobs(cfg, args)

    # Filter
    if args.skip_existing:
        skipped = [j for j in all_jobs if is_eval_complete(j, args.results_root, args.metrics)]
        to_run = [j for j in all_jobs if not is_eval_complete(j, args.results_root, args.metrics)]
    else:
        skipped = []
        to_run = all_jobs

    # Report missing checkpoints
    expected = len(cfg["backbones"]) * len(cfg["expansion_factors"]) * len(cfg["k_values"])
    if "architectures" in cfg:
        expected *= len(cfg["architectures"])
    missing = expected - len(all_jobs)

    print(f"Expected checkpoints: {expected}")
    print(f"  Found (trained):     {len(all_jobs)}")
    if missing > 0:
        print(f"  Missing (not trained): {missing}")
    print(f"  Skip (already evaluated): {len(skipped)}")
    print(f"  To evaluate: {len(to_run)}")
    print()

    if not to_run:
        print("Nothing to run.")
        return

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        for i, job in enumerate(to_run):
            gpu = i % args.num_gpus
            print(f"  [{i+1:>3}/{len(to_run)}] GPU {gpu}  {job['label']}")
        num_rounds = len(to_run) // args.num_gpus + (1 if len(to_run) % args.num_gpus else 0)
        print(f"\n{len(to_run)} eval jobs across {args.num_gpus} GPUs ({num_rounds} rounds)")
        return

    # ── Parallel execution ───────────────────────────────────────────────
    num_gpus = args.num_gpus
    total = len(to_run)
    start_time = time.time()

    os.makedirs(args.results_root, exist_ok=True)

    # active: gpu_id -> (process, job_index, start_time, log_file)
    active: dict[int, tuple[subprocess.Popen, int, float, object]] = {}
    queue = list(range(total))
    completed = 0
    failed = []

    log_dir = os.path.join(PROJECT_ROOT, "logs", "sweep_eval")
    os.makedirs(log_dir, exist_ok=True)

    def launch_on_gpu(gpu_id: int, job_idx: int):
        job = to_run[job_idx]
        cmd = build_eval_command(job, args)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_name = job["label"].replace("/", "_")
        log_path = os.path.join(log_dir, f"{log_name}.log")
        log_file = open(log_path, "w")

        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        active[gpu_id] = (proc, job_idx, time.time(), log_file)
        print(f"  [START] GPU {gpu_id}  job {job_idx+1}/{total}  {job['label']}  "
              f"(log: {log_path})")

    # Initial launch: fill all GPU slots
    for gpu_id in range(min(num_gpus, len(queue))):
        job_idx = queue.pop(0)
        launch_on_gpu(gpu_id, job_idx)

    # Poll until all done
    while active:
        time.sleep(2)
        for gpu_id in list(active.keys()):
            proc, job_idx, t0, log_file = active[gpu_id]
            ret = proc.poll()
            if ret is None:
                continue

            elapsed = time.time() - t0
            job = to_run[job_idx]
            log_file.close()

            if ret == 0:
                completed += 1
                status = "OK"
            else:
                failed.append(job["label"])
                completed += 1
                status = f"FAILED (exit {ret})"

            done = completed
            remaining = total - done
            elapsed_total = time.time() - start_time
            avg_per_job = elapsed_total / done if done > 0 else 0
            eta = avg_per_job * remaining / num_gpus if done > 0 else 0

            print(f"  [{status:>6}] GPU {gpu_id}  job {job_idx+1}/{total}  "
                  f"{job['label']}  ({format_time(elapsed)})  "
                  f"[{done}/{total} done, ETA {format_time(eta)}]")

            del active[gpu_id]

            if queue:
                next_idx = queue.pop(0)
                launch_on_gpu(gpu_id, next_idx)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Evaluation sweep complete")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Succeeded:   {total - len(failed)}/{total}")
    print(f"  Failed:      {len(failed)}/{total}")
    if failed:
        print(f"  Failed runs:")
        for f_label in failed:
            print(f"    - {f_label}")
    print(f"  Results:     {args.results_root}/")
    print(f"  Logs:        {log_dir}/")
    print("=" * 60)

    # Prompt for aggregation
    if completed > 0 and not failed:
        print(f"\nTo aggregate results into a CSV:")
        print(f"  python scripts/aggregate_results.py \\")
        print(f"    --results_dir {args.results_root} \\")
        print(f"    --output_csv {os.path.dirname(args.results_root)}/aggregated/all_results.csv")


if __name__ == "__main__":
    main()
