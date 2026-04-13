"""Launch the full training sweep across multiple GPUs on a single node.

Reads the sweep YAML config, generates all (backbone, arch, expansion, k)
combinations, and distributes them round-robin across available GPUs.
Runs N_GPUS jobs concurrently at all times — when one finishes, the next
queued job takes its GPU slot.

Usage:
    # Dry run — see all 60 jobs and GPU assignments
    python scripts/launch_train_sweep.py --dry_run

    # Run on 8 H100s
    python scripts/launch_train_sweep.py --num_gpus 8

    # Run on GPUs 0-3 only, skip already-trained checkpoints
    python scripts/launch_train_sweep.py --num_gpus 4 --skip_existing

    # Custom sweep config
    python scripts/launch_train_sweep.py --num_gpus 8 \
        --sweep_config configs/sweeps/my_sweep.yaml
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
    "sweep_config": "configs/sweeps/main_sweep.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch training sweep across multiple GPUs on a single node.",
    )
    p.add_argument("--sweep_config", type=str, default=DEFAULTS["sweep_config"],
                   help="Path to sweep YAML config.")
    p.add_argument("--num_gpus", type=int, default=8,
                   help="Number of GPUs to use (default: 8).")
    p.add_argument("--activation_root", type=str, default=DEFAULTS["activation_root"])
    p.add_argument("--val_root", type=str, default=DEFAULTS["val_root"])
    p.add_argument("--checkpoint_root", type=str, default=DEFAULTS["checkpoint_root"])
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs where sae.pt already exists.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the job list without executing anything.")
    return p.parse_args()


def build_runs(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Generate all run configs from sweep YAML."""
    backbones = cfg["backbones"]
    expansion_factors = cfg["expansion_factors"]
    k_values = cfg["k_values"]
    training = cfg.get("training", {})

    if "architectures" in cfg:
        architectures = cfg["architectures"]
    elif "architecture" in cfg:
        architectures = [cfg["architecture"]]
    else:
        architectures = ["batchtopk"]

    seeds = cfg.get("seeds", None)

    runs = []
    for backbone, arch, exp, k in itertools.product(
        backbones, architectures, expansion_factors, k_values,
    ):
        activation_dir = os.path.join(args.activation_root, backbone, "layer_11")
        val_dir = os.path.join(args.val_root, backbone, "layer_11")

        if seeds:
            for seed in seeds:
                name = f"{arch}_{exp}x_k{k}_seed{seed}"
                output_dir = os.path.join(args.checkpoint_root, backbone, name)
                runs.append({
                    "backbone": backbone, "architecture": arch,
                    "expansion_factor": exp, "k": k,
                    "activation_dir": activation_dir, "val_dir": val_dir,
                    "output_dir": output_dir,
                    "lr": training.get("lr", 1e-3),
                    "batch_size": training.get("batch_size", 4096),
                    "num_epochs": training.get("num_epochs", 4),
                    "seed": seed,
                    "label": f"{backbone}/{name}",
                })
        else:
            seed = training.get("seed", 42)
            name = f"{arch}_{exp}x_k{k}"
            output_dir = os.path.join(args.checkpoint_root, backbone, name)
            runs.append({
                "backbone": backbone, "architecture": arch,
                "expansion_factor": exp, "k": k,
                "activation_dir": activation_dir, "val_dir": val_dir,
                "output_dir": output_dir,
                "lr": training.get("lr", 1e-3),
                "batch_size": training.get("batch_size", 4096),
                "num_epochs": training.get("num_epochs", 4),
                "seed": seed,
                "label": f"{backbone}/{name}",
            })

    return runs


def build_command(run: dict) -> list[str]:
    return [
        sys.executable, "-m", "src.training.train_sae",
        "--backbone", run["backbone"],
        "--activation_dir", run["activation_dir"],
        "--val_dir", run["val_dir"],
        "--output_dir", run["output_dir"],
        "--expansion_factor", str(run["expansion_factor"]),
        "--k", str(run["k"]),
        "--architecture", run["architecture"],
        "--lr", str(run["lr"]),
        "--batch_size", str(run["batch_size"]),
        "--num_epochs", str(run["num_epochs"]),
        "--seed", str(run["seed"]),
    ]


def is_complete(run: dict) -> bool:
    return os.path.isfile(os.path.join(run["output_dir"], "sae.pt"))


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

    print(f"Sweep: {cfg.get('sweep_name', '(unnamed)')}")
    print(f"  {cfg.get('description', '')}")
    print(f"  GPUs: {args.num_gpus}")
    print()

    all_runs = build_runs(cfg, args)

    # Filter
    if args.skip_existing:
        skipped = [r for r in all_runs if is_complete(r)]
        to_run = [r for r in all_runs if not is_complete(r)]
    else:
        skipped = []
        to_run = all_runs

    print(f"Total runs: {len(all_runs)}")
    print(f"  Skip (already complete): {len(skipped)}")
    print(f"  To execute: {len(to_run)}")
    print()

    if not to_run:
        print("Nothing to run.")
        return

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        for i, run in enumerate(to_run):
            gpu = i % args.num_gpus
            print(f"  [{i+1:>3}/{len(to_run)}] GPU {gpu}  {run['label']}")
        print(f"\n{len(to_run)} jobs across {args.num_gpus} GPUs "
              f"({len(to_run) // args.num_gpus + (1 if len(to_run) % args.num_gpus else 0)} "
              f"rounds)")
        return

    # ── Parallel execution ───────────────────────────────────────────────
    # Maintain a pool of up to num_gpus concurrent processes.
    # Each GPU slot runs jobs sequentially from its queue.

    num_gpus = args.num_gpus
    total = len(to_run)
    start_time = time.time()

    # Track: gpu_id -> (process, run_index, start_time)
    active: dict[int, tuple[subprocess.Popen, int, float]] = {}
    queue = list(range(total))  # indices into to_run
    completed = 0
    failed = []

    # Log file for each job
    log_dir = os.path.join(PROJECT_ROOT, "logs", "sweep_train")
    os.makedirs(log_dir, exist_ok=True)

    def launch_on_gpu(gpu_id: int, run_idx: int):
        run = to_run[run_idx]
        cmd = build_command(run)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_name = run["label"].replace("/", "_")
        log_path = os.path.join(log_dir, f"{log_name}.log")
        log_file = open(log_path, "w")

        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        active[gpu_id] = (proc, run_idx, time.time(), log_file)
        print(f"  [START] GPU {gpu_id}  job {run_idx+1}/{total}  {run['label']}  "
              f"(log: {log_path})")

    # Initial launch: fill all GPU slots
    for gpu_id in range(min(num_gpus, len(queue))):
        run_idx = queue.pop(0)
        launch_on_gpu(gpu_id, run_idx)

    # Poll until all done
    while active:
        time.sleep(2)
        for gpu_id in list(active.keys()):
            proc, run_idx, t0, log_file = active[gpu_id]
            ret = proc.poll()
            if ret is None:
                continue  # still running

            # Job finished
            elapsed = time.time() - t0
            run = to_run[run_idx]
            log_file.close()

            if ret == 0:
                completed += 1
                status = "OK"
            else:
                failed.append(run["label"])
                completed += 1
                status = f"FAILED (exit {ret})"

            done = completed
            remaining = total - done
            elapsed_total = time.time() - start_time
            avg_per_job = elapsed_total / done if done > 0 else 0
            eta = avg_per_job * remaining / num_gpus if done > 0 else 0

            print(f"  [{status:>6}] GPU {gpu_id}  job {run_idx+1}/{total}  "
                  f"{run['label']}  ({format_time(elapsed)})  "
                  f"[{done}/{total} done, ETA {format_time(eta)}]")

            del active[gpu_id]

            # Launch next job on this GPU if queue is not empty
            if queue:
                next_idx = queue.pop(0)
                launch_on_gpu(gpu_id, next_idx)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Training sweep complete")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Succeeded:   {total - len(failed)}/{total}")
    print(f"  Failed:      {len(failed)}/{total}")
    if failed:
        print(f"  Failed runs:")
        for f_label in failed:
            print(f"    - {f_label}")
    print(f"  Logs:        {log_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
