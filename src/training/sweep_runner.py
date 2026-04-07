"""CLI script to run SAE training sweeps from a YAML config.

Generates all (backbone, architecture, expansion_factor, k) combinations
and executes them sequentially, via SLURM, or as a dry run.

Usage:
    # Dry run — see what would execute
    python -m src.training.sweep_runner \
        --sweep_config configs/sweeps/main_sweep.yaml --mode dry_run

    # Sequential execution on current machine
    python -m src.training.sweep_runner \
        --sweep_config configs/sweeps/main_sweep.yaml --mode sequential --skip_existing

    # Submit to SLURM cluster
    python -m src.training.sweep_runner \
        --sweep_config configs/sweeps/main_sweep.yaml --mode slurm \
        --slurm_partition a100 --slurm_time 2:00:00 --skip_existing
"""

import argparse
import itertools
import os
import subprocess
import sys
import textwrap

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ACTIVATION_ROOT_DEFAULT = "/mnt/NAS/data/ds5725/visaebench/activations"
VAL_ROOT_DEFAULT = "/mnt/NAS/data/ds5725/visaebench/activations_val"
CHECKPOINT_ROOT_DEFAULT = "/mnt/NAS/data/ds5725/visaebench/checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAE training sweeps from a YAML config.")
    p.add_argument("--sweep_config", type=str, required=True,
                   help="Path to sweep YAML file (e.g. configs/sweeps/main_sweep.yaml).")
    p.add_argument("--activation_root", type=str, default=ACTIVATION_ROOT_DEFAULT,
                   help="Base path for cached activations.")
    p.add_argument("--val_root", type=str, default=VAL_ROOT_DEFAULT,
                   help="Base path for val activations.")
    p.add_argument("--checkpoint_root", type=str, default=CHECKPOINT_ROOT_DEFAULT,
                   help="Base path for checkpoints.")
    p.add_argument("--mode", type=str, default="dry_run",
                   choices=["sequential", "slurm", "dry_run"],
                   help="Execution mode (default: dry_run).")
    p.add_argument("--slurm_partition", type=str, default="gpu",
                   help="SLURM partition name (default: gpu).")
    p.add_argument("--slurm_gpus", type=int, default=1,
                   help="GPUs per SLURM job (default: 1).")
    p.add_argument("--slurm_time", type=str, default="4:00:00",
                   help="SLURM time limit per job (default: 4:00:00).")
    p.add_argument("--slurm_mem", type=str, default="32G",
                   help="SLURM memory per job (default: 32G).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs where output_dir/sae.pt already exists.")
    return p.parse_args()


def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_runs(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Build the list of run configs from the sweep YAML."""
    backbones = cfg["backbones"]
    expansion_factors = cfg["expansion_factors"]
    k_values = cfg["k_values"]
    training = cfg.get("training", {})

    # Architecture(s): support both singular "architecture" and plural "architectures"
    if "architectures" in cfg:
        architectures = cfg["architectures"]
    elif "architecture" in cfg:
        architectures = [cfg["architecture"]]
    else:
        architectures = ["topk"]

    # Seeds: if present, generate one run per (backbone, arch, exp, k, seed)
    seeds = cfg.get("seeds", None)

    runs = []
    for backbone, arch, exp, k in itertools.product(backbones, architectures, expansion_factors, k_values):
        activation_dir = os.path.join(args.activation_root, backbone, "layer_11")
        val_dir = os.path.join(args.val_root, backbone, "layer_11")

        if seeds:
            for seed in seeds:
                output_dir = os.path.join(
                    args.checkpoint_root, backbone,
                    f"{arch}_{exp}x_k{k}_seed{seed}",
                )
                runs.append(_make_run(
                    backbone, arch, exp, k, activation_dir, val_dir, output_dir,
                    training, seed_override=seed,
                ))
        else:
            output_dir = os.path.join(
                args.checkpoint_root, backbone,
                f"{arch}_{exp}x_k{k}",
            )
            runs.append(_make_run(
                backbone, arch, exp, k, activation_dir, val_dir, output_dir,
                training, seed_override=None,
            ))

    return runs


def _make_run(
    backbone: str, arch: str, exp: int, k: int,
    activation_dir: str, val_dir: str, output_dir: str,
    training: dict, seed_override: int | None,
) -> dict:
    seed = seed_override if seed_override is not None else training.get("seed", 42)
    return {
        "backbone": backbone,
        "architecture": arch,
        "expansion_factor": exp,
        "k": k,
        "activation_dir": activation_dir,
        "val_dir": val_dir,
        "output_dir": output_dir,
        "lr": training.get("lr", 1e-3),
        "batch_size": training.get("batch_size", 4096),
        "num_epochs": training.get("num_epochs", 3),
        "seed": seed,
    }


def build_command(run: dict) -> list[str]:
    """Build the python -m src.training.train_sae command for a run."""
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


def build_command_str(run: dict) -> str:
    """Build a shell-friendly command string for display and SLURM scripts."""
    return (
        f"python -m src.training.train_sae \\\n"
        f"    --backbone {run['backbone']} \\\n"
        f"    --activation_dir {run['activation_dir']} \\\n"
        f"    --val_dir {run['val_dir']} \\\n"
        f"    --output_dir {run['output_dir']} \\\n"
        f"    --expansion_factor {run['expansion_factor']} \\\n"
        f"    --k {run['k']} \\\n"
        f"    --architecture {run['architecture']} \\\n"
        f"    --lr {run['lr']} \\\n"
        f"    --batch_size {run['batch_size']} \\\n"
        f"    --num_epochs {run['num_epochs']} \\\n"
        f"    --seed {run['seed']}"
    )


def run_label(run: dict) -> str:
    return f"{run['backbone']} {run['architecture']} {run['expansion_factor']}x k{run['k']} seed{run['seed']}"


def is_complete(run: dict) -> bool:
    return os.path.isfile(os.path.join(run["output_dir"], "sae.pt"))


def make_slurm_script(run: dict, args: argparse.Namespace) -> str:
    job_name = f"visaebench_{run['backbone']}_{run['architecture']}_{run['expansion_factor']}x_k{run['k']}"
    cmd_str = build_command_str(run)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={args.slurm_partition}
        #SBATCH --gres=gpu:{args.slurm_gpus}
        #SBATCH --time={args.slurm_time}
        #SBATCH --mem={args.slurm_mem}
        #SBATCH --output=logs/slurm/%x_%j.out
        #SBATCH --error=logs/slurm/%x_%j.err

        cd {PROJECT_ROOT}
        source .visaebench/bin/activate

        {cmd_str}
    """)


def main() -> None:
    args = parse_args()
    cfg = load_sweep_config(args.sweep_config)

    print(f"Sweep: {cfg.get('sweep_name', '(unnamed)')}")
    print(f"  {cfg.get('description', '')}")
    print()

    all_runs = build_runs(cfg, args)
    total = len(all_runs)

    # Determine which runs to skip
    skipped = []
    to_run = []
    for run in all_runs:
        if args.skip_existing and is_complete(run):
            skipped.append(run)
        else:
            to_run.append(run)

    print(f"Total runs: {total}")
    print(f"  Skip (already complete): {len(skipped)}")
    print(f"  To execute: {len(to_run)}")
    print()

    if not to_run:
        print("Nothing to run.")
        return

    # ── dry_run ──────────────────────────────────────────────────────────────
    if args.mode == "dry_run":
        for i, run in enumerate(to_run, 1):
            print(f"--- Run {i}/{len(to_run)}: {run_label(run)} ---")
            print(build_command_str(run))
            print()
        print(f"Summary: {len(to_run)} runs to execute, {len(skipped)} skipped")

    # ── sequential ───────────────────────────────────────────────────────────
    elif args.mode == "sequential":
        failures = []
        for i, run in enumerate(to_run, 1):
            label = run_label(run)
            print(f"{'='*60}")
            print(f"Run {i}/{len(to_run)}: {label}")
            print(f"{'='*60}")
            cmd = build_command(run)
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                print(f"FAILED (exit {result.returncode}): {label}")
                failures.append(label)
            print()

        print(f"\nSweep complete: {len(to_run) - len(failures)}/{len(to_run)} succeeded")
        if failures:
            print("Failed runs:")
            for f in failures:
                print(f"  - {f}")

    # ── slurm ────────────────────────────────────────────────────────────────
    elif args.mode == "slurm":
        scripts_dir = os.path.join(PROJECT_ROOT, "logs", "slurm", "scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        # Also ensure the log output directory exists
        os.makedirs(os.path.join(PROJECT_ROOT, "logs", "slurm"), exist_ok=True)

        job_ids = []
        for i, run in enumerate(to_run, 1):
            label = run_label(run)
            script_content = make_slurm_script(run, args)
            script_name = (
                f"{run['backbone']}_{run['architecture']}"
                f"_{run['expansion_factor']}x_k{run['k']}_seed{run['seed']}.sh"
            )
            script_path = os.path.join(scripts_dir, script_name)
            with open(script_path, "w") as f:
                f.write(script_content)

            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True, text=True, cwd=PROJECT_ROOT,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"[{i}/{len(to_run)}] Submitted {label} -> job {job_id}")
            else:
                print(f"[{i}/{len(to_run)}] FAILED to submit {label}: {result.stderr.strip()}")

        print(f"\nSubmitted {len(job_ids)}/{len(to_run)} jobs")
        if job_ids:
            print(f"Job IDs: {', '.join(job_ids)}")
            print(f"SLURM scripts saved to: {scripts_dir}")


if __name__ == "__main__":
    main()
