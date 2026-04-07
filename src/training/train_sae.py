"""CLI script for production SAE training on cached ViT activations.

Usage:
    python -m src.training.train_sae \
        --backbone dinov2_vitb14 \
        --activation_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \
        --output_dir /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/topk_16x_k192/ \
        --expansion_factor 16 \
        --k 192 \
        --architecture topk \
        --batch_size 4096 \
        --num_epochs 3 \
        --val_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11_val/ \
        --wandb_project visaebench

Trains shard-by-shard to stay within ~8 GB system RAM.
"""

import argparse
import glob
import json
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.training.overcomplete_config import make_sae_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on cached ViT activations.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Backbone name, e.g. dinov2_vitb14 (used for logging only).")
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="Path to directory containing shard_*.pt files and stats.json.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sae.pt, config.yaml, and training_log.json.")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Path to held-out val shard directory (shard_*.pt + stats.json). "
                             "If not set, evaluation is skipped.")
    parser.add_argument("--expansion_factor", type=int, default=16,
                        help="Dictionary size = d_model * expansion_factor (default: 16).")
    parser.add_argument("--k", type=int, default=192,
                        help="TopK sparsity — active features per step (default: 192).")
    parser.add_argument("--architecture", type=str, default="topk",
                        choices=["batchtopk", "topk", "jumprelu"],
                        help="SAE architecture (default: topk).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3).")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Training batch size in patch tokens (default: 4096).")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training loss every N steps (default: 50).")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Batch size for evaluation (default: 512, kept small to limit RAM).")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="If set, log metrics to this W&B project.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    return parser.parse_args()


def _load_stats(activation_dir: str) -> tuple[torch.Tensor, float]:
    """Load mean and std from stats.json in an activation directory."""
    stats_path = os.path.join(activation_dir, "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = float(stats["std"])
    return mean, std


def _discover_shards(activation_dir: str) -> list[str]:
    """Return sorted list of shard_*.pt paths in a directory."""
    paths = sorted(glob.glob(os.path.join(activation_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard files found in {activation_dir}")
    return paths


def evaluate(
    sae: torch.nn.Module,
    shard_paths: list[str],
    mean: torch.Tensor,
    std: float,
    dict_size: int,
    device: str,
    batch_size: int = 512,
) -> dict:
    """Evaluate FVU, L0, and dead features incrementally over shards.

    Uses running sums instead of accumulating tensors to avoid OOM.
    """
    sae.eval()

    n_tok = 0
    sum_x2 = 0.0
    sum_x = 0.0
    sum_res2 = 0.0
    sum_res = 0.0
    sum_l0 = 0.0
    ever_active = torch.zeros(dict_size, dtype=torch.bool)

    with torch.no_grad():
        for shard_path in tqdm(shard_paths, desc="Eval shards"):
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            N, P, D = shard.shape
            tokens = shard.reshape(N * P, D).float()
            tokens = (tokens - mean) / std

            for start in range(0, len(tokens), batch_size):
                batch = tokens[start : start + batch_size].to(device)
                _pre, codes, x_hat = sae(batch)
                res = batch - x_hat

                sum_x2 += float(batch.pow(2).sum())
                sum_x += float(batch.sum())
                sum_res2 += float(res.pow(2).sum())
                sum_res += float(res.sum())
                sum_l0 += float((codes != 0).float().sum(dim=1).sum())
                ever_active |= (codes != 0).any(dim=0).cpu()
                n_tok += batch.shape[0]

                del batch, _pre, codes, x_hat, res

            del shard, tokens

    if device == "cuda":
        torch.cuda.empty_cache()

    var_x = sum_x2 / n_tok - (sum_x / n_tok) ** 2
    var_res = sum_res2 / n_tok - (sum_res / n_tok) ** 2
    fvu = float(var_res / var_x)
    l0 = float(sum_l0 / n_tok)
    dead_count = int((~ever_active).sum())
    dead_pct = 100.0 * dead_count / dict_size

    return {"fvu": fvu, "l0": l0, "dead_features": dead_count, "dead_pct": dead_pct}


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train_sae] device={device}  backbone={args.backbone}  arch={args.architecture}")

    # ── Discover shards and load stats ────────────────────────────────────────
    train_shards = _discover_shards(args.activation_dir)
    mean, std = _load_stats(args.activation_dir)

    # Peek at first shard to get d_model
    first_shard = torch.load(train_shards[0], map_location="cpu", weights_only=True)
    d_model = first_shard.shape[-1]
    patches_per_image = first_shard.shape[1]
    del first_shard

    print(f"[train_sae] {len(train_shards)} training shards  d_model={d_model}  patches={patches_per_image}")
    print(f"[train_sae] Shards loaded one at a time to stay within RAM limits.")

    # ── Instantiate SAE ───────────────────────────────────────────────────────
    config = make_sae_config(
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        architecture=args.architecture,
        batch_size=args.batch_size if args.architecture == "batchtopk" else None,
    )
    dict_size = config["dict_size"]

    from overcomplete import BatchTopKSAE, TopKSAE
    try:
        from overcomplete import JumpReLUSAE
    except ImportError:
        JumpReLUSAE = None

    ctor_kwargs = config["constructor_kwargs"]
    ctor_kwargs["device"] = device

    arch = args.architecture
    if arch == "batchtopk":
        sae = BatchTopKSAE(**ctor_kwargs)
    elif arch == "topk":
        sae = TopKSAE(**ctor_kwargs)
    elif arch == "jumprelu":
        if JumpReLUSAE is None:
            raise ImportError("JumpReLUSAE not found in installed overcomplete version.")
        sae = JumpReLUSAE(**ctor_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    num_params = sum(p.numel() for p in sae.parameters())
    print(f"[train_sae] SAE: {arch}  dict_size={dict_size}  k={args.k}  params={num_params:,}")

    # ── Training loop (shard-by-shard) ────────────────────────────────────────
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    training_log: list[dict] = []
    global_step = 0

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={
            "backbone": args.backbone,
            "architecture": arch,
            "d_model": d_model,
            "dict_size": dict_size,
            "k": args.k,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        })

    sae.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Shuffle shard order each epoch
        epoch_shards = train_shards[:]
        random.shuffle(epoch_shards)

        for shard_path in tqdm(epoch_shards, desc=f"Epoch {epoch+1}/{args.num_epochs} shards", leave=False):
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            N, P, D = shard.shape
            tokens = shard.reshape(N * P, D).float()
            tokens = (tokens - mean) / std
            perm = torch.randperm(tokens.shape[0])
            tokens = tokens[perm]

            loader = DataLoader(
                TensorDataset(tokens), batch_size=args.batch_size,
                shuffle=False, drop_last=False, num_workers=0,
            )

            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                _pre_codes, codes, x_hat = sae(batch)
                loss = (batch - x_hat).square().mean()
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                epoch_loss += loss_val
                epoch_steps += 1
                global_step += 1

                if global_step % args.log_every == 0:
                    entry = {"step": global_step, "loss": loss_val}
                    training_log.append(entry)
                    if wandb_run:
                        wandb_run.log(entry, step=global_step)

            del shard, tokens, loader

        avg = epoch_loss / epoch_steps if epoch_steps > 0 else float("nan")
        print(f"Epoch {epoch+1:>2}/{args.num_epochs}  |  step {global_step:>6}  |  avg loss: {avg:.4f}")

    print("[train_sae] Training complete.")

    # ── Save checkpoint (before eval — safe if eval OOMs) ─────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    sae_path = os.path.join(args.output_dir, "sae.pt")
    # Include running_threshold for BatchTopKSAE — it's a plain attribute,
    # not a registered buffer, so state_dict() doesn't capture it.
    save_dict = sae.state_dict()
    if arch == "batchtopk" and hasattr(sae, "running_threshold") and sae.running_threshold is not None:
        save_dict["_running_threshold"] = sae.running_threshold.detach().cpu()
    torch.save(save_dict, sae_path)
    print(f"[train_sae] Saved weights → {sae_path}")

    config_to_save = {
        "backbone": args.backbone,
        "architecture": arch,
        "d_model": d_model,
        "expansion_factor": args.expansion_factor,
        "dict_size": dict_size,
        "k": args.k,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "total_steps": global_step,
        "seed": args.seed,
        "activation_dir": args.activation_dir,
    }
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False)
    print(f"[train_sae] Saved config  → {config_path}")

    # ── Evaluate on held-out val shards ───────────────────────────────────────
    metrics = None
    if args.val_dir:
        val_shards = _discover_shards(args.val_dir)
        val_mean, val_std = _load_stats(args.val_dir)
        print(f"[train_sae] Evaluating on {len(val_shards)} val shards from {args.val_dir} ...")
        metrics = evaluate(
            sae, val_shards, val_mean, val_std, dict_size,
            device=device, batch_size=args.eval_batch_size,
        )
        print(f"  FVU:           {metrics['fvu']:.4f}  (target < 0.10)")
        print(f"  L0:            {metrics['l0']:.1f}    (target ≈ {args.k})")
        print(f"  Dead features: {metrics['dead_features']} / {dict_size}  ({metrics['dead_pct']:.1f}%)")

        if wandb_run:
            wandb_run.log({"eval/fvu": metrics["fvu"], "eval/l0": metrics["l0"],
                           "eval/dead_pct": metrics["dead_pct"]})
    else:
        print("[train_sae] No --val_dir provided, skipping evaluation.")

    if wandb_run:
        wandb_run.finish()

    log_path = os.path.join(args.output_dir, "training_log.json")
    log_data = {"training_loss": training_log}
    if metrics:
        log_data["final_metrics"] = metrics
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[train_sae] Saved log     → {log_path}")


if __name__ == "__main__":
    main()
