"""CLI script for production SAE training on cached ViT activations.

Usage:
    python -m src.training.train_sae \
        --backbone dinov2_vitb14 \
        --activation_dir ./activations/dinov2_vitb14/layer_11/ \
        --output_dir ./checkpoints/dinov2_vitb14/batchtopk_16x_k192/ \
        --expansion_factor 16 \
        --k 192 \
        --architecture batchtopk \
        --batch_size 4096 \
        --wandb_project visaebench
"""

import argparse
import json
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.training.overcomplete_config import load_training_data, make_sae_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on cached ViT activations.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Backbone name, e.g. dinov2_vitb14 (used for logging only).")
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="Path to directory containing shard_*.pt files and stats.json.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sae.pt, config.yaml, and training_log.json.")
    parser.add_argument("--expansion_factor", type=int, default=16,
                        help="Dictionary size = d_model * expansion_factor (default: 16).")
    parser.add_argument("--k", type=int, default=192,
                        help="BatchTopK sparsity — active features per step (default: 192).")
    parser.add_argument("--architecture", type=str, default="batchtopk",
                        choices=["batchtopk", "topk", "jumprelu"],
                        help="SAE architecture (default: batchtopk).")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4).")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Training batch size in patch tokens (default: 4096).")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Total gradient steps. If not set, trains for 1 epoch.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="If set, log metrics to this W&B project.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    return parser.parse_args()


def compute_metrics(sae, data: torch.Tensor, device: str, batch_size: int = 4096) -> dict:
    """Compute FVU, L0, and dead feature percentage on the given data tensor."""
    sae.eval()
    all_codes = []
    all_x = []
    all_x_hat = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size].to(device)
            _pre_codes, codes, x_hat = sae(batch)
            all_x.append(batch.cpu())
            all_x_hat.append(x_hat.cpu())
            all_codes.append(codes.cpu())

    x = torch.cat(all_x, dim=0)
    x_hat = torch.cat(all_x_hat, dim=0)
    codes = torch.cat(all_codes, dim=0)

    residual = x - x_hat
    fvu = float(residual.var() / x.var())

    l0 = float((codes != 0).float().sum(dim=1).mean())

    feature_activated = (codes != 0).any(dim=0)
    dead_count = int((~feature_activated).sum())
    dead_pct = 100.0 * dead_count / codes.shape[1]

    return {"fvu": fvu, "l0": l0, "dead_features": dead_count, "dead_pct": dead_pct}


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train_sae] device={device}  backbone={args.backbone}  arch={args.architecture}")

    print(f"[train_sae] Loading activations from {args.activation_dir} ...")
    data = load_training_data(args.activation_dir, normalize=True)
    num_samples, d_model = data.shape
    print(f"[train_sae] Loaded {num_samples:,} patch tokens  shape={tuple(data.shape)}")

    perm = torch.randperm(num_samples)
    data = data[perm]

    config = make_sae_config(
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        architecture=args.architecture,
    )

    from overcomplete import BatchTopKSAE, TopKSAE
    try:
        from overcomplete import JumpReLUSAE
    except ImportError:
        JumpReLUSAE = None

    ctor_kwargs = config["constructor_kwargs"]
    ctor_kwargs["device"] = device

    arch = args.architecture
    if arch == "batchtopk":
        sae = BatchTopKSAE(**ctor_kwargs).tied()
    elif arch == "topk":
        sae = TopKSAE(**ctor_kwargs).tied()
    elif arch == "jumprelu":
        if JumpReLUSAE is None:
            raise ImportError("JumpReLUSAE not found in installed overcomplete version.")
        sae = JumpReLUSAE(**ctor_kwargs).tied()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    print(f"[train_sae] SAE: {arch}  d_model={d_model}  dict_size={config['dict_size']}  k={args.k}")

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    if args.num_steps is not None:
        num_steps = args.num_steps
        steps_per_epoch = (num_samples + args.batch_size - 1) // args.batch_size
        num_epochs = max(1, (num_steps + steps_per_epoch - 1) // steps_per_epoch)
    else:
        num_epochs = 1
        num_steps = (num_samples + args.batch_size - 1) // args.batch_size

    print(f"[train_sae] Training for {num_epochs} epoch(s) (~{num_steps} steps) ...")

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={
            "backbone": args.backbone,
            "architecture": arch,
            "d_model": d_model,
            "dict_size": config["dict_size"],
            "k": args.k,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_steps": num_steps,
            "seed": args.seed,
        })

    log_interval = max(1, num_steps // 100)
    training_log = []
    global_step = 0

    sae.train()
    for epoch in range(num_epochs):
        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True):
            if global_step >= num_steps:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            _pre_codes, codes, x_hat = sae(batch)
            loss = (batch - x_hat).square().mean()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            global_step += 1

            if global_step % log_interval == 0 or global_step == 1:
                entry = {"step": global_step, "loss": loss_val}
                training_log.append(entry)
                if wandb_run:
                    wandb_run.log(entry, step=global_step)

        if global_step >= num_steps:
            break

    print("[train_sae] Training complete. Computing metrics ...")
    metrics = compute_metrics(sae, data, device=device, batch_size=args.batch_size)
    print(f"  FVU:           {metrics['fvu']:.4f}  (target < 0.10)")
    print(f"  L0:            {metrics['l0']:.1f}    (target ≈ {args.k})")
    print(f"  Dead features: {metrics['dead_features']} / {config['dict_size']}  ({metrics['dead_pct']:.1f}%)")

    if wandb_run:
        wandb_run.log({"eval/fvu": metrics["fvu"], "eval/l0": metrics["l0"],
                       "eval/dead_pct": metrics["dead_pct"]})
        wandb_run.finish()

    os.makedirs(args.output_dir, exist_ok=True)

    sae_path = os.path.join(args.output_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)
    print(f"[train_sae] Saved weights → {sae_path}")

    config_to_save = {
        "backbone": args.backbone,
        "architecture": arch,
        "d_model": d_model,
        "expansion_factor": args.expansion_factor,
        "dict_size": config["dict_size"],
        "k": args.k,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_steps": global_step,
        "seed": args.seed,
        "activation_dir": args.activation_dir,
    }
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False)
    print(f"[train_sae] Saved config  → {config_path}")

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({"training_loss": training_log, "final_metrics": metrics}, f, indent=2)
    print(f"[train_sae] Saved log     → {log_path}")


if __name__ == "__main__":
    main()
