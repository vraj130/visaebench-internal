# VisaeBench

A benchmarking framework for evaluating Sparse Autoencoders (SAEs) trained on Vision Transformer (ViT) patch-token activations. Measures SAE quality across 7 metrics: reconstruction quality (FVU, downstream preservation), concept detection (sparse probing, monosemanticity, cross-domain), spatial coherence (localization), and disentanglement (absorption).

## Setup

Requires Python 3.11+ and CUDA (tested on RTX 3090).

```bash
# Create and activate virtual environment
python -m venv .visaebench
source .visaebench/bin/activate

# Install dependencies (uses uv for PyTorch CUDA wheels)
uv pip install -e .
```

Set environment variables in `~/.bashrc` or `.env`:

```bash
export HF_HOME=/mnt/NAS/data/ds5725/visaebench/huggingface
export HF_DATASETS_CACHE=/mnt/NAS/data/ds5725/visaebench/huggingface/datasets
export HF_TOKEN=<your_token>
```

## Pipeline Overview

```
ImageNet images
    --> cache_activations (extract ViT patch activations, save as shards)
    --> train_sae (train SAE on cached activations)
    --> evaluate (run M1-M7 metrics on trained SAE)
```

## Step 1: Cache Activations

Extract patch-token activations from ViT backbones and save as sharded `.pt` files.

**Supported backbones:**

| Backbone | Model ID | Patch Count |
|----------|----------|-------------|
| `clip_vitb16` | openai/clip-vit-base-patch16 | 196 |
| `dinov2_vitb14` | facebook/dinov2-base | 256 |
| `siglip_vitb16` | google/siglip-base-patch16-224 | 196 |
| `mae_vitb16` | facebook/vit-mae-base | 196 |
| `deit_vitb16` | facebook/deit-base-patch16-224 | 196 |

**Cache training activations** (100K images, balanced across 1000 classes):

```bash
python -m src.caching.cache_activations \
    --backbone dinov2_vitb14 \
    --use_hf_local \
    --hf_split train \
    --balanced \
    --num_images 100000 \
    --output_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \
    --batch_size 512
```

**Cache validation activations** (full 50K val set, sequential order for label alignment):

```bash
python -m src.caching.cache_activations \
    --backbone dinov2_vitb14 \
    --use_hf_local \
    --hf_split validation \
    --output_dir /mnt/NAS/data/ds5725/visaebench/activations_val/dinov2_vitb14/layer_11/ \
    --batch_size 512
```

**Cache all backbones at once:**

```bash
bash scripts/cache_all_backbones_train.sh   # train (100K balanced)
bash scripts/cache_all_backbones.sh          # val (50K sequential)
```

**Output structure:**

```
activations/{backbone}/layer_11/
    shard_000.pt   # [N, patch_count, 768] float16
    shard_001.pt
    ...
    stats.json     # mean (per-dim), std (scalar), num_images, d_model
```

Each shard contains ~5000 images. `stats.json` provides the normalization statistics used during training.

## Step 2: Train SAE

Train a Sparse Autoencoder on cached activations. Supports TopK and BatchTopK architectures from the `overcomplete` library.

**Train a single SAE:**

```bash
python -m src.training.train_sae \
    --backbone dinov2_vitb14 \
    --activation_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \
    --output_dir /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192/ \
    --expansion_factor 16 \
    --k 192 \
    --architecture batchtopk \
    --lr 1e-3 \
    --batch_size 4096 \
    --num_epochs 4 \
    --seed 42
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--architecture` | `topk` or `batchtopk` | `topk` |
| `--expansion_factor` | Dictionary size = d_model x factor | 16 |
| `--k` | Per-sample sparsity (active features) | 192 |
| `--batch_size` | Patch tokens per batch | 4096 |
| `--num_epochs` | Passes over all shards | 3 |
| `--val_dir` | Val shard directory (optional, for eval after training) | None |
| `--wandb_project` | W&B project name (optional) | None |

**BatchTopK note:** The `top_k` parameter passed to `BatchTopKSAE` is `k * batch_size` (e.g., 192 x 4096 = 786,432). This is handled automatically by `make_sae_config()` when `batch_size` is provided.

**Train all 5 backbones** (same config, skips existing):

```bash
bash scripts/train_batchtopk_all.sh
```

**Checkpoint output:**

```
checkpoints/{backbone}/batchtopk_16x_k192/
    sae.pt            # model state dict (includes _running_threshold for BatchTopK)
    config.yaml       # training config (backbone, architecture, d_model, k, etc.)
    training_log.json # per-step loss + final eval metrics
```

Training processes shards one at a time to stay within RAM limits (~8 GB free).

## Step 3: Run Sweep

For systematic experiments across multiple backbones, expansion factors, and k values.

**Define a sweep** in `configs/sweeps/`:

```yaml
# configs/sweeps/main_sweep.yaml
sweep_name: main_sweep
backbones: [clip_vitb16, dinov2_vitb14, siglip_vitb16, mae_vitb16, deit_vitb16]
architecture: topk
expansion_factors: [8, 16, 32]
k_values: [128, 192, 256]
training:
  lr: 1e-3
  batch_size: 4096
  num_epochs: 3
  seed: 42
```

**Run the sweep:**

```bash
# Preview all commands
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode dry_run

# Run sequentially
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode sequential --skip_existing

# Submit to SLURM cluster
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode slurm --slurm_partition gpu
```

## Step 4: Evaluate

### M1: FVU (Fraction of Variance Unexplained)

Measures reconstruction quality. Computed incrementally over val shards.

```python
from src.evaluation.reconstruction.fvu import FVUMetric

metric = FVUMetric(batch_size=512)
results = metric.evaluate(sae, shard_paths, mean, std, device="cuda", dict_size=12288)
# {"fvu": 0.1009, "l0": 208.0, "dead_features": 172, "dead_pct": 1.4}
```

### M3: Sparse Probing

Measures concept alignment by training k-sparse linear probes on SAE features to predict ImageNet classes.

```bash
python scripts/run_sparse_probing.py \
    --sae_checkpoint /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192/sae.pt \
    --sae_config /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192/config.yaml \
    --activation_dir /mnt/NAS/data/ds5725/visaebench/activations_val/dinov2_vitb14/layer_11/ \
    --output_path results/raw/dinov2_vitb14_batchtopk_16x_k192_sparse_probing.json
```

## Data Storage

All heavy artifacts live on NAS at `/mnt/NAS/data/ds5725/visaebench/`. Never store large files under `/mnt/NAS/home/`.

```
/mnt/NAS/data/ds5725/visaebench/
    activations/         # train shards per backbone/layer
    activations_val/     # val shards per backbone/layer
    checkpoints/         # trained SAE weights + configs
    huggingface/         # HF model & dataset cache
    results/             # evaluation results JSON
```

Import paths from `src.utils.paths`:

```python
from src.utils.paths import ACTIVATION_ROOT, CHECKPOINT_ROOT, DATA_ROOT
```

## Project Structure

```
src/
    backbones/           # ViT backbone adapters (CLIP, DINOv2, SigLIP, MAE, DeiT)
    caching/             # Activation extraction and sharding
    training/            # SAE training (train_sae.py) and sweep runner
    evaluation/          # Metrics: M1 FVU, M3 sparse probing, (M2/M4-M7 planned)
    analysis/            # Post-eval aggregation and visualization (planned)
    utils/               # Path constants, env setup
configs/
    sweeps/              # Sweep YAML configs (main, ablation, seed)
scripts/                 # CLI entry points and batch scripts
notebooks/               # Interactive exploration and pilot experiments
```

## Hardware

- **GPUs:** 2x RTX 3090 (24 GB VRAM each); GPU 0 is primary
- **RAM:** ~62 GB total, ~8-9 GB typically free
- **Storage:** NFS-mounted NAS at `/mnt/NAS/`
- Training is shard-by-shard to stay within RAM limits
