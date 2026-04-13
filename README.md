# VisaeBench

A benchmarking framework for evaluating Sparse Autoencoders (SAEs) trained on Vision Transformer (ViT) patch-token activations. VisaeBench measures SAE quality across **7 metrics** spanning 4 capability dimensions: reconstruction quality, concept detection, spatial coherence, and disentanglement.

## Why VisaeBench?

Existing SAE evaluation benchmarks (e.g., SAEBench) focus on language models. Vision models introduce unique challenges — spatial structure, cross-model embeddings, and object-part localization — that require vision-specific evaluation. VisaeBench fills this gap with a comprehensive suite of metrics designed for the visual domain, including a novel **feature localization** metric that measures spatial coherence against ground-truth segmentation masks.

## Metrics

| # | Metric | Dimension | What it measures |
|---|--------|-----------|------------------|
| M1 | FVU (Fraction of Variance Unexplained) | Reconstruction | How well SAE reconstructs original activations |
| M2 | Downstream Preservation | Reconstruction | Whether SAE reconstruction preserves downstream task performance |
| M3 | Sparse Probing | Concept Detection | Whether k-sparse probes on SAE features can predict ImageNet classes |
| M4 | Monosemanticity | Concept Detection | Whether top-activating images for each feature are semantically coherent (evaluated with cross-model embeddings) |
| M5 | Cross-Domain | Concept Detection | Whether SAE features remain interpretable on OOD datasets (iNaturalist, EuroSAT) |
| M6 | Feature Localization | Spatial Coherence | Whether SAE features are spatially coherent against segmentation masks |
| M7 | Feature Absorption | Disentanglement | Whether SAE features absorb multiple concepts into single features |

## Supported Backbones

| Backbone | Model ID | Patch Count | Special Tokens |
|----------|----------|-------------|----------------|
| `clip_vitb16` | openai/clip-vit-base-patch16 | 196 | CLS |
| `dinov2_vitb14` | facebook/dinov2-base | 256 | CLS + 4 registers |
| `siglip_vitb16` | google/siglip-base-patch16-224 | 196 | None |
| `mae_vitb16` | facebook/vit-mae-base | 196 | CLS |
| `deit_vitb16` | facebook/deit-base-patch16-224 | 196 | CLS + distillation |

All backbones extract activations from **layer 11** with d_model = 768.

## Setup

Requires Python 3.11+ and CUDA.

```bash
python -m venv .visaebench
source .visaebench/bin/activate
uv pip install -e .
```

Set environment variables:

```bash
export HF_HOME=<your_hf_cache_dir>
export HF_DATASETS_CACHE=<your_datasets_cache_dir>
export HF_TOKEN=<your_token>
```

## Pipeline

```
ImageNet images
    --> cache_activations   (extract ViT patch activations, save as shards)
    --> train_sae           (train SAE on cached activations)
    --> evaluate            (run M1-M7 metrics on trained SAE)
```

### Step 1: Cache Activations

Extract patch-token activations from a ViT backbone and save as sharded `.pt` files.

**Training activations** (100K images, balanced across 1000 classes):

```bash
python -m src.caching.cache_activations \
    --backbone dinov2_vitb14 \
    --use_hf_local \
    --hf_split train \
    --balanced \
    --num_images 100000 \
    --output_dir /path/to/activations/dinov2_vitb14/layer_11/ \
    --batch_size 512
```

**Validation activations** (full 50K val set, sequential order for label alignment):

```bash
python -m src.caching.cache_activations \
    --backbone dinov2_vitb14 \
    --use_hf_local \
    --hf_split validation \
    --output_dir /path/to/activations_val/dinov2_vitb14/layer_11/ \
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
    shard_000.pt       # [N, patch_count, 768] float16
    shard_001.pt
    ...
    stats.json          # mean, std, num_images, d_model
```

### Step 2: Train SAE

Train a Sparse Autoencoder on cached activations. Supports **TopK** and **BatchTopK** architectures from the `overcomplete` library.

```bash
python -m src.training.train_sae \
    --backbone dinov2_vitb14 \
    --activation_dir /path/to/activations/dinov2_vitb14/layer_11/ \
    --output_dir /path/to/checkpoints/dinov2_vitb14/batchtopk_16x_k192/ \
    --expansion_factor 16 \
    --k 192 \
    --architecture batchtopk \
    --lr 1e-3 \
    --batch_size 4096 \
    --num_epochs 4 \
    --seed 42
```

**Key training arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--architecture` | `topk` or `batchtopk` | `topk` |
| `--expansion_factor` | Dictionary size = d_model × factor | 16 |
| `--k` | Per-sample sparsity (active features) | 192 |
| `--batch_size` | Patch tokens per batch | 4096 |
| `--num_epochs` | Passes over all shards | 3 |

**Checkpoint output:**

```
checkpoints/{backbone}/batchtopk_16x_k192/
    sae.pt              # model state dict
    config.yaml         # training config snapshot
    training_log.json   # per-step loss + final eval metrics
```

### Step 3: Evaluate

Run individual metrics or all metrics at once.

**M1: FVU**

```python
from src.evaluation.reconstruction.fvu import FVUMetric

metric = FVUMetric(batch_size=512)
results = metric.evaluate(sae, shard_paths, mean, std, device="cuda", dict_size=12288)
# {"fvu": 0.1009, "l0": 208.0, "dead_features": 172, "dead_pct": 1.4}
```

**M3: Sparse Probing**

```bash
python scripts/run_sparse_probing.py \
    --sae_checkpoint /path/to/checkpoints/dinov2_vitb14/batchtopk_16x_k192/sae.pt \
    --sae_config /path/to/checkpoints/dinov2_vitb14/batchtopk_16x_k192/config.yaml \
    --activation_dir /path/to/activations_val/dinov2_vitb14/layer_11/
```

**M7: Feature Absorption**

```bash
python scripts/run_absorption.py \
    --sae_checkpoint /path/to/checkpoints/dinov2_vitb14/batchtopk_16x_k192/sae.pt \
    --sae_config /path/to/checkpoints/dinov2_vitb14/batchtopk_16x_k192/config.yaml \
    --activation_dir /path/to/activations_val/dinov2_vitb14/layer_11/
```

**Run all metrics:**

```bash
bash scripts/run_all_evals.sh
```

### Sweeps

For systematic experiments across backbones, expansion factors, and k values.

```bash
# Preview all commands
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode dry_run

# Run sequentially
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode sequential --skip_existing

# Submit to SLURM cluster
python -m src.training.sweep_runner --sweep_config configs/sweeps/main_sweep.yaml --mode slurm --slurm_partition gpu
```

Sweep configs are defined in `configs/sweeps/`:
- `main_sweep.yaml` — 5 backbones × 3 expansions × 3 k values = 45 runs
- `ablation_sweep.yaml` — TopK + JumpReLU on CLIP & DINOv2
- `seed_sweep.yaml` — 3 seeds on best config per backbone

## Project Structure

```
src/
    backbones/           # ViT backbone adapters (CLIP, DINOv2, SigLIP, MAE, DeiT)
    caching/             # Activation extraction and sharding
    training/            # SAE training and sweep runner
    evaluation/         # Metrics M1–M7 + EvalRunner
        reconstruction/  # M1: FVU, M2: Downstream Preservation
        concept_detection/  # M3: Sparse Probing, M4: Monosemanticity, M5: Cross-Domain
        spatial_coherence/  # M6: Feature Localization
        disentanglement/    # M7: Feature Absorption
    analysis/            # Result aggregation, hypothesis tests, Pareto, visualization
    utils/               # Path constants, IO helpers, logging
configs/
    backbones/           # Per-backbone YAML configs
    sae/                 # SAE architecture configs (BatchTopK, TopK, JumpReLU)
    eval/                # Evaluation metric configs
    sweeps/              # Sweep experiment definitions
scripts/                # CLI entry points and batch scripts
notebooks/              # Exploratory notebooks (backbone shapes, metric design, results)
paper/                  # NeurIPS 2026 submission (main.tex, appendix, references)
```

## Dependencies

- Python 3.11+
- PyTorch 2.6 (CUDA 12.4)
- [overcomplete](https://github.com/nicandris/overcomplete) — SAE architectures (TopK, BatchTopK)
- HuggingFace transformers & datasets
- timm, open-clip-torch
- einops, numpy, pillow, tqdm, pyyaml

See `pyproject.toml` for the full dependency list.
