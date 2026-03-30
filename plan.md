visaebench-internal/
│
├── README.md                              # Internal notes, not for publication
├── .env                                   # API keys: HuggingFace, wandb, etc.
├── .gitignore                             # Ignore data/, activations/, checkpoints/, wandb/
├── pyproject.toml                         # Dependencies: overcomplete, prisma-lens, timm, transformers, open-clip-torch, wandb, etc.
│
│
│ ═══════════════════════════════════════════
│  CONFIGS
│ ═══════════════════════════════════════════
│
├── configs/
│   │
│   ├── backbones/                         # One YAML per backbone — defines loading + extraction behavior
│   │   ├── clip_vitb16.yaml               #   model_id, source (open_clip / hf), layer: 11, patch_count: 196, has_cls: true, special_tokens: [cls], drop_special_tokens: true
│   │   ├── dinov2_vitb14.yaml             #   source: torch_hub, layer: 11, patch_count: 256, has_cls: true, has_registers: true, special_tokens: [cls, reg0..reg3], drop_special_tokens: true
│   │   ├── siglip_vitb16.yaml             #   source: hf, layer: 11, patch_count: 196, has_cls: false, special_tokens: [], drop_special_tokens: false
│   │   ├── mae_vitb16.yaml                #   source: timm, layer: 11, patch_count: 196, has_cls: true, special_tokens: [cls], drop_special_tokens: true, note: "remove reconstruction head before extraction"
│   │   └── deit_vitb16.yaml               #   source: timm, layer: 11, patch_count: 196, has_cls: true, has_dist_token: true, special_tokens: [cls, dist], drop_special_tokens: true
│   │
│   ├── sae/                               # SAE architecture configs — BatchTopK primary, others are ablations
│   │   ├── batchtopk_8x.yaml              #   architecture: batchtopk, expansion_factor: 8, k_values: [128, 192, 256], lr, batch_size, num_steps, dead_feature_threshold, etc.
│   │   ├── batchtopk_16x.yaml             #   expansion_factor: 16, k_values: [128, 192, 256]
│   │   ├── batchtopk_32x.yaml             #   expansion_factor: 32, k_values: [128, 192, 256]
│   │   ├── topk_16x.yaml                  #   [ABLATION] architecture: topk, expansion_factor: 16, k_values: [128, 192, 256]
│   │   └── jumprelu_16x.yaml              #   [ABLATION] architecture: jumprelu, expansion_factor: 16
│   │
│   ├── eval/
│   │   ├── reconstruction.yaml            #   datasets: [imagenet_val], metrics: [fvu, downstream_preservation], downstream_probe_datasets: [imagenet, dtd]
│   │   ├── concept_detection.yaml         #   metrics: [sparse_probing, monosemanticity, cross_domain], cross_model_embeddings: {clip: dinov2, dinov2: clip, siglip: dinov2, mae: clip, deit: clip}, probe_datasets: [imagenet], ood_datasets: [inaturalist, eurosat]
│   │   ├── spatial_coherence.yaml         #   metrics: [feature_localization], segmentation_dataset: imagenet_segmentation, iou_thresholds: [0.25, 0.5]
│   │   └── disentanglement.yaml           #   metrics: [feature_absorption], reference: "Karvonen et al. SAEBench absorption protocol adapted for vision"
│   │
│   └── sweeps/
│       ├── main_sweep.yaml                #   5 backbones × 3 expansions × 3 K values = 45 runs, all BatchTopK, all layer 11
│       ├── ablation_sweep.yaml            #   TopK + JumpReLU on CLIP + DINOv2 at 16× expansion = ~6–12 runs
│       └── seed_sweep.yaml                #   3 seeds on best config per backbone = 15 runs
│
│
│ ═══════════════════════════════════════════
│  SOURCE CODE
│ ═══════════════════════════════════════════
│
├── src/
│   ├── __init__.py
│   │
│   ├── backbones/                         # Thin adapters — NOT a parallel extraction framework
│   │   ├── __init__.py                    #   Exports: load_backbone(config) → model, transform
│   │   ├── base.py                        #   BackboneAdapter ABC: load_model(), extract_layer(images, layer_idx) → Tensor, get_patch_tokens(hidden_states) → Tensor (drops special tokens per config)
│   │   ├── clip_adapter.py                #   Loads via open_clip or HF transformers. Extracts layer 11. Drops CLS token.
│   │   ├── dinov2_adapter.py              #   Loads via torch.hub. Extracts layer 11. Drops CLS + register tokens.
│   │   ├── siglip_adapter.py              #   Loads via HF transformers. Extracts layer 11. No CLS to drop — all tokens are patch tokens.
│   │   ├── mae_adapter.py                 #   Loads via timm. Removes reconstruction head. Extracts layer 11. Drops CLS token.
│   │   ├── deit_adapter.py                #   Loads via timm. Extracts layer 11. Drops CLS + distillation token.
│   │   └── token_utils.py                 #   Shared logic: drop_special_tokens(hidden_states, config) → patch_only_tensor. Validates output shape matches expected patch_count.
│   │
│   ├── caching/                           # Activation caching pipeline
│   │   ├── __init__.py
│   │   ├── cache_activations.py           #   Main script: backbone_config + dataset_path + output_dir → sharded .pt files. Uses BackboneAdapter. Computes and saves mean/std stats. Args: --backbone, --dataset, --num_images, --shard_size, --output_dir
│   │   ├── dataset.py                     #   Wraps ImageNet/other datasets. Applies backbone-specific transform from adapter. Returns (image_tensor, image_id).
│   │   └── shard_utils.py                 #   Write/read sharded .pt activations. Compute running mean/std. Save stats.json.
│   │
│   ├── training/                          # SAE training — thin wrappers around Overcomplete
│   │   ├── __init__.py
│   │   ├── train_sae.py                   #   Main entry: --backbone_config --sae_config --k_value --activation_dir --output_dir. Loads cached activations, constructs Overcomplete SAE, trains, saves checkpoint + config + training_log.json.
│   │   ├── overcomplete_config.py         #   Translates our YAML configs → Overcomplete's expected format. Maps expansion_factor + d_model → dict_size. Sets BatchTopK k parameter.
│   │   └── sweep_runner.py                #   Reads sweep YAML → generates (backbone, sae, k) tuples → launches train_sae.py for each. Supports SLURM array jobs or sequential local runs.
│   │
│   ├── evaluation/                        # All 7 metrics — this becomes the public repo nucleus
│   │   ├── __init__.py
│   │   ├── base.py                        #   MetricBase ABC: compute(sae, backbone_adapter, dataset) → dict. Standard interface for all metrics.
│   │   │
│   │   ├── reconstruction/                #   Capability Dimension 1: Reconstruction Quality
│   │   │   ├── __init__.py
│   │   │   ├── fvu.py                     #   M1: Fraction of Variance Unexplained. Compares original activations vs SAE reconstructed activations on held-out val set.
│   │   │   └── downstream_preservation.py #   M2: Train linear probe on original activations → accuracy_orig. Train on SAE-reconstructed activations → accuracy_recon. Report gap.
│   │   │
│   │   ├── concept_detection/             #   Capability Dimension 2: Concept Detection
│   │   │   ├── __init__.py
│   │   │   ├── sparse_probing.py          #   M3: Train k-sparse linear probes (k=1,2,4,8,16) on SAE features for class prediction. Report accuracy curve over k. Higher = more interpretable features.
│   │   │   ├── monosemanticity.py         #   M4: For each SAE feature, find top-activating images, embed with CROSS-MODEL embeddings (e.g., DINOv2 embeds for CLIP SAE), compute pairwise cosine sim. Accepts --embedding_model parameter. Must not use same backbone as the SAE being evaluated.
│   │   │   └── cross_domain.py            #   M5: Evaluate SAE feature quality on OOD datasets (iNaturalist, EuroSAT). Same sparse probing protocol but on distribution-shifted data.
│   │   │
│   │   ├── spatial_coherence/             #   Capability Dimension 3: Spatial Coherence (NOVEL)
│   │   │   ├── __init__.py
│   │   │   └── localization.py            #   M6: Feature localization score. For features that activate on object parts, measure spatial coherence of activation pattern against ground-truth segmentation masks. Most time-intensive metric to implement.
│   │   │
│   │   ├── disentanglement/               #   Capability Dimension 4: Disentanglement
│   │   │   ├── __init__.py
│   │   │   └── absorption.py              #   M7: Feature absorption rate. Adapted from SAEBench (Karvonen et al.) absorption protocol. Measures whether SAE features absorb multiple concepts into single features.
│   │   │
│   │   └── runner.py                      #   EvalRunner: takes SAE checkpoint + backbone config + eval configs → runs all requested metrics → returns unified results dict. Handles cross-model embedding routing for M4.
│   │
│   ├── analysis/                          # Post-evaluation analysis and paper figure generation
│   │   ├── __init__.py
│   │   ├── aggregate_results.py           #   Collect per-SAE result JSONs → master CSV (all_results.csv). One row per SAE, one column per metric.
│   │   ├── hypothesis_tests.py            #   Statistical tests for H1–H5. Paired comparisons, bootstrap CIs, effect sizes. Outputs hypothesis_tests.json.
│   │   ├── pareto.py                      #   Pareto frontier computation: FVU vs each capability metric. This is a VISUALIZATION TOOL, not a metric.
│   │   ├── radar_charts.py                #   Per-backbone capability radar charts (one axis per capability dimension).
│   │   ├── tables.py                      #   Generate LaTeX tables from all_results.csv. Main table + per-dimension breakdown tables.
│   │   └── feature_viz.py                 #   Qualitative visualization: top-activating image patches for selected SAE features. For paper figures.
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                          #   Checkpoint save/load (SAE weights + config). Results serialization (JSON/CSV).
│       ├── logging.py                     #   Wandb integration + CSV fallback logging.
│       └── constants.py                   #   BACKBONE_NAMES, BACKBONE_PATCH_COUNTS = {"dinov2_vitb14": 256, "clip_vitb16": 196, ...}, BACKBONE_D_MODEL = 768, DATASET_PATHS, DEFAULT_LAYER = 11, K_VALUES = [128, 192, 256], EXPANSION_FACTORS = [8, 16, 32]
│
│
│ ═══════════════════════════════════════════
│  SCRIPTS
│ ═══════════════════════════════════════════
│
├── scripts/
│   ├── cache_all_backbones.sh             #   Loop over 5 backbones, call cache_activations.py for each
│   ├── run_main_sweep.sh                  #   Launch 45 BatchTopK training runs (SLURM or sequential)
│   ├── run_ablation_sweep.sh              #   Launch TopK + JumpReLU ablation runs on CLIP + DINOv2
│   ├── run_seed_sweep.sh                  #   Launch 3-seed runs on best config per backbone
│   ├── run_all_evals.sh                   #   Run all 7 metrics on all trained SAEs
│   ├── generate_all_figures.py            #   Read results/ → produce all paper figures as PDF/SVG
│   ├── generate_all_tables.py             #   Read results/ → produce all LaTeX tables
│   └── upload_to_hf.py                    #   Upload trained SAE checkpoints + configs to HuggingFace Hub
│
│
│ ═══════════════════════════════════════════
│  NOTEBOOKS (exploration & debugging)
│ ═══════════════════════════════════════════
│
├── notebooks/
│   ├── 01_backbone_activation_shapes.ipynb     #   START HERE. Load all 5 backbones. Print exact tensor shapes. Verify patch token extraction and special token dropping.
│   ├── 02_pilot_sae_training.ipynb             #   Train 1 BatchTopK SAE on DINOv2 10K-image cache. Check FVU, L0, dead features.
│   ├── 03_metric_debugging.ipynb               #   Test FVU, sparse probing, monosemanticity on pilot SAE.
│   ├── 04_monosemanticity_cross_model.ipynb    #   Validate cross-model embedding setup: score DINOv2 SAE with CLIP embeddings and vice versa. Compare against Pach et al. benchmarks.
│   ├── 05_localization_score_design.ipynb      #   Design, prototype, and validate M6 feature localization score. Ground-truth segmentation mask pipeline.
│   ├── 06_absorption_rate_design.ipynb         #   Design and test M7 absorption rate adapted from SAEBench.
│   ├── 07_results_exploration.ipynb            #   Explore full results after sweep. Identify patterns, outliers, strongest findings.
│   └── 08_qualitative_features.ipynb           #   Cherry-pick compelling feature examples for paper figures.
│
│
│ ═══════════════════════════════════════════
│  DATA & ARTIFACTS (all .gitignored)
│ ═══════════════════════════════════════════
│
├── data/                                  #   NOT in git
│   ├── imagenet/                          #   ImageNet-1K train + val
│   ├── imagenet_segmentation/             #   ImageNet-S or equivalent — needed for M6 localization
│   ├── inaturalist/                       #   iNaturalist 2021 mini — OOD eval
│   ├── eurosat/                           #   EuroSAT — OOD eval
│   └── dtd/                              #   Describable Textures Dataset — downstream probe
│
├── activations/                           #   NOT in git — ~300GB–2TB
│   ├── clip_vitb16/
│   │   └── layer_11/
│   │       ├── shard_000.pt              #   Each shard: Tensor[num_images_in_shard, 196, 768] (patch tokens only, no CLS)
│   │       ├── shard_001.pt
│   │       ├── ...
│   │       └── stats.json                #   {"mean": [768-dim vector], "std": scalar, "num_images": N, "patch_count": 196, "d_model": 768}
│   ├── dinov2_vitb14/
│   │   └── layer_11/
│   │       ├── shard_000.pt              #   Tensor[num_images, 256, 768] — note 256 patches due to /14 patch size
│   │       └── stats.json
│   ├── siglip_vitb16/
│   │   └── layer_11/
│   ├── mae_vitb16/
│   │   └── layer_11/
│   └── deit_vitb16/
│       └── layer_11/
│
├── checkpoints/                           #   NOT in git
│   ├── clip_vitb16/
│   │   ├── batchtopk_8x_k128/
│   │   │   ├── sae.pt                    #   Trained SAE weights
│   │   │   ├── config.yaml               #   Exact config snapshot used for this run
│   │   │   └── training_log.json         #   Loss curve, L0 over training, dead feature count, wall time
│   │   ├── batchtopk_8x_k192/
│   │   ├── batchtopk_8x_k256/
│   │   ├── batchtopk_16x_k128/
│   │   ├── batchtopk_16x_k192/
│   │   ├── batchtopk_16x_k256/
│   │   ├── batchtopk_32x_k128/
│   │   ├── batchtopk_32x_k192/
│   │   └── batchtopk_32x_k256/
│   ├── dinov2_vitb14/
│   │   └── ...                           #   Same 9 configs
│   ├── siglip_vitb16/
│   │   └── ...
│   ├── mae_vitb16/
│   │   └── ...
│   ├── deit_vitb16/
│   │   └── ...
│   ├── ablations/
│   │   ├── clip_topk_16x_k128/
│   │   ├── clip_topk_16x_k192/
│   │   ├── clip_topk_16x_k256/
│   │   ├── clip_jumprelu_16x/
│   │   ├── dinov2_topk_16x_k128/
│   │   ├── dinov2_topk_16x_k192/
│   │   ├── dinov2_topk_16x_k256/
│   │   └── dinov2_jumprelu_16x/
│   └── seeds/
│       ├── clip_best_seed0/
│       ├── clip_best_seed1/
│       ├── clip_best_seed2/
│       └── ...                           #   3 seeds × 5 backbones = 15 runs
│
├── results/                               #   IN git — small JSON/CSV files
│   ├── raw/                               #   One JSON per SAE with all 7 metric scores
│   │   ├── clip_vitb16_batchtopk_8x_k128.json
│   │   ├── clip_vitb16_batchtopk_8x_k192.json
│   │   ├── clip_vitb16_batchtopk_8x_k256.json
│   │   └── ...                            #   45 core + ~12 ablation + 15 seed = ~72 files
│   ├── aggregated/
│   │   ├── all_results.csv                #   Master table: one row per SAE, columns for backbone, expansion, k, M1–M7
│   │   ├── pareto_data.csv                #   FVU vs each capability metric, for Pareto frontier plots
│   │   └── hypothesis_tests.json          #   H1–H5 test results: effect sizes, p-values, CIs
│   └── figures/
│       ├── fig1_overview.pdf              #   Method overview / pipeline diagram
│       ├── fig2_pareto.pdf                #   Pareto frontiers: FVU vs capability metrics
│       ├── fig3_monosemanticity.pdf        #   M4 results across backbones
│       ├── fig4_radar.pdf                 #   Radar charts per backbone
│       ├── fig5_localization.pdf          #   M6 spatial coherence results + qualitative examples
│       ├── fig6_qualitative.pdf           #   Cherry-picked feature visualizations
│       └── ...
│
│
│ ═══════════════════════════════════════════
│  PAPER
│ ═══════════════════════════════════════════
│
├── paper/
│   ├── main.tex
│   ├── appendix.tex
│   ├── references.bib
│   ├── neurips_2026.sty
│   ├── figures/                           #   Copy or symlink from results/figures/
│   └── tables/                            #   Auto-generated .tex files from scripts/generate_all_tables.py
│
│
│ ═══════════════════════════════════════════
│  IGNORED BY GIT
│ ═══════════════════════════════════════════
│
└── wandb/                                 #   NOT in git