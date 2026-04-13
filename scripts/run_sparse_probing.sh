set -e
source /mnt/NAS/home/ds5725/visaebench-internal/.visaebench/bin/activate
cd /mnt/NAS/home/ds5725/visaebench-internal
export PYTHONPATH=/mnt/NAS/home/ds5725/visaebench-internal:$PYTHONPATH

python3 scripts/run_sparse_probing.py \
    --sae_checkpoint /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192/sae.pt \
    --sae_config /mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192/config.yaml \
    --activation_dir /mnt/NAS/data/ds5725/visaebench/activations_val/dinov2_vitb14/layer_11/ \
    --output_path /mnt/NAS/home/ds5725/visaebench-internal/results/raw/dinov2_vitb14_batchtopk_16x_k192_sparse_probing.json