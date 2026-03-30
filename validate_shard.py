import torch, json

# Check shard shapes
s0 = torch.load("./activations/dinov2_vitb14/layer_11/shard_000.pt")
s1 = torch.load("./activations/dinov2_vitb14/layer_11/shard_001.pt")
print(s0.shape)  # should be [5056, 256, 768] or [5000, 256, 768]
print(s1.shape)  # remaining images

# Check stats
with open("./activations/dinov2_vitb14/layer_11/stats.json") as f:
    stats = json.load(f)
print(f"mean range: [{min(stats['mean']):.4f}, {max(stats['mean']):.4f}]")
print(f"std: {stats['std']:.4f}")
print(f"num_images: {stats['num_images']}, patches: {stats['patch_count']}")
