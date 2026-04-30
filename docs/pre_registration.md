Final Task D predictions, written today (2026-04-29) before we compute the correlation matrix:

1. M6 → segmentation mIoU: positive, expect ρ > 0.7
2. M4 → CUB accuracy: positive, expect ρ > 0.5
3. M2 → retrieval R@1: positive, expect ρ > 0.5
4. M3 → iNat sparse probing: near-tautological positive (sanity check)
5. Codes-lift inversely correlates with raw task performance across all 4 tasks
6. Backbone ranking on seg mIoU: DINOv2 > DeiT > CLIP > SigLIP > MAE