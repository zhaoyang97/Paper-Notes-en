---
title: >-
  [Paper Note] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT
description: >-
  [CVPR 2026][Video Understanding][referring multi-object tracking] FlexHook revitalizes the two-stage Referring-by-Tracking (RBT) paradigm: it introduces C-Hook to directly sample target features from the backbone (replacing dual encoding) and inject language-conditioned cues, and replaces CLIP cosine similarity with PCD (Pairwise Correspondence Decoder) for active correspondence modeling. This marks the first time a two-stage method comprehensively surpasses one-stage RMOT state-of-the-art — achieving HOTA of 42.53 (vs. 10.32 for iKUN) on Refer-KITTI-V2, with training completed in only 1.91 hours (2×4090).
tags:
  - CVPR 2026
  - Video Understanding
  - referring multi-object tracking
  - two-stage RBT
  - language-conditioned sampling
  - pairwise correspondence
date: 2026-05-08
content_hash: 12c8876b02a5f127
---

# FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT

**Conference**: CVPR 2026
**arXiv**: [2503.07516](https://arxiv.org/abs/2503.07516)
**Code**: [GitHub](https://github.com/buptLwz/FlexHook)
**Area**: Video Understanding
**Keywords**: referring multi-object tracking, two-stage RBT, language-conditioned sampling, pairwise correspondence

## TL;DR
FlexHook revitalizes the two-stage Referring-by-Tracking (RBT) paradigm: it introduces C-Hook to directly sample target features from the backbone (replacing dual encoding) and inject language-conditioned cues, and replaces CLIP cosine similarity with PCD (Pairwise Correspondence Decoder) for active correspondence modeling. This marks the first time a two-stage method comprehensively surpasses one-stage RMOT state-of-the-art — achieving HOTA of 42.53 (vs. 10.32 for iKUN) on Refer-KITTI-V2, with training completed in only 1.91 hours (2×4090).

## Background & Motivation
Referring Multi-Object Tracking (RMOT) tracks multiple targets specified by natural language expressions. Three paradigms exist: (1) TBR (GroundingDINO localization + association); (2) one-stage RBT (end-to-end trajectory queries via MOTR); (3) two-stage RBT (offline tracking followed by referring matching). Two-stage methods offer lower training costs and incremental deployability (upgrading the tracker does not affect the referring module). However, iKUN (CVPR24) achieves only 10.32 HOTA on Refer-KITTI-V2, far behind one-stage methods (35+). The root causes are: (1) heuristic feature construction — dual encoding of the full image and cropped patches wastes computation and fails to exploit the contextual capacity of pretrained backbones; (2) fragile CLIP cosine similarity matching — constrained by the CLIP alignment space, additional modules tend to degrade performance.

## Core Problem
How to substantially improve referring matching accuracy in two-stage RBT while preserving its training efficiency and deployment flexibility?

## Method

### Overall Architecture
Input: trajectory bounding box sequences from an offline tracker + video frames + a set of language expressions. FlexHook encodes the full image with a backbone (once only), then applies C-Hook to directly sample target features and language-conditioned reference features from multi-scale feature maps. After temporal integration, PCD decodes matching scores for each expression–trajectory pair.

### Key Designs
1. **C-Hook (Conditioning Hook)**: Comprises two sub-components — (a) *Neighboring Grid Sampling*: constructs a sampling grid from trajectory bounding boxes and performs bilinear interpolation sampling of target features from backbone feature maps. Three perturbation strategies (random segment masking, Gaussian noise, and intra-batch ID swapping) simulate tracking noise to improve train–inference consistency. (b) *Conditioning Enhancement*: uses a Transformer decoder to decode $M$ reference point coordinates from language features (learnable queries + cross-attention + MLP + sigmoid), and additionally samples language-conditioned features at those locations on the feature map. Different semantic expressions (e.g., "person in red" vs. "person on the left") attend to different spatial regions.

2. **Temporal Integration**: Concatenates multi-frame target features with inter-frame grid coordinate displacements (explicit optical flow) and compresses them via MLP, capturing motion cues (e.g., handling motion-based expressions such as "car turning left").

3. **PCD (Pairwise Correspondence Decoder)**: Learnable query vectors access trajectory features (shared) and the corresponding language features plus reference features (private) simultaneously via masked cross-attention. Attention masks ensure each query attends only to its corresponding language branch, while shared trajectory features enable implicit contrastive learning. Multi-scale decoding is performed via FPN, producing $N$ matching scores.

### Loss & Training
Focal Loss supervises matching scores to handle positive/negative sample imbalance. A reference point boundary regularization loss $L_r$ (softplus barrier) prevents learned reference coordinates from degenerating to boundaries. AdamW with lr=$3\times10^{-5}$, 20 epochs, 2×4090.

## Key Experimental Results

### Refer-KITTI-V2 (Primary Benchmark)

| Method | Paradigm | HOTA | DetA | AssA |
|--------|----------|------|------|------|
| TransRMOT | One-stage | 31.00 | 19.40 | 49.68 |
| HFF-Tracker | One-stage | 36.18 | 24.64 | 53.27 |
| iKUN (CVPR24) | Two-stage | 10.32 | 2.17 | 49.77 |
| **FlexHook-best** | **Two-stage** | **42.53** | **30.63** | **59.19** |

### Training Efficiency (Refer-KITTI-V2)

| Method | Training Time | HOTA |
|--------|--------------|------|
| TempRMOT (one-stage) | 51.68h (60ep) | 35.04 |
| iKUN (two-stage) | 2.46h (100ep) | 10.32 |
| **FlexHook** | **1.91h** (20ep) | **42.53** |

### LaMOT (Large-Scale Multi-Scene)
FlexHook-best: HOTA 56.77 vs. LaMOTer 48.45 (+8.32)

### Ablation Study
- **C-Hook contributes the most**: iKUN → +C-Hook = 34.49 HOTA (+24.17!), identifying feature construction as the core bottleneck of prior two-stage methods.
- **PCD provides additional gains**: +PCD = 38.62 (+4.13), replacing CLIP cosine similarity.
- **Conditioning Enhancement ($M=10$)**: Consistently improves HOTA by 0.5–1.5.
- **Neighboring noise perturbation**: Removing it drops HOTA by 1.3, confirming the importance of simulating tracking noise to reduce the train–inference gap.
- **Frozen encoder**: Freezing all encoders reduces HOTA by only ~1.7 (42.53 → 40.86), enabling deployment under extremely limited resources.
- **CLIP independence**: Using RoBERTa + Swin-T (non-aligned embedding space) outperforms CLIP (42.53 vs. 41.42).

## Highlights & Insights
- **Lightweight design via sampling over re-encoding**: C-Hook directly performs grid sampling from backbone feature maps, avoiding redundant full-image re-encoding and patch cropping, while preserving contextual gradient flow from the pretrained backbone — a remarkably clean design.
- **Data augmentation for tracking noise**: The train–test gap caused by using GT trajectories during training vs. tracker outputs during inference is substantially reduced by introducing synthetic noise (segment deletion, position perturbation, ID swapping), a technique transferable to all two-stage pipelines.
- **Masked attention in PCD**: All expression–trajectory pairs share trajectory features in cross-attention, with language branches isolated via masking — achieving pairwise discrimination while implicitly benefiting from cross-pair contrastive learning.
- **Revival of two-stage methods**: This work is the first to demonstrate that a properly designed two-stage RBT can comprehensively surpass one-stage methods with faster training, offering new perspectives on the necessity of end-to-end learning.

## Limitations & Future Work
- Performance still partially depends on tracker quality (HOTA drops from 42.53 to 40.73 when using weaker tracker D-DETR + StrongSORT).
- The number of reference points $M=10$ is manually selected; adaptive determination of $M$ may be preferable.
- Validation is limited to multi-object tracking scenarios; single-object referring tracking is not evaluated.
- Multi-scale decoding in PCD introduces some additional inference overhead.

## Related Work & Insights
- **vs. iKUN**: Both are two-stage; iKUN achieves 10.32 HOTA with dual encoding + CLIP similarity, while FlexHook achieves 42.53 with C-Hook + PCD — the gap stems from fundamental differences in feature construction and matching strategy.
- **vs. TransRMOT/TempRMOT (one-stage)**: These require 60-epoch end-to-end training (51.68h), whereas FlexHook trains in 1.91h with superior performance.
- **vs. LaMOTer (TBR)**: Uses GroundingDINO for open-set localization with strong open-vocabulary generalization, but achieves lower RMOT performance than FlexHook.

## Relevance to My Research
- The "language-conditioned feature sampling" paradigm is transferable to other vision-language grounding tasks.
- The two-stage vs. one-stage trade-off analysis provides useful reference for multimodal system design.
- PCD's masked pairwise attention is applicable to other one-to-many matching scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ C-Hook's sampling-over-re-encoding design is concise and effective; PCD decouples model selection from CLIP's alignment space.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 4 benchmarks with multiple encoder combinations and detailed ablations (C-Hook / PCD / noise / reference points / frozen encoders / efficiency).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is rigorously argued around two fundamental limitations; figures and tables are clear; the "Make it Strong Again" narrative is coherent and complete.
- **Value**: ⭐⭐⭐⭐ Highly relevant to vision-language object tracking; C-Hook and PCD offer strong design transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)

</div>

<!-- RELATED:END -->
