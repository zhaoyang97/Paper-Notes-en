---
title: >-
  [Paper Note] TrackMAE: Video Representation Learning via Track, Mask, and Predict
description: >-
  [CVPR 2026][Self-Supervised Learning][Masked Video Modeling] This paper introduces explicit motion signals into the masked video modeling (MVM) framework. Point trajectories extracted via CoTracker3 serve as auxiliary reconstruction targets, complemented by a motion-aware masking strategy. The model jointly learns spatial reconstruction and motion prediction, achieving substantial gains over existing video self-supervised methods on motion-sensitive benchmarks (SSv2, FineGym).
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Masked Video Modeling
  - Point Tracking
  - Motion Prediction
  - Self-Supervised Pretraining
  - Video Representation
date: 2026-05-08
content_hash: 100920ca93bb2da4
---

# TrackMAE: Video Representation Learning via Track, Mask, and Predict

**Conference**: CVPR 2026
**arXiv**: [2603.27268](https://arxiv.org/abs/2603.27268)
**Code**: [https://github.com/rvandeghen/TrackMAE](https://github.com/rvandeghen/TrackMAE)
**Area**: Self-Supervised Learning / Video Understanding
**Keywords**: Masked Video Modeling, Point Tracking, Motion Prediction, Self-Supervised Pretraining, Video Representation

## TL;DR

This paper introduces explicit motion signals into the masked video modeling (MVM) framework. Point trajectories extracted via CoTracker3 serve as auxiliary reconstruction targets, complemented by a motion-aware masking strategy. The model jointly learns spatial reconstruction and motion prediction, achieving substantial gains over existing video self-supervised methods on motion-sensitive benchmarks (SSv2, FineGym).

## Background & Motivation

Masked Video Modeling (MVM) has emerged as a concise and effective paradigm for video self-supervised pretraining—masking 80–95% of spatiotemporal tokens and reconstructing the visible portion. However, existing MVM methods suffer from a fundamental limitation:

**Implicit motion encoding**: Pixel reconstruction objectives tend to capture low-level appearance statistics (color/texture continuity) rather than temporal dynamics. Due to the strong temporal redundancy in video, pixel reconstruction can often be solved via spatial correlations or short-range consistency, creating a "shortcut" that bypasses motion learning.

**Poor performance on motion-sensitive tasks**: MVM methods perform well on appearance-dominated datasets (K400, UCF101) but fall notably short on SSv2 and FineGym, which require fine-grained temporal modeling.

**Limitations of prior improvements**:
- Improved masking strategies (e.g., flow-based masking): motion information is incorporated only implicitly.
- Improved reconstruction targets (e.g., HOG, DINO, CLIP features): encode high-level semantics but do not explicitly model motion.
- MME uses trajectory signals but relies on precomputed optical flow, requires complex preprocessing, and is sensitive to camera motion.

The paper's central claim is that **temporal correspondence should be treated as a first-class signal in pretraining**, complementary to—rather than competing with—pixel/feature targets.

## Method

### Overall Architecture

TrackMAE (Figure 2) extends standard MVM with two additional components:

- **Motion target extraction**: CoTracker3 extracts sparse point trajectories from video.
- **Dual-decoder architecture**: a shared encoder paired with a spatial decoder $\Psi_{spatial}$ and a motion decoder $\Psi_{motion}$.
- **Motion-aware masking**: trajectory displacements are used to construct a sampling distribution.

### Key Designs

1. **Motion target extraction and upsampling**: introducing point tracking into self-supervised learning.

    - A uniform grid of query points is sampled from the first frame ($G = H/p$, one point per patch center) and tracked to subsequent frames using CoTracker3.
    - Output shape matches video tokens: $T/2 \times H/p \times W/p \times 2$ (predicting displacements rather than absolute positions).
    - **Upsampling trick**: assuming that nearby pixels within a patch undergo similar motion, sparse trajectories are spatially interpolated to produce denser motion targets (14→28, $\upsilon=2$), equivalent to tracking 4 points per patch at no additional computational cost.
    - Design motivation: the cost of dense tracking scales with query grid size; upsampling yields +1.7%/+1.9% performance gains at zero overhead.

2. **Motion prediction loss**: trajectory reconstruction as an auxiliary self-supervised objective.

    - The motion decoder $\Psi_{motion}$ predicts point trajectory displacements at masked positions.
    - Loss is computed only over masked tokens: $\mathcal{L}_{motion} = \frac{1}{|\mathcal{T}^{masked}|} \sum_{i \in \mathcal{T}^{masked}} \|\mathbf{m}_i - \hat{\mathbf{m}}_i\|_2^2$
    - Final objective: $\mathcal{L} = \mathcal{L}_{spatial} + \lambda \cdot \mathcal{L}_{motion}$
    - $\lambda=1$ for pixel reconstruction; $\lambda=0.25$ for CLIP feature reconstruction (due to larger gradient magnitudes from feature targets).
    - Key finding: motion targets and CLIP features are **highly complementary**—CLIP encodes "what is there," while trajectories encode "how it moves."

3. **Motion-aware masking strategy**: improving random tube masking using trajectory information.

    - The average displacement $\bar{\mathbf{M}}$ of each query point over time serves as a sampling distribution.
    - All positions are partitioned into high-motion and low-motion bins.
    - A motion ratio $\rho_{motion}$ controls the number of visible tokens sampled uniformly from each bin.
    - Default $\rho_{motion}=50\%$ (equal-ratio sampling is optimal); skewing toward either bin yields slight degradation.
    - Compared to random tube masking, this strategy provides a consistent ~0.5% improvement at no additional cost, since trajectories are already extracted.

### Loss & Training

- Encoder: ViT-B / ViT-L
- Pretraining data: Kinetics-400 (K700 for ViT-L)
- Pretraining: 800 epochs, following VideoMAE hyperparameter settings
- CoTracker3: offline mode, 14×14 grid, upsampling factor $\upsilon=2$
- Feature reconstruction targets extracted using CLIP ViT-B
- Downstream evaluation: both linear probing and full fine-tuning protocols

## Key Experimental Results

### Main Results

**Linear Probing** (Table 1, ViT-B, K400 pretrained)

| Method | Target | K400 | HMDB | SSv2 | GYM |
|--------|--------|------|------|------|-----|
| VideoMAE | Pixel | 20.7 | 37.7 | 17.5 | 23.9 |
| MGMAE | Pixel | 24.9 | 41.3 | 16.8 | 26.1 |
| MGM | Pixel | 19.8 | 40.3 | 21.7 | 25.8 |
| **TrackMAE** | **Pixel** | **25.7** | **40.6** | **23.6** | **29.0** |
| SIGMA | DINO | 47.5 | 52.3 | 20.8 | 30.1 |
| SMILE | CLIP | 56.2 | 53.4 | 23.7 | 30.2 |
| **TrackMAE** | **CLIP** | **55.2** | **53.1** | **27.3** | **31.8** |

TrackMAE achieves substantial margins on motion-sensitive tasks (SSv2, GYM).

**Full Fine-tuning** (Table 2)

| Method | Backbone | Target | SSv2 | K400 |
|--------|----------|--------|------|------|
| VideoMAE | ViT-B | Pixel | 68.5 | 80.0 |
| SMILE | ViT-B | CLIP | 72.1 | 83.1 |
| **TrackMAE** | **ViT-B** | **CLIP** | **72.8** | **83.9** |
| VideoMAE | ViT-L | Pixel | 74.0 | 85.2 |
| **TrackMAE** | **ViT-L** | **CLIP** | **75.7** | **86.7** |

### Ablation Study

**Reconstruction target combinations** (Table 3, ViT-S)

| Target | K400s | SSv2s | Notes |
|--------|-------|-------|-------|
| Trajectory only | 46.5 | 53.1 | Trajectories alone are a strong signal |
| Pixel only | 46.0 | 52.2 | Baseline |
| Pixel + Trajectory | 48.9 | 55.7 | Complementary gain: +2.9/+3.5 |
| CLIP only | 52.7 | 57.1 | Semantic features are stronger |
| **CLIP + Trajectory** | **55.8** | **61.1** | **Complementary gain: +3.1/+4.0** |

**Upsampling effect** (Table 5)

| Grid size | Upsampling | K400s | SSv2s |
|-----------|------------|-------|-------|
| 14×14 | None | 48.9 | 55.7 |
| 28×28 | None | 49.5 | 56.7 |
| 56×56 | None | 50.0 | 57.0 |
| 14×14 | 14→28 (υ=2) | **50.6** | **57.6** |

Upsampling (14→28) even outperforms direct 56×56 dense tracking at zero additional cost.

### Key Findings

- Trajectory prediction alone is already an effective self-supervised objective (46.5 on K400s) and can independently encode useful video representations.
- The complementarity between motion trajectories and CLIP features is significantly stronger than that with pixel targets (+4.0 vs. +3.5 on SSv2s), since CLIP encodes "what is there" while trajectories encode "how it moves."
- Equal-ratio sampling across high/low motion regions ($\rho=50\%$) is optimal; skewing toward either regime yields minor degradation.
- The upsampling trick achieves performance close to 4× denser tracking at zero cost, validating the assumption of intra-patch motion smoothness.
- TrackMAE demonstrates favorable scaling properties at ViT-L scale (SSv2: 75.7%, K400: 86.7%).

## Highlights & Insights

1. **Motion as a first-class citizen**: Unlike MME, which indirectly constructs trajectory signals from optical flow, TrackMAE directly leverages a modern point tracker (CoTracker3) to extract high-quality trajectories, avoiding the complex preprocessing and camera-motion sensitivity associated with precomputed optical flow.
2. **Simple yet effective design**: The method adds only a lightweight trajectory decoder and a motion-aware masking strategy, leaving the encoder architecture unchanged.
3. **Elegance of the upsampling trick**: By exploiting the spatial smoothness assumption, sparse trajectories are densified at zero cost, with performance even surpassing true dense tracking.
4. **Complementarity analysis of feature targets**: CLIP and trajectory targets exhibit the strongest complementarity (+4.0%), providing compelling evidence for jointly learning high-level semantics and motion signals.
5. **Fully online extraction**: Trajectories are extracted online from RGB video (as opposed to MME's precomputed optical flow), simplifying the training pipeline.

## Limitations & Future Work

- Computational overhead of CoTracker3: despite using offline mode and a sparse grid, training time is still increased.
- Validation is limited to ViT-B/L; scaling behavior to larger models (e.g., ViT-H/Giant) and larger datasets remains unexplored.
- The gains from motion-aware masking are modest (~0.5%), suggesting that more sophisticated sampling strategies may be warranted.
- Initializing query points from the first frame may miss object motion that first appears in later frames.
- The transferability of trajectory prediction to other downstream tasks (e.g., video object tracking, action localization) is not explored.

## Related Work & Insights

- **vs. SMILE**: SMILE injects motion awareness through synthetic motion (copy-paste with random paths), whereas TrackMAE uses genuine pixel-level motion signals derived from real trajectories.
- **vs. Tracktention**: Tracktention injects trajectories into attention layers to achieve temporally consistent features; TrackMAE uses trajectories as reconstruction targets to learn motion semantics.
- **Broader trend of CoTracker3 adoption**: point trackers are increasingly used as general-purpose tools in video understanding, spanning attention routing, dense feature learning, and self-supervised pretraining.
- **Generalizable principle**: any "free" signal from a pretrained model can potentially serve as an auxiliary prediction target in MVM pretraining.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing point trajectories into MVM pretraining is intuitive and well-motivated, though the core idea is relatively natural.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 6 benchmarks, both linear probing and fine-tuning, comprehensive ablations, and two model scales (ViT-B/L).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-analyzed motivation, and fair comparisons.
- **Value**: ⭐⭐⭐⭐ — Improvements on motion-sensitive benchmarks are substantive; CoTracker3 overhead remains the primary practical concern.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[AAAI 2026\] From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models without Finetuning](../../AAAI2026/self_supervised/from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition](sphor_a_representation_learning_perspective_on_ope.md)

</div>

<!-- RELATED:END -->
