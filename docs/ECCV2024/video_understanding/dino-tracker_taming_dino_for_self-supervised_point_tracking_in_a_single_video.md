---
title: >-
  [Paper Note] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video
description: >-
  [ECCV 2024][Video Understanding][Point Tracking] This paper proposes DINO-Tracker, which combines the semantic features of pretrained DINOv2 with test-time single-video optimization. Through Delta-DINO residual fine-tuning and multi-source self-supervised losses, it achieves long-range dense point tracking. It reaches state-of-the-art (SOTA) performance among self-supervised methods and is comparable to supervised trackers, particularly outperforming existing methods by a wid…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Point Tracking"
  - "Self-Supervised"
  - "DINO features"
  - "Test-time training"
  - "Long-range occlusion"
date: 2026-05-08
content_hash: c44d60ef453418ff
---

# DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video

**Conference**: ECCV 2024  
**arXiv**: [2403.14548](https://arxiv.org/abs/2403.14548)  
**Code**: [Yes](https://dino-tracker.github.io)  
**Area**: Video Understanding  
**Keywords**: Point Tracking, Self-Supervised, DINO features, Test-time training, Long-range occlusion

## TL;DR

This paper proposes DINO-Tracker, which combines the semantic features of pretrained DINOv2 with test-time single-video optimization. Through Delta-DINO residual fine-tuning and multi-source self-supervised losses, it achieves long-range dense point tracking. It reaches state-of-the-art (SOTA) performance among self-supervised methods and is comparable to supervised trackers, particularly outperforming existing methods by a wide margin in long-term occlusion scenarios.

## Background & Motivation

The two main paradigms for dense point tracking have their respective limitations:

**Supervised feed-forward methods** (TAP-Net/TAPIR/Co-Tracker):
- Limited diversity of synthetic training data, presenting a domain gap with real video distributions.
- Restricted spatiotemporal receptive fields, making it difficult to aggregate information across the entire video.
- Insufficient capability to handle long-term occlusions.

**Test-time optimization methods** (Omnimotion):
- Solely rely on pre-computed optical flow and video reconstruction, without utilizing any external visual priors.
- Performance drops sharply when optical flow is unavailable (e.g., during long-term occlusions).
- Optimization is extremely time-consuming.

Key Insight: **Semantic features of DINOv2 inherently contain fine-grained semantic information.** While raw DINO feature matching already outperforms RAFT and TAP-Net on DAVIS-256, **its discriminative power is insufficient to support sub-pixel accuracy tracking.**

## Method

### Overall Architecture

DINO-Tracker performs end-to-end test-time training on a single input video:
1. Freeze DINOv2 to extract semantic features.
2. Delta-DINO (CNN) predicts feature residuals.
3. Compute the cost volume based on refined features for tracking.
4. Jointly optimize with multi-source self-supervised losses.

### Delta-DINO Residual Feature Refinement

Core design: Predict residuals rather than directly fine-tuning DINO (to better retain priors):

$$\Phi(I) = \Phi_{DINO}(I) + \Phi_{\Delta}(I)$$

- $\Phi_{DINO}$: Token features from the 16th layer of frozen DINOv2-ViT-L/14.
- $\Phi_{\Delta}$: Residuals predicted by the CNN (zero-initialized to stabilize training).
- Inductive bias of CNN: Encodes similar RGB patches into similar features, naturally providing smoothness.

### Tracking Inference Pipeline

Given a query point $x_q$ in frame $I^k$:
1. Bilinearly sample the query feature.
2. Compute the cost volume (cosine similarity) with the target frame.
3. Pass through a CNN refiner + spatial softmax to obtain the heatmap $H$.
4. Final coordinates: Weighted sum of the neighborhood around the maximum value.

### Sources of Self-Supervision

**1. Optical Flow Correspondences**: Short-range sub-pixel correspondences pre-computed by RAFT (filtered by cycle-consistency).
- Advantages: Accurate sub-pixel matching; Disadvantages: Long-range error accumulation.

**2. DINO Best-Buddy Pairs**: Mutual nearest neighbor matches of raw DINO features.
- Advantages: Semantic matching across distant frames; Disadvantages: Coarse spatial resolution.

**3. Refined Best-Buddy Pairs**: Mutual nearest neighbor matches of refined features dynamically updated during training.

Key Complementarity: Optical flow provides sub-pixel accuracy in temporal vicinity, while DINO BB provides semantic correspondences across distant frames.

### Loss & Training

**Flow Loss**: Aligns tracking estimates with optical flow correspondences using Huber loss.

**DINO BB Contrastive Loss**: Weighted InfoNCE, which enhances the similarity of matched features and reduces the similarity of non-matched features.

**Refined BB Contrastive Loss**: Employs the same structure as above, using dynamic BB pairs of refined features.

**Cycle Consistency Loss**: Encourages the trajectories output by the tracker to remain cycle-consistent.

**Prior Preservation Loss**: Constrains both the direction (cos-sim) and magnitude (norm) of the refined features to match the original DINO features.

**Total Loss**: 

$$L = L_{flow} + \lambda_1 L_{dino\text{-}bb} + \lambda_2 L_{rfn\text{-}bb} + \lambda_3 L_{rfn\text{-}cc} + \lambda_4 L_{prior}$$

### Occlusion Prediction

Based on trajectory consistency: re-track backward from the estimated position, checking if the path aligns with the original trajectory on the anchor frame. If the deviation is large and features are dissimilar, it is classified as occluded.

## Key Experimental Results

### Main Results

| Method | Type | DAVIS-256 delta | DAVIS-480 delta/AJ | Kinetics-256 delta/AJ | BADJA seg/3px |
|------|------|-------------|----------------|--------------------|--------------------|
| RAFT | - | 56.7 | 66.7/- | 50.4/- | 45.0/5.8 |
| DINOv2 | - | 61.4 | 64.7/- | 60.3/- | 62.8/8.4 |
| TAP-Net | Supervised | 53.4 | 66.4/46.0 | 61.7/48.5 | 45.4/9.6 |
| TAPIR | Supervised | 74.7 | 77.3/65.7 | 69.5/57.3 | 68.7/10.5 |
| Co-Tracker | Supervised | 79.2 | 79.4/65.6 | 72.9/59.9 | 64.0/11.2 |
| Omnimotion | Test-time | 67.5 | 74.1/58.4 | 69.2/55.0 | 45.2/6.9 |
| **Ours** | **Test-time** | **78.2** | **80.4/64.6** | **73.3/59.7** | **72.4/14.3** |

### Ablation Study

| Ablation Study | DAVIS-480 delta | OA | AJ |
|--------|-------------|-----|-----|
| Full Model | 80.4 | 88.1 | 64.6 |
| w/o DINO | 71.4 | 79.7 | 51.0 |
| LoRA instead of Delta-DINO | 76.0 | 85.1 | 58.8 |
| w/o L_flow | 78.4 | 86.7 | 62.3 |
| w/o L_dino-bb | 78.3 | 87.3 | 62.7 |
| w/o L_rfn | 78.1 | 87.2 | 62.3 |
| w/o L_prior | 78.5 | 86.7 | 62.3 |

### Occlusion Rate Analysis

Grouped by video occlusion rate, the proposed method shows a huge advantage in high-occlusion scenarios (>30%):
- DINO-Tracker leads all competitors significantly in both delta and AJ.
- Omnimotion's performance drops sharply under high occlusion due to its complete reliance on optical flow.

### Key Findings

- **Raw DINOv2 is already a strong baseline**: Outperforming RAFT and TAP-Net on DAVIS-256 (61.4 vs 56.7/53.4).
- The DINO prior makes a huge contribution: Without DINO, delta drops from 80.4 to 71.4, and AJ drops from 64.6 to 51.0.
- Delta-DINO (CNN residual) is significantly better than LoRA fine-tuning: Sharper heatmaps, less jitter.
- Removing L_flow leads to only a 2% drop in positional accuracy, indicating that the DINO prior + self-distillation already provides most of the tracking signal.
- Significant SOTA on BADJA: seg 72.4 (vs Co-Tracker 64.0), 3px 14.3 (vs Co-Tracker 11.2).

## Highlights & Insights

1. **First to utilize DINO for dense point tracking**: Discovered that pretrained visual features can directly assist motion estimation tasks.
2. **Elegant fusion of test-time training and external priors**: Bypassed the separate limitations of "zero data" and "zero adaptation".
3. **Highly convincing t-SNE visualizations**: Raw DINO features are scattered and intertwined along the trajectories, whereas refined features form tight "trajectory clusters."
4. **Fundamental breakthrough in handling occlusions**: Leveraged semantic priors to associate points across occlusions instead of relying solely on optical flow propagation.
5. **Clever design of prior preservation regularization**: Constraining both the direction and magnitude.

## Limitations & Future Work

- Test-time optimization still takes time (though much faster than Omnimotion, it is slower than feed-forward methods), making it unsuitable for real-time applications.
- OA (occlusion accuracy) is slightly lower than supervised approaches (88.1 vs TAPIR 89.5).
- Only used the 16th layer of DINOv2-ViT-L/14, without exploring dynamic multi-layer feature fusion.
- Did not incorporate 3D priors (e.g., depth estimation). The 3D improvement strategy from Omnimotion is worth exploring.

## Related Work & Insights

- The core innovation comes from combining the test-time optimization concept of Omnimotion with external DINO priors.
- While Time-tuning applies DINO for temporal consistency in video segmentation, this work pushes it further to sub-pixel tracking.
- The "best-buddy" matching of DINO features provides annotation-free, long-range correspondences.
- The approach of residual learning + zero initialization is inspired by works like ControlNet, proving equally effective in feature refinement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Pioneered the paradigm of combining DINO and test-time optimization for point tracking)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three benchmarks + occlusion rate analysis + thorough ablation studies + DINO layer selection + LoRA comparison)
- Writing Quality: ⭐⭐⭐⭐⭐ (Beautifully illustrated, intuitive t-SNE visualization and trajectory consistency diagrams)
- Value: ⭐⭐⭐⭐⭐ (Groundbreaking work with a substantial breakthrough in long-term occlusion tracking)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)
- [\[ECCV 2024\] Local All-Pair Correspondence for Point Tracking](local_all-pair_correspondence_for_point_tracking.md)
- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[CVPR 2026\] Boosting Self-Supervised Tracking with Contextual Prompts and Noise Learning](../../CVPR2026/video_understanding/boosting_self-supervised_tracking_with_contextual_prompts_and_noise_learning.md)
- [\[CVPR 2026\] Generative Point Tracking and Forecasting](../../CVPR2026/video_understanding/generative_point_tracking_and_forecasting.md)

</div>

<!-- RELATED:END -->
