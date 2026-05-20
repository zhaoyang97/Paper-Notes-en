---
title: >-
  [Paper Note] Online Segment Any 3D Thing as Instance Tracking
description: >-
  [NeurIPS 2025][3D Vision][Online 3D instance segmentation] AutoSeg3D reformulates online 3D instance segmentation as an instance tracking problem, leveraging long-term memory for cross-frame instance association…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Online 3D instance segmentation"
  - "instance tracking"
  - "visual foundation models"
  - "temporal modeling"
  - "spatial consistency"
date: 2026-05-08
content_hash: 438d3eb30449feac
---

# Online Segment Any 3D Thing as Instance Tracking

**Conference**: NeurIPS 2025
**arXiv**: [2512.07599](https://arxiv.org/abs/2512.07599)  
**Code**: [https://github.com/AutoLab-SAI-SJTU/AutoSeg3D](https://github.com/AutoLab-SAI-SJTU/AutoSeg3D)  
**Area**: 3D Vision
**Keywords**: Online 3D instance segmentation, instance tracking, visual foundation models, temporal modeling, spatial consistency

## TL;DR

AutoSeg3D reformulates online 3D instance segmentation as an instance tracking problem, leveraging long-term memory for cross-frame instance association, short-term memory for instance update, and spatial consistency learning to mitigate VFM over-segmentation. The method surpasses ESAM by 2.8 AP on ScanNet200 while maintaining real-time performance.

## Background & Motivation

Online, real-time, fine-grained 3D instance segmentation is a foundational capability for embodied agents to perceive and understand their environments. Recent methods exploit visual foundation models (VFMs) such as SAM to predict 2D segmentation results, which are then lifted to 3D superpoint representations via depth information. However, existing pipelines suffer from two core issues:

**Lack of temporal modeling**: Current methods naively concatenate global point features across frames, ignoring instance-level temporal reasoning, which prevents the over-segmentation artifacts produced by VFMs from being corrected in the temporal dimension.

**VFM over-segmentation**: Models such as SAM frequently decompose a single instance into multiple adjacent mask fragments. Post-processing NMS can only partially correct these errors while potentially discarding valid information.

The core insight of this paper draws from classical multi-object tracking (MOT) and video instance segmentation, where temporal consistency is maintained by explicitly preserving and evolving instance-specific representations to propagate semantically rich information across frames. Additionally, inspired by the complementary learning systems of the brain (hippocampus for rapid episodic memory formation vs. neocortex for consolidation into persistent representations), the framework is decomposed into two complementary modules: long-term memory and short-term memory.

## Method

### Overall Architecture

AutoSeg3D is a tracking-centric online 3D segmentation framework consisting of three lightweight modules: (1) Long-Term Memory (LTM) performs cross-frame instance association via Hungarian assignment and a confidence-gated affinity matrix; (2) Short-Term Memory (STM) injects immediate temporal context through distance-aware cross-frame attention; (3) Spatial Consistency Learning (SCL) merges high-affinity mask fragments at inference time and employs one-to-many supervision during training.

### Key Designs

1. **Long-Term Memory (LTM)**: Maintains a bounded track bank. For each frame, appearance affinity (embedding dot product) and geometric affinity (BBox IoU mapped through an MLP) are computed between instance segments and existing tracklets, fused and normalized via softmax to obtain a matching probability matrix $M$, which is then weighted by a sigmoid confidence gate $C$ to yield the final affinity $A = M \cdot C$. The Hungarian algorithm solves the one-to-one optimal assignment. Successfully matched tracklets update their embeddings and bounding boxes via a weighted average (with track age $\alpha$ as the weight). Unmatched segments initialize new tracklets, while tracklets that remain unmatched beyond a timeout are retired into a fixed-capacity buffer queue, from which instances that reappear after prolonged occlusion can be recovered in subsequent frames.

2. **Short-Term Memory (STM)**: Employs a distance-aware attention mechanism to fuse information from adjacent frames. The key design subtracts a distance penalty term $\mathrm{diag}(\tau) \cdot D$ from the softmax of standard attention, where $D$ is the Euclidean distance matrix between instance centroids and $\tau$ is an adaptive receptive field scale predicted per query via a linear layer. Large $\tau$ suppresses long-range attention to encourage local refinement, while small $\tau$ retains global context. This design avoids the background noise introduced by global cross-attention while adaptively regulating the temporal fusion range according to each instance's needs.

3. **Spatial Consistency Learning (SCL)**: Comprises two sub-components. **Learning-Based Mask Integration (LMI)** at inference: reuses the affinity computation module from LTM to compute pairwise affinities among mask fragments within the same frame, then applies hierarchical clustering to merge fragments whose affinity exceeds a threshold $\delta$, followed by feature re-pooling on the merged masks. **Instance Consistency Mask Supervision (ICMS)** during training: for each ground-truth instance, all fragment queries with overlap $> 50\%$ are collected and supervised with the same GT label (one-to-many supervision). To simultaneously preserve fragment selection capability, a dual-branch decoder is adopted: the first branch enables self-attention with one-to-one supervision to maintain selectivity, while the second branch disables self-attention with one-to-many supervision to enhance robustness. The dual-branch mechanism is activated only during training and incurs no additional inference overhead.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_\text{seg} + \beta_\text{ltm} \cdot \mathcal{L}_\text{ltm} + \beta_\text{agg} \cdot \mathcal{L}_\text{agg}$:
- $\mathcal{L}_\text{seg} = \mathcal{L}_{1:1} + \lambda \cdot \mathcal{L}_{1:N} + \gamma \cdot \mathcal{L}_\text{bg}$ (segmentation loss comprising one-to-one, one-to-many, and background penalty terms)
- $\mathcal{L}_\text{ltm} = \mathcal{L}_\text{match} + \beta_\text{conf} \cdot \mathcal{L}_\text{conf}$ (matching loss + BCE loss for confidence gating)
- $\mathcal{L}_\text{agg}$: affinity prediction loss over positive and negative mask-fragment pairs

Training proceeds in two stages: the perception model is first trained on single-view ScanNet200-25k, then fine-tuned on RGB-D sequences (8 frames randomly sampled per scene). AdamW optimizer is used with a learning rate of 0.0001, weight decay of 0.05, batch size of 4, on a single A100 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | AutoSeg3D | ESAM (ICLR'25) | Gain |
|---------|--------|-----------|----------------|------|
| ScanNet200 (SAM) | AP | 45.5 | 42.2 | +3.3 |
| ScanNet200 (SAM) | AP50 | 66.7 | 63.7 | +3.0 |
| ScanNet200 (FastSAM) | AP | 46.2 | 43.4 | +2.8 |
| ScanNet | AP | 43.4 | 41.6 | +1.8 |
| ScanNet | AP50 | 62.5 | 59.6 | +2.9 |
| SceneNN (zero-shot) | AP50 | 53.6 | 52.2 | +1.4 |
| 3RScan (zero-shot) | AP50 | 32.4 | 31.2 | +1.2 |

Real-time FPS is maintained on par with ESAM (0.7 for SAM, 10.1 for FastSAM).

### Ablation Study

| Configuration | AP | AP50 | AP25 | Note |
|---------------|----|------|------|------|
| Baseline (no modules) | 41.6 | 62.9 | 78.7 | — |
| + LTM | 44.1 | 65.8 | 80.7 | Instance association is most critical, +2.5 AP |
| + STM | 42.9 | 63.8 | 80.0 | Short-term context is effective, +1.3 AP |
| + LTM + STM | 44.8 | 66.7 | 81.0 | Two modules are complementary |
| + LTM + STM + ICMS | 45.6 | 66.9 | 81.2 | One-to-many supervision is effective |
| + LTM + STM + LMI | 45.5 | 67.0 | 81.3 | Mask merging is effective |
| Full (all) | 46.2 | 67.9 | 81.7 | All components work synergistically |

### Key Findings

- LTM contributes the largest gain (+2.5 AP), indicating that instance association is the most overlooked critical component in online 3D segmentation.
- The distance-aware attention in STM is essential: without the distance penalty, global cross-attention degrades performance due to spurious associations with background queries.
- LMI and ICMS in SCL each independently contribute approximately 0.7–0.8 AP and their effects are additive.
- Consistent improvements are observed under zero-shot transfer (ScanNet200 training → SceneNN/3RScan evaluation), validating generalization.

## Highlights & Insights

- Reformulating online 3D segmentation as an instance tracking problem is a highly valuable perspective that naturally introduces temporal consistency.
- The LTM + STM design, inspired by the brain's complementary learning systems (hippocampus + neocortex), cleanly separates long-term association from short-term update.
- The dual-branch training strategy in SCL is elegant: one-to-many supervision enhances robustness while one-to-one supervision preserves fragment selectivity.
- Confidence-gated Hungarian matching is more robust than naive matching by suppressing spurious associations.
- The entire framework is lightweight and introduces no additional inference latency.

## Limitations & Future Work

- The method relies on the 2D segmentation quality of VFMs (SAM/FastSAM); poor 2D mask quality imposes an upper bound on performance.
- The fixed-capacity buffer queue in LTM may lose historical information in very long sequences.
- STM only exploits the immediately preceding frame; extending it to multi-frame short-term memory is a natural direction.
- The current evaluation is limited to a class-agnostic setting; open-vocabulary semantic-level 3D segmentation remains unexplored.

## Related Work & Insights

- The key distinction from ESAM (ICLR'25): ESAM lacks instance-level temporal modeling and relies solely on NMS post-processing.
- MOTR and TrackFormer from the MOT community inspire the idea of using queries for temporal propagation.
- The distance-aware attention in STM is analogous to the spatial attention gating in Sparse4D.
- The one-to-many supervision strategy in SCL is conceptually aligned with one-to-many label assignment in object detection.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TAPIP3D: Tracking Any Point in Persistent 3D Geometry](tapip3d_tracking_any_point_in_persistent_3d_geometry.md)
- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](../../ICCV2025/3d_vision/sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](../../ICCV2025/3d_vision/tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)

</div>

<!-- RELATED:END -->
