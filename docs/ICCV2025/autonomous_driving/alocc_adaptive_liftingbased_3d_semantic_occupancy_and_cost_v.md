---
title: >-
  [Paper Note] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions
description: >-
  [ICCV 2025][Autonomous Driving][3D Occupancy Prediction] This paper proposes the ALOcc framework, which achieves state-of-the-art performance on multiple occupancy prediction benchmarks while maintaining high inference speed through three improvements: an occlusion-aware adaptive lifting mechanism, semantic prototype alignment, and BEV cost volume-based flow prediction.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D Occupancy Prediction
  - 2D-to-3D View Transformation
  - Occlusion-Aware Lifting
  - Semantic Prototype
  - Occupancy Flow
date: 2026-05-08
content_hash: fdedc0f251e8cac6
---

# ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions

**Conference**: ICCV 2025
**arXiv**: [2411.07725](https://arxiv.org/abs/2411.07725)
**Code**: [https://github.com/cdb342/ALOcc](https://github.com/cdb342/ALOcc)
**Area**: 3D Vision / Autonomous Driving
**Keywords**: 3D Occupancy Prediction, 2D-to-3D View Transformation, Occlusion-Aware Lifting, Semantic Prototype, Occupancy Flow

## TL;DR

This paper proposes the ALOcc framework, which achieves state-of-the-art performance on multiple occupancy prediction benchmarks while maintaining high inference speed through three improvements: an occlusion-aware adaptive lifting mechanism, semantic prototype alignment, and BEV cost volume-based flow prediction.

## Background & Motivation

Vision-based 3D semantic occupancy prediction requires transforming 2D image features into 3D space, which is an ill-posed problem. Existing 2D-to-3D transformation methods each have inherent limitations:
- **Depth-based LSS**: Explicitly predicts depth probability distributions to guide feature propagation, but the depth target is a delta distribution—weights concentrate on surface points, leaving occluded regions with negligible weight, and spatial density is also low at long ranges. Inaccurate depth estimates in early training can trap the network in local optima.
- **2D-3D Cross Attention**: Passively transfers information via cross-attention without modeling explicit structural information.

Furthermore, joint prediction of semantic occupancy and flow imposes dual pressure on feature encoding (requiring both semantic and motion encoding), and the long-tail distribution of scene categories degrades prediction accuracy for rare classes.

## Core Problem

1. How can 2D-to-3D transformation attend beyond surfaces to cover occluded and sparse regions?
2. How can the semantic quality of 3D features be kept consistent with the original 2D signal while handling class imbalance?
3. How can the multi-task encoding burden on features be reduced during joint semantic and flow prediction?

## Method

### Overall Architecture

N surround-view camera images → ResNet-50 2D feature extraction → **Adaptive Lifting** transforms 2D features into 3D voxel space → 3D encoder (with temporal fusion over 16 historical frames) → two decoding heads output semantic occupancy and occupancy flow, respectively. The architecture is purely convolutional with no Transformer components.

### Key Designs

1. **Occlusion-Aware Adaptive Lifting**:

   Conventional LSS populates the transformation matrix $M_T$ with depth probabilities, but since the depth target is a delta distribution, nearly all weight is placed on the surface, leaving occluded regions without features. ALOcc addresses this with three components:

   - **Soft filling replacing hard rounding**: Trilinear interpolation diffuses depth probabilities to the surrounding 8 voxel centers, making the 2D→3D mapping differentiable with respect to coordinates.
   - **Intra-object occlusion modeling**: A conditional probability matrix $P(o_l|o_d)$ is designed to propagate surface depth probabilities to deeper positions. The physical interpretation is: "if a camera ray reaches a surface at depth $i$, depth $j$ ($j > i$) may also be occupied by the same object." This is implemented via causal conditional matrix multiplication: positions before depth $i$ have zero probability (the ray reaching $i$ implies empty space ahead), depth $i$ itself has probability 1, and positions beyond $i$ are predicted by a network $f_h(x, j-i)$.
   - **Inter-object occlusion modeling**: Multiple offsets $(\Delta u, \Delta v)$ and propagation weights $w$ are predicted per point to spread occupancy probabilities to neighboring pixel locations, covering background objects occluded by foreground objects. For computational efficiency, propagation is applied only to points with top-$k$ depth probabilities.
   - **Depth denoising**: Inspired by query denoising in object detection, training uses a weighted average of GT depth and predicted depth, with the weight annealed from 1 to 0 via cosine scheduling. This ensures GT depth serves as a safety net in early training, preventing local optima due to inaccurate depth estimates.

2. **Semantic Prototype-based Occupancy Head**:

   A learnable prototype vector is initialized for each semantic class, serving simultaneously as classification weights for both 2D and 3D features, naturally establishing a cross-dimensional semantic alignment bridge. During inference, voxel features are matched against prototypes via inner product, and the class with the maximum response is selected.

   Two strategies are designed to address the long-tail problem:
   - **Independent prototype training**: When a class is absent from the current scene's GT, its corresponding prototype is not trained; each class mask is predicted independently to prevent majority classes from suppressing minority classes.
   - **Uncertainty + class-prior sampling**: Logit maps quantify per-voxel uncertainty; combined with class priors as a multinomial distribution, $K$ hard samples are drawn from all voxels, and loss is computed only at these points.

3. **BEV Cost Volume-based Flow Head**:

   Jointly predicting semantics and flow directly on voxel features causes feature encoding conflicts. ALOcc constructs an independent motion prior as follows:
   - Voxel features at heights 0–4 m are average-pooled to the BEV plane and downsampled to increase receptive field.
   - Camera parameters are used to warp the previous frame's BEV features into the current frame's coordinate system.
   - Cosine similarities between the current frame and the warped previous frame features are computed at multiple hypothetical offset points to construct a cost volume.
   - The cost volume together with current-frame voxel features is fed into the flow network, allowing voxel features to focus on semantic encoding.
   - **Classification-regression hybrid prediction**: Flow values are discretized into bins; the probability of each bin is predicted, and the final flow is $\text{flow} = \sum (p_\text{bin} \times \text{bin\_center})$. Both regression loss ($L_2$ + cosine) and classification cross-entropy supervise the prediction.

   The cost volume reuses cached historical frame features, requiring no second forward pass.

### Loss & Training

- Total loss: $\mathcal{L}_\text{total} = \mathcal{L}_\text{3D} + \mathcal{L}_\text{2D} + \mathcal{L}_\text{flow} + \mathcal{L}_\text{depth}$
- $\mathcal{L}_\text{3D}$ / $\mathcal{L}_\text{2D}$: $5\times$ Dice Loss $+ 20\times$ BCE, computed only over the sampled $K$ voxels (pixels)
- $\mathcal{L}_\text{flow} = \mathcal{L}_\text{flow\_reg}$ ($L_2$ + cosine similarity) $+ \mathcal{L}_\text{flow\_cls}$ (bin classification cross-entropy)
- AdamW (lr = 2e-4), batch = 16, 12 epochs (occ) / 18 epochs (occ+flow), CBGS class-balanced sampling
- Three model variants: ALOcc-3D (3D convolution, ch = 32) / ALOcc-2D (height-compressed, ch = 80) / ALOcc-2D-mini (monocular depth + smaller channels)

## Key Experimental Results

| Dataset / Setting | Metric | Ours (R50) | Prev. SOTA (R50) | Gain |
|---|---|---|---|---|
| Occ3D (w/ mask) | mIoU_m | 45.5 (ALOcc-3D) | 44.5 (COTR) | +1.0 |
| Occ3D (w/ mask) | mIoU_D_m | 39.3 (ALOcc-3D) | 38.6 (COTR) | +0.7 |
| Occ3D (w/o mask) | RayIoU | 38.0 (ALOcc-3D) | 31.6 (P-FlashOcc) | +6.4 |
| Occ3D (w/o mask) | mIoU | 43.7 (ALOcc-3D) | 41.2 (OPUS) | +2.5 |
| OpenOcc | Occ Score | 43.0 (Flow-3D) | 41.0 (F-Occ) | +2.0 |
| OpenOcc | mAVE↓ | 0.481 (Flow-3D) | 0.493 (F-Occ) | −0.012 |

- Swin-Base large model: mIoU_m **50.6**, mIoU_D_m **46.1**
- ALOcc-2D-mini real-time variant: **30.5 FPS**, mIoU_m 41.4 (FlashOcc: 29.6 FPS / 32.0; comparable speed but +9.4 higher accuracy)
- **2nd place** at the CVPR 2024 Occupancy and Flow Competition

### Ablation Study

- Removing Adaptive Lifting (AL): mIoU_m 44.5 → 43.5 (−1.0), mIoU_D_m 38.5 → 37.5 (−1.0)
- Removing Semantic Prototype head (SP): mIoU_m 44.5 → 42.1 (−2.4), larger impact
- Removing both AL and SP: mIoU_m 41.2 (−3.3)
- Adding the flow head degrades semantics: RayIoU drops from 42.4 to 39.7
- BEV Cost Volume (CV) alleviates the conflict: RayIoU recovers from 39.7 to 40.2, flow also improves
- Bin Classification (BC) significantly improves flow accuracy: mAVE 0.508 → 0.464
- Channel width 40 → 80 is a bottleneck for joint learning; expanding channels yields substantial overall improvement

## Highlights & Insights

- The theoretical modeling of occlusion-aware probability propagation is rigorous—Bayesian conditional probability is used to derive the "surface → occluded region" probability transfer, analogous to human inference of complete objects from partial observations.
- The depth denoising training strategy is elegant: cosine annealing smoothly transitions from GT depth to predicted depth, avoiding the chicken-and-egg dilemma.
- Shared semantic prototypes jointly constrain 2D and 3D representations; a single prototype set bridges cross-dimensional semantics.
- The BEV cost volume reuses cached historical frame features without requiring a second forward pass, making the design computationally efficient.
- Three speed/accuracy variants ranging from 30.5 FPS to 6 FPS offer strong practical applicability.

## Limitations & Future Work

- Offset propagation in adaptive lifting is restricted to top-$k$ points; extremely dense occlusion scenarios may be insufficiently handled.
- Flow prediction operates only in the BEV plane ($X$/$Y$); motion information along the height direction ($Z$) is compressed away.
- Depth denoising relies on GT depth from LiDAR projection, limiting the independence of purely vision-based systems.
- Semantic prototypes are fixed in class count, precluding open-vocabulary scenarios (could be extended with CLIP).
- Generalization has not been validated on other large-scale datasets such as Waymo.

## Related Work & Insights

- **vs. FB-Occ**: Both follow the LSS improvement paradigm; ALOcc additionally models occlusion propagation and semantic prototypes, achieving +5.7 higher mIoU_m.
- **vs. COTR**: The closest competitor in performance, but COTR runs at only 0.5 FPS, while ALOcc-3D runs at 6.0 FPS—12× faster.
- **vs. FlashOcc**: Among real-time methods with comparable speed (30.5 vs. 29.6 FPS), ALOcc achieves +9.4 higher mIoU.
- **vs. SparseOcc**: SparseOcc introduced the RayIoU metric; ALOcc outperforms it by +7.1 RayIoU.
- **vs. LetOccFlow**: A competing flow method; ALOcc replaces self-supervised flow with BEV cost volume, achieving +6.6 higher OccScore.

The occlusion-aware probability propagation idea is transferable to other occlusion-handling tasks (e.g., person re-identification, indoor scene completion). The 2D-3D bridging via semantic prototypes is conceptually analogous to CLIP's cross-modal alignment; introducing pretrained VLM semantic prototypes could enable open-vocabulary occupancy prediction. The BEV cost volume concept originates from optical flow estimation (e.g., RAFT), serving as a strong example of cross-domain knowledge transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ (Each module offers clear improvements, though not a paradigm shift)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 benchmarks, multiple variants, detailed ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; many equations but logically coherent)
- Value: ⭐⭐⭐⭐ (Speed-accuracy trade-off offers practical engineering value)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)

<!-- RELATED:END -->
