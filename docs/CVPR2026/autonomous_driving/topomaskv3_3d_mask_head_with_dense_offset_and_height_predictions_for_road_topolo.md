---
title: >-
  [Paper Note] TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding
description: >-
  [CVPR 2026][Autonomous Driving][Road Topology Understanding] This paper proposes TopoMaskV3, which upgrades the mask-based road topology understanding paradigm from a 2D auxiliary module to a standalone 3D centerline pre…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Road Topology Understanding"
  - "Mask Paradigm"
  - "Offset Correction"
  - "Height Prediction"
  - "Geographic Data Leakage"
date: 2026-05-08
content_hash: 2e7f0c9aad1bce97
---

# TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.01558](https://arxiv.org/abs/2603.01558)
**Code**: [Project Page](https://artest08.github.io/TopoMaskV3.github.io/)
**Area**: Autonomous Driving
**Keywords**: Road Topology Understanding, Mask Paradigm, Offset Correction, Height Prediction, Geographic Data Leakage

## TL;DR

This paper proposes TopoMaskV3, which upgrades the mask-based road topology understanding paradigm from a 2D auxiliary module to a standalone 3D centerline predictor by introducing dense offset fields and dense height maps as additional prediction heads. The work also introduces, for the first time in road topology evaluation, a geographically non-overlapping split and a long-range benchmark, exposing performance inflation caused by geographic overlap in existing benchmarks. TopoMaskV3 achieves state-of-the-art 28.5 OLS on the geographically non-overlapping benchmark.

## Background & Motivation

1. **Background**: Road topology understanding requires not only detecting static road elements such as lane lines, but also inferring complex relationships among them (e.g., connectivity, merging/diverging, traffic signal associations). Mainstream methods adopt end-to-end transformers to predict vectorized representations in the form of parametric Bézier curves (TopoMLP), path-level predictions (LaneGAP), or rasterization-based masks (TopoMaskV2).

2. **Limitations of Prior Work**: TopoMaskV2 introduced a mask-based paradigm that generates centerlines from rasterized masks, encoding flow direction via four-direction labels. However, it suffers from two critical limitations: (1) rasterization-to-vectorization introduces discretization artifacts, as real centerlines rarely align with grid centers; and (2) the model only supports 2D output, lacking height prediction. These limitations force TopoMaskV2 to fuse with a parametric Bézier head to remain competitive.

3. **Key Challenge**: A deeper issue concerns evaluation fairness. The existing OpenLane-V2 benchmark adopts a temporal split (reasonable for dynamic objects), but roads are static; the training and test sets share significant geographic overlap, allowing models to achieve high scores by "memorizing the map" rather than learning genuine generalization.

4. **Goal**: (1) Upgrade the mask paradigm into a standalone 3D predictor; (2) Establish a rigorous benchmark for evaluating road topology generalization.

5. **Key Insight**: Predict, for each BEV grid cell, the offset to the nearest centerline point and the corresponding height value, fundamentally addressing discretization errors and the 2D limitation.

6. **Core Idea**: Sub-grid precision correction via dense offset fields combined with 3D lifting via dense height maps, alongside a geographically non-overlapping benchmark for fair generalization evaluation.

## Method

### Overall Architecture

TopoMaskV3 takes multi-view RGB images as input. A backbone extracts perspective-view features, which are then projected into a unified BEV feature map. A transformer decoder employs sparse queries—each corresponding to one centerline instance—and outputs five parallel prediction heads: a classification head (four-direction labels), a mask head (instance mask probability maps), an offset head (2D offset fields), a height head (height maps), and an optional Bézier head. The primary pipeline generates 3D centerlines from the classification, mask, offset, and height heads; optionally, outputs from the Bézier path are fused at inference.

### Key Designs

1. **Dense Offset Field**

   - **Function**: Corrects discretization errors introduced during rasterization, enabling sub-grid precision.
   - **Mechanism**: For each BEV grid cell, a 2D offset vector $\mathbf{o}_{ij} = \mathbf{O}(i,j,:)$ is predicted, pointing from the grid center toward the nearest real centerline point. During training, multi-point supervision is applied: the target offset for each foreground pixel is computed as $\mathbf{o}_{ij}^{gt} = \Pi_\mathcal{C}((i,j)) - (i,j)$, where $\Pi_\mathcal{C}$ denotes the nearest-point projection onto the continuous centerline. At inference, two usage modes are provided: single-point proposals (correcting only the initially extracted centerpoints) and multi-point proposals (correcting all foreground pixels within the mask region).
   - **Design Motivation**: Conventional row/column expectation extraction can only locate centerpoints at grid-center positions. The offset field provides a means to surpass grid resolution limits without increasing BEV resolution, thereby reducing computational cost.

2. **Dense Height Map**

   - **Function**: Supplies z-coordinates for 2D centerlines, enabling end-to-end 3D prediction.
   - **Mechanism**: The model predicts $\mathbf{H} \in \mathbb{R}^{H_{BEV} \times W_{BEV}}$, where each grid cell corresponds to a normalized height value. Training supervision follows the same multi-point nearest-neighbor strategy: $h_{ij}^{gt} = h_{norm}(\Pi_\mathcal{C}((i,j)))$. At inference, height values are sampled at offset-corrected $(x,y)$ positions and concatenated to form 3D points.
   - **Design Motivation**: TopoMaskV2 entirely lacks height prediction capability. The height map shares the "dense prediction + multi-point supervision" framework with the offset field, yielding a unified and straightforward design.

3. **Curve Reconstruction Pipeline**

   - **Function**: Converts noisy 3D grid points into smooth, ordered centerlines.
   - **Mechanism**: Grid coordinates are first mapped to real-world coordinates via a transformation matrix. The four-direction label then determines the independent variable for polynomial fitting (e.g., $y=f(x)$ for up/down directions), while a 3D height surface $z=g(x,y)$ is fitted simultaneously. Finally, arc-length interpolation resamples the curve into equidistant points and sorts them.
   - **Design Motivation**: Polynomial fitting combined with arc-length interpolation further smooths discretization artifacts, ensuring high-quality vectorized output.

### Loss & Training

The training objective combines mask segmentation loss, four-direction classification cross-entropy, offset regression L1 loss, and height regression L1 loss. The mask probability threshold is set to $\tau = 0.95$, and 4th-order polynomials are used for curve reconstruction. ML1M (Mixed L1 Matcher) and BDA (Bézier Deformable Attention) are optionally employed to enhance training.

## Key Experimental Results

### Main Results

SOTA comparison on the geographically non-overlapping Near split (V1.1 metrics with score mapping):

| Method | DET_l | DET_l_ch | TOP_ll | OLS_l |
|--------|-------|----------|--------|-------|
| TopoNet | 18.9 | 23.5 | 12.7 | 26.0 |
| TopoMLP | 15.6 | 22.4 | 14.5 | 25.3 |
| TopoLogic | 16.9 | 22.7 | 15.5 | 26.3 |
| TopoMaskV2 (Mask) | 16.4 | 20.1 | 10.9 | 23.2 |
| TopoMaskV2 (Fusion) | 18.5 | 23.8 | 11.7 | 25.5 |
| TopoBDA | 20.8 | 24.9 | 13.0 | 27.3 |
| **TopoMaskV3 (Mask)** | **19.3** | **25.6** | **13.6** | **27.3** |
| **TopoMaskV3 (Fusion)** | **20.5** | **26.2** | **15.1** | **28.5** |

### Ablation Study

Ablation of offset and height prediction (multi-point proposal mode):

| Configuration | DET_l | DET_l_ch | TOP_ll | OLS_l |
|---------------|-------|----------|--------|-------|
| No prediction (baseline) | 31.1 | 31.7 | 22.5 | 36.8 |
| Offset only | 32.5 | 33.1 | 23.8 | 38.2 |
| Height only | 32.6 | 37.2 | 23.4 | 39.4 |
| **Offset + Height** | **33.1** | **37.9** | **25.0** | **40.3** |

Cross-split generalization ablation (camera, ±50 m):

| Output Type | Original (Overlap) | Near | FarA | FarB | FarC |
|-------------|-------------------|------|------|------|------|
| Bezier | 43.4 | 27.8 | 22.2 | 20.9 | 27.8 |
| Mask | 40.8 | 26.4 | 21.2 | 20.0 | 27.1 |
| Fusion | 42.5 | 27.9 | 22.3 | 20.7 | 28.3 |

### Key Findings

- Height prediction contributes more than offset prediction (OLS_l +2.6 vs. +1.4), particularly on the DET_l_ch (spatial localization) metric.
- Multi-point proposals (correcting all foreground pixels) outperform single-point proposals (correcting only the initial centerpoint): OLS_l 40.3 vs. 38.9.
- **Geographic leakage is severe**: performance drops approximately 42% across all methods when moving from the original split to the geographically non-overlapping split, indicating that the standard benchmark substantially inflates reported numbers.
- The Bézier head performs best on the overlapping split but degrades most severely in generalization; the Mask and Fusion heads generalize more robustly.
- LiDAR fusion yields the largest benefit in the long-range (±100 m) setting (relative gain of 40.8%), and its relative gain is greater on the overlapping split than on the non-overlapping split, suggesting that LiDAR fusion partially benefits from geographic memorization.

## Highlights & Insights

- **Upgrading the mask head from "requires fusion to be competitive" to "standalone 3D predictor"**: TopoMaskV3 (Mask) alone matches TopoBDA, validating the potential of the mask paradigm.
- **Geographic data leakage analysis is a significant contribution**: This is the first systematic exposure of performance inflation due to geographic overlap in road topology tasks, serving as an important warning for both the HD map and topology communities.
- **The multi-point supervision strategy for the offset field is elegant**: Unlike conventional offset regression in keypoint detection, every foreground pixel predicts an offset here, providing denser training signals and redundant corrections at inference.
- **The introduction of long-range benchmarks (±100 m) is forward-looking**: High-speed driving scenarios demand longer perception ranges, making the ±100 m benchmark more representative of real-world requirements.

## Limitations & Future Work

- The curve reconstruction pipeline (polynomial fitting + arc-length interpolation) is not fully differentiable, limiting end-to-end optimization.
- The mask paradigm may produce topological inconsistencies at complex intersections where multiple centerline masks overlap.
- Under the geographically non-overlapping split, all methods perform poorly (OLS_l < 30), indicating that road topology generalization itself remains a formidable challenge.
- Validation is limited to Argoverse2; cross-dataset verification on NuScenes and other benchmarks is absent.

## Related Work & Insights

- **vs. TopoMaskV2**: TopoMaskV3 addresses both core limitations of its predecessor (discretization errors and lack of height prediction), elevating the mask head from an "auxiliary module" to a "primary predictor."
- **vs. TopoBDA**: TopoBDA relies on pure Bézier representations and leads on the standard benchmark, but is surpassed by TopoMaskV3 on the non-overlapping benchmark, suggesting that Bézier representations are more susceptible to overfitting geographic information.
- **vs. StreamMapNet/MapTR**: These HD map works introduced the concept of geographic splits; TopoMaskV3 is the first to adapt this practice to the more complex road topology task.

## Rating

- Novelty: ⭐⭐⭐⭐ The offset + height design is intuitive, but the multi-point supervision and geographic benchmark analysis demonstrate depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis across multiple splits, sensors, and distances with highly detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough benchmark analysis.
- Value: ⭐⭐⭐⭐ Dual contributions from the upgraded mask paradigm and the fair benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[AAAI 2026\] Minimum-Cost Network Flow with Dual Predictions](../../AAAI2026/autonomous_driving/minimum-cost_network_flow_with_dual_predictions.md)
- [\[AAAI 2026\] Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving](../../AAAI2026/autonomous_driving/invisible_triggers_visible_threats_road-style_adversarial_creation_attack_for_vi.md)
- [\[ICCV 2025\] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions](../../ICCV2025/autonomous_driving/alocc_adaptive_liftingbased_3d_semantic_occupancy_and_cost_v.md)

</div>

<!-- RELATED:END -->
