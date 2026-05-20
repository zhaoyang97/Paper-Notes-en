---
title: >-
  [Paper Note] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][occupancy prediction] This paper proposes Dr.Occ, a unified camera-only 3D occupancy prediction framework. It introduces a Depth-guided Dual-projection View Former (D2-VFormer) that levera…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "occupancy prediction"
  - "depth guidance"
  - "MoGe-2"
  - "Mixture-of-Experts"
  - "region-guided"
  - "view transformation"
date: 2026-05-08
content_hash: 059d2887bea99f69
---

# Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2603.01007](https://arxiv.org/abs/2603.01007)  
**Code**: N/A  
**Area**: Autonomous Driving / 3D Occupancy Prediction
**Keywords**: occupancy prediction, depth guidance, MoGe-2, Mixture-of-Experts, region-guided, view transformation

## TL;DR

This paper proposes Dr.Occ, a unified camera-only 3D occupancy prediction framework. It introduces a Depth-guided Dual-projection View Former (D2-VFormer) that leverages high-quality depth priors from MoGe-2 for accurate geometric alignment, and a Region-guided Expert Transformer (R-EFormer / R2-EFormer) that adaptively assigns spatial region experts to address semantic imbalance. Dr.Occ improves the BEVDet4D baseline by 7.43% mIoU on Occ3D-nuScenes.

## Background & Motivation

**Background**: 3D semantic occupancy prediction is a core perception task in autonomous driving, aiming to produce dense voxel-level scene representations that provide geometric and semantic information for motion planning and obstacle avoidance. Camera-only approaches (LSS, BEVFormer, COTR, etc.) that perform 2D-to-3D view transformation represent the dominant paradigm.

**Limitations of Prior Work**:
1. **Geometric misalignment**: Existing forward-projection methods (LSS, BEVDepth) rely on low-resolution, noisy depth estimates for 2D→3D feature transformation, limiting projection accuracy.
2. **Spatial semantic imbalance**: Different semantic categories exhibit strong spatial preferences in 3D space — pedestrians concentrate near road edges, vehicles in the center lane, and buildings at higher elevations — yet existing methods model all regions uniformly.
3. Approximately 90% of voxels are empty, making direct fitting over all voxels inefficient.
4. Naively concatenating MoGe depth maps with images or converting them to pseudo point clouds for forward projection actually degrades performance.

**Key Challenge**: How to exploit high-quality geometric priors from advanced depth estimation models to improve occupancy prediction, while simultaneously addressing the severe spatial imbalance in semantic category distributions.

**Goal**: Jointly address the two core challenges in camera-only occupancy prediction: inaccurate geometric reconstruction and imbalanced semantic learning.

**Key Insight**: Depth guidance ensures geometric alignment; region-guided experts enhance semantic learning — the two are complementary by design.

**Core Idea**: Use MoGe-2 depth to generate occupancy masks that guide non-empty voxel refinement, and apply MoE-style spatial region routing to handle semantic heterogeneity.

## Method

### Overall Architecture

Dr.Occ introduces two modifications to the standard occupancy prediction pipeline:
1. **D2-VFormer** replaces the original view transformer, constructing 3D voxel features via depth-guided dual projection using MoGe-2 priors.
2. **R2-EFormer** is inserted during the 3D feature refinement stage for region-guided recursive semantic enhancement.

The pipeline is: $T$-frame surround-view images → image encoder → D2-VFormer (3D voxel features) → R2-EFormer (semantic refinement) → OCC decoder → $\hat{\mathbf{O}} \in \mathbb{R}^{X \times Y \times Z \times C}$.

### Key Designs

1. **Depth-Guided 2D-to-3D View Former (D2-VFormer)**:
    - **Geometry-aware occupancy mask**: The depth map $\mathbf{D}_i$ estimated by MoGe-2 is used to generate a pseudo point cloud $\mathcal{P}$, which is projected and voxelized to produce a binary mask $M(\mathbf{v})$ marking non-empty voxel locations.
    - Projection formulae: $\mathbf{x}_{\text{cam}}^T = d \cdot \mathbf{K}_i^{-1}[u, v, 1]^T$, $\mathbf{p}_i = \mathbf{R}_i^\top(\mathbf{x}_{\text{cam}} - \mathbf{t}_i)$, $M(\mathbf{v}) = \mathbf{1}[\mathbf{v} \in \text{Voxelize}(\mathcal{P}, r)]$
    - **Three-stage progressive refinement**:
        - Stage 1 (forward projection + downsampling): Multi-frame image features are projected into voxel space via depth, then downsampled by factor $\lambda$ to improve computational efficiency and depth robustness.
        - Stage 2 (backward projection densification): Deformable Cross-Attention (DCA) aggregates multi-view image features: $\mathbf{F}_{\text{dense}} = \text{DCA}(\mathbf{F}_{\text{down}}, \mathbf{F}^{(I)})$.
        - Stage 3 (depth-guided non-empty voxel refinement): Only mask-identified non-empty voxels undergo two-step refinement — geometric refinement (fusing depth features $\mathbf{F}^{(D)}$) followed by semantic enhancement (fusing image features $\mathbf{F}^{(I)}$); empty voxels are filled with a learnable embedding $\mathbf{e}_{\text{empty}}$.
    - Key insight: Rather than using MoGe depth directly for forward projection (which degrades feature quality by removing implicit depth constraints), the depth is used to generate masks that guide attention toward meaningful regions.

2. **Region-guided Expert Transformer (R-EFormer)**:
    - **Spatial semantic analysis**: Statistical analysis reveals that different semantic categories exhibit strong anisotropic distributions along distance and height dimensions — road surfaces cluster at low elevations near range, vegetation/buildings appear at higher elevations mid-range, and dynamic objects occupy narrow spatial bands.
    - The 3D space is partitioned into $3 \times 3 = 9$ regions along distance (near 0–10 m / mid 10–30 m / far 30+ m) and height (low −1.0–0.2 m / mid 0.2–2.2 m / high 2.2–5.4 m).
    - A routing network computes per-region importance scores $s_m = \text{Router}(\mathbf{F}_{\text{out}})$, and the top-$K$ most relevant regions activate their corresponding experts.
    - Each expert $E_m$ applies DCA restricted to its region mask $\mathcal{M}_m$: $E_m(\mathbf{F}_{\text{out}}, \mathbf{F}^{(I)}; \mathcal{M}_m) = \text{DCA}(\mathbf{F}_{\text{out}}, \mathbf{F}^{(I)}; \mathcal{M}_m)$
    - Final output is a weighted fusion: $\mathbf{F}_{\text{final}} = \sum_{m \in \mathcal{S}} w_m \cdot E_m(\mathbf{F}_{\text{out}}, \mathbf{F}^{(I)}; \mathcal{M}_m)$

3. **Recursive Variant R2-EFormer**:
    - Inspired by Mixture-of-Recursions (MoR), a single shared expert is applied iteratively for $n$ steps, with the router generating progressively shrinking spatial masks at each iteration.
    - The mask sequence satisfies $\mathcal{M}^{(t)} \subset \mathcal{M}^{(t-1)}$ with decreasing coverage (100% → 75% → 50%).
    - Each iteration: $\mathbf{F}^{(t)} = \text{DCA}(\mathbf{F}^{(t-1)}, \mathbf{F}^{(I)}; \mathcal{M}^{(t)})$
    - Advantages: fewer parameters via weight sharing, no need for manual region boundary definition, and progressive focus on hard-to-classify voxels.

### Loss & Training

- AdamW optimizer, learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-2}$
- 24 training epochs, batch size = 2/GPU × 8 GPUs (NVIDIA L20)
- Image encoder: ResNet-50; depth estimation: moge-2-vits-normal
- Forward projection voxel channels $C=32$, resolution $200 \times 200 \times 16$, covering $80 \times 80 \times 6.4$ m
- Multi-head attention: 8 heads, $N_{\text{ref}} = 4$ reference points

## Key Experimental Results

### Main Results

Occ3D-nuScenes benchmark (mIoU / IoU %):

| Method | Backbone | mIoU | IoU |
|--------|----------|---:|---:|
| BEVFormer | R101 | 26.9 | — |
| TPVFormer | R101 | 27.8 | — |
| SparseOcc | R50 | 30.9 | — |
| BEVDet4D | R50 | 36.0 | — |
| FlashOcc | R50 | 37.8 | — |
| FB-Occ | R50 | 39.1 | — |
| ViewFormer | R50 | 41.9 | — |
| COTR | R50 | 43.1 | — |
| **BEVDet4D + Dr.Occ** | R50 | **43.4** (+7.43) | (+3.09) |
| **COTR + Dr.Occ** | R50 | **44.1** (+1.0) | — |

Dr.Occ achieves substantial foreground class IoU improvements over BEVDet4D (e.g., bicycle +20.4, pedestrian +13.4, motorcycle +6.9), with consistent gains on background classes as well.

### Ablation Study

Contribution of individual modules:

| D2-VFormer | R-EFormer | R2-EFormer | IoU (%) | mIoU (%) |
|:---:|:---:|:---:|---:|---:|
| | | | 70.36 | 36.01 |
| ✔ | | | 71.29 (+0.93) | 41.45 (+5.44) |
| ✔ | ✔ | | **73.45** (+2.16) | 43.03 (+1.58) |
| ✔ | | ✔ | 72.87 | **43.43** (+1.98) |

Key observations:
- D2-VFormer alone contributes +5.44% mIoU, validating the importance of depth guidance for geometric completeness and semantics.
- R-EFormer on top of D2-VFormer adds a further +1.58% mIoU and achieves the highest IoU.
- Replacing R-EFormer with R2-EFormer yields a slightly lower IoU but the highest mIoU (43.43%), as recursive refinement more effectively handles rare and hard-to-classify categories.

### Key Findings

1. Directly applying MoGe depth for forward projection degrades performance, as image features lose implicit depth constraints; using depth to generate masks is a superior strategy.
2. With ~90% of voxels being empty, geometry-aware masking focuses the model on the meaningful ~10% of voxels, substantially improving both efficiency and accuracy.
3. The spatial anisotropy of semantic categories (road surfaces at the bottom, buildings at higher elevations) is an objective phenomenon; MoE-style region experts effectively exploit this prior.
4. The recursive mask shrinkage strategy of R2-EFormer (100% → 75% → 50%) automatically focuses on hard voxels without requiring manually defined region boundaries.
5. As a plug-and-play module, Dr.Occ further improves the current state-of-the-art COTR by 1.0% mIoU, demonstrating generalizability.

## Highlights & Insights

1. **Elegant use of depth priors**: Rather than applying depth directly for projection (which can introduce domain bias), depth is used to generate occupancy masks for attention guidance — a conceptually novel strategy.
2. **MoE localized to 3D physical space**: The expert routing concept from MoE is transferred from token space to physically defined spatial regions, aligning well with the spatial semantic structure of autonomous driving scenes.
3. **MoR-inspired recursive variant** reduces parameters via weight sharing while adaptively discovering important regions, avoiding sensitivity to manually defined region boundaries.
4. The geometric and semantic modules are decoupled by design and can be independently inserted into different baselines.

## Limitations & Future Work

1. The framework depends on an external depth estimation model (MoGe-2), increasing inference latency and deployment complexity.
2. The region partition in R-EFormer (3×3 grid) is a manually defined hyperparameter that may require tuning for different scenes.
3. Evaluation is limited to Occ3D-nuScenes; other benchmarks (e.g., OpenOccupancy, SurroundOcc) are not tested.
4. Only ResNet-50 backbone is used; stronger backbones (e.g., InternImage, ViT) are not explored.
5. In-depth optimization of temporal fusion is not investigated; only the basic multi-frame fusion from BEVDet4D is adopted.

## Related Work & Insights

- **LSS / BEVDepth / BEVStereo**: Classical forward-projection methods; D2-VFormer builds upon these by introducing externally guided depth priors.
- **BEVFormer**: Classical backward-projection approach that samples image features via transformer queries.
- **COTR**: Dual-projection design; D2-VFormer further introduces depth mask guidance.
- **MoGe-2**: High-quality monocular depth estimation model that supplies geometric priors to Dr.Occ.
- **Mixture-of-Recursions (MoR)**: An efficient design that replaces multiple experts with a single recursively applied shared expert.
- **Insight**: As foundation models (e.g., MoGe, Depth Anything) continue to advance, effectively leveraging these off-the-shelf tools to provide strong priors for downstream tasks is a direction worth deeper exploration.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)
- [\[AAAI 2026\] Exploring Surround-View Fisheye Camera 3D Object Detection](../../AAAI2026/autonomous_driving/exploring_surround-view_fisheye_camera_3d_object_detection.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)

</div>

<!-- RELATED:END -->
