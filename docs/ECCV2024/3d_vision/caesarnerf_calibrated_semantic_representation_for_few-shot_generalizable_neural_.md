---
title: >-
  [Paper Note] CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering
description: >-
  [ECCV2024][3D Vision][Few-shot novel view synthesis] This paper proposes CaesarNeRF, which introduces scene-level semantic representations built upon generalizable NeRF (such as GNT). By leveraging camera pose calibration (aligning feature rotations with the target view) and sequential refinement (gradually updating global features across Transformer layers), CaesarNeRF improves PSNR by 1.74dB (on LLFF) compared to GNT in a 1-view setup, and seamlessly enhances other baseline…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "Few-shot novel view synthesis"
  - "Generalizable NeRF"
  - "Semantic representation calibration"
  - "Transformer"
  - "Sequential refinement"
date: 2026-05-08
content_hash: 124ad0727fd0e4af
---

# CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering

**Conference**: ECCV2024  
**arXiv**: [2311.15510](https://arxiv.org/abs/2311.15510)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: Few-shot novel view synthesis, Generalizable NeRF, Semantic representation calibration, Transformer, Sequential refinement

## TL;DR
This paper proposes CaesarNeRF, which introduces scene-level semantic representations built upon generalizable NeRF (such as GNT). By leveraging camera pose calibration (aligning feature rotations with the target view) and sequential refinement (gradually updating global features across Transformer layers), CaesarNeRF improves PSNR by 1.74dB (on LLFF) compared to GNT in a 1-view setup, and seamlessly enhances other baselines like IBRNet and MatchNeRF.

## Background & Motivation

**Background**: Generalizable NeRFs (such as GNT, IBRNet, and MatchNeRF) achieve feed-forward novel view synthesis through cross-scene training. However, their performance drops significantly under few-shot settings (1-3 views), as the reference views are insufficient to cover the target regions.

**Limitations of Prior Work**: Existing approaches primarily rely on pixel-wise feature matching for view interpolation. When the reference views are extremely limited, the matching information becomes sparse and unreliable. They lack a global understanding of "what the scene looks like as a whole," failing to complete unseen regions.

**Key Challenge**: Pixel-level features provide precise local correspondences but suffer from insufficient coverage under few-shot settings. Conversely, scene-level semantic features can provide global priors, but the semantic features from different viewpoints are inconsistent (global features extracted from different angles of the same scene vary greatly).

**Goal**: How to effectively introduce scene-level semantic representations into generalizable NeRF to enhance few-shot rendering?

**Key Insight**: It is observed that directly averaging global features from different reference views causes semantic conflicts due to viewpoint variations. The solution is to use the rotation matrix of the camera extrinsic parameters to "rotate" and align each viewpoint's semantic features onto the target viewpoint, eliminating viewpoint discrepancies before aggregation.

**Core Idea**: Calibrated semantic representations—aligning multi-view global features with camera pose rotations, concatenating them with pixel-level features, and sequentially refining them across Transformer layers.

## Method

### Overall Architecture
Built on the GNT baseline, CaesarNeRF incorporates: (1) a CNN encoder to extract global semantic features $S_n$ (using Global Average Pooling) $\rightarrow$ (2) calibrating features using the camera rotation matrix as $\tilde{S}_n = \mathcal{P}(T_n \cdot \mathcal{P}^{-1}(S_n))$ $\rightarrow$ (3) averaging the multi-view calibrated features to obtain $\tilde{S}$ $\rightarrow$ (4) concatenating it with pixel-level features to form global-local embeddings $\tilde{E}$ $\rightarrow$ (5) sequentially refining $\tilde{S}$ through cross-attention in each Transformer layer.

### Key Designs

1. **Camera Pose Calibration**:

    - **Function**: Rotate the global semantic features of each reference view into the coordinate system of the target viewpoint.
    - **Mechanism**: Given the transformation $T_n = T_{out}^{w2c} \cdot T_n^{c2w}$, the $3\times3$ rotation submatrix is extracted to rotate the features. The semantic features are first reshaped into a 3D volume, rotated, and then reshaped back.
    - **Design Motivation**: Global features of a scene observed from different views are semantically inconsistent (e.g., front vs. side). Only after rotating and aligning them to eliminate viewpoint discrepancies does averaging become meaningful. Ablations show that adding only calibration improves PSNR by 0.67 on MVImgNet (1-view).

2. **Sequential Refinement**:

    - **Function**: Learn a residual update $\Delta^{(k)}$ from per-frame features $\{S_n\}$ via cross-attention in each Transformer layer, progressively refining the global representation as $\tilde{S}^{(k+1)} = \tilde{S}^{(k)} + \Delta^{(k)}$.
    - **Mechanism**: Different Transformer layers demand global information of varying granularities—shallow layers require coarse structural information, while deep layers demand precise fine details. A static global feature cannot adapt to this requirement.
    - **Design Motivation**: This is analogous to the concept of FPN using different scales of features across different layers, but here the hierarchy is built upon the feature dimension rather than the spatial dimension.

3. **Central Loss**:

    - **Function**: Constrain the L1 distance between each calibrated view feature $\tilde{S}_n$ and the mean feature $\tilde{S}$.
    - **Mechanism**: $\mathcal{L}_{central} = \frac{1}{N}\sum ||\tilde{S}_n - \tilde{S}||_1$, encouraging feature consistency after calibration.
    - **Design Motivation**: Ensure that features from different views indeed converge toward consistency after rotation alignment, preventing calibration collapse.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{central} + \lambda_2 \mathcal{L}_{perc}$ ($\lambda_1=1, \lambda_2=0.001$). Training lasts for 500K iterations, sampling 4096 rays per batch from a single scene. Features are processed with an 8-layer Transformer and 96-dimensional semantic features.

## Key Experimental Results

### Main Results

**LLFF Dataset (Few-shot Rendering)**:

| Method | 1-view PSNR | 2-view PSNR | 3-view PSNR |
|------|-----------|-----------|-----------|
| IBRNet | 16.85 | 21.25 | 23.00 |
| GNT (baseline) | 16.57 | 20.88 | 23.21 |
| MatchNeRF | — | 21.08 | 22.30 |
| **CaesarNeRF** | **18.31** | **21.94** | **23.45** |

Using 1-view achieves an improvement of **+1.74 PSNR** over GNT, showing the most significant gain when the number of views is minimal.

**Plug-and-play Enhancement on Other Baselines (LLFF 2-view)**:

| Baseline | Original PSNR | +Caesar PSNR | Gain |
|------|---------|-------------|------|
| IBRNet | 21.25 | 22.39 | +1.14 |
| MatchNeRF | 20.59 | 21.55 | +0.96 |

### Ablation Study (LLFF "orchid" Scene)

| Configuration | PSNR | LPIPS |
|------|------|-------|
| GNT baseline | 20.93 | 0.185 |
| + Semantic representation (96-dim) | 21.46 | 0.150 |
| + Sequential refinement | 21.53 | 0.146 |
| + Calibration | 21.51 | 0.147 |
| **+ Semantic + Sequential refinement + Calibration (Full)** | **21.67** | **0.139** |

### Key Findings
- **Fewer views bring higher gains**: 1-view improvement of 1.74 PSNR is much larger than 3-view's 0.24 PSNR, indicating semantic representations primarily compensate for missing information in extremely sparse views.
- **Calibration is more critical in object-centric scenes**: Calibration yields a 0.67 PSNR improvement on MVImgNet, where surrounding viewpoints undergo larger transformations.
- **Strong generalization capability**: The same method can be plugged into three different baselines (GNT, IBRNet, and MatchNeRF), yielding consistent improvements in all cases.
- **96-dimensional semantic features are sufficient**: Performance increases continuously from 32 to 96 dimensions, but brings negligible gain from 96 to 128.

## Highlights & Insights
- **The idea of using camera pose to rotate features is simple yet sound**: It shares a similar philosophy with SE(3)-equivariant methods but is extremely lightweight—requiring only a single matrix multiplication. The key insight is that the "orientation" of semantic features is coupled with the viewing angle, and rotation alignment eliminates this coupling.
- **Plug-and-play design**: It does not alter the core of the baseline architectures, but only appends a global semantic channel to the feature space, facilitating easy integration into any feature-matching-based generalizable NeRF.
- **Regularization effect of Central Loss**: A simple L1 consistency constraint effectively guides calibration learning and prevents collapse.

## Limitations & Future Work
- **Still reliant on reference view coverage**: Unlike generative methods, CaesarNeRF cannot render regions completely unobserved by the reference views (e.g., the backside of an object).
- **Rotation alignment assumption**: It assumes that the rotation matrix is sufficient to align semantic features, which may be inadequate for scenes containing heavy occlusion or perspective transformations.
- **Insufficient ablation**: Ablations are primarily conducted on a single scene (orchid), lacking a comprehensive cross-scene effectiveness analysis.
- **Potential improvements**: (1) combining with diffusion models to handle completely unseen regions; (2) incorporating translation and scale calibration rather than just rotation; (3) pre-training the semantic encoder on larger-scale datasets.

## Related Work & Insights
- **vs GNT**: CaesarNeRF introduces a global semantic channel on top of GNT while keeping the core Transformer architecture intact. The 1.74 PSNR improvement on 1-view setups demonstrates that global understanding is crucial for sparse views.
- **vs PixelNeRF**: PixelNeRF uses pixel-aligned features but lacks global context, achieving a 1-view PSNR of only 9.32 compared to CaesarNeRF's 18.31.
- **vs GeoNeRF**: GeoNeRF utilizes geometric priors (depth estimation) to enhance few-shot rendering, whereas CaesarNeRF employs semantic priors. The two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of pose-calibrated semantic features is simple yet effective, and sequential refinement is also a solid design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets + 3 baseline plug-and-play experiments + ablations, though ablations are primarily evaluated on a single scene.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the formula derivations are fully presented.
- Value: ⭐⭐⭐⭐ A practical improvement for few-shot generalizable NeRF, with a plug-and-play design that is easy for the community to adopt.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views](gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)
- [\[CVPR 2026\] PP-Brep: Few-Shot B-rep Classification with Hybrid Graph Representation](../../CVPR2026/3d_vision/pp-brep_few-shot_b-rep_classification_with_hybrid_graph_representation.md)
- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](../../CVPR2025/3d_vision/ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[ECCV 2024\] High-Resolution and Few-shot View Synthesis from Asymmetric Dual-Lens Inputs](high-resolution_and_few-shot_view_synthesis_from_asymmetric_dual-lens_inputs.md)
- [\[AAAI 2026\] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios](../../AAAI2026/3d_vision/epsegfz_efficient_point_cloud_semantic_segmentation_for_few-_and_zero-shot_scena.md)

</div>

<!-- RELATED:END -->
