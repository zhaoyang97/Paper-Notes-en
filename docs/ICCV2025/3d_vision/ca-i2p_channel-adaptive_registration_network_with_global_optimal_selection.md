---
title: >-
  [Paper Note] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection
description: >-
  [ICCV 2025][3D Vision][Image-to-point cloud registration] This paper proposes CA-I2P, which introduces a Channel Adaptive Adjustment Module (CAA) to enhance and filter channel-level discrepancies between image and point cloud features, and a Global Optimal Selection (GOS) module that replaces top-k selection with optimal transport to reduce many-to-one matching errors, achieving state-of-the-art image-to-point cloud registration performance on RGB-D Scenes V2 and 7-Scenes.
tags:
  - ICCV 2025
  - 3D Vision
  - Image-to-point cloud registration
  - cross-modal feature matching
  - channel adaptivity
  - optimal transport
  - detector-free method
date: 2026-05-08
content_hash: 2d8b7cb7729b35a2
---

# CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection

**Conference**: ICCV 2025
**arXiv**: [2506.21364](https://arxiv.org/abs/2506.21364)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: Image-to-point cloud registration, cross-modal feature matching, channel adaptivity, optimal transport, detector-free method

## TL;DR

This paper proposes CA-I2P, which introduces a Channel Adaptive Adjustment Module (CAA) to enhance and filter channel-level discrepancies between image and point cloud features, and a Global Optimal Selection (GOS) module that replaces top-k selection with optimal transport to reduce many-to-one matching errors, achieving state-of-the-art image-to-point cloud registration performance on RGB-D Scenes V2 and 7-Scenes.

## Background & Motivation

Image-to-point cloud registration (I2P) estimates the rigid transformation from the point cloud coordinate system to the camera coordinate system, and serves as a foundation for 3D reconstruction, SLAM, and visual localization. However, the substantial domain gap between images (2D regular dense grids) and point clouds (3D unordered sparse irregular data) makes cross-modal registration highly challenging.

Existing detector-free methods (e.g., 2D3D-MATR) adopt a coarse-to-fine pipeline: image and point cloud features are first extracted, followed by patch-level matching and subsequent refinement to pixel-point correspondences. However, two core issues persist:

**Channel-level feature inconsistency**: Differences in the imaging ranges of LiDAR and cameras lead to distinct attention distributions across channels for each modality. Features output by 2D and 3D encoders exhibit systematic bias along the channel dimension, causing misaligned receptive fields and erroneous matching in occluded regions. As illustrated in the paper, the channel color distributions of the two modalities differ markedly and only converge after processing by the proposed module.

**Redundant correspondences from local selection**: Indoor scenes commonly contain similar structures (e.g., repeated furniture, symmetric architecture). Conventional top-k selection strategies cause multiple similar structures to be incorrectly matched to the same cross-modal object, producing redundant many-to-one correspondences that degrade registration accuracy.

## Method

### Overall Architecture

CA-I2P extracts image features using FPN-ResNet and point cloud features using KPFCNN. After Fourier positional encoding and Transformer-based enhancement, the features are passed to the Channel Adaptive Adjustment Module (CAA) for channel-level refinement. A cosine similarity score map is then computed and processed by the Global Optimal Selection (GOS) module to obtain accurate patch-level matches, which are subsequently refined into dense pixel-point correspondences; the final transformation is estimated via PnP+RANSAC.

### Key Designs

1. **Intra-Modal Enhancement Stage (IME)**: Separate enhancement units are designed for each modality:

    - **Image Channel Enhancement Unit (ICE)**: Three parallel branches are designed to capture cross-dependencies among the $(H,W)$, $(C,H)$, and $(C,W)$ dimensions, respectively. Dimensional interaction is achieved by rotating the tensor along different axes. Each branch applies GO-Pool (concatenation of max and average pooling) for dimensionality reduction, followed by Conv+BN+Sigmoid to produce attention weights; the outputs of the three branches are averaged:
    $F'_I = \frac{1}{3}\sum_{y=1}^{3} F_{Iy} \sigma(\text{Conv}(\text{GO-Pool}(F_{Iy})))$
    - **Point Channel Enhancement Unit (PCE)**: A channel self-attention mechanism is applied to point cloud features, using Q/K/V linear projections to compute inter-channel correlations and adaptively recalibrate them:
    $A = \text{Softmax}\left(\frac{QK^T S}{\sqrt{C}}\right), \quad F'_P = AV$
    - The enhanced features are fused with the original features via learnable weights: $F_I = \alpha F_I + \beta F'_I$

2. **Cross-Modal Channel Filtering Stage (CMCF)**:

    - Instance Normalization (IN) is first applied per sample to render features independent of training set statistics.
    - The covariance matrices $\mathbf{V}^x_I$ and $\mathbf{V}^x_P$ are computed for the image and point cloud features, respectively, and then used to compute the cross-modal covariance matrix $\mathbf{Cov}_x$.
    - $\mathbf{Cov}_x(i,j)$ measures the sensitivity of the $i$-th and $j$-th channels across modalities.
    - High-variance channels contain more attention directed at non-corresponding regions and are suppressed via a selective mask $M_x$.
    - Only the upper triangular portion is optimized to prevent the model from overfitting to modality-specific statistics.
    - The filtered features are fused with the original features to preserve information integrity.
    - Filtering loss: $L_f = \frac{1}{X}\sum_{x=1}^X (\|\hat{\mathbf{V}}^x_I \odot M_x\|_1 + \|\hat{\mathbf{V}}^x_P \odot M_x\|_1)$

3. **Global Optimal Selection Module (GOS)**:

    - The patch-level matching problem is formulated as an optimal transport (OT) problem.
    - The similarity cost matrix is $(1-S)$, and the globally optimal transport plan is solved via Sinkhorn iterations:
    $T^* = \min_{T \in \mathcal{T}} \text{Tr}(T^T(1-S)) - \epsilon H(\mathcal{T})$
    - The constraint $\mathcal{T} = \{T | T\mathbf{1} = \frac{1}{N}\mathbf{1}, T^T\mathbf{1} = \frac{1}{N}\mathbf{1}\}$ enforces uniform marginal distributions.
    - Convergence is achieved in approximately 10 Sinkhorn iterations, enabling efficient GPU-parallelized computation.
    - Matching is optimized from a global perspective, effectively reducing many-to-one errors.

### Loss & Training

The total loss comprises three terms:
$$L_{\text{total}} = \lambda_1 L_f + \lambda_2 L_{ic} + \lambda_3 L_{if}$$
- $L_f$: channel filtering loss from CMCF
- $L_{ic}$: Circle Loss for coarse matching
- $L_{if}$: Circle Loss for fine matching

Circle Loss balances learning on hard samples through adaptive weighting of positive and negative pairs.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (2D3D-MATR) | Gain |
|--------|------|------|-----------|------|
| RGB-D Scenes V2 | IR↑ | 35.5 | 32.4 | +3.1pp |
| RGB-D Scenes V2 | FMR↑ | 93.6 | 90.8 | +2.8pp |
| RGB-D Scenes V2 | RR↑ | 63.3 | 56.4 | +6.9pp |
| 7-Scenes | IR↑ | 51.6 | 50.1 | +1.5pp |
| 7-Scenes | FMR↑ | 92.4 | 92.1 | +0.3pp |
| 7-Scenes | RR↑ | 79.5 | 75.8 | +3.7pp |

Compared with FreeReg: RR improves by 6pp (63.3 vs. 57.3) and FMR improves by 12pp.

### Ablation Study

| Configuration | PIR | IR | FMR | RR | Note |
|------|-----|-----|-----|-----|------|
| Baseline (M1) | 48.5 | 32.5 | 91.0 | 55.8 | No enhancement |
| +ICE (M2) | 56.3 | 34.6 | 92.4 | 56.9 | Image channel enhancement |
| +PCE (M3) | 54.7 | 33.6 | 93.2 | 56.0 | Point cloud channel enhancement |
| +ICE+PCE (M4) | 56.3 | 34.6 | 92.3 | 58.2 | Joint enhancement |
| +CMCF (M5) | 59.2 | 35.4 | 91.4 | 59.7 | Cross-modal filtering |
| +CAA (M6) | 59.1 | 35.1 | 93.3 | 61.8 | Full channel adaptation |
| +GOS (M7) | 56.3 | 35.3 | 91.7 | 58.1 | Global optimal selection |
| Full CA-I2P (M8) | 58.3 | 35.5 | 93.6 | 63.3 | All components |

CMCF contributes the largest PIR gain (+10.7pp); GOS contributes the largest IR gain (+2.8pp); overall RR improves by +7.5pp.

### Key Findings

- Intra-modal feature enhancement and cross-modal channel filtering each contribute significantly and in a complementary manner.
- The Instance Normalization combined with covariance matrix analysis in CMCF effectively reveals channel-level modality discrepancies.
- The optimal transport strategy in GOS is particularly effective in indoor scenes with repetitive structures.
- The most pronounced improvements are observed on the most challenging scenes, Heads (close-range with amplified small errors) and Stairs (repetitive patterns).
- The proposed method outperforms approaches that additionally incorporate overlap region detectors in indoor settings.

## Highlights & Insights

- Addressing cross-modal feature alignment from the channel dimension is a novel and effective perspective; prior work has predominantly focused on the spatial dimension.
- The three-branch rotation design in ICE elegantly captures channel–spatial cross-dependencies while remaining lightweight.
- Covariance matrix analysis identifies which channels are sensitive to modality variation, providing an interpretable basis for feature filtering.
- Replacing top-k selection with optimal transport elevates a locally greedy problem to a globally optimized one.

## Limitations & Future Work

- Evaluation is limited to indoor datasets; applicability to large-scale outdoor scenes (e.g., KITTI) remains unverified.
- The three-tier thresholding strategy for mask $M_x$ in CMCF may lack flexibility; learnable soft masks could be explored.
- Although Sinkhorn iterations are efficient, computational bottlenecks may arise in very large-scale matching scenarios.
- The authors acknowledge that indoor and outdoor methods generally do not transfer directly, and generalizability warrants further investigation.

## Related Work & Insights

- The coarse-to-fine framework of 2D3D-MATR provides a strong baseline upon which CA-I2P applies channel-level enhancements.
- Detector-free methods such as LoFTR and GeoTransformer inspired the detector-free design for the I2P task.
- Optimal transport has been employed for 2D matching in SuperGlue; this work is the first to introduce it to patch-level matching in I2P.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of channel adaptivity and global optimal selection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations on two benchmarks; introduces the new PIR metric.
- Writing Quality: ⭐⭐⭐ Technical descriptions are adequate but notation is inconsistent in places.
- Value: ⭐⭐⭐⭐ Proposes an effective channel-level solution to the cross-modal registration problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[ICLR 2026\] COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](../../ICLR2026/3d_vision/coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)
- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](../../NeurIPS2025/3d_vision/mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)
- [\[CVPR 2026\] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](../../CVPR2026/3d_vision/hg-i2p_bridging_modalities_for_generalizable_image-to-point-cloud_registration_v.md)

</div>

<!-- RELATED:END -->
