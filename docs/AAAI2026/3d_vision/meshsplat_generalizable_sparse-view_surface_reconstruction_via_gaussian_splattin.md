---
title: >-
  [Paper Note] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][sparse-view reconstruction] This paper proposes MeshSplat, the first generalizable sparse-view surface reconstruction framework based on 2DGS. It regularizes depth prediction via a Weighted Chamfer Distance loss and aligns 2DGS orientations through an uncertainty-guided normal prediction network, learning geometric priors in a self-supervised manner from novel view synthesis. MeshSplat achieves state-of-the-art performance on both sparse-view mesh reconstruction and cross-dataset generalization.
tags:
  - AAAI 2026
  - 3D Vision
  - sparse-view reconstruction
  - surface reconstruction
  - 2D Gaussian splatting
  - feed-forward network
  - cross-scene generalization
date: 2026-05-08
content_hash: 0c420ef9db4dd577
---

# MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2508.17811](https://arxiv.org/abs/2508.17811)
**Code**: [https://hanzhichang.github.io/meshsplat_web/](https://hanzhichang.github.io/meshsplat_web/)
**Area**: 3D Vision
**Keywords**: sparse-view reconstruction, surface reconstruction, 2D Gaussian splatting, feed-forward network, cross-scene generalization

## TL;DR

This paper proposes MeshSplat, the first generalizable sparse-view surface reconstruction framework based on 2DGS. It regularizes depth prediction via a Weighted Chamfer Distance loss and aligns 2DGS orientations through an uncertainty-guided normal prediction network, learning geometric priors in a self-supervised manner from novel view synthesis. MeshSplat achieves state-of-the-art performance on both sparse-view mesh reconstruction and cross-dataset generalization.

## Background & Motivation

3D surface reconstruction is a fundamental task in 3D vision, critical for applications such as AR/VR and embodied AI. Per-scene optimization methods based on NeRF/3DGS perform poorly under sparse-view settings, where limited multi-view geometric constraints are insufficient to support high-quality per-scene geometry optimization.

**Limitations of Prior Work**:

**NeuS-based methods** (e.g., SparseNeuS): estimate implicit SDF fields via geometric voxels and extract meshes. They suffer from low efficiency due to implicit representations, slow rendering speeds, and are restricted to object-level scenes.

**3DGS feed-forward methods** (e.g., pixelSplat, MVSplat): effective for novel view synthesis, but the ellipsoidal shape of 3DGS produces different cross-sectional planes across viewpoints, causing **surface inconsistency** and making mesh extraction unreliable.

**Core Idea**: 2DGS (2D Gaussian Splatting) serves as a natural bridge between NVS and surface reconstruction. Unlike 3DGS, 2DGS maintains consistent cross-sectional planes across viewpoints and is inherently better suited to represent thin surfaces, enabling simultaneous novel view synthesis and mesh extraction. However, integrating 2DGS into a feed-forward framework is non-trivial—2DGS is **more sensitive** to errors in position and orientation estimation:

- **Position sensitivity**: The thin nature of 2DGS means that depth prediction errors directly cause pronounced positional offsets (3DGS tolerates larger errors due to its volumetric extent).
- **Orientation sensitivity**: The orientation of 2DGS directly determines scene surface normals; incorrect orientation prediction leads to distorted surfaces.

## Method

### Overall Architecture

Given two images and their projection matrices, MeshSplat proceeds as follows:
1. A CNN with a Multi-View Transformer extracts feature maps.
2. Plane sweeping constructs per-view cost volumes.
3. A Weighted Chamfer Distance loss regularizes the cost volumes.
4. A Gaussian Prediction Network (comprising a depth refinement network and a normal prediction network) generates pixel-aligned 2DGS.
5. 2DGS renders novel views for supervision and extracts scene meshes.

Formally: $\{I_i, \Pi_i\}_{i=1}^{2} \rightarrow \{\mu_j, s_j, r_j, \alpha_j, c_j\}_{j=1}^{2 \times H \times W}$

### Key Designs

#### 1. **Cost Volume Construction and Depth Prediction**

Following the MVSplat framework, plane sweeping is used to construct cost volumes. For input view $i$, the depth range is discretized into $D=128$ depth candidates. Features from the other view are warped to the current view at each depth candidate, and dot products are computed to obtain the cost volume:

$$V_i^{d_k} = \frac{F_i \cdot F_{j \to i}^{d_k}}{\sqrt{C}}$$

A Softmax over the depth dimension yields depth probabilities, which are used in a weighted sum to produce the coarse depth map:

$$D_i^{\text{coarse}} = \sum_k W_i^k d_k$$

#### 2. **Weighted Chamfer Distance Loss (WCD Loss)**

Ideally, Gaussians predicted from adjacent views should exhibit substantial overlap. Standard Chamfer Distance assigns equal weight to all points; however, due to occlusion and viewpoint differences, non-corresponding pixels produce large chamfer distances, and uniform weighting introduces erroneous constraints.

**Solution**: A per-pixel matching confidence map is extracted from the cost volume:

$$M_i = \max_{d_k} \text{Softmax}_D(V_i)$$

The WCD loss applies strong constraints only in high-confidence regions:

$$\mathcal{L}_{\text{WCD}} = \frac{1}{2}\left(\frac{1}{N_1}\sum_{i=1}^{N_1} M_1(i)\min_j ||p_1^i - p_2^j|| + \frac{1}{N_2}\sum_{i=1}^{N_2} M_2(i)\min_j ||p_2^i - p_1^j||\right)$$

The confidence map clearly identifies texture-less and non-overlapping regions (low confidence), preventing erroneous constraints from being applied to these areas.

#### 3. **Uncertainty-Guided Normal Prediction Network**

The orientation of 2DGS directly determines scene surface normals. A lightweight CNN $\phi_{\text{rot}}$ is designed to predict the rotation quaternion $q$ and uncertainty $\kappa$ for each 2DGS:

$$\{q, \kappa\} = \phi_{\text{rot}}(V_i || F_i || I_i), \quad n = R(q) \cdot [0, 0, 1]^T$$

Supervision is provided via the negative log-likelihood (NLL) loss of the Angular von Mises-Fisher distribution:

$$\mathcal{L}_{\text{AngMF}}(n_i, \hat{n}_i, \kappa_i) = -\log(\kappa_i^2 + 1) + \log(1 + \exp(-\kappa_i\pi)) + \kappa_i \cos^{-1} n_i^T \hat{n}_i$$

Pseudo ground-truth normals are provided by a pretrained Omnidata model. Uncertainty-guided sampling based on $\kappa$ selects the top 70% of pixels with the lowest $\kappa$ along with a random 30% for loss computation.

### Loss & Training

The total training loss is:

$$\mathcal{L} = w_1\mathcal{L}_{\text{pho}} + w_2\mathcal{L}_{\text{WCD}} + w_3\mathcal{L}_{\text{normal}}$$

where $\mathcal{L}_{\text{pho}} = w_{11}\text{MSE}(I, \hat{I}) + w_{12}\text{LPIPS}(I, \hat{I})$

Loss weights: $w_1=1.0$, $w_2=5.0\times10^{-3}$, $w_3=5.0\times10^{-3}$, $w_{11}=1.0$, $w_{12}=0.1$

Training configuration:
- Re10K: cropped to 256×256, trained for 200k steps, batch size 12
- ScanNet: cropped to 512×384, trained for 75k steps, batch size 4
- Adam optimizer, maximum learning rate $2\times10^{-4}$
- Single NVIDIA A800 GPU

## Key Experimental Results

### Main Results

Surface reconstruction on Re10K and ScanNet:

| Method | Re10K CD↓ | Re10K F1↑ | ScanNet CD↓ | ScanNet F1↑ |
|--------|-----------|-----------|-------------|-------------|
| **MeshSplat** | **0.3566** | **0.3758** | **0.2606** | **0.3824** |
| MVSplat | 0.4015 | 0.3100 | 0.3748 | 0.2095 |
| pixelSplat | 1.4423 | 0.0944 | 0.3285 | 0.2948 |
| MVSNeRF | 0.6139 | 0.1407 | 0.5761 | 0.1514 |
| SparseNeuS | 6.0473 | 0.0020 | 7.1860 | 0.0107 |

Cross-dataset zero-shot transfer (trained on Re10K only):

| Method | Re10K→ScanNet F1↑ | Re10K→Replica F1↑ |
|--------|-------------------|-------------------|
| **MeshSplat** | **0.2956** | **0.0809** |
| MVSplat | 0.1418 | 0.0564 |
| SparseNeuS | 0.0006 | 0.0003 |

Depth and normal prediction quality:

| Method | Depth AbsRel↓ | Normal Mean↓ | Normal <30°↑ |
|--------|--------------|--------------|--------------|
| **MeshSplat** | **0.0910** | **33.84** | **0.6026** |
| MVSplat | 0.1692 | 57.16 | 0.1357 |

### Ablation Study

Ablation on ScanNet:

| # | Configuration | CD↓ | Note |
|---|---------------|-----|------|
| 1 | 3DGS (MVSplat baseline) | 0.3748 | Baseline |
| 2 | 2DGS | 0.2948 | 2DGS better suited for surface reconstruction |
| 3 | 2DGS + WCD Loss | 0.2769 | Cross-view depth consistency improved |
| 4 | 2DGS + NPN | 0.2642 | Normal prediction network contributes most |
| 5 | 2DGS + WCD + NPN | **0.2606** | Two components are complementary |

Model efficiency:

| Method | Rendering Time (s) | Parameters (M) |
|--------|--------------------|----------------|
| MeshSplat | 0.102 | 13.3 |
| MVSplat | 0.072 | 12.0 |
| SparseNeuS | 7.048 | 0.843 |

### Key Findings

- **2DGS vs. 3DGS**: Simply replacing 3DGS with 2DGS reduces CD from 0.3748 to 0.2948, validating 2DGS as an effective bridge between NVS and surface reconstruction.
- **The normal prediction network contributes most** (CD: 0.2948→0.2642), highlighting the critical influence of 2DGS orientation on mesh quality.
- The WCD loss effectively addresses erroneous constraints in non-overlapping regions; the confidence map accurately identifies texture-less and non-overlapping areas.
- The method adds only 1.3M parameters and 30ms rendering time, incurring minimal overhead.
- Cross-dataset generalization: training on Re10K and transferring zero-shot to ScanNet/Replica yields F1 scores substantially higher than baselines.
- High-uncertainty regions in the $\kappa$ map typically correspond to object boundaries, consistent with intuition.

## Highlights & Insights

1. **2DGS as a bridge**: The framework converts the richness of NVS training data into geometric priors for surface reconstruction, elegantly avoiding the need for expensive 3D ground-truth annotations.
2. **Elegant WCD loss design**: The confidence map is naturally derived from the cost volume without requiring additional modules.
3. **Uncertainty-guided sampling**: Sampling based on $\kappa$ in the normal loss directs the network to focus learning on uncertain regions, improving training efficiency.
4. **Self-supervised geometric learning**: The entire framework requires no 3D ground truth and learns geometry solely through NVS supervision.

## Limitations & Future Work

- Weakly textured regions may yield discontinuous depth maps, even when RGB rendering remains reliable.
- The method cannot reconstruct regions not observed in the input views.
- Only two input images are used; incorporating more views could further improve performance.
- Generative approaches for completing unobserved regions have not been explored.
- Re10K lacks ground-truth meshes; dense point clouds reconstructed by COLMAP are used as approximate ground truth.

## Related Work & Insights

- MVSplat is the most direct baseline, sharing the same feed-forward framework but using 3DGS.
- 2DGS (Huang et al.) demonstrated its advantage for surface reconstruction in per-scene optimization settings; this work is the first to extend it to a generalizable setting.
- DUSt3R/MASt3R can predict 3D point maps but do not support novel view synthesis or surface reconstruction.
- Insight: 2DGS holds potential in other feed-forward 3D tasks, such as panoramic reconstruction and object-level reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of 2DGS to generalizable sparse-view surface reconstruction
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation, cross-dataset generalization, depth/normal assessment, and ablation studies
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation; 2DGS vs. 3DGS comparison is intuitive
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction for feed-forward reconstruction with 2DGS; high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
