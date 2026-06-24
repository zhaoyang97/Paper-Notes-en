---
title: >-
  [Paper Note] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][Sparse-view reconstruction] MeshSplat is proposed as the first generalizable sparse-view surface reconstruction framework based on 2D GS. By introducing a weighted Chamfer Distance loss to regularize depth predictions and an uncertainty-based normal prediction network to align 2D GS orientations, it learns geometric priors from novel view synthesis in a self-supervised manner, achieving state-of-the-art performance in sparse-view mesh reconstruction and…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Sparse-view reconstruction"
  - "Surface reconstruction"
  - "2D Gaussian Splatting"
  - "Feed-forward networks"
  - "Cross-scene generalization"
date: 2026-05-08
content_hash: 96f540d418da4b00
---

# MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2508.17811](https://arxiv.org/abs/2508.17811)  
**Code**: [https://hanzhichang.github.io/meshsplat_web/](https://hanzhichang.github.io/meshsplat_web/)  
**Area**: 3D Vision  
**Keywords**: Sparse-view reconstruction, Surface reconstruction, 2D Gaussian Splatting, Feed-forward networks, Cross-scene generalization

## TL;DR

MeshSplat is proposed as the first generalizable sparse-view surface reconstruction framework based on 2D GS. By introducing a weighted Chamfer Distance loss to regularize depth predictions and an uncertainty-based normal prediction network to align 2D GS orientations, it learns geometric priors from novel view synthesis in a self-supervised manner, achieving state-of-the-art performance in sparse-view mesh reconstruction and cross-dataset generalization.

## Background & Motivation

3D scene surface reconstruction is a fundamental task in 3D vision, critical for applications such as AR/VR and embodied AI. Scene-specific optimization methods based on NeRF or 3D GS perform poorly under sparse views, where limited multi-view geometric constraints fail to support high-quality per-scene geometric optimization.

**Limitations of Prior Work**:

**NeuS-based methods** (e.g., SparseNeuS): Extract meshes by estimating implicit SDF fields through geometric voxels. They suffer from low representation efficiency, slow rendering speeds, and are often restricted to object-level scenes.

**Feed-forward 3D GS methods** (e.g., pixelSplat, MVSplat): Perform well in novel view synthesis but suffer from **surface inconsistency** because the ellipsoidal shape of 3D Gaussians produces different intersecting planes under different viewpoints, preventing effective mesh extraction.

**Key Insight**: 2D GS (2D Gaussian Splatting) serves as a natural bridge between novel view synthesis (NVS) and surface reconstruction. Having consistent intersecting planes across different viewpoints, 2D GS is inherently better suited for representing thin surfaces, enabling simultaneous NVS and mesh extraction. However, integrating 2D GS into a feed-forward framework is challenging because 2D GS is **more sensitive** to position and orientation estimation:

- **Position Sensitivity**: The thin nature of 2D GS means that even minor errors in depth estimation directly cause noticeable position offsets, whereas 3D GS can tolerate larger errors due to its ellipsoidal volume.
- **Orientation Sensitivity**: The orientation of 2D GS directly determines the scene surface normals, and incorrect orientation predictions lead to highly distorted scene surfaces.

## Method

### Overall Architecture

Given two input images and their projection matrices, the pipeline of MeshSplat is as follows:
1. Extract feature maps using a CNN and a Multi-View Transformer.
2. Construct a per-view cost volume via plane sweeping.
3. Apply a Weighted Chamfer Distance Loss to regularize the cost volume.
4. Use a Gaussian Prediction Network (comprising a depth refinement network and a normal prediction network) to generate pixel-aligned 2D GS.
5. Render novel views with 2D GS for supervision and extract the scene mesh.

Formal formulation: $\{I_i, \Pi_i\}_{i=1}^{2} \rightarrow \{\mu_j, s_j, r_j, \alpha_j, c_j\}_{j=1}^{2 \times H \times W}$

### Key Designs

#### 1. **Cost Volume Construction and Depth Prediction**

Following the framework of MVSplat, plane sweeping is introduced to construct the cost volume. For the input view $i$, the depth range is discretized into $D=128$ depth candidates. The feature map of the other view is warped according to the current depth candidate, and the dot product is calculated to obtain the cost volume:

$$V_i^{d_k} = \frac{F_i \cdot F_{j \to i}^{d_k}}{\sqrt{C}}$$

Softmax is applied along the depth dimension of the cost volume to obtain depth probabilities, and a weighted sum yields the coarse depth map:

$$D_i^{\text{coarse}} = \sum_k W_i^k d_k$$

#### 2. **Weighted Chamfer Distance Loss (WCD Loss)**

Ideally, the predicted Gaussian positions from adjacent views should exhibit substantial overlap. Standard Chamfer Distance assigns equal weights to all points. However, due to occlusions and view differences, non-corresponding pixels can have large Chamfer distances, and applying a uniform constraint leads to unreasonable optimization behaviors.

**Solution**: A match confidence map is extracted for each pixel from the cost volume:

$$M_i = \max_{d_k} \text{Softmax}_D(V_i)$$

The WCD Loss imposes strong constraints only on regions with high confidence:

$$\mathcal{L}_{\text{WCD}} = \frac{1}{2}\left(\frac{1}{N_1}\sum_{i=1}^{N_1} M_1(i)\min_j ||p_1^i - p_2^j|| + \frac{1}{N_2}\sum_{i=1}^{N_2} M_2(i)\min_j ||p_2^i - p_1^j||\right)$$

The confidence map clearly highlights textureless and non-overlapping regions (exhibiting low confidence), preventing erroneous constraints in these areas.

#### 3. **Uncertainty-Guided Normal Prediction Network**

The orientation of 2D GS directly determines the scene surface normals. A lightweight CNN $\phi_{\text{rot}}$ is designed to predict the rotation quaternion $q$ and uncertainty $\kappa$ of the 2D GS:

$$\{q, \kappa\} = \phi_{\text{rot}}(V_i || F_i || I_i), \quad n = R(q) \cdot [0, 0, 1]^T$$

Supervision is applied using the negative log-likelihood (NLL) loss of the Angular von Mises-Fisher distribution:

$$\mathcal{L}_{\text{AngMF}}(n_i, \hat{n}_i, \kappa_i) = -\log(\kappa_i^2 + 1) + \log(1 + \exp(-\kappa_i\pi)) + \kappa_i \cos^{-1} n_i^T \hat{n}_i$$

The output from a pre-trained Omnidata model serves as the pseudo-ground-truth normal supervision. An uncertainty-guided sampling strategy based on $\kappa$ is adopted: the top 70% pixels with the lowest uncertainty $\kappa$ along with 30% randomly sampled pixels are selected for loss computation.

### Loss & Training

Total training loss:

$$\mathcal{L} = w_1\mathcal{L}_{\text{pho}} + w_2\mathcal{L}_{\text{WCD}} + w_3\mathcal{L}_{\text{normal}}$$

where $\mathcal{L}_{\text{pho}} = w_{11}\text{MSE}(I, \hat{I}) + w_{12}\text{LPIPS}(I, \hat{I})$

Weight settings: $w_1=1.0$, $w_2=5.0\times10^{-3}$, $w_3=5.0\times10^{-3}$, $w_{11}=1.0$, $w_{12}=0.1$

Training strategy:
- Re10K: Cropped to 256×256, trained for 200k steps with a batch size of 12.
- Scannet: Cropped to 512×384, trained for 75k steps with a batch size of 4.
- Optimizer: Adam, with a peak learning rate of $2\times10^{-4}$.
- Hardware: A single NVIDIA A800 GPU.

## Key Experimental Results

### Main Results

Surface reconstruction results on Re10K and Scannet datasets:

| Method | Re10K CD↓ | Re10K F1↑ | Scannet CD↓ | Scannet F1↑ |
|------|----------|----------|------------|------------|
| **MeshSplat** | **0.3566** | **0.3758** | **0.2606** | **0.3824** |
| MVSplat | 0.4015 | 0.3100 | 0.3748 | 0.2095 |
| pixelSplat | 1.4423 | 0.0944 | 0.3285 | 0.2948 |
| MVSNeRF | 0.6139 | 0.1407 | 0.5761 | 0.1514 |
| SparseNeuS | 6.0473 | 0.0020 | 7.1860 | 0.0107 |

Cross-dataset zero-shot transfer (trained only on Re10K):

| Method | Re10K→Scannet F1↑ | Re10K→Replica F1↑ |
|------|-------------------|-------------------|
| **MeshSplat** | **0.2956** | **0.0809** |
| MVSplat | 0.1418 | 0.0564 |
| SparseNeuS | 0.0006 | 0.0003 |

Depth and normal prediction quality:

| Method | Depth AbsRel↓ | Normal Mean↓ | Normal <30°↑ |
|------|-------------|-------------|-------------|
| **MeshSplat** | **0.0910** | **33.84** | **0.6026** |
| MVSplat | 0.1692 | 57.16 | 0.1357 |

### Ablation Study

Ablation study on Scannet dataset:

| # | Configuration | CD↓ | Description |
|---|------|-----|------|
| 1 | 3D GS (MVSplat baseline) | 0.3748 | Baseline |
| 2 | 2D GS | 0.2948 | 2D GS is better suited for surface reconstruction |
| 3 | 2D GS + WCD Loss | 0.2769 | Dynamic cross-view depth consistency improvement |
| 4 | 2D GS + NPN | 0.2642 | Normal prediction network contributes the most |
| 5 | 2D GS + WCD + NPN | **0.2606** | The two are complementary |

Model efficiency:

| Method | Rendering Time (s) | Parameters (M) |
|------|-----------|----------|
| MeshSplat | 0.102 | 13.3 |
| MVSplat | 0.072 | 12.0 |
| SparseNeuS | 7.048 | 0.843 |

### Key Findings

- **2D GS vs 3D GS**: Simply replacing 3D GS with 2D GS reduces CD from 0.3748 to 0.2948, validating the efficacy of 2D GS as a bridge between NVS and surface reconstruction.
- **The normal prediction network contributes the most** (reducing CD from 0.2948 to 0.2642), highlighting the critical impact of 2D GS orientation on mesh quality.
- The WCD Loss effectively addresses the incorrect constraint issues in non-overlapping regions, as the confidence map accurately reflects both textureless and non-overlapping areas.
- The model introduces minimal overhead, requiring only an additional 1.3M parameters and 30ms rendering time.
- Cross-dataset generalization: Zero-shot transfer from Re10K to Scannet and Replica shows significantly superior F1 scores compared to baselines.
- High-uncertainty regions in the predicted $\kappa$ map typically correspond to object boundaries, which aligns with intuition.

## Highlights & Insights

1. **Insight on 2D GS as a Bridge**: Leverages the rich training data of NVS to derive geometric priors for surface reconstruction, cleverly avoiding costly 3D ground-truth labels.
2. **Elegant Design of WCD Loss**: Automatically derives confidence maps directly from the cost volume without requiring extra modules.
3. **Uncertainty-Guided Sampling**: Samples based on $\kappa$ for the normal loss to allow the network to focus on highly uncertain regions, thereby improving training efficiency.
4. **Self-Supervised Geometric Learning**: The entire framework does not require 3D ground truth, learning geometry purely under NVS supervision.

## Limitations & Future Work

- Weakly textured regions may yield discontinuous depth maps (even though RGB rendering remains reliable).
- Unobserved regions from input views cannot be reconstructed.
- Only two input images are used; leveraging more view inputs may yield further improvements.
- Generative methods for hallucinating or completing unseen regions are yet to be explored.
- The Re10K dataset lacks ground-truth meshes, requiring dense point clouds reconstructed via COLMAP to serve as approximate ground truths.

## Related Work & Insights

- MVSplat is the most direct baseline (employing a similar feed-forward framework but using 3D GS).
- 2D GS (Huang et al.) demonstrated its advantages in surface reconstruction within per-scene optimization setups; this work is the first to extend it to a generalizable setting.
- While models like DUSt3R and MASt3R can predict 3D point maps, they do not support novel view synthesis and surface reconstruction.
- Inspiration: 2D GS holds significant potential for other feed-forward 3D tasks (e.g., panorama reconstruction, object-level reconstruction).

## Rating

- Novelty: ⭐⭐⭐⭐ — First to apply 2D GS to generalizable sparse-view surface reconstruction
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation + cross-dataset generalization + depth/normal evaluation + ablation study
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, with an intuitive comparison between 2D GS and 3D GS
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction for feed-forward 2D GS reconstruction, carrying high practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[CVPR 2026\] Generalizable Sparse-View 3D Reconstruction from Unconstrained Images](../../CVPR2026/3d_vision/generalizable_sparse-view_3d_reconstruction_from_unconstrained_images.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[ICLR 2026\] D²GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction](../../ICLR2026/3d_vision/d2gs_depth-and-density_guided_gaussian_splatting_for_stable_and_accurate_sparse-.md)
- [\[CVPR 2026\] FSFSplatter: Geometrically Accurate Reconstruction with Free Sparse-view Images within 2 minutes](../../CVPR2026/3d_vision/fsfsplatter_geometrically_accurate_reconstruction_with_free_sparse-view_images_w.md)

</div>

<!-- RELATED:END -->
