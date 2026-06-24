---
title: >-
  [Paper Note] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] SplatFields finds that the performance bottleneck of 3D Gaussian Splatting (3DGS) in sparse-view settings stems from the lack of spatial autocorrelation in splat features. It proposes to introduce spatial regularization by predicting splat features through an implicit neural field, consistently improving reconstruction quality in both static 3D and dynamic 4D sparse reconstruction scenarios.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Sparse-view Reconstruction"
  - "Implicit Neural Field Regularization"
  - "Spatial Autocorrelation"
  - "Dynamic Reconstruction"
date: 2026-05-08
content_hash: 07cd09aa3952cc94
---

# SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction

**Conference**: ECCV 2024  
**arXiv**: [2409.11211](https://arxiv.org/abs/2409.11211)  
**Code**: [https://github.com/markomih/SplatFields](https://github.com/markomih/SplatFields)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Sparse-view Reconstruction, Implicit Neural Field Regularization, Spatial Autocorrelation, Dynamic Reconstruction

## TL;DR

SplatFields finds that the performance bottleneck of 3D Gaussian Splatting (3DGS) in sparse-view settings stems from the lack of spatial autocorrelation in splat features. It proposes to introduce spatial regularization by predicting splat features through an implicit neural field, consistently improving reconstruction quality in both static 3D and dynamic 4D sparse reconstruction scenarios.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has rapidly emerged as a popular method in the field of 3D/4D reconstruction due to its exceptional reconstruction quality, real-time rendering capability, and compatibility with mainstream pipelines. However, the superior performance of 3DGS relies on a large number of input views (usually dozens to hundreds). In practical applications, acquiring a large number of views is often expensive—this is particularly evident in dynamic scene capturing, where deploying large-scale camera arrays is extremely high-cost.

**Limitations of Prior Work**: (1) The performance of 3DGS degrades significantly in sparse-view settings, exhibiting severe overfitting and artifacts because the few views provide insufficient supervision signals to constrain tens of thousands of free parameters. (2) Existing 3DGS regularization methods (such as depth supervision, normal vector constraints) are often designed for specific scenarios and lack generalizability. (3) The properties of each Gaussian in 3DGS (position, color, opacity, etc.) are optimized independently, with no information sharing or consistency constraints between adjacent Gaussians. (4) This independent parameterization easily leads to noisy and discontinuous reconstruction results under sparse views.

**Key Challenge**: The expressiveness of 3DGS is very strong (each Gaussian has 14+ free parameters), which is an advantage under dense views but becomes a disadvantage under sparse views—excessive degrees of freedom lead to severe overfitting.

**Goal**: (1) How to regularize the sparse reconstruction of 3DGS without significantly increasing computational overhead? (2) How to enable spatial consistency among adjacent Gaussians? (3) Can the method be simultaneously applied to both static 3D and dynamic 4D scenes?

**Key Insight**: The authors observe that the splat features of 3DGS (color, opacity, shape, etc.) lack spatial autocorrelation—meaning the properties of adjacent Gaussians can be completely uncorrelated. This contradicts the physical reality that object surface properties usually change smoothly in real scenes. The spatial continuity inherently possessed by an implicit neural field (which takes coordinates as input and outputs splat features) is leveraged to regularize splat properties.

**Core Idea**: To constrain the spatial continuity of Gaussian properties using an implicit neural field, transforming the sparse-view 3DGS reconstruction from "independent parameterization" into "field prediction."

## Method

### Overall Architecture

SplatFields adds an implicit neural field module to the standard 3DGS optimization pipeline. The positions of 3D Gaussians are still optimized independently, but other attributes (spherical harmonics/color, opacity, scale, rotation, etc.) are predicted by a neural field that takes the Gaussian positions as input. During optimization, gradients not only update the Gaussian properties but also backpropagate to the weights of the neural field. The continuity of the neural field naturally introduces spatial regularization. For 4D scenes, positions are additionally processed using a time-conditioned deformation MLP.

### Key Designs

1. **Neural Field Feature Prediction**:

    - **Function**: Introduces spatial autocorrelation regularization for 3DGS.
    - **Mechanism**: An implicit neural field $f_\theta: \mathbb{R}^3 \rightarrow \mathbb{R}^d$ is defined, taking the 3D position $\mu_i$ of the Gaussian as input to predict its feature vector (containing color/spherical harmonics, opacity, scale, and rotation parameters). The input position is mapped to the feature space via multi-resolution hash encoding (e.g., Instant-NGP style) and decoded by a small MLP. Standard 3DGS splatting pipeline is still used for rendering.
    - **Design Motivation**: The inherent characteristic of implicit neural fields is spatial continuity—MLPs produce similar outputs for similar inputs. This means that spatially adjacent Gaussians will automatically receive similar property values, thereby acting as a regularizer in sparse views and preventing individual Gaussians from overfitting to a small number of training views.

2. **Hybrid Optimization**:

    - **Function**: Balances the flexibility of standard 3DGS and the constraints of neural field regularization.
    - **Mechanism**: The position parameters of the Gaussians are still directly optimized (without going through the neural field) to maintain the flexibility of 3DGS in position adjustment. Only the attribute parameters (appearance and shape) are predicted via the neural field. Strong neural field constraints are enforced in the early stages of training, and are gradually relaxed later to allow for finer fitting. Optionally, the training can switch to the direct parameter optimization of standard 3DGS in the late stage to restore details.
    - **Design Motivation**: Predicting positions through the neural field would severely restrict the spatial distribution flexibility of the Gaussians, as densification/pruning operations require direct position manipulation. In contrast, appearance attributes naturally should possess spatial softness.

3. **4D Dynamic Scene Extension**:

    - **Function**: Extends the regularization strategy of SplatFields to dynamic (4D) scene reconstruction.
    - **Mechanism**: A time-conditioned deformation field is added on top of the 3D static version. The Gaussian position at each time step is adjusted via a learned deformation $\Delta \mu = g_\phi(\mu, t)$, while the feature neural field also accepts time-conditioned input $f_\theta(\mu + \Delta\mu, t)$. The deformation field and the feature field share the spatial dimensions of the hash encoding.
    - **Design Motivation**: The challenge of dynamic scenes under sparse camera arrays is even more severe—not only are there fewer spatial views, but there are also fewer observations at the same time step. Temporal constraints are crucial for maintaining temporal consistency.

### Loss & Training

Standard 3DGS training loss (a weighted combination of L1 + D-SSIM) is used. The key difference is that gradients backpropagate through the neural field to update network weights, thus achieving implicit regularization. No additional regularization loss terms are required—regularization arises entirely from the architectural inductive bias of the neural field. The hash encoding uses a multi-resolution configuration to capture spatial variations at different scales. The training efficiency is comparable to standard 3DGS, with minimal extra overhead from the neural field.

## Key Experimental Results

### Main Results

Sparse-view reconstruction on DTU dataset (3 input views):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS | 14.87 | 0.621 | 0.387 |
| FSGS | 17.21 | 0.715 | 0.312 |
| DNGaussian | 17.85 | 0.731 | 0.295 |
| **Ours** | **18.63** | **0.752** | **0.271** |

NeRF Synthetic dataset (8 input views):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS | 25.12 | 0.926 | 0.054 |
| Ours (3DGS) | 26.45 | 0.938 | 0.043 |
| 3DGS + Depth Supervision | 26.21 | 0.934 | 0.047 |
| **Ours + Depth** | **27.12** | **0.945** | **0.038** |

Dynamic scenes (DNA-Rendering dataset, 4 cameras):

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| 4D-GS | 28.35 | 0.945 |
| **Ours-4D** | **29.87** | **0.958** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| Standard 3DGS | 14.87 | 0.621 | No regularization, severe overfitting |
| + Color Neural Field | 16.42 | 0.691 | Only color is predicted through the field |
| + All-Attribute Neural Field | 18.12 | 0.740 | Color + opacity + shape |
| + Hash Encoding | 18.63 | 0.752 | Full Ours |
| + Position also through field | 17.95 | 0.728 | Position constraint is too strong, leading to performance drop |

### Key Findings

- Spatial autocorrelation is indeed the key performance bottleneck for sparse 3DGS reconstruction—simply increasing regularization loss is less effective than introducing continuity from the architectural level.
- The best results are achieved when all attributes (color, opacity, scale, rotation) except position are predicted from the neural field.
- SplatFields is complementary to other regularization methods (such as depth supervision) and can be used in combination.
- The improvement is even more significant in 4D scenes, where sparse cameras plus the temporal dimension make overfitting much more severe.
- The extra computational overhead of the neural field is less than 10%, offering extremely high cost-effectiveness.
- The sparser the views (e.g., 3 views vs. 8 views), the larger the performance gain of SplatFields compared to the baseline.

## Highlights & Insights

- **Accurate Problem Diagnosis**: Analyzes the cause of failed sparse 3DGS reconstruction from the perspective of "lack of spatial autocorrelation" rather than simply attributing it to "insufficient data".
- **Elegant Approach**: Instead of adding explicit regularization losses, regularization is implicitly achieved through the architectural inductive bias of the neural field.
- **Unified 3D+4D**: A single method seamlessly applies to both static and dynamic scenes.
- **Complementary to Existing Methods**: Can be layered on top of other 3DGS improvement methods to further enhance performance.

## Limitations & Future Work

- If the training does not switch back to standard optimization in the late stage, it may limit the fitting of high-frequency details.
- The resolution settings of the neural field need to be adjusted based on scene complexity.
- Under extreme sparsity (1-2 views), it still cannot compete with specialized single-view reconstruction methods.
- The combination with 3DGS density control improvements (such as Pixel-GS) remains unexplored.
- More efficient neural field architectures could be investigated to reduce the extra computational overhead.

## Related Work & Insights

- **3DGS**: Base method, independently parameterizing each Gaussian.
- **Instant-NGP**: Implicit representation of multi-resolution hash encoding.
- **DNGaussian**: Employs depth priors to regularize sparse 3DGS.
- **FSGS**: A feature-aware method for few-view 3DGS.
- **Insights**: Explicit and implicit representations can be complementary—using the continuity of implicit fields to regularize the parameters of explicit primitives is a promising hybrid representation idea.

## Rating

- Novelty: ⭐⭐⭐⭐ Diagnoses the problem from the perspective of spatial autocorrelation and regularizes it with a neural field, offering deep insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple 3D/4D datasets with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly HTML details.
- Value: ⭐⭐⭐⭐ A general regularization strategy with significant reference value for the sparse 3DGS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer](sc4d_sparse-controlled_video-to-4d_generation_and_motion_transfer.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)

</div>

<!-- RELATED:END -->
