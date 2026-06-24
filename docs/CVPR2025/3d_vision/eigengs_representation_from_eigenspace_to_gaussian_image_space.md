---
title: >-
  [Paper Note] EigenGS: Representation from Eigenspace to Gaussian Image Space
description: >-
  [CVPR 2025][3D Vision][Gaussian Representation] This paper proposes EigenGS, which bridges the eigenspace representation of classical PCA with the 2D Gaussian Splatting image representation. By learning unified Gaussian parameters on the eigenbasis, instant initialization of new images is achieved (without optimization from scratch). Furthermore, a frequency-aware learning mechanism is introduced to prevent high-resolution reconstruction artifacts…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Representation"
  - "PCA"
  - "Image Reconstruction"
  - "2D Gaussian"
  - "Frequency-aware Learning"
date: 2026-05-08
content_hash: 5d3b667452413cfb
---

# EigenGS: Representation from Eigenspace to Gaussian Image Space

**Conference**: CVPR 2025  
**arXiv**: [2503.07446](https://arxiv.org/abs/2503.07446)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Gaussian Representation, PCA, Image Reconstruction, 2D Gaussian, Frequency-aware Learning

## TL;DR

This paper proposes EigenGS, which bridges the eigenspace representation of classical PCA with the 2D Gaussian Splatting image representation. By learning unified Gaussian parameters on the eigenbasis, instant initialization of new images is achieved (without optimization from scratch). Furthermore, a frequency-aware learning mechanism is introduced to prevent high-resolution reconstruction artifacts, comprehensively outperforming GaussianImage in both convergence speed and final quality.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has been widely applied to 3D scene representations. GaussianImage adapts this concept to 2D, fitting a single image for reconstruction with a set of 2D Gaussians. As a classic dimensionality reduction technique, PCA is extensively used in computer vision, but its pixel-wise independence assumption ignores local spatial relationships.

**Limitations of Prior Work**: (1) GaussianImage requires independent training for each new image starting from random initialization, which converges slowly (PSNR is only around 10 dB in the first 100 iterations); (2) PCA assumes pixel independence, failing to exploit local and non-local pixel relationships; (3) During high-resolution image optimization, Gaussians easily shrink into uniform small sizes, generating "penny-round-tile" circular artifacts.

**Key Challenge**: GaussianImage requires image-by-image training and cannot utilize common knowledge from the training set for initialization; PCA possesses good initialization capabilities but lacks local modeling capabilities.

**Goal**: (1) How to convert the eigenbasis knowledge of PCA into Gaussian parameters to achieve instant initialization of new images? (2) How to adapt Gaussians to different spatial frequencies to avoid artifacts under high-resolution conditions?

**Key Insight**: The authors observe that the linear combination coefficients of the PCA eigenbasis can be directly merged with the visual weights of Gaussians—if the eigenbasis is rendered using the same set of Gaussians, the Gaussians of a new image can be instantly derived via the weighted sum of the coefficients.

**Core Idea**: Fit all components of the PCA eigenbasis using the same set of 2D Gaussians. A new image can obtain instant Gaussian initialization through PCA projection coefficients, which then quickly converges with a small amount of fine-tuning.

## Method

### Overall Architecture

The input is a training image set $\{I_1, ..., I_m\}$, and the PCA eigenbasis $\{\Psi_j\}_{j=1}^k$ is first computed. Then, a set of shared 2D Gaussians $\mathcal{N}$ is learned to simultaneously approximate all $k$ eigenbasis components. For a new image $I$, its PCA coefficients $\{w_j\}$ are computed, and the weight of each Gaussian is instantly obtained via the linear combination $c'_n = \sum_j w_j \psi'_{n,j}$, yielding an initial reconstruction $\tilde{I}^{(0)}$. Finally, the Gaussian parameters are further optimized by minimizing the reconstruction loss.

### Key Designs

1. **EigenGS Representation**:

    - **Function**: Uniformly represent all PCA eigenbasis components with the same set of Gaussians, enabling a seamless transition from eigenspace to image space.
    - **Mechanism**: The rendering of each eigenbasis component $\Psi_j$ at pixel location $(x,y)$ is $\tilde{\Psi}_j(x,y) = \sum_{n=1}^{|\mathcal{N}|} \psi'_{n,j} \cdot \exp(-\sigma_n(x,y))$, where the spatial parameters (position, covariance) of the Gaussians are shared across all components, and only the weights $\psi'_{n,j}$ differ. The Gaussian weights of a new image $I$ are $c'_n = \sum_j w_j \psi'_{n,j}$, which is a linear combination of the PCA coefficients and eigenbasis weights. This mathematically guarantees that the initial reconstruction quality equals the standard PCA reconstruction.
    - **Design Motivation**: Traditional PCA reconstruction is a pixel-level linear combination. Transforming it into a linear combination of Gaussian weights not only preserves PCA's initialization benefits but also allows subsequent optimization to leverage the local modeling capability of Gaussians, surpassing the upper bound of PCA.

2. **Frequency-aware Learning (FL)**:

    - **Function**: Prevent all Gaussians from shrinking to a uniform small size, maintaining a mixture of large and small Gaussians to cover different spatial frequencies.
    - **Mechanism**: The Gaussian set is divided into two groups, $\mathcal{N}_l$ and $\mathcal{N}_h$. The eigenbasis components are divided into low-frequency $\{\tilde{\Psi}_l\}$ and high-frequency $\{\tilde{\Psi}_h\}$ based on the magnitude of the eigenvalues. Training occurs in two stages: In the first stage, about 10% of the Gaussians are allocated to model the large eigenvalue (low-frequency) components, forcing these Gaussians to maintain larger sizes; in the second stage, the remaining Gaussians model the small eigenvalue (high-frequency) components. The final representation is a mixture of large and small Gaussians.
    - **Design Motivation**: The optimization process naturally prefers small Gaussians to minimize pixel-level differences, which leads to all Gaussians shrinking and causing "penny-round-tile" artifacts under high resolution. By training the high-frequency and low-frequency components separately, a bi-modal size distribution is naturally formed, replacing explicit regularization.

3. **YCbCr Color Space Processing**:

    - **Function**: Reduce performance degradation caused by value truncation in PCA reconstruction.
    - **Mechanism**: Deconstruct in the YCbCr space instead of the RGB space, processing luminance (Y: 16-235) and chrominance (Cb/Cr: 16-240) channels separately. Since the range structure of YCbCr provides a natural margin for out-of-bound values in PCA reconstruction, and because the chrominance channels are less complex, they are more robust to outliers.
    - **Design Motivation**: All three channels of RGB are prone to outlier effects, and out-of-bound values of PCA reconstruction lead to truncation in all three channels. YCbCr compresses color information into two channels, reducing the quality loss caused by truncation, and yielding up to a 7+ dB PSNR improvement.

### Loss & Training

Standard image reconstruction loss (pixel-level L2) is used. Training is conducted in two stages: the low-frequency stage handles about 10% of the Gaussians corresponding to large eigenvalue components; the high-frequency stage handles the remaining Gaussians corresponding to small eigenvalue components. A training set of 10,000 images is used for PCA decomposition, with 300 or 500 eigencomponents and a default of 20,000 Gaussian points, trained on a single V100 GPU.

## Key Experimental Results

### Main Results

FFHQ dataset (512×512), 20,000 Gaussian points:

| Method | ITER=0 PSNR | ITER=100 PSNR | ITER=1000 PSNR | ITER=10000 PSNR |
|------|-------------|---------------|----------------|-----------------|
| GaussianImage | - | 10.4 | 29.4 | 40.1 |
| EigenGS (300 comp) | 28.0 | 34.4 | 37.5 | 41.8 |
| EigenGS (500 comp) | 28.9 | 34.8 | 37.7 | 41.8 |

At 1000 iterations, 83-84% of EigenGS samples already achieve PSNR > 35dB, while GaussianImage is at 0%.

### Ablation Study

| Configuration | CelebA PSNR | FFHQ PSNR | Cats PSNR | Cars PSNR |
|------|-------------|-----------|-----------|-----------|
| Ours-YCbCr | 47.2 | 41.8 | 45.7 | 44.7 |
| Ours-YCbCr (w/o FL) | 48.0 | 40.7 | 46.1 | 43.5 |
| Ours-RGB | 39.5 | 34.9 | 38.5 | 36.4 |
| Ours-RGB (w/o FL) | 39.9 | 33.3 | 38.9 | 35.1 |

### Key Findings

- The initial PSNR of EigenGS (28-29 dB) already far exceeds the random initialization of GaussianImage, reaching 34+ dB within 100 iterations.
- The YCbCr color space is the largest contributing factor, offering a ~7 dB improvement on FFHQ compared to RGB.
- Frequency-aware learning (FL) brings a significant boost (+1.1/+1.2 dB) on high-resolution datasets (FFHQ 512×512, Cars), but slightly degrades (-0.8 dB) on low-resolution datasets (CelebA 256×256).
- Strong cross-dataset generalization: EigenGS trained on ImageNet and applied to CelebA still achieves an initial PSNR of 28.7 dB, reaching 35.4 dB in 100 iterations.
- The number of components mainly affects early convergence (300 vs 500), while the final quality remains almost identical (41.8 vs 41.8 dB).

## Highlights & Insights

- **Elegant Bridging of PCA and Gaussian Splatting**: Utilizing the commutative property of linear combination, the image reconstruction from PCA coefficients is seamlessly transformed into the calculation of Gaussian weights, which is mathematically elegant and practical. This idea can be generalized to any basis-function-based initialization strategy.
- **Frequency Separation as a Substitute for Explicit Regularization**: Instead of directly constraining Gaussian sizes, training across different frequency bands naturally yields a multi-scale Gaussian distribution. This both resolves artifacts and preserves optimization flexibility.
- **Insights on Cross-domain Generalization**: A general-purpose EigenGS trained on ImageNet can provide effective initialization across various datasets, implying the potential existence of a universal image Gaussian basis.

## Limitations & Future Work

- Requires pre-computing PCA on a training set, which is not applicable to completely prior-free scenarios.
- Verified only on 2D Gaussian image representation and not extended to 3D scenes in 3DGS.
- FL has a slightly negative impact on low-resolution images, requiring manual selection of whether to enable it based on the resolution.
- The linearity assumption of PCA limits the initialization quality under highly non-linear visual variations (e.g., large posture changes).
- Training still requires 10,000 iterations (approx. 13 seconds), which is still some distance away from "real-time".

## Related Work & Insights

- **vs GaussianImage**: GaussianImage trains from scratch with random initialization for 10,000 iterations per image; EigenGS utilizes PCA initialization to reach comparable quality in 100-1000 iterations, yielding a speedup of 10-100x.
- **vs Traditional PCA**: PCA sets the upper bound of linear reconstruction (PSNR around 28-29 dB); EigenGS starts from PCA and continues to optimize with Gaussians, achieving a final PSNR of over 41 dB, breaking the linear limitation.
- **vs Gaussian Optimization Methods like Mini-Splatting**: These methods optimize Gaussian density/regularization. EigenGS orthogonally solves the initialization problem, and the two can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of bridging PCA and Gaussians is novel and mathematically elegant, though the core contribution leans towards a combination of innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough experiments across multiple datasets, cross-domain tests, and ablation studies, but lacks comparison with a wider range of baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear mathematical derivations, intuitive figures/tables, and well-structured paper.
- **Value**: ⭐⭐⭐ The application scenario is relatively narrow (2D Gaussian image representation), offering limited inspiration to mainstream 3D scene reconstruction with 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[ICLR 2026\] Weight Space Representation Learning on Diverse NeRF Architectures](../../ICLR2026/3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)

</div>

<!-- RELATED:END -->
