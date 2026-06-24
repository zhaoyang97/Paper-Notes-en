---
title: >-
  [Paper Note] HyperGS: Hyperspectral 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][hyperspectral imaging] This paper successfully extends 3DGS to hyperspectral novel view synthesis (HNVS) for the first time. By performing hyperspectral rendering in a learned latent space combined with adaptive density control and pixel-level spectral pruning, it achieves efficient and accurate reconstruction of high-dimensional spectral data.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "hyperspectral imaging"
  - "3D Gaussian Splatting"
  - "novel view synthesis"
  - "latent space"
  - "spectral reconstruction"
date: 2026-05-08
content_hash: 40a1cf32d0a1c105
---

# HyperGS: Hyperspectral 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.12849](https://arxiv.org/abs/2412.12849)  
**Code**: Unreleased  
**Area**: 3D Vision  
**Keywords**: hyperspectral imaging, 3D Gaussian Splatting, novel view synthesis, latent space, spectral reconstruction

## TL;DR

This paper successfully extends 3DGS to hyperspectral novel view synthesis (HNVS) for the first time. By performing hyperspectral rendering in a learned latent space combined with adaptive density control and pixel-level spectral pruning, it achieves efficient and accurate reconstruction of high-dimensional spectral data.

## Background & Motivation

**Background**: 3DGS has achieved remarkable success in RGB novel view synthesis, enabling high-performance and high-quality real-time rendering. However, RGB imaging only captures three-channel information and fails to represent the spectral characteristics of materials. Hyperspectral imaging (HSI) captures continuous spectra across 128–141 narrow-band channels, which is crucial for fields such as remote sensing, medical diagnosis, and robotics.

**Limitations of Prior Work**: (1) NeRF-based HNVS methods like HS-NeRF suffer from unstable training and slow rendering speeds; (2) directly extending 3DGS to high-dimensional spectral data yields optimization challenges and difficulties in threshold settings under naive approaches; (3) there is a lack of systematic HNVS benchmarking.

**Key Challenge**: The extreme dimensionality of hyperspectral data (128–141 channels) leads to massive computational overhead during direct optimization. Furthermore, the signal-to-noise ratios (SNR) vary significantly across channels, making stable training difficult with traditional L1+SSIM loss functions.

**Key Insight**: Model 3DGS inside a learned latent space to compress high-dimensional spectral information via an autoencoder, coupled with depth-aware density control and pixel-level spectral pruning strategies.

## Method

### Overall Architecture

1. **Preprocessing**: A convolutional autoencoder (AE) compresses high-dimensional spectral images into a low-dimensional latent space.
2. **SfM Initialization**: SfM point clouds are estimated from grayscale channel slices, and 3D points are reprojected into the latent hyperspectral space to obtain initial spectral signatures.
3. **Latent Space 3DGS**: Performs Gaussian rendering in the latent space and employs an MLP to predict view-dependent effects.
4. **Decoding & Training**: Decodes latent predictions to full spectral images using a frozen decoder to compute spectral loss.

### Key Designs

**1. Hyperspectral Compression Autoencoder**
- **Function**: Employs a symmetric AE built with 1D convolutions and Squeeze-and-Excitation (SE) blocks to compress high-dimensional spectral data into low-dimensional latent representations.
- **Mechanism**: The encoder compresses the spectral dimension using max-pooling, and the decoder reconstructs it via upsampling; no skip connections are used to ensure the decoder can operate independently.
- **Loss Function**: Huber Loss $L_{ae} = L_{Huber}(C^*(p), Dec(Enc(C^*(p))))$ is robust to outliers and handles varying SNRs across different cameras.
- **Design Motivation**: The latent space reduces the computational overhead of 3DGS optimization and encodes the camera's spectral sensitivity, establishing a bounded upper limit for errors.

**2. Depth-Aware Adaptive Density Control**
- **Function**: Modifies the split/clone criteria of 3DGS using a depth scaling function $h(d,i) = (|\mathbf{E}_d \mathbf{X}_i| / (\beta_{field} \times R))^2$ to modulate gradient influence.
- **Mechanism**: Scales NDC gradients by the square of the depth to reduce false high-gradient signals of close-to-camera Gaussians, stabilizing density control under the wide dynamic range of hyperspectral data.
- **Design Motivation**: Due to the massive number of channels and wide range of values in hyperspectral data, conventional threshold-based 3DGS density control fails, and near-field Gaussians tend to split inconsistently across multiple views.

**3. Pixel-Level Spectral Gaussian Pruning**
- **Function**: Calculates a pixel-level spectral importance score $\mathcal{I}[g_i, p, d] = (1 - |C^*_d(p) - Dec(f_i)|) \alpha_i T_i$ for each Gaussian, retaining only those within the Top-K for any pixel.
- **Mechanism**: Avoids pruning based on average scores (which leads to over-pruning), and instead decides to retain a Gaussian based on whether it falls within the Top-K importance list of at least one pixel.
- **Design Motivation**: Direct cross-view pruning can lead to over-pruning and the loss of fine spectral details; the pixel-level approach guarantees sufficient spectral expressiveness for every pixel.

### Loss & Training

$$L_d(p) = (1-\lambda)(\beta L_{CB}(p) + L_{CS}(p)) + \lambda L_{SSIM}(p)$$

- $L_{CB}$: Charbonnier Loss (smoother than L1, avoiding extreme errors in sensitive bands)
- $L_{CS}$: Cosine similarity loss (measures the angular distance of spectral vectors, suitable for spectral comparison)
- $L_{SSIM}$: Maintains spatial and geometric consistency
- $\beta$ balances the Charbonnier and Cosine terms in the spectral loss, while $\lambda$ controls the SSIM weight

## Key Experimental Results

### Main Results

**BaySpec Dataset** (141 channels, high noise, ~360 images/scene):

| Method | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ |
|---|---|---|---|---|
| MipNeRF360 | 26.53 | 0.7442 | 0.0280 | 0.0476 |
| HS-NeRF | 19.82 | 0.6714 | 0.0534 | 0.1071 |
| 3DGS | 22.91 | 0.6321 | 0.1335 | 0.0810 |
| **HyperGS** | **27.11** | **0.7804** | **0.0254** | **0.0440** |

**SOP Dataset** (128 channels, low noise, ~40 images/scene):

| Method | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ |
|---|---|---|---|---|
| MipNeRF360 | 12.28 | 0.6824 | 0.1369 | 0.2658 |
| 3DGS | 28.58 | 0.9627 | 0.0301 | 0.0478 |
| **HyperGS** | **30.51** | **0.9756** | **0.00415** | **0.0354** |

### Ablation Study

| Ablation Step | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ | N.Prim↓ |
|---|---|---|---|---|---|
| Base 3DGS | 22.91 | 0.6320 | 0.1335 | 0.0810 | 440k |
| + Spec. SFM | 23.05 | 0.6331 | 0.1310 | 0.0799 | 421k |
| + Latent AE | 24.87 | 0.7101 | 0.0548 | 0.0602 | 500k |
| + Densification | 25.25 | 0.7356 | 0.0365 | 0.0548 | 1.3M |
| + Pruning | 25.17 | 0.7199 | 0.0374 | 0.0555 | 412k |
| + View MLP | 27.05 | 0.7792 | 0.0253 | 0.0443 | 309k |
| + Custom Loss | **27.11** | **0.7804** | **0.0254** | **0.0440** | **309k** |

### Key Findings

1. **Latent space modeling is the core contribution**: Moving from Base 3DGS to +Latent AE yields a PSNR improvement of nearly 2dB, and a reduction in SAM from 0.1335 to 0.0548, showing the most significant gain in spectral accuracy.
2. **Model compression via pruning and density control**: Although the Gaussians expand from 500k to 1.3M after densification, pruning successfully reduces them to 309k (or 412k before MLP tuning), leading to a smaller model and cleaner spectra.
3. **Discrepancy across cameras and scenes**: NeRF-based methods perform reasonably well on the high-frame-rate, high-noise BaySpec dataset. However, on the low-frame-rate, low-noise SOP dataset, 3DGS methods vastly outperform NeRF. This is because the explicit representation of 3DGS is inherently better suited for sparse-view interpolation.
4. **All-around superiority**: HyperGS achieves the best results across all scenes and metrics.

## Highlights & Insights

- Systematically adapts 3DGS to the hyperspectral field for the first time, establishing a complete HNVS benchmark.
- The joint latent space and AE strategy is highly generalizable, and can be extended to 3DGS modeling of other high-dimensional signals (such as multispectral or infrared).
- A view-dependent MLP simultaneously predicts spectral effects and anisotropic opacity, elegantly handling view-dependent spectral variations.
- The loss function design — using Huber Loss to train the AE, and Charbonnier + Cosine similarity to train the 3DGS — is specifically optimized for spectral characteristics.
- The pixel-level Top-K pruning strategy is better suited for high-dimensional data compared to traditional average- or total-volume-based pruning.

## Limitations & Future Work

- Requires a pre-trained AE, which introduces an extra step.
- Relies on COLMAP for grayscale SfM, which may fail in textureless scenes.
- The experiments are only validated on two small datasets provided by HS-NeRF, lacking evaluation on large-scale scenes.
- Synthetic datasets rely on substituting spectral signatures based on semantic labels, which deviates from real-world hyperspectral scenes.
- Online inference speed and real-time rendering capabilities are not discussed.

## Related Work & Insights

- **HS-NeRF**: The first HNVS method, but it is not end-to-end and suffers from instability, providing the dataset and motivation for this work.
- **Scaffold-GS / Mip-Splatting**: Representative improvements of 3DGS (focusing on compression and anti-aliasing), but they do not consider high-dimensional extensions.
- **VDGS**: Uses a hybrid NeRF-MLP to predict color/opacity for multispectral reconstruction, though it still relies on a 3DGS backbone.
- **Insight**: The combined paradigm of latent space compression and explicit 3D representation can be generalized to multi-modal extensions in radiance fields.

## Rating

⭐⭐⭐⭐ — Achieves hyperspectral novel view synthesis under the 3DGS framework for the first time. The method is logically designed with solid ablations, though the experimental scale is relatively small, and a discussion on real-time rendering is missing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
