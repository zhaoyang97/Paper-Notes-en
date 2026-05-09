---
title: >-
  [Paper Note] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation
description: >-
  [CVPR 2026][3D Vision][2D Gaussian Splatting] SGI proposes a seed-based structured 2D Gaussian representation framework that organizes unstructured Gaussian primitives into seed-driven neural Gaussians, coupled with context-guided entropy coding and a multi-scale fitting strategy, achieving up to 7.5× compression and 6.5× optimization speedup in high-resolution image representation while maintaining or improving reconstruction fidelity.
tags:
  - CVPR 2026
  - 3D Vision
  - 2D Gaussian Splatting
  - image representation
  - neural compression
  - entropy coding
  - multi-scale optimization
date: 2026-05-08
content_hash: 40baa5293f0c5d38
---

# SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation

**Conference**: CVPR 2026
**arXiv**: [2603.07789](https://arxiv.org/abs/2603.07789)
**Code**: None
**Area**: 3D Vision
**Keywords**: 2D Gaussian Splatting, image representation, neural compression, entropy coding, multi-scale optimization

## TL;DR

SGI proposes a seed-based structured 2D Gaussian representation framework that organizes unstructured Gaussian primitives into seed-driven neural Gaussians, coupled with context-guided entropy coding and a multi-scale fitting strategy, achieving up to 7.5× compression and 6.5× optimization speedup in high-resolution image representation while maintaining or improving reconstruction fidelity.

## Background & Motivation

**State of the Field**: 2D Gaussian Splatting has emerged as a promising image representation technique that enables efficient rendering on low-end devices. However, scaling it to high-resolution images requires independently optimizing and storing millions of unstructured Gaussian primitives, resulting in two core bottlenecks:

1. **Parameter Redundancy**: Existing methods (e.g., GaussianImage, LIG) optimize each Gaussian independently without exploiting spatial locality — neighboring pixels tend to share similar colors, textures, and structures, leading to substantial redundancy among adjacent primitives.
2. **Optimization Overhead**: Independent optimization of millions of Gaussian primitives at high resolution converges slowly, especially when quantization-aware fine-tuning is introduced for compression.

**Limitations of Prior Work**:
- INR-based methods (SIREN, I-NGP) require extensive MLP forward passes, incurring high computational cost.
- 3D anchor-based methods (Scaffold-GS, HAC) cannot achieve comparable compression gains when directly applied to 2D image modeling, since 2D Gaussians already omit parameters such as opacity, leaving limited room for storage savings.
- LIG's Level-of-Gaussian uses only a subset of Gaussians for residual fitting, which is not a global optimization strategy.

## Method

### Overall Architecture

SGI comprises three core components:

1. **Seed-based 2D Neural Gaussians**: Decomposes the image into multi-scale local regions, where each seed point predicts a set of structured Gaussian primitives via a lightweight MLP.
2. **Neural Entropy Coding**: Employs a binary hash grid and a context model to estimate the probability distribution of seed attributes, enabling adaptive bit allocation.
3. **Multi-scale Fitting**: Adopts a Gaussian pyramid for coarse-to-fine progressive optimization to accelerate convergence.

### Key Design 1: Seed-Driven 2D Neural Gaussians

Given $N$ predefined seed points uniformly initialized to cover the image, each seed is located at position $\boldsymbol{x_a} \in \mathbb{R}^2$ and is associated with a set of attributes:

$$\mathcal{A} = \{\boldsymbol{f_a} \in \mathbb{R}^D, \boldsymbol{s_o} \in \mathbb{R}^2, \boldsymbol{s_a} \in \mathbb{R}^2, \boldsymbol{\delta} \in \mathbb{R}^{K \times 2}\}$$

The semantics of each attribute are as follows:

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\boldsymbol{f_a}$ | $\mathbb{R}^D$ | Seed feature vector encoding local region information |
| $\boldsymbol{s_o}$ | $\mathbb{R}^2$ | Offset scale factor controlling the spatial distribution range of associated Gaussians |
| $\boldsymbol{s_a}$ | $\mathbb{R}^2$ | Scale factor adjusting the final scale of Gaussians |
| $\boldsymbol{\delta}$ | $\mathbb{R}^{K \times 2}$ | Learned offsets for $K$ associated Gaussians |

**Gaussian Position Computation**: The positions of the $K$ Gaussians associated with each seed are computed by adding scaled offsets to the seed position:

$$\{\boldsymbol{\mu}^{(k)}\}_{k=0}^{K-1} = \boldsymbol{x_a} + \{\boldsymbol{\delta}^{(k)}\}_{k=0}^{K-1} \cdot \boldsymbol{s_o}$$

**Attribute Decoding**: Two shared lightweight MLPs decode Gaussian attributes from the seed feature $\boldsymbol{f_a}$:
- $\text{MLP}_c$: outputs opacity-weighted color coefficients $\mathbf{c'} \in \mathbb{R}^3$
- $\text{MLP}_\Sigma$: predicts base scale $\boldsymbol{s_{\text{base}}}$ and rotation angle $\theta$

The final scale is $\boldsymbol{s} = \boldsymbol{s_{\text{base}}} \cdot \boldsymbol{s_a}$, and the covariance matrix is constructed via positive-definite decomposition:

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$$

where $\mathbf{R}(\theta)$ is the 2D rotation matrix and $\mathbf{S}$ is the diagonal scale matrix.

**Rendering Formula**: Pixel color is computed by accumulating contributions from all relevant Gaussians:

$$\boldsymbol{C} = \sum_{i \in I} \mathbf{c'}_i G_i(\mathbf{x}), \quad G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Design Insight**: Each Gaussian requires only 8 parameters (2 position + 3 covariance + 3 weighted color), while abundant spatial redundancy is implicitly encoded in the seed features and shared MLPs, substantially reducing the number of independent parameters.

### Key Design 2: Context Model-Based Entropy Coding

Using the seed structure alone reduces storage by only approximately 3% compared to plain 2D Gaussians (e.g., LIG), yielding limited gains. The core innovation of SGI is to exploit the structural regularity introduced by seeds for entropy coding compression.

**Quantization Strategy**: During training, quantization is simulated via noise injection; during inference, rounding quantization is applied:

$$\hat{\boldsymbol{f}}_j^{(i)} = \begin{cases} \boldsymbol{f}_j^{(i)} + \mathcal{U}(-\frac{1}{2}, \frac{1}{2}) \cdot q_j^{(i)} & \text{training} \\ \text{Round}(\boldsymbol{f}_j^{(i)} / q_j^{(i)}) \cdot q_j^{(i)} & \text{inference} \end{cases}$$

The quantization step size is $q_j^{(i)} = Q_j \times (1 + \tanh(r_j^{(i)}))$, where $r_j^{(i)}$ is adaptively predicted by the context model.

**Probability Modeling**: A learnable binary hash grid $\mathcal{H}$ is introduced to capture spatial coherence among seeds, and a context model $\text{MLP}_p$ estimates the Gaussian distribution parameters for each attribute component:

$$\{\mu_j^{(i)}, \sigma_j^{(i)}, r_j^{(i)}\}_{j=0}^{3} = \text{MLP}_p(\mathcal{H}(\boldsymbol{x}_a^{(i)}))$$

Attribute probabilities are obtained by integrating the Gaussian distribution over quantization intervals, driving arithmetic coding for adaptive bit allocation.

### Key Design 3: Multi-Scale Fitting Strategy

Directly optimizing seed parameters on high-resolution images is computationally expensive and converges slowly. SGI constructs an $M$-level Gaussian pyramid for coarse-to-fine optimization:

1. Build pyramid $\{I_0=I, I_1, \ldots, I_{M-1}\}$, each level downsampled by 2×.
2. Begin optimization of seed and MLP parameters at the coarsest level $l=M-1$.
3. Transfer optimized parameters to the next level: positions and scales are multiplied by 2 to adapt to the doubled resolution.
4. Iterate level by level until the finest level $l=0$.

### Loss & Training

The total loss combines reconstruction fidelity and bit-rate regularization:

$$L = L_{\text{img}} + \frac{\lambda}{N \cdot d_\mathcal{A}}(L_{\text{entropy}} + L_{\text{hash}})$$

- $L_{\text{img}}$: L1 loss between the rendered and target images
- $L_{\text{entropy}}$: information entropy loss on seed attributes, driving the probability model to learn compact distributions
- $L_{\text{hash}}$: upper bound on the bit consumption of the binary hash grid
- $\lambda = 0.001$: rate-distortion trade-off hyperparameter
- $d_\mathcal{A} = D + 4 + 2K$: total attribute dimensionality per seed

## Key Experimental Results

### Experimental Setup

- **Datasets**: FGF2 (4 satellite images, ~51MP), ICB (2 natural images, 27.7/39.1MP), STimage (3 pathology images, ~76MP)
- **Metrics**: PSNR, SSIM, LPIPS, optimization time (minutes), model size (MB)
- **SGI Configurations**: low bit-rate (3.5M Gaussians) and high bit-rate (10M Gaussians)

### Main Results

| Method | FGF2 PSNR↑ | FGF2 Size(MB)↓ | FGF2 Time(min)↓ | ICB PSNR↑ | ICB Size(MB)↓ | ICB Time(min)↓ |
|--------|-----------|----------------|-----------------|----------|--------------|----------------|
| SIREN | 22.05 | 15.79 | 649.71 | 27.62 | 15.79 | 363.34 |
| I-NGP | 28.55 | 21.07 | 72.32 | 33.09 | 21.07 | 48.11 |
| HAC | 25.15 | 16.78 | 261.69 | 34.47 | 13.52 | 270.57 |
| GaussianImage | 27.30 | 23.37 | 322.17 | 31.09 | 23.37 | 282.61 |
| **SGI (low)** | **31.24** | **16.33** | **48.43** | **35.27** | **12.30** | **44.75** |
| 3DGS | 34.93 | 787.73 | 642.85 | 37.52 | 787.73 | 515.99 |
| Scaffold-GS | 28.25 | 112.61 | 248.83 | 35.76 | 105.81 | 162.11 |
| LIG | 32.10 | 106.81 | 87.56 | 36.40 | 106.81 | 68.73 |
| **SGI (high)** | **36.27** | **41.74** | **97.75** | **39.09** | **32.15** | **86.11** |

### Ablation Study

**Ablation on Number of Gaussians per Seed $K$**:

| $K$ | FGF2 PSNR↑ | FGF2 Size(MB)↓ | ICB PSNR↑ | ICB Size(MB)↓ |
|-----|-----------|----------------|----------|--------------|
| 5 | 31.29 | 18.48 | 35.03 | 13.64 |
| **10** | **31.24** | **16.33** | **35.27** | **12.30** |
| 15 | 30.61 | 15.32 | 34.88 | 11.48 |
| 20 | 30.62 | 14.83 | 34.57 | 10.87 |

Larger $K$ yields more compact models at the cost of slightly reduced fidelity; $K=10$ represents the optimal trade-off.

**Key Findings**:

1. **Entropy Coding Is the Core of Compression**: Without entropy coding ($\lambda=0$), the FGF2 storage is 104.08 MB; with $\lambda=0.001$, it drops to 16.33 MB — a 6.4× reduction. The seed structure itself contributes only ~3% reduction; entropy coding accounts for the vast majority of compression gain.
2. **Multi-Scale Fitting Significantly Accelerates Optimization**: With $M=3$ pyramid levels, optimization time drops from 71.59 minutes ($M=1$) to 48.43 minutes (32% speedup), while PSNR improves from 30.58 to 31.24 dB.
3. **Storage Breakdown**: Seed features $\boldsymbol{f_a}$ account for 48% of total storage, offsets $\boldsymbol{\delta}$ for 35%, and the hash grid and MLPs for only ~1%, incurring negligible overhead.
4. **Low Bit-Rate Advantage**: In the low-bpp regime, SGI substantially outperforms JPEG (26.09 dB at 0.245 bpp on ICB vs. JPEG's 22.77 dB at 0.296 bpp).

## Highlights & Insights

- **First Structured 2D Gaussian Representation**: Adapts anchor-based ideas from 3D to 2D image representation, with entropy coding bridging the limited compression gain inherent to 2D settings.
- **Outstanding Compression Efficiency**: Achieves 7.5× compression over LIG and 1.6× over quantized GaussianImage.
- **High Optimization Efficiency**: The multi-scale fitting strategy delivers 1.6–6.5× speedup without sacrificing reconstruction quality.
- **Low-End Device Friendly**: Differentiable rasterization-based fast rendering enables processing of 76MP images on a 24GB A10 GPU.

## Limitations & Future Work

- Large images still require tens of minutes of optimization (per-image optimization paradigm), making the approach unsuitable for real-time encoding scenarios.
- Each image requires training an independent model, with no generalization across images.
- Validation is currently limited to static images; extension to video or dynamic scenes has not been explored.
- At high bit-rates, storage remains in the 30–40 MB range, offering no absolute advantage over learned image codecs (e.g., VVC/AV1).

## Rating ⭐⭐⭐⭐

A solid and systematic work that organically integrates seed structure, entropy coding, and multi-scale optimization, achieving significant improvements in storage, speed, and quality for large-scale image representation. The experiments are comprehensive and the ablations are thorough. However, the per-image optimization paradigm and lack of generalization remain inherent limitations.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [\[CVPR 2026\] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)
- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](lumimotion_gaussian_relighting_dynamics.md)

<!-- RELATED:END -->
