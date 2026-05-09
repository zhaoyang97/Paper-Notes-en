---
title: >-
  [Paper Note] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space
description: >-
  [AAAI 2026][3D Vision][Implicit Neural Representation] This paper proposes Split-Layer, which decomposes fully connected layers in MLPs into multiple parallel branches and integrates their outputs via the Hadamard product. Without increasing parameter count or computational cost, this approach exponentially expands the feature space dimensionality from $C$ to $\binom{C/\sqrt{N}+N-1}{N}$, significantly enhancing the representational capacity of implicit neural representations (INRs).
tags:
  - AAAI 2026
  - 3D Vision
  - Implicit Neural Representation
  - Feature Space
  - MLP Restructuring
  - Hadamard Product
  - Multi-task
date: 2026-05-08
content_hash: 345e9be521ddec87
---

# Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space

**Conference**: AAAI 2026
**arXiv**: [2511.10142](https://arxiv.org/abs/2511.10142)
**Code**: None
**Area**: 3D Vision
**Keywords**: Implicit Neural Representation, Feature Space, MLP Restructuring, Hadamard Product, Multi-task

## TL;DR

This paper proposes Split-Layer, which decomposes fully connected layers in MLPs into multiple parallel branches and integrates their outputs via the Hadamard product. Without increasing parameter count or computational cost, this approach exponentially expands the feature space dimensionality from $C$ to $\binom{C/\sqrt{N}+N-1}{N}$, significantly enhancing the representational capacity of implicit neural representations (INRs).

## Background & Motivation

### Problem Definition

Implicit neural representations (INRs) model signals as continuous functions via neural networks and have broad applications in inverse problem solving. However, the representational capacity of INRs is constrained by the dimensionality of the feature space in MLP architectures. Specifically:

- **Limitations of fully connected layers**: For a fully connected layer of width $C$, the feature space spanned by its outputs is a $C$-dimensional Euclidean space. Each output element is a linear combination of input elements, so the feature space dimensionality scales linearly with layer width.
- **High cost of scaling**: Linearly increasing layer width (e.g., to $2C$) yields only linear growth in feature space dimensionality, while parameter count grows quadratically (from $C^2$ to $4C^2$), making this approach computationally prohibitive.

### Limitations of Prior Work

Existing methods for enhancing INR representational capacity fall into two main categories:

**Coordinate embedding methods**: e.g., Fourier positional encodings (PEMLP), hash tables (InstantNGP), which map low-dimensional coordinates to high-dimensional manifolds.

**Specialized activation functions**: e.g., SIREN (periodic sine), WIRE (wavelets), FINER (variable-period), etc.

However, these methods fundamentally introduce **learning biases**, making certain features easier to learn rather than structurally expanding the range of learnable features. They do not increase feature space dimensionality at the architectural level.

### Core Motivation

The authors attribute the root cause to the fully connected mechanism of MLPs, where feature space dimensionality scales **linearly** with the number of neurons. By reorganizing the connectivity into a "split" paradigm, it is possible to **exponentially expand** the feature space dimensionality while maintaining the same parameter count.

## Method

### Overall Architecture

The core idea of Split-Layer is straightforward: a fully connected layer is decomposed into $N$ parallel branches, each with independent weight matrices, and the outputs are integrated via the **Hadamard product (element-wise multiplication)**. This operation elevates linear combinations to **high-order polynomials**, thereby constructing a feature space far exceeding that of the original MLP.

### Key Designs

#### 1. **Split-Layer Structure**: Decomposing Fully Connected Layers into Multi-Branch Hadamard Product Form

**Mechanism**: The output of the $l$-th fully connected layer is:

$$z_i^l = \sum_{j=1}^{C} w_{ij} z_j^{l-1}$$

This is a linear combination of $C$ linearly independent elements, yielding a feature space of dimension $C$.

Split-Layer decomposes this layer into $N$ branches, each with weight matrix $\mathbf{W}_n^l \in \mathbb{R}^{C/\sqrt{N} \times C/\sqrt{N}}$, with outputs integrated via the Hadamard product:

$$z_i^l = \prod_{n=1}^{N} \left(\sum_{j=1}^{C/\sqrt{N}} w_{ij}^n z_j^{l-1}\right)$$

Expanding this yields:

$$z_i^l = \sum_{(j_1,j_2,...,j_N)} \left(\prod_{n=1}^{N} w_{ij_n}^n\right) \left(z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}\right)$$

This forms an **$N$-th order homogeneous polynomial**, in which distinct terms $z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}$ are mutually linearly independent.

**Design Motivation**: By leveraging polynomial products, the feature space is expanded from a linear space to a polynomial space. The total number of distinct terms equals the number of multiset combinations of $N$ elements chosen from $C/\sqrt{N}$:

$$\text{Feature space dimensionality} = \binom{C/\sqrt{N}+N-1}{N}$$

For $C=256, N=2$, the original feature space has dimension 256, whereas Split-Layer expands it to $\binom{181+1}{2} = 16{,}471$ dimensions—approximately a 64-fold increase—with no change in parameter count (each branch has $(C/\sqrt{N})^2$ parameters; $N$ branches total $C^2$).

#### 2. **Optimal Split Count Selection**: Balancing Feature Space Dimensionality and Weight Matrix Degrees of Freedom

**Mechanism**: Increasing the split count $N$ enlarges the feature space dimensionality but reduces the size of each branch's weight matrix, limiting its freedom to explore feature combinations. Empirically, the optimal split count is:

$$N^* \approx (0.17C)^{2/3}$$

**Design Motivation**: This represents the optimal trade-off between feature space expansion and weight matrix expressiveness. The authors validate the robustness of this formula through 2D image fitting experiments across different network widths $C$—optimal results consistently appear near the theoretical surface.

#### 3. **Plug-and-Play Design**: Adapting to Various INR Backbones as a Drop-in Module

**Mechanism**: Split-Layer replaces all hidden fully connected layers in an INR and is compatible with diverse INR architectures (ReLU MLP, SIREN, Gauss, PEMLP, WIRE, FINER).

**Design Motivation**: Split-Layer is an architectural improvement orthogonal to input encoding schemes and activation functions, allowing it to be combined additively with existing INR enhancement methods.

### Validation from the Neural Tangent Kernel Perspective

From the NTK perspective, Split-MLP exhibits a more uniform NTK eigenvalue distribution, expanding from $[10^{-3}, 10^{0}]$ to $[10^{-2}, 10^{2}]$, indicating superior convergence on high-frequency components. This further validates the effectiveness of Split-Layer from a theoretical standpoint.

### Loss & Training

- Standard loss functions are used for all tasks (e.g., L2 loss, cross-entropy loss); no special design is required.
- Weight initialization: SIREN and FINER use their respective specific initialization schemes; all others use default LeCun initialization.
- Optimizer: Adam; the number of training epochs varies by task.
- In all experiments, $N=2$ (2-split) is used as the default and already achieves strong performance.

## Key Experimental Results

### Main Results

Split-Layer is comprehensively evaluated across 6 INR backbones and 4 tasks.

| Task | Backbone | Baseline | Split | Gain |
|------|----------|----------|-------|------|
| 2D Image Fitting (PSNR↑) | ReLU | 21.24 | 30.89 | **+45.43%** |
| 2D Image Fitting (PSNR↑) | SIREN | 38.52 | 39.25 | +1.90% |
| 2D Image Fitting (PSNR↑) | PEMLP | 29.60 | 40.78 | **+37.77%** |
| 2D Image Fitting (PSNR↑) | Gauss | 31.74 | 40.84 | **+28.67%** |
| CT Reconstruction (PSNR↑) | SIREN | 18.32 | 29.11 | **+58.90%** |
| CT Reconstruction (PSNR↑) | PEMLP | 28.11 | 32.29 | +14.87% |
| 3D Shape Representation (CD↓) | ReLU | 1.00e-4 | 2.01e-5 | **+79.90%** |
| 3D Shape Representation (CD↓) | Gauss | 2.19e-5 | 5.33e-6 | **+75.66%** |

**5D Novel View Synthesis (NeRF scenes, PSNR↑)**:

| Method | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship | Avg. |
|--------|-------|-------|-------|--------|------|-----------|-----|------|------|
| NeRF | 31.37 | 24.50 | 28.90 | 34.94 | 30.71 | 28.60 | 28.99 | 27.27 | 29.41 |
| Split-NeRF | 31.78 | 24.81 | 29.34 | 35.33 | 31.76 | 28.87 | 31.85 | 27.83 | **30.20** |
| DINER | 34.49 | 25.43 | 33.28 | 36.45 | 34.82 | 29.58 | 33.43 | 29.25 | 32.09 |
| Split-DINER | **34.85** | 25.47 | **33.39** | **36.92** | **35.14** | 29.59 | **34.01** | **29.49** | **32.36** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| $N=2$ (default) | Best or near-best | Optimal balance between practicality and performance |
| Optimal $N$ at different $C$ | Consistent with $(0.17C)^{2/3}$ surface | Validates robustness of the optimal split formula |
| Feature visualization | More diverse features after split | 9 features → 45/84 distinct feature bases |
| NTK eigenvalue distribution | More uniform in Split-MLP | Better convergence on high-frequency components |

### Key Findings

1. **ReLU and SIREN benefit the most**: ReLU achieves a 45% gain on image fitting, and SIREN achieves a 59% gain on CT reconstruction, indicating that weaker baseline backbones benefit more from Split-Layer.
2. **Split-PEMLP achieves outstanding performance**: It reaches the best results in both image fitting and shape representation tasks.
3. **Zero additional cost**: Split-Layer achieves substantial gains without increasing parameter count or computational overhead.
4. **Strong generality**: Consistent improvements are observed across all combinations of 6 backbones × 4 tasks.

## Highlights & Insights

1. **Theoretical elegance**: The representational capacity of INRs is analyzed through the lens of feature space dimensionality, reducing the problem to a combinatorial question of multiset coefficients, with concise and clear theoretical derivations.
2. **Simple implementation**: Only the fully connected layers need to be replaced with multi-branch Hadamard product structures; no modifications to the training pipeline or loss functions are required.
3. **Orthogonal to existing methods**: Split-Layer does not conflict with positional encodings or activation function methods and can be combined with them additively.
4. **NTK-based theoretical validation**: The neural tangent kernel perspective provides additional theoretical grounding for the observed performance improvements.

## Limitations & Future Work

1. **Optimal split formula is empirical**: $N^* \approx (0.17C)^{2/3}$ lacks rigorous theoretical derivation and is obtained solely through experimental fitting.
2. **Hadamard product may introduce training instability**: High-order polynomials could lead to gradient explosion or vanishing; the paper does not analyze this in depth.
3. **Validation limited to INR settings**: The generality of Split-Layer has not been verified on broader deep learning tasks such as classification or detection.
4. **Combination with hash grid methods unexplored**: The effect of combining Split-Layer with methods such as InstantNGP warrants further investigation.

## Related Work & Insights

- **Connection to MFN (Multiplicative Filter Networks)**: MFN also employs multiplicative operations to combine multi-branch outputs, but Split-Layer provides a clearer theoretical analysis (explicit computation of feature space dimensionality).
- Hilbert or Gaussian kernels may offer further avenues for expanding the feature space (noted by the authors in the conclusion).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Rethinking INR representational capacity from the perspective of fully connected layer structure; a genuinely novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 4 tasks × 6 backbones, though quantitative comparison of computational efficiency is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — A plug-and-play general INR enhancement module with strong practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](../../CVPR2026/3d_vision/ntk-guided_implicit_neural_teaching.md)
- [\[ICLR 2026\] Weight Space Representation Learning on Diverse NeRF Architectures](../../ICLR2026/3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)

<!-- RELATED:END -->
