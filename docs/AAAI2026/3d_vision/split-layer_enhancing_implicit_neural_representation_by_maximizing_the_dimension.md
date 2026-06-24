---
title: >-
  [Paper Note] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space
description: >-
  [AAAI 2026][3D Vision][Implicit Neural Representation] Proposed Split-Layer, which splits the MLP fully connected layer into multiple parallel branches and integrates their outputs using the Hadamard product. Without increasing parameters or computation, this exponentially expands the feature space dimensionality from $C$ to $\binom{C/\sqrt{N}+N-1}{N}$, significantly enhancing the representation capability of Implicit Neural Representations (INR).
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Implicit Neural Representation"
  - "Feature Space"
  - "MLP Reconstruction"
  - "Hadamard Product"
  - "Multi-tasking"
date: 2026-05-08
content_hash: a0ae5f4a117f503e
---

# Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space

**Conference**: AAAI 2026  
**arXiv**: [2511.10142](https://arxiv.org/abs/2511.10142)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Implicit Neural Representation, Feature Space, MLP Reconstruction, Hadamard Product, Multi-tasking

## TL;DR

Proposed Split-Layer, which splits the MLP fully connected layer into multiple parallel branches and integrates their outputs using the Hadamard product. Without increasing parameters or computation, this exponentially expands the feature space dimensionality from $C$ to $\binom{C/\sqrt{N}+N-1}{N}$, significantly enhancing the representation capability of Implicit Neural Representations (INR).

## Background & Motivation

### Problem Definition

Implicit Neural Representations (INRs) model signals as continuous functions using neural networks and are widely applied in solving inverse problems. However, the representation capability of INR is limited by the feature space dimensionality of the MLP architecture. Specifically:

- **Limitations of Fully Connected Layers**: For a fully connected layer with a width of $C$, the feature space formed by its outputs is a $C$-dimensional Euclidean space. Each output element is a linear combination of input elements, meaning the feature space dimensionality scales linearly with the layer width.
- **Prohibitively Expensive Scaling**: Linearly increasing the layer width (e.g., to $2C$) results in a linear growth of the feature space dimensionality, but the parameters grow quadratically (from $C^2$ to $4C^2$), leading to unacceptable computational overhead.

### Limitations of Prior Work

Existing methods to enhance INR representation capabilities mainly fall into two categories:

**Coordinate Encoding**: e.g., Fourier positional encoding (PEMLP), hash tables (InstantNGP), which map low-dimensional coordinates to high-dimensional manifolds.

**Special Activation Functions**: e.g., SIREN (periodic sine), WIRE (wavelet), FINER (variable-periodic), etc.

However, these methods essentially introduce **learning bias** to make certain features easier to learn, rather than fundamentally expanding the range of learnable features. They do not expand the dimensionality of the feature space from a model architecture perspective.

### Design Motivation

The authors argue that the root of the problem lies in the fully connected mechanism of MLPs—the feature space dimensionality is **linearly related** to the number of neurons. If the connection mechanism can be reorganized into a "split" style, the feature space dimensionality can be **exponentially expanded** while maintaining the same parameter count.

## Method

### Overall Architecture

The core idea of Split-Layer is remarkably simple: split the fully connected layer into $N$ parallel branches, each with an independent weight matrix, and then integrate the outputs of each branch using the **Hadamard product (element-wise multiplication)**. This operation elevates the linear combination to a **high-order polynomial**, thereby constructing a high-dimensional feature space far exceeding that of a standard MLP.

### Key Designs

#### 1. **Split-Layer Structure**: Splitting the fully connected layer into a multi-branch Hadamard product form

**Mechanism**: The output of the $l$-th layer of a standard fully connected layer is:

$$z_i^l = \sum_{j=1}^{C} w_{ij} z_j^{l-1}$$

This is a linear combination of $C$ linearly independent elements, resulting in a feature space dimensionality of $C$.

Split-Layer splits this layer into $N$ branches, where the weight matrix of each branch is $\mathbf{W}_n^l \in \mathbb{R}^{C/\sqrt{N} \times C/\sqrt{N}}$, and the outputs are integrated via the Hadamard product:

$$z_i^l = \prod_{n=1}^{N} \left(\sum_{j=1}^{C/\sqrt{N}} w_{ij}^n z_j^{l-1}\right)$$

Upon expansion, this yields:

$$z_i^l = \sum_{(j_1,j_2,...,j_N)} \left(\prod_{n=1}^{N} w_{ij_n}^n\right) \left(z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}\right)$$

This forms an **$N$-th order homogeneous polynomial**, where the different terms $z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}$ are linearly independent of each other.

**Design Motivation**: Through the polynomial product, the feature space is expanded from a linear space to a polynomial space. The total number of different terms is equivalent to combinations with repetition of choosing $N$ elements from $C/\sqrt{N}$:

$$\text{特征空间维度} = \binom{C/\sqrt{N}+N-1}{N}$$

When $C=256, N=2$, the original feature space is 256-dimensional, while Split-Layer expands it to $\binom{181+1}{2} = 16,471$ dimensions—an expansion of approximately 64 times, while keeping the parameter count unchanged (each branch has $(C/\sqrt{N})^2$ parameters, resulting in a total of $C^2$ across $N$ branches).

#### 2. **Selection of the Optimal Split Number**: Balancing Feature Space Dimensionality and Weight Matrix Freedom

**Mechanism**: Increasing the split number $N$ enlarges the feature space dimensionality but also shrinks the weight matrix size of each branch, reducing its freedom to explore feature combinations. Empirically, the optimal split number is found to be:

$$N^* \approx (0.17C)^{2/3}$$

**Design Motivation**: This is the optimal trade-off point between feature space dimensionality expansion and the expressiveness of the weight matrix. The authors validated the robustness of this formula through 2D image fitting experiments across different network widths $C$—the optimal results consistently appeared near the theoretical surface.

#### 3. **Universality Design**: Serving as a Plug-and-Play Module to Adapt to Different INR Backbones

**Mechanism**: Split-Layer replaces all hidden fully connected layers in the INR, making it applicable to various INR architectures (ReLU MLP, SIREN, Gauss, PEMLP, WIRE, FINER).

**Design Motivation**: Split-Layer represents an architectural improvement that is orthogonal to input encoding methods and activation functions, allowing it to be combined with various existing methods that enhance INR representation capabilities.

### Neural Tangent Kernel Perspective Validation

From the Neural Tangent Kernel (NTK) perspective, the NTK eigenvalue distribution of Split-MLP is more uniform, expanding from $[10^{-3}, 10^{0}]$ to $[10^{-2}, 10^{2}]$, which indicates better convergence performance on high-frequency components. This further theoretically validates the effectiveness of Split-Layer.

### Loss & Training

- All tasks use standard loss functions (e.g., L2 distance, cross-entropy loss), requiring no special designs.
- Weight Initialization: SIREN and FINER use their respective specific initialization schemes, while the rest utilize the default LeCun initialization.
- Optimizer: Adam, with the number of training epochs varying by task.
- In experiments, $N=2$ (i.e., 2-split) is set consistently, which already yields outstanding results.

## Key Experimental Results

### Main Results

Split-Layer was comprehensively evaluated on 6 INR backbones across 4 tasks.

| Task | Backbone | Baseline | Split | Gain |
|------|------|----------|-------|------|
| 2D Image Fitting (PSNR↑) | ReLU | 21.24 | 30.89 | **+45.43%** |
| 2D Image Fitting (PSNR↑) | SIREN | 38.52 | 39.25 | +1.90% |
| 2D Image Fitting (PSNR↑) | PEMLP | 29.60 | 40.78 | **+37.77%** |
| 2D Image Fitting (PSNR↑) | Gauss | 31.74 | 40.84 | **+28.67%** |
| CT Reconstruction (PSNR↑) | SIREN | 18.32 | 29.11 | **+58.90%** |
| CT Reconstruction (PSNR↑) | PEMLP | 28.11 | 32.29 | +14.87% |
| 3D Shape Representation (CD↓) | ReLU | 1.00e-4 | 2.01e-5 | **+79.90%** |
| 3D Shape Representation (CD↓) | Gauss | 2.19e-5 | 5.33e-6 | **+75.66%** |

**5D Novel View Synthesis (NeRF scenes, PSNR↑)**:

| Method | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship | Mean |
|------|-------|-------|-------|--------|------|-----------|-----|------|------|
| NeRF | 31.37 | 24.50 | 28.90 | 34.94 | 30.71 | 28.60 | 28.99 | 27.27 | 29.41 |
| Split-NeRF | 31.78 | 24.81 | 29.34 | 35.33 | 31.76 | 28.87 | 31.85 | 27.83 | **30.20** |
| DINER | 34.49 | 25.43 | 33.28 | 36.45 | 34.82 | 29.58 | 33.43 | 29.25 | 32.09 |
| Split-DINER | **34.85** | 25.47 | **33.39** | **36.92** | **35.14** | 29.59 | **34.01** | **29.49** | **32.36** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| $N=2$ (Default) | Best or near-best | Optimal balance between practicality and performance |
| Optimal $N$ under different $C$ | Matches the $(0.17C)^{2/3}$ surface | Validates the robustness of the optimal split formula |
| Feature Visualization | More diverse features after Split | 9 features -> 45/84 different feature bases |
| NTK Eigenvalue Distribution | Split-MLP is more uniform | Better convergence performance on high frequencies |

### Key Findings

1. **ReLU and SIREN Benefit the Most**: ReLU achieves a 45% improvement on the image fitting task, and SIREN gains 59% on CT reconstruction, indicating that the weaker the representation capability of the original backbone, the more significant the improvement brought by Split-Layer.
2. **Split-PEMLP Performs Outstandingly**: Achieving the best performance in both image fitting and shape representation tasks.
3. **Zero Extra Cost**: Split-Layer achieves significant improvements without adding parameters or computational overhead.
4. **Strong Versatility**: Improvements are observed across all combinations of 6 backbones $\times$ 4 tasks.

## Highlights & Insights

1. **Theoretical Elegance**: Analyzing the INR representation capability from the perspective of feature space dimensionality, which reduces the problem to combinations with repetition in combinatorics, offering clean and clear theoretical derivations.
2. **Simple Implementation**: Merely requires replacing the fully connected layer with a multi-branch Hadamard product structure, with no modifications needed for the training pipeline or loss functions.
3. **Orthogonal to Existing Methods**: Split-Layer is non-conflicting with positional encoding or activation functions, allowing them to be combined.
4. **Validation via NTK Perspective**: Further elucidating the theoretical foundation of performance improvements from the viewpoint of the Neural Tangent Kernel (NTK).

## Limitations & Future Work

1. **Empirical Nature of the Optimal Split Formula**: $N^* \approx (0.17C)^{2/3}$ lacks a rigorous theoretical derivation and is derived solely through experimental fitting.
2. **Potential Training Instability from Hadamard Product**: High-order polynomials may introduce gradient explosion or vanishing issues, which are not deeply analyzed in the paper.
3. **Only Validated in INR Scenarios**: The universality of Split-Layer has not been validated in broader deep learning tasks such as classification or detection.
4. **Unexplored Combinations with Hash Grid Methods**: The effects of combining Split-Layer with methods like InstantNGP are worth exploring.

## Related Work & Insights

- Connection to MFN (Multiplicative Filter Networks): MFN also uses multiplicative operations to combine multi-branch outputs, but Split-Layer offers a clearer theoretical analysis (feature space dimensionality calculation).
- Hilbert kernels or Gaussian kernels may provide pathways for further expanding the feature space (as mentioned in the authors' conclusion).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Rethinking INR representation capability from the perspective of fully connected layer structure, which is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 4 tasks $\times$ 6 backbones, but lacks quantitative comparison of computational efficiency.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear theoretical derivations and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Highly practical as a plug-and-play general-purpose INR enhancement module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICLR 2026\] Neural Compression of 3D Meshes using Sparse Implicit Representation](../../ICLR2026/3d_vision/neural_compression_of_3d_meshes_using_sparse_implicit_representation.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](../../CVPR2026/3d_vision/ntk-guided_implicit_neural_teaching.md)

</div>

<!-- RELATED:END -->
