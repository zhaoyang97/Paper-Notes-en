---
title: >-
  [Paper Note] CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices
description: >-
  [NeurIPS 2025][Image Generation][Normalizing flows] CDFlow is proposed to construct invertible linear layers via alternating products of circulant and diagonal matrices…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Normalizing flows"
  - "circulant matrices"
  - "diagonal matrices"
  - "invertible linear layers"
  - "fast Fourier transform"
  - "density estimation"
date: 2026-05-08
content_hash: 506845e91c94f19f
---

# CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices

**Conference**: NeurIPS 2025
**arXiv**: [2510.25323](https://arxiv.org/abs/2510.25323)  
**Code**: To be confirmed  
**Area**: Image Generation
**Keywords**: Normalizing flows, circulant matrices, diagonal matrices, invertible linear layers, fast Fourier transform, density estimation

## TL;DR

CDFlow is proposed to construct invertible linear layers via alternating products of circulant and diagonal matrices, reducing parameter complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(mn)$, matrix inversion complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(mn\log n)$, and log-determinant computation from $\mathcal{O}(n^3)$ to $\mathcal{O}(mn)$, outperforming comparable methods on density estimation and periodic data modeling.

## Background & Motivation

Normalizing flows achieve exact likelihood estimation and efficient sampling via invertible transformations. The central challenge lies in designing linear layers that are simultaneously **expressive, and efficient in computing Jacobian determinants and inverse transformations**.

Limitations of existing approaches:
- **Glow's 1×1 convolution**: LU decomposition reduces determinant cost to $\mathcal{O}(n)$, but inversion remains $\mathcal{O}(n^2)$ with $\mathcal{O}(n^2)$ parameters.
- **Emerging/Periodic convolutions**: Determinant computation for periodic convolutions costs $\mathcal{O}(n^3)$; 2D FFT incurs large memory overhead.
- **ButterflyFlow**: Butterfly matrices reduce parameters, but inversion remains $\mathcal{O}(n^2)$.
- **Woodbury transforms**: Theoretical complexity $\mathcal{O}(dn)$, but repeated input transformations lead to suboptimal practical efficiency.

Key mathematical foundation: any $n \times n$ matrix can be expressed as an alternating product of at most $2n-1$ circulant and diagonal matrices, and circulant matrices are diagonalizable via FFT. This paper introduces such decomposition into flow models, simultaneously addressing both the determinant and inversion bottlenecks.

## Method

### Overall Architecture

Each flow module in CDFlow consists of three components: ActNorm layer → CD-Convolution layer → Coupling layer. Multiple flow modules are stacked $K$ times to form a block, and multiple blocks compose a multi-scale architecture (with split and squeeze operations).

### Key Designs

**Weight matrix construction**: An alternating product of $m$ diagonal matrices and $m-1$ circulant matrices:

$$\mathbf{W} = \text{diag}(\mathbf{d}_1) \times \text{circ}(\mathbf{c}_2) \times \cdots \times \text{diag}(\mathbf{d}_{2m-1})$$

In practice, $m=2$ is used (two diagonal matrices + one circulant matrix), storing only the diagonal vectors and the frequency-domain eigenvalues $\hat{\mathbf{c}}_{2j} = \mathbf{F} \times \mathbf{c}_{2j}$ of the circulant matrix.

**Efficient log-determinant computation**:

$$\log|\det(\mathbf{W})| = \sum_{j=1}^{m} \sum_{i=1}^{n} \log|\hat{c}_{2j}^i| + \sum_{j=1}^{m-1} \sum_{i=1}^{n} \log|d_{2j-1}^i|$$

Leveraging the properties that the determinant of a diagonal matrix equals the product of its diagonal entries and that of a circulant matrix equals the product of its frequency-domain eigenvalues, the full computation reduces to simple summation with complexity $\mathcal{O}(mn)$.

**Efficient matrix inversion**:

$$\mathbf{W}^{-1} = \mathbf{D}_{2m-1}^{-1} \times \cdots \times \mathbf{C}_2^{-1} \times \mathbf{D}_1^{-1}$$

Diagonal matrix inversion amounts to element-wise reciprocals ($\mathcal{O}(n)$); circulant matrix inversion corresponds to reciprocals of frequency-domain eigenvalues followed by IFFT ($\mathcal{O}(n \log n)$), yielding total complexity $\mathcal{O}(mn \log n)$.

**Frequency-domain parameterization**: Directly storing frequency-domain parameters $\hat{\mathbf{c}}_{2j}$ rather than time-domain $\mathbf{c}_{2j}$ avoids repeated FFT operations across matrix multiplication, determinant, and inversion computations.

### Loss & Training

Standard normalizing flow negative log-likelihood:

$$\mathcal{L} = -\log p_\theta(x) = -\log p_Z(z) - \sum_{i=1}^{L} \log\left|\det \frac{\partial f_i}{\partial f_{i-1}}\right|$$

where $z = f_\theta(x)$ and $p_Z$ is the standard Gaussian. The core contribution of CDFlow lies in the efficient computation of the intermediate term $\log|\det \mathbf{W}|$.

## Key Experimental Results

### Main Results

Density estimation BPD (bits per dimension, lower is better):

| Model | Parameters | CIFAR-10 BPD↓ | ImageNet 32×32 BPD↓ | Galaxy BPD↓ |
|:--|:--:|:--:|:--:|:--:|
| Real NVP | 6.4M/46.2M | 3.49 | 4.28 | 2.11 |
| Glow | 44.2M/66.2M | 3.36 | 4.09 | 2.02 |
| Emerging | 46.6M/44.0M | 3.34 | 4.09 | 1.98 |
| Woodbury | 45.3M/45.3M | 3.42 | 4.09 | 2.01 |
| ButterflyFlow | 44.4M/44.4M | 3.33 | 4.09 | 1.95 |
| Residual Flows | 25.2M/47.1M | 3.28 | 4.01 | 3.60 |
| i-DenseNet | 24.9M/47.0M | 3.25 | 3.98 | 4.06 |
| **CDFlow (Ours)** | **44.2M/44.2M** | **3.31** | **4.04** | **1.92** |

CDFlow achieves the best results among comparable architectures (Glow/Emerging/Woodbury/ButterflyFlow) and demonstrates a substantial advantage on the Galaxy dataset (1.92 vs. 1.95), highlighting its superiority in modeling periodic data.

### Ablation Study

Effect of hyperparameter $m$ (CIFAR-10):

| $m$ | CD-Conv Parameters (K) | BPD↓ |
|:--:|:--:|:--:|
| 1 | ~2K | ~3.36 |
| 2 | ~3K | 3.31 |
| 3 | ~4K | ~3.30 |
| 4 | ~5K | ~3.30 |

BPD improvement becomes marginal for $m \geq 2$ while parameters and computation grow linearly, making $m=2$ the optimal trade-off point.

Runtime comparison (at 96 channels):

| Method | Logdet Speedup | Inverse Speedup |
|:--|:--:|:--:|
| CDFlow vs. 1×1 Conv | **4.31×** | **1.17×** |

### Key Findings

1. The circulant-diagonal decomposition substantially reduces parameters and computation while preserving expressiveness — at $m=2$, only three vectors (two diagonal + one circulant) are needed to approximate an arbitrary matrix.
2. Galaxy experiments confirm the natural advantage of circulant matrices for periodic data (4.95% BPD improvement over Glow).
3. In 2D toy experiments under the Flow Matching framework, CDMLP achieves optimal or near-optimal NLL/FID with the fewest parameters.
4. Frequency-domain parameterization is a critical engineering decision — avoiding repeated FFT operations ensures that practical speedups align with theoretical complexity reductions.

## Highlights & Insights

- ⭐ Elegantly addresses both major computational bottlenecks in normalizing flows (determinant + inversion) simultaneously, unlike prior works that optimize only one.
- ⭐ Grounded in classical matrix decomposition theory (Huhtanen 2015), providing a rigorous mathematical foundation with controllable approximation accuracy.
- The periodic structure of circulant matrices confers a natural advantage for data with periodic characteristics (e.g., astronomical images).
- Compatibility with the Flow Matching framework validates the generality of the approach.

## Limitations & Future Work

- Currently supports only 1×1 convolutions; extension to $d \times d$ convolutions has not yet been addressed, limiting spatial information fusion.
- On CIFAR-10, CDFlow surpasses comparable architectures but falls short of distinct architectures such as Residual Flows and i-DenseNet.
- Experiments are limited to 32×32 resolution; performance on high-resolution image generation remains unvalidated.
- Stable training requires spectral normalization and channel-aware learning rate scaling, increasing tuning complexity.

## Related Work & Insights

From Glow's LU decomposition to ButterflyFlow's butterfly matrices and now to CDFlow's circulant-diagonal decomposition, a consistent thread emerges: exploiting the structural properties of structured matrices to trade expressiveness for computational efficiency. Future directions include exploring circulant-diagonal structures in other settings requiring invertible transformations (e.g., invertible ResNets, ODE solvers) and extending the approach to $d \times d$ convolutions.

## Rating

⭐⭐⭐⭐ (4/5)

| Dimension | Rating |
|:--|:--:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

The mathematical derivations are concise and elegant, simultaneously resolving both the determinant and inversion bottlenecks. Weaknesses include limited experimental scale, restriction to 1×1 convolutions, and lack of comprehensive comparison with modern generative models (diffusion models/Flow Matching).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](../../CVPR2026/image_generation/leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[NeurIPS 2025\] Knowledge Distillation Detection for Open-weights Models](knowledge_distillation_detection_for_open-weights_models.md)

</div>

<!-- RELATED:END -->
