---
title: >-
  [Paper Note] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation
description: >-
  [3D Vision] This paper proposes SL2A-INR, a hybrid architecture combining a single-layer learnable activation block parameterized by Chebyshev polynomials with a ReLU-MLP fusion block…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: ea655b2462a5206f
---

# SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation

| Info | Content |
|------|------|
| Conference | ICCV2025 |
| arXiv | [2409.10836](https://arxiv.org/abs/2409.10836) |
| Code | [GitHub](https://github.com/) |
| Area | 3D Vision / Implicit Neural Representation |
| Keywords | Implicit neural representation, learnable activation function, Chebyshev polynomials, spectral bias, NeRF |

## TL;DR

This paper proposes SL2A-INR, a hybrid architecture combining a single-layer learnable activation block parameterized by Chebyshev polynomials with a ReLU-MLP fusion block, effectively alleviating spectral bias in implicit neural representations and achieving state-of-the-art performance on image fitting, 3D shape reconstruction, and novel view synthesis.

## Background & Motivation

### Spectral Bias

Implicit neural representations (INRs) use MLPs to map continuous coordinates to attribute values (e.g., color, SDF), but suffer from **spectral bias**: networks tend to learn low-frequency components first, making it difficult to accurately represent high-frequency details such as fine textures and complex shapes.

### Existing Solutions and Their Limitations

**Positional encoding** (Fourier Features / NeRF's PE): maps inputs to a high-dimensional space via sinusoidal functions to extract high-frequency features, but residual spectral bias remains.

**Specialized activation functions**:
   - SIREN (sinusoidal activation): performance is highly sensitive to hyperparameters (frequency $\omega_0$) and initialization.
   - Gaussian activation: sensitive to learning rate and batch size.
   - WIRE (wavelet activation): performance degrades under limited parameter budgets.
   - FINER: improves sinusoidal activation but retains residual spectral bias.

**Root cause**: The polynomial expansion coefficients of activation functions decay rapidly; harmonic analysis demonstrates that this decay is the fundamental source of spectral bias.

## Method

### Overall Architecture

SL2A-INR adopts a two-block hybrid architecture:
1. **Learnable Activation Block (LA Block)**: a single layer employing learnable activation functions parameterized by high-order Chebyshev polynomials.
2. **Fusion Block**: a multi-layer ReLU-MLP with low-rank linear layers modulated via skip connections from the LA Block.

### Key Design 1: Learnable Activation Block (LA Block)

Inspired by KAN (Kolmogorov-Arnold Networks), but learnable activations are applied only at the first layer:

$$\Psi(\mathbf{x}) = \begin{pmatrix} \psi_{1,1}(\cdot) & \cdots & \psi_{1,d_0}(\cdot) \\ \vdots & \ddots & \vdots \\ \psi_{d_1,1}(\cdot) & \cdots & \psi_{d_1,d_0}(\cdot) \end{pmatrix} \mathbf{x}$$

Each activation function $\psi_{i,j}$ is expanded using $K$-th order Chebyshev polynomials:

$$\psi_{i,j}(x) = \sum_{k=0}^{K} a_{i,j,k} T_k(\sigma(x))$$

where $T_k: [-1,1] \to [-1,1]$ denotes the Chebyshev polynomial of the first kind, $\sigma(x) = \tanh(x)$ normalizes the input to $(-1,1)$, and $a_{i,j,k}$ are learnable coefficients initialized with Xavier uniform initialization.

**Why Chebyshev polynomials over B-splines:**
- Minimize the maximum approximation error (minimax property), yielding higher accuracy.
- Strong spectral approximation capability, efficiently representing high-frequency components.
- Substantially more efficient than B-splines used in KAN (see Tab. 4: KAN B-spline requires 210 minutes; Chebyshev requires 4.3 minutes; SL2A requires only 0.77 minutes).

### Key Design 2: Fusion Block

A standard ReLU-MLP in which each layer's input is modulated by the LA Block output:

$$\mathbf{z}_1 = \Psi(\mathbf{x})$$
$$\mathbf{z}_l = \phi(\mathbf{W}_l(\mathbf{z}_{l-1} \odot \mathbf{z}_1) + \mathbf{b}_l), \quad l=2,...,L-1$$
$$f_\theta(\mathbf{x}) = \mathbf{W}_L(\mathbf{z}_{L-1} \odot \mathbf{z}_1) + \mathbf{b}_L$$

where $\odot$ denotes element-wise multiplication and $\Psi(\mathbf{x})$ serves as a modulation signal injecting high-frequency information into each layer. Linear layers use low-rank parameterization to balance efficiency.

### Design Rationale

- **A single learnable activation layer suffices**: early MLP layers select low-frequency features; applying high-order polynomials at the first layer is therefore sufficient to capture high-frequency details.
- **Necessity of skip connections**: ensures that high-frequency information learned by the LA Block propagates to subsequent layers.
- **Importance of ReLU**: ablation studies show that removing ReLU leads to a PSNR drop of up to 6.22 dB, confirming that the nonlinearity provided by ReLU is indispensable for expressive capacity.

### Neural Tangent Kernel Analysis

Analysis of the NTK eigenvalue distribution reveals:
- Increasing $K$ slows the decay rate of eigenvalues → stronger high-frequency learning capability.
- Removing skip connections accelerates eigenvalue decay.
- Decay rate ordering: ReLU (fastest) > SIREN > FINER > SL2A (slowest).

## Key Experimental Results

### 2D Image Fitting (DIV2K, 16 images, 512×512)

| Method | Params (K) | Mean PSNR↑ | Mean SSIM↑ |
|------|----------|-----------|-----------|
| WIRE | 91.6 | 30.63 | 0.818 |
| SIREN | 198.9 | 33.47 | 0.896 |
| Gauss | 198.9 | 34.96 | 0.914 |
| ReLU+P.E. | 204.0 | 35.27 | 0.916 |
| FINER | 198.9 | 36.35 | 0.924 |
| **SL2A** | 330.2 | **36.88** | **0.933** |

SL2A surpasses FINER by +0.53 dB PSNR to achieve state-of-the-art, ranking first or second on most individual images.

### 3D Shape Reconstruction (Stanford 3D Scanning Repository)

| Method | Armadillo | Dragon | Lucy | Thai Statue | BeardedMan |
|------|-----------|--------|------|-------------|------------|
| FINER | 0.9899 | 0.9895 | 0.9832 | 0.9848 | 0.9943 |
| Gauss | 0.9768 | 0.9968 | 0.9601 | 0.9900 | 0.9932 |
| ReLU+P.E. | 0.9870 | 0.9763 | 0.9760 | 0.9406 | 0.9939 |
| SIREN | 0.9895 | 0.9409 | 0.9721 | 0.9799 | 0.9948 |
| **SL2A** | **0.9983** | **0.9989** | **0.9988** | **0.9986** | **0.9987** |

SL2A significantly outperforms all competing methods on all five shapes, with IoU approaching 1.0.

### Novel View Synthesis (NeRF Blender dataset, 25 training images)

| Method | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|------|-------|-------|-------|--------|------|-----------|-----|------|
| ReLU+P.E. | 31.32 | 20.18 | 24.49 | 30.59 | 25.90 | 25.16 | 26.38 | 21.46 |
| SIREN | 33.31 | 24.89 | 27.26 | 32.85 | 29.60 | 27.13 | 33.28 | 22.25 |
| FINER | 33.90 | 24.90 | 28.70 | 33.05 | 30.04 | 27.05 | 33.96 | 22.47 |
| **SL2A** | **34.70** | **24.33** | **28.31** | **33.83** | **30.63** | **28.62** | **33.88** | **23.43** |

SL2A outperforms FINER on most scenes, with particularly notable gains on Materials (+1.57 dB), Ship (+0.96 dB), and Lego (+0.59 dB).

### Key Ablation: Chebyshev Order and Skip Connections

- Increasing $K$ generally improves performance.
- Skip connections yield substantial gains (cf. results with and without the asterisk marker).
- $K=256$ with skip connections achieves the best trade-off for image fitting.

### Comparison with KAN

| Method | Params (M) | Time (min) | PSNR | SSIM |
|------|----------|-----------|------|------|
| KAN (B-Spline) | 0.329 | 210.1 | 25.40 | 0.722 |
| KAN (Chebyshev) | 0.203 | 4.27 | 30.50 | 0.845 |
| **SL2A** | 0.330 | **0.77** | **33.40** | **0.892** |

SL2A is 273× faster than KAN B-Spline and achieves 8 dB higher PSNR, demonstrating that the hybrid architecture is substantially superior to a full KAN.

## Highlights & Insights

1. **Solid theoretical grounding**: the solution is motivated by the observation that rapidly decaying polynomial expansion coefficients cause spectral bias, and addresses this by making the coefficients learnable.
2. **Minimalist yet effective design**: only a **single layer** of learnable activations is required (cf. KAN's fully learnable layers), dramatically reducing computational overhead.
3. **Strong robustness**: less sensitive to learning rate and batch size than FINER, Gauss, and SIREN.
4. **NTK analysis**: rigorously explains the effectiveness of the design from a kernel-method perspective.
5. **Bridge from KAN to practical INR**: retains the learnable activation concept from KAN while replacing the costly full-KAN structure with low-rank MLPs.

## Limitations & Future Work

- Parameter count is slightly higher than comparable methods (330K vs. 199K), though efficiency remains acceptable.
- Scalability issues inherited from KAN: very large-scale architectures may be prohibitively expensive.
- Image fitting experiments involve per-image optimization, without evaluation of generalization capability.
- Evaluation is limited to standard NeRF benchmarks; performance on larger-scale 3D reconstruction scenarios remains unverified.

## Related Work & Insights

- **KAN** (Kolmogorov-Arnold Networks): the primary inspiration for learnable activation functions; SL2A simplifies this concept to a single layer.
- **SIREN**: a pioneering INR with sinusoidal activations; SL2A demonstrates that learnable activations outperform hand-crafted designs.
- **FINER**: the primary competing method, which improves sinusoidal activations but retains residual spectral bias.
- **NTK theory**: provides a rigorous theoretical framework for spectral bias analysis.
- The learnable activation paradigm is potentially generalizable to other MLP-based settings, such as NeRF acceleration and 3DGS.

## Rating

⭐⭐⭐⭐ — Theoretically rigorous, elegantly designed, and experimentally comprehensive, achieving consistent state-of-the-art performance across multiple INR tasks. Increased parameter count and large-scale scalability remain potential concerns.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[ICCV 2025\] LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

</div>

<!-- RELATED:END -->
