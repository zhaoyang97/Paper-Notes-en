---
title: >-
  [Paper Note] Learning Unified Representation of 3D Gaussian Splatting
description: >-
  [ICLR 2026][3D Vision][3D Gaussian Splatting] The native 3DGS parameters $\boldsymbol{\theta}=\{\mu,\mathbf{q},\mathbf{s},\mathbf{c},o\}$ suffer from non-uniqueness and numerical heterogeneity…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Submanifold Field Representation"
  - "Representation Uniqueness"
  - "VAE"
  - "Optimal Transport"
date: 2026-05-08
content_hash: 33a5911ccf62462c
---

# Learning Unified Representation of 3D Gaussian Splatting

**Conference**: ICLR 2026
**arXiv**: [2509.22917](https://arxiv.org/abs/2509.22917)  
**Code**: [GitHub](https://github.com/cilix-ai/gs-embedding)  
**Area**: 3D Vision / Representation Learning
**Keywords**: 3D Gaussian Splatting, Submanifold Field Representation, Representation Uniqueness, VAE, Optimal Transport

## TL;DR

The native 3DGS parameters $\boldsymbol{\theta}=\{\mu,\mathbf{q},\mathbf{s},\mathbf{c},o\}$ suffer from non-uniqueness and numerical heterogeneity, making them unsuitable as a learning space for neural networks. This paper proposes the **Submanifold Field (SF)** representation: each Gaussian primitive is mapped to a continuous color field defined on its iso-probability ellipsoidal surface. The paper proves this mapping is injective, fundamentally eliminating parameter ambiguity. Combined with a VAE trained using an optimal-transport-based Manifold Distance (M-Dist), the approach comprehensively outperforms parameter-based baselines in reconstruction fidelity, cross-domain generalization, and latent space stability.

## Background & Motivation

3DGS has become a core method for 3D reconstruction and rendering, and an increasing number of downstream tasks—compression (Shin et al.), generation (Yi et al.), semantic understanding (Guo et al.)—directly use Gaussian parameters $\boldsymbol{\theta}$ as network inputs/outputs. However, this practice implicitly entails three fundamental problems:

1. **Non-uniqueness**: Quaternion sign ambiguity ($\mathbf{q}$ and $-\mathbf{q}$ represent the same rotation), geometric symmetry, and rotation–SH interactions yield equivalent parameter combinations, forming many-to-one mappings that produce contradictory gradient signals during training. Experimentally, simply negating the quaternion ($\mathbf{q}\to-\mathbf{q}$) causes a parameter autoencoder to fail entirely at decoding.
2. **Numerical Heterogeneity**: Position $\mu\in\mathbb{R}^3$ can span a large range, quaternions are unit-normalized, pre-activation scales range from $-15$ to $3$, and SH coefficients decay exponentially. Concatenating these into a single vector violates the identical-distribution assumption of standard modules such as BatchNorm.
3. **Manifold Mismatch**: Position lies in $\mathbb{R}^3$, rotation in $\text{SO}(3)$, and scale in $(\mathbb{R}^+)^3$—variables from different manifolds are forced into Euclidean space, destroying intrinsic geometric structure.

In downstream generative tasks, these issues manifest as geometric "jitter" during latent space interpolation, high noise sensitivity, and poor cross-domain (indoor ↔ outdoor) generalization. The paper's **Key Insight**: rather than learning the parameters themselves, learn a geometric-photometric representation with a provably unique mapping.

## Method

### Overall Architecture

Gaussian parameters $\boldsymbol{\theta}$ → Submanifold Field $\mathcal{E}=(\mathcal{M}, F)$ (iso-probability ellipsoidal surface + color field) → discretized into a colored point cloud $\mathcal{P}$ → PointNet encoder producing a 32-dimensional latent variable $\mathbf{z}$ → coordinate transformation network $g_c$ + color field network $g_f$ decoding to reconstructed point cloud $\hat{\mathcal{P}}$ → PCA-fitted covariance + SH fitting to recover original parameters $\hat{\boldsymbol{\theta}}$.

### Key Designs

**1. Submanifold Field Representation**

For each Gaussian primitive, the iso-probability surface at a constant Mahalanobis distance $r$ is taken as the 2D submanifold:

$$\mathcal{M} = \{\mathbf{x}\in\mathbb{R}^3 \mid (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) = r^2 \}$$

A color field $F(\mathbf{x})=\sigma(o)\cdot\text{Color}(\mathbf{d}_\mathbf{x})$ is defined on this ellipsoidal surface, where the viewing direction is $\mathbf{d}_\mathbf{x}=(\mathbf{x}-\mu)/\|\mathbf{x}-\mu\|$. The shape of the ellipsoid encodes rotation and scale, while the color field encodes appearance and opacity. The paper proves **Proposition 2**: distinct SGRFs correspond to distinct submanifold fields $\mathcal{E}$—i.e., the mapping is **injective**—fundamentally eliminating parameter ambiguity.

**2. Manifold Distance (M-Dist)**

Directly applying $L_1/L_2$ metrics in parameter space fails to reflect perceptual quality. The paper defines a Wasserstein-2 distance based on optimal transport:

$$W_2^2(\mathcal{E}, \hat{\mathcal{E}}) = \inf_{\gamma\in\Gamma} \int_{\mathcal{M}\times\hat{\mathcal{M}}} \left(\|\mathbf{x}-\mathbf{y}\|^2 + \lambda\|c_x - c_y\|^2\right) d\gamma$$

This is computed between two colored point clouds in discretized form, with $\lambda$ balancing spatial and color terms. Experiments demonstrate that M-Dist correlates with PSNR/LPIPS far more strongly than $L_1$ parameter distance.

**3. SF-VAE Architecture**

- **Encoder**: PointNet encodes $P=12^2=144$ sampled points into a 32-dimensional latent variable.
- **Decoder**: Seed points are sampled from the unit sphere and transformed by the coordinate network $g_c$ and color field network $g_f$ to produce the reconstructed point cloud.
- **Parameter Recovery**: PCA fits the covariance matrix $\Sigma$; SH basis functions fit the color coefficients $\mathbf{c}$.
- **Training Loss**: $\mathcal{L}_\text{VAE} = \hat{W}_2^2(\mathcal{P}, \hat{\mathcal{P}}) + \beta \cdot D_\text{KL}(f(\mathbf{z}|\mathcal{P}) \| \mathcal{N}(0,\mathbf{I}))$
- **Training Data**: 500,000 randomly generated Gaussian primitives—individual primitives carry no scene semantics, making the model inherently domain-agnostic.

## Key Experimental Results

### Zero-Shot Reconstruction Quality (Trained on Random Data)

| Setting | Input Representation | Encoder/Decoder | PSNR↑ | SSIM↑ | LPIPS↓ | M-Dist |
|------|---------|-------------|-------|-------|--------|--------|
| ShapeSplat | Parameters $\boldsymbol{\theta}$ | MLP/MLP | 37.51 | 0.888 | 0.152 | 0.184 |
| ShapeSplat | Parameters $\boldsymbol{\theta}$ | MLP/SF-Dec | 44.73 | 0.896 | 0.136 | 0.051 |
| **ShapeSplat** | **Submanifold Field** | **SF-VAE** | **63.41** | **0.990** | **0.010** | **0.041** |
| Mip-NeRF 360 | Parameters $\boldsymbol{\theta}$ | MLP/MLP | 18.82 | 0.564 | 0.452 | 0.510 |
| Mip-NeRF 360 | Parameters $\boldsymbol{\theta}$ | MLP/SF-Dec | 20.92 | 0.730 | 0.359 | 0.055 |
| **Mip-NeRF 360** | **Submanifold Field** | **SF-VAE** | **29.83** | **0.953** | **0.079** | **0.048** |

The submanifold field representation surpasses the best parameter baseline by **+18.7 dB** at the object level (ShapeSplat) and **+8.9 dB** at the scene level (Mip-NeRF 360) in PSNR. All three models are matched in parameter count (0.62M / 0.66M / 0.62M); differences are attributable entirely to the choice of representation.

### Cross-Domain Generalization (Train on A → Test on B)

| Train Set | Test Set | Input Representation | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|--------|---------|-------|-------|--------|
| ShapeSplat | Mip-NeRF 360 | Parameters (MLP/MLP) | 9.75 | 0.356 | 0.615 |
| ShapeSplat | Mip-NeRF 360 | **Submanifold Field** | **19.19** | **0.821** | **0.309** |
| Mip-NeRF 360 | ShapeSplat | Parameters (MLP/MLP) | 55.62 | 0.957 | 0.067 |
| Mip-NeRF 360 | ShapeSplat | **Submanifold Field** | **62.58** | **0.990** | **0.014** |

The advantage of the submanifold field representation is even larger in cross-domain settings (object → scene gains nearly **+10 dB**). Notably, the model trained on random data outperforms models transferred from a real domain, confirming the inherent domain-agnostic nature of the submanifold field representation.

### Gaussian Neural Field (GNF) Downstream Validation

| Regression Target | PSNR↑ | SSIM↑ | LPIPS↓ | Parameters |
|---------|-------|-------|--------|--------|
| Raw parameters $\boldsymbol{\theta}$ (ShapeSplat) | 51.66 | 0.925 | 0.141 | 0.21M |
| **SF Embedding** (ShapeSplat) | **58.62** | **0.980** | **0.043** | 0.20M |
| Raw parameters $\boldsymbol{\theta}$ (Mip-NeRF 360) | 19.92 | 0.648 | 0.410 | 1.87M |
| **SF Embedding** (Mip-NeRF 360) | **24.40** | **0.804** | **0.261** | 1.85M |

Regressing SF embeddings from spatial coordinates with a lightweight MLP is substantially easier than regressing raw parameters, validating that the proposed representation is more amenable to neural network learning.

### Sensitivity Analysis & Ablation Study

- **Quaternion Negation Robustness**: Under $\mathbf{q}\to-\mathbf{q}$, the parameter VAE fails completely at decoding; SF-VAE is unaffected, as the submanifold field is inherently invariant to quaternion sign.
- **Noise Robustness**: When noise of varying magnitude is injected into the embedding space, the M-Dist degradation of SF embeddings is far slower than that of parameter embeddings.
- **Interpolation Smoothness**: Linear interpolation in the parameter latent space produces rotation/scale jitter; interpolation in the SF latent space yields smooth transitions.
- **Embedding Dimensionality**: 32 dimensions is the optimal trade-off; quality degrades significantly below 32, with diminishing returns above.
- **Training Data Volume**: Only 2% of data (10,000 samples) suffices to approach near-baseline performance.
- **Discretization Resolution**: $P=12^2=144$ is the optimal sampling point count; higher values yield negligible improvement.
- **Encoding Efficiency**: On an RTX 5090 with batch size 4096, encoding 1M Gaussians takes only 1.72s and decoding takes 4.20s (including 0.48s for the fitting module).

## Highlights & Insights

- **The diagnostic value exceeds that of the solution itself**: The paper clearly formalizes three fundamental defects of parameter-based representations (non-uniqueness, heterogeneity, manifold mismatch), sounding a warning for all methods that use 3DGS parameters as network inputs. The experiment in which a mere quaternion negation causes complete failure is highly persuasive.
- **Mathematical elegance of the submanifold field**: The iso-probability surface is an intrinsic geometric structure of a 3D Gaussian—the ellipsoidal surface encodes rotation and scale, while the color field encodes appearance and opacity. Proposition 2 guarantees uniqueness without requiring special parameterization tricks or normalization.
- **Ingenuity of the domain-agnostic design**: An individual Gaussian primitive, divorced from a scene, carries no semantics, enabling the embedding model to be trained on randomly generated data. As a result, the randomly-trained model generalizes better on real domains than models transferred across real domains.
- **Practical value of M-Dist**: M-Dist provides a metric that correlates more closely with perceptual indicators than $L_1/L_2$ parameter distances, offering direct guidance for training loss selection in future 3DGS learning systems.

## Limitations & Future Work

- The current approach operates at the **single-Gaussian level** and does not model inter-Gaussian relationships; scaling to scene level requires permutation-invariant attention mechanisms.
- Submanifold field discretization introduces a resolution–efficiency trade-off ($P=144$ is empirically optimal).
- M-Dist, being based on optimal transport, incurs higher computational cost than $L_2$.
- Only reconstruction, embedding quality, and unsupervised clustering are demonstrated; integration with a complete generative pipeline (diffusion / flow matching) remains to be explored.
- Temporal extension to dynamic scenes (4D GS) is not investigated.

## Related Work & Insights

- **vs. Direct Parameter Regression (pixelSplat, MVSplat)**: These methods train networks to directly output $\boldsymbol{\theta}$; this paper identifies a fundamental deficiency at the representation level.
- **vs. 3DGS Generation (DiffGS, GaussianDiffusion)**: Performing diffusion in parameter space could benefit from a switch to the submanifold field space, yielding a smoother latent space.
- **vs. 3DGS Compression (HAC, CompGS)**: Compression methods reduce the number of parameters, while the submanifold field improves parameter quality—the two approaches are orthogonal and can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic diagnosis of defects in 3DGS parameter representations, with a theoretically guaranteed unique alternative representation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Rigorous controlled variables; multi-angle validation via quaternion negation, cross-domain, and GNF experiments is convincing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ A complete logical chain from definitions → propositions → proofs → experiments; problem motivation is articulated with exceptional clarity.
- **Value**: ⭐⭐⭐⭐⭐ Provides foundational guidance for all learning systems built on 3DGS parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[ACL 2026\] CodeBind: Decoupled Representation Learning for Multimodal Alignment with Unified Compositional Codebook](../../ACL2026/3d_vision/codebind_decoupled_representation_learning_for_multimodal_alignment_with_unified.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[ICLR 2026\] Weight Space Representation Learning on Diverse NeRF Architectures](weight_space_representation_learning_on_diverse_nerf_architectures.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](../../AAAI2026/3d_vision/point-sra_self-representation_alignment_for_3d_representation_learning.md)

</div>

<!-- RELATED:END -->
