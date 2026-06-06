---
title: >-
  [Paper Note] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints
description: >-
  [NeurIPS 2025][3D Vision][Depth from Focus] This paper proposes DualFocus, which achieves robust and accurate depth estimation from focal stacks via two complementary constraints: a spatial variational constraint (exploi…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Depth from Focus"
  - "Variational Constraints"
  - "Focal Stack"
  - "Depth Estimation"
  - "Spatio-Focal Dual Constraints"
date: 2026-05-08
content_hash: 2827fd9dc402fa43
---

# DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2509.21992](https://arxiv.org/abs/2509.21992)  
**Code**: Not released  
**Area**: 3D Vision
**Keywords**: Depth from Focus, Variational Constraints, Focal Stack, Depth Estimation, Spatio-Focal Dual Constraints

## TL;DR

This paper proposes DualFocus, which achieves robust and accurate depth estimation from focal stacks via two complementary constraints: a spatial variational constraint (exploiting focus-dependent gradient patterns to distinguish depth edges from texture artifacts) and a focal variational constraint (enforcing a unimodal and monotonic focus probability distribution along the focal axis).

## Background & Motivation

Depth-from-Focus (DFF) infers depth from focal stacks—image sequences captured at varying focus distances—and offers advantages such as no specialized hardware requirements and no scale ambiguity. However, existing learning-based methods suffer from two key limitations:

1. **Texture–depth edge confusion**: When regressing depth directly from image features, strong texture gradients are easily misidentified as depth discontinuities, particularly in regions with repetitive textures.
2. **Insufficient modeling along the focal dimension**: Existing methods typically process each focal plane independently or apply only limited regularization, failing to exploit the physical continuity of focus probability along the focal axis (which should follow a unimodal distribution).

The core insight is that the same scene point exhibits distinct gradient patterns across focal planes in a focal stack—in-focus regions yield consistent and strong gradients, while out-of-focus regions produce diffuse or noisy gradients. Leveraging this cross-focal-plane gradient variation enables indirect sharpness inference and facilitates the discrimination of genuine depth edges.

## Core Problem

How can spatial-domain and focal-domain physical priors be jointly exploited in DFF to improve depth estimation accuracy and robustness in complex scenes with fine textures and abrupt depth transitions?

## Method

### Focal Volume Construction

Given $N$ images captured at different focus distances, features are extracted and stacked along the focal dimension to form a 4D focal volume $V \in \mathbb{R}^{H \times W \times C_1 \times N}$. Focal-dimension differences are computed and concatenated to obtain an enhanced volume:

$$V_n^* = \begin{cases} [V_n, V_{n+1} - V_n], & n = 1, \ldots, N-1 \\ [V_n, V_n - V_{n-1}], & n = N \end{cases}$$

### Spatial Variational Constraint

The network predicts multi-channel gradient features $\Gamma_n \in \mathbb{R}^{2HW \times C_2}$ for each focal plane, encoding depth variation cues along the x/y directions. To ensure global integrability (avoiding noisy gradients that cannot correspond to a real surface), these are projected onto an integrable gradient field via least-squares:

$$z_n^{*(c)} = \arg\min_z \|Pz - \Gamma_n^{(c)}\|_2^2 = (P^\top P)^{-1} P^\top \Gamma_n^{(c)}$$

where $P$ is a fixed finite-difference operator. The reconstructed implicit surface $z_n^*$ yields consistent geometric structures at in-focus planes and noisy surfaces at out-of-focus planes—a disparity that naturally encodes geometric reliability.

Supervision of $z_n^*$ is applied only at in-focus regions, with per-pixel per-plane sharpness weights defined as:

$$q_n(\mathbf{x}) = \frac{\exp(-|f_n - D^*(\mathbf{x})|)}{\sum_{m=1}^N \exp(-|f_m - D^*(\mathbf{x})|)}$$

The spatial variational loss is:

$$L_{\text{sv}} = \sum_{\mathbf{x},n} q_n(\mathbf{x}) \|\nabla D^*(\mathbf{x}) - \theta_{\text{grad}}(z_n^*)(\mathbf{x})\|_1$$

### Focal Variational Constraint

The focus probability $p_n(\mathbf{x})$ should peak at the correct depth and decrease monotonically on both sides. A bidirectional soft-monotonicity loss is defined as:

$$L_{\text{fv}} = \sum_{\mathbf{x}} \left( \sum_{i=1}^{k(\mathbf{x})-1} (\max(0, p_i - p_{i+1}))^2 + \sum_{i=k(\mathbf{x})}^{N-1} (\max(0, p_{i+1} - p_i))^2 \right)$$

where $k(\mathbf{x}) = \arg\max_n p_n(\mathbf{x})$. Decreasing values before the peak and increasing values after the peak are penalized respectively.

### Depth Fusion and Total Loss

The reconstructed surface features are concatenated with the focal volume and decoded via 3D convolutions to produce a focus probability map $p \in \mathbb{R}^{H' \times W' \times N}$, from which depth is obtained by weighted summation over focal distances:

$$\hat{D}(\mathbf{x}) = \sum_{n=1}^N p_n(\mathbf{x}) f_n$$

The total loss is $L = L_{\text{depth}} + \lambda_{\text{sv}} L_{\text{sv}} + \lambda_{\text{fv}} L_{\text{fv}}$, where $L_{\text{depth}}$ is the smooth L1 loss.

## Key Experimental Results

### NYU Depth v2 (Synthetic Focal Stack)

| Method | Type | RMSE ↓ | AbsRel ↓ | δ₁ ↑ |
|--------|------|--------|----------|------|
| Depth Anything | SIDE | 0.206 | 0.056 | 0.984 |
| HybridDepth | DFF | 0.128 | 0.026 | 0.995 |
| DFV | DFF | 0.094 | 0.020 | 0.998 |
| **DualFocus** | **DFF** | **0.075** | **0.013** | **0.999** |

Compared to DFV: RMSE reduced by 20.2%, AbsRel reduced by 35.0%.

### FoD500

| Method | MSE ↓ | RMSE ↓ | Bump ↓ |
|--------|-------|--------|--------|
| DFV | 0.020 | 0.129 | 1.43 |
| **DualFocus** | **0.015** | **0.112** | **1.31** |

### DDFF 12-Scene

| Method | MSE ↓ | RMSE ↓ | δ₁ ↑ |
|--------|-------|--------|------|
| HybridDepth | 5.1×10⁻⁴ | 0.0200 | 0.789 |
| **DualFocus** | **4.7×10⁻⁴** | **0.0194** | **0.800** |

### Zero-Shot Transfer (ARKitScenes)

| Method | Type | RMSE ↓ | AbsRel ↓ | Params |
|--------|------|--------|----------|--------|
| Depth Anything | SIDE | 0.53 | 0.32 | 336M |
| HybridDepth | DFF | 0.29 | 0.42 | 67M |
| **DualFocus** | **DFF** | **0.28** | **0.40** | **27M** |

### Ablation Study

Removing both constraints increases RMSE from 0.075 to 0.094. The spatial variational constraint contributes more than the focal variational constraint, as it directly encodes surface gradient information at each focal plane.

## Highlights & Insights

1. The projection of gradient fields onto an integrable space is elegant—it not only regularizes gradients but also naturally encodes the distinction between in-focus and out-of-focus regions.
2. The focal variational constraint leverages a physical prior (unimodality of focus probability) in a form that is both concise and effective.
3. With only 27M parameters—far fewer than SIDE models (336M)—DualFocus achieves superior zero-shot transfer performance.
4. State-of-the-art results are demonstrated across four datasets.

## Limitations & Future Work

- The NYU dataset relies on synthetic focal stacks, introducing a domain gap with real focal scans.
- The $N$-frame input requirement of focal stacks limits real-time applicability.
- Performance on regions with severely lacking texture (e.g., plain white walls) is not analyzed in detail.
- Comparisons with recent large-scale models such as Depth Anything V2 are absent.

## Related Work & Insights

- **vs. DFV**: DFV captures only first-order derivatives along the focal dimension, whereas DualFocus jointly models spatial gradient variation and focal probability distributions.
- **vs. HybridDepth**: HybridDepth relies on a pretrained relative depth model; DualFocus is fully end-to-end with fewer parameters.
- **vs. VA-DepthNet**: VA-DepthNet applies variational constraints to single images; DualFocus extends this paradigm to focal stacks, exploiting cross-focal-plane gradient discrepancies.

The variational constraint paradigm is generalizable to other multi-view or multi-condition depth estimation tasks. Incorporating integrability constraints as an inductive bias into network training represents a compelling approach to injecting physical priors.

## Rating

- ⭐ Novelty: 8/10 — The spatio-focal dual variational constraint design is novel; integrability projection is applied to DFF for the first time.
- ⭐ Experimental Thoroughness: 8/10 — Four datasets, zero-shot transfer, and ablation studies are included, though large-scale validation on real focal stacks is lacking.
- ⭐ Writing Quality: 8/10 — Mathematical derivations are clear and motivation is well articulated.
- ⭐ Value: 7/10 — A strong contribution to the DFF field, though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](../../ICCV2025/3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[ICML 2026\] FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](../../ICML2026/3d_vision/fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)
- [\[NeurIPS 2025\] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)

</div>

<!-- RELATED:END -->
