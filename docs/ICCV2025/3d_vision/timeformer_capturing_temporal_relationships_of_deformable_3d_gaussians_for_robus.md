---
title: >-
  [Paper Note] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction
description: >-
  [ICCV 2025][3D Vision][Dynamic scene reconstruction] This paper proposes the TimeFormer module, which implicitly learns temporal relationships among deformable 3D Gaussians via a cross-time Transformer encoder…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Dynamic scene reconstruction"
  - "3D Gaussian Splatting"
  - "Transformer"
  - "temporal relationship modeling"
  - "plug-and-play"
date: 2026-05-08
content_hash: 95bba0286fd9414d
---

# TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2411.11941](https://arxiv.org/abs/2411.11941)  
**Code**: [Project Page](https://patrickddj.github.io/TimeFormer/)  
**Area**: 3D Vision
**Keywords**: Dynamic scene reconstruction, 3D Gaussian Splatting, Transformer, temporal relationship modeling, plug-and-play

## TL;DR

This paper proposes the TimeFormer module, which implicitly learns temporal relationships among deformable 3D Gaussians via a cross-time Transformer encoder, and introduces a dual-stream optimization strategy that transfers motion knowledge during training with no additional overhead at inference.

## Background & Motivation

In dynamic scene reconstruction, existing methods extend 3DGS to dynamic scenes via deformation fields, but suffer from a critical limitation: **insufficient temporal relationship modeling**.

Existing deformation field designs (MLP, spatiotemporal planes, polynomial, Fourier series, etc.) learn motion patterns from each timestamp independently, ignoring intrinsic cross-time relationships:
- Some methods introduce motion flow regularization, but only capture **local temporal relationships between adjacent frames**
- For complex scenarios involving **large motions, extreme geometries, and reflective surfaces**, such local perspectives fail to model motion effectively

The root cause is that deformation fields treat each timestamp as an independent input, lacking a global view of the entire temporal sequence. This leads to significant degradation in reconstruction quality under complex motion patterns.

## Method

### Overall Architecture

TimeFormer is a **plug-and-play module** that requires no modification to the architecture of the base deformable 3DGS method. It consists of:
1. **Cross-time Transformer encoder** — models implicit motion patterns across multiple timestamps
2. **Dual-stream optimization strategy** — transfers motion knowledge via shared weights; TimeFormer can be removed at inference

### Cross-Time Transformer Encoder

Different timestamps are treated as a "special batch" analogous to BatchFormer.

Let $\mathcal{T}_s = \{t_i\}_{i=0}^{B-1}$ denote randomly sampled timestamps and $\mathcal{G} \in \mathbb{R}^{N \times (3+C)}$ denote Gaussians in canonical space.

Positions are replicated $B$ times to obtain $\mathcal{G}_c \in \mathbb{R}^{B \times N \times 3}$, which are concatenated with timestamps and passed through positional encoding:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

After $M$ layers of Transformer encoding, a small MLP projects the features into positional offsets:

$$\mathcal{O} = MLP(F_{M-1}), \quad \mathcal{G}_t = \mathcal{G}_c + \mathcal{O}$$

### Gradient Analysis

Without TimeFormer: $\frac{\partial \Delta\mu_i}{\partial \mu} = \frac{\partial \mathcal{D}(\mu, t_i)}{\partial \mu}$

With TimeFormer: $\frac{\partial \Delta\mu_i}{\partial \mu} = \frac{\partial \mathcal{D}(a, t_i)}{\partial a} \cdot (1 + \frac{\partial \mathcal{P}(\mu, \mathcal{T}_s)}{\partial \mu})$

The additional gradient term $\frac{\partial \mathcal{P}(\mu, \mathcal{T}_s)}{\partial \mu}$ allows the current state to be influenced by any past or future state, enabling motion pattern capture from a global temporal perspective.

### Dual-Stream Optimization Strategy

The original branch and the TimeFormer branch share deformation field weights:

$$\mathcal{L}_c = \|Splatting(\mathcal{D}(\mathcal{G}_c, \mathcal{T}), \mathcal{V}) - \mathcal{I}_{gt}\|_1$$
$$\mathcal{L}_t = \|Splatting(\mathcal{D}(\mathcal{G}_t, \mathcal{T}), \mathcal{V}) - \mathcal{I}_{gt}\|_1$$
$$\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_t \mathcal{L}_t$$

where $\lambda_c > \lambda_t$ to prevent the TimeFormer branch from overfitting and degrading inference quality. At inference, the TimeFormer branch is removed, preserving the original rendering speed.

## Key Experimental Results

### Multi-view Dynamic Scenes (Neural 3D Video Dataset)

| Method | Sear Steak PSNR | Flame Salmon PSNR | Coffee Martini PSNR | Mean PSNR | Mean SSIM |
|--------|----------------|-------------------|---------------------|-----------|-----------|
| K-Plane | 32.52 | 30.44 | 29.99 | 31.63 | 0.960 |
| GS4D | 32.92 | 26.39 | 25.23 | 30.07 | 0.936 |
| 4D-Rotor | 32.86 | 28.25 | 27.95 | 31.06 | 0.938 |

### Plug-and-Play Effectiveness

TimeFormer can be applied to various deformable 3DGS backbone networks, consistently yielding improvements in both quality and rendering speed.

### Key Findings

1. **TimeFormer guides a more efficient canonical space distribution** — implicit cross-time relationship learning encourages Gaussians with similar deformations to automatically cluster during optimization, accelerating rendering
2. **Inference FPS actually improves** — although TimeFormer is only used during training, the improved canonical space distribution leads to faster inference
3. **Greatest gains on complex motions** — the largest improvements are observed on scenes with reflective surfaces and fluid dynamics, such as Coffee Martini

## Highlights & Insights

1. **Plug-and-play design** — requires no modification to the base method's architecture; seamlessly integrates into existing deformable 3DGS methods
2. **Zero inference overhead** — through the weight-sharing dual-stream strategy, motion knowledge learned during training incurs no additional computational cost at inference
3. **Motion modeling from a learning perspective** — unlike explicit motion flow or optical flow supervision, TimeFormer automatically extracts motion patterns from RGB supervision
4. **Theoretical analysis of gradient flow** — clearly explains how TimeFormer enables cross-time information propagation via additional gradient terms

## Limitations & Future Work

- Random timestamp sampling may not always cover the most critical motion variations
- The self-attention mechanism in Transformers incurs high computational cost over a large number of timestamps
- Additional training time is required for dual-stream optimization

## Related Work & Insights

- **Deformation field design**: D-NeRF, Deformable-3DGS (MLP / K-Plane / polynomial)
- **Motion modeling**: optical flow supervision (MD-Splatting, D3DG), adjacent-frame-based (DN-4DGS)
- **4D representations**: GS4D, 4D-Rotor

## Rating

- Novelty: ⭐⭐⭐⭐ (cross-time Transformer + zero-inference-overhead design is novel)
- Technical Depth: ⭐⭐⭐⭐ (clear gradient analysis; dual-stream strategy is well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐ (validated across multiple datasets and backbones)
- Value: ⭐⭐⭐⭐⭐ (plug-and-play; deployment-friendly)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa_gaussians_anti_aliased_artifact_free_3d_gaussian_rendering.md)

</div>

<!-- RELATED:END -->
