---
title: >-
  [Paper Note] LIM: Large Interpolator Model for Dynamic Reconstruction
description: >-
  [CVPR 2025][3D Vision][4D reconstruction] This paper proposes LIM, the first feed-forward, category-agnostic dynamic 4D asset reconstruction model, which achieves high-quality continuous-time interpolation and mesh tracking with consistent topology in seconds by interpolating between implicit triplane representations via a Transformer and introducing a causal consistency loss.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "triplane interpolation"
  - "feed-forward"
  - "mesh tracking"
  - "causal consistency"
date: 2026-05-08
content_hash: 89360bd56b0d6445
---

# LIM: Large Interpolator Model for Dynamic Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2503.22537](https://arxiv.org/abs/2503.22537)  
**Code**: Provided on the project page  
**Area**: 3D Vision  
**Keywords**: 4D reconstruction, triplane interpolation, feed-forward, mesh tracking, causal consistency

## TL;DR

This paper proposes LIM, the first feed-forward, category-agnostic dynamic 4D asset reconstruction model, which achieves high-quality continuous-time interpolation and mesh tracking with consistent topology in seconds by interpolating between implicit triplane representations via a Transformer and introducing a causal consistency loss.

## Background & Motivation

**Background**: Existing 4D reconstruction methods are either limited to specific categories (e.g., humans, animals) or rely on slow optimization methods (requiring minutes to hours). LRM (Large Reconstruction Model) has demonstrated the potential of feed-forward methods in static 3D reconstruction.

**Limitations of Prior Work**: Although methods like L4GM achieve feed-forward 4D reconstruction, they can only reconstruct discrete keyframes and fail to continuously interpolate over time. Additionally, Gaussian mixture representations struggle to establish correspondences across time steps, making them unable to output tracked meshes ready for production pipelines.

**Key Challenge**: Production environments require time-varying mesh sequences with fixed topology and shared UV textures, which existing methods cannot directly output.

**Key Insight**: To learn temporal interpolation within LRM's triplane representation space, rather than in the image space or parameter space.

**Core Idea**: To perform cross-attention interpolation between the triplane features of two keyframes using a Transformer, combined with a causal consistency loss to achieve continuous-time generalization.

## Method

### Overall Architecture

1. Encoders each keyframe's multi-view images into triplane representations using a pre-trained multi-view LRM.
2. LIM takes the LRM intermediate features $\mathcal{F}_k$ of the $k$-th frame, the image $\mathcal{I}_{k+1}$ of the $(k+1)$-th frame, and the interpolation time $\alpha \in [0,1]$.
3. Outputs the interpolated triplane $\hat{\mathcal{T}}_{k+\alpha}$ through a 6-layer Transformer with cross-attention.
4. Optionally paired with a diffusion model to convert monocular video into multi-view inputs.

### Key Designs

**1. LIM Architecture—Transformer Interpolator Based on LRM Features**
- **Function**: Extracts the intermediate features $\mathcal{F}_k$ of the last 6 layers of LRM, concatenates them with the time encoding $\alpha$, and interacts with the image tokens of the next keyframe via cross-attention to generate the interpolated triplane.
- **Mechanism**: $\hat{\mathcal{T}}_{k+\alpha} = \text{LIM}_\psi(\mathcal{F}_k(\mathcal{I}_k, \Pi_k), \mathcal{I}_{k+1}, \alpha)$.
- **Design Motivation**: Reuses pre-trained LRM features to avoid learning 3D representations from scratch; cross-attention allows the interpolator to perceive appearance changes in the target frame.

**2. Causal Consistency Loss**
- **Function**: Constrains the result of "directly interpolating from $t_0$ to $t_\delta$" to be consistent with the result of "first interpolating to an intermediate time $t_{\alpha_{rand}}$ and then to $t_\delta$".
- **Mechanism**: $\mathcal{L}_{\text{causal}} = \|\text{LIM}(\hat{\mathcal{F}}_{k+\alpha_{rand}}, \mathcal{I}_{k+\delta}, \frac{\delta - \alpha_{rand}}{1-\alpha_{rand}}) - \hat{\mathcal{T}}_{k+\delta}\|^2$, where $\alpha_{rand} \sim \mathcal{U}(0, \delta)$.
- **Design Motivation**: Since supervision is only available at discrete keyframes during training, the causal consistency loss introduces self-supervised signals at arbitrary continuous times $\alpha$, rendering the model a truly temporally smooth interpolator.

**3. Canonical Surface Coordinates and Mesh Tracking**
- **Function**: Trains an auxiliary $\overline{\text{LRM}}$ and $\overline{\text{LIM}}$ to predict canonical surface coordinates, mapping 3D surface points at each time step to the XYZ coordinates of the starting frame.
- **Mechanism**: Extracts the mesh using Marching Cubes on the starting frame, and utilizes nearest-neighbor matching of canonical coordinates in subsequent frames to track vertex positions, maintaining fixed topology and shared UV textures.
- **Design Motivation**: Canonical coordinates provide time-invariant surface identities, avoiding the difficulty of directly finding correspondences between Gaussian mixtures or implicit surfaces.

### Loss & Training

- Triplane MSE loss: $\mathcal{L}_{\mathcal{T}} = \|\hat{\mathcal{T}}_{k+\alpha_m} - \mathcal{T}_{k_m}\|^2$, aligned with the pseudo-ground-truth triplane of LRM.
- Causal consistency loss: $\mathcal{L}_{\text{causal}}$ (as above).
- Total loss: $\mathcal{L}_{\mathcal{T}} + \mathcal{L}_{\text{causal}}$.
- LRM weights are frozen, and only LIM is trained; Adam optimizer with a learning rate of $10^{-4}$.
- Training data: A large-scale artist-created animated mesh dataset.

## Key Experimental Results

### Main Results—Triplane Interpolation Quality

| Method | PSNR↑ | PSNR_FG↑ | LPIPS↓ |
|---|---|---|---|
| Linear (Linear Interpolation) | 20.96 | 11.04 | 0.093 |
| FILM (Frame Interpolation) | 22.05 | 14.98 | 0.082 |
| **LIM (Ours)** | **23.11** | **16.12** | **0.075** |
| Oracle (Upper Bound) | 24.43 | 17.51 | 0.064 |

### Ablation Study—Causal Consistency Loss

| Model | PSNR↑ | PSNR_FG↑ | LPIPS↓ |
|---|---|---|---|
| LIM w/o $\mathcal{L}_{\text{causal}}$ | 22.2 | 15.38 | 0.084 |
| LIM (Full) | **23.11** | **16.12** | **0.075** |

### Mesh Tracking Quality

| Method | PSNR↑ | PSNR_FG↑ | LPIPS↓ |
|---|---|---|---|
| NN-tracing | 20.33 | 16.09 | 0.122 |
| **LIM (Ours)** | **21.56** | **17.11** | **0.096** |

### Monocular Video 4D Reconstruction

| Method | Feed-forward | Inference Time | LPIPS↓ | FVD↓ |
|---|---|---|---|---|
| Consistent4D | ✗ | ~1.5h | 0.429 | 1136.3 |
| TripoSR | ✓ | ~30s | 0.504 | 1427.2 |
| **LIM (Ours)** | ✓ | ~3min | **0.142** | **811.1** |

### Key Findings

1. **Linear interpolation completely fails in moving areas**, with dynamic parts often disappearing; image-space interpolation (FILM) produces ghosting artifacts due to multi-view inconsistency.
2. **The causal consistency loss contributes significantly**: removing it drops PSNR by 0.91 dB, confirming the necessity of continuous-time self-supervision.
3. **Dense temporal interpolation avoids explicit correspondence solving**: LIM can accurately interpolate RGB and XYZ, bypassing the difficult problem of establishing direct keyframe-to-keyframe correspondences.
4. **LIM significantly outperforms optimization-based methods in monocular 4D reconstruction**: achieving an LPIPS of only 0.142 vs 0.429 for Consistent4D, while being much faster.

## Highlights & Insights

- The first feed-forward category-agnostic 4D asset reconstruction model, filling the gap for LRM from static to dynamic.
- The causal consistency loss elegantly solves the continuous-time generalization challenge under only discrete keyframe supervision.
- The canonical coordinates + mesh tracking design allows the output to be directly deployed in game/film production pipelines.
- The recursive nature of LIM (which can accept its own intermediate features as input) enables flexible cascaded inference.

## Limitations & Future Work

- Only trained on synthetic data; generalizing to real data requires an LRM trained on real-world data.
- The tracking performance degrades for thin or fine structures.
- Currently supports only temporal interpolation, not extrapolation.
- Relying on diffusion models to generate multi-view inputs can introduce additional cumulative errors.
- The accumulation of error over longer sequences (e.g., hundreds of frames) remains unexplored.

## Related Work & Insights

- The feed-forward paradigm of LRM demonstrates the feasibility of training on large-scale 3D datasets, which LIM extends to 4D.
- Key difference compared to L4GM: LIM performs interpolation in the triplane space instead of separate reconstructions, and supports mesh tracking.
- The core design of the causal consistency loss can be generalized to other tasks requiring continuous-time generalization.
- The canonical coordinates paradigm can inspire other dynamic reconstruction methods requiring cross-time correspondence.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First feed-forward general-purpose 4D reconstruction model; the causal consistency loss is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple evaluation dimensions including interpolation quality, ablation, mesh tracking, and monocular reconstruction.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clear, and the technical pipeline is progressively introduced.
- **Value**: ⭐⭐⭐⭐ Addresses a core demand in physical production (tracked meshes), demonstrating strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)
- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[CVPR 2026\] iLRM: An Iterative Large 3D Reconstruction Model](../../CVPR2026/3d_vision/ilrm_an_iterative_large_3d_reconstruction_model.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)
- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)

</div>

<!-- RELATED:END -->
