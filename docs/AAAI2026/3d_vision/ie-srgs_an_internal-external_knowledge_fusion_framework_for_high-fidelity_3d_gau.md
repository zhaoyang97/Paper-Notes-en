---
title: >-
  [Paper Note] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes IE-SRGS, a framework that fuses external knowledge (high-frequency texture priors from a pretrained 2D super-resolution model) with internal knowledge (cr…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Super-Resolution"
  - "Internal-External Knowledge Fusion"
  - "Mip-Splatting"
  - "Depth Estimation"
date: 2026-05-08
content_hash: 9eb9de3bb49013aa
---

# IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2511.22233](https://arxiv.org/abs/2511.22233)
**Code**: Not released (to be released post-review)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Super-Resolution, Internal-External Knowledge Fusion, Mip-Splatting, Depth Estimation

## TL;DR

This paper proposes IE-SRGS, a framework that fuses external knowledge (high-frequency texture priors from a pretrained 2D super-resolution model) with internal knowledge (cross-view consistent depth and texture features from a multi-scale 3DGS model), coordinated via a mask-guided fusion strategy, to achieve high-fidelity 3DGS super-resolution reconstruction from low-resolution inputs, attaining state-of-the-art performance on both synthetic and real-world scenes.

## Background & Motivation

3D Gaussian Splatting (3DGS) achieves excellent performance in novel view synthesis, yet reconstructing high-resolution (HR) scenes from **low-resolution (LR)** inputs remains a significant challenge, as LR inputs lack fine-grained texture and geometric detail. Acquiring, storing, and transmitting HR multi-view data is often prohibitively costly or infeasible in practice.

Existing 3DGS super-resolution methods (e.g., SRGS, GaussianSR, SuperGaussian) primarily rely on pretrained 2D super-resolution (2DSR) models to upsample LR views, but directly applying 2DSR models introduces two fundamental problems:

**Cross-view inconsistency**: 2D models process each view independently and cannot guarantee multi-view consistency, leading to ambiguities during 3D Gaussian optimization.

**Domain gap**: The distribution mismatch between 2D training data and target 3D scenes causes SR model performance to degrade on unseen 3D scenes.

The authors' key insight is that 2DSR models provide strong HR detail priors but lack cross-view consistency, whereas multi-scale 3DGS models naturally enforce cross-view consistency and adapt to scene geometry but struggle to recover fine-grained texture from LR inputs. The two approaches offer complementary strengths.

## Method

### Overall Architecture

IE-SRGS consists of three key steps:
1. Generate HR images and depth maps as **external knowledge** using a pretrained 2DSR model (SwinIR) and a depth estimation model (Depth Anything V2).
2. Generate cross-view consistent internal reference images and depth maps as **internal knowledge** using a multi-scale 3DGS model (based on Mip-Splatting).
3. Integrate internal and external knowledge via a **mask-guided fusion strategy** to jointly supervise HR 3DGS optimization.

### Key Designs

#### 1. **External Knowledge: HR Detail Recovery**

SwinIR generates super-resolved images $E_{\text{image}}$, and Depth Anything V2 estimates depth maps $E_{\text{depth}}$.

Texture guidance loss (weighted combination of L1 and D-SSIM):

$$\mathcal{L}^E_{\text{tex}} = (1-\lambda)\mathcal{L}_1(E_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(E_{\text{image}}, R_{\text{image}})$$

Geometry guidance loss (relaxed relative depth loss based on Pearson correlation):

$$\mathcal{L}^E_{\text{gem}} = \frac{1}{N}\sum_{i=1}^{N}\left(1 - \frac{\text{Cov}(R_{\text{depth}}^i, E_{\text{depth}}^i)}{\sqrt{\text{Var}(R_{\text{depth}}^i)\text{Var}(E_{\text{depth}}^i)}}\right)$$

Pearson correlation is used instead of direct L1 because monocular depth estimation produces relative depth, which is not scale-aligned with rendered depth.

#### 2. **Internal Knowledge: Ambiguity Correction**

A multi-scale 3DGS model is built upon Mip-Splatting, leveraging its 3D smoothing operation to suppress aliasing and high-frequency noise:

$$\mathbf{g}^{\text{3D}}_{\text{reg}}(\boldsymbol{x}) = (\mathbf{g}^{\text{3D}} \otimes \mathbf{g}_{\text{low}})(\boldsymbol{x})$$

**Multi-View Regularization (MV-Regulation)** is introduced to jointly supervise multiple views, reducing overfitting to single views and enhancing geometric consistency. During training, 3 views are randomly sampled for joint optimization.

HR internal references are generated via **SR-Splatting**: 3D Gaussians are projected to 2D screen space, upsampled, and then rasterized to produce internal scale images $I_{\text{image}}$ and depth maps $I_{\text{depth}}$.

Internal losses similarly comprise texture and geometry components:

$$\mathcal{L}^I_{\text{tex}} = (1-\lambda)\mathcal{L}_1(I_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(I_{\text{image}}, R_{\text{image}})$$

$$\mathcal{L}^I_{\text{gem}} = \mathcal{L}_1(I_{\text{depth}}, R_{\text{depth}})$$

Note that the internal geometry loss uses direct L1 (since internal depth is scale-consistent with rendered depth), whereas the external loss uses Pearson correlation.

#### 3. **Mask-Guided Fusion Strategy**

**Texture fusion**: Inconsistencies and artifacts from 2DSR are typically local. A per-pixel uncertainty map is computed as:

$$D(p) = \frac{|I_{\text{image}}(p) - E_{\text{image}}(p)|}{I_{\text{image}}(p) + \epsilon}$$

A binary mask $M(p)$ is generated via threshold $T$: regions with large discrepancy use the internal reference (ensuring consistency), while regions with small discrepancy use the external reference (preserving HR detail).

$$\mathcal{L}_{\text{tex}} = \mathcal{L}^{I'}_{\text{tex}} + \mathcal{L}^{E'}_{\text{tex}}$$

where $\mathcal{L}^{I'}_{\text{tex}} = \mathcal{L}^I_{\text{tex}} \odot M(p)$ and $\mathcal{L}^{E'}_{\text{tex}} = \mathcal{L}^E_{\text{tex}} \odot (1-M(p))$.

**Geometry fusion**: Geometric structure is relatively coarse and insensitive to local variation; a weighted sum is applied directly:

$$\mathcal{L}_{\text{gem}} = \lambda_i \mathcal{L}^I_{\text{gem}} + \lambda_e \mathcal{L}^E_{\text{gem}}$$

### Loss & Training

- Final loss: $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{tex}} + \mathcal{L}_{\text{gem}}$
- Internal model trained for 30,000 iterations
- $\lambda_i=0.001$, $\lambda_e=0.0001$
- Threshold $T=0.9$ for real scenes, $T=0.6$ for synthetic scenes
- Single NVIDIA RTX 4090

## Key Experimental Results

### Main Results

4× 3D super-resolution on the NeRF Synthetic dataset:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| **IE-SRGS (Ours)** | **30.97** | **0.952** | **0.054** |
| SRGS | 30.83 | 0.948 | 0.056 |
| CROC | 30.71 | 0.945 | 0.067 |
| FastSR-NeRF | 30.47 | 0.944 | 0.075 |
| SwinIR-3DGS | 30.38 | 0.945 | 0.059 |
| 3DGS | 21.77 | 0.867 | 0.104 |
| Mip-Splatting | 24.59 | 0.909 | 0.101 |
| Upper Bound (HR input) | 33.37 | 0.969 | 0.032 |

Real-world datasets (Mip-NeRF360 / Deep Blending / Tanks&Temples):

| Method | Mip360 PSNR↑ | DB PSNR↑ | T&T PSNR↑ |
|------|-------------|----------|----------|
| **IE-SRGS** | **27.15** | **29.63** | **23.52** |
| Sequence Matters | 27.02 | — | 23.43 |
| SRGS | 26.88 | 29.49 | 23.41 |
| Mip-Splatting | 26.43 | 28.93 | 23.04 |
| Upper Bound | 27.23 | 29.73 | 23.51 |

IE-SRGS achieves the best performance across all datasets and approaches the HR upper bound. Compared to the backbone Mip-Splatting, PSNR improves by 25.9% and LPIPS improves by 46.5%.

### Ablation Study

Incremental component addition on Mip-NeRF360:

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|------|-------|-------|--------|------|
| Mip-Splatting (Baseline) | 26.43 | 0.754 | 0.304 | Baseline |
| + External Texture ($E_{\text{image}}$) | 26.69 | 0.762 | 0.300 | Texture detail improved |
| + External Geometry ($E_{\text{depth}}$) | 26.72 | 0.763 | 0.299 | Geometry prior beneficial |
| + Internal Texture ($I_{\text{image}}$) | 27.00 | 0.775 | 0.283 | Cross-view consistency significantly improved |
| + Internal Geometry ($I_{\text{depth}}$) | 27.05 | 0.775 | 0.282 | Internal geometry further improves results |
| + Mask-Guided Fusion | **27.15** | **0.779** | **0.278** | Fusion strategy provides final improvement |

### Key Findings

- Internal knowledge contributes more (+0.33 PSNR) than external knowledge (+0.29 PSNR), demonstrating that cross-view consistency is critical for 3D SR.
- Mask-guided fusion yields an additional +0.10 PSNR, effectively integrating the complementary advantages.
- IE-SRGS inference speed (260 FPS on NeRF Synthetic, 119 FPS on MipNeRF360) surpasses SRGS (191/92 FPS), as joint guidance accelerates convergence.
- Total training time increases by only 7–8 minutes.
- The method is robust to the choice of depth estimator: results with DepthAnythingV2 / V2-Small / DepthPro differ negligibly.
- The framework generalizes across backbones: consistent gains are observed when SwinIR or Mip-Splatting are replaced.
- The method maintains superiority over state-of-the-art under the extreme condition of 8× super-resolution.

## Highlights & Insights

1. **A paradigm of complementary internal-external knowledge**: 2DSR provides detail but is inconsistent; multi-scale 3DGS is consistent but lacks detail. Mask-guided fusion elegantly combines both.
2. **First introduction of geometric depth priors into 3DGS SR**, with a relaxed relative loss to accommodate monocular depth estimators.
3. **Threshold $T$ is highly robust**: performance remains stable for $T \in [0.3, 0.9]$, requiring no careful tuning.
4. **Modular framework design**: both the external SR model and the internal 3DGS backbone are replaceable, making this a general-purpose framework.

## Limitations & Future Work

- Training an additional internal 3DGS model increases total training time by approximately 40%.
- The mask threshold $T$ requires different settings for synthetic and real-world scenes.
- The potential of using video super-resolution (VSR) models as an external backbone remains unexplored (only the SISR model SwinIR is used).
- Systematic evaluation of extreme super-resolution beyond 8× has not been conducted.

## Related Work & Insights

- SRGS is the most direct baseline, relying solely on external knowledge.
- The 3D smoothing operation in Mip-Splatting forms the foundation for internal knowledge.
- The mask-guided fusion concept is generalizable to other 3D tasks requiring multi-source supervision.
- The internal-external fusion paradigm may be applicable to tasks such as 3D editing and 3D inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ — The internal-external knowledge fusion paradigm is clearly articulated and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, extensive ablations, backbone generalization, and robustness analysis are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with thorough motivation.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for 3DGS super-resolution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
