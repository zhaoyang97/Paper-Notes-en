---
title: >-
  [Paper Note] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes the IE-SRGS framework, which reconstructs high-fidelity super-resolution 3DGS from low-resolution inputs. It fuses high-frequency texture priors from external 2D super-resolution models (external knowledge) with cross-view consistent depth and texture features from multi-scale 3DGS models (internal knowledge) through a mask-guided fusion strategy, achieving state-of-the-art performance on both synthetic and rea…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Super-Resolution"
  - "Internal-External Knowledge Fusion"
  - "Mip-Splatting"
  - "Depth Estimation"
date: 2026-05-08
content_hash: 86ca09d930a559e1
---

# IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution

**Conference**: AAAI 2026  
**arXiv**: [2511.22233](https://arxiv.org/abs/2511.22233)  
**Code**: Not released (to be released after review)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Super-Resolution, Internal-External Knowledge Fusion, Mip-Splatting, Depth Estimation

## TL;DR

This paper proposes the IE-SRGS framework, which reconstructs high-fidelity super-resolution 3DGS from low-resolution inputs. It fuses high-frequency texture priors from external 2D super-resolution models (external knowledge) with cross-view consistent depth and texture features from multi-scale 3DGS models (internal knowledge) through a mask-guided fusion strategy, achieving state-of-the-art performance on both synthetic and real-world scenes.

## Background & Motivation

3D Gaussian Splatting (3DGS) performs exceptionally well in novel view synthesis, but reconstructing high-resolution (HR) scenes from **low-resolution (LR)** inputs remains a significant challenge, as LR inputs lack fine texture and geometric details. Moreover, acquiring, storing, and transmitting HR multi-view data is often costly or impractical in real-world scenarios.

Existing 3DGS super-resolution methods (e.g., SRGS, GaussianSR, SuperGaussian) mainly rely on pre-trained 2D super-resolution (2DSR) models to upscale LR views. However, directly applying 2DSR models introduces two fundamental problems:

**Cross-view inconsistency**: The 2D models process each view independently and cannot guarantee multi-view consistency, leading to ambiguities during 3D Gaussian optimization.

**Domain gap**: There is a distribution mismatch between 2D training data and the target 3D scenes, causing the SR models to degrade in performance on unseen 3D scenes.

The authors' key insight is that while 2DSR models provide strong HR detail priors but lack cross-view consistency, multi-scale 3DGS models naturally enforce cross-view consistency and adapt well to scene geometry, but struggle to recover fine-grained textures from LR inputs. The advantages of the two are highly complementary.

## Method

### Overall Architecture

IE-SRGS consists of three key steps:
1. Generate HR images and depth maps as **external knowledge** using a pre-trained 2DSR model (SwinIR) and a depth estimation model (Depth Anything V2).
2. Generate cross-view consistent internal reference images and depth maps as **internal knowledge** using a multi-scale 3DGS model (built on Mip-Splatting).
3. Integrate internal and external knowledge via a **mask-guided fusion strategy** to jointly guide the optimization of the HR 3DGS.

### Key Designs

#### 1. **External Knowledge: HR Detail Recovery**

SwinIR is utilized to generate the super-resolved image $E_{\text{image}}$, and Depth Anything V2 is used to estimate the depth map $E_{\text{depth}}$.

Texture guidance loss (a weighted combination of L1 and D-SSIM):

$$\mathcal{L}^E_{\text{tex}} = (1-\lambda)\mathcal{L}_1(E_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(E_{\text{image}}, R_{\text{image}})$$

Geometric guidance loss (relaxed relative depth loss based on Pearson correlation):

$$\mathcal{L}^E_{\text{gem}} = \frac{1}{N}\sum_{i=1}^{N}\left(1 - \frac{\text{Cov}(R_{\text{depth}}^i, E_{\text{depth}}^i)}{\sqrt{\text{Var}(R_{\text{depth}}^i)\text{Var}(E_{\text{depth}}^i)}}\right)$$

Pearson correlation is preferred over direct L1 loss because the monocular depth estimation outputs relative depth, which is scale-misaligned with the rendered depth.

#### 2. **Internal Knowledge: Ambiguity Correction**

Construct a multi-scale 3DGS model based on Mip-Splatting, utilizing its 3D smoothing operation to suppress aliasing and high-frequency noise:

$$\mathbf{g}^{\text{3D}}_{\text{reg}}(\boldsymbol{x}) = (\mathbf{g}^{\text{3D}} \otimes \mathbf{g}_{\text{low}})(\boldsymbol{x})$$

**Multi-View Regulation (MV-Regulation)** is introduced to jointly supervise multiple views, reducing overfitting to a single view and enhancing geometric consistency. During training, three views are randomly sampled for joint optimization.

The HR internal reference is generated via **SR-Splatting**: 3D Gaussians are projected onto the 2D screen space, upsampled, and then rasterized to obtain the internal-scale image $I_{\text{image}}$ and depth map $I_{\text{depth}}$.

The internal loss likewise consists of both texture and geometric components:

$$\mathcal{L}^I_{\text{tex}} = (1-\lambda)\mathcal{L}_1(I_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(I_{\text{image}}, R_{\text{image}})$$

$$\mathcal{L}^I_{\text{gem}} = \mathcal{L}_1(I_{\text{depth}}, R_{\text{depth}})$$

Note that the internal geometric loss employs a direct L1 loss (as the internal depth is scale-aligned with the rendered depth), whereas the external geometric loss uses Pearson correlation.

#### 3. **Mask-Guided Fusion Strategy**

**Texture Fusion**: Inconsistencies and artifacts from 2DSR are typically local. An uncertainty map is computed for each pixel:

$$D(p) = \frac{|I_{\text{image}}(p) - E_{\text{image}}(p)|}{I_{\text{image}}(p) + \epsilon}$$

A binary mask $M(p)$ is generated via a threshold $T$: regions with large differences use the internal reference (to ensure consistency), while regions with small differences use the external reference (to preserve HR details).

$$\mathcal{L}_{\text{tex}} = \mathcal{L}^{I'}_{\text{tex}} + \mathcal{L}^{E'}_{\text{tex}}$$

where $\mathcal{L}^{I'}_{\text{tex}} = \mathcal{L}^I_{\text{tex}} \odot M(p)$ and $\mathcal{L}^{E'}_{\text{tex}} = \mathcal{L}^E_{\text{tex}} \odot (1-M(p))$.

**Geometric Fusion**: Geometric structures are relatively coarse and insensitive to local variations, so they are directly merged via a weighted sum:

$$\mathcal{L}_{\text{gem}} = \lambda_i \mathcal{L}^I_{\text{gem}} + \lambda_e \mathcal{L}^E_{\text{gem}}$$

### Loss & Training

- Final Loss: $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{tex}} + \mathcal{L}_{\text{gem}}$
- The internal model is trained for 30,000 iterations.
- $\lambda_i=0.001$, $\lambda_e=0.0001$
- Threshold $T=0.9$ for real-world scenes, $T=0.6$ for synthetic scenes.
- Single NVIDIA RTX 4090 GPU.

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
| Upper Bound (HR Input) | 33.37 | 0.969 | 0.032 |

Real-world datasets (Mip-NeRF360 / Deep Blending / Tanks&Temples):

| Method | Mip360 PSNR↑ | DB PSNR↑ | T&T PSNR↑ |
|------|-------------|----------|----------|
| **IE-SRGS** | **27.15** | **29.63** | **23.52** |
| Sequence Matters | 27.02 | — | 23.43 |
| SRGS | 26.88 | 29.49 | 23.41 |
| Mip-Splatting | 26.43 | 28.93 | 23.04 |
| Upper Bound | 27.23 | 29.73 | 23.51 |

IE-SRGS achieves the best performance across all datasets, close to the HR upper bound. Compared to the Mip-Splatting backbone, it shows a 25.9% improvement in PSNR and a 46.5% improvement in LPIPS.

### Ablation Study

Stepwise component addition on Mip-NeRF360:

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| Mip-Splatting (Baseline) | 26.43 | 0.754 | 0.304 | Baseline |
| + External Texture ($E_{\text{image}}$) | 26.69 | 0.762 | 0.300 | Texture detail improvement |
| + External Geometry ($E_{\text{depth}}$) | 26.72 | 0.763 | 0.299 | Helpful geometric priors |
| + Internal Texture ($I_{\text{image}}$) | 27.00 | 0.775 | 0.283 | Significant improvement in cross-view consistency |
| + Internal Geometry ($I_{\text{depth}}$) | 27.05 | 0.775 | 0.282 | Further improvement in internal geometry |
| + Mask-Guided Fusion | **27.15** | **0.779** | **0.278** | Final optimization by fusion strategy |

### Key Findings

- The contribution of internal knowledge (+0.33 PSNR) is larger than that of external knowledge (+0.29 PSNR), showing that cross-view consistency is critical for 3D SR.
- Mask-guided fusion yields an additional 0.10 PSNR gain, effectively integrating both advantages.
- The inference speed of IE-SRGS (260 FPS on NeRF Synthetic, 119 FPS on Mip-NeRF360) outperforms SRGS (191/92 FPS), as joint guidance accelerates convergence.
- Training time increases by only 7-8 minutes.
- Robust to depth estimators: Minimal performance difference across DepthAnythingV2, V2-Small, and DepthPro.
- Generalizable to different backbones: Consistent improvements are achieved when replacing SwinIR or Mip-Splatting.
- Outperforms SOTA even under extreme 8× super-resolution conditions.

## Highlights & Insights

1. **Complementary Internal-External Knowledge Paradigm**: 2DSR provides details without consistency, whereas multi-scale 3DGS is consistent but lacks details. Mask-guided fusion elegantly combines the two.
2. **First introduction of geometric depth priors in 3DGS SR**, using a relaxed relative loss to accommodate monocular depth estimators.
3. **Strong robustness of threshold $T$**: Performance remains stable within the range of $0.3$ to $0.9$, eliminating the need for fine-tuning.
4. **Modular framework design**: Both the external SR model and the internal 3DGS backbone can be replaced, establishing it as a general-purpose framework.

## Limitations & Future Work

- Training the additional internal 3DGS model increases total training time (by approximately 40%).
- The mask threshold $T$ requires different settings for synthetic and real-world scenes.
- The potential of Video Super-Resolution (VSR) as an external backbone has not been explored (the paper only uses SwinIR, a SISR model).
- Performance under extreme super-resolution beyond 8× has not been systematically evaluated.

## Related Work & Insights

- SRGS serves as the most direct baseline, relying solely on external knowledge.
- The 3D smoothing operation of Mip-Splatting serves as the foundation for the internal knowledge.
- The concept of mask-guided fusion can be generalized to other 3D tasks that require the fusion of multi-source supervision.
- Inspiration: This internal-external fusion paradigm can be applied to tasks such as 3D editing and 3D inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ — The paradigm of fusing internal and external knowledge is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive analysis across 4 datasets, various ablations, backbone generalization, and robustness.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with well-justified motivations.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for 3DGS super-resolution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting](../../CVPR2026/3d_vision/splatsure_selective_super-resolution_for_multi-view_consistent_3d_gaussian_splat.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] Urban-GS: A Unified 3D Gaussian Splatting Framework for Compact and High-Fidelity Aerial-to-Street Reconstruction](../../CVPR2026/3d_vision/urban-gs_a_unified_3d_gaussian_splatting_framework_for_compact_and_high-fidelity.md)

</div>

<!-- RELATED:END -->
