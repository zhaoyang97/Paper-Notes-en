---
title: >-
  [Paper Note] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] To address overfitting in 3DGS under sparse-view settings, this paper proposes DropAnSH-GS, which replaces independent random Dropout with Anchor-based Dropout—dropping entire clusters of spatially correlated Gaussians around selected anchors to disrupt local redundancy compensation—while introducing Spherical Harmonics (SH) Dropout to suppress high-order SH overfitting and enable lossless post-training compression.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - sparse-view
  - dropout regularization
  - spherical harmonics
  - novel view synthesis
date: 2026-05-08
content_hash: cca3a70288b1e002
---

# DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2602.20933](https://arxiv.org/abs/2602.20933)
**Code**: [Project Page](https://sk-fun.fun/DropAnSH-GS)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, sparse-view, dropout regularization, spherical harmonics, novel view synthesis

## TL;DR

To address overfitting in 3DGS under sparse-view settings, this paper proposes DropAnSH-GS, which replaces independent random Dropout with Anchor-based Dropout—dropping entire clusters of spatially correlated Gaussians around selected anchors to disrupt local redundancy compensation—while introducing Spherical Harmonics (SH) Dropout to suppress high-order SH overfitting and enable lossless post-training compression.

## Background & Motivation

3D Gaussian Splatting (3DGS) represents 3D scenes via a large collection of explicit Gaussian primitives, achieving an excellent balance between rendering speed and visual quality under dense-view inputs. However, under sparse-view settings (e.g., only 3 training views), severe overfitting leads to artifacts, blurriness, and geometric distortion.

**Limitations of Prior Work**: Inspired by Dropout in deep learning, DropGaussian and DropoutGS randomly zero out the opacity of individual Gaussians during training. This paper identifies two critical issues:

**Problem 1 — Neighbor Compensation Effect**: 3DGS renders scenes through large numbers of overlapping Gaussians that collaboratively contribute to each pixel. Gaussians in local regions share highly similar opacity and color attributes (validated via Moran's I, which shows spatial autocorrelation is inversely proportional to distance). When a single Gaussian is dropped, its rendering contribution is easily compensated by neighboring Gaussians, resulting in negligible pixel color change $\Delta C$, weak gradient signals during backpropagation, and severely diminished regularization.

**Problem 2 — High-order SH Overfitting**: Existing Dropout methods only manipulate opacity and neglect spherical harmonic coefficients. Experiments (Figure 3) show that increasing SH degree improves performance under dense views, but under sparse views, high-order SH degrades performance and inflates model size, constituting another source of overfitting.

**Core Idea**: For Dropout to be truly effective, entire spatially correlated clusters of Gaussians (rather than individual primitives) must be dropped, creating large-scale "information voids" that force the model to learn more robust global representations.

## Method

### Overall Architecture

DropAnSH-GS integrates two regularization strategies into the 3DGS training pipeline:
1. **Anchor-based Dropout**: Selects anchor Gaussians and drops them together with their spatial neighborhoods.
2. **SH Dropout**: Randomly drops high-order spherical harmonic coefficients.

Both strategies are seamlessly embedded into the standard 3DGS training pipeline without modifying the loss function; they are applied only during the forward pass.

### Key Designs

1. **Anchor-based Dropout**: Executed in three steps:

    - **Anchor Selection**: Randomly sample a subset of anchors $\mathcal{A}$ from all $N$ Gaussians $\mathcal{G}$ at sampling rate $p_a$.
    - **Neighborhood Construction**: For each anchor, find $k$ nearest neighbors in Euclidean space.
    - **Structured Dropping**: Collect all anchors and their neighborhoods into a drop set $\mathcal{D}$, and zero out their opacities via a binary mask $m_i$:

    $\hat{\alpha}_i = \alpha_i \cdot m_i, \quad m_i = \begin{cases} 0 & G_i \in \mathcal{D} \\ 1 & \text{otherwise} \end{cases}$

   **Design Motivation**: Dropping entire Gaussian clusters creates large-scale information voids that actively disrupt spatial coherence and prevent neighbor compensation. The optimization is forced to exploit long-range contextual information to reconstruct dropped regions, encouraging the learning of more robust global scene representations. The kNN search is implemented in CUDA, adding less than 2.8% training overhead.

2. **SH Dropout**: The color of each Gaussian $\mathbf{c} = [\mathbf{c}^{(0)}, \mathbf{c}^{(1)}, \dots, \mathbf{c}^{(L)}]$ is represented by multi-order SH coefficients. For a subset of Gaussians selected with probability $p_{sh}$, SH coefficients above order $l_{\max}$ are dropped:

    $\tilde{\mathbf{c}} = [\mathbf{c}^{(0)}, \dots, \mathbf{c}^{(l_{\max})}, \mathbf{0}, \dots, \mathbf{0}]$

   During training, $l_{\max}$ increases progressively (iteration 2000 = order 0, 4000 = order 1, 6000 = order 2), forming a coarse-to-fine appearance learning curriculum. Dual benefits: (1) suppresses color overfitting; (2) prioritizes appearance information in low-order SH coefficients, enabling post-training truncation of high-order SH for model compression without retraining.

3. **Hyperparameter Design**: $p_a$ grows linearly from 0 to 0.02 (no dropping in early training to preserve geometric initialization); $k=10$ (neighborhood size); $p_{sh}=0.2$ (SH Dropout probability).

### Loss & Training

Standard 3DGS loss without modification:
$$\mathcal{L} = \mathcal{L}_{\text{L1}}(\hat{C}, C_{gt}) + \lambda \mathcal{L}_{\text{SSIM}}(\hat{C}, C_{gt})$$

Notably, DropAnSH-GS is a pure regularization strategy that imposes implicit constraints solely by modifying opacity and SH during the forward pass, introducing no additional explicit loss terms.

## Key Experimental Results

### Main Results

| Dataset (# views) | Metric | DropAnSH-GS | DropGaussian | 3DGS | Gain vs. DropGaussian |
|--------|------|-------------|-------------|------|------|
| LLFF (3-view) | PSNR↑ | **20.68** | 20.33 | 19.17 | +0.35 |
| LLFF (3-view) | SSIM↑ | **0.724** | 0.709 | 0.646 | +0.015 |
| LLFF (3-view) | LPIPS↓ | **0.194** | 0.201 | 0.268 | −0.007 |
| MipNeRF-360 (12-view) | PSNR↑ | **19.95** | 19.66 | 18.58 | +0.29 |
| Blender (8-view) | PSNR↑ | **25.50** | 25.17 | 22.13 | +0.33 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Notes |
|------|------|------|-------|------|
| No Dropout (3DGS) | 19.17 | 0.646 | 0.268 | Baseline |
| Anchor Dropout only | 20.47 | 0.713 | 0.200 | +1.30 PSNR from Anchor Dropout |
| SH Dropout only | 19.59 | 0.641 | 0.247 | SH Dropout alone is also effective |
| Anchor + SH Dropout | **20.68** | **0.724** | **0.194** | Complementary gains |

### Key Findings

- **Model Compression**: Retaining only order-0 SH (SH0) already surpasses vanilla 3DGS, at only 25% of the model size. On MipNeRF-360: SH0 = 33.8 MB (PSNR 19.71) vs. 3DGS = 143.4 MB (PSNR 18.58).
- **Strong Compatibility**: Plugging DropAnSH-GS into other 3DGS variants consistently yields improvements — FSGS +0.29 PSNR, CoR-GS +0.38, DNGaussian +0.59, Scaffold-GS +1.22.
- **Training Efficiency**: Less than 2.8% additional training time over 3DGS (LLFF: 760 s vs. 742 s).
- Dropping SH by order outperforms randomly dropping individual SH coefficients (Blender: 25.50 vs. 25.12 PSNR), as it preserves the hierarchical structure of SH.

## Highlights & Insights

- **Rigorous Problem Analysis**: The neighbor compensation effect is substantiated by quantitatively measuring spatial autocorrelation among Gaussians via Moran's I, which is more convincing than intuitive arguments alone.
- **Minimal yet Effective Method**: No loss function modifications, no additional networks — only a change in the random masking strategy during training.
- **Post-training Compression as a Byproduct**: SH Dropout enables high-order SH truncation without retraining, offering flexible trade-offs between performance and model size.
- The research is problem-driven, starting from the question "why does Dropout have weak effect in 3DGS," a methodology worth emulating.

## Limitations & Future Work

- kNN search may become a bottleneck when the number of Gaussians is extremely large, despite CUDA acceleration.
- The anchor sampling rate $p_a$ and neighborhood size $k$ require tuning; sensitivity analysis shows performance is relatively sensitive to $p_a$ (PSNR drops sharply to 19.97 at $p_a = 0.04$).
- The method is a general regularization strategy that does not exploit any 3D priors or pretrained models, which represents a direction for further improvement.
- Validation is limited to sparse-view scenarios; performance under other degradation conditions (e.g., noisy camera poses) remains unexplored.

## Related Work & Insights

- DropGaussian and DropoutGS pioneered the use of Dropout in 3DGS but overlooked the spatial redundancy characteristics of Gaussian primitives.
- The approach shares conceptual similarity with Spatial Dropout / DropBlock in deep learning: dropping spatially contiguous regions is more effective than dropping independent units.
- CoR-GS regularizes via mutual constraint between two 3DGS models, whereas the proposed method achieves regularization within a single model, offering greater simplicity.
- The idea of SH truncation for model compression can be generalized to other scene representations that employ spherical harmonics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of revisiting Dropout through the lens of Gaussian spatial redundancy is novel, and the method is elegantly simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multiple view counts, comprehensive ablations, compatibility validation, and hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The pilot study analysis is rigorous, and figures are clear.
- **Value**: ⭐⭐⭐⭐ — Simple, practical, and plug-and-play; offers direct value to the 3DGS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)
- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
