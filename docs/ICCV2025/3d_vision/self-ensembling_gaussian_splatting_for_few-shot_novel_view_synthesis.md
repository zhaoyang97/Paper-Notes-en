---
title: >-
  [Paper Note] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis
description: >-
  [3D Vision] SE-GS dynamically generates diverse 3DGS models during training via an uncertainty-aware perturbation strategy…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 590dcb6bb2c538c0
---

# Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis

## Paper Info
- **Conference**: ICCV 2025
- **arXiv**: [2411.00144](https://arxiv.org/abs/2411.00144)
- **Code**: [Project Page](https://chen-zhao.com/SE-GS)
- **Area**: 3D Vision
- **Keywords**: 3D Gaussian Splatting, Few-Shot Novel View Synthesis, Self-Ensembling Learning, Uncertainty-Aware Perturbation

## TL;DR
SE-GS dynamically generates diverse 3DGS models during training via an uncertainty-aware perturbation strategy, and leverages a self-ensembling mechanism to allow the Σ-model to aggregate information from perturbed models, effectively mitigating overfitting under sparse-view settings and achieving state-of-the-art few-shot novel view synthesis performance across multiple datasets.

## Background & Motivation

3D Gaussian Splatting (3DGS) excels at novel view synthesis but is prone to overfitting when trained with **sparse views**:

**Severe Overfitting**: Experiments show (Fig. 2) that training-set performance improves continuously with iterations, while test-set performance begins to degrade after approximately 2,000 iterations; overfitting is more pronounced in the 3-view setting.

**Limitations of Prior Work**:
   - Depth-prior-based methods (DNGaussian, FSGS) introduce noisy depth estimates and may actually hurt performance as the number of training views increases.
   - Multi-model regularization (CoR-GS) incurs high computational cost and lacks sufficient diversity among models.

**Potential of Ensemble Learning**: Ensemble learning has been proven effective at alleviating overfitting in detection and segmentation tasks, yet efficient ensembling within 3DGS remains unexplored.

## Method

### Overall Architecture

SE-GS jointly trains two models:
- **Δ-model**: Trained normally on available training images and dynamically perturbed to produce diverse model variants.
- **Σ-model**: Achieves self-ensembling by minimizing discrepancy with the perturbed models; used at inference time.

### Key Design 1: Uncertainty-Aware Perturbation

Naïve global random perturbation causes the model to deviate too far, leading to instability. SE-GS applies targeted perturbation via the following steps:

**1. Create Pseudo-Views**: Generate $M$ pseudo-views via spherical linear interpolation (SLERP) between training views:

$$\hat{\mathbf{R}} = \text{SLERP}(\mathbf{R}_1, \mathbf{R}_2, \beta)$$

**2. Compute Uncertainty Maps**: Store pseudo-view renderings from different training steps in a buffer and compute pixel-level uncertainty:

$$\mathbf{U} = \sqrt{\frac{1}{S}\sum_{i=1}^S (\mathbf{I}_i - \bar{\mathbf{I}})^2}$$

followed by local smoothing with $k=5$ to obtain $\hat{\mathbf{U}}$.

**3. Selective Perturbation**: Perturb only Gaussians whose projected regions overlap with high-uncertainty pixels:

$$\hat{G}_\Delta^t = G_\Delta^t + \delta_t \cdot h(G_\Delta^t, \hat{\mathcal{U}}^t)$$

where the indicator function $h$ checks whether the maximum uncertainty in a Gaussian's projected region exceeds threshold $\tau$. A 6D continuous rotation representation is used for perturbation to ensure continuity.

### Key Design 2: Self-Ensembling Regularization

The Σ-model is trained normally on training views while being regularized to maintain consistency with the perturbed models:

$$\mathcal{L}_r = (1-\lambda)\|\mathbf{I}_\Sigma^t - \mathbf{I}_\Delta^t\|_1 + \lambda\mathcal{L}_{\text{D-SSIM}}(\mathbf{I}_\Sigma^t, \mathbf{I}_\Delta^t)$$

where $\lambda=0.2$. The regularization is applied on pseudo-views in a self-supervised manner, requiring no additional ground-truth supervision.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \gamma\mathcal{L}_r$$

where $\gamma=1$ and $\mathcal{L}_{\text{RGB}}$ is the photometric loss on training views.

### Key Advantages
- Perturbed models are derived from the Δ-model rather than trained from scratch, making the computational overhead negligible.
- Uncertainty is computed in 2D rendering space, naturally accommodating changes in the number of Gaussians during training.
- Regularization is applied on pseudo-views, independent of external information such as depth.

## Key Experimental Results

### Main Results: LLFF Dataset

| Method | 3-view PSNR | 6-view PSNR | 9-view PSNR |
|------|-----------|-----------|-----------|
| 3DGS | 19.22 | 23.80 | 25.44 |
| DNGaussian | 19.12 | 22.01 | 22.62 |
| FSGS | 20.43 | 24.09 | 25.31 |
| CoR-GS | 20.45 | 24.49 | 26.06 |
| **SE-GS** | **20.79** | **24.78** | **26.36** |

SE-GS achieves the best PSNR across 3/6/9-view settings, and also leads comprehensively on SSIM and LPIPS.

### Ablation Study: Perturbation Strategy Comparison

| Perturbation | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| No perturbation (cross-model only) | baseline | - | - |
| Global random perturbation | degraded | - | - |
| **Uncertainty-aware perturbation** | **highest** | **highest** | **lowest** |

- Naïve global perturbation degrades performance due to excessive deviation.
- Uncertainty-aware selective perturbation significantly outperforms other strategies.

### Key Findings
- Buffer size $S=5$ and pseudo-view count $M=10$ are the optimal configurations.
- SE-GS is more efficient and effective than explicit ensembling of $k$ independently trained models.
- SE-GS maintains substantial gains over vanilla 3DGS even as the number of training views increases (e.g., 9 views).
- Unlike depth-prior-based methods, SE-GS does not suffer from performance degradation as more views are added.

## Highlights & Insights

1. **First introduction of self-ensembling into 3DGS**: Cleverly exploits uncertainty signals emerging from training dynamics.
2. **Computationally efficient**: Incurs negligible additional training cost compared to explicit multi-model ensembling.
3. **Plug-and-play**: Orthogonal to depth-prior methods and can be combined with them.
4. **Self-supervised regularization**: Requires no external ground-truth depth or generated novel-view images.

## Limitations & Future Work
- Still requires a certain number of initial SfM points for 3DGS initialization.
- Pseudo-views must lie within the interpolation range of training views, limiting effectiveness on extrapolated scenes.
- Uncertainty estimation may be unreliable in extremely sparse settings (e.g., 1–2 images).

## Related Work & Insights
- **3DGS**: Explicit point-based representation supporting real-time rendering.
- **DNGaussian / FSGS**: Leverage monocular depth priors to address sparse-view challenges.
- **CoR-GS**: Trains multiple 3DGS models for cross-regularization.
- **Ensemble Learning**: Improves robustness by aggregating multi-model predictions; related to temporal ensembling and consistency regularization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of uncertainty-aware perturbation and self-ensembling is novel.
- **Practicality**: ⭐⭐⭐⭐ — Low overhead, no external data required, consistent improvements across datasets.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensively validated on LLFF/DTU/Mip-NeRF360/MVImgNet with extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and rigorous methodological derivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
