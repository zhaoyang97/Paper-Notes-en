---
title: >-
  [Paper Note] Learning a Distance Measure from the Information-Estimation Geometry of Data
description: >-
  [ICLR 2026][Image Generation][information-estimation metric] This paper proposes the Information-Estimation Metric (IEM), a novel distance function induced by the geometry of the data probability density. IEM measures th…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "information-estimation metric"
  - "denoising error"
  - "probability density geometry"
  - "perceptual distance"
  - "diffusion models"
date: 2026-05-08
content_hash: 653bbd610798599a
---

# Learning a Distance Measure from the Information-Estimation Geometry of Data

**Conference**: ICLR 2026
**arXiv**: [2510.02514](https://arxiv.org/abs/2510.02514)  
**Code**: [GitHub](https://github.com/ohayonguy/information-estimation-metric)  
**Area**: Metric Learning / Perceptual Quality Assessment
**Keywords**: information-estimation metric, denoising error, probability density geometry, perceptual distance, diffusion models

## TL;DR
This paper proposes the Information-Estimation Metric (IEM), a novel distance function induced by the geometry of the data probability density. IEM measures the distance between signals by comparing their score vector fields at multiple noise levels. Without any supervised training, IEM achieves perceptual judgment prediction performance competitive with fully supervised methods.

## Background & Motivation
- **Background**: Distance functions are fundamental tools in science and engineering, yet no precise mathematical definition exists for the perceptual distance between natural signals such as images.
- **Limitations of Prior Work**: The best-performing perceptual metrics (LPIPS, DISTS) rely on human-annotated data for training, which is costly and yields limited interpretability.
- **Key Challenge**: Information-theoretic quantities (e.g., mutual information) are insensitive to the global geometry of the density, whereas estimation-theoretic quantities (e.g., denoising error) are directly tied to density geometry.
- **Goal**: The relationship between denoising error and the score function (Tweedie–Miyasawa formula) is foundational to diffusion models. This paper investigates whether this relationship can be exploited to construct a principled perceptual metric.

## Method

### Overall Architecture
IEM is built upon the pointwise I-MMSE identity: the log-probability of a signal can be decomposed into optimal denoiser errors across different SNR levels. Distance is defined by comparing the score vector fields in the neighborhoods of two signals.

### Key Designs

1. **IEM Definition**: The metric compares the score vector field discrepancy of the blurred densities around two points $\boldsymbol{x}_1, \boldsymbol{x}_2$:
$$\text{IEM}(\boldsymbol{x}_1, \boldsymbol{x}_2, \Gamma) = \left(\int_0^\Gamma \mathbb{E}\left[\|\nabla \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x}_1 + \mathbf{w}_\gamma) - \nabla \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x}_2 + \mathbf{w}_\gamma)\|^2\right] d\gamma\right)^{1/2}$$
   where $\gamma$ denotes the signal-to-noise ratio and $\mathbf{w}_\gamma$ is Wiener process noise. IEM can be approximated numerically using a pre-trained denoiser (analogous to a diffusion model).

2. **Metric Properties**: The paper proves that IEM constitutes a valid distance metric satisfying symmetry, non-negativity, positive definiteness, and the triangle inequality. Under a Gaussian distribution, IEM reduces to the Mahalanobis distance: $\text{IEM} = \sqrt{(\boldsymbol{x}_1 - \boldsymbol{x}_2)^\top \Sigma^{-1} (\boldsymbol{x}_1 - \boldsymbol{x}_2)}$.

3. **Local Riemannian Metric**: A second-order expansion yields the Riemannian metric tensor $\boldsymbol{G}(\boldsymbol{x}, \Gamma)$:
$$\boldsymbol{G}(\boldsymbol{x}, \Gamma) = \int_0^\Gamma \gamma^2 \mathbb{E}\left[(\nabla^2 \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x} + \mathbf{w}_\gamma))^2\right] d\gamma$$
   Intuitively, the metric is more sensitive in regions of high log-density curvature and along perturbation directions that produce large probability changes.

4. **Generalized IEM**: A learnable function $f$ is introduced to modulate the weighting of score differences, enabling IEM to adapt to diverse perceptual tasks (e.g., texture similarity vs. distortion assessment).

### Loss & Training
- A Hourglass Diffusion Transformer (HDiT) is trained as the denoiser on ImageNet-1k at 256×256 resolution.
- MSE loss is used with a log-uniform noise level schedule.
- IEM is computed by substituting the trained denoiser into the definition and numerically solving the one-dimensional integral.

## Key Experimental Results

### Main Results (SRCC with Human MOS)

| Method | Supervised | TID2013 | LIVE | CSIQ | TQD (Texture) |
|--------|-----------|---------|------|------|---------------|
| PSNR | No | 0.69 | 0.87 | 0.81 | 0.34 |
| SSIM | No | 0.64 | 0.91 | 0.82 | 0.51 |
| LPIPS | Yes | 0.71 | 0.94 | 0.88 | 0.48 |
| DISTS | Yes | 0.83 | 0.95 | 0.93 | **0.83** |
| TOPIQ | Yes | **0.86** | **0.97** | **0.95** | 0.67 |
| **IEM (unsupervised)** | **No** | **0.83** | **0.96** | **0.94** | 0.51 |
| **IEM_sq (unsupervised)** | **No** | 0.66 | 0.82 | 0.79 | **0.79** |
| **IEM_fω (supervised f)** | **Partial** | **0.84** | **0.96** | **0.94** | **0.77** |

### Ablation Study (Max Differentiation Competition)

| Operation | IEM Result | DISTS Result | Remarks |
|-----------|-----------|-------------|---------|
| Minimize metric (PSNR=10dB) | Artifact-free, high quality | Visible artifacts | IEM is more robust as an optimization target |
| Maximize metric (PSNR=10dB) | Unstructured noise | Patterned artifacts | IEM is most sensitive to perturbations off the data support |

### Key Findings
- Unsupervised IEM is competitive with the best supervised methods on TID2013/LIVE/CSIQ (SRCC 0.83–0.96).
- $\text{IEM}_{sq}$ excels on texture similarity (TQD); the choice of $f$ controls sensitivity to global vs. local distortions.
- Learning $f_\omega$ yields consistently strong results across all benchmarks.
- IEM minimization produces artifact-free images, demonstrating its suitability as a standalone optimization objective.

## Highlights & Insights
- The deep connection between information theory and estimation theory provides a principled foundation for constructing perceptual metrics.
- The reduction to the Mahalanobis distance in the Gaussian case offers an elegant theoretical anchor.
- Isometric curves can be disconnected (in Gaussian mixture settings), reflecting the metric's adaptation to the global geometry of the density.
- This work represents a breakthrough toward deriving perceptual metrics from unlabeled data.

## Limitations & Future Work
- **Computational cost**: Running the denoiser across multiple SNR levels and integrating numerically is substantially slower than LPIPS.
- The selection of the hyperparameter $\Gamma$ currently lacks a systematic principled criterion.
- Validation is limited to 256×256 images.
- Applications as an optimization objective (e.g., image restoration, compression) remain to be explored.

## Related Work & Insights
- The method builds upon the I-MMSE identity (Guo et al., 2005) and the Tweedie–Miyasawa formula.
- IEM shares its theoretical foundation with diffusion models (score function = denoising error) but serves a fundamentally different purpose.
- The framework offers a new perspective on unsupervised feature learning and metric learning.
- The approach is extensible to other continuous signal domains such as audio.

## Technical Details
- The denoiser employs the Hourglass Diffusion Transformer (HDiT), which scales linearly with image resolution.
- Training is conducted on ImageNet-1k at 256×256 with a log-uniform noise level schedule.
- Under a Gaussian prior, IEM equals the Mahalanobis distance, admitting a closed-form solution.
- The Laplacian prior example demonstrates IEM's differentiated local sensitivity along probability density ridges.
- $\Gamma=1/4$ performs best on standard IQA benchmarks; $\Gamma=10^6$ performs best on texture datasets.
- IEM minimization as an optimization objective produces artifact-free results, outperforming supervised methods such as DISTS.
- A mismatched IEM variant can be used to evaluate differences between generative models.
- The code is fully open-sourced, including all details for denoiser training, IEM computation, and experiment reproduction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Deriving a perceptual metric from information-estimation theory represents an outstanding theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive benchmark comparisons and visualizations, though large-scale downstream applications are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and elegant; figures are intuitive and compelling.
- Value: ⭐⭐⭐⭐⭐ — Provides a fundamentally new theoretical framework for perceptual metric learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](../../NeurIPS2025/image_generation/mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICLR 2026\] Monocular Normal Estimation via Shading Sequence Estimation](monocular_normal_estimation_via_shading_sequence_estimation.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](../../NeurIPS2025/image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)

</div>

<!-- RELATED:END -->
