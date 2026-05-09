---
title: >-
  [Paper Note] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising
description: >-
  [ICCV 2025][3D Vision][Point cloud denoising] This paper proposes Noise2Score3D, a fully unsupervised point cloud denoising framework based on Tweedie's formula. It learns the score function directly from noisy data and achieves single-step denoising, while introducing point cloud total variation to estimate unknown noise parameters.
tags:
  - ICCV 2025
  - 3D Vision
  - Point cloud denoising
  - unsupervised learning
  - Tweedie's formula
  - score function
  - total variation
date: 2026-05-08
content_hash: 214c3cd3b8d40e61
---

# Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising

**Conference**: ICCV 2025
**arXiv**: [2503.09283](https://arxiv.org/abs/2503.09283)
**Code**: [GitHub](https://github.com/)
**Area**: 3D Vision
**Keywords**: Point cloud denoising, unsupervised learning, Tweedie's formula, score function, total variation

## TL;DR

This paper proposes Noise2Score3D, a fully unsupervised point cloud denoising framework based on Tweedie's formula. It learns the score function directly from noisy data and achieves single-step denoising, while introducing point cloud total variation to estimate unknown noise parameters.

## Background & Motivation

3D point clouds are frequently corrupted by noise due to sensor errors and environmental factors. Existing deep learning-based denoising methods face two fundamental challenges:

**Supervised methods depend on paired data** — obtaining clean data in real-world scenarios is difficult or impossible.

**Existing unsupervised methods suffer from significant limitations**:
- TotalDn relies on spatial priors and involves slow iterative processes.
- The "score" in ScoreDenoise is based on displacement rather than likelihood, precluding the use of Bayesian statistical tools.
- Poor generalization (retraining required across datasets and noise levels).
- Low inference efficiency (iterative denoising required).

Noise2Score has demonstrated the power of Tweedie's formula in 2D image denoising: by reformulating denoising as a score function estimation problem, it offers a theoretically elegant and flexible framework. This paper extends this idea to 3D point clouds.

## Method

### Theoretical Foundation — Tweedie's Formula

For Gaussian noise: $\mathbf{y} = \mathbf{x} + \sigma\epsilon$

Tweedie's formula yields an explicit expression for the posterior expectation:
$$E_{p(\mathbf{x}|\mathbf{y})}(\mathbf{x}) = \mathbf{y} + \sigma^2 \nabla_\mathbf{y} \log p(\mathbf{y})$$

where $\nabla_\mathbf{y} \log p(\mathbf{y})$ is the score function of the noisy data distribution. **Key advantage**: given the score function and noise parameters, denoising is accomplished in a single step.

### Score Estimation Network

Based on the KPConv architecture:
- 5 encoding stages (14 blocks): 3→256→512→1024→2048 channels
- 4 decoding blocks: 3072→128
- Final fully connected layer outputting a 3D score vector
- 24.3M parameters

The **Amortized Residual DAE (AR-DAE)** loss is employed:
$$\mathcal{L}_{AR-DAE} = \frac{1}{N}\sum_{i=1}^N \|\sigma_t \cdot S(y'_i) + u\|^2$$

where $y'_i = y_i + u \cdot \sigma_t$ is the perturbed version and $u \sim \mathcal{N}(0, I)$.

### Unknown Noise Parameter Estimation — Point Cloud Total Variation

$$TV_{PC} = \sum_{i=1}^N \sum_{j \in \text{neighbors}(i)} w_{i,j} \cdot \sqrt{\|\mathbf{p}_i - \mathbf{p}_j\|^2 + \epsilon^2}$$

The optimal noise parameter is estimated by minimizing the total variation of the denoised result:
$$\sigma^* = \arg\min_\sigma TV_{PC}(\hat{x}(\sigma))$$

### Denoising Pipeline

1. **Training**: Learn the score function $S(y)$ from noisy point clouds.
2. **Denoising**: $\hat{x} = y + \sigma^2 S(y)$ (completed in a single step).
3. **Unknown noise**: Search for the optimal $\sigma$ over a set of candidates.

## Key Experimental Results

### Denoising Performance on ModelNet-40 (CD×$10^4$)

| Method | 10K pts 1% noise CD | 10K pts 3% noise CD | 50K pts 1% noise CD |
|--------|---------------------|---------------------|---------------------|
| Bilateral | 5.865 | 31.034 | 3.711 |
| GLR | 6.592 | 12.890 | 1.860 |
| TotalDn | 8.079 | 29.617 | 5.044 |
| Score-U | 5.514 | 18.239 | 2.696 |
| **Noise2Score3D** | **4.891** | **12.456** | **1.654** |

### Generalization Evaluation

| Training Set | Test Set | CD↓ | P2M↓ |
|--------------|----------|-----|------|
| ModelNet-40 | ModelNet-40 | Best | Best |
| ModelNet-40 | PU-Net | Still superior | Still superior |

### Key Findings

1. Achieves state-of-the-art performance among unsupervised methods, approaching supervised methods in certain settings.
2. **Strong generalization** — a single pretrained model transfers across datasets and noise levels without retraining.
3. **Efficient inference** — single-step denoising when noise parameters are known, outperforming iterative approaches.
4. Point cloud total variation serves as an effective no-reference quality metric for automatic noise parameter selection.

## Highlights & Insights

1. **Integration of Bayesian statistical tools** — Tweedie's formula decouples denoising from score estimation, offering theoretical elegance.
2. **Single-step denoising** — distinguishes this work from all existing unsupervised methods that rely on iterative procedures.
3. **Noise model agnosticism** — the same loss function and pretrained weights apply across different noise models.
4. **Point cloud total variation** — introduces a no-reference quality metric to point cloud denoising for the first time.

## Limitations & Future Work

- Validation is primarily conducted under Gaussian noise; performance on other noise distributions remains to be verified.
- The KPConv-based network has a relatively large parameter count (24.3M).
- When noise parameters are unknown, a search over a candidate range is required.

## Related Work & Insights

- **Traditional methods**: Bilateral filtering, low-rank approximation, graph Laplacian regularization.
- **Supervised methods**: PointCleanNet, ScoreDenoise, IterativePFN.
- **Unsupervised methods**: TotalDn, DMR-U, Score-U.
- **Score matching**: Denoising Score Matching, AR-DAE.

## Rating

- Novelty: ⭐⭐⭐⭐ (Extension of Tweedie's formula to 3D point clouds)
- Technical Depth: ⭐⭐⭐⭐⭐ (Solid theoretical foundation, complete Bayesian framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple noise levels, cross-dataset generalization)
- Value: ⭐⭐⭐⭐⭐ (No clean data required, single-step denoising, automatic noise estimation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

</div>

<!-- RELATED:END -->
