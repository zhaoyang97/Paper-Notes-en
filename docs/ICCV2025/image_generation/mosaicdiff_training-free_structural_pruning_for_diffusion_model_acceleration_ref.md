---
title: >-
  [Paper Note] MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics
description: >-
  [ICCV 2025][Image Generation][Structural Pruning] This paper proposes MosaicDiff, a training-free structural pruning method for diffusion models that dynamically partitions the inference trajectory into three stages based on pretraining learning-speed dynamics and applies stage-specific subnetworks with varying sparsity, achieving significant acceleration on DiT and SDXL without sacrificing generation quality.
tags:
  - ICCV 2025
  - Image Generation
  - Structural Pruning
  - Training-free Acceleration
  - Pretraining Dynamics
  - SNR-aware
  - Second-order Pruning
date: 2026-05-08
content_hash: 70fb58baa340be0a
---

# MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics

**Conference**: ICCV 2025
**arXiv**: [2510.11962](https://arxiv.org/abs/2510.11962)
**Code**: [https://github.com/bwguo105/MosaicDiff](https://github.com/bwguo105/MosaicDiff)
**Area**: Diffusion Models / Model Acceleration
**Keywords**: Structural Pruning, Training-free Acceleration, Pretraining Dynamics, SNR-aware, Second-order Pruning

## TL;DR

This paper proposes MosaicDiff, a training-free structural pruning method for diffusion models that dynamically partitions the inference trajectory into three stages based on pretraining learning-speed dynamics and applies stage-specific subnetworks with varying sparsity, achieving significant acceleration on DiT and SDXL without sacrificing generation quality.

## Background & Motivation

**Background**: Diffusion models offer strong generative capabilities but incur substantial computational cost. The community has primarily addressed inference acceleration through reduced sampling steps (DDIM, DPM-Solver), knowledge distillation, structural pruning, quantization, and feature caching. However, existing methods universally overlook the inherent variation in learning speed during diffusion model pretraining.

**Core Observation**: Diffusion model pretraining exhibits a "slow–fast–slow" three-phase learning pattern—early stages involve slow learning (dominated by high-noise inputs), the middle stage shows a sharp increase in learning speed (rapid capture of coarse-grained features), and the late stage slows again (refinement of fine-grained details). This critical insight has been entirely ignored by prior work.

**Limitations of Prior Work**:
- Diff-Pruning applies a single pruned model across all timesteps, ignoring step-level importance differences.
- EcoDiff employs learnable masks but similarly lacks step-adaptive behavior.
- Existing methods suffer severe performance degradation at high sparsity ratios.

## Method

### Overall Architecture

MosaicDiff follows a "Divide → Prune → Conquer" three-stage pipeline:

1. **Divide**: The inference trajectory is partitioned into three stages via quantitative analysis of pretraining dynamics.
2. **Prune**: Second-order structural pruning is applied to each stage using SNR-aware calibration data.
3. **Conquer**: Subnetworks with different sparsity levels are assembled to complete the final sampling.

### Key Design 1: Stage Partitioning and Importance Scoring

Learning dynamics are characterized by monitoring the MSE between the intermediate latent representation $\hat{x}_t$ and the final output $\hat{x}_0$:

$$\text{MSE}(t) = \frac{1}{d}\|\hat{x}_t - \hat{x}_0\|_2^2$$

The authors derive closed-form expressions for the expected MSE and its gradient (Theorem 1):

$$\mathbb{E}[\text{MSE}(t)] = \frac{1}{d}\left[(1-\sqrt{\bar{\alpha}_t})^2\|\hat{x}_0\|_2^2 + (1-\bar{\alpha}_t)\|\mathbf{I}\|_2^2\right]$$

$$\mathbb{E}[\text{Grad}(t)] = \frac{1}{d}\left[(\delta_t + 2(\sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_t}))\|\hat{x}_0\|_2^2 - \delta_t\|\mathbf{I}\|_2^2\right]$$

The final importance score integrates SNR information:

$$score(t) = \mathbb{E}[\text{Grad}(t)] + \lambda \ln \text{SNR}(t)$$

Sampling steps are automatically partitioned into three stages via a threshold $threshold = M \cdot \max_t(score(t))$.

### Key Design 2: SNR-aware Calibration Dataset

Stage-specific calibration data are constructed by encoding images from standard datasets (e.g., ImageNet-1K) into latent representations, randomly sampling $t$ within the corresponding stage's timestep range, and adding noise according to $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$, ensuring the calibration data accurately reflects the SNR characteristics of each stage. For models employing CFG, unconditional (null-label) calibration samples are additionally provided.

### Key Design 3: Second-order Structural Pruning

Given a linear layer weight $\mathbf{W} \in \mathbb{R}^{m \times n}$ and calibration input $\mathbf{X} \in \mathbb{R}^{b \times n}$, the objective is:

$$\arg\min_{\widehat{\mathbf{W}}} \|\mathbf{X}\widehat{\mathbf{W}}^\top - \mathbf{X}\mathbf{W}^\top\|_2^2$$

Column-level saliency scores are obtained by computing the Hessian matrix $\mathbf{H} = \mathbf{X}^\top\mathbf{X}$ and extending the OBS formulation:

$$\arg\min_{\mathbf{M}} \sum_{i=0}^{m-1} \mathbf{W}_{i,\mathbf{M}} \cdot (\mathbf{H}_{\mathbf{M},\mathbf{M}}^{-1})^{-1} \cdot \mathbf{W}_{i,\mathbf{M}}^\top$$

After pruning, a compensatory weight update $\delta = -\mathbf{W}_{:,\mathbf{M}} \cdot (\mathbf{H}_{\mathbf{M},\mathbf{M}}^{-1})^{-1} \cdot \mathbf{H}_{\mathbf{M},:}^{-1}$ is applied to further reduce reconstruction error.

### Sparsity Allocation Strategy

Per-stage sparsity is inversely proportional to the average importance score: $s_i \propto 1 - \overline{score}_i$. The fast-learning middle stage retains more parameters, while the slow-learning early and late stages permit more aggressive pruning.

## Key Experimental Results

### Main Results

| Method | Steps | MACs(T) | Speedup | IS↑ | FID↓ | Precision↑ |
|--------|-------|---------|---------|-----|------|------------|
| Vanilla DiT-XL/2 | 50 | 5.72 | 1.00× | 238.6 | 2.26 | 80.16 |
| Diff-Pruning-0.3 | 50 | 4.10 | 1.29× | 4.68 | 180.76 | 7.24 |
| Learning-to-Cache | 50 | 4.36 | 1.27× | 244.1 | 2.27 | 80.94 |
| **MosaicDiff-0.33** | **50** | **3.92** | **1.32×** | **267.8** | **2.24** | **82.01** |
| Vanilla DiT-XL/2 | 20 | 2.29 | 1.00× | 223.5 | 3.48 | 78.76 |
| **MosaicDiff-0.30** | **20** | **1.64** | **1.28×** | **266.7** | **3.20** | **81.13** |

### Pruning Comparison on SDXL

| Method | Sparsity | FID↓ | CLIP↑ | SSIM↑ |
|--------|----------|------|-------|-------|
| Diff-Pruning | 10% | 108.96 | 0.22 | 0.31 |
| EcoDiff | 10% | 33.75 | 0.31 | 0.53 |
| **MosaicDiff** | **10%** | **23.18** | **0.32** | **0.67** |
| Diff-Pruning | 20% | 404.87 | 0.05 | 0.26 |
| **MosaicDiff** | **20%** | **23.79** | **0.32** | **0.64** |

### Key Findings

- MosaicDiff achieves FID 2.24 under 50-step DDIM with 33% sparsity, outperforming the uncompressed baseline (2.26) while reducing MACs by 31%.
- The advantage is more pronounced at high sparsity: Diff-Pruning at 30% sparsity yields FID 180.76, whereas MosaicDiff attains 2.24.
- The method generalizes effectively to SDXL: at 10% sparsity, FID improves to 23.18 (vs. EcoDiff's 33.75).
- Closed-form MSE/gradient curves closely match empirical observations, validating the theoretical analysis.

## Highlights & Insights

1. **First alignment of pretraining learning dynamics with post-training acceleration**: The perspective is novel and theoretically grounded; the closed-form solution eliminates any dependency on actual pretraining runs.
2. **Training-free and fine-tuning-free**: Second-order pruning leveraging Hessian information requires no retraining whatsoever.
3. **Strong generality**: Applicable to both Transformer (DiT) and U-Net (SDXL) architectures.
4. **SNR-aware calibration** ensures precision in stage-specific pruning.

## Limitations & Future Work

- The thresholds $M$ and $\lambda$ require hyperparameter tuning (though the authors provide recommended values).
- Whether the three-stage partition is optimal for all noise schedules remains unclear.
- Hessian computation incurs non-trivial calibration data and computational overhead.

## Related Work & Insights

- **Sampling Acceleration**: DDIM, DPM-Solver, Consistency Models
- **Structural Pruning**: Diff-Pruning, EcoDiff
- **Caching Methods**: DeepCache, Learning-to-Cache
- **Training-time Compression**: DiP-GO, Knowledge Distillation

## Rating

- Novelty: ⭐⭐⭐⭐ — The pretraining dynamics perspective is original.
- Technical Depth: ⭐⭐⭐⭐ — Closed-form theoretical analysis is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across DiT and SDXL.
- Practical Value: ⭐⭐⭐⭐⭐ — Training-free and plug-and-play.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](../../CVPR2026/image_generation/just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)

<!-- RELATED:END -->
