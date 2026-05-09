---
title: >-
  [Paper Note] Blind2Sound: Self-Supervised Image Denoising without Residual Noise
description: >-
  [ICCV2025][Image Restoration][Self-supervised denoising] This paper proposes the Blind2Sound framework, which perceives noise levels and achieves personalized denoising via an adaptive re-visible loss, complemented by a Cramer Gaussian loss that improves noise parameter estimation accuracy. The framework eliminates residual noise in self-supervised blind denoising and outperforms all contemporary self-supervised methods and even some supervised baselines.
tags:
  - ICCV2025
  - Image Restoration
  - Self-supervised denoising
  - Poisson-Gaussian noise
  - blind denoising
  - residual noise elimination
  - noise awareness
date: 2026-05-08
content_hash: 4b5ff770dbaffb4f
---

# Blind2Sound: Self-Supervised Image Denoising without Residual Noise

**Conference**: ICCV2025
**arXiv**: [2303.05183](https://arxiv.org/abs/2303.05183)
**Authors**: Zejin Wang, Jiazheng Liu, Hao Zhai, Hua Han (Institute of Automation, Chinese Academy of Sciences)
**Code**: Provided in the paper's supplementary material
**Area**: Image Restoration
**Keywords**: Self-supervised denoising, Poisson-Gaussian noise, blind denoising, residual noise elimination, noise awareness

## TL;DR

This paper proposes the Blind2Sound framework, which perceives noise levels and achieves personalized denoising via an adaptive re-visible loss, complemented by a Cramer Gaussian loss that improves noise parameter estimation accuracy. The framework eliminates residual noise in self-supervised blind denoising and outperforms all contemporary self-supervised methods and even some supervised baselines.

## Background & Motivation

### Problem Definition

Noise in practical imaging sensors typically follows a Poisson-Gaussian mixture model: $\mathbf{y} = \alpha P + N$, where $P \sim \text{Poisson}(\mathbf{x}/\alpha)$ is signal-dependent Poisson noise and $N \sim \mathcal{N}(0, \sigma^2)$ is signal-independent Gaussian noise. Self-supervised blind denoising requires learning to denoise from a single noisy image without relying on paired clean data.

### Limitations of Prior Work

**Pseudo-supervised pair methods** (e.g., Noise2Noise variants): Constructing training pairs from a single noisy image doubly corrupts the signal, degrading performance.

**Blind-spot methods** (e.g., N2V, BSN series): Input masking causes information loss, producing severe artifacts.

**Blind2Unblind**: Achieves lossless denoising via a re-visible transition, but the MSE objective cannot perceive noise levels, leading to pixel-level greedy fitting that results in **pronounced residual noise**.

**FBI-Denoiser**: The Gaussian loss lacks fine-grained constraints, yielding insufficient noise estimation accuracy; moreover, post-processing steps amplify prior denoising errors.

### Core Motivation

MSE as an objective function cannot adaptively adjust denoising strength based on dynamic noise levels. A loss function that is compatible with the lossless framework while performing personalized denoising according to perceived noise levels is needed to fundamentally eliminate residual noise.

## Method

### Overall Architecture

Blind2Sound consists of two modules:

1. **Denoising network** $f_\omega(\cdot)$: An improved U-Net that outputs denoising results (mean $\mu_m, \mu_v$ and covariance $\Sigma_m, \Sigma_v$) for both the masked branch and the visible branch.
2. **Noise estimator** $g_\theta(\cdot)$: Predicts Poisson-Gaussian noise parameters $(\alpha, \sigma_1, \sigma_2)$.

Both modules are jointly optimized during training; **the noise estimator and masked branch are removed at inference**, and the denoiser directly generates results from the original noisy image.

### Adaptive Re-Visible Loss

The core idea is to model the masked branch and the visible branch as two independent Gaussian generative processes:

- **Masked branch**: $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{z}_1 | \mu_m, \Sigma_m)$, generating latent clean images from the masked noisy volume $\Omega\mathbf{y}$.
- **Visible branch**: $\mathbf{z}_2 \sim \mathcal{N}(\mathbf{z}_2 | \mu_v, \Sigma_v)$, generated from the original noisy image $\mathbf{y}$ with gradients disabled (excluded from backpropagation).

The noise model is explicitly incorporated via marginal likelihood:

$$p(\mathbf{y}) = \int p(\mathbf{y}|\mathbf{x}) p(\mathbf{x}|\mathbf{y}, \Omega\mathbf{y}) d\mathbf{x}$$

Minimizing the negative log-likelihood yields the adaptive re-visible loss:

$$\mathcal{L}_{arv} = \frac{1}{2}[(\mathbf{y} - \mu_y)^T \Sigma_y^{-1} (\mathbf{y} - \mu_y)] + \frac{1}{2}\log|\Sigma_y| + \text{const}$$

where $\mu_y = \frac{\mu_m + \lambda \mu_v}{1+\lambda}$ and $\lambda$ is a visibility factor that grows progressively from 3 to 11.

**Key designs**:

- The two branches are modeled as i.i.d., decoupling their correlation and preventing the masked results from suppressing visible denoising.
- Gradient analysis on the intermediate medium $\mu_m$ indicates that gradients through $\text{diag}(\alpha \mu_y)$ must be disabled to stabilize training.
- At convergence, the optimal estimate is $\tilde{\mathbf{x}} = \frac{\mu_m + \lambda \mu_v}{1+\lambda}$, satisfying $\mu_m \leq \tilde{\mathbf{x}} \leq \mu_v$.
- No post-processing MAP as in Laine19 is required, since the loss itself already incorporates information from $\mathbf{y}$.

### Cramer Gaussian Loss

To address the limitation of the original Gaussian loss—which estimates noise only at the global image level and ignores local noise knowledge—fine-grained constraints are introduced:

**Single-channel images** (e.g., grayscale): Sub-patch constraints are employed by cropping four overlapping sub-patches (3/4 the size of the original image) from the four corners. After GAT transformation, the noise variance of both sub-patches and the full image should approximate unit variance:

$$\mathcal{L}_{est} = \sum_{s=1}^{4} \|\eta(G_{g_\theta}(\mathbf{y}_s)) - 1\|_2^2 + \|\eta(G_{g_\theta}(\mathbf{y})) - 1\|_2^2$$

**Multi-channel images** (e.g., sRGB): A cross-channel noise level consistency constraint is introduced to prevent estimation errors across channels from canceling each other out:

$$\mathcal{L}_{est} = \sum_{j \neq k}^{c} \|\eta(G_{g_\theta}(\mathbf{y}_j)) - 1\|_2^2 + \|\eta(G_{g_\theta}(\mathbf{y}_j)) - \eta(G_{g_\theta}(\mathbf{y}_k))\|_2^2$$

The Cramer Gaussian loss serves only as a regularization term (weight 0.01), as the actual noise level of the denoised image may differ from that of the original input.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{arv} + 0.01 \cdot \mathcal{L}_{est}$$

## Key Experimental Results

### Noise Estimation Accuracy

On BSD68 (grayscale) and CBSD68 (sRGB), the Cramer Gaussian loss compared to FBI-D:
- On grayscale images, it eliminates the severe Gaussian parameter estimation errors of FBI-D.
- On sRGB images, the cross-channel constraint produces predictions closer to ground truth.

### Synthetic Grayscale Denoising (PSNR/SSIM)

| Noise | Method | BSD68 | Set12 | Urban100 |
|-------|--------|-------|-------|----------|
| PG1 | Blind2Unblind | 30.61/0.869 | 31.45/0.880 | 30.70/0.900 |
| PG1 | **Blind2Sound** | **30.83/0.875** | **31.68/0.886** | **31.14/0.908** |
| PG3 | Blind2Unblind | 27.02/0.757 | 27.65/0.796 | 26.54/0.805 |
| PG3 | **Blind2Sound** | **27.17/0.766** | **27.96/0.805** | **26.96/0.819** |

- Maximum gain of **0.44 dB** and minimum gain of **0.15 dB** over Blind2Unblind.
- **Surpasses supervised baselines** N2C and N2N on Set12 and Urban100, with a maximum gain of 0.4 dB.

### Synthetic sRGB Denoising

| Noise | Method | KODAK | SET14 | BSD300 |
|-------|--------|-------|-------|--------|
| PG1 | Blind2Unblind | 33.88/0.915 | 32.47/0.886 | 32.53/0.913 |
| PG1 | **Blind2Sound** | **34.23/0.920** | **32.75/0.896** | **33.00/0.921** |

### Real-World Denoising

| Method | SIDD Benchmark (RAW) | SIDD Validation (RAW) | FMD Confocal |
|--------|---------------------|----------------------|--------------|
| N2C (supervised) | 50.61/0.991 | 51.19/0.991 | 38.40/0.966 |
| Blind2Unblind | 50.79/0.991 | 51.36/0.992 | 38.44/0.964 |
| **Blind2Sound** | **50.92/0.991** | **51.50/0.992** | **38.46/0.965** |

- Outperforms all self-supervised methods and the supervised baseline on SIDD RAW.
- Nearly **0.3 dB** gain over FBI-D in the RAW space.

### Ablation Study

| Experiment | Finding |
|------------|---------|
| Granularity size | Fine-grained sub-patch constraints improve coarse-grained estimation accuracy, but excessively small patches provide insufficient noise context. |
| Cramer loss weight | 0.01 is optimal; both 0 (no regularization) and 100 (over-regularization) are inferior. |
| Training scheme | Joint training > fixed pre-trained estimator ≈ fixed ground-truth noise (larger gap at low noise). |
| Noise model | Enhanced model $\mathcal{M}_E$ shows a clear advantage at high noise levels. |
| Branch independence | IID (independent) greatly outperforms non-IID, validating the importance of the decoupled design. |
| Visibility factor | $\lambda_f = 11$ is optimal; the relationship is non-monotonically increasing. |

## Highlights & Insights

1. **Unifying noise awareness with lossless denoising**: Embedding noise level estimation into the re-visible loss enables adaptive adjustment of denoising strength—this is the key to eliminating residual noise.
2. **Zero inference overhead**: The noise estimator and masked branch are used only during training; at inference, the pipeline is identical to Blind2Unblind with no additional computation.
3. **Bayesian re-formulation**: The MSE loss of Blind2Unblind is elevated to the negative log-likelihood of a mixed Gaussian marginal likelihood, yielding a theoretically sounder objective.
4. **Cross-channel/sub-patch fine-grained constraints**: The Cramer Gaussian loss narrows the solution space via multi-scale noise consistency, resolving the inaccurate estimation issue of FBI-D.
5. **Surpassing supervised methods on grayscale**: Outperforming N2C/N2N on Set12 and Urban100 demonstrates the potential of self-supervised methods under noise-aware training.

## Limitations & Future Work

1. **Limited gains in sRGB space**: Denoising across channels is more challenging at high noise levels, yielding smaller gains than in grayscale; cross-channel modeling can be further strengthened.
2. **Noise model assumption**: The method relies on the Poisson-Gaussian model and may require extensions for more complex real-world noise (e.g., spatially correlated noise).
3. **Visibility factor tuning**: The initial and final values of $\lambda$ must be set manually; an adaptive schedule may be more effective.
4. **Single network architecture**: Only an improved U-Net is employed; incorporating a stronger backbone (e.g., Transformer) may yield further improvements.
5. **No video denoising**: The framework targets static images and does not exploit temporal information.

## Related Work & Insights

- **Blind2Unblind (CVPR 2022)**: The direct predecessor; Blind2Sound extends its re-visible framework with noise awareness.
- **FBI-Denoiser (CVPR 2021)**: The first to introduce a Gaussian loss for noise estimation within a self-supervised framework, but with insufficient accuracy.
- **Noise2Void / Noise2Self**: Foundational works in blind-spot denoising, but suffer from severe information loss.
- **NBR2NBR (CVPR 2021)**: Constructs training pairs via sub-sampling, but the neighboring-pixel approximation causes over-smoothing.
- **GAT + BM3D**: Traditional methods, now comprehensively outperformed by data-driven approaches.
- **Insight**: Explicitly incorporating noise model knowledge into self-supervised loss design is an effective path to performance improvement; future work could explore applying this idea to other degradation types (e.g., blur, compression artifacts).

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)
- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)

</div>

<!-- RELATED:END -->
