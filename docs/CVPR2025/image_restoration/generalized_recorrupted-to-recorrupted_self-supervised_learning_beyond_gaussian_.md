---
title: >-
  [Paper Note] Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
description: >-
  [CVPR 2025][Image Restoration][Self-Supervised Denoising] This paper proposes Generalized R2R (GR2R), generalizing the original self-supervised denoising framework Recorrupted-to-Recorrupted (R2R) from Gaussian noise to natural exponential family (NEF) distributions—including Poisson, Gamma, and Binomial noise. It proves that the GR2R loss is an unbiased estimator of supervised loss, with SURE being its special case, achieving performance close to supervised learning in appli…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Self-Supervised Denoising"
  - "Non-Gaussian Noise"
  - "Natural Exponential Family"
  - "Unbiased Estimation"
  - "Recorrupted-to-Recorrupted"
date: 2026-05-08
content_hash: cfd47eef16820db3
---

# Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise

**Conference**: CVPR 2025  
**arXiv**: [2412.04648](https://arxiv.org/abs/2412.04648)  
**Code**: [https://github.com/bemc22/GeneralizedR2R](https://github.com/bemc22/GeneralizedR2R)  
**Area**: Image Restoration  
**Keywords**: Self-Supervised Denoising, Non-Gaussian Noise, Natural Exponential Family, Unbiased Estimation, Recorrupted-to-Recorrupted

## TL;DR
This paper proposes Generalized R2R (GR2R), generalizing the original self-supervised denoising framework Recorrupted-to-Recorrupted (R2R) from Gaussian noise to natural exponential family (NEF) distributions—including Poisson, Gamma, and Binomial noise. It proves that the GR2R loss is an unbiased estimator of supervised loss, with SURE being its special case, achieving performance close to supervised learning in applications like low-light imaging and SAR.

## Background & Motivation
1. **Background**: Deep learning-based image denoising relies on clean-noisy paired data (supervised learning), but clean data is scarce in areas like medical imaging and scientific imaging. Self-supervised methods have emerged: Noise2Noise requires independent noisy pairs (hard to obtain), Blind Spot Networks do not look at the center pixel (suboptimal), and SURE is only applicable to Gaussian or partial exponential family noise and requires calculating divergence.
2. **Limitations of Prior Work**: R2R generates training pairs through synthetic recorruption and is an unbiased estimator of the supervised loss, but it has only been validated on Gaussian noise. In practical applications, non-Gaussian noise is frequently encountered: Poisson noise (photon counting/low-light imaging), Gamma noise (Synthetic Aperture Radar - SAR), log-Rayleigh noise, etc.
3. **Key Challenge**: The theoretical proof of R2R relies on properties of the Gaussian distribution (independence of $\epsilon - \omega/\tau$ and $\epsilon + \tau\omega$), which cannot be directly generalized to other noise distributions.
4. **Goal**: Generalize R2R to the entire natural exponential family (NEF) noise distributions, providing theoretical guarantees and practical algorithms.
5. **Key Insight**: Leverage the decomposability of NEF, $\boldsymbol{y} = (1-\alpha)\boldsymbol{y}_1 + \alpha\boldsymbol{y}_2$, to construct independent noisy pairs from a single noisy observation.
6. **Core Idea**: Use the additivity/divisibility of natural exponential family distributions to generate independent noisy pairs from a single noisy image, making the R2R framework applicable to non-Gaussian noise such as Poisson and Gamma.

## Method

### Overall Architecture
Given a noisy image $\boldsymbol{y} \sim p(\boldsymbol{y}|\boldsymbol{x})$ (belonging to NEF), GR2R generates two independent noisy images $\boldsymbol{y}_1$ and $\boldsymbol{y}_2$ using distribution-specific recorruption strategies. Here, $\boldsymbol{y}_1$ serves as the input and $\boldsymbol{y}_2$ acts as the target to train the denoising network $f$ with MSE or negative log-likelihood (NLL) loss. At inference, Monte Carlo averaging is used to mitigate the additional noise introduced by recorruption.

### Key Designs

1. **Generalization of R2R to Non-Gaussian Additive Noise (Proposition 1)**:
    - Function: Generalize the original corruption scheme of R2R to arbitrary additive noise.
    - Mechanism: For $\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{\epsilon}$, the original R2R corruption $\boldsymbol{y}_1 = \boldsymbol{y} + \tau\boldsymbol{\omega}$ and $\boldsymbol{y}_2 = \boldsymbol{y} - \boldsymbol{\omega}/\tau$ is still utilized. However, it is proved that as long as $\boldsymbol{\omega}$ matches the low-order moments of $\boldsymbol{\epsilon}$, the estimator is approximately unbiased: a linear estimator only needs to match the second-order moment $\mathbb{E}\omega_i^2 = \mathbb{E}\epsilon_i^2$, while a quadratic estimator additionally requires matching the third-order moment $\mathbb{E}\omega_i^3 = \frac{1}{\tau}\mathbb{E}\epsilon_i^3$.
    - Design Motivation: For asymmetric noise like log-Rayleigh, original R2R using Gaussian $\omega$ (symmetric distribution, third-order moment is 0) introduces bias. Matching the third-order moment improves PSNR on log-Rayleigh noise from 25.32 to 29.47 dB.

2. **Generalization of R2R to the Natural Exponential Family (Theorem 1)**:
    - Function: Provide accurate unbiased estimation for non-additive noise such as Poisson, Gamma, and Binomial.
    - Mechanism: Design distribution-specific corruption schemes utilizing the decomposability of NEF: for Poisson noise $\boldsymbol{z} \sim \mathcal{P}(\boldsymbol{x}/\gamma)$, draw binomial samples $\boldsymbol{\omega} \sim \text{Bin}(\boldsymbol{z}, \alpha)$ to obtain $\boldsymbol{y}_1 = (\boldsymbol{y}-\gamma\boldsymbol{\omega})/(1-\alpha)$; for Gamma noise $\boldsymbol{y} \sim \mathcal{G}(\ell, \ell/\boldsymbol{x})$, split using a Beta distribution $\boldsymbol{\omega} \sim \text{Beta}(\ell\alpha, \ell(1-\alpha))$ in $\boldsymbol{y}_1 = \boldsymbol{y} \circ (1-\omega)/(1-\alpha)$. The formulation $\boldsymbol{y}_2 = \frac{1}{\alpha}\boldsymbol{y} - \frac{1-\alpha}{\alpha}\boldsymbol{y}_1$ applies universally. It is proved that $\mathbb{E}_{\boldsymbol{y}|\boldsymbol{x}} \mathcal{L}_{GR2R} = \mathbb{E}_{\boldsymbol{y}_1|\boldsymbol{x}} \|f(\boldsymbol{y}_1)-\boldsymbol{x}\|_2^2$.
    - Design Motivation: Exploiting the decomposability of NEF distributions is the key mathematical insight—a NEF random variable can be decomposed into the sum of two independent random variables of the same family.

3. **Negative Log-Likelihood (NLL) Loss**:
    - Function: Improve denoising performance by incorporating noise distribution prior information.
    - Mechanism: While MSE loss does not exploit the noise distribution prior, NLL loss $\mathcal{L}_{GR2R-NLL}^\alpha = \mathbb{E}\{\phi(f(\boldsymbol{y}_1)) - \boldsymbol{y}_2^\top \eta(f(\boldsymbol{y}_1))\}$ directly maximizes the likelihood of the recorrupted observations. For Poisson, this is $-\gamma\boldsymbol{y}_2^\top\log f(\boldsymbol{y}_1) + \mathbf{1}^\top f(\boldsymbol{y}_1)$; for Gamma, it is $\log f(\boldsymbol{y}_1) + \boldsymbol{y}_2/f(\boldsymbol{y}_1)$. The proof shows that its optimal solution remains the conditional mean MMSE estimator.
    - Design Motivation: NLL loss is more robust than MSE at low signal-to-noise ratios (high noise) because it accounts for the actual shape of the noise distribution.

### Loss & Training
- MSE Loss: $\mathcal{L}_{GR2R-MSE}^\alpha = \mathbb{E}_{\boldsymbol{y}_1,\boldsymbol{y}_2|\boldsymbol{y}} \|f(\boldsymbol{y}_1) - \boldsymbol{y}_2\|_2^2$
- NLL Loss: Distribution-specific negative log-likelihood
- Hyperparameter $\alpha$: Controls the allocation of signal-to-noise ratios between $\boldsymbol{y}_1$ and $\boldsymbol{y}_2$, with the optimal value typically in $[0.1, 0.2]$
- Monte Carlo Averaging during Inference: $\hat{\boldsymbol{x}} \approx \frac{1}{J}\sum_{j=1}^J \hat{f}(\boldsymbol{y}_1^{(j)})$ with $J=5\sim15$
- Network Architecture: DRUnet (Poisson) and DnCNN (Gamma/Gaussian); GR2R is architecture-agnostic

## Key Experimental Results

### Main Results (Poisson Denoising, PSNR/SSIM on DIV2K)

| Noise Level (γ) | PURE | Neigh2Neigh | GR2R-NLL | GR2R-MSE | Supervised |
|------------|------|-------------|----------|----------|---------|
| 0.01 | 32.69/0.919 | 33.37/0.929 | 33.90/0.935 | 33.92/0.935 | 33.96/0.933 |
| 0.1 | 24.37/0.631 | 28.27/0.827 | 28.30/0.827 | 28.35/0.827 | 28.39/0.827 |
| 0.5 | 22.98/0.623 | 24.90/0.651 | **25.07/0.716** | 24.69/0.698 | 25.32/0.727 |
| 1.0 | 17.94/0.469 | 23.56/0.653 | **23.69/0.658** | 23.49/0.646 | 23.85/0.668 |

### Ablation Study (log-Rayleigh Additive Noise, PSNR/SSIM)

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| R2R (only 2nd-order moment matched) | 25.32±0.79 | 0.576±0.08 | Original R2R with Gaussian ω |
| **GR2R (3rd-order moment matched)** | **29.47±1.51** | **0.813±0.04** | +4.15dB |
| Supervised | 29.93±1.50 | 0.831±0.04 | Upper bound |

### Key Findings
- Matching the third-order moment yields a giant boost of 4.15 dB (25.32 $\to$ 29.47), demonstrating the critical importance of high-order moment matching for asymmetric noise.
- GR2R nearly matches supervised learning performance in Poisson denoising (with a margin of only 0.06-0.16 dB), substantially outperforming PURE.
- NLL loss outperforms MSE under high noise levels (low counts $\gamma=0.5, 1.0$) as it leverages the shape information of the Poisson distribution.
- MSE loss performs on par with NLL under low noise levels (high counts $\gamma=0.01$), where Poisson noise approximates Gaussian noise.
- The optimal $\alpha$ lies in $[0.1, 0.2]$, balancing the input signal-to-noise ratio against the target noise.
- GR2R is architecture-agnostic and can be directly integrated with any denoising network.

## Highlights & Insights
- **Theoretical Elegance**: It is proved that GR2R-MSE degenerates to SURE as $\alpha \to 0$, establishing a mathematical bridge between R2R and SURE. SURE can be viewed as a special case of GR2R, whereas GR2R avoids SURE's drawback of requiring the divergence term calculation.
- **Ingenious Exploitation of NEF Decomposability**: Generating independent pairs via binomial splitting for Poisson variables and Beta splitting for Gamma variables serves as a stellar template of utilizing classical probability theory to solve deep learning problems.
- **Introduction of NLL Loss**: Breaks the convention of using only MSE in self-supervised denoising by customizing loss functions for diverse noise distributions.
- The method is highly transferable to other inverse problems with NEF noise: medical MRI reconstruction (Rician noise), photoacoustic imaging, astronomical image processing, etc.

## Limitations & Future Work
- The current Monte Carlo inference ($J$ forward passes) increases test-time computation; single-inference alternatives could be explored.
- For mixed noise (e.g., Poisson-Gaussian), further derivation of corruption schemes is necessary.
- The optimal selection of $\alpha$ currently relies on empirical ablation and lacks theoretical guidance.
- For signal-dependent noise (e.g., heteroscedastic Gaussian), where noise variance changes with the signal, extra treatment is required.
- The experimental dataset (DIV2K with 900 training images) is relatively small, showing a lack of validation on large-scale medical datasets.

## Related Work & Insights
- **vs R2R**: The original R2R only handles Gaussian noise; GR2R extends this to the entire NEF, achieving a genuine theoretical generalization.
- **vs SURE/PURE/GSURE**: SURE requires calculating divergence (numerical approximation), and PURE and GSURE both suffer from distinct limitations; GR2R entirely bypasses divergence computation and mathematically proves SURE to be its special case.
- **vs Noise2Void/Neigh2Neigh**: These methods implement self-supervision by constraining network structures (blind spots), yet their optimal solution is not MMSE; the solution of GR2R precisely is the MMSE optimal estimator.
- **vs Noise2Score**: Requires approximating the score function of the noise distribution, relying on Tweedie's formula; GR2R constructs training pairs directly from noisy samples, making it simpler and more straightforward.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contribution, generalizing R2R from Gaussian to NEF and establishing connections with SURE.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Gaussian, Poisson, Gamma, and log-Rayleigh noise, though on limited datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clear tabular summaries, and modularized theorem-proposition-proof structures.
- Value: ⭐⭐⭐⭐ Provides a unified theoretical framework for self-supervised denoising under non-Gaussian noise; the code is open-source and highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[CVPR 2026\] Convexity-Aware Noise Calibration: A Self-Supervised Framework for Noise-Level-Unknown Image Denoising](../../CVPR2026/image_restoration/convexity-aware_noise_calibration_a_self-supervised_framework_for_noise-level-un.md)
- [\[NeurIPS 2025\] The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](../../NeurIPS2025/image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)

</div>

<!-- RELATED:END -->
