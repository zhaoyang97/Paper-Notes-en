---
title: >-
  [Paper Note] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models
description: >-
  [NeurIPS 2025][Image Generation][Gaussianity Regularization] This paper proposes a unified Gaussianity regularization framework that combines moment matching in the spatial domain with power spectrum matching in the freq…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Gaussianity Regularization"
  - "Normalizing Flow"
  - "Power Spectrum"
  - "Moment Matching"
  - "Reward Alignment"
date: 2026-05-08
content_hash: 3c77ef18d6d953e4
---

# Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.07027](https://arxiv.org/abs/2509.07027)
**Code**: None
**Area**: Image Generation / Regularization / Generative Models
**Keywords**: Gaussianity Regularization, Normalizing Flow, Power Spectrum, Moment Matching, Reward Alignment

## TL;DR

This paper proposes a unified Gaussianity regularization framework that combines moment matching in the spatial domain with power spectrum matching in the frequency domain. It subsumes existing regularizers (KL divergence, kurtosis, norm) as special cases, and achieves the equivalent effect of PRNO's $\mathcal{O}(D^2)$ approach at $\mathcal{O}(D\log D)$ complexity, significantly outperforming all baselines on reward alignment tasks for text-to-image models.

## Background & Motivation

In generative models, the standard Gaussian distribution serves as a core prior over the latent space. Measuring and maintaining the Gaussianity of latent samples is critical for the following reasons:

**Reward hacking**: Directly optimizing latent samples to maximize a reward function causes samples to deviate from the Gaussian prior, leading to image quality degradation (e.g., cartoonish artifacts).

**Limitations of existing regularization methods**:
   - KL divergence, norm regularization, and similar approaches only constrain marginal statistics without accounting for inter-component dependencies.
   - PRNO addresses dependencies via covariance matrix matching, but requires $\mathcal{O}(D^2)$ time and memory complexity.

**Insufficiency of single-domain regularization**: Matching in either the spatial or frequency domain alone is insufficient — latent vectors matched only in the spatial domain may still retain spectral structure that leads to generation degradation (illustrated intuitively in Figure 1).

## Method

### Overall Architecture

The high-dimensional latent vector $\mathbf{x} \in \mathbb{R}^D$ (where each component $x_i \sim \mathcal{N}(0,1)$ is i.i.d.) is treated as a collection of one-dimensional standard Gaussian variables. Regularization losses are defined separately in the spatial and frequency domains and used jointly.

### Key Designs

1. **Spatial Domain: Moment-Based Regularization Loss**:

    - **Theoretical basis** (Theorem 1): The standard Gaussian distribution is uniquely characterized by all of its moments.
    - Odd-order moments = 0; even-order moments = $(2k)!/(2^k k!)$.
    - **$n$-th order moment loss**: $\mathcal{L}_n = \left| \left|\frac{1}{D}\sum_{k=1}^D x_k^n\right|^{1/n} - \mu_n^{1/n} \right|$
    - Computational complexity $\mathcal{O}(D)$, efficiently applicable to high-dimensional latent spaces.
    - **Unifying existing methods**:
        - KL regularization $\approx$ $\mathcal{L}_1 + \mathcal{L}_2$
        - Kurtosis regularization $\approx$ $\mathcal{L}_4$
        - Norm regularization $\approx$ $\mathcal{L}_2$ (asymptotically equivalent as $D \to \infty$)

2. **Frequency Domain: Power-Spectrum-Based Regularization Loss**:

    - **Theoretical basis** (Lemma 1): The normalized DFT coefficient magnitudes of an i.i.d. standard Gaussian vector follow a $\chi_2/\sqrt{2}$ distribution.
    - Directly maximizing likelihood (NLL loss) pushes all spectral components toward the mode $1/\sqrt{2}$, suppressing natural variance.
    - **Batch-averaging strategy**: Frequency indices are grouped into batches (size $|B|=16$); the mean within each batch is compared against the target mean.
    - **Power spectrum loss**: $\mathcal{L}_{\text{power}} = \frac{1}{|\mathcal{B}|}\sum_{B\in\mathcal{B}}\left|\frac{1}{|B|}\sum_{k\in B}\frac{|\hat{x}_k|}{\sqrt{D}} - \mu_{\text{power}}\right|$
    - Target mean $\mu_{\text{power}} = 0.875$ (the expected value of $\chi_2/\sqrt{2}$).
    - Equivalent in objective to PRNO (covariance matrix matching), but reduces complexity from $\mathcal{O}(D^2)$ to $\mathcal{O}(D\log D)$.

3. **Random Permutation Invariance**:

    - Spatial-domain moment losses are naturally permutation-invariant.
    - Frequency-domain losses are order-sensitive; the latent vector is therefore randomly shuffled prior to computation.

### Loss & Training

The final Gaussianity regularization loss is:
$$\mathcal{L}_{\mathcal{N}(0,I)} = \sum_{n \in \mathcal{K}} \mathcal{L}_n + \lambda_{\text{power}} \mathcal{L}_{\text{power}}$$

- $\mathcal{K} = \{1, 2\}$ (first- and second-order moments)
- $\lambda_{\text{power}} = 25.0$
- Optimization: Nesterov momentum 0.9, gradient clipping 0.01, 500 iterations
- Regularization gradients are normalized to be on the same scale as reward gradients

## Key Experimental Results

### Regularization Method Comparison

| Method | Time Complexity | Memory Complexity | Relation to Proposed Losses |
|--------|----------------|-------------------|-----------------------------|
| KL | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | $\mathcal{L}_1, \mathcal{L}_2$ |
| Kurtosis | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | $\mathcal{L}_4$ |
| Norm (ReNO) | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | $\mathcal{L}_2$ |
| PRNO | $\mathcal{O}(Dk)$ | $\mathcal{O}(Dk)$ | $\mathcal{L}_1, \mathcal{L}_{\text{power}}$ |
| **Ours** | $\mathcal{O}(D\log D)$ | $\mathcal{O}(D)$ | — |

### Toy Experiment: Checkerboard Initialization Recovery

| Method | Iterations | Time | Spatial Matching | Spectral Matching | Image Quality |
|--------|-----------|------|-----------------|-------------------|---------------|
| KL | 10K | 11.2s | ✓ | ✗ | Residual checkerboard artifacts |
| Kurtosis | 10K | 16.1s | ✓ | ✗ | Residual checkerboard artifacts |
| Norm (ReNO) | 10K | 10.4s | ✓ | ✗ | Residual checkerboard artifacts |
| PRNO | 100 | 14.1s | ✓ | Partial | Texture residuals remain |
| **Ours** | **100** | **0.26s** | **✓** | **✓** | **Clean, high quality** |

### Aesthetic Image Generation

- Base model: FLUX (single-step text-to-image model)
- Target reward: aesthetic score
- Held-out rewards: ImageReward, HPSv2
- **The proposed method achieves consistently best performance across all metrics**: highest target reward and highest held-out rewards.
- Without regularization, reward hacking occurs: target reward increases while held-out rewards continuously decline.
- Spatial-only regularizers (KL, Kurtosis, ReNO) plateau quickly.
- PRNO improves results but underperforms the proposed method.

### Text-Aligned Image Generation

- Target reward: PickScore
- Evaluation prompts: 60 prompts from T2I-CompBench++ (6 categories, 10 each)
- **The proposed method achieves higher rewards with fewer iterations.**
- Spatial-only methods rapidly hit a performance ceiling.
- The proposed method promotes stable gradient flow by keeping latent vectors close to the Gaussian prior.

### Key Findings

1. **Joint spatial and frequency domain regularization is necessary**: Figure 1 clearly demonstrates the differences among four combinations.
2. **Approximately 50× speedup**: Compared to PRNO, the proposed method completes 100 iterations in 0.26s vs. 14.1s for PRNO.
3. **Effective prevention of reward hacking**: All held-out metrics improve continuously rather than declining.
4. **Accelerated convergence**: Higher reward scores are achieved within the same number of iterations.
5. **Low loss values do not guarantee perfect Gaussianity**: This is an acknowledged limitation.

## Highlights & Insights

- **Strong theoretical unification**: KL, kurtosis, norm regularization, and other methods are unified under the moment matching framework.
- **Equivalence and efficiency of frequency-domain regularization**: Equivalent in objective to PRNO but with substantially reduced complexity.
- **High pedagogical value of Figures 1 and 3**: They intuitively illustrate how spatial and frequency domains each control distinct aspects of Gaussianity.
- **Lemma 1** provides a rigorous theoretical foundation for spectral regularization.
- **The batch-averaging strategy** elegantly avoids the over-compression of spectral variance that arises with NLL loss.

## Limitations & Future Work

- Loss values alone cannot reliably indicate how closely the latent vector approximates a true standard Gaussian distribution.
- The method inherits biases and artifacts from the pre-trained generative model.
- Validation is limited to the FLUX model; generalization to other architectures such as Stable Diffusion has not been tested.
- Only moments of orders $\mathcal{K} = \{1, 2\}$ are used; the effect of higher-order moments remains underexplored.
- A systematic sensitivity analysis with respect to different values of $\lambda_{\text{power}}$ is absent.

## Related Work & Insights

- ReNO demonstrates the effectiveness of noise optimization in single-step generative models.
- The covariance matching approach of PRNO is conceptually sound but computationally inefficient.
- The proposed method can be naturally extended to other domains that employ Gaussian priors, such as motion generation and music generation.
- The framework provides an efficient and theoretically grounded toolkit for regularizing high-dimensional latent spaces.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (Unified theoretical framework + frequency-domain equivalence discovery + efficiency improvement)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Toy experiments + two application tasks, but limited to a single base model)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Rigorous mathematical derivations, high-quality figures, strong intuitive clarity)
- **Value**: ⭐⭐⭐⭐ (Broadly applicable guidance for latent space optimization in generative models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)

</div>

<!-- RELATED:END -->
