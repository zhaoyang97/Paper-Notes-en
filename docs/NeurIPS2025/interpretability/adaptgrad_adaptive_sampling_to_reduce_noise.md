---
title: >-
  [Paper Note] AdaptGrad: Adaptive Sampling to Reduce Noise
description: >-
  [NeurIPS 2025][Interpretability][Gradient smoothing] AdaptGrad analyzes the theoretical origin of noise in SmoothGrad—out-of-boundary (OOB) sampling behavior—and proposes adaptively adjusting the Gaussian sampling varian…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Gradient smoothing"
  - "saliency maps"
  - "adaptive sampling"
  - "SmoothGrad"
date: 2026-05-08
content_hash: 8d388cd1fcd95c1b
---

# AdaptGrad: Adaptive Sampling to Reduce Noise

**Conference**: NeurIPS 2025  
**arXiv**: [2410.07711](https://arxiv.org/abs/2410.07711)  
**Code**: [GitHub](https://github.com/AiShare-WHU/AdaptGrad)  
**Area**: Interpretability  
**Keywords**: Gradient smoothing, saliency maps, interpretability, adaptive sampling, SmoothGrad

## TL;DR
AdaptGrad analyzes the theoretical origin of noise in SmoothGrad—out-of-boundary (OOB) sampling behavior—and proposes adaptively adjusting the Gaussian sampling variance for each input dimension to bound the additional noise. The method nearly eliminates gradient noise while revealing richer fine-grained features, requires minimal implementation effort, and is composable with arbitrary gradient-based explanation methods.

## Background & Motivation

**Background** Gradients are a key source of information for interpreting deep models, but raw gradients contain substantial noise. SmoothGrad reduces noise by adding Gaussian perturbations and averaging the resulting gradients, and remains the most widely adopted gradient smoothing technique.

**Limitations of Prior Work** The critical hyperparameter $\sigma$ in SmoothGrad is typically set empirically as $\sigma = \alpha(x_{max} - x_{min})$ with $\alpha=0.2$. This setting causes a large proportion of sampled points to fall outside the valid input range $\Omega=[x_{min}, x_{max}]$, producing meaningless samples and introducing additional noise.

**Key Challenge** SmoothGrad is designed for denoising, yet its hyperparameter configuration introduces new noise. The fundamental cause is the mismatch between the sampling distribution domain ($\mathbb{R}^D$) and the data domain ($\Omega$).

**Goal** To theoretically analyze and minimize the additional noise introduced by SmoothGrad, achieving nearly noise-free gradient smoothing.

**Key Insight** SmoothGrad is reinterpreted as a Monte Carlo approximation of a convolution operation. An analytic expression for the additional noise is derived, which then guides the design of adaptive sampling variances.

**Core Idea** Compute the sampling variance for each input pixel adaptively based on its distance to the boundary of the data range, ensuring that the OOB sampling probability does not exceed a preset upper bound $1-c$.

## Method

### Overall Architecture
AdaptGrad retains the basic framework of SmoothGrad (sampling gradients in the neighborhood and averaging), but replaces the shared fixed variance $\sigma^2$ across all dimensions with dimension-wise adaptive variances $\sigma_i^2$, such that the OOB sampling probability for each dimension is bounded by $1-c$.

### Key Designs

1. **Theoretical Analysis of SmoothGrad as Convolution**:
    - Function: Reveals the theoretical origin of SmoothGrad noise.
    - Mechanism: SmoothGrad is essentially a convolution of the gradient function $G(\mathbf{x})$ with a Gaussian kernel $p(\cdot)$: $G_{sg}(\mathbf{x}) \simeq (G * p)(\mathbf{x})$. The convolution domain should be $\Omega$, but the sampling distribution is defined over $\mathbb{R}^D$. The OOB sampling probability is quantified as $A^i_{sg} = 1 - \frac{1}{2}\text{erf}(\frac{x_{max}-x_i}{\sqrt{2}\sigma}) + \frac{1}{2}\text{erf}(\frac{x_{min}-x_i}{\sqrt{2}\sigma})$.
    - Design Motivation: Spearman correlation tests confirm that the out-of-boundary ratio (OBA) and out-of-boundary value (OBV) are significantly negatively correlated with the noise metric (Sparseness).

2. **Adaptive Variance Computation**:
    - Function: Computes the optimal sampling variance for each input dimension.
    - Mechanism: Given a desired noise level $c$ (e.g., 0.95), the variance for each dimension is solved as: $\sigma_i = \frac{\min(|x_i - x_{min}|, |x_i - x_{max}|)}{\sqrt{2}\text{erfinv}(\frac{1+c}{2})}$. When $x_i = x_{min}$ or $x_i = x_{max}$, $\sigma_i = 0$ (boundary pixels are not perturbed). The covariance matrix is diagonal: $\Sigma_{ag} = \text{diag}(\sigma_1^2, ..., \sigma_D^2)$.
    - Design Motivation: Pixels closer to the boundary have less safe sampling space available and should receive smaller variances. This is a natural application of confidence-interval-based parameter estimation.

3. **Composition with Other Methods**:
    - Function: AdaptGrad can replace SmoothGrad when combined with any gradient-based explanation method.
    - Mechanism: For methods such as GI (Gradient×Input), IG (Integrated Gradients), and NoiseGrad, the SmoothGrad component is directly substituted with AdaptGrad (denoted with an A- prefix), yielding variants A-GI, A-IG, and A-NG.
    - Design Motivation: AdaptGrad is fully interface-compatible with SmoothGrad, making it a general-purpose drop-in improvement for gradient smoothing.

## Key Experimental Results

### Quantitative Evaluation — Sparseness on VGG16

| Method | Grad | SG | **AG** | S-GI | **A-GI** | S-IG(W) | **A-IG(W)** |
|------|------|-----|--------|------|----------|---------|-------------|
| Sparseness ↑ | 0.558 | 0.529 | **0.574** | - | **Higher** | - | **Higher** |

### Consistency and Invariance Tests (MNIST + MLP)

| Method | Consistency ↓ | Invariance ↓ |
|------|-------------|-------------|
| Grad | 0.02076 | 0.3483 |
| SmoothGrad | 0.01911 | 0.3613 |
| **AdaptGrad** | **0.02024** | **0.3484** |

### Ablation Study

| Configuration | Effect | Note |
|------|------|------|
| c=0.95 | Best visual quality | Recommended default |
| c=0.99 | Near optimal | Slightly conservative |
| c=0.999 | Slight degradation | Variance too small |
| Direct clipping of samples | Weaker than AdaptGrad | Truncation distorts distribution shape |

### Key Findings
- AdaptGrad already demonstrates clear denoising capability with as few as $N=10$ samples, whereas SmoothGrad requires $N=50$–$70$ to converge.
- The Invariance test confirms that AdaptGrad maintains gradient stability under constant input shifts (0.3484 vs. 0.3613 for SmoothGrad).
- Improvement is especially pronounced when combined with GI and IG, revealing finer semantic features such as facial details.
- Performance is consistent across VGG16, ResNet50, and InceptionV3 architectures.

## Highlights & Insights
- The theoretical analysis is concise and compelling: the convolution perspective exposes the paradox of a denoising method that introduces noise, and yields a closed-form expression for the additional noise.
- The method is minimally invasive—only the variance computation is modified, introducing no additional computational cost.
- Borrowing from confidence interval estimation gives the hyperparameter $c$ a clear probabilistic interpretation.

## Limitations & Future Work
- The theoretical analysis assumes the data domain is a simple hyperrectangle $[x_{min}, x_{max}]^D$; real data manifolds may be considerably more complex.
- Metrics such as Sparseness are imperfect proxies for noise and have limited alignment with human perception.
- Setting $\sigma=0$ for boundary pixels implies no smoothing at all, which may be overly conservative.

## Related Work & Insights
- **vs. SmoothGrad**: AdaptGrad addresses the fundamental flaw of SmoothGrad (OOB sampling) and outperforms it both theoretically and empirically.
- **vs. NoiseGrad**: NoiseGrad perturbs model parameters, whereas AdaptGrad perturbs inputs; the two approaches are complementary and can be combined (A-NG).
- **vs. Integrated Gradients**: IG addresses gradient saturation while AdaptGrad addresses noise—the two are orthogonal and mutually beneficial (A-IG achieves the best results).
- **vs. FusionGrad**: FusionGrad is a hybrid of NoiseGrad and SmoothGrad; AdaptGrad can replace the SmoothGrad component therein.

## Additional Notes
- The computational overhead of AdaptGrad is identical to that of SmoothGrad—only the variance computation differs, with no increase in the number of samples.
- Setting $c=0.95$ and $N=50$ is recommended as sufficient for most scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — The theoretical analysis of the noise source in SmoothGrad is novel and insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four metrics, three model architectures, and five method combinations are evaluated, though the number of datasets is limited.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and visual comparisons are intuitive.
- Value: ⭐⭐⭐ — Addresses a practical problem, but the scope of application is limited to gradient visualization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](../../ICML2026/interpretability/adaptive_querying_with_ai_persona_priors.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)

</div>

<!-- RELATED:END -->
