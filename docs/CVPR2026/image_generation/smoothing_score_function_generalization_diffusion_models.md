---
title: >-
  [Paper Note] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] This paper theoretically demonstrates that memorization in diffusion models stems from the "sharpness" of empirical score function weights (concentration of softmax weights…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Memorization"
  - "Generalization"
  - "Score Function Smoothing"
  - "Temperature Smoothing"
date: 2026-05-08
content_hash: 0d031f2f64c2191d
---

# Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework

**Conference**: CVPR 2026
**arXiv**: [2601.19285](https://arxiv.org/abs/2601.19285)
**Code**: [GitHub](https://github.com)
**Area**: Diffusion Models / Generative Model Theory
**Keywords**: Diffusion Models, Memorization, Generalization, Score Function Smoothing, Temperature Smoothing

## TL;DR

This paper theoretically demonstrates that memorization in diffusion models stems from the "sharpness" of empirical score function weights (concentration of softmax weights), and proposes two methods — noise unconditioning and temperature smoothing — that improve generalization and reduce memorization by smoothing score function weights while preserving generation quality.

## Background & Motivation

1. **Background**: Diffusion models achieve state-of-the-art generation quality, yet research has revealed that some generated samples are identical to training data (memorization), raising privacy and copyright concerns.
2. **Limitations of Prior Work**: In theory, a neural network that perfectly learns the empirical score function would only reproduce training samples. In practice, however, models do generate novel samples — this theory–practice discrepancy lacks a principled explanation.
3. **Key Challenge**: Why do neural networks partially mitigate memorization? And how can generalization be further improved?
4. **Goal**: To establish a theoretical framework explaining the causes of memorization and the mechanisms of generalization, and to propose corresponding improvements.
5. **Key Insight**: Analysis of the mathematical structure of empirical score function weights $w_{ij}(x)$, which take the form of a softmax function.
6. **Core Idea**: In high-dimensional spaces, empirical score function weights become extremely concentrated (sharp) at low noise levels, causing a single training point to dominate the sampling process. Neural networks achieve generalization by implicitly smoothing these weights.

## Method

### Overall Architecture

Theoretical analysis proves that memorization is directly related to the sharpness of score function weights. Two smoothing methods are proposed: (1) **noise unconditioning** — removing noise conditioning so that each training point adaptively selects its optimal noise level; (2) **temperature smoothing** — introducing a temperature parameter into the softmax weights to control the degree of smoothing.

### Key Designs

1. **Mathematical Explanation of Memorization ($\sigma$- and $\mu$-dominance)**:

    - **Function**: Formally proves why memorization occurs.
    - **Mechanism**: Score function weights take the form $w_{ij}(x) = \text{Softmax}(f(x, \mu_j, \sigma_i))$, where $f = -(d-2)\ln\sigma_i - \|x-\mu_j\|^2 / (2\sigma_i^2)$. *$\sigma$-dominance*: given position $x$ and center $\mu_j$, there exists an optimal noise level $\sigma_j^*$ whose weight exponentially dominates all other noise levels. *$\mu$-dominance*: given $x$, the weight of the nearest training point dominates others by a factor of $\exp(\delta\|\mu_j-\mu_l\|^2/\sigma_j^{*2})$. As $\sigma^* \to 0$, a single training point completely dominates the process, yielding memorization.
    - **Design Motivation**: In high-dimensional geometry, Gaussian distributions concentrate on thin shells, rendering the score function weights a sharp softmax.

2. **Noise Unconditioning**:

    - **Function**: Removes noise conditioning so that more training samples contribute to the score at each query point.
    - **Mechanism**: Replaces $s_\theta(x, t) \to s_\theta(x)$, unifying all noise levels under a single Gaussian mixture distribution $p_{\text{MN}}$. Each training point can adaptively find its optimal noise shell, enabling more training points to contribute meaningfully to the score function. Sampling is reformulated as gradient ascent on $\log p_{\text{MN}}$.
    - **Design Motivation**: In standard diffusion, fixed noise levels may place a query point outside the optimal shell of most training points, suppressing their weights. Unconditioning allows each point to "self-select" its optimal shell.

3. **Temperature Smoothing**:

    - **Function**: Explicitly controls the smoothness of score function weights.
    - **Mechanism**: Introduces a temperature vector $T$, modifying weights to $w_j^*(x;T) = \exp(f(x,\mu_j,\sigma_j^*)/T_j^*) / \sum_l \exp(f(x,\mu_l,\sigma_l^*)/T_l^*)$. For $\sigma_i \leq \sigma_{\text{collapse}}$, the score function is approximated using top-$K$ nearest training points, with $T_i = \max(\sigma_{\text{collapse}}/\sigma_i, 1)$.
    - **Design Motivation**: Higher temperature produces a more uniform weight distribution, delaying "collapse" to a single point during sampling and allowing continued exploration on the local manifold.

### Loss & Training

Unconditioning loss $\mathcal{L}_u$: identical to the standard NCSN objective but with noise input removed. Temperature loss $\mathcal{L}_T$: uses $\mathcal{L}_u$ for $\sigma > \sigma_{\text{collapse}}$, and temperature-modified score matching otherwise. Feature-space KNN consistently outperforms pixel-space KNN.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 FID(train) | FID(test) | CelebA FID(train) | FID(test) |
|--------|--------------------|-----------|--------------------|-----------|
| Conditioning | 6.49 | 6.56 | 7.25 | 7.81 |
| Unconditioning | 7.33 | 7.34 | 7.07 | 7.34 |
| Temp (feature KNN, K=100) | 7.96 | 7.98 | 8.40 | 8.19 |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Pixel-space KNN, T=7/σ, K=100 | FID=50.81 | Manifold curvature too large, causing collapse |
| Feature-space KNN, T=7/σ, K=100 | FID=7.96 | Low-curvature space supports strong smoothing |
| Weight ratio at α=1/3 | ≈403:1 | Even a slight distance advantage yields a large weight disparity |

### Key Findings

- The score function learned by neural networks is substantially smoother than the empirical score function — the expansion ratio $\gamma_{ex}$ is smaller by two orders of magnitude.
- Feature-space KNN consistently outperforms pixel-space KNN, supporting the hypothesis that neural networks achieve generalization by smoothing the local manifold.
- On the cat–caracal dataset, clear generalization emerges: the model generates "novel species" with caracal-like faces but short ears and gray fur.

## Highlights & Insights

- **Elegant Theoretical Framework**: The memorization problem is translated into an actionable mathematical framework via high-dimensional geometry and softmax analysis.
- **Optimization Perspective on Unconditioning**: Reformulating sampling as gradient ascent unifies diffusion theory with optimization theory.
- **Intuitive Geometric Analogy**: The mountaineer analogy makes the complex theory accessible — standard diffusion is a "step-by-step guide," while unconditioning provides a "panoramic map."

## Limitations & Future Work

- Temperature smoothing requires KNN search over training points, which incurs non-trivial cost at scale.
- FID slightly degrades (trading quality for generalization), requiring application-specific trade-off considerations.
- Current validation is limited to VE-SDE; extension to DDPM or Flow Matching remains unexplored.
- Theoretical analysis primarily assumes a Gaussian mixture model, whereas real data distributions are more complex.
- Joint tuning of the temperature parameter $T$ and early stopping timing requires practical experience.

## Related Work & Insights

- **vs. Yoon et al.**: Their work explains memorization from the perspective of model capacity; this paper explains it through the mathematical properties of the score function.
- **vs. Bonnaire et al.**: Their work studies how neural networks learn the score function; this paper further explains why the learned deviation from the empirical score function promotes generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The theoretical framework is highly insightful; both methods are grounded in rigorous mathematical analysis, and the mountaineer analogy is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative and qualitative validation is thorough; the expansion ratio experiment is cleverly designed, and the cat–caracal dataset provides an intuitive demonstration of generalization.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical exposition is clear, intuitive analogies are excellent, and proofs are rigorous yet readable.
- **Value**: ⭐⭐⭐⭐⭐ Establishes an important theoretical foundation for generalization in diffusion models, with far-reaching implications for understanding and improving generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization](verify_claimed_text-to-image_models_via_boundary-aware_prompt_optimization.md)

</div>

<!-- RELATED:END -->
