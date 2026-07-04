---
title: >-
  [Paper Note] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)
description: >-
  [ICML 2025][Image Generation][Discrete Diffusion Models] This paper proposes the Di4C method, which captures correlations between dimensions through a "mixture" model. Combined with a consistency loss function, it distills multi-step discrete diffusion models into few-step models, demonstrating effectiveness across both image and language tasks.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Discrete Diffusion Models"
  - "Knowledge Distillation"
  - "Dimensional Correlation"
  - "Mixture Models"
  - "Few-step Sampling"
date: 2026-05-08
content_hash: d2931a5057632b80
---

# Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)

**Conference**: ICML 2025  
**arXiv**: [2410.08709](https://arxiv.org/abs/2410.08709)  
**Code**: [sony/di4c](https://github.com/sony/di4c)  
**Area**: Diffusion Models / Discrete Generative Models  
**Keywords**: Discrete Diffusion Models, Knowledge Distillation, Dimensional Correlation, Mixture Models, Few-step Sampling

## TL;DR

This paper proposes the Di4C method, which captures correlations between dimensions through a "mixture" model. Combined with a consistency loss function, it distills multi-step discrete diffusion models into few-step models, demonstrating effectiveness across both image and language tasks.

## Background & Motivation

Discrete diffusion models face unique challenges in reducing sampling steps that do not exist in continuous models:

**Dimensional Independence Limitation**: Traditional discrete diffusion models use a "product model", where each dimension models the sampling distribution independently, i.e., $p_{s|t}(x_s|x_t) = \prod_{d=1}^D p_{s|t}^d(x_s^d|x_t)$. Although this ensures scalability (reducing the output space size from $O(D^{|S|})$ to $O(D|S|)$), it neglects the dependencies between dimensions.

**Fundamental Obstacle to Few-step Sampling**: In extreme cases (such as single-step denoising in masked diffusion), product models are completely incapable of approximating complex joint distributions. Theorem 1 of this paper proves that the total variation (TV) distance of an $N$-step product model has an $\Omega(1/N)$ lower bound, indicating that modeling dimensional correlation is essential to reducing the number of steps.

**Limitations of Training Losses**: The loss functions of continuous-time score-based discrete diffusion only require marginal distributions. Consequently, even if a model possesses the capacity to represent dimensional correlations, it cannot learn them from these losses.

Key Insight: **The composition of multi-step product models can implicitly capture dimensional correlation**, even though each individual step is dimension-independent.

## Method

### Overall Architecture

The core idea of Di4C is to distill a multi-step teacher (a product model) into a few-step student (a mixture model):
$$p_{0|t_n}^\theta \approx p_{0|t_1}^\psi \circ \cdots \circ p_{t_{n-1}|t_n}^\psi$$

### Key Designs

#### 1. Mixture Model (Section 3.2)

To capture dimensional correlations, a mixture model is proposed:
$$p_{s|t}^\theta(x_s|x_t) = \mathbb{E}_\lambda[p_{s|t}^\theta(x_s|x_t;\lambda)]$$
$$\text{where } p_{s|t}^\theta(x_s|x_t;\lambda) = \prod_{d=1}^D p_{s|t}^{\theta,d}(x_s^d|x_t;\lambda)$$

- Each component conditioned on $\lambda$ remains a product model (dimension-independent), but taking the expectation over $\lambda$ introduces dimensional correlation.
- **Proposition 1** proves the **universal approximation** property of the mixture model: any discrete distribution can be represented as a mixture of product distributions.
- It introduces almost no additional computational overhead during inference, requiring only sampling and injecting $\lambda$.

#### 2. Distillation Loss Functions (Section 3.3)

**Distillation Loss**—approximating the teacher at a small timestep $\delta$:
$$\mathcal{L}_{\text{distil}}(\theta;\psi,r_\delta,\delta) = \mathbb{E}_{x_\delta \sim r_\delta}[D_{\text{KL}}(p_{0|\delta}^\psi(\cdot|x_\delta) \| p_{0|\delta}^\theta(\cdot|x_\delta))]$$

**Consistency Loss**—learning the dimensional correlations manifested by the composition of the teacher:
$$\mathcal{L}_{\text{consis}}(\theta;\psi,r_t,s,u,t) = \mathbb{E}_{x_t \sim r_t}[D_{\text{KL}}(p_{s|u}^\theta \circ p_{u|t}^\psi(\cdot|x_t) \| p_{s|t}^\theta(\cdot|x_t))]$$

- The key to consistency loss lies in the joint use of both the student denoiser and the teacher denoiser.
- It is approximated computationally using Monte Carlo or control variates.

### Loss & Training

- Distillation loss is used at small $\delta$ (where a single step of the teacher suffices), while dimensional correlation is primarily introduced via consistency loss.
- The reference distribution $r_t$ can be either $q_t$ generated from data or the distribution obtained from multi-step teacher denoising.
- In practice, $\lambda$ can be implemented by feeding an additional random noise vector into the network.

## Key Experimental Results

### Main Results

**1. CIFAR-10 Pixel-level Discrete Gaussian Diffusion**
- Uses Campbell et al. (2022) as the teacher.
- Significantly improves the teacher's FID metrics under few-step sampling (2-4 steps).

**2. ImageNet Class-Conditional Generation (Masked Diffusion)**
- Teacher: Besnier & Chen (2023)
- Achieves a 2x speedup while maintaining generation quality comparable to the teacher.

**3. Language Modeling (OpenWebText, Masked Diffusion LM)**
- Further distills on top of an already distilled model (Deschenaux & Gulcehre, 2025).
- Further improves performance by capturing dimensional correlations, without significantly compromising sampling diversity.

### Ablation Study

- Impact of the number of mixture components in the mixture model on quality.
- Selection of distillation steps and consistency loss timesteps.
- Performance comparison between consistency loss and pure distillation loss.

### Key Findings

| Theoretical Result | Content |
|---------|------|
| Theorem 1 Upper Bound | TV distance of the $N$-step product model = $O(1/N)$ |
| Theorem 1 Lower Bound | Even in a simple case where $|S|=D=2$, the lower bound is $\Omega(1/N)$ |
| Theorem 2 | The Di4C loss upper-bounds the distance between teacher and student output distributions |
| Lemma 1 | TV distance of a single-step product model = $O(\epsilon^2)$ |

## Highlights & Insights

1. **Elegant Combination of Theory and Practice**: Theorem 1 rigorously proves that product models require $\Omega(1/\epsilon)$ steps, establishing the theoretical necessity of modeling dimensional correlation.
2. **Elegance and Simplicity of the Mixture Model**: By indexing a family of product models via a simple random variable $\lambda$, it achieves a universal representation of dimensional correlation with minimal inference overhead.
3. **Cross-Modal Generality**: The method is effective for both images (pixel-level and masked diffusion) and language (masked diffusion LM).
4. **Connection to Continuous Consistency Models**: The design of consistency loss is conceptually similar to consistency models in the continuous domain, but operates on the composition of conditional probabilities.
5. **Stepwise Approximation Error of $O(\epsilon^2)$**: Leveraging the dimension-factorized nature of the forward process, the paper proves the good approximation performance of product models over small time intervals.

## Limitations & Future Work

1. **Computation of Consistency Loss**: Exact computation requires summing over high-dimensional discrete spaces, and Monte Carlo approximation introduces variance.
2. **Mixture Model Capacity**: The distribution and dimension choices of $\lambda$ require empirical tuning, and the theoretically required number of mixtures can be quite large.
3. **Dependency on Teacher Quality**: The upper bound of the distillation performance is constrained by the multi-step sampling quality of the teacher.
4. **Continuous-Time Setting**: The theoretical analysis is conducted in a continuous-time framework, but practical training employs discrete timesteps.
5. **Scaling to Larger Scales**: The model scales used in the ImageNet experiments are relatively limited, and the effectiveness on large-scale text generation remains to be further validated.

## Related Work & Insights

- **Continuous-Domain Distillation**: Consistency distillation of Salimans & Ho (2022), Kim et al. (2024).
- **Foundations of Discrete Diffusion**: Austin et al. (2021) D3PM, Campbell et al. (2022) continuous-time discrete diffusion.
- **Concurrent Work**: Park et al. (2025), Liu et al. (2024), Xu et al. (2025) also highlight the issue of product models neglecting dimensional correlation.
- **Insights**: The mixture model concept could inspire other scenarios requiring efficient modeling of high-dimensional discrete joint distributions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to systematically address few-step distillation in discrete diffusion, introducing innovations in both theory and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Cross-modal validation (image and language) across multiple discrete diffusion variants.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations, with Figure 1 intuitively showcasing the core problem.
- Value: ⭐⭐⭐⭐⭐ — Holds significant theoretical and practical importance for accelerating discrete diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variational Autoencoding Discrete Diffusion with Enhanced Dimensional Correlations Modeling](../../ICLR2026/image_generation/variational_autoencoding_discrete_diffusion_with_enhanced_dimensional_correlatio.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](../../NeurIPS2025/image_generation/learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[ICML 2025\] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](../../NeurIPS2025/image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[ICML 2025\] Privacy Amplification Through Synthetic Data: Insights from Linear Regression](privacy_amplification_through_synthetic_data_insights_from_linear_regression.md)

</div>

<!-- RELATED:END -->
