---
title: >-
  [Paper Note] Provable Maximum Entropy Manifold Exploration via Diffusion Models
description: >-
  [ICML2025][LLM Pretraining][Diffusion Models] Proposes the S-MEME algorithm, formulating the exploration problem of diffusion models as entropy maximization on an approximate data manifold. By leveraging the intrinsic relationship between the score function and the first-order variation of entropy, it bypasses density estimation, iteratively fine-tuning the pre-trained diffusion model via mirror descent, and proves convergence to the optimal exploration strategy.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Diffusion Models"
  - "Maximum Entropy Exploration"
  - "Manifold Exploration"
  - "Mirror Descent"
  - "Score Function"
  - "Fine-Tuning"
date: 2026-05-08
content_hash: df5bb26ffb752b72
---

# Provable Maximum Entropy Manifold Exploration via Diffusion Models

**Conference**: ICML2025  
**arXiv**: [2506.15385](https://arxiv.org/abs/2506.15385)  
**Code**: To be confirmed  
**Area**: Diffusion Model Exploration  
**Keywords**: Diffusion Models, Maximum Entropy Exploration, Manifold Exploration, Mirror Descent, Score Function, Fine-Tuning

## TL;DR

Proposes the S-MEME algorithm, formulating the exploration problem of diffusion models as entropy maximization on an approximate data manifold. By leveraging the intrinsic relationship between the score function and the first-order variation of entropy, it bypasses density estimation, iteratively fine-tuning the pre-trained diffusion model via mirror descent, and proves convergence to the optimal exploration strategy.

## Background & Motivation

- **Core Problem**: Generative models (especially diffusion models) excel at fitting data distributions. However, in scenarios like scientific discovery, the goal is not to mimic the data distribution but to **explore new regions on the data manifold**. How to leverage the representation capabilities of generative models to guide exploration is a fundamental challenge.
- **Limitations of Prior Work**:
    - Traditional generative models can only approximate $p_{\text{data}}$, tending to sample from high-density regions and failing to cover low-density regions.
    - Explicit uncertainty quantification is computationally expensive in high-dimensional spaces.
    - Density estimation ($p_T^{\text{pre}}(x)$) is highly challenging in high-dimensional practical scenarios.
- **Key Insight**: Pre-trained diffusion models implicitly define an approximate data manifold $\Omega_{\text{pre}} = \text{supp}(p_T^{\text{pre}})$, enabling uniform exploration by maximizing entropy on this manifold.

## Method

### 1. Maximum Entropy Manifold Exploration Problem

Formulates exploration as entropy maximization on the approximate data manifold $\Omega_{\text{pre}}$:

$$\pi^* \in \arg\max_{\pi} \; \mathcal{H}(p_T^\pi), \quad \text{s.t.} \; p_T^\pi \in \mathbb{P}(\Omega_{\text{pre}})$$

where $\mathcal{H}(\mu) = -\int d\mu \log \frac{d\mu}{dx}$ is the differential entropy functional. The compactness of $\Omega_{\text{pre}}$ is guaranteed by Proposition 1 (when the score function is Lipschitz and the noise distribution is a truncated Gaussian).

### 2. First-order Variation of Entropy and Surprise Maximization

Taking the first-order variation of the entropy functional yields the **regularized surprise maximization** principle:

$$\pi^* \in \arg\max_\pi \; \mathbb{E}_{x \sim \pi}\left[-\log p_T^{\text{pre}}(x)\right] - \alpha \, D_{KL}(p_T^\pi, p_T^{\text{pre}})$$

- The first term $-\log p_T^{\text{pre}}(x)$ is the surprise: encouraging sampling in the low-density regions of the pre-trained model.
- The second term is the KL regularization: constraining the fine-tuned model from deviating from the data manifold.

### 3. Key Connection Bypassing Density Estimation

The core theoretical contribution of the paper—the gradient of the first-order variation of entropy equals the negative score function:

$$\nabla_x \delta\mathcal{H}(p_T^\pi)(x) = -\nabla_x \log p_T^\pi(x) = -s^\pi(x, T)$$

This means that without estimating the probability density $p_T^{\text{pre}}(x)$, the score function of the pre-trained model $s^{\text{pre}}(x, T)$ can be directly used as the reward gradient, solved in conjunction with first-order fine-tuning methods such as Adjoint Matching.

### 4. The S-MEME Algorithm

**Score-based Maximum Entropy Manifold Exploration (S-MEME)** decomposes non-linear entropy optimization into iterative linear fine-tuning:

```text
Input: Pre-trained model π_pre, number of iterations K, regularization coefficients {α_k}
Initialize: π_0 = π_pre
for k = 1 to K:
    Set reward gradient: ∇f_k = -s^{k-1}(·, T)
    Solve fine-tuning: π_k = LinearFineTuningSolver(∇f_k, α_k, π_{k-1})
Return π_K
```

Each iteration step corresponds to a Mirror Descent step, where the KL divergence plays the role of the Bregman divergence.

### 5. Convergence Guarantees

- **Ideal Case (Theorem 5.2)**: Under the assumptions of exact score estimation and exact optimization, a single fine-tuning step is sufficient to achieve the optimal exploration strategy.
- **Realistic Case (Theorem 7.1)**: Under the condition that noise and bias satisfy Assumption 7.3 (bias $\|b_k\|_\infty \to 0$, step size schedule satisfies the Robbins-Monro conditions), S-MEME converges almost surely to the optimal exploration strategy.
- The convergence rate is $\tilde{\mathcal{O}}((\log\log k)^{-1})$.

**Key Theoretical Property**: The negative entropy $\mathcal{F} = -\mathcal{H}$ is **1-smooth and 1-strongly convex** with respect to itself (Lemma 5.1), which guarantees the applicability of the mirror descent framework.

## Key Experimental Results

### Synthetic Data Experiments

| Metrics | Pre-trained Model $\pi^{\text{pre}}$ | S-MEME $\pi_4$ (4 steps) |
|------|------|------|
| Entropy Estimation | Low (concentrated in high-density regions) | Significantly improved |
| Low-density Region Coverage | Poor | Good (more uniform density) |
| Data Support Maintenance | — | ✓ Maintained |

- Monte Carlo entropy estimation on 80,000 samples shows that just 4 steps of S-MEME can significantly increase the entropy.

### Text-to-Image Experiments

| Settings | Pre-trained Model $\pi^{\text{pre}}$ | S-MEME $\pi_3$ (3 steps) |
|------|------|------|
| Prompt | "A creative architecture." | Same as left |
| Generation Diversity | Conventional architectural designs | Higher complexity and originality |
| Semantic Fidelity | ✓ | ✓ Maintained |

- Based on the Stable Diffusion pre-trained model, the architectural images generated after S-MEME fine-tuning exhibit higher creative complexity.
- Comparisons with fixed initial noise reveal that the fine-tuned model tends to sample from the low-density regions of the pre-trained model.

## Highlights & Insights

1. **Elegant Theoretical Connection**: Establishes an equivalence relationship between the gradient of the first-order variation of entropy and the score function (Eq. 12), fundamentally bypassing the bottleneck of high-dimensional density estimation.
2. **Mirror Descent Perspective**: Interprets diffusion model fine-tuning as mirror descent on the probability space, with KL divergence acting as the Bregman divergence—a perspective that is highly natural and theoretically elegant.
3. **Trade-off between Exploration and Validity**: The regularization coefficient $\alpha$ flexibly controls the balance between exploration (low $\alpha$) and conservatism (high $\alpha$).
4. **Complete Convergence Theory**: Progresses from idealized single-step optimality to asymptotic convergence under realistic scenarios, providing a step-by-step and complete theoretical foundation.
5. **No External Reward Required**: Purely utilizes the model's own score function as an intrinsic reward, achieving self-guided exploration.

## Limitations & Future Work

1. **Slow Convergence Rate**: $\tilde{\mathcal{O}}((\log\log k)^{-1})$ is theoretically optimal at present, but may require a large number of iterations in practice.
2. **Computational Overhead**: Each iteration step requires a full diffusion model fine-tuning procedure (Adjoint Matching), resulting in high cumulative computational costs for multi-step iterations.
3. **Limited High-Dimensional Experiments**: Although text-to-image experiments demonstrate qualitative effects, they lack systematic quantitative evaluation metrics (such as FID, diversity metrics, etc.).
4. **Strength of the Manifold Assumption**: The method relies on the quality of the manifold $\Omega_{\text{pre}}$ implicitly defined by the pre-trained model. If pre-training is insufficient, the exploration scope is limited.
5. **Validation Only in Continuous Domains**: It has not been validated on discrete/structured domains such as molecular design, which are precisely the core scenarios of scientific discovery.
6. **Support Compatibility Assumption** (Assumption 7.1) is difficult to verify precisely in practice.

## Related Work & Insights

- **Diffusion Model Fine-Tuning**: DPPO, DDPO, Adjoint Matching (Domingo-Enrich et al., 2024)—Ours uses Adjoint Matching as the LinearFineTuningSolver.
- **Exploration and Intrinsic Rewards**: Count-based exploration, RND—Ours' surprise maximization can be viewed as a density estimation-based intrinsic reward.
- **Continuous-Time RL**: Doya (2000), Zhao et al. (2024)—treating the reverse process of diffusion models as a continuous-time RL policy.
- **Mirror Descent / Mirror Flow**: Lu et al. (2018), Hsieh et al. (2019)—providing a theoretical framework for optimization on probability spaces.
- **Insight**: This framework can be generalized to any scenario requiring optimal exploration within an implicit space defined by a generative model (e.g., drug discovery, materials design).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The connection "first-order variation of entropy = negative score" is very beautiful)
- Experimental Thoroughness: ⭐⭐⭐ (Synthetic experiments are thorough, but high-dimensional validation is relatively weak)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical hierarchy, with motivation, methodology, and guarantees closely linked)
- Value: ⭐⭐⭐⭐ (Provides a solid theoretical foundation for exploration in diffusion models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/llm_pretraining/composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[CVPR 2025\] Exploration-Driven Generative Interactive Environments](../../CVPR2025/llm_pretraining/exploration-driven_generative_interactive_environments.md)
- [\[ICLR 2026\] Soft-Masked Diffusion Language Models](../../ICLR2026/llm_pretraining/soft-masked_diffusion_language_models.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[ICLR 2026\] Autoregressive Models Rival Diffusion Models at Any-Order Generation](../../ICLR2026/llm_pretraining/autoregressive_models_rival_diffusion_models_at_any-order_generation.md)

</div>

<!-- RELATED:END -->
