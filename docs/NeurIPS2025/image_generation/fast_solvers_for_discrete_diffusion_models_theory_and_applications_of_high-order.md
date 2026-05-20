---
title: >-
  [Paper Note] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms
description: >-
  [NeurIPS 2025][Image Generation][Discrete Diffusion] This work introduces high-order numerical methods into discrete diffusion model inference for the first time…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Discrete Diffusion"
  - "High-Order Solvers"
  - "τ-leaping"
  - "Trapezoidal"
  - "Text Generation"
date: 2026-05-08
content_hash: 64a3aa434126e8b8
---

# Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms

**Conference**: NeurIPS 2025
**arXiv**: [2502.00234](https://arxiv.org/abs/2502.00234)  
**Code**: [DiscreteFastSolver](https://github.com/yuchen-zhu-zyc/DiscreteFastSolver)  
**Area**: Diffusion Models
**Keywords**: Discrete Diffusion, High-Order Solvers, τ-leaping, Trapezoidal, Text Generation, Image Generation

## TL;DR

This work introduces high-order numerical methods into discrete diffusion model inference for the first time, proposing two second-order solvers — θ-RK-2 and θ-Trapezoidal — and theoretically proving that θ-Trapezoidal improves the discretization error from first-order $\mathcal{O}(\kappa T)$ to second-order $\mathcal{O}(\kappa^2 T)$. Experiments spanning 200M–8B models consistently demonstrate improvements across text, image, and mathematical reasoning tasks.

## Background & Motivation

**Background**: Discrete diffusion models have achieved significant progress in generating discrete data such as text, images, and molecules. Unlike continuous diffusion models, discrete models are naturally suited for data in combinatorial spaces such as language tokens, molecular sequences, and quantized images.

**Limitations of Prior Work**: Inference efficiency is the core bottleneck for discrete diffusion models. Existing methods fall into two categories: (1) **Exact simulation methods** (e.g., uniformization, First-Hitting Sampler) are theoretically unbiased but have unpredictable inference times — the number of jumps grows sharply near the end of the reverse process, causing excessive redundant computation, and variable-length schedules across samples hinder batch parallelism; (2) **Approximate methods** (e.g., τ-leaping) are simple and parallelizable, but as first-order methods, they require very small step sizes to control discretization error, yielding limited sample quality under compute constraints.

**Key Challenge**: Continuous diffusion models already benefit from a rich set of high-order acceleration methods (DPM-Solver, DDIM, etc.), yet in the discrete domain, high-order methods have remained absent due to the discontinuous nature of the state space and the mathematical complexity of Poisson jump processes. The first-order accuracy of τ-leaping — analogous to the Euler method in the continuous setting — has been shown to be far from optimal, suggesting that discrete diffusion should likewise benefit from higher-order methods.

**Goal**: To develop high-order numerical inference methods for discrete diffusion models that achieve higher-quality samples with fewer function evaluations (NFE), without modifying the underlying model.

**Key Insight**: Drawing inspiration from Runge-Kutta methods for ODEs and high-order τ-leaping methods from chemical reaction simulation, the paper adapts these classical numerical techniques to the Poisson jump processes of discrete diffusion models.

**Core Idea**: Replace the first-order τ-leaping with a two-stage predictor-corrector high-order solver that exploits additional information at an intermediate point in each step to elevate the discretization error from first-order to second-order.

## Method

### Overall Architecture

In the reverse process of discrete diffusion models, a continuous-time Markov chain over the state space $\mathbb{X} = [S]^d$ defines transition probabilities through a rate matrix. Inference requires sampling from a Poisson jump process driven by the learned reverse rates, parameterized via a score function. Standard τ-leaping performs Poisson sampling at each step using only the rate at the current time point (first-order). The proposed methods additionally evaluate the rate at an intermediate time point and use a weighted combination to achieve second-order accuracy.

### Key Designs

1. **θ-RK-2 (Warm-Up Scheme)**:

    - **Function**: Analogous to second-order Runge-Kutta for ODEs; provides a two-stage predictor-corrector inference scheme for discrete diffusion.
    - **Mechanism**: In the first stage, τ-leaping advances $\theta\Delta$ steps to an intermediate point $\hat{y}_{\rho_n}^*$ and evaluates the rate $\hat{\mu}_{\rho_n}^*$ there. In the second stage, starting from the original point $\hat{y}_{s_n}$, a full step $\Delta_n$ is taken using the weighted sum of the initial and intermediate rates: $[(1-\frac{1}{2\theta})\hat{\mu}_{s_n} + \frac{1}{2\theta}\hat{\mu}_{\rho_n}^*]$, where the weights are interpolation coefficients.
    - **Design Motivation**: A direct adaptation of the ODE RK-2 idea to Poisson processes. However, theoretical analysis reveals that second-order convergence holds only for $\theta \in (0, 1/2]$ (conditional second-order), because the weighted sum transitions from interpolation to extrapolation when $\theta > 1/2$, creating technical difficulties.

2. **θ-Trapezoidal (Core Contribution)**:

    - **Function**: A more robust second-order solver for discrete diffusion that achieves unconditional second-order convergence for all $\theta \in (0,1]$.
    - **Mechanism**: The first stage is identical to θ-RK-2 — τ-leaping advances $\theta\Delta$ to an intermediate point. The second stage differs in two key ways: (a) it departs from the intermediate point $\hat{y}_{\rho_n}^*$ rather than the original point and takes $(1-\theta)\Delta_n$ steps; (b) it uses extrapolation coefficients $(\alpha_1, -\alpha_2)$ rather than interpolation coefficients, where $\alpha_1 = \frac{1}{2\theta(1-\theta)}$, $\alpha_2 = \frac{(1-\theta)^2+\theta^2}{2\theta(1-\theta)}$, satisfying $\alpha_1 - \alpha_2 = 1$. The rate is given by $(\alpha_1 \hat{\mu}_{\rho_n}^* - \alpha_2 \hat{\mu}_{s_n})_+$.
    - **Design Motivation**: Each time interval is split into two sub-intervals and handled separately, with extrapolation rather than interpolation weights. This decomposed treatment enables the proof of second-order convergence without any restriction on $\theta$ — the extrapolation structure allows key error terms to be unconditionally bounded via Dynkin's formula.

3. **Multi-Jump Rejection Mechanism**:

    - **Function**: Ensures the algorithm is well-defined.
    - **Mechanism**: Following standard practice in the discrete diffusion literature, updates that produce multiple jumps in the same dimension are rejected. Analysis shows that the rejection probability is only $\mathcal{O}(\kappa)$, leaving asymptotic accuracy unaffected.
    - **Design Motivation**: In Poisson jump processes, large step sizes may cause multiple transitions in the same dimension, potentially producing invalid states in the discrete space.

### Theoretical Guarantees

**Theorem (Unconditional Second-Order Convergence of θ-Trapezoidal)**: Under regularity assumptions, $D_{\text{KL}}(p_\delta \| \hat{q}_{T-\delta}^{\text{trap}}) \lesssim \exp(-T) + (\epsilon_I + \epsilon_{II})T + \kappa^2 T$. Compared to τ-leaping's $\mathcal{O}(\kappa T)$, the discretization error is improved to $\mathcal{O}(\kappa^2 T)$, holding unconditionally for all $\theta \in (0,1]$. The second-order convergence of θ-RK-2 holds only for $\theta \in (0, 1/2]$.

## Key Experimental Results

### Main Results

**Text Generation (RADD/GPT-2 scale, d=1024, S=50258)**:

| Method | NFE=128 | NFE=1024 |
|--------|---------|----------|
| FHS (exact simulation) | ≤122.7 | ≤109.4 |
| Euler | ≤86.3 | ≤44.7 |
| Tweedie τ-leap | ≤85.7 | ≤44.3 |
| τ-leaping | ≤52.4 | ≤28.8 |
| Semi-AR | ≤360.8 | ≤147.4 |
| θ-RK-2 | ≤64.3 | ≤36.3 |
| **θ-Trapezoidal** | **≤49.1** | **≤27.6** |

θ-Trapezoidal consistently outperforms all existing methods across all NFE values. Notably, exact simulation (FHS) performs worse than approximate methods at high NFE — zero discretization error does not guarantee better samples, since score estimation errors are amplified near the end of the process.

**Image Generation (MaskGIT/ImageNet 256×256, d=256, S=1025)**:

| Method | NFE=16 FID↓ | NFE=32 FID↓ | NFE=64 FID↓ |
|--------|------------|------------|------------|
| Euler | ~12 | ~8 | ~7 |
| τ-leaping | ~10 | ~7 | ~6 |
| **θ-Trapezoidal** | **~8** | **~6** | **~5** |

θ-Trapezoidal consistently achieves lower FID than first-order methods for NFE≥16. FHS and parallel decoding have an advantage at very low NFE (≤8) but saturate quickly.

### Ablation Study

| Configuration | Text PPL (NFE=128) | Image FID (NFE=32) | Notes |
|--------------|-------------------|-------------------|-------|
| τ-leaping (first-order) | 52.4 | ~7 | Baseline |
| θ-RK-2 (θ=0.5) | 64.3 | — | Conditional second-order; degrades for θ>0.5 |
| θ-Trapezoidal (θ=0.5) | **49.1** | **~6** | Unconditional second-order; stably optimal |
| θ-Trapezoidal (θ=0.3) | ~49.5 | ~6 | Robust across θ∈[0.3,0.5] |
| θ-Trapezoidal (θ=0.8) | ~50 | ~6.5 | Slight degradation for larger θ |

**LLaDA-Instruct 8B Mathematical Reasoning (GSM8K)**:

| Method | NFE=64 | NFE=128 | NFE=256 |
|--------|--------|---------|---------|
| Semi-AR (Rand.) | 33.8% | 34.3% | **40.3%** |
| **θ-Trapezoidal** | **35.1%** | **38.4%** | 39.7% |

The advantage is especially pronounced in low-NFE (compute-constrained) settings.

### Key Findings

- **θ-Trapezoidal is Comprehensively Superior**: It consistently outperforms τ-leaping and θ-RK-2 across text, image, and mathematical reasoning tasks, in full agreement with theoretical predictions.
- **θ Hyperparameter is Robust**: θ∈[0.3, 0.5] is a robust choice across all tasks; the performance surface is flat and does not require careful tuning.
- **Exact Simulation ≠ Optimal**: The unbiasedness of FHS becomes a disadvantage in the presence of score estimation errors — a large number of redundant NFEs are wasted in regions where the score is inaccurate. This supports the view that "controlled approximation outperforms uncontrolled exactness."
- **Consistently Effective from 200M to 8B Models**: Performance gains are observed on both MaskGIT (~200M scale) and LLaDA (8B scale), demonstrating that the advantages of high-order solvers are not tied to any particular model scale.

## Highlights & Insights

- **Filling an Important Gap**: While continuous diffusion has high-order methods such as DPM-Solver, discrete diffusion has lacked them entirely. This paper presents the first rigorously proven and experimentally validated second-order solver for discrete diffusion, filling a critical gap in both the theory and practice of discrete generative models.
- **Deep Reason Why Trapezoidal Outperforms RK-2**: The second stage of RK-2 departs from the original point and traverses the full step (interpolation), whereas Trapezoidal departs from the intermediate point and traverses the remaining step (extrapolation). Although extrapolation is intuitively more "aggressive," it is precisely this structure that allows error terms to be unconditionally bounded in theory — a classical insight from chemical reaction simulation successfully transferred to AI.
- **Practical Value of Training-Free Acceleration**: The method is a purely inference-time improvement that requires no model retraining. It can be applied as a plug-and-play accelerator to already-deployed discrete diffusion models, reducing inference costs.
- **Counterintuitive Finding: Exact ≠ Better**: Exact methods such as FHS are outperformed by approximate methods at high NFE. This finding has important implications for the selection of inference strategies in discrete diffusion models.

## Limitations & Future Work

- **Non-Negativity Assumption for Extrapolation**: The theoretical proof of θ-Trapezoidal relies on the assumption $\alpha_1 \hat{\mu}_{\rho}^* - \alpha_2 \hat{\mu}_{s} \geq 0$. Experiments show this holds in over 95% of cases, but theoretical confirmation remains an open problem (a long-standing unresolved question from the chemical reaction simulation literature).
- **Only Second-Order Methods Considered**: Whether higher-order (third- or fourth-order) schemes can yield further improvements, and the trade-off between additional NFE overhead and accuracy gains for multi-stage methods, remain to be studied.
- **Text Experiments Limited to GPT-2 Scale**: Although the GSM8K evaluation on an 8B model validates scalability, systematic evaluation on larger-scale LLMs is still lacking.
- **Combination with Other Acceleration Methods**: How to jointly use the proposed approach with orthogonal acceleration techniques such as caching and speculative sampling remains an open question.
- **Theoretical Analysis of Covariance Error**: The chemistry literature notes that Trapezoidal also achieves second-order advantages in covariance error, but formal proof in the discrete diffusion setting is left for future work.

## Related Work & Insights

- **vs τ-leaping (Campbell et al., 2022)**: A first-order approximate method and the direct target of improvement in this work. θ-Trapezoidal requires 2 score evaluations per step instead of 1, but reduces the discretization error from $\mathcal{O}(\kappa)$ to $\mathcal{O}(\kappa^2)$ — a highly significant improvement when $\kappa$ is not sufficiently small.
- **vs DPM-Solver (Lu et al., 2022)**: A high-order method for continuous diffusion models. DPM-Solver exploits exponential integrators for ODEs, whereas this paper addresses Poisson jump processes — a mathematically distinct setting that requires an entirely new theoretical framework.
- **vs FHS (Zheng et al., 2024)**: An exact simulation method. This paper demonstrates that "exact does not mean optimal" — controlled approximation is in practice superior when score estimates carry errors, since exact methods waste excessive computation in regions of high score error.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First introduction of high-order numerical methods into discrete diffusion, filling an important gap in theory and practice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 200M–8B models across text, image, and mathematical reasoning, though large-scale LLM evaluation remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical exposition is rigorous and experimental analysis is detailed, though dense mathematical notation may affect readability.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play inference acceleration with direct practical impact on the deployment of discrete diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)

</div>

<!-- RELATED:END -->
