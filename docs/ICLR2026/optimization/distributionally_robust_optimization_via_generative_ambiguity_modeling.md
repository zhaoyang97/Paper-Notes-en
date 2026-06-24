---
title: >-
  [Paper Note] Distributionally Robust Optimization via Generative Ambiguity Modeling
description: >-
  [ICLR 2026][Optimization][Distributionally Robust Optimization] This paper defines the "ambiguity set" of DRO directly on the **parameter space** of generative models (Diffusion Models / VAEs). By using reconstruction loss to constrain the consistency between generated and nominal distributions, and solving the inner maximization via dual learning and policy optimization, the authors develop GAS-DRO—a tractable DRO algorithm capable of searching for worst-case distributions a…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Distributionally Robust Optimization"
  - "DRO"
  - "Generative Models"
  - "Diffusion Models"
  - "Ambiguity Set"
  - "OOD Generalization"
  - "Policy Optimization"
date: 2026-05-08
content_hash: db5c5ea73a1a730b
---

# Distributionally Robust Optimization via Generative Ambiguity Modeling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=q67t0gFrdY](https://openreview.net/forum?id=q67t0gFrdY)  
**Code**: [https://github.com/CIGLAB-Houston/GAS-DRO](https://github.com/CIGLAB-Houston/GAS-DRO)  
**Area**: optimization  
**Keywords**: Distributionally Robust Optimization, DRO, Generative Models, Diffusion Models, Ambiguity Set, OOD Generalization, Policy Optimization  

## TL;DR
This paper defines the "ambiguity set" of DRO directly on the **parameter space** of generative models (Diffusion Models / VAEs). By using reconstruction loss to constrain the consistency between generated and nominal distributions, and solving the inner maximization via dual learning and policy optimization, the authors develop GAS-DRO—a tractable DRO algorithm capable of searching for worst-case distributions across different support sets.

## Background & Motivation
**Background**: Distributionally Robust Optimization (DRO) enhances Out-of-Distribution (OOD) robustness in statistical learning by optimizing worst-case performance within a min-max framework. Its effectiveness relies heavily on the design of the "ambiguity set," where the inner layer searches for the worst-case distribution and the outer layer optimizes decisions against it.

**Limitations of Prior Work**: Classical ambiguity sets face trade-offs. $\phi$-divergence sets (e.g., KL) require every distribution $P$ in the set to be absolutely continuous with respect to the nominal distribution $P_0$ ($P \ll P_0$, identical support), which limits robustness when **support set shifts** occur. While Wasserstein-based sets allow support shifts, solving the inner maximization over an infinite-dimensional probability space is computationally difficult. Common convex assumptions fail in deep learning, and adversarial approximations tend to be **overly conservative**, including invalid distributions inconsistent with $P_0$.

**Key Challenge**: There is a fundamental tension between **expressiveness and tractability** in ambiguity set design. Restricting support (as in $\phi$-divergence) leads to insufficient expressiveness and poor robustness, while unrestricted support (as in Wasserstein) leads to intractable infinite-dimensional optimization. Existing works integrating generative models either remain constrained by support set restrictions or suffer from high approximation errors.

**Goal**: Construct an ambiguity set that simultaneously satisfies three properties: (1) coverage of diverse distributions with different supports to identify worst-case scenarios; (2) consistency with the nominal distribution to avoid over-conservatism; and (3) computational tractability.

**Key Insight**: **This work is the first to define the ambiguity set directly on the finite parameter space of generative models.** Generative models can approximate the true data distribution (ensuring consistency) and generate diverse samples beyond the nominal support (ensuring expressiveness), while offering a finite-dimensional parameterized space (ensuring tractability).

**Core Idea**: The theoretical anchor is using **inclusive KL divergence** rather than exclusive KL to constrain consistency. This allows $P_\theta$ to have a broader support than $P_0$.

## Method

### Overall Architecture
GAS-DRO reformulates the inner maximization of DRO from an "infinite-dimensional probability space search" to a "constrained optimization over generative model parameters $\theta$." The framework is a nested loop: the InnerMax updates parameters $\theta$ to find the worst-case distribution $P_\theta$, and the outer minimization updates decision variables $w$ using samples from $P_\theta$.

```mermaid
flowchart LR
    P0[Nominal P0 + Dataset S0] --> GM[Generative Model Pθ]
    GM --> GAS[Generative Ambiguity Set: J θ,P0 ≤ ε]
    GAS --> Inner[InnerMax: Dual + Policy Opt<br/>Find worst Pθ]
    Inner --> Sample[Sample adversarial Sj ~ Pθ]
    Sample --> Outer[Outer Min: Update decision w]
    Outer --> Inner
```

### Key Designs

**1. Generative Ambiguity Set (GAS): Defining the "parameter ball" via inclusive KL.** Instead of traditional sets $B(P_0,\epsilon)=\{P\mid D(P,P_0)<\epsilon\}$, this work uses the **reconstruction loss** of generative models. Lemma 1 proves that for likelihood-based models, the inclusive KL divergence between the nominal and generated distributions is upper-bounded by the reconstruction loss: $D_{KL}(P_0\|P_\theta)\le J(\theta,P_0)+R(p',P_0)+C_1$, where $J(\theta,P_0)$ is the denoising loss $J_{DM}$ for Diffusion or $J_{VAE}$ for VAE. The DRO is reformulated as:
$$\min_{w\in W}\max_{\theta\in\Theta}\mathbb{E}_{x\sim P_\theta}[f(w,x)]\quad\text{s.t.}\quad J(\theta,P_0)\le\epsilon$$
Using **inclusive** $D_{KL}(P_0\|P_\theta)$ is critical because it allows $P_0\ll P_\theta$, enabling the generated distribution to exceed the nominal support—breaking the support set constraint of $\phi$-divergence DRO.

**2. Dual Learning for Constrained Inner Maximization.** The inner problem is a constrained optimization on the diffusion parameter space. By introducing the Lagrangian multiplier $\mu>0$, it is transformed into: $\max_\theta \mathbb{E}_{x\sim P_\theta}[f(w,x)]-\mu J(\theta,P_0)$. The multiplier is updated via dual gradient ascent: $\mu\leftarrow\max\{0,\mu+\eta(J(\theta_k,P_0)-\epsilon)\}$. This dynamically balances adversarial strength and consistency.

**3. Policy Optimization for Differentiable Objectives.** The objective $\mathbb{E}_{x\sim P_\theta}[f]$ is not directly differentiable with respect to $\theta$. Treating the diffusion reverse process as a sampling trajectory (policy), the objective is rewritten using policy gradient techniques as $\max_\theta \hat{\mathbb{E}}_{P_\theta(x_{0:T})}[\ln P_\theta(x_{0:T})\cdot f(w,x_0)]-\mu J_{DM}(\theta,P_0)$. To stabilize training, a **PPO**-style objective with a clipping term $\text{clip}(r_\theta, 1-\kappa, 1+\kappa)$ is used. For efficiency, only the final $T'$ steps of the reverse process are optimized.

**4. Convergence Guarantees.** Theorem 1 proves the inner maximization error converges to the optimal oracle at $O(1/\sqrt{K})$. Theorem 2 provides stationary point convergence for the overall objective $\phi(w)=\max_\theta\mathbb{E}_{P_\theta}[f]$. The authors also define **$\Gamma$-expressivity** of the generative model, showing that smaller $\Gamma$ (better approximation of any test distribution) leads to better robustness.

## Key Experimental Results

### Main Results
Task: Electricity carbon emission time-series forecasting (Electricity Maps). Training set: BANC 2324; OOD test sets: different years/regions. Metric: MSE (lower is better).

| Method | Average MSE | Worst MSE | Gain vs ML |
|------|------------|-----------|------------|
| **GAS-DRO** | **0.0163** | **0.0509** | **63.7%** |
| DRAGEN | 0.0230 | 0.0681 | 48.9% |
| P-DRO | 0.0259 | 0.0820 | 42.5% |
| DML | 0.0271 | 0.0834 | 39.7% |
| KL-DRO | 0.0288 | 0.0831 | 36.1% |
| W-DRO | 0.0342 | 0.0879 | 24.0% |
| ML | 0.0450 | 0.0946 | — |

GAS-DRO outperforms all baselines, achieving a 63.7% improvement over standard ML.

### Ablation Study
- **Budget $\epsilon$**: Controls the trade-off between adversarial strength and nominal consistency.
- **Expressivity**: Stronger generative models (smaller $\Gamma$) yield better robustness, validating the theory.
- **Partial Optimization**: Optimizing only the last $T'$ steps significantly reduces computation with minimal performance loss.

### Key Findings
- Traditional DRO methods like W-DRO and KL-DRO are restricted by support sets or conservative approximations.
- Defining the ambiguity set in the parameter space is more effective than fitting generative models into traditional frameworks.
- Robustness gains are consistent across both time-series and image (MNIST to USPS) tasks.

## Highlights & Insights
- **Perspective Shift**: Moving the ambiguity set from "distribution space" to "parameter space" resolves the expressiveness-tractability dilemma.
- **Inclusive KL**: Theoretically justifies why the model can look "beyond" the training support points.
- **Unified Framework**: Combines solid convergence analysis with a practical PPO-based algorithm.

## Limitations & Future Work
- **Dependence on Generative Quality**: Limited expressivity of the base generative model restricts the identification of the true worst-case distribution.
- **Computational Cost**: Training and sampling from diffusion models in the inner loop is more expensive than closed-form DRO.
- **Scalability**: Performance on high-resolution, large-scale image datasets remains to be fully explored.

## Related Work & Insights
- **Standard DRO**: Addresses support limits of KL-DRO and intractability of W-DRO.
- **Generative DRO (DRAGEN, P-DRO)**: Argues that direct parameter space definition is superior to using generative models for latent Wasserstein balls or parameterized KL distributions.
- **Insight**: The strategy of "parameterizing search spaces + RL-based optimization" can be extended to robust reinforcement learning and environment generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to define DRO ambiguity sets in parameter space via inclusive KL.
- **Experimental Thoroughness**: ⭐⭐⭐ Strong results in time-series; image tasks are limited to small-scale datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and comprehensive theoretical-to-algorithmic pipeline.
- **Value**: ⭐⭐⭐⭐ Provides a new, tractable paradigm for OOD generalization and robust learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICLR 2026\] Generative Bayesian Optimization: Generative Models as Acquisition Functions](generative_bayesian_optimization_generative_models_as_acquisition_functions.md)
- [\[ICLR 2026\] Gen-DFL: Decision-Focused Generative Learning for Robust Decision Making](gen-dfl_decision-focused_generative_learning_for_robust_decision_making.md)
- [\[ICLR 2026\] Distributionally Robust Linear Regression with Block Lewis Weights](distributionally_robust_linear_regression_with_block_lewis_weights.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)

</div>

<!-- RELATED:END -->
