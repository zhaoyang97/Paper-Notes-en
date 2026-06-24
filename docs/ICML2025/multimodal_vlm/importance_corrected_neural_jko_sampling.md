---
title: >-
  [Paper Note] Importance Corrected Neural JKO Sampling
description: >-
  [ICML 2025][Multimodal VLM][Neural JKO] Proposes Importance Corrected Neural JKO Sampling (Neural JKO IC), which alternates between the local JKO steps of continuous normalizing flows (CNFs) and rejection resampling steps based on importance weights. This approach overcomes the local optima issues of Wasserstein gradient flows on multimodal distributions while maintaining independent and identically distributed (i.i.d.) sampling and density tractability.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Neural JKO"
  - "Rejection Sampling"
  - "Continuous Normalizing Flows"
  - "Importance Sampling"
  - "Multimodal Distributions"
date: 2026-05-08
content_hash: fb7838be385c5551
---

# Importance Corrected Neural JKO Sampling

**Conference**: ICML 2025  
**arXiv**: [2407.20444](https://arxiv.org/abs/2407.20444)  
**Code**: [github.com/johertrich/neural_JKO_ic](https://github.com/johertrich/neural_JKO_ic)  
**Area**: Sampling Methods, Wasserstein Gradient Flows, Normalizing Flows  
**Keywords**: Neural JKO, Rejection Sampling, Continuous Normalizing Flows, Importance Sampling, Multimodal Distributions

## TL;DR

Proposes Importance Corrected Neural JKO Sampling (Neural JKO IC), which alternates between the local JKO steps of continuous normalizing flows (CNFs) and rejection resampling steps based on importance weights. This approach overcomes the local optima issues of Wasserstein gradient flows on multimodal distributions while maintaining independent and identically distributed (i.i.d.) sampling and density tractability.

## Background & Motivation

Sampling from unnormalized probability density functions is a core problem in machine learning. Given an integrable function $g: \mathbb{R}^d \to \mathbb{R}_{>0}$, the goal is to sample from $q(x) = g(x)/Z_g$, where the normalization constant $Z_g$ is unknown.

Traditional methods fall into two categories:
- **MCMC methods** (e.g., Langevin, HMC): Based on local transitions, **struggle to correctly allocate mass across multimodal distributions**
- **Generative models** (e.g., normalizing flows, diffusion models): Optimize reverse KL divergence, but this objective is non-convex when the target distribution is non-log-concave, making them **prone to mode collapse**

Neural JKO connects CNF training with the JKO scheme through a regularized velocity field, enhancing stability. However, the underlying Wasserstein gradient flows are still plagued by non-convexity, experiencing slow convergence or becoming trapped in sub-optimal local minima on multimodal targets.

## Method

### Overall Architecture

Neural JKO IC alternates between two steps:
1. **Neural JKO step**: Locally adjusts sample positions using a CNF.
2. **Importance rejection step**: Non-locally corrects the inferred distribution based on importance weights.

### Neural JKO Scheme

Parameterizes the Wasserstein proximal operator of the JKO scheme using a neural ODE:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mu_\tau^k}\left[-\log(g(z_\theta(x,\tau))) - \ell_\theta(x,\tau) + \omega_\theta(x,\tau)\right]$$

where $(z_\theta, \ell_\theta, \omega_\theta)$ satisfy the ODE system:
- $\dot{z}_\theta = v_\theta(z_\theta, t)$ (position evolution)
- $\dot{\ell}_\theta = \text{trace}(\nabla v_\theta)$ (density change)
- $\dot{\omega}_\theta = \|v_\theta\|^2$ (regularization term)

### Theoretical Guarantees

**Theorem 3.3**: Proves that the sequence of velocity fields $v_{\tau_l}$ of the JKO scheme **strongly converges** to the velocity field of the Wasserstein gradient flow as $\tau_l \to 0$.

### Importance Rejection Step

Given the current approximate distribution $\mu$ (density $p$) and target distribution $\nu$ (density $q$), the acceptance probability for a sample $X \sim \mu$ is computed as:

$$\alpha(X) = \min\left\{1, \frac{g(X)}{c \cdot f(X)}\right\}$$

The sample is retained with probability $\alpha(X)$, and otherwise replaced by resampling from $\mu$.

**Key conclusion of Theorem 4.2**:
1. The density after the rejection step, $\tilde{p}(x) = p(x)(\alpha(x) + 1 - \mathbb{E}[\alpha(X)])$, **can be explicitly computed**.
2. $\text{KL}(\tilde{\mu}, \nu) \leq \text{KL}(\mu, \nu)$, meaning **the KL divergence decreases monotonically at each step**.

The hyperparameter $c$ is determined via binary search to ensure that approximately $r = 20\%$ of the samples are resampled.

### Density Tractability

Unlike MCMC methods, Neural JKO IC can track the density values of samples at each step, enabling:
- Independent and identically distributed (i.i.d.) sampling
- Density evaluation

## Key Experimental Results

### Main Results: Energy Distance

| Distribution | MALA | HMC | DDS | CRAFT | Neural JKO | **Neural JKO IC** |
|------|------|-----|-----|-------|---------|-----------------|
| Mustache | 4.6e-2 | 1.7e-2 | 6.9e-2 | 9.2e-2 | 1.8e-2 | **2.9e-3** |
| Shifted 8 Modes | 5.3e-3 | 4.1e-5 | 1.2e-2 | 5.2e-2 | 1.3e-1 | **1.2e-5** |

### Key Findings

- On high-dimensional multimodal targets (up to $d = 1600$), Neural JKO IC **significantly outperforms** existing methods across almost all tested distributions.
- Pure Neural JKO fails completely on the Shifted 8 Modes setup (energy distance of 0.13), while incorporating the importance correction reduces the error by 4 orders of magnitude to 1.2e-5.
- The rejection step effectively corrects the incorrect allocation of mode weights caused by the gradient flow.

## Highlights & Insights

1. **Combining Local and Non-local Steps**: CNFs perform local adjustments while rejection sampling executes non-local corrections, complementing each other.
2. **Tractable Density**: Unlike MCMC, which generates non-independent samples with unknown density, Neural JKO IC produces i.i.d. samples with tractable densities.
3. **Theoretical Rigor**: Provides proofs for the strong convergence of velocity fields and the monotonic decrease of the KL divergence.
4. **Self-generated Proposals**: The proposal distribution for the rejection step is generated by the model itself, avoiding the sensitivity to proposals typical of classical rejection sampling.

## Limitations & Future Work

- In high-dimensional settings, importance weights can become highly unbalanced, resulting in an increased rejection rate.
- Requires evaluation of the unnormalized density $g(x)$, making it inapplicable to purely data-driven scenarios.
- A new CNF must be trained at every JKO step, leading to relatively high overall computational overhead.
- Theoretical convergence requires the target function to satisfy conditions such as $\lambda$-convexity.

## Related Work & Insights

- MCMC (MALA, HMC)
- Sequential Monte Carlo (SMC)
- Neural ODE / Continuous Normalizing Flows (CNF)
- OT-flow (regularized velocity fields)
- Stochastic Normalizing Flows
- Stein Variational Gradient Descent (SVGD)

## Rating

⭐⭐⭐⭐⭐ — Solid theoretical contributions (proof of the strong convergence of velocity fields), extensive experiments, and significant improvements on multimodal high-dimensional targets. The methodology is elegantly designed, unifying the advantages of gradient flows and rejection sampling into a single framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization](../../ICLR2026/multimodal_vlm/importance_sampling_for_multi-negative_multimodal_direct_preference_optimization.md)
- [\[ICCV 2025\] Evading Data Provenance in Deep Neural Networks](../../ICCV2025/multimodal_vlm/evading_data_provenance_in_deep_neural_networks.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/multimodal_vlm/conditional_diffusion_sampling.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](../../ICML2026/multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems](../../NeurIPS2025/multimodal_vlm/hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)

</div>

<!-- RELATED:END -->
