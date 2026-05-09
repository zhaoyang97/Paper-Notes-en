---
title: >-
  [Paper Note] Diffusion Models Meet Contextual Bandits
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] This paper proposes diffusion Thompson Sampling (dTS), which employs a pretrained diffusion model as an expressive prior over action parameters in contextual bandit problems. Through an efficient hierarchical posterior approximation, dTS enables fast updates and sampling, significantly outperforming conventional methods in large action spaces.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Contextual Bandits
  - Thompson Sampling
  - Bayesian Prior
  - Posterior Approximation
date: 2026-05-08
content_hash: af9f56e4b34e4897
---

# Diffusion Models Meet Contextual Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2402.10028](https://arxiv.org/abs/2402.10028)
**Code**: Available (GitHub)
**Area**: Image Generation / Diffusion Models / Online Learning
**Keywords**: Diffusion Models, Contextual Bandits, Thompson Sampling, Bayesian Prior, Posterior Approximation

## TL;DR

This paper proposes diffusion Thompson Sampling (dTS), which employs a pretrained diffusion model as an expressive prior over action parameters in contextual bandit problems. Through an efficient hierarchical posterior approximation, dTS enables fast updates and sampling, significantly outperforming conventional methods in large action spaces.

## Background & Motivation

- **Background**: In large-scale action spaces (large $K$), standard exploration strategies such as LinUCB and LinTS face computational or statistical efficiency bottlenecks.
- **Limitations of Prior Work**: Correlations among actions are typically ignored; in practice, actions are often related (e.g., similar movies in recommendation systems), and observing one action can provide information about others.
- **Key Challenge**: Diffusion models excel at approximating complex high-dimensional distributions and can encode structured priors over action parameters.
- **Core Idea**: A diffusion model is pretrained on offline data to capture the distributional structure of action parameters, serving as an informative prior for Thompson sampling, with efficient posterior updates during online interaction.

## Method

### Overall Architecture

#### Problem Formulation

A hierarchical Bayesian model combining contextual bandits with a diffusion model prior:

$$\psi_L \sim \mathcal{N}(0, \Sigma_{L+1})$$
$$\psi_{\ell-1} | \psi_\ell \sim \mathcal{N}(f_\ell(\psi_\ell), \Sigma_\ell), \quad \forall \ell \in [L] \setminus \{1\}$$
$$\theta_a | \psi_1 \sim \mathcal{N}(f_1(\psi_1), \Sigma_1), \quad \forall a \in [K]$$
$$Y_t | X_t, \theta_{A_t} \sim P(\cdot | X_t; \theta_{A_t}), \quad \forall t \in [n]$$

- $\psi_1, \ldots, \psi_L$: $L$ layers of latent variables corresponding to the denoising layers of the diffusion model.
- $f_\ell$: link functions (linear or nonlinear neural networks).
- $\theta_a$: parameter vector for each action.
- The reward distribution follows a generalized linear model (GLM) with mean $g(x^\top \theta_a)$.

#### dTS Algorithm (Algorithm 1)

Hierarchical sampling at each round $t$:
1. Sample the top-level latent variable from the posterior $p(\psi_L | H_t)$.
2. Sample downward layer by layer: $\psi_{\ell-1} | \psi_\ell, H_t$.
3. Given $\psi_1$, independently sample each action parameter $\theta_a | \psi_1, H_{t,a}$.
4. Select $A_t = \arg\max_a r(X_t, a; \theta_t)$.

### Key Designs

#### 1. Two-Level Posterior Approximation

**Problem**: Nonlinear rewards and nonlinear diffusion links render the posterior intractable.

**(i) Likelihood Approximation**: The GLM likelihood is approximated as Gaussian via a Laplace-style approach:

$$\prod_{i \in S_{t,a}} P(Y_i | X_i; \theta_a) \approx \mathcal{N}(\theta_a; \hat{B}_{t,a}, \hat{G}_{t,a}^{-1})$$

where $\hat{B}_{t,a}$ is the MLE and $\hat{G}_{t,a}$ is the Hessian of the negative log-likelihood.

**(ii) Diffusion Approximation**: Based on the closed-form solution for the linear case, the linear term $W_\ell \psi_\ell$ is replaced by the nonlinear $f_\ell(\psi_\ell)$.

#### 2. Closed-Form Posterior Expressions

**Action posterior** (precision-weighted form):

$$\hat{\Sigma}_{t,a}^{-1} = \underbrace{\Sigma_1^{-1}}_{\text{prior precision}} + \underbrace{\hat{G}_{t,a}}_{\text{data precision}}$$

$$\hat{\mu}_{t,a} = \hat{\Sigma}_{t,a}\left(\underbrace{\Sigma_1^{-1} f_1(\psi_1)}_{\text{prior contribution}} + \underbrace{\hat{G}_{t,a} \hat{B}_{t,a}}_{\text{data contribution}}\right)$$

**Latent variable posterior** (computed recursively):

$$\bar{\Sigma}_{t,\ell-1}^{-1} = \Sigma_\ell^{-1} + \bar{G}_{t,\ell-1}, \quad \bar{\mu}_{t,\ell-1} = \bar{\Sigma}_{t,\ell-1}\left(\Sigma_\ell^{-1} f_\ell(\psi_\ell) + \bar{B}_{t,\ell-1}\right)$$

Information propagates upward from the action layer to all latent variable layers via the recursion $\bar{G}_{t,\ell}, \bar{B}_{t,\ell}$.

#### 3. Computational Complexity Advantage

| Method | Time Complexity | Space Complexity |
|--------|----------------|-----------------|
| Joint posterior | $O(K^3 d^3)$ | $O(K^2 d^2)$ |
| **dTS** | $O((L+K)d^3)$ | $O((L+K)d^2)$ |
| LinTS (independent) | $O(Kd^3)$ | $O(Kd^2)$ |

When $K \gg L$, dTS approaches the computational cost of LinTS while exploiting action correlations.

### Loss & Training

- **Offline Pretraining**: The diffusion model learns $f_\ell$ and $\Sigma_\ell$ from historical data.
- **Online Updates**: Each round only requires updating Gaussian statistics (MLE $\hat{B}_{t,a}$, Hessian $\hat{G}_{t,a}$); no retraining of the diffusion model is needed.
- Posterior updates are closed-form matrix operations, ensuring computational efficiency.

## Key Experimental Results

### Main Results: Synthetic Data

Experimental settings: $d \in \{5, 20\}$, $L \in \{2, 4\}$, $K \in \{10^2, 10^4\}$, $n = 5000$

| Setting | dTS vs LinTS | dTS vs HierTS | dTS vs UCB |
|---------|-------------|---------------|------------|
| Linear diffusion + linear reward | Significantly lower regret | Significantly lower | Significantly lower |
| Linear diffusion + nonlinear reward | Significantly lower | — | Significantly lower |
| Nonlinear diffusion + linear reward | Significantly lower | Significantly lower | Significantly lower |
| Nonlinear diffusion + nonlinear reward | Significantly lower | — | Significantly lower |

### Ablation Study

#### Advantage Amplification with Larger $K$

The ratio of cumulative regret between LinTS and dTS increases monotonically as $K$ grows from 10 to $5 \times 10^4$—the larger the action space, the greater the benefit of exploiting action correlations.

#### Parameter Sensitivity

| Parameter | Effect |
|-----------|--------|
| Larger $K$ | Higher regret (more parameters to learn) |
| Larger $d$ | Higher regret |
| Larger $L$ | Higher regret (more latent variables to learn) |

#### Prior Robustness

- With noise added to prior parameters ($v \in \{0.5, 1.0, 1.5\}$), dTS still outperforms baselines (only approximately matching at $v = 1.5$).
- Swiss Roll data (non-diffusion prior): pretraining with $L \approx 40$ is optimal.
- MovieLens (real-world recommendation): dTS significantly outperforms LinTS.

### Key Findings

1. **Latent structure matters more than reward distribution**: dTS using the correct diffusion prior but incorrect reward distribution still outperforms GLM-TS, which uses the correct reward but ignores structural correlations.
2. **Advantage amplifies with action space size**: dTS's margin over baselines is far larger at $K = 10^4$ than at $K = 10^2$.
3. **Robust to prior misspecification**: Effective on Swiss Roll (non-diffusion generative process) and MovieLens (real data).
4. **Few pretraining samples suffice**: As few as 50 pretraining samples enable dTS to more than double its performance advantage over LinTS.

## Highlights & Insights

- **Conceptual Innovation**: The paper reframes diffusion models from "generators" to "Bayesian prior encoders"—a fundamentally new role.
- **Theory-Practice Unity**: The linear case admits exact posteriors and regret bounds; the nonlinear case is approximated naturally from the closed-form solution.
- **Elegance of Two-Level Approximation**: The likelihood approximation preserves the expressiveness of the diffusion hierarchy (vs. a global Laplace approximation); the diffusion approximation leverages the structure of the closed-form solution.
- **Bayes Regret Analysis**: The regret of dTS contains $K\sigma_1^2$ (conditional variance) in the action term, whereas LinTS contains $K\Sigma$ (marginal variance, which is substantially larger).
- **Sparsity Acceleration**: When the mixing matrix has column sparsity, the effective dimension for latent variable learning reduces from $d$ to $d_\ell$.

## Limitations & Future Work

1. **Theory restricted to the linear case**: Bayes regret bounds hold only under linear Gaussian settings; theoretical guarantees for the nonlinear case are absent.
2. **Approximation errors unquantified**: The errors introduced by the likelihood approximation and diffusion approximation are not bounded.
3. **Dependence on pretraining quality**: Insufficient or biased offline data may lead to an under-regularized prior.
4. **Static bandit assumption**: Action parameters are assumed fixed; extension to non-stationary settings remains open.
5. **No matching lower bound**: A Bayesian lower bound remains an open problem.

## Related Work & Insights

- **HierTS** (Hong et al., 2022): Hierarchical Bayesian bandits, but limited to linear priors; dTS generalizes to nonlinear diffusion priors.
- **Hsieh et al. (2023)**: Multi-armed bandits with diffusion priors; this paper is the first contextual bandit extension.
- **Kveton et al. (2024)**: Concurrent work in a similar direction.
- **Broader Insight**: The paradigm of using pretrained generative models as structured priors can be extended to reinforcement learning, Bayesian optimization, and other online decision-making problems.

## Rating

| Dimension | Score | Comment |
|-----------|-------|---------|
| Novelty | ★★★★★ | First use of diffusion models as priors for contextual bandits; conceptually original. |
| Technical Depth | ★★★★☆ | Rigorous posterior derivation; complete theoretical analysis for the linear case. |
| Experimental Thoroughness | ★★★★☆ | Synthetic and real data with multi-dimensional ablations; large-scale scenarios are limited. |
| Value | ★★★★☆ | Direct applicability to recommendation systems, advertising, and other large action-space settings. |
| Writing Quality | ★★★★☆ | Clear structure with detailed derivations; somewhat lengthy. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contextual Thompson Sampling via Generation of Missing Data](contextual_thompson_sampling_via_generation_of_missing_data.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](when_are_concepts_erased_from_diffusion_models.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
