---
title: >-
  [Paper Note] Federated ADMM from Bayesian Duality
description: >-
  [ICLR 2026][ADMM] This paper derives a Bayesian dual structure for ADMM from a variational Bayes (VB) perspective, proving that classical ADMM is a special case of VB over isotropic Gaussian families. Two novel extension…
tags:
  - "ICLR 2026"
  - "ADMM"
  - "Variational Bayes"
  - "Natural Gradient"
  - "Federated Learning"
  - "Bayesian Duality"
date: 2026-05-08
content_hash: 76576fb2109b5dd4
---

# Federated ADMM from Bayesian Duality

**Conference**: ICLR 2026
**arXiv**: [2506.13150](https://arxiv.org/abs/2506.13150)  
**Code**: [Available](https://github.com/xxx)  
**Area**: Others
**Keywords**: ADMM, Variational Bayes, Natural Gradient, Federated Learning, Bayesian Duality

## TL;DR
This paper derives a Bayesian dual structure for ADMM from a variational Bayes (VB) perspective, proving that classical ADMM is a special case of VB over isotropic Gaussian families. Two novel extensions are introduced: a Newton-like variant (one-round convergence on quadratic objectives) and an Adam-like variant (IVON-ADMM, achieving +7% accuracy in heterogeneous deep learning settings).

## Background & Motivation

### State of the Field

**Background**: ADMM has served as a core algorithmic backbone for federated learning since its introduction in the 1970s, with its form remaining largely unchanged. Its robust structure naturally raises the question of whether a more general formulation exists.

**Limitations of Prior Work**: Accelerated variants of ADMM (over-relaxation, momentum, scaled norms, etc.) introduce additional variables without altering the algorithmic form. Swaroop et al. observed line-by-line similarities between VB and ADMM but could not establish an exact correspondence.

**Key Challenge**: The deterministic optimization framework underlying ADMM does not extend naturally to heterogeneous deep learning scenarios. A more general framework is needed to unify and generalize ADMM.

**Key Insight**: The key insight is that solutions to the VB objective exhibit a dual structure that not only resembles the fixed-point structure of ADMM but also naturally generalizes it. The critical missing link is the **natural gradient**.

**Core Idea**: The duality between natural parameters and expectation parameters in exponential family distributions establishes a "Bayesian duality" structure, of which ADMM is a special case under isotropic Gaussians.

### Mechanism

**Goal**: ### Overall Architecture
Classical ADMM: primal-dual structure over $(\theta_g^*, \theta_k^*, \mathbf{v}_k^*, \mathbf{v}_g^*)$; Bayesian ADMM: expectation-natural parameter dual structure over $(\mu_g^*, \mu_k^*, \eta_k^*, \lambda_g^*)$.

## Method

### Overall Architecture
Classical ADMM operates with a primal-dual structure over $(\theta_g^*, \theta_k^*, \mathbf{v}_k^*, \mathbf{v}_g^*)$; Bayesian ADMM adopts an expectation-natural parameter dual structure over $(\mu_g^*, \mu_k^*, \eta_k^*, \lambda_g^*)$. The core distinction is: gradient $\to$ natural gradient, parameters $\to$ distributions.

### Key Designs

1. **Bayesian Dual Structure**:

    - VB fixed-point condition: $\lambda_g^* = -\sum_{k=0}^{K} \nabla \mathcal{L}_k(\mu_g^*)$
    - Introducing local distributions $q_k^*$ and dual variables $\eta_k^*$ yields a four-condition structure analogous to ADMM
    - When $q$ is chosen as an isotropic Gaussian, the natural gradient reduces to the ordinary gradient, recovering classical ADMM

2. **Newton-like Extension (Full-Covariance Gaussian)**:

    - **Function**: $q$ is parameterized as a full-covariance Gaussian distribution
    - **Mechanism**: The natural gradient incorporates the inverse Fisher information matrix, which is equivalent to Newton's method on quadratic objectives, enabling one-round communication convergence
    - **Design Motivation**: Classical ADMM requires multiple rounds of iteration even on quadratic objectives

3. **Adam-like Extension (Diagonal Gaussian, IVON-ADMM)**:

    - **Function**: $q$ is parameterized as a diagonal-covariance Gaussian, implemented efficiently via the IVON method
    - **Mechanism**: Diagonal Fisher approximation produces adaptive learning rates analogous to Adam
    - **Design Motivation**: Full Fisher information is computationally prohibitive; diagonal approximation is more practical for deep learning

### Loss & Training
- Client: minimizes local loss plus KL regularization (Bayesian formulation)
- Server: aggregates natural gradient parameters rather than raw gradients

## Key Experimental Results

### Main Results
Deep heterogeneous federated learning:

| Method | Accuracy | Runtime | Notes |
|--------|----------|---------|-------|
| FedADMM | Baseline | Baseline | Classical ADMM |
| FedAvg | Baseline-level | Baseline-level | Standard federated |
| IVON-ADMM | **+7%** | **Comparable** | Adam-like extension |

### Theoretical Validation (Quadratic Objectives)

| Method | Rounds to Converge | Notes |
|--------|--------------------|-------|
| Classical ADMM | Multiple rounds | Linear convergence |
| Newton-like ADMM | **1 round** | One-step convergence |

### Key Findings
- IVON-ADMM achieves +7% accuracy in deep heterogeneous settings (non-IID data) without additional communication or computational overhead
- The Newton-like variant achieves one-round convergence on quadratic objectives, confirming the theoretical prediction
- The natural gradient is the key link connecting VB and ADMM — precisely the missing element identified in Swaroop et al.

## Highlights & Insights
- **Mathematical Elegance**: Classical ADMM turns out to be a special case of a Bayesian method under the simplest distribution family. This connection is not only aesthetically appealing but also opens a new avenue for generalizing optimization algorithms via families of probability distributions.
- **The Central Role of Natural Gradients**: Prior work using ordinary gradients failed to establish an exact correspondence; substituting natural gradients resolves this immediately, underscoring the deep role of information geometry in algorithm design.
- **A Free Lunch**: IVON-ADMM leverages IVON's efficient diagonal Fisher implementation, incurring no additional runtime while substantially improving performance in heterogeneous settings.

## Limitations & Future Work
- Deep learning experiments are conducted at a relatively small scale (7-layer CNN); performance on larger models (e.g., LLMs) remains unknown
- Diagonal Fisher approximation may be insufficiently accurate for certain model architectures
- The framework requires selecting an exponential family distribution as a prior assumption; guidelines for choosing the distribution are not clearly specified
- Communication efficiency analysis is relatively straightforward and could be further developed

## Related Work & Insights
- **vs. FedADMM**: Classical ADMM is recovered as a special case; Bayesian ADMM provides a rigorous generalization
- **vs. FedAvg**: IVON-ADMM demonstrates significant advantages in heterogeneous settings
- **vs. PVI (Swaroop 2025)**: This work resolves the missing exact correspondence between PVI and ADMM

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Bayesian dual structure is original and elegant, unifying two major paradigms
- Experimental Thoroughness: ⭐⭐⭐ Theoretical validation is rigorous, but deep learning experiments are limited in scale
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and the duality diagrams are intuitive
- Value: ⭐⭐⭐⭐ Provides a new theoretical foundation and practical extensions for federated optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/others/singular_bayesian_neural_networks.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](../../AAAI2026/others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)

</div>

<!-- RELATED:END -->
