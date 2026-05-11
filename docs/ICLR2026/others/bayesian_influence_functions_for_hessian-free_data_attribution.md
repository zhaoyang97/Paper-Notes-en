---
title: >-
  [Paper Note] Bayesian Influence Functions for Hessian-Free Data Attribution
description: >-
  [ICLR 2026][influence functions] This paper proposes the Local Bayesian Influence Function (BIF), which replaces the intractable Hessian inversion in classical influence functions with a covariance estimate obtained via…
tags:
  - "ICLR 2026"
  - "influence functions"
  - "training data attribution"
  - "Bayesian inference"
  - "SGMCMC"
  - "Hessian-free"
  - "data valuation"
date: 2026-05-08
content_hash: 7cf65e3081d7986c
---

# Bayesian Influence Functions for Hessian-Free Data Attribution

**Conference**: ICLR 2026
**arXiv**: [2509.26544](https://arxiv.org/abs/2509.26544)
**Code**: None
**Area**: Other
**Keywords**: influence functions, training data attribution, Bayesian inference, SGMCMC, Hessian-free, data valuation

## TL;DR
This paper proposes the Local Bayesian Influence Function (BIF), which replaces the intractable Hessian inversion in classical influence functions with a covariance estimate obtained via SGLD sampling, enabling architecture-agnostic data attribution for models with billions of parameters and achieving state-of-the-art performance on retraining experiments.

## Background & Motivation
**Background**: Training Data Attribution (TDA) studies how training data shapes model behavior, forming a foundational problem in AI interpretability and safety. Classical Influence Functions (IF) measure the impact of data points via the inverse Hessian.

**Limitations of Prior Work**: (a) The Hessian of deep neural networks is typically degenerate (non-invertible), violating the theoretical premise of classical IF; (b) Direct Hessian computation is infeasible for large models, and approximation methods such as EK-FAC introduce structural bias while supporting only Linear/Conv2D layers; (c) Fine-grained per-token attribution requires serial computation over tokens in classical methods and does not scale.

**Key Challenge**: A data attribution method is needed that is both theoretically sound (without relying on Hessian invertibility) and computationally feasible (scalable to billions of parameters).

**Goal**: Can a Bayesian inference framework entirely bypass the Hessian while maintaining or surpassing the attribution quality of classical IF?

**Key Insight**: Replace the inverse Hessian in classical IF with a covariance estimate over a local posterior distribution, realized via SGLD sampling.

**Core Idea**: $\text{BIF}(z_i, \phi) = -\text{Cov}_\gamma(\ell_i(\boldsymbol{w}), \phi(\boldsymbol{w}))$ — influence is defined as the negative covariance between the training loss and the target observable under the local posterior.

## Method

### Overall Architecture
Classical influence functions are reformulated from a point-estimation problem to a distributional one: rather than computing gradients and the Hessian at a single optimal parameter $\boldsymbol{w}^*$, covariances are estimated over a local posterior distribution centered at $\boldsymbol{w}^*$.

### Key Designs

1. **Theoretical Derivation from IF to BIF**:

    - Classical IF: $\text{IF}(z_i, \phi) = -\nabla\phi(\boldsymbol{w}^*)^\top \boldsymbol{H}^{-1} \nabla\ell_i(\boldsymbol{w}^*)$ (requires inverse Hessian)
    - Bayesian IF: $\text{BIF}(z_i, \phi) = -\text{Cov}(\ell_i(\boldsymbol{w}), \phi(\boldsymbol{w}))$ (standard result from statistical physics)
    - Key equivalence: Under non-degenerate models, the first-order Taylor expansion of BIF is equivalent to classical IF (Appendix A)

2. **Localization Mechanism (Local BIF)**:

    - An isotropic Gaussian prior centered at $\boldsymbol{w}^*$ is introduced: $p_\gamma \propto \exp(-\sum \ell_i - \frac{\gamma}{2}\|\boldsymbol{w}-\boldsymbol{w}^*\|^2)$
    - The precision parameter $\gamma$ controls locality, analogous to Hessian dampening $(\boldsymbol{H} + \gamma\boldsymbol{I})$ in classical IF
    - Local BIF serves as a higher-order generalization of dampened IF

3. **SGLD Sampling Estimation**:

    - Stochastic Gradient Langevin Dynamics is used to sample from the local posterior
    - Multiple independent SGLD chains improve coverage, yielding $N_{\text{draws}} = C \times T$ total samples
    - Each step requires only a forward pass to compute training and query losses — no backward pass through the network is needed for attribution

4. **Per-Token Attribution**:

    - In autoregressive models, the loss decomposes over tokens: $\ell_i = \sum_s \ell_{i,s}$
    - BIF naturally supports token-level covariance: $\text{BIF}(z_{i,s}, z_{j,s'}) = -\text{Cov}_\gamma(\ell_{i,s}, \ell_{j,s'})$
    - A single sampling run enables parallel computation of the full token-token influence matrix, whereas classical methods require serial computation per token

5. **Normalized BIF (Posterior Correlation)**:

    - Raw covariance is dominated by high-variance data points
    - Normalization to the Pearson correlation coefficient yields values in $[-1, 1]$, providing greater stability and comparability

## Key Experimental Results

### Table 1: Complexity Comparison of Local BIF vs. EK-FAC

| Dimension | Local BIF | EK-FAC |
|-----------|-----------|--------|
| Time Complexity | $O(N_{\text{draws}}(n+q)d)$, no fit phase | Fit: $O(N_{\text{fit}}d + \sum d_\ell^3)$; Score: $O(nqd)$ |
| Additional Storage | $O(N_{\text{draws}}(n+q))$ loss trajectories | $O(\sum(d_{\text{in},l}^2 + d_{\text{out},l}^2))$ factors |
| Error Source | Finite sampling + SGLD bias | Finite sampling + structural bias (Kronecker/Fisher) |
| Architecture Support | Any differentiable model | Linear and Conv2D layers only |

### Table 2: LDS Scores on CIFAR-10 Retraining Experiment

| Data Scale $\alpha$ | BIF | EK-FAC | TRAK | GradSim |
|---------------------|-----|--------|------|---------|
| Small dataset (high variance) | **Slightly better than EK-FAC** | SOTA baseline | Substantially below BIF/EK-FAC | Lowest |
| Large dataset | Comparable to EK-FAC (within error margin) | SOTA | Below both | Lowest |
| Pythia-14M fine-tuning | Below EK-FAC | Better | — | — |

> On ResNet-9/CIFAR-10, both BIF and EK-FAC achieve LDS scores substantially higher than TRAK and GradSim. BIF shows a slight advantage in the small-data, high-variance regime.

### Scalability Analysis (Pythia Model Family)
- On Pythia-2.8B, BIF evaluation is approximately **2 orders of magnitude** faster than EK-FAC
- EK-FAC incurs significant upfront fitting costs (storing Kronecker factors and performing eigendecompositions), whereas BIF has no such overhead
- GPU memory usage is comparable for both methods (4×A100 nodes)
- The advantage of BIF over EK-FAC becomes more pronounced as model scale increases

## Highlights & Insights
- **Conceptual Elegance**: Reformulating Hessian inversion as covariance estimation unifies the entire method under a single expression $-\text{Cov}(\ell_i, \phi)$ — theoretically clean and intuitively straightforward to implement.
- **Practical Breakthrough for Per-Token Attribution**: Per-token computation is infeasible for classical methods, whereas BIF's batch covariance is naturally parallel, making token-level attribution for LLMs practically viable.
- **Semantic Quality**: Figure 2 demonstrates that posterior correlations between tokens on Pythia-2.8B capture semantic relationships such as translation, synonymy, and numeral-spelling correspondences, yielding strong qualitative results.
- **Statistical Physics Perspective**: BIF shares deep connections with susceptibilities, local learning coefficients, and other quantities from Singular Learning Theory (SLT), situating it as an important component of the developmental interpretability research agenda.
- **Broad Generality**: The method imposes no constraints on layer type, making it applicable to any differentiable architecture, including attention and normalization layers that EK-FAC does not support.

## Limitations & Future Work
1. **Sampling Quality Dependence**: The accuracy of BIF depends on the quality of SGLD sampling from the local posterior; the singular loss landscapes of DNNs may invalidate standard SGLD convergence guarantees.
2. **Hyperparameter Sensitivity**: There is no theoretical guidance for selecting the optimal step size $\epsilon$, localization strength $\gamma$, and inverse temperature $\beta$, particularly in language model settings.
3. **Sequence-Level Attribution Underperforms EK-FAC**: In the Pythia-14M fine-tuning setting, BIF achieves a lower LDS than EK-FAC, reflecting the difficulty of posterior sampling for LLMs.
4. **Computational Cost at Scale**: Although no fit phase is required, each SGLD draw necessitates a forward pass over the full training and query sets, which remains substantial when the attribution dataset is very large.
5. **No Theoretical Convergence Rate**: A rigorous analysis of how the sampling error of BIF decays with $N_{\text{draws}}$ is currently absent.

## Related Work & Insights
- **Classical Influence Functions**: Originally proposed by Cook (1977), introduced to deep learning by Koh & Liang (2020), and extended with the EK-FAC approximation (current SOTA) by Grosse et al. (2023).
- **Gradient Similarity Methods**: TRAK (Park et al., 2023b) approximates attribution via representation-space similarity; GradSim uses gradient inner products directly.
- **Distributional TDA**: Mlodozeniec et al. (2025) propose the d-TDA framework, of which BIF can be viewed as a mean-shift special case.
- **Bayesian Infinitesimal Jackknife**: Giordano & Broderick (2024) propose a similar idea in the analysis of Bayesian models, without localization or scaling to large LLMs.
- **Singular Learning Theory**: Lau et al. (2025) propose SGMCMC-based estimation of the local learning coefficient; the present paper adopts the same localization mechanism.

## Rating
- **Novelty**: ★★★★★ — The theoretical insight of replacing the inverse Hessian with covariance estimation is elegant and profound; the definition of Local BIF is natural and broadly applicable.
- **Practicality**: ★★★★☆ — Architecture-agnostic for large models with high practical value for per-token attribution, though hyperparameter tuning still relies on empirical judgment.
- **Theoretical Depth**: ★★★★☆ — The asymptotic equivalence between BIF and classical IF is rigorously proved, but theoretical guarantees for sampling convergence and hyperparameter selection are lacking.
- **Experimental Thoroughness**: ★★★★☆ — Covers vision (CIFAR-10, ImageNet) and language (Pythia series) models with both qualitative and quantitative evaluations, though retraining experiments are conducted at relatively small scale.
- **Writing Quality**: ★★★★★ — The derivation from IF to BIF is clearly structured, Figures 1–2 provide intuitive and compelling visualizations, and Table 1 comparing the method against baselines is highly informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](../../NeurIPS2025/others/position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[ICLR 2026\] Federated ADMM from Bayesian Duality](federated_admm_from_bayesian_duality.md)
- [\[ICLR 2026\] On the Impact of the Utility in Semivalue-based Data Valuation](on_the_impact_of_the_utility_in_semivalue-based_data_valuation.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)

</div>

<!-- RELATED:END -->
