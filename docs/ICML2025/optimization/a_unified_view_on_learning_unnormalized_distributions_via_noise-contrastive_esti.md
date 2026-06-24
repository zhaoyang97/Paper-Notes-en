---
title: >-
  [Paper Note] A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation
description: >-
  [ICML 2025][Optimization][noise-contrastive estimation] Proposes two estimator families, alpha-CentNCE and f-CondNCE, based on f-NCE, to unify methods for learning unnormalized distributions (such as MLE, MC-MLE, GlobalGISO, pseudo-likelihood, and ISO), corrects the misleading connection between CondNCE and score matching, and establishes the first finite-sample convergence guarantees for bounded exponential families.
tags:
  - "ICML 2025"
  - "Optimization"
  - "noise-contrastive estimation"
  - "unnormalized distributions"
  - "energy-based models"
  - "exponential families"
  - "convergence rates"
date: 2026-05-08
content_hash: a0bc299bc29260ba
---

# A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation

**Conference**: ICML 2025  
**arXiv**: [2409.18209](https://arxiv.org/abs/2409.18209)  
**Code**: None  
**Area**: Optimization / Statistical Estimation  
**Keywords**: noise-contrastive estimation, unnormalized distributions, energy-based models, exponential families, convergence rates

## TL;DR
Proposes two estimator families, alpha-CentNCE and f-CondNCE, based on f-NCE, to unify methods for learning unnormalized distributions (such as MLE, MC-MLE, GlobalGISO, pseudo-likelihood, and ISO), corrects the misleading connection between CondNCE and score matching, and establishes the first finite-sample convergence guarantees for bounded exponential families.

## Background & Motivation

**Background**: Unnormalized distributions (energy-based models) $\phi_\theta(x)$ are widely used in generative modeling, density estimation, and reinforcement learning. Since the partition function (normalization constant) $Z(\theta) = \int\phi_\theta(x)dx$ is intractable, alternative methods such as NCE, score matching, MC-MLE, and contrastive divergence have been proposed.  
**Limitations of Prior Work**: These methods were proposed independently by different communities: NCE (machine learning), pseudo-likelihood (statistics), and ISO (graphical models), lacking systematic comparison and a unified understanding in the literature. More seriously, Ceylan & Gutmann (2018) claimed that CondNCE converges to score matching in the small-noise limit, a connection that might be misleading.  
**Key Challenge**: Seemingly distinct estimators are actually special cases of the same underlying principle, yet a unified perspective revealing these connections is missing.  
**Goal**: Unify various methods through the NCE framework and establish rigorous theoretical guarantees.  
**Key Insight**: f-NCE is based on density ratio estimation using Bregman divergence, where different generating functions $f$ and noise distributions $q_n$ yield different estimators.  
**Core Idea**: An alpha-centering transform (an alpha-dependent normalization method) combined with conditional noise contrastive estimation to unify and extend existing methods.

## Method

### Overall Architecture
Starting from the f-NCE objective, its core is to fit the model density ratio $\rho_\theta(x) = \phi_\theta(x)/[\nu q_n(x)]$ to the true density ratio $q_d(x)/[\nu q_n(x)]$ using the Bregman divergence $\Delta_f$: $\mathcal{L}_f^{\text{nce}} = -\frac{1}{\nu}\mathbb{E}_{q_d}[f'(\rho_\theta)] + \mathbb{E}_{q_n}[\rho_\theta f'(\rho_\theta) - f(\rho_\theta)]$. Two families of variants are derived based on this formulation.

### Key Designs

1. **alpha-CentNCE**:
    - **Function**: Converts NCE into a framework unifying different estimators via an alpha-dependent normalization method.
    - **Mechanism**: Defines the alpha-centered model $\tilde{\phi}_{\theta;\alpha}(x) = \phi_\theta(x)/Z_\alpha(\theta)$, where $Z_\alpha(\theta) = (\mathbb{E}_{q_n}[(\phi_\theta/q_n)^\alpha])^{1/\alpha}$. Substituting this into the $f_\alpha$-NCE objective yields $\mathcal{L}_\alpha^{\text{cent}} = \mathbb{E}_{q_d}[\rho_\theta^{\alpha-1}]\cdot(\mathbb{E}_{q_n}[\rho_\theta^\alpha])^{(1-\alpha)/\alpha}/(1-\alpha)$.
    - **Design Motivation**: When alpha=1, $Z_1(\theta)=\int\phi_\theta dx$, representing standard normalization $\to$ MLE; when alpha=0, geometric-mean normalization $\to$ GlobalGISO; intermediate alpha values provide a continuous trade-off between bias and variance.

2. **f-CondNCE**:
    - **Function**: Replaces the global noise $q_n$ with a conditional noise distribution $\pi(y|x)$ to avoid the challenge of choosing a proper global noise distribution.
    - **Mechanism**: Compares the joint distribution $q_d(x)\pi(y|x)$ vs. $q_d(y)\pi(x|y)$, simplifying the density ratio to $\rho_\theta(x,y) = \phi_\theta(x)/\phi_\theta(y)$ (under a symmetric channel). **Crucial Correction**: While the population objective indeed approaches score matching as $\epsilon\to 0$ (Theorem 3.3), the $O(\epsilon)$ term dominates in the empirical objective, causing the variance to diverge at a rate of $1/\epsilon$ (Theorem 3.4) unless the number of conditional samples grows as $1/\epsilon^2$.
    - **Design Motivation**: Corrects the misleading connection in literature showing that f-CondNCE is not equivalent to score matching under finite samples.

3. **Finite-Sample Convergence Guarantees (Exponential Families)**:
    - **Function**: Establishes the first finite-sample convergence rates for alpha-CentNCE and f-CondNCE.
    - **Mechanism**: Utilizes the analytical framework of GlobalGISO (Shah et al., 2023) and proves through Theorem 3.2 that the $f_\alpha$-NCE and alpha-CentNCE estimators are equivalent (sharing global optima), thereby unifying the analysis of all variants. For bounded exponential families $\phi_\theta(x) = \exp(\langle\theta,\psi(x)\rangle)$, a sample complexity of $n = O(p^2\log p/\epsilon^2)$ guarantees a parameter estimation error $\leq\epsilon$.
    - **Design Motivation**: Prior to this work, almost all NCE variants lacked finite-sample guarantees; the unified analytical framework allows a single proof to cover all variants.

### Loss & Training
- Standard f-NCE requires optimizing the augmented parameters $\underline{\theta} = (\theta, c)$, where $c$ compensates for normalization.
- alpha-CentNCE automatically handles normalization (alpha-centering absorbs the constant), and Fisher consistency only requires $c\phi_{\theta^\star} = q_d$ (scale recovery).
- f-CondNCE also automatically eliminates the constant factor, as the constant cancels out in the density ratio $\phi_\theta(x)/\phi_\theta(y)$.
- The support of the noise distribution $q_n$ must cover the support of the data distribution: $\text{supp}(q_d) \subset \text{supp}(q_n)$.

## Key Experimental Results

### Main Results: Unified Mapping of Methods

| Estimator | NCE Perspective | $\alpha$ Value | Original Proposing Community |
|--------|---------|-----------|------------|
| MLE | $\alpha$-CentNCE | $\alpha=1$, $Z_1$ tractable | Statistics (Fisher, 1922) |
| MC-MLE | $\alpha$-CentNCE | $\alpha=1$, $Z_1$ estimated via sampling | Computational Statistics (Geyer, 1994) |
| GlobalGISO | $\alpha$-CentNCE | $\alpha=0$ | Graphical Models (Shah et al., 2023) |
| Pseudo-likelihood | Local $\alpha$-CentNCE | $\alpha=1$ | MRF (Besag, 1975) |
| ISO/GISO | Local $\alpha$-CentNCE | $\alpha=0$ | MRF (Vuffray et al., 2016) |
| InvIS | $f_0$-NCE | — | NCE (Pihlaja et al., 2010) |
| eNCE | $f_{1/2}$-NCE | — | NCE (Liu et al., 2021) |
| IS | $f_1$-NCE | — | NCE (Pihlaja et al., 2010) |

### Ablation Study: f-CondNCE vs Score Matching

| Condition | CondNCE Behavior | Score Matching Behavior |
|------|------------|-------------------|
| Population (Infinite samples) | $\mathcal{L}_f^{\text{cond}} = -f(1) + f''(1)\mathcal{L}^{\text{sm}}\epsilon^2 + o(\epsilon^2)$ | Consistent |
| Empirical ($K$ conditional samples) | $O(\epsilon)$ term dominates, variance $\propto 1/\epsilon$ | Does not diverge |
| $\epsilon\to 0$, $K$ fixed | **Diverges** | Converges |
| $\epsilon\to 0$, $K \propto 1/\epsilon^2$ | Converges to SM | Converges |

### Key Findings
- alpha-CentNCE reveals the intrinsic connection among 5+ estimators—transitioning from MLE to GlobalGISO requires only changing the $\alpha$ value.
- The equivalence in Theorem 3.2 (optimal solution of $f_\alpha$-NCE = optimal solution of alpha-CentNCE) is the key bridge to unification.
- The non-convergence of f-CondNCE to score matching under finite samples is an important negative result—Theorem 3.4 indicates that the variance of the $O(\epsilon)$ term in the empirical objective diverges as $\epsilon\to 0$, and the variance of its estimator cannot converge with analytical tricks.
- Different alpha values provide a continuous trade-off between bias and variance—larger alpha leads to lower bias but higher variance (variance-bias duality in partition function estimation).
- Local versions (leveraging MRF graph structures) exhibit better statistical efficiency than global versions, as local density ratios only involve low-dimensional conditional distributions.

## Highlights & Insights
- The mathematical construction of alpha-centering is exceptionally elegant—by normalizing with Lp means under different alpha values, a continuous parameter connects a complete spectrum ranging from exact normalization (MLE) to geometric mean normalization (GISO), spanning the communities of statistics, computational statistics, and graphical models.
- Clarifying that f-CondNCE does not equal score matching is a crucial correction—many works cite Ceylan & Gutmann (2018) to justify CondNCE, but this work points out that with finite samples, the estimator's variance explodes under small noise. This "detail" fundamentally changes the practical utility of the method.
- Practical value of the unified perspective: realizing that various methods are special cases of alpha-CentNCE allows directly transferring the analytical framework of GlobalGISO to obtain finite-sample guarantees for all other methods.

## Limitations & Future Work
- Finite-sample guarantees are limited to bounded exponential family distributions—deep energy-based models and non-parametric models are not covered.
- There is a lack of theoretical guidance on the optimal choice of different alpha values (when to use MLE vs. GISO?).
- The choice of the noise distribution $q_n$ still relies on heuristics—although CondNCE circumvents this issue, it introduces new requirements on the sample size.
- Systematic numerical experiments comparing the practical performance of different alpha-CentNCE variants are lacking.
- Practical validation on large-scale energy-based models (e.g., EBMs for generation) is missing.

## Related Work & Insights
- **vs Gutmann & Hyvärinen (2012)**: Original NCE uses the log generating function (first row of Table 1); this work extends it to general $f$ and unifies a broader class of methods via the centering transform.
- **vs Shah et al. (2023)**: GlobalGISO is a special case with $\alpha=0$; this work generalizes its analytical framework to the entire alpha spectrum.
- **vs Riou-Durand & Chopin (2018)**: They connect MC-MLE and IS objectives (special cases of $\alpha=1$) using a Poisson transform; the alpha-centering in this work acts as a "generalized inverse Poisson transform".

## Rating
- Novelty: ⭐⭐⭐⭐ The unified perspective and correction of the CondNCE error hold significant theoretical value.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work, lacking numerical validation.
- Writing Quality: ⭐⭐⭐⭐ Clear comparisons in Tables 1 and 2, rigorous theorem statements.
- Value: ⭐⭐⭐⭐ A significant contribution to unifying energy-based model learning and NCE theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2025\] FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation](fsl-sage_accelerating_federated_split_learning_via_smashed_activation_gradient_e.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](../../NeurIPS2025/optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](../../ICML2026/optimization/bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)

</div>

<!-- RELATED:END -->
