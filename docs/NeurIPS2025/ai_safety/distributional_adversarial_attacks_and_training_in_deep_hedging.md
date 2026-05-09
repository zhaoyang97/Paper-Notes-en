---
title: >-
  [Paper Note] Distributional Adversarial Attacks and Training in Deep Hedging
description: >-
  [NeurIPS 2025][AI Safety][Adversarial Training] This paper is the first to introduce distributional adversarial attacks into the deep hedging framework. It proposes computationally tractable adversarial training methods based on Wasserstein balls (WPGD and WBPGD), achieving substantial improvements in robustness and out-of-sample performance under distribution shift and real market data.
tags:
  - NeurIPS 2025
  - AI Safety
  - Adversarial Training
  - Distributionally Robust Optimization
  - Deep Hedging
  - Wasserstein Distance
  - Financial Derivatives
date: 2026-05-08
content_hash: 90c2f9590a7bc9be
---

# Distributional Adversarial Attacks and Training in Deep Hedging

**Conference**: NeurIPS 2025
**arXiv**: [2508.14757](https://arxiv.org/abs/2508.14757)
**Code**: [github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging](https://github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging)
**Area**: AI Safety
**Keywords**: Adversarial Training, Distributionally Robust Optimization, Deep Hedging, Wasserstein Distance, Financial Derivatives

## TL;DR
This paper is the first to introduce distributional adversarial attacks into the deep hedging framework. It proposes computationally tractable adversarial training methods based on Wasserstein balls (WPGD and WBPGD), achieving substantial improvements in robustness and out-of-sample performance under distribution shift and real market data.

## Background & Motivation

**State of the Field**: Deep Hedging (Buehler et al., 2019) parameterizes hedging strategies with neural networks and trains them by minimizing risk measures, and has been widely adopted in industry. Training data is drawn from stochastic model simulations or historical observations.

**Limitations of Prior Work**: Model misspecification between the training distribution and the deployment distribution is pervasive; even small distributional shifts can cause severe degradation in hedging performance. Existing robust methods only perturb terminal distributions or randomize model parameters, lacking systematic analysis of distributional robustness.

**Root Cause**: The deep hedging loss (combining a neural network policy with a risk measure) is highly non-convex, whereas the tractability results for standard Wasserstein DRO rely on convexity assumptions and are therefore inapplicable here.

**Paper Goals**: (a) Quantify the vulnerability of deep hedging to distribution shift; (b) design computationally tractable distributional adversarial attack methods; (c) improve robustness via adversarial training.

**Starting Point**: Extend pointwise adversarial attacks (FGSM/PGD) to the distributional level by leveraging sensitivity analysis for Wasserstein DRO (Bartl et al., 2021) to obtain a tractable first-order approximation.

**Core Idea**: Realize distributional adversarial attacks via projected gradient descent on a Wasserstein ball, and incorporate adversarial examples into the training loop to improve the robustness of deep hedging.

## Method

### Overall Architecture
Deep hedging under OCE (Optimized Certainty Equivalent) risk measure: $\min_{\tilde{\theta}} \mathbb{E}_{\mathbf{I} \sim \mu}[l_{\text{DH}}(\tilde{\theta}, \mathbf{I})]$, where $l_{\text{DH}}(\tilde{\theta}, \mathbf{I}) = \omega + \ell(-\text{PnL}(\theta, \mathbf{I}) - \omega)$ and $\omega$ is a trainable parameter. The DRO formulation is $\min_{\tilde{\theta}} \max_{\eta \in B_\delta(\mu)} \mathbb{E}_{\mathbf{I} \sim \eta}[l_{\text{DH}}(\tilde{\theta}, \mathbf{I})]$.

### Key Designs

1. **Tractable Reformulation of Distributional Adversarial Attacks (Theorem 3.3)**

    - Function: Approximates the infinite-dimensional optimization over a Wasserstein ball as a finite-dimensional sample perturbation problem.
    - Mechanism: Based on DRO sensitivity analysis, as $\delta \to 0$ the optimal perturbed distribution $\eta_\delta$ can be expressed as a pointwise perturbation of each sample $X_n$: $\hat{X}_n = X_n + \delta \cdot h(\nabla_x l(\theta; X_n)) \|\nabla_x l(\theta; X_n)\|_*^{q-1} \Upsilon^{1-q}$
    - Here $\Upsilon = (N^{-1} \sum_{n=1}^N \|\nabla_x l(\theta; X_n)\|_*^q)^{1/q}$ and $h(x) = \text{sign}(x)|x|^{q-1}$ corresponds to the dual under $\ell_p$ norms.
    - Design Motivation: Decomposing distributional perturbation into sample-level perturbations, where each sample's perturbation magnitude is proportional to its gradient norm, allocating more budget to paths with larger gradients.

2. **WPGD (Wasserstein PGD, Algorithm 1)**

    - Function: A distributional analogue of PGD attacks.
    - Mechanism: Iteratively performs (a) updating each path according to the sensitivity formula $\hat{S}_n \leftarrow \hat{S}_n + \beta \cdot \text{sign}(\nabla_x l_{\text{DH}}(\theta; \hat{S}_n)) \|\nabla_x l(\theta; \hat{S}_n)\|_*^{q-1} \hat{\Upsilon}^{1-q}$, and (b) projecting back onto the Wasserstein constraint set $\hat{B}_\delta(\mu)$.
    - Projection: $\hat{S}_n \leftarrow S_n + \max(1, \delta/\text{dist})(\hat{S}_n - S_n)$, with global rescaling to enforce the constraint.

3. **WBPGD (Wasserstein Budget PGD, Algorithm 2)**

    - Function: Decomposes perturbations into budget and direction variables that are optimized independently.
    - Mechanism: Sets $\hat{S}_n = S_n + \text{budget}_n \times \text{direction}_n$, where $\text{budget}_n \in \mathbb{R}_{\ge 0}$ controls perturbation magnitude and $\text{direction}_n \in [-1,1]^{T+1}$ controls perturbation direction.
    - Update rules (Lemma 4.1): $\text{budget}_n \leftarrow \text{budget}_n + \beta \cdot (g_n^b)^{q-1} \hat{\Upsilon}^{1-q}$, $\text{direction}_n \leftarrow \text{direction}_n + (\beta/\delta) \cdot \text{sign}(g_n^d)$
    - Advantage: Decoupling budget allocation from direction optimization enables more thorough exploration of the adversarial space.

4. **Extension to the Heston Model (Corollary 4.2)**

    - Function: Handles dual-sequence inputs of price and volatility.
    - Mechanism: Defines a weighted distance $d((S,v),(\hat{S},\hat{v})) = (\|S-\hat{S}\|_\infty^p + (\lambda \|v-\hat{v}\|_\infty)^p)^{1/p}$, with $\lambda$ balancing different scales.
    - Equivalent to applying independent $\ell_\infty$ perturbations to $S$ and $\lambda v$ separately.

### Loss & Training
The adversarial training loss is $\mathcal{L}_{\text{adv}}(\theta) = \alpha \sum_n l_{\text{DH}}(\theta; X_n) + \sum_n l_{\text{DH}}(\theta; \hat{X}_n)$, where $\alpha$ balances clean and adversarial samples. Adversarial attacks and parameter updates are performed in alternation.

## Key Experimental Results

### Adversarial Attack Evaluation on the Heston Model (Classical Deep Hedging, CVaR Risk Measure)

| Attack Method | $\delta=0$ | $\delta=0.01$ | $\delta=0.05$ | $\delta=0.1$ | $\delta=0.3$ | $\delta=0.5$ |
|--------------|-----------|--------------|--------------|-------------|-------------|-------------|
| S-WBPGD | 1.928 | 1.964 | 2.136 | 2.447 | 4.577 | 8.075 |
| SV-WBPGD | 1.928 | 1.966 | 2.145 | 2.466 | 4.590 | 7.739 |
| S-WPGD | 1.928 | 1.964 | 2.134 | 2.437 | 4.515 | 7.541 |

### Out-of-Sample Performance: Adversarial Training vs. Standard Training

| Model/Method | Clean Loss | Attacked Loss ($\delta=0.1$) | Improvement |
|-------------|------------|---------------------------|-------------|
| BS Standard | Baseline | Severe degradation | — |
| BS Adversarial Training | Slightly higher | Significantly improved | 30–50% reduction in attacked loss |
| Heston Standard | 1.928 | 2.447 | — |
| Heston Adversarial Training | ~2.0 | ~2.1 | Attacked loss approaches baseline |

### Key Findings
- Even a minimal distributional perturbation of $\delta=0.01$ increases CVaR loss by approximately 2%; at $\delta=0.5$, loss quadruples, demonstrating the extreme fragility of standard deep hedging.
- WBPGD consistently outperforms WPGD, especially under large perturbations — budget-direction decoupling enables more effective utilization of the attack budget.
- The Frobenius distance of the covariance matrix remains small for $\delta < 0.1$ (< 1.0 vs. baseline ~386), indicating that statistically "small" distributional shifts can induce large losses.
- Adversarial training remains effective on real market data, with particularly prominent gains during periods of high market volatility.

## Highlights & Insights
- **Unified View of Distributional and Pointwise Attacks**: As $p \to \infty$, the distributional constraint degenerates to an $L_\infty$ pointwise constraint, with WPGD/WBPGD subsuming FGSM/PGD as special cases. This establishes a complete theoretical bridge from DRO to adversarial training.
- **Budget-Direction Decoupling**: WBPGD separates "how much perturbation to allocate to each path" from "in which direction to perturb" into independent optimization sub-problems, offering greater flexibility than direct gradient ascent in path space and transferable to other sequential DRO problems.
- **Elegant Use of the OCE Risk Measure**: By treating $\omega$ as a trainable parameter, the DRO formulation of deep hedging reduces to standard expected loss minimization, directly admitting the theoretical tools from the DRO literature.

## Limitations & Future Work
- The theoretical results (Theorem 3.3, Lemma 3.4) are fundamentally first-order approximations as $\delta \to 0$; approximation quality at large $\delta$ is not guaranteed.
- The analysis assumes Lipschitz continuity of the loss with respect to inputs, which may be violated in practice (e.g., barrier option knock-in events).
- Adversarial training requires additional gradient computations and projection steps, increasing training time by approximately 2–3×.
- The effects of transaction costs and market impact on robustness are not addressed.

## Related Work & Insights
- **vs. Lutkebohmert et al. (2022) (parameter-randomized robust DH)**: Their approach introduces uncertainty by randomizing model parameters, whereas this paper perturbs at the data distribution level directly — a more flexible approach with a stronger theoretical foundation.
- **vs. Wu et al. (2023) (terminal distribution robustness)**: That work perturbs only the terminal payoff distribution without touching intermediate paths; this paper perturbs entire price paths, covering a broader class of distributional shifts.
- **vs. Madry et al. (2018) (pointwise PGD)**: The classical adversarial training approach from computer vision, which this paper generalizes to the distributional level and adapts to financial time-series data.

## Rating
- Novelty: ⭐⭐⭐⭐ — First application of distributional adversarial attacks to deep hedging; the budget-direction decoupling in WBPGD is a creative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers BS/Heston/General Affine Diffusion models, validated on real market data, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, theory and experiments are well integrated, notation is consistent.
- Value: ⭐⭐⭐⭐ — Directly applicable to financial ML practice; the framework generalizes to other data-driven decision-making problems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)

<!-- RELATED:END -->
