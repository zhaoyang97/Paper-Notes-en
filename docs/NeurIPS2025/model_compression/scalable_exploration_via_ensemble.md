---
title: >-
  [Paper Note] Ensemble++: Scalable Exploration via Ensemble
description: >-
  [NeurIPS 2025][Model Compression][Thompson Sampling] This paper proposes Ensemble++, which achieves regret bounds comparable to exact Thompson Sampling using only $\Theta(d\log T)$ ensemble size via an incremental update…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Thompson Sampling"
  - "Ensemble Sampling"
  - "Linear Bandit"
  - "Exploration-Exploitation"
  - "Approximate Posterior"
date: 2026-05-08
content_hash: 154c041bb88b306f
---

# Ensemble++: Scalable Exploration via Ensemble

**Conference**: NeurIPS 2025  
**arXiv**: [2407.13195](https://arxiv.org/abs/2407.13195)  
**Code**: [https://github.com/szrlee/Ensemble_Plus_Plus](https://github.com/szrlee/Ensemble_Plus_Plus)  
**Area**: Model Compression  
**Keywords**: Thompson Sampling, Ensemble Sampling, Linear Bandit, Exploration-Exploitation, Approximate Posterior

## TL;DR
This paper proposes Ensemble++, which achieves regret bounds comparable to exact Thompson Sampling using only $\Theta(d\log T)$ ensemble size via an incremental update mechanism over shared factor matrices, with natural extension to nonlinear/neural network settings.

## Background & Motivation
Thompson Sampling (TS) is a classical Bayesian method for balancing exploration and exploitation in sequential decision-making. However, exact posterior sampling is computationally prohibitive in high-dimensional or non-conjugate settings (e.g., neural networks). Ensemble Sampling approximates TS by maintaining $M$ model replicas, but achieving optimal regret bounds theoretically requires ensemble size $M = \Omega(T \cdot |\mathcal{X}|)$ (Qin et al., 2022), which is entirely infeasible at long time horizons or with large action spaces.

**Key Challenge**: How can a practically small ensemble size approximate the posterior sampling of TS while retaining near-optimal regret bounds?

**Key Insight**: The paper designs a novel "shared-factor ensemble" architecture that compresses $M$ ensemble directions into an approximate representation of the posterior covariance matrix via a random linear combination mechanism, fundamentally reducing the ensemble size requirement.

## Method

### Overall Architecture
Ensemble++ maintains a shared matrix factor $\mathbf{A}_t \in \mathbb{R}^{d \times M}$, which approximates the posterior covariance square root $\Sigma_t^{1/2}$ via incremental updates. At action selection, a "pseudo-posterior sample" is generated via the random linear combination $\theta_t(\zeta_t) = \mu_{t-1} + \mathbf{A}_{t-1}\zeta_t$, rather than randomly selecting from $M$ independent models.

### Key Designs

1. **Shared Factor Incremental Update**: Each step requires only $O(d^2 M)$ updates — after observing the reward, the mean $\mu_t$ and ensemble matrix are updated as:
    $\mathbf{A}_t = \Sigma_t(\Sigma_{t-1}^{-1}\mathbf{A}_{t-1} + X_t \mathbf{z}_t^\top)$
   where $\mathbf{z}_t \in \mathbb{R}^M$ is a perturbation vector. This avoids retraining from scratch or large-scale matrix factorization.

2. **Random Linear Combination Sampling**: At action selection, a reference vector $\zeta_t$ is sampled from distribution $P_\zeta$, and an approximate posterior sample is constructed as $\theta_t = \mu_{t-1} + \mathbf{A}_{t-1}\zeta_t$. Unlike conventional ensemble sampling that randomly selects a single model, this approach linearly combines all columns, substantially improving information utilization efficiency.

3. **Symmetrized Regression Objective**: The base and ensemble parameters are unified under a symmetrized regression objective:
    $L(\theta; D, f) = \sum_{m=1}^M \sum_{s \in D} \sum_{\beta \in \{\pm 1\}} (Y_s + \beta \mathbf{z}_{s,m} - f(X_s, \beta e_m))^2 + \lambda\|\theta\|^2$
   This admits a closed-form solution in the linear case and is optimized via SGD in the neural network case, providing a seamless bridge from theory to practice.

4. **Neural Network Extension**: The linear features are replaced by a learnable neural feature extractor $h(x;w)$, while maintaining the same incremental update principle. A FIFO buffer and a fixed number of SGD steps ensure constant-time updates.

### Loss & Training
- Linear case: closed-form update of the shared factor matrix $\mathbf{A}_t$
- Neural case: symmetrized loss + SGD, FIFO buffer of capacity $C$, $G$ gradient updates per step

## Key Experimental Results

### Main Results

| Setting | Metric | Ensemble++ | Ensemble+ | EpiNet | Notes |
|---------|--------|-----------|-----------|--------|-------|
| Quadratic Bandit | Regret | Sublinear convergence | Linear regret | Linear regret | Significant advantage under nonlinear rewards |
| Neural Bandit | Accuracy | Highest | Second best | Second best | 2-layer MLP setting |
| UCI Shuttle | Accuracy | Best | Second best | Second best | Real-world dataset |
| Hate Speech (GPT-2) | Accuracy | +5% vs Ensemble+ | Baseline | N/A | Validated at Transformer scale |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Ensemble size $M$ vs. dimension $d$ | $M$ scales linearly with $d$ | Validates $M = \Theta(d\log T)$ theory |
| $M$ vs. action set $|\mathcal{X}|$ | Essentially independent | Decoupled from action space size |
| Gaussian vs. coordinate reference distribution | Gaussian superior | Smaller $\rho/p$ ratio for continuous distributions |
| Buffer size $C$ | Small buffer suffices | Full history storage not required |

### Key Findings
- Ensemble++ with only $M=8$ matches exact TS performance at half the computational cost of conventional ensemble sampling.
- Ensemble size scales linearly with dimension and is independent of action space size, validating the theoretical prediction $M = \Theta(d\log T)$.
- The method scales effectively to GPT-2-level Transformers, demonstrating feasibility for extension to large models.

## Highlights & Insights
- **Theoretical Contribution**: The first proof in linear bandits that incremental ensemble updates with $\Theta(d\log T)$ ensemble size achieves TS-level regret bounds — an exponential reduction compared to the $\Omega(|\mathcal{X}|T)$ requirement of Qin et al.
- **Unified Framework**: The same algorithm handles all four combinations of compact/finite action sets × stationary/time-varying contexts without modification.
- **Sequential JL Lemma**: A sequential Johnson-Lindenstrauss variant is proposed for adaptive data collection, resolving the limitation of standard JL that requires independent projections.

## Limitations & Future Work
- The neural extension lacks rigorous theoretical guarantees; theoretical analysis is confined to the linear case.
- Computational complexity $O(d^3\log T)$ incurs an additional $\log T$ factor compared to exact TS at $O(d^3)$.
- Validation in truly large-scale LLM agent settings is absent; experiments are limited to the GPT-2 scale.

## Related Work & Insights
- **vs. Ensemble+ (Osband et al., 2019)**: Ensemble++ requires neither a large ensemble nor retraining, achieving a superior regret-computation trade-off via shared factor updates.
- **vs. EpiNet (Osband et al., 2023)**: Both inject uncertainty via architectural modifications, but Ensemble++ provides theoretical guarantees in the linear setting.
- **vs. LMC-TS (Xu et al., 2022)**: LMC-TS costs $O(d^2T)$ per step versus $O(d^2M)$ for Ensemble++, conferring a significant advantage at long horizons.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The shared-factor ensemble architecture and Sequential JL Lemma constitute entirely novel technical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of linear and nonlinear bandit settings, though large-scale experiments are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, and the extension from linear to nonlinear settings follows naturally.
- Value: ⭐⭐⭐⭐⭐ Addresses the core bottleneck of ensemble sampling and provides a foundational framework for exploration in LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](../../AAAI2026/model_compression/credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](../../ICLR2026/model_compression/flyprompt_brain-inspired_random-expanded_routing.md)
- [\[CVPR 2026\] LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration](../../CVPR2026/model_compression/llava-le_large_language-and-vision_assistant_for_lunar_exploration.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

</div>

<!-- RELATED:END -->
