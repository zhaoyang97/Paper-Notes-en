---
title: >-
  [Paper Note] Exploration via Feature Perturbation in Contextual Bandits
description: >-
  [NeurIPS 2025 Spotlight][Reinforcement Learning][contextual bandits] Proposes Feature Perturbation (FP) as a novel randomized exploration strategy for contextual bandits. By directly injecting noise into the feature inputs instead of perturbing parameters or rewards, it achieves an optimal regret bound of $\tilde{O}(d\sqrt{T})$ in generalized linear bandits, eliminating the $\sqrt{d}$ factor disadvantage of randomized algorithms compared to deterministic methods for the first…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Reinforcement Learning"
  - "contextual bandits"
  - "feature perturbation"
  - "generalized linear bandits"
  - "regret bound"
  - "exploration"
date: 2026-05-08
content_hash: b313a3ddfbdaa914
---

# Exploration via Feature Perturbation in Contextual Bandits

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.17390](https://arxiv.org/abs/2510.17390)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: contextual bandits, feature perturbation, generalized linear bandits, regret bound, exploration

## TL;DR

Proposes Feature Perturbation (FP) as a novel randomized exploration strategy for contextual bandits. By directly injecting noise into the feature inputs instead of perturbing parameters or rewards, it achieves an optimal regret bound of $\tilde{O}(d\sqrt{T})$ in generalized linear bandits, eliminating the $\sqrt{d}$ factor disadvantage of randomized algorithms compared to deterministic methods for the first time.

## Background & Motivation

Exploration strategies in contextual bandits are divided into two main categories:
- **Deterministic methods** (UCB/OFU): Achieve the theoretically optimal $\tilde{O}(d\sqrt{T})$ regret, but are overly conservative in practice.
- **Randomized methods** (Thompson Sampling, PHE): Perform exceptionally well in practice, but suffer from a regret bound of $\tilde{O}(d^{3/2}\sqrt{T})$, which incurs an extra $\sqrt{d}$ factor.

Hamidi & Bayati (2021) proved that this gap **is not an artifact of loose analysis** but an inherent cost of parameter-level randomization: eliminating the $\sqrt{d}$ factor within the Thompson Sampling framework leads to a linear dependence on $T$.

This leads to the core problem: **Does there exist a randomized algorithm that can achieve the optimal $\tilde{O}(d\sqrt{T})$ regret?**

Key Insight: Prior randomized methods either perturb parameters (TS) or rewards (PHE). This work proposes a third paradigm—**perturbing features**. This fundamentally changes the action space of randomization, breaking the inherent limitations.

## Method

### Overall Architecture

**GLM-FP Algorithm** (Algorithm 1) workflow:
1. At each round $t$: Compute the MLE $\hat{\theta}_t$.
2. Sample a **single** noise vector $\zeta_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$.
3. Construct the perturbed feature for each arm $i$: $\tilde{x}_{ti} = x_{ti} + c_t \cdot \frac{\|x_{ti}\|_{\hat{H}_t^{-1}}}{\|\hat{\theta}_t\|} \cdot \zeta_t$.
4. Select the arm that maximizes $\mu(\tilde{x}_{ti}^\top\hat{\theta}_t)$.

Core: All arms share the same noise vector $\zeta_t$, and the perturbation magnitude is adaptively scaled according to each arm's uncertainty $\|x_{ti}\|_{\hat{H}_t^{-1}}$.

### Key Designs

**Why does feature perturbation avoid the $\sqrt{d}$ factor?** Comparing the perturbation terms of TS and FP directly:

- **TS perturbation**: $|x_t^\top(\tilde{\theta}_t - \hat{\theta}_t)| = c_t \|x_{ti}\|_{V_t^{-1}} \cdot \|\zeta_t\|$ $\rightarrow$ requires controlling the norm of a $d$-dimensional Gaussian vector.
- **FP perturbation**: $|(\tilde{x}_t - x_t)^\top\hat{\theta}_t| = c_t \|x_{ti}\|_{V_t^{-1}} \cdot |u^\top\zeta_t|$ $\rightarrow$ only requires controlling a 1-dimensional projection.

TS requires a union bound over $d$ dimensions $\rightarrow$ introducing $\sqrt{d}$; FP only involves the scalar random variable $u^\top\zeta_t$ $\rightarrow$ avoiding the union bound.

**Weighted Gram Matrix** $\hat{H}_t = \lambda\boldsymbol{I} + \nabla^2 L_t(\hat{\theta}_t)$: Unlike prior works using standard Gram matrices $V_t$, this work utilizes a weighted version containing curvature information of the link function, achieving more precise perturbation control.

**Shared Noise Design**: All arms use the same noise vector $\zeta_t$, eliminating the dependence of the regret bound on the number of arms $K$.

### Loss & Training

**Parameter Estimation**: Uses MLE with negative log-likelihood, weighted least squares for the linear case, and IRLS for the logistic case.

**Hyperparameter Tuning Strategy**: $c_t = \beta_t(\delta')$, where $\beta_t$ is the confidence width and $\delta' = \delta/(4T)$; regularization is set to $\lambda = O(d)$.

**Key Technical Steps in Proof**:
1. **Confidence Set Construction** (Lemma 1): $\theta^* \in \{\theta: \|\theta - \hat{\theta}_t\|_{\hat{H}_t} \leq \beta_t(\delta)\}$
2. **Concentration** (Lemma 2): The perturbed features remain close to the original features with high probability.
3. **Randomized Optimism** (Lemma 3): $\Pr_t(\mu(\tilde{x}_t^\top\hat{\theta}_t) \geq \mu(x_{t*}^\top\theta^*)) \geq \frac{1}{4\sqrt{e\pi}}$ — a constant lower bound.
4. **Regret Decomposition**: $R(T) = \text{Reg}_{\text{FP}} + \text{Reg}_{\text{EST}}$, where both components are individually bounded.

## Key Experimental Results

### Main Results: Linear & Logistic Bandits

| Algorithm | Linear Bandit (d=10) | Linear Bandit (d=50) | Logistic Bandit (d=10) | Logistic Bandit (d=50) |
|------|-------------------|-------------------|----------------------|----------------------|
| ε-greedy | Highest regret | Highest regret | Highest regret | Highest regret |
| UCB | Medium | Medium | Medium | Medium |
| TS | Medium-low | Medium | Medium | Medium-high |
| PHE | Medium-low | Medium | Medium | Medium-high |
| RandUCB | Near-optimal | Near-optimal | Medium-low | Medium-low |
| **GLM-FP** | **Lowest** | **Lowest** | **Lowest** | **Lowest** |

GLM-FP consistently achieves the lowest regret across all tested dimensions and settings.

### Dimension Dependency Experiments

| Dimension d | TS Terminal Regret | FP Terminal Regret | Ratio |
|--------|-----------|-----------|------|
| 10 | ~400 | ~200 | 2.0x |
| 20 | ~900 | ~400 | 2.25x |
| 30 | ~1500 | ~600 | 2.5x |
| 50 | ~3000 | ~1000 | 3.0x |

Verifies that FP regret grows linearly with $d$, whereas TS regret grows with $d^{3/2}$.

### Neural Bandit Experiments

| Dataset | ε-greedy | NeuralUCB | NeuralTS | FTPL | **DeepFP** |
|--------|----------|-----------|----------|------|-----------|
| shuttle | Medium | Medium-low | Medium-low | Medium | **Lowest** |
| isolet | High | Medium | Medium | Medium | **Lowest** |
| mushroom | Medium | Low | Low | Medium | **Lowest** |

DeepFP outperforms all baselines across three UCI datasets, demonstrating the scalability of feature perturbation on non-parametric models.

### Key Findings

1. **Empirically Consistent with Theory**: The regret ratio of FP vs. TS increases as $d$ increases, matching the theoretical prediction of $d$ vs. $d^{3/2}$.
2. **More Pronounced Advantage in Logistic Settings**: GLM-FP benefits from modeling the link function's curvature accurately using the weighted Gram matrix.
3. **Simple and Effective Neural Extension**: DeepFP drives exploration with only a decaying noise $\zeta_{ti} \sim \mathcal{N}(0, I/t)$, without requiring access to model parameters or gradients.

## Highlights & Insights

- **Conceptually Simple yet Profound**: Shifting the randomness of exploration from parameter/reward space to feature space is a seemingly minor perspective shift with deep theoretical implications.
- **Clear Geometric Interpretation**: TS projects all arms onto a shared random direction in the $V_t^{-1/2}$-transformed space, which might assign a large bonus to well-explored arms. FP, scaled independently by uncertainty, systematically biases towards unexplored arms.
- **Outstanding Practicality**: Avoids the computational overhead of parameter sampling, naturally adapting to high-dimensional parametric models like neural networks.

## Limitations & Future Work

- Theoretical analysis is limited to generalized linear models; DeepFP for neural bandits lacks theoretical guarantees.
- Requires $\|\hat{\theta}_t\|$ for scaling, which may become unstable when the estimate is close to zero.
- Experiments are restricted to synthetic or UCI data, lacking validation on large-scale real-world recommendation or advertising systems.
- Compatibility with batched/delayed feedback settings has not been explored.

## Related Work & Insights

- **LinTS / GLM-TS**: Classic parameter randomization methods with an unimprovable $\tilde{O}(d^{3/2}\sqrt{T})$ regret.
- **RandUCB**: Another randomized method achieving $\tilde{O}(d\sqrt{T})$, but lacks instance-dependent constant $\kappa_*$.
- **PHE**: Reward perturbation method, also limited by the $\sqrt{d}$ factor.
- **Neural bandits**: NeuralUCB/NeuralTS require gradient information to construct confidence sets, whereas FP only needs input perturbation, making it much more lightweight.

## Rating

- **Novelty**: 9/10 — Conceptual perspective of feature perturbation is novel, with significant theoretical breakthroughs.
- **Theoretical Depth**: 9/10 — Rigorous proof of the optimal regret bound with elegant geometric explanations.
- **Experimental Thoroughness**: 8/10 — Comprehensive coverage of synthetic and UCI datasets, but lacks large-scale experiments.
- **Value**: 8/10 — Simple implementation and naturally adaptable to deep models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)

</div>

<!-- RELATED:END -->
