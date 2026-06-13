---
title: >-
  [Paper Note] Approximating Shapley Explanations in Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Shapley values] This paper proposes FastSVERL, a scalable parametric learning framework that separately approximates the two computational bottlenecks of Shapley values in reinforce…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Shapley values"
  - "RL interpretability"
  - "feature attribution"
  - "parametric approximation"
  - "off-policy learning"
date: 2026-05-08
content_hash: 03e9100872055f40
---

# Approximating Shapley Explanations in Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.06094](https://arxiv.org/abs/2511.06094)  
**Code**: [Available](https://github.com/djeb20/fastsverl)  
**Area**: Reinforcement Learning / Interpretability
**Keywords**: Shapley values, RL interpretability, feature attribution, parametric approximation, off-policy learning

## TL;DR

This paper proposes FastSVERL, a scalable parametric learning framework that separately approximates the two computational bottlenecks of Shapley values in reinforcement learning—the characteristic function and the Shapley summation—while supporting off-policy learning and continuous explanation updates as the policy evolves.

## Background & Motivation

Shapley values provide a principled, theoretically grounded framework (satisfying fairness and consistency axioms) for feature attribution in RL. The SVERL framework defines three explanation targets: **behavioral** (how features influence action selection), **outcome** (how features influence expected return), and **predictive** (how features influence value predictions).

However, exact Shapley value computation incurs cost $\mathcal{O}(2^{|\mathcal{F}|} \cdot |\mathcal{S}|)$—requiring enumeration of all feature subsets for each input, with each subset requiring an expectation over the state space. This is entirely intractable for high-dimensional RL problems.

RL also poses additional challenges beyond supervised learning: (1) explanations require temporal dependencies across multi-step trajectories; (2) explanations must update as the policy evolves; and (3) when environment interaction is limited, explanations must be learned from off-policy data.

## Method

### Overall Architecture

FastSVERL decomposes the approximation problem into two levels:

1. **Approximating the characteristic function**: learning a parametric model to predict conditional expectations given a subset of features.
2. **Approximating the Shapley summation**: learning a parametric model to directly predict all Shapley values, analogous to FastSHAP.

Both levels apply uniformly to all three explanation targets (behavioral, outcome, predictive) and share the same model architecture and training procedure.

### Key Designs

1. **Parametric Characteristic Function Approximation**

    **Function**: A parametric model $\hat{\pi}(s, a | \mathcal{C}; \beta)$ is trained to approximate the behavioral characteristic function $\tilde{\pi}_s^a(\mathcal{C})$—the conditional action probability given feature subset $\mathcal{C}$.

    **Mechanism**: Features not in $\mathcal{C}$ are replaced with out-of-support values, and the model is trained to minimize $\mathcal{L}(\beta) = \mathbb{E}_{p^\pi(s)} \mathbb{E}_{\text{Unif}(a)} \mathbb{E}_{p(\mathcal{C})} |\pi(s,a) - \hat{\pi}(s,a|\mathcal{C};\beta)|^2$. Because different states share the same masked representation under the same subset $\mathcal{C}$, the model cannot recover the exact target and instead learns its mean—i.e., the characteristic function value. The resulting characteristic function is exactly unbiased over all $(s, a, \mathcal{C})$ with $p^\pi(s) > 0$.

    **Design Motivation**: Compared to Monte Carlo sampling (which does not generalize across states and requires recomputation when the policy changes), the parametric model amortizes approximation cost across all states and feature subsets.

2. **Conditional Policy + Parametric Value Function for Outcome Characteristic Function**

    **Function**: A conditional policy $\hat{\pi}(a|s; s_e, \mathcal{C})$ is defined, which uses the characteristic function behavior at the state to be explained $s_e$ and the original policy elsewhere. A parametric value function $V(s|s_e, \mathcal{C}; \beta)$ is then trained to estimate the expected return under this conditional policy.

    **Mechanism**: The outcome characteristic function $\tilde{v}_s^\pi(\mathcal{C})$ requires solving independent RL problems for $2^{|\mathcal{F}|} \times |\mathcal{S}|$ distinct $(s_e, \mathcal{C})$ pairs. A single conditional policy and parametric value function unify all these problems, with both on-policy (Eq. 14) and off-policy (Eq. 15) training variants provided.

    **Design Motivation**: Outcome explanations represent a challenge unique to RL—evaluating the long-term consequences of acting under partial information—which cannot be handled directly by supervised learning approaches.

3. **Single-Sample Approximation Eliminating the Characteristic Function Model**

    **Function**: A single sampled feature value directly replaces the pretrained characteristic function model, embedding feature estimation into Shapley model training.

    **Mechanism**: In the Shapley loss, $\tilde{\pi}_s^a(\mathcal{C})$ is replaced by $\pi(s', a)$ where $s' \sim p^\pi(\cdot | s^\mathcal{C})$, yielding the revised loss $\mathcal{L}(\theta) = \mathbb{E}_{p^\pi(s)} \mathbb{E}_{p(\mathcal{C})} \mathbb{E}_{s' \sim p^\pi(\cdot|s^\mathcal{C})} |\pi(s',a) - \pi_{s,a}(\emptyset) - \sum_{i \in \mathcal{C}} \hat{\phi}^i(s,a;\theta)|^2$. The authors prove that this loss recovers exact, unbiased Shapley values at its global optimum.

    **Design Motivation**: Training the characteristic function model is the dominant computational bottleneck (~50% of compute). The single-sample approximation trades higher gradient variance for the elimination of pretraining overhead and error propagation. Empirically, this halves total training time while improving accuracy.

### Loss & Training

- The Shapley model uses FastSHAP's weighted least-squares objective (Eq. 10), with subsets $\mathcal{C}$ sampled according to the Shapley weighting distribution (Eq. 6).
- The efficiency constraint is enforced via post-hoc correction: $\phi^i \leftarrow \hat{\phi}^i + \frac{1}{|\mathcal{F}|}(\pi(s,a) - \tilde{\pi}_s^a(\emptyset) - \sum_j \hat{\phi}^j)$.
- Off-policy learning uses importance sampling to correct for distributional mismatch, with weights $\frac{\pi(s_t, a_t)}{\pi_t(s_t, a_t)}$.
- Continual learning is achieved through joint updates of the agent and explanation model; update ratios of 10:1 or 50:1 yield the best performance.

## Key Experimental Results

### Main Results (Tables)

**Convergence in large-scale Mastermind domains (10 runs)**:

| Domain | Model | Steps to Convergence | Final Loss |
|--------|-------|----------------------|------------|
| Mastermind-443 (24 features, ≥4.3×10⁷ states) | Characteristic model | (1.10±0.11)×10⁶ | (3.83±0.02)×10⁻³ |
| | Shapley model | (7.31±0.68)×10⁵ | (1.30±0.04)×10⁻³ |
| Mastermind-463 (36 features, ≥2.8×10¹¹ states) | Characteristic model | (1.18±0.12)×10⁶ | (3.70±0.01)×10⁻³ |
| | Shapley model | (7.12±0.51)×10⁵ | (1.88±0.04)×10⁻³ |

Two key observations: (1) convergence is stable and reliable; (2) the number of required training steps does not increase with the number of states or features.

### Ablation Study (Tables)

**Single-sample vs. model-based vs. exact characteristic values (Mastermind-222, behavioral Shapley)**:

| Method | Total Training Steps | Final Shapley MSE |
|--------|----------------------|-------------------|
| Exact characteristic values | Baseline | Lowest (slightly slower convergence) |
| Model-based characteristics (standard) | 2× baseline | Higher (error propagation) |
| **Single-sample approximation** | **1× baseline** | **Comparable to exact** |

The single-sample method converges before the characteristic model begins Shapley training, halving total training time.

### Key Findings

- **Hypercube scalability**: For fixed feature count, training cost grows approximately polynomially with the number of states (near-linear on a log-log scale); for fixed state count, increasing the number of features has negligible effect on training cost.
- **Off-policy learning**: Training with importance-sampled replay buffers reduces approximation error, though it does not match the on-policy baseline.
- **Continual learning**: An update ratio of 10:1 during joint training is sufficient for the explanation model to track policy changes, avoiding the error spikes caused by large policy updates.
- **Interpretability validation**: Behavioral explanations for Mastermind-463 show that the most recent guess contributes most to the next action, while unused slots contribute zero (satisfying the Shapley nullity axiom).

## Highlights & Insights

- The paper systematically extends FastSHAP's amortized approximation paradigm to all three explanation targets in RL, addressing the RL-specific challenges of temporal dependency, off-policy data, and policy non-stationarity.
- The theoretical contribution of the single-sample approximation is particularly elegant—trading higher variance for the elimination of an entire computational stage, with direct applicability to supervised learning settings.
- The work establishes a complete theoretical-practical framework for RL interpretability, providing a foundation for future extensions.

## Limitations & Future Work

- Experiments are limited to discrete action spaces in tabular or small-scale domains; validation in high-dimensional settings such as continuous control or pixel-based observations is absent.
- Off-policy importance sampling may exhibit high variance in high-dimensional or long-horizon settings.
- No user studies are conducted to assess the comprehensibility or practical utility of the explanations.
- Approximating the stationary distribution in high-dimensional continuous state spaces remains an open problem.

## Related Work & Insights

- The paper builds on the theoretical framework of SVERL (Beechey et al., 2023/2025), serving as its first scalable and practical instantiation.
- Its relationship to FastSHAP (Jethani et al., 2021): the approach extends single-step predictive Shapley approximation to multi-step sequential decision-making.
- A noteworthy direction for future work: the single-sample approximation idea for eliminating the characteristic function model can be directly transferred back to supervised learning to improve FastSHAP.

## Rating

⭐⭐⭐⭐ Theoretically elegant and architecturally complete, this work systematically addresses the scalability of Shapley values in RL. The single-sample approximation is a standout contribution, though the experimental scale remains modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](../../ACL2026/reinforcement_learning/savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)
- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](../../ICML2026/reinforcement_learning/shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
