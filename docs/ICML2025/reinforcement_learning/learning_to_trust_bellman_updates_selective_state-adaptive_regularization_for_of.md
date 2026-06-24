---
title: >-
  [Paper Note] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL
description: >-
  [ICML2025][Reinforcement Learning][Offline RL] Proposes Selective State-Adaptive Regularization (SSAR), which dynamically generates individual regularization coefficients for each state using a neural network and enforces constraints strictly on high-quality actions. This framework unifies the two major offline RL paradigms: CQL (value regularization) and TD3+BC (policy constraint), achieving massive performance gains over baselines on both offline and O2O scenarios in D4RL.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Offline RL"
  - "State-Adaptive Regularization"
  - "Selective Constraint"
  - "Value Function Regularization"
  - "Policy Constraint"
  - "Offline-to-Online RL"
date: 2026-05-08
content_hash: 9e56329eff182d17
---

# Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL

**Conference**: ICML2025  
**arXiv**: [2505.19923](https://arxiv.org/abs/2505.19923)  
**Code**: [QinwenLuo/SSAR](https://github.com/QinwenLuo/SSAR)  
**Area**: Offline Reinforcement Learning (Offline RL)  
**Keywords**: Offline RL, State-Adaptive Regularization, Selective Constraint, Value Function Regularization, Policy Constraint, Offline-to-Online RL

## TL;DR
Proposes Selective State-Adaptive Regularization (SSAR), which dynamically generates individual regularization coefficients for each state using a neural network and enforces constraints strictly on high-quality actions. This framework unifies the two major offline RL paradigms: CQL (value regularization) and TD3+BC (policy constraint), achieving massive performance gains over baselines on both offline and O2O scenarios in D4RL.

## Background & Motivation

The core challenge in offline RL lies in balancing the **performance gains brought by Bellman updates** against the **safety constraints enforced by regularization**:

* **Limitations of global fixed coefficients**: Existing methods (such as CQL and TD3+BC) employ fixed global regularization coefficients $\beta$. However, the optimal coefficient varies by task, training phase, and data density. Weak regularization leads to value overestimation, whereas strong regularization degenerates the policy into behavior cloning.
* **Dynamic training dynamics**: The policy is unstable and unreliable in the early training phases, calling for strict constraints. As it aligns closer to the dataset in later phases, the constraints should be relaxed to unleash the true potential of Bellman updates.
* **Varied data quality**: States with high data density can confidently trust Bellman updates, whereas low-density states require tighter regularization.
* **Offline-to-Online efficiency**: Fixed coefficients yield massive discrepancies between offline and online Q-values, degrading fine-tuning efficiency.

## Method

### Core Idea: Unified Framework

**Proposition 3.1 (Equivalence between CQL and Policy Constraints)**: When modeling the policy $\pi(a|s) \propto \exp(Q(s,a))$ as a Boltzmann distribution, the CQL value regularization term is equivalent to the negative log-likelihood of dataset actions:

$$\min_Q \beta \, \mathbb{E}_{s \sim D}\left[\log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim D}[Q(s,a)]\right] \iff \min_\pi \beta \, \mathbb{E}_{(s,a) \sim D}[-\log \pi(a|s)]$$

This equivalence unifies value regularization and explicit policy constraints under a single framework, facilitating the simultaneous application of state-adaptive coefficients to both paradigms.

### Component 1: State-Adaptive Coefficients

A neural network $\beta_\phi(s)$ is introduced to output individual regularization coefficients for each state, which is adaptively optimized via the following objective:

$$L_\beta(\phi) = \mathbb{E}_{(s,a) \sim D}\left[\log \pi(a|s) - C_n(s)\right] \beta_\phi(s)$$

where the threshold $C_n(s) = \min\{\log \pi(\mu + n\sigma | s), \log \pi(\mu - n\sigma | s)\}$, $\mu, \sigma$ represent the mean and standard deviation of the policy, and $n$ adjustably defines the width of the trust region.

**Mechanism**: When the log-probability of dataset actions exceeds the threshold, the coefficient decays (relying more on Bellman updates). Conversely, when it falls below the threshold, the coefficient rises (strengthening constraints).

### Component 2: Distribution-Aware Threshold

The parameter $n$ is increased linearly from $n_{start}$ to $n_{end}$:

$$n \leftarrow n + \Delta n, \quad \Delta n = (n_{end} - n_{start}) \cdot T_{inc} / T$$

The optimization halts when $\mathbb{E}_{(s,a) \sim D}[\log \pi(a|s) - C_n(s)] > 0$ is met, ensuring the trust region adaptively broadens as training advances.

### Component 3: Selective Regularization

Constraints are selectively imposed on a high-quality action subset $\hat{D}$ to steer the policy clear of low-quality dataset actions:

- **Low-variance datasets**: Trajectories are filtered based on returns to establish $\hat{D}$ where $G > G_T$.
- **High-variance datasets**: Pre-train $Q, V$ networks using IQL, filtering actions based on positive advantages $Q(s,a) - V(s) > 0$.

**Q-function Update for CQL + SSAR**:

$$\min_Q \beta_\phi(s) \, \mathbb{E}_{s \sim \hat{D}}\left[\log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim D}[Q(s,a)]\right] + \frac{1}{2} \mathbb{E}_{s,a,s' \sim D}\left[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2\right]$$

**Policy Update for TD3+BC + SSAR**:

$$\max_\pi \mathbb{E}_{s,a \sim D}\left[Q_{norm}(s, \pi(s)) - \mathbb{I}((s,a) \in \hat{D}) \, \beta_\phi(s) (\pi(s) - a)^2\right]$$

### Component 4: Offline-to-Online Fine-Tuning

Freeze the parameters of the coefficient network and apply a linear decay to the output:

$$\beta_{on}(s) = \min\left\{1 - \frac{N}{N_{end}}, 0\right\} \cdot \beta(s)$$

By leveraging the generalization capabilities of the offline-trained coefficient network, online fine-tuning can proceed efficiently—even allowing the complete disposal of the offline dataset.

## Key Experimental Results

### Offline Performance (D4RL Benchmark)

| Dataset | TD3+BC Base | TD3+BC+SSAR | CQL Base | CQL+SSAR |
|--------|:-----------:|:-----------:|:--------:|:--------:|
| halfcheetah-m | 48.3 | **56.5** | 47.1 | **63.9** |
| hopper-m | 58.7 | **101.6** | 65.6 | **89.1** |
| walker2d-m | 82.3 | **87.9** | 81.6 | **84.9** |
| halfcheetah-mr | 44.4 | **49.6** | 45.7 | **53.8** |
| hopper-mr | 66.4 | **101.6** | 92.3 | **101.4** |
| walker2d-mr | 81.6 | **93.5** | 79.2 | **94.7** |
| halfcheetah-me | 92.9 | **94.9** | 93.0 | **102.1** |
| hopper-me | 101.4 | **103.8** | 97.8 | **109.6** |
| **Locomotion Total Score** | 1000.8 | **1116.7** | 1030.4 | **1139.1** |
| **AntMaze Total Score** | 131.8 | **276.0** | 294.1 | **406.8** |

- Locomotion tasks: SSAR yields a **+11.6%** boost to TD3+BC and a **+10.5%** boost to CQL.
- AntMaze tasks (sparse reward): SSAR achieves a massive **+109%** improvement on TD3+BC and a **+38%** improvement on CQL.

### Offline-to-Online Fine-Tuning (250k steps)

| Dataset | IQL | SPOT | CQL | TD3+BC(SA) | CQL(SA) |
|--------|:---:|:----:|:---:|:----------:|:-------:|
| halfcheetah-m | 49.7 | 58.6 | 48.0 | **82.9** | **95.3** |
| hopper-m | 75.2 | 99.9 | 63.8 | **103.5** | **99.3** |
| walker2d-m | 80.8 | 82.5 | 82.8 | **101.6** | **105.9** |
| **Locomotion Total Score** | 1057.4 | 1107.1 | 1068.4 | **1218.6** | **1278.4** |

In the O2O setting, CQL(SA) tops the benchmark with a total score of 1278.4, outperforming all baseline frameworks by roughly 10%.

## Highlights & Insights

1. **Theoretical elegance**: Establishes the equivalence between CQL and explicit policy constraints in Proposition 3.1, providing a solid theoretical link that unifies both paradigms.
2. **State-level adaptation**: Deploys a neural network to bypass fixed global coefficients, resolving critical parameter tuning issues across tasks, training phases, and data densities.
3. **Selective constraints**: Constraining only the high-quality actions is an elegant and effective approach, yielding massive gains especially in sparse-reward AntMaze environments (boosting TD3+BC from 131.8 to 276.0).
4. **Seamless O2O transition**: Freezing the coefficient network coupled with simple linear decay facilitates highly efficient online fine-tuning, with the added benefit of privacy preservation by dropping the offline data.
5. **Plug-and-play**: Formulated as a general and versatile module that concurrently strengthens both value-regularized (CQL) and policy-constrained (TD3+BC) methods.

## Limitations & Future Work

1. **Additional computational overhead**: Training the coefficient network $\beta_\phi$ introduces extra complexity. High-variance datasets require further pre-training of the auxiliary IQL Q/V networks.
2. **Incomplete elimination of hyperparameters**: Parameters like $n_{start}$, $n_{end}$, $T_{inc}$, and the selective threshold $G_T$ still require calibration. Though much more robust than tuning a static $\beta$, it is not fully parameter-free.
3. **Continuous control limitation**: Evaluation remains restricted to Mujoco and AntMaze environments in D4RL, without extension to discrete action spaces or real-world physical systems.
4. **Dependence on data quality priors**: High- and low-variance datasets require distinct filtering procedures (return thresholding vs. advantage estimation), which may be challenging to predict automatically.
5. **Simplistic decay in O2O phases**: Utilizing more sophisticated decay functions or adaptive checkpointing schemes could further accelerate fine-tuning.

## Related Work & Insights

- **CQL** (Kumar et al., 2020): Representative value-regularized offline RL method, serving as an immediate base for improvement.
- **TD3+BC** (Fujimoto & Gu, 2021): A minimalist explicit policy constraint scheme, significantly enhanced by the integration of SSAR.
- **IQL** (Kostrikov et al., 2021): Provides advantage functions via expectile regression, which are utilized for quality filtering in SSAR.
- **Cal-QL** (Nakamoto et al., 2024): Another adaptive CQL variant; SSAR serves as a complementary mechanism in O2O setups.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The structural unification of CQL and policy constraints is theoretically sound, and the state-adaptive, selective constraint formulation is remarkably intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive task coverage across D4RL environments under both offline and O2O conditions, accompanied by robust ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and formal derivations, though math formula density is relatively high.
- **Value**: ⭐⭐⭐⭐ — A highly versatile plug-and-play component of significant utility to the offline RL research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[ICLR 2026\] Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/beyond_penalization_diffusion-based_out-of-distribution_detection_and_selective_.md)
- [\[ICML 2025\] Embedding Safety into RL: A New Take on Trust Region Methods](embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[ICLR 2026\] Sample Efficient Offline RL via T-Symmetry Enforced Latent State-Stitching](../../ICLR2026/reinforcement_learning/sample_efficient_offline_rl_via_t-symmetry_enforced_latent_state-stitching.md)

</div>

<!-- RELATED:END -->
