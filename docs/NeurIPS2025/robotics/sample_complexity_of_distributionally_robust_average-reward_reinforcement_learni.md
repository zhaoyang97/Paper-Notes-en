---
title: >-
  [Paper Note] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][Robotics][Distributionally robust optimization] This paper establishes the first finite-sample convergence guarantees for distributionally robust average-reward reinforcement learning (DR-AMDP), proposing two algorithms (discount reduction and anchoring) that achieve near-optimal sample complexity of $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$ under both KL and $f_k$-divergence uncertainty sets.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Distributionally robust optimization"
  - "average-reward reinforcement learning"
  - "sample complexity"
  - "Markov decision processes"
  - "KL divergence"
date: 2026-05-08
content_hash: ca467efa6e64f542
---

# Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.10007](https://arxiv.org/abs/2505.10007)  
**Authors**: Zijun Chen (HKUST), Shengbo Wang (USC), Nian Si (HKUST)
**Code**: Not released  
**Area**: Reinforcement Learning
**Keywords**: Distributionally robust optimization, average-reward reinforcement learning, sample complexity, Markov decision processes, KL divergence

## TL;DR

This paper establishes the first finite-sample convergence guarantees for distributionally robust average-reward reinforcement learning (DR-AMDP), proposing two algorithms (discount reduction and anchoring) that achieve near-optimal sample complexity of $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$ under both KL and $f_k$-divergence uncertainty sets.

## Background & Motivation

### State of the Field
Reinforcement learning has achieved remarkable success in robotics, operations research, healthcare, and related domains. However, standard RL assumes that training and deployment environments are identical—an assumption that frequently fails in practice. Distributionally robust RL (DR-RL) addresses environment mismatch by introducing adversarial perturbations to the transition probabilities.

### Limitations of Prior Work
- Existing DR-RL theory focuses primarily on the **discounted reward** or **finite-horizon** settings, where sample complexity is well understood.
- The **average-reward** setting is more relevant to many practical applications (e.g., steady-state robot control, inventory management, queueing systems), yet remains theoretically underdeveloped.
- Prior work on average-reward robust MDPs (Wang et al., 2023) establishes only asymptotic convergence, without **finite-sample complexity guarantees**.
- In the non-robust setting, the minimax sample complexity for discounted rewards was resolved as early as 2013, whereas analogous results for average-reward were established only recently.

### Root Cause
There is a fundamental theoretical gap in DR-RL under the average-reward criterion. The paper aims to fill this gap by establishing the first finite-sample convergence guarantees and designing practical algorithms that require no prior knowledge (e.g., mixing time $t_{\mathrm{mix}}$).

## Method

### Problem Formulation
- **Nominal MDP**: A tabular MDP $(S, A, r, P)$ with unknown transition kernel $P$, accessible via a generative model.
- **Uncertainty set**: SA-rectangular uncertainty sets defined under KL divergence or $f_k$-divergence: $\mathcal{P}_{s,a}(D,\delta) = \{p : D(p \| p_{s,a}) \leq \delta\}$.
- **Objective**: Learn the optimal robust average-reward policy $g_{\mathcal{P}}^* = \max_\pi \inf_{\mathbf{P} \in \mathcal{P}} g_{\mathbf{P}}^\pi$ under the uncertainty set.
- **Key assumptions**: The nominal MDP is uniformly ergodic (Assumption 1), and the adversarial perturbation radius $\delta$ is sufficiently small (Assumption 2).

### Key Theoretical Contribution 1: Stability of the Uncertainty Set
Conventional uncertainty sets may include non-unichain MDPs, causing standard Bellman equations to fail. Under Assumption 2, the paper proves that for all $Q \in \mathcal{P}$ and $\pi \in \Pi$, the minorization time satisfies $t_{\mathrm{minorize}}(Q_\pi) \leq 2 t_{\mathrm{minorize}}$, ensuring that adversarial perturbations do not destroy uniform ergodicity (Proposition 4.2).

### Algorithm 1: Reduction to DR-DMDP
1. Collect $n$ samples from the generative model to construct empirical transition probabilities $\hat{p}_{s,a}$.
2. Set a calibrated discount factor $\gamma = 1 - 1/\sqrt{n}$ to balance statistical error and reduction bias.
3. Solve the empirical DR discounted Bellman equation to obtain $V_{\hat{\mathcal{P}}}^*$ and the optimal policy $\hat{\pi}^*$.
4. Output: policy $\hat{\pi}^*$ and average-reward estimate $V_{\hat{\mathcal{P}}}^*/\sqrt{n}$.

### Algorithm 2: Anchored DR-AMDP
1. Construct empirical transition probabilities identically to Algorithm 1.
2. Select an anchor state $s_0$ and intensity parameter $\xi = 1/\sqrt{n}$.
3. Construct the anchored uncertainty set: $\underline{\hat{\mathcal{P}}}_{s,a} = \{(1-\xi)p + \xi \mathbf{1} e_{s_0}^\top : D(p\|\hat{p}_{s,a}) \leq \delta\}$.
4. Solve the empirical DR average-reward Bellman equation directly, avoiding the discounted subproblem.

### Sample Complexity Results
Both algorithms achieve the same sample complexity upper bound:

$$\widetilde{O}\left(|S||A| \cdot t_{\mathrm{minorize}}^2 \cdot \mathfrak{p}_\wedge^{-1} \cdot \varepsilon^{-2}\right)$$

where $\mathfrak{p}_\wedge$ denotes the minimum transition probability support. This result is optimal with respect to the dependence on $\varepsilon$ and $|S||A|$.

### Proof Mechanism
1. Proposition 4.2 establishes uniform ergodicity for all MDPs within the uncertainty set.
2. Policy error is reduced to estimation error of the DR Bellman operator.
3. Strong duality of DR functions and Bernstein-type inequalities are used to derive concentration bounds.
4. The final bound is completed via $\mathrm{Span}(V_{\mathcal{P}}^\pi) \leq O(t_{\mathrm{minorize}})$.

## Key Experimental Results

### Experiment 1: Convergence Rate Verification on Hard MDPs

Experiments use the Hard MDP family from Wang et al. (2023), consisting of 2 states and 2 actions, with mixing time controlled by parameter $\mathfrak{p}$.

| Setting | Uncertainty Set | Algorithm | Empirical Rate | Theoretical Prediction |
|---------|----------------|-----------|----------------|------------------------|
| Hard MDP | KL divergence | Algorithm 2 (Discount Reduction) | $n^{-1/2}$ slope | $O(n^{-1/2})$ |
| Hard MDP | $\chi^2$ divergence | Algorithm 2 (Discount Reduction) | $n^{-1/2}$ slope | $O(n^{-1/2})$ |
| Hard MDP | KL divergence | Algorithm 3 (Anchoring) | $n^{-1/2}$ slope | $O(n^{-1/2})$ |
| Hard MDP | $\chi^2$ divergence | Algorithm 3 (Anchoring) | $n^{-1/2}$ slope | $O(n^{-1/2})$ |

On a log-log scale, the empirical slope is approximately $-1/2$, confirming the $n^{-1/2}$ convergence rate. Each data point corresponds to a single independent run, with very low variance around the regression line.

### Experiment 2: Comparison of State-of-the-Art Sample Complexities

| Problem Type | Setting | Sample Complexity | Source |
|-------------|---------|-------------------|--------|
| Standard RL | Discounted | $\widetilde{\Theta}(|S||A|(1-\gamma)^{-3}\varepsilon^{-2})$ | Azar et al. 2013 |
| Standard RL | Discounted + Mixing | $\widetilde{\Theta}(|S||A|t_{\mathrm{mix}}(1-\gamma)^{-2}\varepsilon^{-2})$ | Wang et al. 2023 |
| Standard RL | Average + Mixing | $\widetilde{\Theta}(|S||A|t_{\mathrm{mix}}\varepsilon^{-2})$ | Wang et al. 2024 |
| DR-RL | Discounted | $\widetilde{O}(|S||A|(1-\gamma)^{-4}\varepsilon^{-2})$ | Shi & Chi; Wang et al. |
| **DR-RL** | **Discounted + Mixing** | $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2(1-\gamma)^{-2}\varepsilon^{-2})$ | **Ours (Thm 4.3)** |
| **DR-RL** | **Average + Mixing** | $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$ | **Ours (Thm 4.4 & 4.5)** |

This paper improves the effective horizon dependence for DR-DMDP from $(1-\gamma)^{-4}$ to $(1-\gamma)^{-2}$ and provides the first sample complexity result for DR-AMDP.

## Highlights & Insights

- **Pioneering contribution**: This is the first work to establish finite-sample guarantees for distributionally robust average-reward RL, closing a significant theoretical gap.
- **Dual algorithm design**: The discount reduction method elegantly connects average-reward to discounted settings; the anchoring method operates directly in the average-reward framework, avoiding subproblems.
- **Stability analysis of uncertainty sets**: The paper rigorously establishes sufficient conditions under which adversarial perturbations preserve uniform ergodicity, resolving a key modeling challenge in DR-AMDP.
- **No prior knowledge required**: Both algorithms adapt parameter selection without requiring knowledge of $t_{\mathrm{mix}}$ in advance.
- **Optimal dependence**: The results achieve information-theoretic lower bounds in $|S||A|$ and $\varepsilon$; the $(1-\gamma)^{-2}$ effective horizon dependence is known to be optimal in the non-robust setting.

## Limitations & Future Work

- **Strong ergodicity assumption**: Uniform ergodicity is required across all policies, excluding weakly communicating and multichain MDPs.
- **Constraint on uncertainty set radius**: Assumption 2 requires $\delta \leq O(\mathfrak{p}_\wedge / m_\vee^2)$, which limits adversarial power.
- **$t_{\mathrm{mix}}^2$ vs. $t_{\mathrm{mix}}$**: Compared to the non-robust $\widetilde{\Theta}(t_{\mathrm{mix}})$ result, this paper incurs an additional $t_{\mathrm{mix}}$ factor; whether this gap can be closed remains open.
- **Tabular setting**: The results apply only to finite state-action spaces and do not extend to function approximation.
- **Limited divergence classes**: Only KL and $f_k$-divergences are considered; $\ell_p$-ball and Wasserstein uncertainty sets are not covered.
- **Generative model assumption**: The algorithms require sampling from arbitrary $(s,a)$ pairs, which is incompatible with online interaction settings.

## Related Work & Insights

- **Shi & Chi (2023), Wang et al. (2024c)**: Achieve $\widetilde{O}((1-\gamma)^{-4}\varepsilon^{-2})$ for DR-DMDP under KL uncertainty sets; this paper improves the dependence to $(1-\gamma)^{-2}$ by exploiting mixing structure.
- **Clavier et al. (2024)**: Establish the minimax rate $\widetilde{\Theta}((1-\gamma)^{-3}\varepsilon^{-2})$ under $\ell_p$-norm constraints; this paper does not cover such uncertainty set types.
- **Wang et al. (2023d/e, 2024d)**: Provide asymptotic convergence results for robust average-reward RL without finite-sample guarantees.
- **Grand-Clément et al. (2025)**: Study the structure of optimal policies under $(s,a)$-rectangular uncertainty sets, without addressing sample complexity.
- **Wang & Si (2024)**: Establish existence of Bellman equations under $s$-rectangular uncertainty sets; this paper provides quantitative results in the SA-rectangular setting.
- **Jin & Sidford (2020), Wang et al. (2024a)**: Achieve the optimal $\widetilde{\Theta}(t_{\mathrm{mix}}\varepsilon^{-2})$ for non-robust average-reward RL; the robust version in this paper incurs an additional $t_{\mathrm{mix}}$ factor.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First resolution of the open problem of sample complexity for DR-AMDP.
- Experimental Thoroughness: ⭐⭐⭐ — Validation is limited to simple Hard MDPs; larger-scale and real-world experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, theoretically complete, with thorough contextual citations.
- Value: ⭐⭐⭐⭐ — Fills an important theoretical gap, though strong assumptions limit practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)
- [\[AAAI 2026\] Distributionally Robust Online Markov Game with Linear Function Approximation](../../AAAI2026/robotics/distributionally_robust_online_markov_game_with_linear_function_approximation.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

</div>

<!-- RELATED:END -->
