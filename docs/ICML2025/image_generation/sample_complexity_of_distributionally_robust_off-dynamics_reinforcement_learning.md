---
title: >-
  [Paper Note] Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction
description: >-
  [ICML2025][Image Generation][Robust MDP] Proposes the supremal visitation ratio $C_{vr}$ to measure the exploration difficulty of online robust MDPs, designs ORBIT, the first efficient online algorithm supporting general $f$-divergences (TV/KL/$\chi^2$), and provides matching upper and lower bounds, proving that $C_{vr}$ is a tight measure characterizing the online learnability of off-dynamics RL.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Robust MDP"
  - "Off-dynamics RL"
  - "Sample Complexity"
  - "online learning"
  - "f-divergence"
  - "Regret Bounds"
date: 2026-05-08
content_hash: 34a85c9d06220a06
---

# Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction

**Conference**: ICML2025  
**arXiv**: [2511.05396](https://arxiv.org/abs/2511.05396)  
**Code**: [panxulab/Online-Robust-Bellman-Iteration](https://github.com/panxulab/Online-Robust-Bellman-Iteration)  
**Area**: Image Generation  
**Keywords**: Robust MDP, Off-dynamics RL, Sample Complexity, online learning, f-divergence, Regret Bounds

## TL;DR

Proposes the supremal visitation ratio $C_{vr}$ to measure the exploration difficulty of online robust MDPs, designs ORBIT, the first efficient online algorithm supporting general $f$-divergences (TV/KL/$\chi^2$), and provides matching upper and lower bounds, proving that $C_{vr}$ is a tight measure characterizing the online learnability of off-dynamics RL.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Off-dynamics RL addresses the discrepancy in transition dynamics between the training and deployment environments, typically modeled as a Robust Markov Decision Process (RMDP). Prior work primarily relies on two settings:

**Generative Model Setting**: Allows querying transitions for any $(s,a)$, avoiding exploration difficulties.

**Offline Dataset Setting**: Assumes the pre-collected data has good coverage over the deployment environment.

Both settings are overly optimistic in practice. A more realistic scenario is where the agent can only perform **online interaction** with the training environment, yet needs to learn a policy that performs well under the deployment environment (where distribution shift may occur).

**Key Challenge - Information Deficit**: States $s$ that are rarely visited in the nominal environment may become crucial due to dynamics shifts in the deployment environment. While the impact of these rare states is negligible in a standard MDP, they can be critical factors in worst-case decision-making within an RMDP.

Existing work on online RMDPs is limited to the special case of **TV distance + fail-state assumption**, lacking a viable online learning theory for more general $f$-divergences (KL/$\chi^2$). This work provides the first systematic answer to: **Under what conditions can provably efficient online learning be achieved in general $f$-divergence RMDPs?**

## Method

### Framework: CRMDP and RRMDP

The paper unifies the treatment of two robust MDP frameworks:

| Framework | Objective | Uncertainty Modeling |
|------|------|-------------|
| **CRMDP** (Constrained Robust MDP) | $\max_\pi \min_{P \in \mathcal{U}^\rho} V^\pi$ | Uncertainty set $D_f(P \| P^o) \le \rho$ |
| **RRMDP** (Regularized Robust MDP) | $\max_\pi \min_P V^\pi + \beta \cdot D_f(P, P^o)$ | Regularization penalizing deviation |

Both adopt the $(s,a)$-rectangular uncertainty set structure, considering three $f$-divergences: TV, KL, and $\chi^2$.

### Supremal Visitation Ratio $C_{vr}$

The core contribution of this work is defining a new metric to measure the exploration difficulty of online RMDPs:

$$C_{vr} := \sup_{\pi, h, s} \frac{q_h^\pi(s)}{d_h^\pi(s)}$$

where $d_h^\pi(s)$ is the probability of policy $\pi$ visiting state $s$ at step $h$ under the nominal environment, and $q_h^\pi(s)$ is the corresponding probability under the worst-case transition.

- When $C_{vr} = 1$, it degenerates to a standard (non-robust) MDP.
- When $C_{vr}$ is bounded (polynomial with respect to $S, A, H$), online learning is feasible.
- When $C_{vr}$ is unbounded, hard instances exist that make any algorithm exponentially difficult.

### ORBIT Algorithm

**Online Robust Bellman Iteration (ORBIT)** is the meta-algorithm proposed in this paper, with the following core steps:

1. **Backward Value Iteration**: From $h = H$ to $1$, estimate the $Q$-function using the robust Bellman equation.
2. **Optimistic Exploration**: Incorporate a bonus term $b_h^k(s,a)$ to achieve optimism in the face of uncertainty.
3. **Execute Greedy Policy** to collect trajectories and update the empirical transition kernel $\hat{P}_h^{k+1}$.

The unified formulation of the $Q$-function estimate:

$$\hat{Q}_h^k(s,a) = \min\{RB_h^k(s,a) + b_h^k(s,a),\; H - h + 1\}$$

where $RB_h^k$ is the robust Bellman estimator (solved via dual reformulation of the inner optimization), and $b_h^k$ is the exploration bonus.

**Dual formulations differ across divergences**:

- **TV**: The inner optimization is reformulated as a one-dimensional search over $\eta$.
- **KL**: Utilizing the duality of exponential functions, RRMDP-KL admits a closed-form solution.
- **$\chi^2$**: Reformulated as a variance optimization over $\lambda$.

## Theoretical Results

### Regret Upper Bounds (Theorem 5.9 & 5.16)

| Setting | CRMDP Regret Upper Bound | RRMDP Regret Upper Bound |
|------|---------------|---------------|
| **TV** | $\tilde{O}(C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ | $\tilde{O}(C_{vr}^{1/2} S A^{1/2} H^2 \sqrt{K})$ |
| **KL** | $\tilde{O}((1 + \frac{H\sqrt{S}}{\rho C_{MP}}) C_{vr}^{1/2} S^{1/2} A^{1/2} H \sqrt{K})$ | $\tilde{O}((1+\beta e^{\beta^{-1}H}\sqrt{S}) C_{vr}^{1/2} S^{1/2} A^{1/2} H \sqrt{K})$ |
| **$\chi^2$** | $\tilde{O}((1+\sqrt{\rho}) C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ | $\tilde{O}((1+\frac{H}{\beta}) C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ |

### Regret Lower Bounds (Theorem 5.12 & 5.18)

For all three divergences (TV/KL/$\chi^2$), both CRMDP and RRMDP satisfy:

$$\mathbb{E}[\text{Regret}(K)] = \Omega(C_{vr}^{1/2} \sqrt{K})$$

**Key Conclusion**: The orders of $C_{vr}$ and $K$ in the leading terms of the upper bounds tightly match those of the lower bounds, demonstrating that $C_{vr}$ is a tight measure of exploration difficulty in online RMDPs.

### Hard Instance (Lemma 5.14)

When $C_{vr} = 2^{2A}$, the regret of any algorithm is $\Omega(2^A \sqrt{K})$, which is exponentially hard. This proves that a bounded $C_{vr}$ is a necessary condition for online learnability.

### Loss & Training
The model is trained end-to-end, and the optimization objective comprehensively accounts for both task loss and regularization terms.


## Key Experimental Results

1. **Verification of the impact of $C_{vr}$** (synthetic MDP, $H=3, S=6, A=10$): As $C_{vr}$ increases, the gap between the learned policy and the optimal policy continuously widens, consistent with the theory.
2. **Simulated RMDP** ($H=3, S=5, A=5$): When the perturbation exceeds 0.6, all robust policies (CRMDP/RRMDP with TV/KL/$\chi^2$) outperform non-robust algorithms.
3. **Frozen Lake**: ORBIT converges stably during training; RRMDP-TV/KL possesses closed-form dual solutions and trains the fastest; CRMDP-$\chi^2$ incurs the highest computational cost.

### Key Findings
- The primary components/modules contribute to the most critical performance gains.


## Highlights & Insights

- **Highly Complete Theory**: Unifies the treatment of 6 settings (2 frameworks $\times$ 3 divergences) under a single framework, with matching upper and lower bounds for each.
- **Elegant Concept of $C_{vr}$**: Unifies special assumptions like fail-state into a single scalar metric and proves its tightness.
- **Novel Information Deficit Perspective**: Understands the problem through the mismatch of visitation distributions between nominal and worst-case environments, explaining why the fail-state assumption is effective (Proposition 5.4).
- **Evidence of RRMDP Outperforming CRMDP**: The regret of RRMDP-TV is smaller than CRMDP-TV by a factor of $\sqrt{S}$; RRMDP-KL requires no additional assumptions and admits a closed-form solution.

## Limitations & Future Work

1. **Tabular Setting Only**: Both the algorithm and theory are restricted to finite state-action spaces and have not been extended to function approximation.
2. **$C_{vr}$ Agnostic in Practice**: $C_{vr}$ cannot be calculated prior to deployment (as it depends on the unknown worst-case transitions), making it difficult to anticipate the algorithm's performance.
3. **CRMDP-KL Requires Additional Assumptions** (Assumption 5.7): It requires transition probabilities to have a lower bound $C_{MP}$, limiting its scope of application.
4. **Large Factor $\beta e^{\beta^{-1}H}$ in RRMDP-KL**: This factor can be large, leading to regret degradation under long horizons for KL regularization.
5. **Upper and Lower Bounds Do Not Fully Match in $S, A, H$**: They are optimal only with respect to $C_{vr}$ and $K$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Generates matching upper and lower bounds for general f-divergence online RMDPs for the first time)
- Experimental Thoroughness: ⭐⭐⭐ (Small scale of experiments, verified only in tabular environments)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theory with abundant notation but maintaining clear logic)
- Value: ⭐⭐⭐⭐⭐ (Makes a critical contribution to the theoretical foundation of online learning in off-dynamics RL)


## Related Work & Insights
- **vs Representative Methods in the Same Field**: This work makes unique contributions to method design, complementing existing approaches.
- **vs Traditional Methods**: Compared to traditional schemes, the proposed method achieves significant improvements in key metrics.
- **Insights**: The technical route of this work holds valuable reference relevance for subsequent related work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ORVIT: Near-Optimal Online Distributionally Robust Reinforcement Learning](../../AAAI2026/image_generation/orvit_near-optimal_online_distributionally_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](../../NeurIPS2025/image_generation/towards_robust_zero-shot_reinforcement_learning.md)
- [\[ICML 2025\] PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](pak-ucb_contextual_bandit_an_online_learning_approach_to_prompt-aware_selection_.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICML 2025\] Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals](hierarchical_reinforcement_learning_with_uncertainty-guided_diffusional_subgoals.md)

</div>

<!-- RELATED:END -->
