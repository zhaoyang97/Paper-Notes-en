---
title: >-
  [Paper Note] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][general utility reinforcement learning] This paper establishes global optimality guarantees for policy gradient methods in reinforcement learning with general utilities (RLGU): in the tabular setting, global convergence is proved via a novel gradient dominance inequality; in large-scale state-action spaces, an occupancy measure approximation algorithm PG-OMA based on maximum likelihood estimation (MLE) is proposed, whose sample complexity depends only on the dimension $m$ of the function approximation class rather than the size of the state-action space.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - general utility reinforcement learning
  - policy gradient
  - gradient dominance inequality
  - occupancy measure estimation
  - maximum likelihood estimation
date: 2026-05-08
content_hash: 48a6894a1f562158
---

# On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2410.04108](https://arxiv.org/abs/2410.04108)
**Code**: None
**Area**: Reinforcement Learning Theory
**Keywords**: general utility reinforcement learning, policy gradient, gradient dominance inequality, occupancy measure estimation, maximum likelihood estimation

## TL;DR

This paper establishes global optimality guarantees for policy gradient methods in reinforcement learning with general utilities (RLGU): in the tabular setting, global convergence is proved via a novel gradient dominance inequality; in large-scale state-action spaces, an occupancy measure approximation algorithm PG-OMA based on maximum likelihood estimation (MLE) is proposed, whose sample complexity depends only on the dimension $m$ of the function approximation class rather than the size of the state-action space.

## Background & Motivation

**Background**: Reinforcement learning with general utilities (RLGU, also known as convex RL) provides a unified framework encompassing a wide range of problems beyond standard expected-return RL, including imitation learning, pure exploration, safe RL, skill discovery, and experimental design. While standard RL optimizes a linear functional of the occupancy measure, RLGU generalizes the objective to an arbitrary (possibly nonlinear) functional thereof.

**Limitations of Prior Work**: Although policy gradient methods in standard RL are supported by mature global optimality theory (e.g., the gradient dominance inequality established by Agarwal et al. 2021), these results do not directly extend to RLGU. Existing RLGU work is primarily limited in two respects: (1) global optimality proofs in the tabular setting rely on "hidden convexity" techniques, lacking a direct connection to standard RL policy gradient analysis; (2) the vast majority of algorithms are applicable only to the tabular setting, using count-based Monte Carlo estimators for occupancy measure estimation that cannot scale to large state-action spaces.

**Key Challenge**: The RLGU objective is nonconcave in the policy parameters, and computing the "pseudo-reward" requires estimating the unknown occupancy measure. In large-scale spaces, counting over individual state-action pairs is computationally and storage-prohibitive. Moreover, existing mean-squared-error (MSE) approximation methods have theoretical guarantees that depend on the size of the state space.

**Goal**: (1) Can the gradient dominance structure of standard RL be extended to RLGU? (2) Can policy gradient algorithms be designed whose sample complexity in large-scale spaces is independent of the state-action space size?

**Key Insight**: The authors observe that the RLGU policy gradient can be expressed as the standard RL policy gradient evaluated at the pseudo-reward $\nabla_\lambda F(\lambda(\theta))$ (Equation 5). This key identity directly connects the RLGU gradient to the standard RL gradient. Leveraging this insight together with a convexity assumption, the gradient dominance property of standard RL can be extended to RLGU. For scalability, MLE is adopted in place of MSE for occupancy measure estimation, as the total variation error bound of MLE depends only on the parameter dimension.

**Core Idea**: Establish a gradient dominance inequality by reducing the RLGU policy gradient to a standard RL gradient evaluated at a pseudo-reward, and approximate the occupancy measure via MLE to enable scalability to large spaces.

## Method

### Overall Architecture

The paper considers a discounted MDP $(S, A, P, F, \rho, \gamma)$, where $F$ is a general utility function defined on the space of occupancy measures. The objective is $\max_\theta F(\lambda^{\pi_\theta})$. The algorithmic development proceeds at two levels: establishing structural theoretical properties (gradient dominance) in the tabular setting, and designing a practical actor-critic-style algorithm PG-OMA for large-scale settings.

### Key Designs

1. **RLGU Gradient Dominance Inequality (Tabular Setting)**

   - *Function*: Proves that the RLGU objective satisfies a gradient dominance property under direct policy parameterization, implying that every stationary point is a global optimum.
   - *Mechanism*: By the chain rule, the RLGU gradient is decomposed as $\nabla_\theta F(\lambda(\theta)) = [\nabla_\theta \lambda(\theta)]^T \nabla_\lambda F(\lambda(\theta))$, which coincides exactly with the standard RL policy gradient evaluated at the pseudo-reward $r_\theta = \nabla_\lambda F(\lambda(\theta))$. The standard RL gradient dominance result (Agarwal 2021, Lemma 4) is then applied to this pseudo-reward, and the concavity of $F$ is used to translate the value function gap into a utility gap: $V^{\pi^*(r_\theta)}(r_\theta) - V^{\pi_\theta}(r_\theta) \geq \langle r_\theta, \lambda^{\pi^*} - \lambda^{\pi_\theta} \rangle \geq F(\lambda(\theta^*)) - F(\lambda(\theta))$.
   - *Design Motivation*: To directly connect the optimization structure of RLGU with existing theory for standard RL, thereby opening a pathway for analyzing softmax and other parameterizations within the RLGU framework.

2. **MLE-Based Occupancy Measure Approximation (PG-OMA Algorithm)**

   - *Function*: Scalably estimates the occupancy measure in large state-action spaces.
   - *Mechanism*: The normalized state occupancy measure $d^{\pi_\theta}$ is treated as a probability distribution and approximated via maximum likelihood estimation within a parametric family $\Lambda = \{p_\omega : \omega \in \Omega \subseteq \mathbb{R}^m\}$: $\omega^* = \arg\max_\omega \frac{1}{n}\sum_{i=1}^n \log p_\omega(s_i)$. The key theoretical result is a total variation error bound $\|\hat\lambda - \lambda\|_1 \leq O(\sqrt{m/n})$ that depends only on the approximation class dimension $m$ and is independent of the state-action space size.
   - *Design Motivation*: Compared to MSE-based methods, MLE is naturally suited to probability distribution estimation and is unaffected by the size of the space. The authors illustrate this with a simple yet instructive counterexample: when the true distribution is uniform, the MSE of a non-uniform estimator vanishes as the space grows, rendering it unable to distinguish good from poor estimates, whereas the MLE total variation error remains informative throughout.

3. **Two-Phase Iteration of PG-OMA**

   - *Function*: At each iteration, first estimates the occupancy measure (Critic step), then updates the policy parameters (Actor step).
   - *Mechanism*: Each iteration proceeds in two steps: (i) approximate the occupancy measure $\hat\lambda_t$ via MLE from states sampled under the current policy, and compute the pseudo-reward $\hat r_t = \nabla_\lambda F(\hat\lambda_t)$; (ii) perform stochastic policy gradient ascent using the REINFORCE estimator under pseudo-reward $\hat r_t$. The entire procedure requires no estimation of the transition kernel and is model-free.
   - *Design Motivation*: Decoupling occupancy measure estimation into an independent statistical learning subproblem exploits the statistical efficiency of MLE; the pseudo-reward need only be evaluated at state-action pairs visited along the current trajectory (Remark 4), further reducing memory requirements.

### Loss & Training

The policy gradient is approximated via the REINFORCE estimator with step size satisfying $\alpha_t \leq 1/(2L_\theta)$. For nonconcave utilities, convergence to a first-order stationary point is guaranteed; for concave utilities, global optimality of the last iterate is established by combining the hidden convexity technique with a policy overparameterization assumption (Assumption 4.3). The key parameters governing total sample complexity are the function approximation dimension $m$, accuracy $\epsilon$, and discount factor $\gamma$.

## Key Experimental Results

### Main Results

This is a purely theoretical paper. The following presents a theoretical comparison with the most closely related prior work:

| Method | First-Order Stationarity Complexity | Global Optimality Complexity | Beyond Tabular | No $|S \times A|$ Dependence |
|---|---|---|---|---|
| Zhang et al. 2020 | $\tilde{O}(\epsilon^{-2})$* | $\tilde{O}(\epsilon^{-1})$* | ✘ | ✘ |
| Zhang et al. 2021 | $\tilde{O}(\epsilon^{-3})$ | $\tilde{O}(\epsilon^{-2})$ | ✘ | ✘ |
| Barakat et al. 2023 (sec. 5) | $\tilde{O}(\epsilon^{-4})$ | ✘ | ✓ | ✘ |
| **Ours (PG-OMA)** | $\tilde{O}(m\epsilon^{-4})$ | $\tilde{O}(m\epsilon^{-4})$ | ✓ | ✓ |

*Deterministic setting; iteration count only.

### Ablation Study

| Dimension | Ours | Barakat et al. 2023 | Huang & Jiang 2024 |
|---|---|---|---|
| Global convergence guarantee | ✓ (concave utility) | ✘ (first-order only) | ✘ (first-order only) |
| Occupancy measure estimation | MLE | MSE | MLE + recursive regression |
| No $|S|$ dependence | ✓ | ✘ (implicit $1/\rho_{min}$) | Not specified |
| Last-iterate guarantee | ✓ | ✓ | ✘ (best-iterate) |

### Key Findings

- MLE-based occupancy measure estimation is strictly superior to MSE in theory: MSE cannot detect distributional differences in large spaces (uniform distribution counterexample), whereas the MLE total variation bound is independent of space size.
- The distribution mismatch coefficient in the gradient dominance inequality depends on the pseudo-reward $\nabla_\lambda F(\lambda(\theta))$; when $F$ is linear, it reduces to the constant coefficient of standard RL.
- The proof uses concavity at only a single point $\lambda^{\pi^*}$, suggesting that the results may be extendable to weaker local concavity conditions.

## Highlights & Insights

- The **policy gradient–pseudo-reward equivalence** is the central contribution: $\nabla_\theta F(\lambda(\theta))$ equals exactly the standard policy gradient evaluated at $r = \nabla_\lambda F(\lambda(\theta))$. This brings the full toolkit of standard RL analysis (gradient dominance, variance reduction, etc.) into the RLGU setting, representing a significant methodological advance.
- The **scalability analysis of MLE vs. MSE** is an independent theoretical contribution: the counterexample clearly demonstrates the fundamental limitation of MSE for probability distribution estimation—when estimating a non-uniform distribution over a uniform base, the MSE loss is $O(1/|X|^2)$, which vanishes as the space grows, making it impossible to distinguish estimator quality.
- The proof technique extends to softmax policy parameterization; the authors explicitly note that Lemma 8 of Mei et al. 2020 can be applied, paving the way for future work.

## Limitations & Future Work

- The policy overparameterization assumption (Assumption 4.3) is difficult to verify for practical neural network policies.
- The theoretical analysis is restricted to finite state-action spaces; extension to continuous spaces remains a major open problem.
- The $\tilde{O}(m\epsilon^{-4})$ complexity for concave utilities is two orders worse than the tabular optimum of $\tilde{O}(\epsilon^{-2})$.
- As a purely theoretical paper, no numerical experiments are provided.
- The MLE subproblem is assumed to be solvable to global optimality, whereas it is in practice a nonconvex optimization problem.

## Related Work & Insights

- **vs. Zhang et al. 2021**: Their approach uses hidden convexity to directly prove global optimality without connecting to standard RL analysis. The present paper establishes this connection via gradient dominance.
- **vs. Barakat et al. 2023**: MSE-based estimation introduces $|S|$ dependence and yields only first-order guarantees. The MLE scheme proposed here entirely eliminates space-size dependence.
- **vs. Huang & Jiang 2024**: Their approach requires additional estimation of the log-gradient occupancy measure, whereas the present paper requires only a single MLE step.

## Rating

- Novelty: ⭐⭐⭐⭐ Significant theoretical contributions in extending gradient dominance and proposing MLE as an alternative estimation scheme.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous structure with thorough comparison to prior work.
- Value: ⭐⭐⭐⭐ Establishes theoretical foundations for scalable algorithms in RLGU.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)

<!-- RELATED:END -->
