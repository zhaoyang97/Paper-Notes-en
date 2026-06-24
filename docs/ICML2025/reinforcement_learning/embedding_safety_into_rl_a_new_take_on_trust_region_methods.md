---
title: >-
  [Paper Note] Embedding Safety into RL: A New Take on Trust Region Methods
description: >-
  [ICML 2025][Reinforcement Learning][Safe Reinforcement Learning] The C-TRPO algorithm is proposed, which modifies the geometry of the policy space (by embedding a constraint-aware barrier term into the KL divergence) so that the trust region naturally contains only safe policies. This guarantees constraint satisfaction throughout the entire training process while maintaining return performance comparable to SOTA.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Safe Reinforcement Learning"
  - "Trust Region Methods"
  - "Constrained MDP"
  - "Natural Policy Gradient"
  - "Barrier Functions"
date: 2026-05-08
content_hash: 6a62fbcf43aa7193
---

# Embedding Safety into RL: A New Take on Trust Region Methods

**Conference**: ICML 2025  
**arXiv**: [2411.02957](https://arxiv.org/abs/2411.02957)  
**Code**: [github.com/milosen/ctrpo](https://github.com/milosen/ctrpo)  
**Area**: Reinforcement Learning  
**Keywords**: Safe Reinforcement Learning, Trust Region Methods, Constrained MDP, Natural Policy Gradient, Barrier Functions

## TL;DR

The C-TRPO algorithm is proposed, which modifies the geometry of the policy space (by embedding a constraint-aware barrier term into the KL divergence) so that the trust region naturally contains only safe policies. This guarantees constraint satisfaction throughout the entire training process while maintaining return performance comparable to SOTA.

## Background & Motivation

The core framework of safe reinforcement learning is the **Constrained Markov Decision Process (CMDP)**, which requires maximizing cumulative rewards while satisfying cost constraints $V_{c_i}^\pi(\mu) \le b_i$. Existing methods suffer from three main **Limitations of Prior Work**:

**Lagrangian Methods** (PPO-Lag, TRPO-Lag): These convert constraints into penalty terms. However, the update of dual variables is unstable, easily oscillating near the constraint boundary and causing frequent constraint violations during training.

**Penalty/Barrier Methods** (IPO, P3O): These simplify optimization by introducing penalty terms with fixed weights. However, they alter the original objective function, leading to **optimization bias**—the optimal solution is no longer the optimal solution of the original CMDP.

**CPO-like Trust Region Methods**: These optimize within the intersection of the trust region and the safe policy set. Although theoretically safe, they rely in practice on noisy cost advantage estimates, leading to high-frequency oscillations and overshooting near the constraint boundary.

The key insight of this paper: The state-averaged KL divergence $D_K$ used by TRPO can be viewed as the Bregman divergence induced by the negative conditional entropy $\Phi_K$. If this mirror function $\Phi$ is modified to approach infinity at the constraint boundary, any trust region with a finite radius will never touch the boundary—essentially eliminating the possibility of violating constraints from a geometric perspective.

## Method

### Overall Architecture

The core idea of C-TRPO is to **embed constraint information into the metric structure of the policy space**, rather than treating constraints as additional optimization constraints. Specifically, it consists of three steps:

1. **Constructing the Safe Mirror Function $\Phi_C$**: Superimpose a convex barrier term regarding cost constraints on top of the standard conditional entropy.
2. **Deriving the Constrained KL Divergence $D_C$**: The Bregman divergence induced by the safe mirror function.
3. **Designing a Practical Approximation Algorithm**: Replace the exact divergence with a surrogate divergence $\bar{D}_C$, combined with conjugate gradient and backtracking line search.

The overall algorithmic flow: At each step, first check if the current policy is safe. If safe, execute a constrained trust region update (using $\bar{D}_C$ as the divergence); if unsafe, execute a recovery step (minimizing only the cost).

### Key Designs

#### 1. Safe Mirror Function

The mirror function used in standard TRPO is the negative conditional entropy:

$$\Phi_K(d_\pi) = \sum_{s,a} d_\pi(s,a) \log \pi(a|s)$$

C-TRPO extends this to:

$$\Phi_C(d) = \Phi_K(d) + \sum_{i=1}^{m} \beta_i \phi(b_i - c_i^\top d)$$

where $\phi: \mathbb{R}_{>0} \to \mathbb{R}$ is a convex function satisfying $\phi'(x) \to +\infty$ as $x \searrow 0$. This means that when the occupancy measure $d$ approaches the constraint boundary $b_i - c_i^\top d = 0$, the gradient of the mirror function approaches infinity. In the authors' experiments, using $\phi(x) = x\log(x)$ (entropy form) outperforms the logarithmic barrier $\phi(x) = -\log(x)$.

#### 2. Constrained KL Divergence

The Bregman divergence induced by $\Phi_C$ is:

$$D_C(d_1 \| d_2) = D_K(d_1 \| d_2) + \sum_{i=1}^{m} \beta_i D_{\phi_i}(d_1 \| d_2)$$

where $D_{\phi_i}$ is the Bregman divergence term with respect to the $i$-th constraint. **Key Property**: $D_C$ is defined only inside the safe occupancy measure set $\mathscr{D}_{\text{safe}}$, and the divergence approaches infinity as $d_2$ approaches the constraint boundary. Therefore, for any finite $\delta$, the trust region $\{d : D_C(d_k \| d) \le \delta\}$ is guaranteed to be contained within the safe set.

#### 3. Surrogate Divergence (Practical Computable Version)

The exact $D_C$ depends on the cost return $V_c(\pi)$ of the new policy, which is unavailable before the update. C-TRPO replaces it with a surrogate divergence:

$$\bar{D}_C(\pi \| \pi_k) = \bar{D}_{KL}(\pi \| \pi_k) + \beta \bar{D}_\phi(\pi \| \pi_k)$$

where $\bar{D}_\phi$ approximates the cost return difference $V_c(\pi) - V_c(\pi_k)$ using the **cost policy advantage** $\mathbb{A}_c^{\pi_k}(\pi)$. Defining the constraint margin as $\delta_b = b - V_c^{\pi_k}$, the surrogate divergence term is:

$$\bar{D}_\phi(\pi \| \pi_k) = \phi(\delta_b - \mathbb{A}_c^{\pi_k}(\pi)) - \phi(\delta_b) + \phi'(\delta_b) \mathbb{A}_c^{\pi_k}(\pi)$$

This surrogate is consistent with the second-order expansion of the exact divergence at the policy parameters $\theta$, theoretically providing sufficient accuracy.

#### 4. Relationship and Differences with CPO

- **When $\beta \to 0$**: C-TRPO degenerates to CPO (where constraints enter as original linear constraints within the trust region constraint).
- **When $\beta > 0$**: C-TRPO is more conservative than CPO—the update direction is deflected to be more parallel to the constraint surface rather than crossing it directly.
- **When $\beta \to +\infty$**: Updates in the direction of increasing cost are completely suppressed.

Compared to CPO, the inner optimization of C-TRPO is simpler: it only needs to approximate a single quadratic constraint, whereas CPO needs to handle both a quadratic constraint (trust region) and a linear constraint (safety) simultaneously.

#### 5. Hysteretic Recovery Mechanism

When the policy leaves the safe set due to estimation errors, C-TRPO executes a recovery step. To avoid repeated oscillations near the constraint boundary, a **hysteretic condition** is introduced: the recovery target is not just to return to $V_c \le b$, but rather to return to a stricter $V_c \le b_H$ (where $b_H = 0.8b$), leaving a safety margin for subsequent updates.

### Loss & Training

The parameter update in C-TRPO follows the same framework as TRPO:

$$\theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{g_k^\top H_k^{-1} g_k}} \cdot H_k^{-1} g_k$$

where $g_k = \nabla_\theta \mathbb{A}_r^{\pi_k}$ is the reward advantage gradient, and $H_k = \nabla_\theta^2 \bar{D}_C(\pi_\theta \| \pi_{\theta_k})$ is the Hessian of the constrained divergence, which is approximated using the conjugate gradient method to compute $H^{-1}g$. $\alpha^i$ is determined via backtracking line search to ensure $\bar{D}_C \le \delta$.

The exact form of the Hessian is:

$$\bar{H}_C(\theta_k) = G_K(\theta_k) + \beta \phi''(b - V_c(\theta)) \nabla_\theta V_c(\theta) \nabla_\theta V_c(\theta)^\top$$

which is the standard Fisher Information Matrix plus a **rank-one correction** of the cost gradient. Its computational cost is comparable to CPO, requiring only the evaluation of the cost value function in addition to TRPO.

**C-NPG (Constrained Natural Policy Gradient)** is the continuous-time limit of C-TRPO: $\partial_t \theta_t = G_C(\theta_t)^+ \nabla V_r(\theta_t)$. Theoretical analysis shows that C-NPG guarantees safe-set invariance (Theorem 4.4) and global convergence to the optimal safe policy (Theorem 4.5).

## Key Experimental Results

### Main Results

Compared with 9 safe RL algorithms on the Safety Gymnasium benchmark, across 4 navigation tasks + 4 locomotion tasks, with each task run with 5 seeds for 10M steps.

| Algorithm | Final Cost (Normalized, 0=threshold) | Reward (Normalized, PPO=1) | Cost Regret (Normalized, CPO=1) | Converge Safe? |
|------|------|------|------|------|
| **C-TRPO** | **< 0 (Safe)** | **~0.85** | **~0.6** | **✓** |
| CPO | > 0 (Unsafe) | ~0.9 | 1.0 | ✗ |
| PCPO | < 0 (Safe) | ~0.7 | ~0.5 | ✓ |
| PPO-Lag | < 0 (Safe) | ~0.8 | ~0.8 | ✓ |
| TRPO-Lag | < 0 (Safe) | ~0.8 | ~1.2 | ✓ |
| FOCOPS | > 0 (Unsafe) | ~0.9 | ~0.9 | ✗ |
| CUP | > 0 (Unsafe) | ~0.85 | ~0.7 | ✗ |
| P3O | < 0 (Safe) | ~0.6 | ~0.3 | ✓ |
| IPO | < 0 (Safe) | ~0.65 | ~0.7 | ✓ |

### Ablation Study

| Configuration | Final Cost | Reward | Description |
|------|------|------|------|
| C-TRPO (Full, β=1, b_H=0.8b) | Safe | High | Default configuration |
| W/o $\bar{D}_\phi$ (KL only) | Unsafe | High | Degenerates to approximate CPO, constraint violations increase |
| W/o Hysteresis (b_H=b) | Boundary Oscillation | High | Violations occur again immediately after recovery |
| β=0.01 | Slightly Unsafe | High | Close to CPO behavior |
| β=10 | Very Safe | Decreased | Overly conservative, noise amplified |
| $\phi(x)=-\log(x)$ (Log barrier) | Safe | Slightly lower | Entropy form works better |
| $\phi(x)=x\log(x)$ (Entropy) | Safe | Higher | Default choice |

### Key Findings

1. **C-TRPO achieves the best trade-off between safety and return**: It achieves the highest return among algorithms that are safe upon convergence, and the best safety among high-return algorithms.
2. **Cost regret is significantly lower than CPO**: C-TRPO is about 60% of CPO, greatly reducing constraint violations throughout training.
3. **Although Lagrangian methods can eventually become safe, they suffer from severe oscillations during training**: TRPO-Lag has the highest cost regret.
4. **P3O has the lowest cost regret but at the expense of return**: Its final cost is far below the threshold, indicating it is overly conservative.
5. **β=1 is a robust default choice**: No parameter tuning was needed across all 8 environments, and β≤1 barely affects returns.

## Highlights & Insights

1. **Innovative Geometric Perspective**: Instead of adding constraints at the optimization level, it redesigns the metric structure from the information geometry of the policy space. It makes the trust region itself naturally safe, rather than relying on post-hoc checks or projections.
2. **Unification of Theory and Practice**: The continuous-time analysis of C-NPG provides theoretical guarantees for safe-set invariance and global convergence, and C-TRPO is its computable discrete approximation, with second-order consistency bridging the two.
3. **Elegant Connection with CPO**: Through the $\beta$ parameter, it smoothly interpolates between CPO ($\beta=0$) and fully conservative updates ($\beta \to \infty$), providing a continuous safety-performance control knob.
4. **Extremely Simple Implementation**: Compared to TRPO, it only adds a rank-one matrix correction term; compared to CPO, it actually simplifies the inner optimization.

## Limitations & Future Work

1. **Lack of analysis on the finite-sample properties of divergence estimation**: The surrogate divergence relies on the accuracy of cost advantage and value function estimation, which may cause constraint violations when samples are insufficient.
2. **Only handles average cost constraints**: The CMDP framework limits constraints to trajectory average costs, making it unable to directly model state-wise or trajectory-wise safety constraints.
3. **Extension to discrete action spaces**: The experiments focus on continuous control, and the performance on discrete action spaces has not been fully verified.
4. **Adaptive adjustment of hyperparameter $\beta$**: Although $\beta=1$ performs robustly in experiments, theoretically, the optimal $\beta$ should dynamically adjust with training progress and constraint margins.
5. **Integration with model-based methods**: It can be combined with methods like ActSafe to further reduce estimation noise using model information.

## Related Work & Insights

- **CPO (Achiam et al., 2017)**: The direct predecessor of C-TRPO. It treats constraints as additional linear constraints added to the trust region optimization. This paper proves CPO is a special case when $\beta \to 0$.
- **PCPO (Yang et al., 2020)**: Guarantees safety through projection, but the projection step limits return maximization.
- **P3O (Zhang et al., 2022)**: The best constraint satisfaction among penalty methods, but suffers from optimization bias.
- **Control Barrier Functions (Ames et al., 2017)**: The design of the barrier term in C-TRPO is directly inspired by CBF methods in control theory.
- **Information Geometry and Policy Optimization (Neu et al., 2017; Müller & Montúfar, 2023)**: Understanding policy optimization as mirror descent on the occupancy measure space is the foundation of the theoretical framework of this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ - Redesigns the safe trust region from an information-geometric perspective. The concept is elegant, though the core technique is a combination of existing tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ - 8 tasks, 9 baselines, 5 seeds, and rich ablations. The cost regret metric is innovative, but it lacks high-dimensional, complex tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ - Clear explanation of the geometric intuition behind TRPO/CPO/C-TRPO, with theoretical and experimental aspects well-coordinated.
- Value: ⭐⭐⭐⭐ - Provides a simple and practical direction for improvement in safe RL with extremely low implementation overhead; worthy of validation in more scenarios.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TRAPO: Trust-Region Adaptive Policy Optimization](../../ICLR2026/reinforcement_learning/trust-region_adaptive_policy_optimization.md)
- [\[ICML 2025\] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[ICML 2025\] Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics](safety_certificate_against_latent_variables_with_partially_unidentifiable_dynami.md)
- [\[ICLR 2026\] From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones](../../ICLR2026/reinforcement_learning/from_fx_and_gx_to_fgx_llms_learn_new_skills_in_rl_by_composing_old_ones.md)

</div>

<!-- RELATED:END -->
