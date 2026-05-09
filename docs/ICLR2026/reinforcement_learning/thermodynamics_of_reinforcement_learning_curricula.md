---
title: >-
  [Paper Note] Thermodynamics of Reinforcement Learning Curricula
description: >-
  [ICLR 2026][Reinforcement Learning][Curriculum Learning] This paper formalizes curriculum learning in RL as a geodesic optimization problem over task space using a framework of excess work minimization drawn from non-equilibrium thermodynamics, and derives the MEW temperature annealing algorithm based on a friction tensor, outperforming standard SAC temperature scheduling on the MuJoCo Humanoid task.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Curriculum Learning
  - Non-equilibrium Thermodynamics
  - Maximum Entropy RL
  - Temperature Annealing
  - Riemannian Geometry
  - Geodesics
date: 2026-05-08
content_hash: 0d37dc2fa6d7a1fd
---

# Thermodynamics of Reinforcement Learning Curricula

**Conference**: ICLR 2026
**arXiv**: [2603.12324](https://arxiv.org/abs/2603.12324)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Curriculum Learning, Non-equilibrium Thermodynamics, Maximum Entropy RL, Temperature Annealing, Riemannian Geometry, Geodesics

## TL;DR
This paper formalizes curriculum learning in RL as a geodesic optimization problem over task space using a framework of excess work minimization drawn from non-equilibrium thermodynamics, and derives the MEW temperature annealing algorithm based on a friction tensor, outperforming standard SAC temperature scheduling on the MuJoCo Humanoid task.

## Background & Motivation

**Background**: Modern RL systems are rarely trained on a single, static task; instead, agents are exposed to a sequence of related tasks through curriculum learning, temperature annealing, reward shaping, and similar techniques. However, principled guidance on how to vary task parameters remains lacking.

**Limitations of Prior Work**: The common practice of linearly interpolating task parameters implicitly assumes that task space is flat and isotropic. In reality, the learning difficulty induced by parameter changes varies greatly across directions—some directions impose high adaptation costs, others low.

**Key Challenge**: The absence of a principled framework for quantifying the "adaptation cost" of task parameter changes forces curriculum design to rely on heuristics (e.g., fixed decay schedules, manual tuning), which may vary parameters too rapidly in high-friction regions and destabilize the policy.

**Goal**: (1) How can a measure of "learning difficulty" over task space be defined and computed? (2) What constitutes an optimal curriculum path? (3) Can a practical temperature annealing algorithm be derived?

**Key Insight**: Drawing from statistical mechanics, the paper analogizes the policy's response to task parameter changes to the driving of a non-equilibrium physical system, and applies linear response theory to approximate excess work as a quadratic form, thereby constructing a pseudo-Riemannian metric over the task parameter space.

**Core Idea**: The optimal curriculum corresponds to a geodesic under the geometry induced by the friction tensor in task space—slowing down in directions where learning is difficult and accelerating where it is easy.

## Method

### Overall Architecture
The parameters $\lambda \in \mathbb{R}^L$ of a parameterized reward function $r_\lambda(s,a)$ are treated as coordinates on a task manifold. A curriculum $\lambda(t)$ is a path connecting two tasks. By minimizing the cumulative adaptation cost (excess work) incurred by the agent, curriculum design is cast as a geometric optimization problem.

### Key Designs

1. **Excess Work and the Friction Tensor**:

    - Function: Quantifies the additional adaptation cost arising from finite-rate changes to task parameters.
    - Mechanism: In the quasi-static limit, linear response theory is used to approximate excess work as $\mathcal{W}_{\text{excess}} = \int_0^\infty \dot{\lambda}_i(t) \zeta_{ij}(\lambda(t)) \dot{\lambda}_j(t) \, dt$
    - The friction tensor is given by the Green–Kubo relation: $\zeta_{ij}(\lambda) = \beta \sum_{t=0}^{\infty} \mathbb{E}_{\tau \sim p_\lambda}(\delta X_i(\mathbf{s}_t, \mathbf{a}_t) \cdot \delta X_j(\mathbf{s}_0, \mathbf{a}_0))$
    - Design Motivation: Large values of the friction tensor correspond to directions in which reward gradient fluctuations persist over long time scales, resulting in high adaptation costs.

2. **Geodesic Optimal Curriculum**:

    - Function: Solves for the path that minimizes excess work.
    - Mechanism: The quadratic form of excess work endows parameter space with a pseudo-Riemannian metric; the optimal curriculum satisfies the geodesic equation $\ddot{\lambda}^k + \Gamma^k_{ij}(\lambda) \dot{\lambda}^i \dot{\lambda}^j = 0$
    - Key Corollary: **Linear curricula are optimal only when the induced geometry is flat**, i.e., $\zeta_{ij}(\lambda) = c$.
    - Design Motivation: GridWorld experiments intuitively illustrate how geodesics circumvent regions of maximum friction (the phase transition at $\lambda_1 = \lambda_2$).

3. **MEW Temperature Annealing Algorithm**:

    - Function: Applies the framework to temperature annealing in MaxEnt RL (e.g., $\alpha$ scheduling in SAC).
    - Mechanism: The inverse temperature $\beta = \alpha^{-1}$ is treated as the control parameter, reducing friction to the auto-covariance of rewards. Minimizing excess work yields the update rule $\dot{\alpha} \propto \alpha^2 / \sqrt{\sum \langle \delta r_k \delta r_{t+k} \rangle}$
    - Design Motivation: Temperature is varied slowly in regions of high reward variance and more rapidly where fluctuations are small, providing a principled mechanism for adaptive regularization.

### Loss & Training
MEW requires no additional loss function; it instead provides a temperature scheduling strategy. Using ASAC (Average-reward SAC) as the base algorithm, only the temperature annealing scheme is replaced. Friction (reward auto-covariance) can be computed efficiently from reward samples collected during training.

## Key Experimental Results

### Main Results (MuJoCo Humanoid-v5)

| Method | Performance | Temperature Schedule Characteristics |
|--------|-------------|--------------------------------------|
| Fixed High Temperature | Converges but suboptimal | Constant throughout |
| Fixed Low Temperature | Unstable early | Constant throughout |
| SAC Automatic Tuning | Suboptimal, non-monotone temperature | Initial rapid drop followed by recovery |
| **MEW** | **Best** | **Monotone decay, high consistency across runs** |

### Key Findings
- The standard SAC method (Haarnoja et al., 2018b) causes premature policy determinism due to its rapid initial temperature reduction, requiring subsequent compensatory increases.
- MEW produces monotone temperature trajectories and dynamically adjusts step size according to the relative adaptation cost.
- MEW exhibits significantly higher consistency across runs compared to standard methods (narrower confidence intervals).
- GridWorld experiments clearly demonstrate the suboptimality of linear paths that traverse the region of maximum friction.

### Ablation Study
- Regret is compared between linear and geodesic paths in GridWorld; geodesic paths that circumvent the phase transition yield substantially lower regret.
- Visualization of the friction tensor confirms that the task space geometry is genuinely curved (non-Euclidean).

## Highlights & Insights
- **Deep Connection between Statistical Mechanics and RL**: This is not a superficial analogy—the Boltzmann distribution structure of MaxEnt RL enables an exact and computable mapping. The Green–Kubo relation for the friction tensor has a well-defined form in the RL setting.
- **Geometrization of Learning Difficulty**: The abstract notion of "where learning is hard" is translated into a measurable geometric quantity (friction), removing the need for ad hoc curriculum design.
- **Practical Utility**: MEW requires only reward variance estimates, incurs low computational overhead, and can be directly integrated into existing deep RL algorithms.
- **Explanatory Power**: The framework offers a principled explanation for certain empirically observed RL instabilities—they may arise from driving a high-dimensional non-equilibrium system too aggressively along a curved manifold.

## Limitations & Future Work
- The current theory relies on the quasi-static assumption (linear response theory), which may break down when parameters change rapidly.
- Empirical validation of MEW is limited to one-dimensional temperature annealing; geodesic computation in higher-dimensional task spaces requires scalable friction tensor estimators.
- Friction tensor computation assumes the policy has approximately converged, and may be inaccurate during the non-stationary early phase of training.
- Integration with distributional RL methods (which provide variance estimates directly) represents a promising future direction.

## Related Work & Insights
- **vs. Heuristic Curriculum Learning**: Existing curriculum learning methods lack theoretical optimality guarantees. This framework provides an optimality criterion grounded in physical principles.
- **vs. Automatic Temperature Tuning (SAC)**: SAC's minimum entropy constraint approach is reactive, whereas MEW is proactive—it anticipates future adaptation costs based on friction.
- **vs. Reward Shaping**: Potential-based reward shaping (PBRS) corresponds within this framework to degenerate directions of the metric (zero eigenvalues), theoretically unifying this phenomenon.
- **vs. Optimal Transport Curriculum Learning (Huang et al., 2022)**: OT-based methods require source and target task distributions, whereas MEW relies only on online reward statistics.
- **vs. Linear Interpolation / Fixed Decay**: Linear curricula are optimal only when the geometry is flat—the framework precisely characterizes when and why linearity is insufficient.
- **vs. Fisher Information Metric**: The friction tensor resembles the Fisher information matrix in form but differs in meaning—the Fisher metric measures information geometry in parameter space, while the friction tensor measures adaptation cost in task space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The deep connection between thermodynamics and RL is remarkably elegant; the concept of geodesic curricula is groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐ GridWorld validates the geometric concepts, but deep RL experiments are limited to the single Humanoid task.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, physical intuitions are clear, and the paper is well-structured.
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for curriculum learning and temperature scheduling; the MEW algorithm is immediately applicable.
- Scalability: ⭐⭐⭐ One-dimensional temperature annealing is validated; multi-dimensional task spaces require scalable friction estimators.
- Theoretical Depth: ⭐⭐⭐⭐⭐ The mapping from non-equilibrium thermodynamics to RL curricula is mathematically rigorous and physically well-motivated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)

</div>

<!-- RELATED:END -->
