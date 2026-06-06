---
title: >-
  [Paper Note] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][time reversal symmetry] This paper proposes the TR-DRL framework, which exploits time reversal symmetry in robotic manipulation tasks—via trajectory reversal augmentation (for fully…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "time reversal symmetry"
  - "data augmentation"
  - "reward shaping"
  - "robotic manipulation"
  - "sample efficiency"
date: 2026-05-08
content_hash: 8c72148b4ade93a2
---

# Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.13925](https://arxiv.org/abs/2505.13925)  
**Code**: [https://github.com/jyp9961/TR-DRL](https://github.com/jyp9961/TR-DRL)  
**Area**: Reinforcement Learning
**Keywords**: time reversal symmetry, data augmentation, reward shaping, robotic manipulation, sample efficiency

## TL;DR
This paper proposes the TR-DRL framework, which exploits time reversal symmetry in robotic manipulation tasks—via trajectory reversal augmentation (for fully reversible transitions) and time-reversal-guided potential-based reward shaping (for partially reversible transitions)—to significantly improve sample efficiency and final performance of DRL on paired tasks (e.g., door opening/closing).

## Background & Motivation

**Background**: Symmetry exploitation in DRL has focused primarily on spatial symmetries (reflection, rotation, translation), with successful applications to both state-based and image-based settings. Temporal symmetry—particularly time reversal symmetry—remains almost entirely unexplored.

**Limitations of Prior Work**:
   - Many robotic manipulation tasks are inherently time-reversal symmetric (e.g., door open↔close, drawer push↔pull), yet current DRL methods completely ignore this structural information.
   - Naively negating actions ($\vec{a} = -a$) to produce reversed transitions **frequently yields invalid results**. For example, closing a door requires only pushing (no handle grasping), so the reversed "open door" action lacks the handle-grasping step and is physically invalid.
   - Existing time reversal methods (Barkley et al., 2023) assume global full reversibility and known reversal actions, which are overly restrictive assumptions.

**Key Challenge**: Time reversal symmetry is pervasive yet difficult to exploit safely—invalid reversed transitions contaminate training data.

**Key Insight**: Distinguish between **fully reversible** and **partially reversible** cases, and address each with a dedicated technique.

**Core Idea**: Fully reversible → learn an inverse dynamics model with a dynamics consistency filter for data augmentation; partially reversible → use reversible state components (e.g., object angle) to construct a potential-based reward shaping signal.

## Method

### Overall Architecture
Given a pair of tasks with a time reversal relationship (e.g., door open/close), TR-DRL comprises four components: (1) an inverse dynamics model $h$ that predicts the reversed action $\vec{a}$ from $(s', s)$; (2) a forward dynamics model $g$ that validates the physical validity of reversed transitions; (3) trajectory reversal augmentation, which adds fully reversible transitions that pass validation into the replay buffer; and (4) reward shaping, which learns a potential function from successful trajectories of the reversed task to guide policy learning. Both techniques benefit **both tasks** in the pair.

### Key Designs

1. **Full Time Reversal (FTR) Symmetry Exploitation — Trajectory Reversal Augmentation**:

    - Function: Reverses a valid transition $(s,a,s')$ from task A into an augmented transition $(s', \vec{a}, s)$ for task B, which is added to the replay buffer.
    - Mechanism: An inverse dynamics model $a = h(s, s')$ is trained to obtain reversed actions (MSE loss, Eq. 6); a forward dynamics model $g$ performs **dynamics consistency filtering**: $\hat{s} = g(s', h(s', s))$, retaining only transitions where $\|\hat{s} - s\| < \epsilon$.
    - Design Motivation: Not all transitions are reversible—contact, friction, and object release instants are irreversible. The dynamics filter automatically identifies which transitions can be safely reversed, preventing spurious data from being introduced.

2. **Partial Time Reversal (PTR) Symmetry Exploitation — Reward Shaping**:

    - Function: For transitions where the object state is reversible but the robot state is not, the reversible components are used to guide policy learning.
    - Mechanism: The state is decomposed into a reversible component $x$ (e.g., door angle) and an irreversible component $y$ (e.g., end-effector position). A potential function $\Phi(s)$ is learned from successful trajectories of the reversed task and used to construct potential-based reward shaping: $\mathcal{F}(s,a,s') = \gamma \Phi(s') - \Phi(s)$.
    - Design Motivation: Ng et al. (1999) proved that potential-based reward shaping **does not alter the optimal policy**—making it theoretically safe. Even when a full transition is irreversible, guiding the agent toward states where the object state matches successful reversed trajectories remains valuable.

3. **Formal Definition of Partial Time Reversal (PTR) — A New Contribution of This Work**:

    - Function: Extends the FTR definition of Barkley et al. to partially reversible scenarios.
    - Mechanism: The state $s = (x, y)$ is decomposed into a reversible part $x$ and an irreversible part $y$. PTR symmetry holds if there exist some $\vec{y}, \vec{y}'$ such that $T(s'|s,a) = T(\vec{s}|\vec{s}', \vec{a})$ (where $\vec{x} = f_\mathcal{X}(x)$).
    - Design Motivation: In practice, the vast majority of robotic tasks are PTR rather than FTR—when pushing a door, the robot arm position is irreversible, but the door angle is reversible.

### Loss & Training
- Base RL algorithm: SAC (Soft Actor-Critic)
- Inverse dynamics model loss: $L_h = \hat{\mathbb{E}}[(h(s,s') - a)^2]$
- Forward dynamics model loss: $L_g = \hat{\mathbb{E}}[(g(s,a) - s')^2]$
- The potential function is fitted from state sequences of successful trajectories (value function approximation)
- Dynamics models are shared across both tasks in a pair (since the underlying physics is identical)

## Key Experimental Results

### Main Results — Robosuite Benchmark

| Task Pair | SAC Baseline | +Reversal Aug. Only | +Reward Shaping Only | +**TR-DRL Full** | Sample Efficiency Gain |
|-----------|-------------|--------------------|--------------------|-----------------|----------------------|
| Door Open/Close | Slow convergence (~500K) | 2× faster | 1.5× faster | **2.5–3× faster** | Significant |
| Lift/Place | Slow convergence | 1.5× faster | 1.3× faster | **2× faster** | Significant |

### Ablation Study — Component Contributions

| Configuration | Sample Efficiency | Final Performance | Notes |
|---------------|------------------|------------------|-------|
| SAC baseline | Reference | Reference | No symmetry exploitation |
| +Reversal aug. (no filter) | Sometimes harmful | May decrease | Invalid transitions contaminate training |
| +Reversal aug. (with dynamics filter) | Significant gain | Improved | Filter is critical |
| +Reward shaping | Moderate gain | Moderate gain | More important in PTR scenarios |
| +**Both combined** | **Largest gain** | **Highest** | FTR and PTR are complementary |

### Main Results — MetaWorld Multi-Task

| Setting | SAC | TR-DRL | Notes |
|---------|-----|--------|-------|
| Single-task (Door Close) | Partial convergence | Full convergence | FTR augmentation effective |
| Multi-task (4 pairs) | Some tasks fail | **All succeed** | Symmetry information transfers across tasks |

### Key Findings
- **Trajectory reversal without dynamics filtering can be harmful**—this is the central finding. Invalid reversed transitions introduce spurious dynamics information, causing the policy to learn incorrect causal relationships. The filter is the turning point from "potentially harmful" to "consistently beneficial."
- The two techniques are complementary: FTR augmentation is most effective in fully reversible scenarios (grasping + moving), while PTR reward shaping is most effective in partially reversible scenarios (pushing a door/cup). Their combination covers both cases.
- TR-DRL's advantage is more pronounced in the multi-task setting—symmetry information is shared across task pairs, and experience from one task directly benefits the other.
- The quality of the inverse dynamics model directly determines augmentation effectiveness—the model is less accurate early in training when data is scarce, and improves progressively as more data accumulates.

## Highlights & Insights
- **Distinguishing full from partial reversibility** is the key conceptual contribution—in practice, almost no manipulation task is fully time-reversible (contact, friction, and gravity all break full reversibility), and the formalization of PTR greatly expands the applicability of time symmetry exploitation.
- The **dynamics consistency filter** transforms a "potentially harmful" technique into a "consistently beneficial" one—this principle of building in a **safety mechanism** offers a transferable design lesson for all data augmentation methods.
- The combination of potential-based reward shaping with time reversal is natural and elegant—reversed successful trajectories naturally provide "good state sequences" as training signal for the potential function.
- The method is an **orthogonal enhancement** to SAC and can be combined with any off-policy algorithm.

## Limitations & Future Work
- The method requires prior knowledge of task pair relationships (which two tasks are time-reversal symmetric)—automatically discovering symmetric pairs remains an open problem.
- Training the inverse dynamics model may be unstable in high-dimensional or high-degree-of-freedom systems.
- The state decomposition for PTR (which components are reversible/irreversible) currently requires domain knowledge—automatic decomposition warrants further investigation.
- Temporal scale asymmetry is not addressed—when opening is slow but closing is fast, the number of time steps after reversal may not match.

## Related Work & Insights
- **vs. Barkley et al., 2023**: Their method assumes global FTR and known reversal actions ($\vec{a}=-a$), which are overly restrictive; TR-DRL learns reversal actions and introduces the PTR concept, substantially broadening applicability.
- **vs. TRASS (Nair et al., 2020)**: TRASS reverses exploration starting only from the goal state; TR-DRL exploits all transitions along entire trajectories.
- **vs. Eysenbach et al., 2018**: Their approach learns a reset policy; TR-DRL treats paired tasks independently while sharing symmetry information—complementary rather than substitutive.
- The PTR concept can be extended to additional task pairs: assembly↔disassembly, throwing↔catching, painting↔erasing, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ The formal definition of PTR is a new contribution; the full/partial reversibility distinction has conceptual depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Robosuite + MetaWorld, two standard benchmarks; single- and multi-task settings; detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ FTR/PTR concepts and examples are clearly explained; method diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Directly applicable to robotic tasks with paired symmetric structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)

</div>

<!-- RELATED:END -->
