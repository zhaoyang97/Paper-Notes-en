---
title: >-
  [Paper Note] Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations
description: >-
  [ICLR 2026][Reinforcement Learning][Hamilton-Jacobi-Bellman] This paper extends HJ-RL (Hamilton-Jacobi Reinforcement Learning) from single "Reach / Avoid / Reach-Avoid" objectives to two types of dual-objective problems—Reach-Always-Avoid (RAA) and Reach-Reach (RR). It demonstrates that their Bellman equations can be **exactly decomposed** into combinations of previously studied
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Hamilton-Jacobi-Bellman
date: 2026-05-08
content_hash: 8a1dcb4339d8e7f0
---
# Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=1SdPgRQrr5](https://openreview.net/forum?id=1SdPgRQrr5)  
**Code**: TBD  
**Area**: Reinforcement Learning  
**Keywords**: Hamilton-Jacobi-Bellman, Safe Reinforcement Learning, Reach-Avoid, Dual-Objective, Value Function Decomposition

## TL;DR
This paper extends HJ-RL (Hamilton-Jacobi Reinforcement Learning) from single "Reach / Avoid / Reach-Avoid" objectives to two types of dual-objective problems—Reach-Always-Avoid (RAA) and Reach-Reach (RR). It demonstrates that their Bellman equations can be **exactly decomposed** into combinations of previously studied simple reach/avoid sub-problems. Based on this, the DOHJ-PPO algorithm is designed, surpassing 10 Lagrangian-based and HJ-RL baselines in success rate, safety, and speed.

## Background & Motivation
**Background**: In Safe RL, the core requirement is to complete tasks (reaching targets) while satisfying constraints (avoiding hazards). The mainstream approach involves folding constraints into rewards—Lagrangian methods (CPPO, PPO-LAG, etc.) introduce a multiplier to optimize a hybrid objective where the discounted cumulative sum of safety violations is below a threshold. Another line of research is HJ-RL: originating from the Hamilton-Jacobi perspective of dynamic programming, it uses **non-conventional Bellman updates** to solve safety and reachability tasks, exemplified by the Safety Bellman Equation (SBE) and Reach-Avoid Bellman Equation (RABE).

**Limitations of Prior Work**: Lagrangian methods require meticulous reward engineering and multiplier tuning. Furthermore, because they optimize discounted cumulative sums, constraint satisfaction is only achieved "in expectation" and lacks direct interpretability. While HJ-RL is theoretically elegant—its value function propagates the **extremum** (worst penalty, best reward) along trajectories rather than discounted sums, meaning a positive value is equivalent to "the task will definitely be completed"—it has thus far only handled three atomic operations: Reach, Avoid, and Reach-Avoid, failing to express more complex composite tasks.

**Key Challenge**: Many practical tasks are inherently **dual-objective coupled**. For example, a fighter jet must fly into a designated airspace and then continuously avoid crashing (reaching a target followed by perpetual safety), or a robot must visit two target points in any order. Balancing these competing objectives using discounted rewards is extremely difficult to tune, and existing HJ-RL lacks corresponding Bellman formulations. While Linear Temporal Logic (LTL) methods can describe such composite specifications, they typically require constructing an automaton and performing state-product expansion; solving path and HJ extremum optimization are two separate systems, making it difficult to leverage the safety advantages of HJ-RL directly.

**Goal**: To derive **explicit, solvable** Bellman formulations for two complementary dual-objective problems—RAA (reach a target and avoid danger forever) and RR (reach two targets in any order)—that can directly utilize existing HJ-RL solvers.

**Key Insight**: The authors observe that while RAA and RR have different themes, they are mathematically **complementary** (differing only by a max/min operation). Their value functions possess a "decomposable" basic structure where composite objectives can be split into several already solvable atomic objectives.

**Core Idea**: Use "value function decomposition" instead of "automata construction." It is proved that the RAA value equals a combination of an Avoid sub-problem and a Reach-Avoid sub-problem defined with a "modified reward," and the RR value equals a combination of three Reach sub-problems. This reduces composite HJ-RL problems to solved simple problems.

## Method

### Overall Architecture
The paper addresses the objective of maximizing the "worst-case outcome of dual objectives" in a deterministic MDP $M = \langle S, A, f \rangle$. The objective functions for the two problems are written as combinations of trajectory extrema:

$$V^\pi_{RAA}(s) = \min\left\{ \max_t r(s^\pi_t),\ \min_t q(s^\pi_t) \right\}, \qquad V^\pi_{RR}(s) = \min\left\{ \max_t r_1(s^\pi_t),\ \max_t r_2(s^\pi_t) \right\}$$

where $r$ is the reward to be maximized and $p$ is the penalty to be minimized; for convenience, $q = -p$ is used (thus, "minimizing the worst penalty" becomes "maximizing the minimum $q$"). RAA takes the minimum of the "best reward" and the "worst penalty," forcing the agent to enter the target at some point and avoid danger throughout. RR takes the minimum of two "best rewards," forcing the agent to visit both targets. These values are **fundamental different** from the standard infinite discounted sums in RL—they do not accumulate along the trajectory but are determined by specific extremum points on the trajectory, carrying direct semantics: if $r$ is positive only in goal $G$ and $q$ is negative only in danger zone $H$, then $V^\pi_{RAA}(s) > 0$ if and only if policy $\pi$ actually completes the RAA task.

The method proceeds in three steps: (1) highlighting that a naive policy $\pi: S \to A$ is **inherently insufficient** for such multi-objective infinite-horizon problems, necessitating state augmentation; (2) proving that augmented RAA/RR values can be **precisely decomposed** into combinations of simple HJ sub-problems (Theorems 1, 2) and proving the optimality of this augmentation (Theorem 3); (3) relaxing the deterministic decomposition to a stochastic version (SRBE/SRABE) and designing DOHJ-PPO to learn the composite and decomposed representations simultaneously.

### Key Designs

**1. State Augmentation: Bridging the Markovian Gap with Historical Extremas**

A naive policy $\pi: S \to A$ that depends only on the current state is flawed here. The problem lies not in the state transitions (which remain Markovian) but in the **non-Markovian nature** of the reward. When balancing multiple objectives over an infinite horizon, the optimal action often depends on the trajectory history rather than just the current state. Figure 3 provides an intuitive example: in RR, a memoryless agent at an intermediate state can only bind it to one fixed action, thus reaching only one target; in RAA, whether a robot should risk passing through a conical obstacle to reach a goal depends on whether it has **already reached the goal** previously.

The solution is to augment the MDP to $\bar M$, where the state becomes $\mathcal S = S \times Y \times Z$. Two auxiliary variables are added to track the "best reward $y$ and worst penalty $z$ achieved so far." For RAA:

$$y^{\bar\pi}_{t+1} = \max\left\{ r(s^{\bar\pi}_{t+1}),\ y^{\bar\pi}_t \right\}, \qquad z^{\bar\pi}_{t+1} = \min\left\{ q(s^{\bar\pi}_{t+1}),\ z^{\bar\pi}_t \right\}$$

For RR, the update for $z$ is also changed to max (since both rewards $r_1, r_2$ are to be maximized). The augmented policy $\bar\pi: \mathcal S \to A$ can then utilize historical information. Crucially, the purpose of augmentation is **not** to make the rewards Markovian (they are already deterministic functions of the state), but because the HJ-style extremum optimization used here requires explicitly feeding the "already achieved extrema" to the policy.

**2. RAA Decomposition Theorem: Avoid + Reach-Avoid with Modified Reward (Theorem 1)**

This is a core theoretical contribution. Directly solving the RAA value is difficult, but Theorem 1 proves it can be obtained by first solving an Avoid problem for penalty $q$ to get $V^*_A(s) = \max_\pi \min_t q(s^\pi_t)$, and then solving a standard Reach-Avoid problem with a **modified reward**:

$$V^*_{RAA}(s) = \max_\pi \max_t \min\left\{ \tilde r_{RAA}(s^\pi_t),\ \min_{\tau \le t} q(s^\pi_\tau) \right\}, \qquad \tilde r_{RAA}(s) := \min\left\{ r(s),\ V^*_A(s) \right\}$$

The intuition is that the modified reward $\tilde r_{RAA}$ takes the minimum of the "original reward $r$" and the "safety guarantee $V^*_A$ from the current state"—meaning a state is truly good only if it is in the target and located in a safe zone where hazard avoidance can still be guaranteed afterward. RAA thus reduces to a Reach-Avoid problem for $\tilde r_{RAA}$, which can be solved using Hsu et al.'s RABE solver. The corresponding Bellman equation (Corollary 1) is:

$$V^*_{RAA}(s) = \min\left\{ \max\left\{ \max_{a \in A} V^*_{RAA}(f(s,a)),\ \tilde r_{RAA}(s) \right\},\ q(s) \right\}$$

This is structurally isomorphic to RABE, with the reward term replaced by $\tilde r_{RAA}$. Unlike Boolean algebra decomposition of predicates in temporal logic, this decomposition is tailored for HJ-style optimal control (Appendix L argues why TL predicate algebra is insufficient).

**3. RR Decomposition Theorem: Three Reach Problems (Theorem 2)**

The RR decomposition is dual to RAA. First, solve two Reach problems for each reward to get $V^*_{R1}(s) = \max_\pi \max_t r_1(s^\pi_t)$ and $V^*_{R2}(s)$. Then, solve a third Reach problem with a modified reward:

$$V^*_{RR}(s) = \max_\pi \max_t \tilde r_{RR}(s^\pi_t), \qquad \tilde r_{RR}(s) := \max\left\{ \min\left\{ r_1(s), V^*_{R2}(s) \right\},\ \min\left\{ r_2(s), V^*_{R1}(s) \right\} \right\}$$

The meaning of $\tilde r_{RR}$ is: a state is worth staying in if and only if it is either "already in Target 1 and can later reach Target 2" or "already in Target 2 and can later reach Target 1"—taking the better of the two paths. RR thus becomes a pure Reach problem for $\tilde r_{RR}$, and the Bellman equation (Corollary 2) reduces to the standard Reach form $V^*_{RR}(s) = \max\{ \max_a V^*_{RR}(f(s,a)),\ \tilde r_{RR}(s) \}$.

**4. Optimality Guarantees (Theorem 3): History Beyond Two Variables is Redundant**

Proving decomposition is not enough; one must confirm that this specific $(y, z)$ augmentation is "sufficient." Theorem 3 provides a sandwich-style conclusion:

$$\max_\pi V^\pi_{RAA}(s) \le \max_{\bar\pi} V^{\bar\pi}_{RAA}(s) = \max_{a_0, a_1, \dots} \min\left\{ \max_t r(s_t),\ \min_t q(s_t) \right\}$$

The right side is the theoretical optimum when the action sequence is known in advance; the left side is the upper bound achievable by a memoryless policy. The theorem states that the optimal augmented policy exactly reaches the theoretical optimum (usually $\ge$ memoryless policy) and that **adding more historical information into the state cannot improve** the performance of the optimal policy. This provides a rigorous basis for tracking only the two extremum variables $(y, z)$—it is both necessary and sufficient.

### Loss & Training
Theoretical decomposition assumes deterministic dynamics and the ability to pre-solve decomposition sub-values, both of which are limited in practical RL. DOHJ-PPO bridges theory to practice in two steps.

**Stochastic Relaxation (SRBE / SRABE)**: High-performance RL requires stochastic policies. Following the Stochastic Reachability Bellman Equation (SRBE) by So et al., this paper generalizes it to the Stochastic Reach-Avoid Bellman Equation (SRABE):

$$\hat V^\pi_{RAA}(s) = \mathbb E_{a \sim \pi}\left[ \min\left\{ \max\left\{ \hat V^\pi_{RAA}(f(s,a)),\ \tilde r_{RAA}(s) \right\},\ q(s) \right\} \right]$$

In practice, a version with a discount factor $\gamma$ is used (to ensure contraction), recovering the non-discounted value as $\gamma \to 1^-$. The corresponding action value $\hat Q^{\gamma,\pi}_{RAA}(s,a)$ is obtained by removing the outer expectation. The PPO advantage function is $\hat A^\pi_{RAA} = \hat Q_{RAA} - \hat V_{RAA}$. This relaxation essentially swaps the order of extremum and expectation operators—while not strictly interchangeable, the paper proves this yields a **conservative estimate** (Appendix D) and validates its effectiveness in stochastic dynamics experiments.

**Three Minimal Modifications to PPO + Synergistic Bootstrapping**: DOHJ-PPO makes three changes to PPO. First, it introduces additional actors and critics for decomposed sub-objectives (as per Theorems 1 and 2, the composite value is a combination of simpler R/A/RA values; thus, each decomposed term is learned by its own network). Second, the composite and decomposed actors/critics are **trained simultaneously** rather than sequentially. In each iteration, each actor performs rollouts independently. When the composite representation is updated, decomposition values are bootstrapped from the current decomposition critic along the composite trajectory and integrated via GAE and targets with special RAA/RR modified rewards. Third, **coupled resets**: trajectories for training decomposition actors/critics are sampled from states in the composite trajectories. This prevents bias from "solving decomposition objectives independently"—for instance, in RAA, if an avoid decomposition converges alone, it might only focus on safety and hide in a corner unrelated to the reward; using states from the composite trajectory anchors decomposition learning to the PPO on-policy data distribution.

## Key Experimental Results

### Main Results
The experiments consist of two layers: qualitative demonstrations using DQN in a 2D Gridworld (Figure 2) to verify that the decomposition theorem induces different behaviors, and quantitative comparisons using DOHJ-PPO in four continuous control environments—Hopper, F16, SafetyGym, and HalfCheetah (Figure 4). Each algorithm is evaluated over 1000 trajectories on three metrics: **Success** (simultaneous completion of both sub-objectives), **Partial Success** (completion of at least one), and average steps required.

| Setting | Comparison Dimension | DOHJ-PPO Performance | Note |
|------|---------|--------------|------|
| Gridworld DQN (RAA) | vs. Classic RA | RA stops in regions doomed to collide; RAA learns to stop in safe zones | Qualitative validation of Theorem 1 |
| Gridworld DQN (RR) | vs. Single Reach | Single Reach stops at one goal; RR induces trajectories visiting both | Qualitative validation of Theorem 2 |
| 4 Continuous Control Envs | vs. 10 SOTA Baselines | Rank 1st or 2nd in all tasks/envs; lead increases with env complexity | Comprehensive lead in Success metric |

The 10 baselines cover three categories: Lagrangian (CPPO, PPO-LAG, P2BPO, LOGBAR), HJ-RL (RESPO, RCPPO, RA), and STL/LTL-RL and MORL (D-STL PPO, SPARSE STL PPO, MORL-PPO). A key observation is: **almost all algorithms achieve high Partial Success, but very few achieve high Success**—this indicates that completing a single goal is easy, but balancing two competing objectives is difficult, especially under discounted reward frameworks. DOHJ-PPO is the only algorithm strong in both RAA and RR tasks with the fastest completion speed.

### Ablation Study
The core ablation focuses on **robustness to stochastic dynamics** (Figure 5): affine Gaussian noise is injected into HalfCheetah velocity/angular velocity at four levels: null, low (0.5), medium (1.0), and high (2.0). Success rates ($V_{RAA} > 0$ or $V_{RR} > 0$) across 256 trajectories are compared against the top three baselines for each task, using the maximum learning curve from three seeds.

| Task | Noise Level | DOHJ-PPO vs. 2nd Place | Note |
|------|--------|---------------------|------|
| RAA | Medium | Peak performance lead of 8%–22% | Fastest to reach peak; all degrade at extreme noise |
| RR | Most levels | Peak performance lead of >30% | Lead of >15% over best baseline DSTL even at medium noise |
| RR | Medium | Slower than DSTL to peak, but peak is ~2x higher | Trade-off between speed and performance |

### Key Findings
- **Extremum propagation is the key to victory**: The advantage of DOHJ-PPO stems from the fact that the new Bellman update propagates trajectory **extrema** (max/min) rather than short-term averages (discounted sums). Discounted sums squashed competing objectives into a scalar, often sacrificing one for the other; the extremum form naturally preserves "worst-case sub-objective" semantics, enabling stable simultaneous satisfaction of dual objectives.
- The **gap between Partial vs. Full Success** highlights the true difficulty of dual-objective RL: achieving a single goal (Partial) is easy for almost all methods; what distinguishes the best is the ability to secure both (Full).
- **Stochastic relaxation is a conservative yet effective approximation**: Swapping expectation and extremum operators in SRBE/SRABE is not strictly rigorous (the two are not interchangeable), but experiments prove it provides reliable (conservative) estimates under stochastic policies and moderate stochastic dynamics.

## Highlights & Insights
- **"Decomposition" as an elegant paradigm shift from "Automata"**: While LTL-RL handles composite tasks via automaton-based state-product expansion, this paper proves composite HJ values can be **precisely** split into solved atomic HJ sub-problems. This transforms composite Safe RL from "requiring new solvers" to "reusing old solvers + combination," offering high reusability.
- **Clever construction of modified reward $\tilde r$**: In RAA, $\tilde r_{RAA} = \min\{r, V^*_A\}$ couples the "goal status" with "future safety guarantees" into a scalar reward, reducing the composite problem to a standard RA problem. This technique of "reshaping rewards using sub-problem optimal values" is transferable to other tasks requiring look-ahead constraints.
- **Synergistic Bootstrapping + Coupled Resets** solves a common malaise in decomposition methods: learning sub-objectives separately can cause sub-policies to drift to regions irrelevant to the total task. Using states from the composite trajectory to initialize decomposition rollouts nails the decomposition learning to the correct state distribution.
- **Direct interpretability of value functions**: A positive value $\iff$ the task is achievable. this transforms "constraint satisfaction" in Safe RL from probabilistic expectation to deterministic semantics, which is highly valuable for mission-critical applications.

## Limitations & Future Work
- **Lack of theoretical guarantees for stochastic dynamics**: The authors explicitly acknowledge that because expectation and extremum operators are not interchangeable, SRBE/SRABE is only an approximation in stochastic **dynamics** (as opposed to stochastic policies). Currently, robustness is supported only by empirical evidence from noise-injected ablation experiments.
- **Support for only two objectives**: The current framework targets RAA/RR dual-objective problems. The authors envision that hierarchical objectives (corresponding to more complex temporal logic specifications) could be iteratively decomposed into a graph of Bellman values. However, this requires deriving generalized decomposition principles for non-trivial combinations of logic operations and more efficient representation and sampling heuristics.
- **Deterministic dynamics assumption**: Theoretical decomposition (Theorems 1–3) is built on deterministic transitions $s_{t+1} = f(s_t, a_t)$. Rigor may be compromised when extending this to general stochastic MDPs.
- **Note on lateral comparisons**: Difficulty across different environments and noise levels is not directly comparable; numbers like "8%–22% lead" are relative to the second place within specific settings.

## Related Work & Insights
- **vs. Lagrangian / CMDP Methods (CPPO, PPO-LAG, etc.)**: These fold constraints into expected discounted sums with multipliers, requiring meticulous reward engineering and tuning; constraint satisfaction is only in expectation. **Ours** treats reachability and avoidance as **hard constraints** without multipliers or similar hyperparameters, and the learned value function has direct interpretability regarding constraint satisfaction.
- **vs. Temporal Logic RL (LTL/STL, e.g., D-STL)**: These express composite specifications through automata and state-product expansion; the solution system differs from HJ extremum optimization. **Ours** is designed to reuse HJ sub-problem solvers and proves that optimal policies can be reached (theoretically exactly, practically approximately) even without state expansion.
- **vs. Existing HJ-RL (RABE/Hsu et al. 2021, RCPPO/So et al. 2024, RESPO/Ganai et al. 2023)**: These cover only R, A, and RA atomic operations and do not handle composite tasks. **Ours** generalizes HJ-RL to RAA and RR, analogous to the progression from MDP to NMRDP in LTL-RL literature, and directly utilizes these existing solvers as sub-modules.
- **vs. Multi-Objective / Reward Decomposition RL (MORL, HRA)**: Those works pursue Pareto-optimal discounted sums or decompose discounted sums. **Ours** targets safe optimal control under **extremum-type** (non-additive) objectives, a scenario where discounted sum decomposition methods typically struggle.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to generalize HJ-RL to RAA/RR dual-objective problems with precise value decomposition theorems and optimality guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four environments, 10 baselines, plus stochastic dynamics ablations; however, theoretical validation outside deterministic dynamics remains empirical.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and step-by-step motivation; theorems are dense and heavily rely on appendices.
- Value: ⭐⭐⭐⭐⭐ Provides a "decomposition-combination" paradigm and interpretable value functions for composite Safe RL, with practical potential for mission-critical control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)
- [\[ICLR 2026\] A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning](a_reward-free_viewpoint_on_multi-objective_reinforcement_learning.md)
- [\[ICLR 2026\] BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management](borearl_a_multi-objective_reinforcement_learning_environment_for_climate-adaptiv.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[ICLR 2026\] Dual Goal Representations](dual_goal_representations.md)

</div>

<!-- RELATED:END -->
