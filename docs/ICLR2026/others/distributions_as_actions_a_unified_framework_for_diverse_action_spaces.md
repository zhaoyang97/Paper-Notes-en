---
title: >-
  [Paper Note] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces
description: >-
  [ICLR 2026][unified action space] DA-AC proposes treating the parameters of an action distribution (e.g., softmax probabilities or Gaussian mean/variance) as the agent's output "actions," relocating the action sampling process to the environment side. This enables a unified deterministic policy gradient framework for discrete, continuous, and hybrid action spaces. The approach is theoretically proven to achieve strictly lower variance than LR and RP estimators, and attains competitive or state-of-the-art performance across 40+ environments.
tags:
  - ICLR 2026
  - unified action space
  - distribution parameterization
  - deterministic policy gradient
  - discrete-continuous hybrid control
  - variance reduction
date: 2026-05-08
content_hash: c50efc598650c28e
---

# DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces

**Conference**: ICLR 2026  
**arXiv**: [2506.16608](https://arxiv.org/abs/2506.16608)  
**Code**: [GitHub](https://github.com/hejm37/da-ac)  
**Area**: Other  
**Keywords**: unified action space, distribution parameterization, deterministic policy gradient, discrete-continuous hybrid control, variance reduction  

## TL;DR
DA-AC proposes treating the parameters of an action distribution (e.g., softmax probabilities or Gaussian mean/variance) as the agent's output "actions," relocating the action sampling process to the environment side. This enables a unified deterministic policy gradient framework for discrete, continuous, and hybrid action spaces. The approach is theoretically proven to achieve strictly lower variance than LR and RP estimators, and attains competitive or state-of-the-art performance across 40+ environments.

## Background & Motivation
**State of the Field**: Current RL algorithms are tightly coupled to action space types — DQN/DSAC for discrete, DDPG/TD3/SAC for continuous, and specialized algorithms such as PADDPG for hybrid action spaces. The architectural divergence across estimator types makes it difficult to design general-purpose algorithms that operate across domains.

**Limitations of Prior Work**:
   - The Likelihood Ratio (LR) estimator is general but suffers from high variance, requiring careful baseline design.
   - DPG/RP estimators exhibit low variance but are restricted to continuous action spaces.
   - Hybrid action spaces (combining discrete and continuous dimensions) require additional engineering effort.

**Root Cause**: Low-variance gradient estimators such as DPG/RP require continuous action spaces — yet achieving low variance on discrete action spaces remains an open challenge.

**Paper Goals**: Design a unified actor-critic algorithm that operates over arbitrary action space types with theoretical guarantees of low variance.

**Starting Point**: Reconsidering the agent-environment boundary — the agent's "action" need not be the raw action defined by the environment; it can instead be the **distribution parameters**. A policy can typically be decomposed into $\bar{\pi}_\theta$ (mapping states to distribution parameters) and $f$ (sampling from the distribution). By relocating $f$ to the environment side, the agent's action space becomes the continuous parameter space $\mathcal{U}$, regardless of the original action space type.

**Core Idea**: Distributions-as-Actions — distribution parameters serve as actions, and sampling is treated as part of the environment.

## Method

### Overall Architecture
In standard RL, the agent's policy $\pi_\theta$ consists of two components: $\bar{\pi}_\theta$ (mapping states to distribution parameters) and $f$ (sampling actions from the distribution). The DA framework relocates $f$ to the environment side, so the agent directly outputs distribution parameters $u = \bar{\pi}_\theta(s)$. This defines a new MDP — the DA-MDP $\langle \mathcal{S}, \mathcal{U}, \bar{p}, d_0, \bar{r}, \gamma \rangle$ — where transitions and rewards become expectations over the original actions:

$$\bar{p}(s'|s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[p(s'|s,A)], \quad \bar{r}(s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[r(s,A)]$$

Key invariants: $\bar{v}_{\bar{\pi}}(s) = v_\pi(s)$ — state values are preserved; $\bar{q}_{\bar{\pi}}(s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[q_\pi(s,A)]$ — the Q-value of distribution parameters equals the expectation of the original Q-value under the distribution.

### Key Designs

1. **DA-PG Gradient Estimator (Theorem 4.2)**:

    - Function: Deterministic policy gradient in the distribution parameter space.
    - Core formula: $\hat{\nabla}_\theta^{\text{DA-PG}} = \nabla_\theta \bar{\pi}_\theta(S_t)^\top \nabla_U \bar{Q}_w(S_t, U)|_{U=\bar{\pi}_\theta(S_t)}$
    - Relation to DPG: Mathematically identical in form, but $\bar{\pi}$ outputs distribution parameters rather than a single action, and $\bar{Q}$ estimates the expected return under the distribution.
    - DPG as a special case of DA-PG (Prop 4.3): The two are equivalent when $f(\cdot|u)$ degenerates to a Dirac delta.
    - Design Motivation: Since $\mathcal{U}$ is always continuous, DPG-style gradients can be applied to any action space type.

2. **Theoretical Guarantee of Strictly Reduced Variance (Prop 4.4 & 4.5)**:

    - DA-PG is the conditional expectation of the LR estimator (over action $A$); by the law of total variance, its variance is strictly lower.
    - DA-PG is likewise the conditional expectation of the RP estimator (over noise $\epsilon$), also with strictly lower variance.
    - Trade-off: This may introduce additional bias, as the critic must operate over a larger input space.
    - This is the first method to provide an unbiased, RP-style low-variance estimator for discrete action spaces.

3. **ICL (Interpolated Critic Learning)**:

    - Function: Improve critic learning quality in the distribution parameter space.
    - Mechanism: Standard TD updates train the critic only at the current policy's distribution parameters $U_t$, leading to inaccurate critic estimates elsewhere in parameter space. ICL linearly interpolates between the current parameters $U_t$ and the deterministic parameters corresponding to the sampled action $U_{A_t}$: $\hat{U}_t = \omega U_t + (1-\omega) U_{A_t}$, $\omega \sim \text{Uniform}[0,1]$.
    - Design Motivation: Encourages the critic to learn smooth curvature information across the distribution parameter space, enabling the policy gradient to point toward high-value regions. This resembles off-policy learning but operates over distributions rather than policies.
    - Validated via bandit experiments: The ICL-trained critic exhibits richer curvature, whereas the standard critic is accurate only near the current policy.

4. **DA-AC Algorithm**:

    - Built on TD3: dual critics, delayed policy updates, target policy smoothing.
    - Replaces DPG actor updates with DA-PG; replaces standard TD critic updates with ICL.
    - Removes the actor target network (ablation shows it is unnecessary).
    - Adapts to different action spaces: Gaussian (continuous), Softmax (discrete), Gaussian+Softmax (hybrid).

## Key Experimental Results

### Main Results — Continuous Control (MuJoCo + DMC, 20 environments, 1M steps)

| Algorithm | MuJoCo (normalized) | DMC (normalized) |
|-----------|---------------------|------------------|
| TD3 | ~0.82 | ~0.70 |
| SAC | ~0.78 | ~0.65 |
| RP-AC | ~0.80 | ~0.72 |
| PPO | ~0.55 | ~0.48 |
| **DA-AC** | **~0.85** | **~0.78** |

Across 20 individual environments, DA-AC outperforms TD3 in the majority of cases, with particularly pronounced advantages in high-dimensional action spaces (e.g., Humanoid, Dog).

### Discrete Control (Classic Control + MinAtar, 9 environments)
DA-AC is competitive with DQN on both Classic Control and MinAtar, and substantially outperforms LR-AC and ST-AC.

### High-Dimensional Discrete Control ($7^{17}$ action space, Humanoid)
DQN, DSAC, and EAC fail entirely due to the intractable action space, whereas DA-AC handles it seamlessly and achieves performance comparable to the continuous control setting.

### Hybrid Control (7 PAMDP environments)
DA-AC is competitive with or superior to PATD3, an algorithm specifically designed for hybrid action spaces.

### Ablation Study — Contribution of ICL

| Configuration | MuJoCo | DMC | MinAtar | Hybrid |
|---------------|--------|-----|---------|--------|
| DA-AC (w/ ICL) | **0.85** | **0.78** | **0.73** | **0.82** |
| DA-AC w/o ICL | 0.80 | 0.72 | 0.68 | 0.77 |

ICL yields consistent improvements across all settings; paired t-tests confirm statistical significance.

### Key Findings
- **Unified framework**: A single algorithm achieves competitive performance across discrete, continuous, and hybrid action spaces — demonstrated for the first time.
- **Variance advantage of DA-PG**: Bias-variance analysis experiments confirm that DA-PG achieves lower variance than both LR and RP estimators, consistent with theoretical predictions.
- **ICL is critical for critic quality**: Visualizations show that ICL produces better gradient signals for the critic across the distribution parameter space.

## Highlights & Insights
- **Reconceptualizing the agent-environment boundary** is an elegant and principled insight — not a trick, but a deep conceptual shift: reframing the problem unifies previously incompatible methods.
- **DA-PG as the conditional expectation of LR and RP** is a theoretically elegant result; variance reduction follows directly from the law of total variance.
- **The intuition behind ICL**: Standard critics are accurate only near the current policy, yet policy optimization requires Q-value information in other regions of parameter space. ICL enables the critic to "see further" via interpolated sampling — analogous to off-policy learning but operating in distribution space.
- **High-dimensional discrete control** experiments are particularly compelling: a $7^{17}$ action space renders DQN/DSAC/EAC completely infeasible, yet DA-AC handles it without modification.

## Limitations & Future Work
- ICL introduces bias (without importance sampling correction); the theoretical implications for convergence remain to be analyzed.
- The current implementation is built solely on TD3; integration with SAC (maximum entropy) or PPO (on-policy) is a natural next step.
- Critic learning may become more challenging in high-dimensional distribution parameter spaces — for Gaussian policies, critic input dimensionality doubles (mean + variance).
- Extensions to model-based RL, hierarchical control, and other directions remain unexplored.

## Related Work & Insights
- **vs. TD3/DDPG**: DA-AC is a natural generalization of TD3; DPG is a special case of DA-PG.
- **vs. SAC**: SAC uses the RP estimator with entropy regularization; DA-AC uses DA-PG (lower variance) without entropy. The two approaches are complementary and can be combined.
- **vs. EPG (Expected Policy Gradient)**: EPG also targets zero variance but is only tractable in low-dimensional discrete or special continuous settings; DA-AC generalizes to arbitrary high-dimensional spaces.
- **vs. Gumbel-Softmax/ST**: These are biased discrete relaxation methods; DA-PG provides the first unbiased, low-variance estimator for discrete action spaces.
- Implications for agent system design: when agents must simultaneously make discrete decisions and perform continuous control, the DA framework provides a unified interface.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining the agent-environment boundary is an elegant and profound conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 40+ environments covering four action space types, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and visualizations are effective.
- Value: ⭐⭐⭐⭐⭐ A significant step toward a unified RL framework with practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](speculative_actions_faster_ai_agents.md)
- [\[ICLR 2026\] Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)
- [\[ICLR 2026\] Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)
- [\[NeurIPS 2025\] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](../../NeurIPS2025/others/a_unified_framework_for_variable_selection_in_modelbased_clu.md)
- [\[AAAI 2026\] Finding Diverse Solutions Parameterized by Cliquewidth](../../AAAI2026/others/finding_diverse_solutions_parameterized_by_cliquewidth.md)

<!-- RELATED:END -->
