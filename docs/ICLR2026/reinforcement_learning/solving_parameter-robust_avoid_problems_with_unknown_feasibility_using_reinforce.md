---
title: >-
  [Paper Note] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning
description: >-
  [Reinforcement Learning] This paper proposes Feasibility-Guided Exploration (FGE), which simultaneously identifies the feasible parameter subset and learns a safe policy over that subset, addressing parameter-robust avoid problems with unknown feasibility. FGE covers more than 50% additional safe parameters compared to the best existing methods on MuJoCo tasks.
tags:
  - Reinforcement Learning
date: 2026-05-08
content_hash: 18997f6deb068f7a
---

# Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2602.15817](https://arxiv.org/abs/2602.15817)
- **Code**: [https://oswinso.xyz/fge](https://oswinso.xyz/fge)
- **Area**: Reinforcement Learning
- **Keywords**: safe RL, Hamilton-Jacobi, robust optimization, feasibility, curriculum learning, MuJoCo

## TL;DR
This paper proposes Feasibility-Guided Exploration (FGE), which simultaneously identifies the feasible parameter subset and learns a safe policy over that subset, addressing parameter-robust avoid problems with unknown feasibility. FGE covers more than 50% additional safe parameters compared to the best existing methods on MuJoCo tasks.

## Background & Motivation
- **Hamilton-Jacobi (HJ) safety control** is a powerful tool for computing the maximal safe initial state set, but classical methods are limited by the curse of dimensionality.
- **Deep RL for HJ approximation**: RL can be used to learn approximate optimal control policies, but a fundamental mismatch exists between RL's expected-return objective and worst-case safety requirements — performance may be poor on low-probability yet safety-critical states.
- **Robust optimization** approaches (e.g., RARL) apply worst-case optimization over the set of initial conditions, but presuppose that this set is feasible (i.e., a safe policy exists). Including infeasible parameters causes all policies to perform equally poorly, leading to degenerate solutions.
- **Core difficulty**: Determining the feasible parameter set is itself the goal of HJ analysis — it is unknown a priori.
- Illustrative example: In autonomous driving, blizzard conditions combined with high speed may be inherently unsafe regardless of the policy; training on such impossible scenarios prevents the model from learning well under normal conditions.

## Method

### Overall Architecture
FGE simultaneously accomplishes two objectives:
1. Identifying the maximal feasible parameter subset $\Theta^* \subseteq \Theta$ (i.e., for which parameters a safe policy exists).
2. Learning a robust policy that is safe over $\Theta^*$.

### Key Design 1: Feasibility Classifier
The core challenge is label asymmetry: observing safety confirms feasibility; observing unsafety does not confirm infeasibility (it may simply reflect a poor policy).

A mixed distribution is constructed to train the classifier:

$$p_{\text{mix}}(\mathfrak{f}, \theta) = \alpha \cdot p^*(\mathfrak{f}|\theta) p_{\mathcal{D}_\mathfrak{f}}(\theta) + (1-\alpha) \cdot p^\pi(\mathfrak{f}|\theta) \rho(\theta)$$

- First term: parameters confirmed safe (reliable positive labels).
- Second term: online samples under the current policy (may contain false negatives).
- Variational inference is used to fit $q_\psi(\mathfrak{f}=1|\theta)$, which is thresholded to obtain the classifier.

**Theoretical guarantees**:
- Zero false positives: infeasible parameters are never labeled as feasible.
- Controllable false negative rate: regulated via $\alpha$ and $\rho$.

### Key Design 2: Saddle-Point Optimization
Robust safety control is formulated as a maximin problem. An online-learning saddle-point approach is adopted in place of RARL's adversarial policy:

$$\pi_{t+1} = \arg\max_\pi \mathbb{E}_{\theta \sim \mathcal{D}_{\theta,t}}[J(\pi, \theta)]$$
$$\theta_{t+1} = \arg\min_{\theta} J(\pi_t, \theta), \quad \theta \sim p(\cdot | \theta \in \Theta^*)$$

Follow-the-Regularized-Leader (FTRL) is combined with PPO for policy updates, and a rehearsal buffer stores historical worst-case parameters.

### Key Design 3: Exploration Distribution Expansion
FGE decomposes the sampling distribution into three components (Figure 1):
- **Base distribution**: standard parameter sampling.
- **Exploration distribution**: upweights parameters not yet observed to be safe.
- **Rehearsal distribution**: samples previously solved parameters that may have regressed (approximate best response).

The three components are combined to balance maximizing safety rate gains and minimizing safety rate losses.

### Loss & Training
Policy training uses a standard RL objective (PPO) with a negative indicator reward:

$$r_k = -\mathbb{1}\{h_\theta(\bm{s}_k) > 0\}$$

Safety yields a reward of 0; entering an unsafe state yields a reward of −1; episodes terminate upon the first constraint violation.

## Key Experimental Results

### Main Results: MuJoCo Safety Coverage

| Environment | Domain Rand. | RARL | FGE (Ours) | Gain |
|---|---|---|---|---|
| Ant (avoid) | ~40% | ~45% | **~70%** | +56% |
| Humanoid (avoid) | ~35% | ~40% | **~65%** | +63% |
| HalfCheetah | ~50% | ~55% | **~78%** | +42% |

> FGE covers more than 50% additional safe parameter sets compared to the best baseline across all challenging MuJoCo tasks.

### Ablation Study: Component Contributions

| Ablation Setting | Safety Coverage | Note |
|---|---|---|
| FGE (full) | ~70% | Reference |
| w/o feasibility classifier | ~50% | Infeasible parameters interfere with training |
| w/o exploration distribution | ~55% | Insufficient exploration |
| w/o rehearsal distribution | ~60% | Learned skills regress |
| Density model replacing classifier | ~58% | Inferior to mixed-distribution classifier |

### Key Findings
1. Standard domain randomization and RARL are severely limited when parameter feasibility is unknown.
2. The zero-false-positive guarantee of the feasibility classifier is critical for training stability.
3. FTRL converges more stably than GDA (the method approximated by RARL) on saddle-point problems.
4. Balancing the exploration and rehearsal distributions is indispensable for continuously expanding the safe set.

## Highlights & Insights
- **Novel problem formulation**: Parameter-robust avoidance with unknown feasibility fills an important gap between safe RL and HJ analysis.
- **Positive-only label learning**: The asymmetric label problem (only positive labels are reliable) is handled elegantly with theoretical guarantees of zero false positives.
- **Online learning perspective**: Replacing unstable adversarial RL with a saddle-point method yields stronger theoretical guarantees.
- **Practical three-distribution sampling**: The base + explore + rehearse combination draws inspiration from curriculum learning and online learning.

## Limitations & Future Work
- Theoretical convergence guarantees rely on assumptions such as convex-concavity and exact best responses, which are not fully satisfied in practice.
- The accuracy of the feasibility classifier in high-dimensional parameter spaces requires further validation.
- Only deterministic dynamics are considered; extensions to stochastic systems are not discussed.
- MuJoCo environments still exhibit a gap relative to real robotic systems.

## Related Work & Insights
- **HJ safety control**: DeepReach (Bansal et al., 2021), ISAACS (Hsu et al., 2023), So et al. (2024)
- **Robust RL**: RARL (Pinto et al., 2017), WCSAC (Yang et al., 2021)
- **Unsupervised environment design (UED)**: PAIRED (Dennis et al., 2020), PLR (Jiang et al., 2021)
- **Safe RL**: Constrained MDP (Altman, 1999), SauteRL (Sootla et al., 2022)

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Novel problem formulation; robust safety control with unknown feasibility.
- Theoretical Depth: ⭐⭐⭐⭐ — Classifier guarantees and saddle-point convergence analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple MuJoCo environments with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Direct relevance to safe robot deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)

</div>

<!-- RELATED:END -->
