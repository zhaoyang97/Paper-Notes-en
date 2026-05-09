---
title: >-
  [Paper Note] TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design
description: >-
  [ICLR 2026][Reinforcement Learning][Unsupervised Environment Design] TRACED improves regret approximation in Unsupervised Environment Design (UED) by augmenting the conventional PVL with an Approximate Transition Prediction Loss (ATPL) to capture dynamics model mismatch, and introduces a Co-Learnability measure to quantify inter-task transfer benefits. On MiniGrid and BipedalWalker, TRACED surpasses all baselines' 20k-update performance using only 10k updates.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Unsupervised Environment Design
  - Curriculum Learning
  - Regret Approximation
  - Transition Prediction Error
  - Co-Learnability
  - Zero-shot Transfer
date: 2026-05-08
content_hash: 1862dea58390b89d
---

# TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design

**Conference**: ICLR 2026
**arXiv**: [2506.19997](https://arxiv.org/abs/2506.19997)
**Code**: [https://github.com/Cho-Geonwoo/TRACED](https://github.com/Cho-Geonwoo/TRACED)
**Area**: Reinforcement Learning
**Keywords**: Unsupervised Environment Design, Curriculum Learning, Regret Approximation, Transition Prediction Error, Co-Learnability, Zero-shot Transfer

## TL;DR
TRACED improves regret approximation in Unsupervised Environment Design (UED) by augmenting the conventional PVL with an Approximate Transition Prediction Loss (ATPL) to capture dynamics model mismatch, and introduces a Co-Learnability measure to quantify inter-task transfer benefits. On MiniGrid and BipedalWalker, TRACED surpasses all baselines' 20k-update performance using only 10k updates.

## Background & Motivation

**Background**: UED is a co-evolutionary framework in which a teacher adaptively generates tasks with high learning potential while a student learns a robust policy. Existing methods (PLR⟂, ACCEL, etc.) measure learning potential via regret, but since the true optimal policy $\pi^*$ is unknown, only coarse proxies (PVL, MaxMC) are available.

**Limitations of Prior Work**: (1) PVL approximates regret solely through value function error, ignoring the contribution of dynamics model mismatch to future returns. (2) Existing methods treat each task independently, without considering how training on one task affects performance on others.

**Key Challenge**: Regret = optimal return − current return, yet the optimal return is intractable. As a proxy, PVL primarily reflects value estimation error and fails to capture the contribution of transition prediction error to regret.

**Goal**: (1) Provide a more accurate regret approximation; (2) model inter-task transfer relationships for curriculum optimization.

**Key Insight**: Starting from a decomposition of regret, the paper identifies a dynamics component within the future value gap that PVL does not cover, and supplements it with a Transition Prediction Loss (ATPL). Co-Learnability is further introduced to exploit correlations in regret changes across tasks.

**Core Idea**: Augment PVL with ATPL to capture dynamics uncertainty, and incorporate Co-Learnability to quantify inter-task transfer benefits, forming a unified Task Priority score to guide curriculum design.

## Method

### Overall Architecture
TRACED follows the UED loop of ACCEL, with the sole modification of replacing PVL with Task Priority (combining Task Difficulty and Co-Learnability) as the task scoring function. The teacher samples tasks according to priority; after the student trains on a task, the difficulty buffer is updated.

### Key Designs

1. **Regret Approximation via Transition Prediction Loss**:

    - Function: Enhance the accuracy of regret estimation.
    - Theoretical Basis: The one-step regret is decomposed as $\text{Regret}(s,a) = \underbrace{V^*(s) - \hat{V}^*(s)}_{\text{(i) value error}} + \underbrace{r(s,a^*) - r(s,a)}_{\text{(ii) reward gap}} + \gamma \underbrace{(\mathbb{E}[\hat{V}^*(s'')] - \mathbb{E}[V^\pi(s')])}_{\text{(iii) future value gap}}$
    - PVL corresponds only to term (i), whereas term (iii) is affected by the mismatch between the learned transition model $\hat{P}$ and the true transition $P$.
    - ATPL is defined as: $\text{ATPL}(\tau) = \frac{1}{T}\sum_{t=0}^T L_{\text{trans}}(s_t, a_t)$
    - Combined regret approximation: $\widehat{\text{Regret}}(\tau) = \text{PVL}(\tau) + \alpha \cdot \text{ATPL}(\tau)$
    - Theoretical Support: The appendix proves that ATPL upper-bounds the dynamics component in term (iii).

2. **Co-Learnability Measure**:

    - Function: Quantify the transfer benefit of training on one task to other tasks.
    - Definition: $\text{CoLearnability}_i(k) = \frac{1}{|\mathcal{T}_{k+1}|}\sum_{j \in \mathcal{T}_{k+1}}[\text{TaskDifficulty}(j,k) - \text{TaskDifficulty}(j,k+1)]$
    - A positive value indicates that training on task $i$ reduces the difficulty of other replayed tasks.
    - Analogy: Spanish–English exhibits high Co-Learnability (shared word roots), whereas Japanese–English exhibits low Co-Learnability.
    - Design Motivation: Prevent the curriculum from focusing exclusively on difficult tasks while neglecting their positive transfer effects on other tasks.

3. **Task Priority**:

    - Combination formula: $\text{TaskPriority}(i,t) = \text{Rank}(\text{TaskDifficulty}(i,t) + \beta \cdot \text{CoLearnability}(i,t))$
    - Rank transformation: eliminates the influence of outliers while preserving relative ordering.
    - Sampling probability: $p(i|t) \propto 1/\text{TaskPriority}(i,t)$

### Loss & Training
The student is trained with PPO. The transition model $f_\phi$ is a recurrent network trained concurrently with the agent. MiniGrid uses 16 workers; BipedalWalker uses 4 workers.

## Key Experimental Results

### MiniGrid Zero-shot Transfer

| Method | 10k updates IQM | 20k updates IQM | Wall-clock (h) |
|--------|-----------------|-----------------|----------------|
| DR | Low | Low | 5.82±0.12 |
| PLR⟂ | Medium | Medium | 14.87±0.62 |
| ADD | Medium | Medium | 22.48±0.27 |
| ACCEL | Medium | Medium | 12.94±0.66 |
| **TRACED** | **Highest** | **—** | **13.78±0.36** |

TRACED at 10k updates surpasses all baselines at 20k updates, with wall-clock time reduced by approximately half.

### BipedalWalker Zero-shot Transfer
- TRACED at 10k updates outperforms ACCEL-CENIE at 20k updates across all metrics: median, IQM, mean, and optimality gap.
- TRACED consistently leads on all 6 terrain types.

### PerfectMaze Stress Test
- PerfectMazeLarge (51×51): TRACED 10k solved rate 27%±23% > ACCEL 20k 20%±25%.
- PerfectMazeXL (100×100): TRACED 10k 10%±14% approaches ACCEL 20k 12%±28%.

### Ablation Study

| Configuration | MiniGrid IQM | Description |
|---------------|-------------|-------------|
| TRACED (full) | Highest | ATPL + CL |
| TRACED − CL | Second | ATPL only; still outperforms baseline |
| TRACED − ATPL | Lowest | CL only; limited improvement |

ATPL is the primary driver; Co-Learnability provides additional gains when combined with ATPL.

### Curriculum Complexity Analysis
- Under TRACED, the shortest path length and obstacle count increase far more rapidly than under ACCEL.
- The curriculum progression from easy → moderate → challenging is markedly more efficient than the baseline.

## Highlights & Insights
- **Theoretical Contribution via Regret Decomposition**: The paper precisely identifies the shortcoming of PVL as a regret proxy—namely, the absence of a dynamics mismatch term. This insight is transferable to any UED method that employs regret.
- **Dual Effect of ATPL**: ATPL simultaneously improves regret estimation accuracy and accelerates curriculum complexity ramp-up, since tasks with high dynamics uncertainty are indeed more challenging.
- **Doubled Sample Efficiency**: 10k updates match baseline performance at 20k updates, with wall-clock time nearly halved.
- **Lightweight Co-Learnability**: The measure leverages existing difficulty-change information without requiring additional modeling overhead.

## Limitations & Future Work
- Co-Learnability uses a simple difficulty-change difference as a Shapley value surrogate, which may lack precision.
- The transition model $f_\phi$ requires additional training, introducing approximately 6% computational overhead.
- Experiments are limited to MiniGrid and BipedalWalker; validation in more complex 3D environments remains future work.
- Sensitivity analyses for $\alpha$ and $\beta$ are provided in the appendix, but adaptive tuning schemes are worth exploring.

## Related Work & Insights
- **vs. ACCEL**: TRACED builds directly on ACCEL, replacing only the scoring function. The improvement is orthogonal to ACCEL's contributions.
- **vs. CENIE**: CENIE uses environmental novelty as the scoring signal; TRACED implicitly captures novelty through ATPL.
- **vs. PLR⟂**: PLR⟂ employs PVL/MaxMC; TRACED demonstrates that these proxies are insufficiently accurate.

## Rating
- Novelty: ⭐⭐⭐⭐ — Regret decomposition that identifies PVL's shortcomings, ATPL augmentation, and Co-Learnability transfer measure constitute a meaningful combinatorial innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual environments, ablations, curriculum analysis, PerfectMaze stress tests, and statistical significance are all covered.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, effective integration of theory and experiments, and consistent notation.
- Value: ⭐⭐⭐⭐ — Provides clear improvements to the UED field; the method is concise and straightforward to integrate into existing frameworks.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Don't Just Fine-tune the Agent, Tune the Environment](dont_just_fine-tune_the_agent_tune_the_environment.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](../../NeurIPS2025/reinforcement_learning/scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/reinforcement_learning/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)

<!-- RELATED:END -->
