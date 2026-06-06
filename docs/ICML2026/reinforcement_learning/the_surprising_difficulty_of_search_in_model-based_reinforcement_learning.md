---
title: >-
  [Paper Note] The Surprising Difficulty of Search in Model-Based Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Model Predictive Control] The authors counter-intuitively identify that the failure of search in model-based RL stems not from model inaccuracy…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Model Predictive Control"
  - "Value Overestimation"
  - "Ensemble Minimum"
  - "Model-as-Representation"
  - "Search"
date: 2026-05-08
content_hash: ad6d6daeaad3e5dd
---

# The Surprising Difficulty of Search in Model-Based Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2601.21306](https://arxiv.org/abs/2601.21306)  
**Code**: https://github.com/facebookresearch/MRSQ  
**Area**: Reinforcement Learning / Model-Based RL / Planning  
**Keywords**: Model Predictive Control, Value Overestimation, Ensemble Minimum, Model-as-Representation, Search

## TL;DR
The authors counter-intuitively identify that the failure of search in model-based RL stems not from model inaccuracy, but from overestimation bias caused by the discrepancy between the MPC behavioral policy and the value function training policy. They propose the MRS.Q algorithm, which utilizes the "minimum" over an ensemble of 10 value functions, outperforming SOTA methods like TD-MPC2, BMPC, BOOM, and SimbaV2 across 50+ continuous control tasks.

## Background & Motivation
**Background**: The premise of MBRL is to "learn a dynamics model → imagine the future → plan." TD-MPC2, using MPPI short-horizon search guided by value functions, serves as a strong baseline. Concurrently, MR.Q, a representative model-free method, treats the model only as an auxiliary objective for representation learning without search, yet achieves SOTA performance. The mainstream diagnosis for MBRL failure has been that "dynamics prediction errors accumulate over time, making longer searches worse," leading to research focus on more accurate models or uncertainty modeling.

**Limitations of Prior Work**: (a) More accurate models and longer search horizons often fail to improve performance; (b) Directly adding MPC to MR.Q degrades its performance; (c) Existing "search-aided" works (TD-M(PC)², BMPC, BOOM) constrain the policy to imitate search actions, which are unstable during training. These observations suggest the implicit assumption that "better models lead to better search" is flawed.

**Key Challenge**: The behavioral policy $\pi_{\text{MPC}}$ for search and the target policy $\pi$ assumed during value function training are inconsistent. The value function is evaluated using $\pi$ but queried with data collected by $\pi_{\text{MPC}}$. This causes query points to fall outside the training distribution, triggering offline-RL-like overestimation bias (Fujimoto 2018). Model accuracy cannot resolve this mismatch.

**Goal**: The problem is decomposed into three questions: (i) Does search have intrinsic difficulties even with perfect dynamics and values? (ii) Does model accuracy predict search gains? (iii) What actually determines search success if not model accuracy?

**Key Insight**: Leveraging the architectural similarities between MR.Q and TD-MPC2, the authors perform cross-hybridization experiments—adding MPC to MR.Q and removing it from TD-MPC2—to isolate the variables of "search" and "model quality," thereby pinpointing the true causal factors.

**Core Idea**: The search space expansion is controlled by fixing the horizon to 3 steps. Pessimistic estimation via the minimum of a 10-value-function ensemble is applied both during "target calculation" and "MPC final value calculation" to suppress overestimation at its source. The resulting algorithm is named MRS.Q (Model-based Representations for Search and Q-learning).

## Method

### Overall Architecture
The paper provides two diagnostic phases followed by an algorithm. Diagnosis: (1) Using the N-chain MDP with an analytical solution, it is shown that even with perfect dynamics and values, the probability of uniform random search finding a non-zero reward trajectory is $1-(1-\frac{1}{A^n})^m$, which decays exponentially with horizon $n$. (2) Comparing MR.Q and TD-MPC2 on 17 DMC/Gym tasks regarding "dynamics error + unroll error" and "performance delta with/without MPC" proves that while error magnitudes are similar, the impact of MPC is opposite. Algorithm: MRS.Q makes minimal changes to MR.Q—adding MPPI short-horizon search, increasing the ensemble size from 2 to 10 with a "min" reduction, adding Simplicial Embeddings (SEM), removing extra exploration noise, and increasing the termination loss weight from 0.1 to 1.

### Key Designs

1. **Dual Diagnosis of Intrinsic Search Difficulty and Model Accuracy Independence**:
    - **Function**: Transitions the narrative of search failure from "insufficient model accuracy" to "search mechanism and value learning coupling."
    - **Mechanism**: On N-chain, the success rate for $m$ trajectories of length $n$ is $1-(1-A^{-n})^m$. At $A=10, m=1000$, success is 0.63 for $n=3$ but drops to $10^{-7}$ for $n=10$. This combinatorial difficulty exists even with a "perfect model." Empirically, after aligning embedding scales using SEM, dynamics MSE and unroll errors for MR.Q+MPC are found to be comparable to or lower than TD-MPC2. However, MPC helps TD-MPC2 while hurting MR.Q (e.g., Humanoid-stand −757).
    - **Design Motivation**: To steer research away from solely pursuing model accuracy and clarify that short horizons are a necessity rather than an engineering compromise.

2. **Overestimation Bias as the Core Mechanism of Search Failure**:
    - **Function**: Identifies a specific, measurable, and treatable metric—the percentage of value overestimation under MPC behavior.
    - **Mechanism**: Value updates $Q(s,a)\approx r+\gamma Q(s',\pi(s'))$ use the learned policy $\pi$, but data is collected by $\pi_{\text{MPC}}$. Querying $a\sim\pi_{\text{MPC}}$ moves outside the distribution of $\pi$. Measuring $|Q_{\text{learned}}-\hat{G}_{\text{behavior}}|/\hat{G}_{\text{behavior}}$ reveals consistent positive overestimation for MR.Q+MPC across 17 tasks, strongly correlating with performance degradation.
    - **Design Motivation**: To provide empirical evidence that "mitigating overestimation = fixing search."

3. **MRS.Q: 10-Value-Function Ensemble Min + Short-horizon MPPI + Refinements**:
    - **Function**: Safely integrates search into the MR.Q representation learning framework.
    - **Mechanism**: Pessimistic estimation is applied by taking the minimum of 10 $Q_i$ functions in the update: $Q(s,a)\approx r+\gamma\min_{i\in\{1,...,10\}} Q_i(s',\pi(s'))$. Crucially, this min is also used for evaluating the final value of MPC trajectories: $V(\tau)=\sum_{t=0}^{N-1}\gamma^t R(\tilde{z}_t,a_t)+\gamma^N \min_i Q_i(\tilde{z}_N,pi(\tilde{z}_N))$. Additional refinements include: (a) 3-step MPPI horizon; (b) Removal of $\mathcal{N}(0,0.2^2)$ noise as MPC provides sufficient perturbation; (c) SEM for stable multi-step rollouts and increased dynamics loss weight; (d) Increased termination loss weight to 1.0.
    - **Design Motivation**: Avoiding policy imitation of search (unlike TD-M(PC)²/BMPC) because search actions drift rapidly (3-5x the variance of policy networks). Instead, the method addresses the root cause by suppressing value overestimation.

### Loss & Training
Inherits MR.Q's $\mathcal{L}(z_s,W_p,W_r) = \mathcal{L}_{\text{Dyn}}(z_{sa}^\top W_p - z_{s'}) + \mathcal{L}_{\text{Reward}}(z_{sa}^\top W_r - r)$. Value learning utilizes $\mathcal{L}_{\text{Value}}(r+\gamma\min_{i=1..10}Q_i(z_{s'a'})-Q(z_{sa}))$. MPPI uses default TD-MPC2 hyperparameters. All 50+ tasks are trained for 1M steps across 10 seeds using a single hyperparameter set.

## Key Experimental Results

### Main Results: Aggregate Performance at 1M Steps (10 seeds, 95% CI)

| Algorithm | MPC | Gym (TD3-norm) | DMC | HB (No Hand) | HB (Hand) |
|---|---|---|---|---|---|
| MR.Q | × | 1.46 [1.41, 1.52] | 0.84 [0.83, 0.84] | 0.48 [0.46, 0.49] | 0.31 [0.29, 0.32] |
| MR.Q + MPC | ✓ | 0.67 [0.55, 0.88] | 0.65 [0.63, 0.68] | 0.46 [0.45, 0.48] | 0.38 [0.37, 0.39] |
| TD-MPC2 | ✓ | 0.41 [0.27, 0.57] | 0.78 [0.77, 0.80] | 0.58 [0.56, 0.60] | 0.22 [0.19, 0.25] |
| TD-M(PC)² | ✓ | 0.62 | 0.76 | 0.51 | 0.44 |
| BMPC | ✓ | 0.54 | 0.86 | 0.40 | 0.38 |
| BOOM | ✓ | 0.61 | 0.83 | 0.55 | 0.23 |
| SimbaV2 | × | 1.44 | 0.84 | 0.38 | 0.18 |
| **MRS.Q (Ours)** | ✓ | **1.54 [1.46, 1.60]** | 0.81 [0.79, 0.82] | **0.59 [0.58, 0.60]** | **0.58 [0.57, 0.58]** |

Ours ranks first in 3 out of 4 benchmarks, with performance on HB-Hand being 1.3x the runner-up.

### Ablation Study: Performance Change Relative to Full MRS.Q (10 seeds)

| Configuration | Gym | DMC | HB (No Hand) | HB (Hand) | Description |
|---|---|---|---|---|---|
| Ensemble size 2 | −0.63 | −0.04 | −0.18 | −0.40 | Default ensemble is insufficient to suppress overestimation |
| Ensemble size 5 | −0.37 | 0.00 | −0.02 | −0.13 | Close to optimal but Gym still degrades |
| Ensemble size 20 | −0.05 | −0.03 | +0.03 | +0.03 | Saturated returns; double computation not cost-effective |
| Re-add exploration noise | −0.13 | −0.02 | −0.02 | −0.02 | MPC already introduces enough noise; extra noise interferes |
| Remove SEM | −0.28 | −0.04 | +0.06 | +0.10 | SEM primarily maintains Gym/DMC multi-step stability |
| Min of 2 (randomly sampled) | −0.49 | +0.01 | −0.05 | −0.19 | Proves full ensemble minimum is critical |
| Min not used for MPC eval | −0.33 | −0.02 | −0.04 | −0.19 | Target min is insufficient; MPC final value requires it |
| Backport Min(10) to TD-MPC2 | +0.07 | +0.03 | +0.02 | +0.11 | Proves overestimation is a universal issue |

### Key Findings
- The benefits of ensemble size saturate at $\approx 10$.
- Using the minimum for MPC trajectory ranking is crucial; otherwise, search explores paths with high overestimation. Search acts as a maximization-bias amplifier.
- Applying Min(10) to TD-MPC2 also yields gains, demonstrating the generalizability of the solution.
- The high step-to-step variance of MPC actions compared to policy networks explains why imitation-based methods (TD-M(PC)²) struggle.

## Highlights & Insights
- **Paradigm Shift**: Redirects the focus from "improving model accuracy" to "managing value overestimation," a significant diagnostic contribution to the MBRL community.
- **Minimalist SOTA**: Achieves superior results with simple modifications (Ensemble 10 + Min), providing a low engineering barrier for adoption.
- **N-chain Argument**: Uses an analytical toy environment to prove why short-horizon search is a necessity.
- **Harmonized Perspectives**: Reconciles the "model for representation" and "model for search" perspectives by showing they can coexist if value estimation is controlled.

## Limitations & Future Work
- The computational overhead of a 10-Q ensemble is significant and was not deeply discussed regarding memory/throughput.
- Validated primarily on continuous control with short horizons (3 steps); generalizability to discrete actions (e.g., MuZero) or long-horizon planning remains untested.
- Min-of-ensemble is an empirical pessimistic estimate without formal bounds; conditions for detrimental "pessimistic underestimation" are not characterized.

## Related Work & Insights
- **vs TD-MPC2 (Hansen 2024)**: Differing in ensemble treatment (TD-MPC2 uses mean for MPC and random min-of-2 for targets), MRS.Q shows that the size and location of the "min" operation are key.
- **vs TD-M(PC)² / BMPC / BOOM**: While these works constrain policy to match search, MRS.Q directly suppresses overestimation, avoiding the noise of volatile search actions.
- **vs MR.Q (Fujimoto 2025)**: Transitioning from "no search" to "safe search," demonstrating that search is viable if its associated distribution shift is managed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[ICLR 2026\] BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](../../ICLR2026/reinforcement_learning/bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)

</div>

<!-- RELATED:END -->
