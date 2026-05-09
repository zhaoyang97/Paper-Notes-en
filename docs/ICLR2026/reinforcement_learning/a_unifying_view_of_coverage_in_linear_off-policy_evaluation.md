---
title: >-
  [Paper Note] A Unifying View of Coverage in Linear Off-Policy Evaluation
description: >-
  [ICLR 2026][Reinforcement Learning][off-policy evaluation] This paper proposes a novel coverage parameter—**feature-dynamics coverage**—and conducts a new finite-sample analysis of the classical LSTDQ algorithm through an instrumental variable lens, unifying the various fragmented coverage definitions in linear off-policy evaluation.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - off-policy evaluation
  - coverage
  - linear function approximation
  - LSTDQ
  - feature-dynamics coverage
date: 2026-05-08
content_hash: 0bff4c5661bee11c
---

# A Unifying View of Coverage in Linear Off-Policy Evaluation

**Conference**: ICLR 2026
**arXiv**: [2601.19030](https://arxiv.org/abs/2601.19030)
**Code**: None
**Area**: Reinforcement Learning / Off-Policy Evaluation
**Keywords**: off-policy evaluation, coverage, linear function approximation, LSTDQ, feature-dynamics coverage

## TL;DR
This paper proposes a novel coverage parameter—**feature-dynamics coverage**—and conducts a new finite-sample analysis of the classical LSTDQ algorithm through an instrumental variable lens, unifying the various fragmented coverage definitions in linear off-policy evaluation.

## Background & Motivation
Off-Policy Evaluation (OPE) is a fundamental problem in reinforcement learning: given data collected by a behavior policy, evaluate the value of a different target policy. This is critical in settings where online interaction is infeasible (e.g., healthcare, recommendation systems).

In the classical linear OPE setting, finite-sample guarantees typically take the form:

$$\text{evaluation error} \leq \text{poly}(C^\pi, d, 1/n, \log(1/\delta))$$

where $d$ is the feature dimension, $n$ is the sample size, and $C^\pi$ is the **coverage parameter**—characterizing how well the data distribution covers the feature space accessed by the target policy.

**Root Cause / Fragmentation Problem**:

Under stronger assumptions (e.g., Bellman completeness), the notion of coverage is well-defined and the guarantees of various classical algorithms are well understood. However, under the **minimal assumption setting** (requiring only linear realizability of the target value function), the landscape becomes highly fragmented:
- No consensus exists on the "correct" notion of coverage.
- Coverage definitions used across different analyses are mutually inconsistent and exhibit undesirable properties (e.g., not distribution-free, unable to recover standard definitions in special cases).
- The lack of connections among definitions leads to a fragmented theoretical understanding.

**Goal**: To propose a unified coverage concept that yields tight finite-sample guarantees under minimal assumptions and gracefully recovers known standard coverage definitions under stronger assumptions.

## Method

### Overall Architecture
- **Algorithm**: The core algorithm analyzed is LSTDQ (Least-Squares Temporal Difference for Q-values), a classical algorithm in linear OPE.
- **Analytical Tool**: The instrumental variable (IV) perspective.
- **Core Contribution**: Proposing feature-dynamics coverage and deriving new finite-sample bounds for LSTDQ under this notion.

### Key Designs
1. **Instrumental Variable Perspective**:

    - **Mechanism**: LSTDQ is reinterpreted as an instrumental variable regression problem. In econometrics and causal inference, IV methods address endogeneity—when regressors are correlated with the error term, an "instrument" that is correlated with the regressor but uncorrelated with the error enables consistent estimation.
    - **Design Motivation**: In linear OPE, the structure of the Bellman equation is naturally amenable to an IV interpretation—features of the current state-action pair serve as "endogenous variables," while features mapped through the transition dynamics serve as "instruments."
    - **Key Insight**: This IV perspective naturally gives rise to the definition of feature-dynamics coverage.

2. **Feature-Dynamics Coverage**:

    - **Definition**: Interpreted as a linear coverage measure in a system induced by the evolution of features under the environment dynamics.
    - **Intuition**: Measures how well the data distribution of the behavior policy covers the "trajectory of features under dynamics"—accounting not only for coverage of current features but also of features after environment transitions.
    - **Mathematical Properties**: Satisfies natural desiderata—distribution-dependent yet naturally defined, and recovers standard definitions in special cases.

3. **Unification Results**:

    - Under Bellman completeness, feature-dynamics coverage reduces to the concentrability coefficient, the standard coverage notion in that setting.
    - In the tabular setting, it recovers the classical state-action visitation ratio.
    - In the general linear realizability setting, it yields tighter bounds than prior analyses.
    - This is the first work to connect all these seemingly disparate coverage definitions within a unified framework.

### Theoretical Results
- **Main Theorem**: A finite-sample error bound for LSTDQ under feature-dynamics coverage $C^\pi_{FD}$:
    - The evaluation error grows polynomially in $C^\pi_{FD}$ and $d$, and decreases at a rate of $1/\sqrt{n}$.
    - A high-probability bound (depending on $\log(1/\delta)$).
- **Reduction Properties**: Under stronger assumptions, $C^\pi_{FD}$ can be replaced by smaller coverage parameters, recovering known optimal rates.
- **Necessity Argument**: Information-theoretic tools are used to argue that $C^\pi_{FD}$ is unavoidable under minimal assumptions.

## Key Experimental Results

### Main Results
This paper is primarily a theoretical contribution, though numerical experiments are included to validate the theoretical results.

| Setting | Metric | Key Finding |
|---------|--------|-------------|
| Synthetic MDP (linear realizability) | MSE vs. $n$ | LSTDQ error follows the theoretically predicted $1/\sqrt{n}$ rate |
| Synthetic MDP (Bellman completeness) | MSE vs. $C^\pi$ | Error dependence on feature-dynamics coverage matches theory |
| Data distributions with varying coverage | Comparison of definitions | Feature-dynamics coverage yields tighter bounds than prior definitions |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Bellman completeness + feature-dynamics | Error bound | Reduces to concentrability bound, validating unification |
| Linear realizability only + prior coverage | Error bound | Prior definitions yield looser bounds |
| Linear realizability only + feature-dynamics | Error bound | Proposed definition yields tighter bounds |
| Varying dimension $d$ | Error bound | Polynomial dependence on $d$ is verified |

### Key Findings
- Feature-dynamics coverage is a more natural and tighter parameter than previously proposed coverage definitions.
- Under Bellman completeness, it perfectly reduces to the known optimal coverage parameter.
- LSTDQ has a cleaner statistical interpretation through the IV lens.
- Several "unusual" coverage definitions from prior analyses (e.g., algorithm-dependent definitions) emerge as special cases of feature-dynamics coverage.

## Highlights & Insights
- **Theoretical Elegance**: Linear OPE has long suffered from a proliferation of seemingly incompatible coverage definitions. This paper ties them together under a single concept—a clarifying contribution that cuts through longstanding confusion.
- **Novelty of the IV Perspective**: Connecting the OPE problem in RL to the IV theory in econometrics opens a new avenue of analytical tools.
- **Understanding Under Minimal Assumptions**: Tight analysis under the minimal assumption of linear realizability, a setting previously poorly understood.
- **Conceptual Contribution**: The "induced dynamical system" interpretation of feature-dynamics coverage is highly illuminating—it suggests that the difficulty of OPE depends not only on the data distribution but also on how the environment dynamics amplify coverage deficiencies.

## Limitations & Future Work
- **Purely Theoretical**: No empirical validation on real RL tasks.
- **Restricted to Linear Settings**: Modern RL predominantly uses nonlinear function approximation (e.g., neural networks); whether the framework extends remains an open question.
- **Focus on OPE, Not OPL**: Off-policy evaluation and off-policy learning are technically distinct; whether the proposed coverage notion extends to the latter is unclear.
- **Computational Feasibility**: Whether feature-dynamics coverage can be efficiently estimated in practice is unaddressed; if not, the practical utility of the theoretical guarantees may be limited.
- **Single-Policy Evaluation**: Extension to simultaneous evaluation of multiple policies or policy optimization scenarios is left for future work.

## Related Work & Insights
- **Classical Linear OPE Algorithms**: LSTD, LSTDQ, FQE (Fitted Q Evaluation), etc.—this paper revisits the most classical among them, LSTDQ.
- **Coverage / Concentrability Coefficient**: The concentrability coefficient is a central concept in OPE theory; this paper provides its correct generalization to the general setting.
- **Instrumental Variables**: A tool from econometrics for addressing endogeneity, introduced here into RL theory.
- **Intersection of Econometrics and RL**: An increasing body of work interprets RL through the lens of causal inference; this paper is a significant contribution to this trend.
- **Open Question**: Can feature-dynamics coverage be leveraged to design adaptive data collection strategies that maximize OPE efficiency?

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](../../NeurIPS2025/reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)
- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

</div>

<!-- RELATED:END -->
