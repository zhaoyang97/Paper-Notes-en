---
title: >-
  [Paper Note] A Unifying View of Coverage in Linear Off-Policy Evaluation
description: >-
  [ICLR 2026][Reinforcement Learning][LSTDQ] This paper proposes a new coverage parameter—**feature-dynamics coverage**, providing a novel finite-sample analysis of the classic LSTDQ algorithm through an instrumental variable perspective, unifying various fragmented coverage definitions in linear off-policy evaluation.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - LSTDQ
date: 2026-05-08
content_hash: bbd7fc7c3ce0bb88
---
# A Unifying View of Coverage in Linear Off-Policy Evaluation

**Conference**: ICLR 2026  
**arXiv**: [2601.19030](https://arxiv.org/abs/2601.19030)  
**Code**: None  
**Area**: Reinforcement Learning / Off-Policy Evaluation  
**Keywords**: Off-policy evaluation, coverage, linear function approximation, LSTDQ, feature-dynamics coverage

## TL;DR
This paper proposes a new coverage parameter—**feature-dynamics coverage**, providing a novel finite-sample analysis of the classic LSTDQ algorithm through an instrumental variable perspective, unifying various fragmented coverage definitions in linear off-policy evaluation.

## Background & Motivation
Off-Policy Evaluation (OPE) is a fundamental problem in reinforcement learning: evaluating the value of a target policy using data collected by a different behavior policy. This is critical in scenarios where online interaction is impossible, such as healthcare or recommendation systems.

In the classic setting of linear OPE, finite-sample guarantees typically take the following form:

$$\text{Evaluation Error} \leq \text{poly}(C^\pi, d, 1/n, \log(1/\delta))$$

Where $d$ is the feature dimension, $n$ is the number of samples, and $C^\pi$ is the **coverage parameter**, which describes how well the data distribution covers the feature space visited by the target policy.

**Key Challenge**:

While coverage definitions are clear under strong assumptions (e.g., Bellman completeness), the situation is chaotic under the **minimal assumption setting** (where only the target value function is required to be linearly realizable):
- No consensus exists on the "correct" concept of coverage.
- Different analyses use conflicting coverage definitions with undesirable properties (e.g., not being distribution-independent or failing to recover standard definitions in special cases).
- The lack of connection between various definitions leads to fragmented theoretical understanding.

**Goal**: To propose a unified coverage concept that provides tight finite-sample guarantees under minimal assumptions while gracefully reducing to known standard coverage definitions under stronger assumptions.

## Method

### Overall Architecture
The paper does not propose a new algorithm. Instead, it develops a new finite-sample theory for the classic LSTDQ (Least-Squares Temporal Difference for Q-values) by reinterpreting it as an instrumental variable regression. This naturally derives the **feature-dynamics coverage** parameter and proves evaluation error bounds under it. These bounds gracefully recover familiar standard coverage definitions under stronger assumptions, integrating fragmented definitions into a single framework.

### Key Designs

**1. Instrumental Variable Perspective: Reinterpreting LSTDQ as IV Regression to bypass endogeneity**

The difficulty of linear OPE lies in the endogeneity within the Bellman equation—when performing least squares directly on $Q(s,a)\approx\phi(s,a)^\top w$, the regressor $\phi(s,a)$ is correlated with the TD error term, causing bias in ordinary least squares. Instrumental Variables (IV) are standard tools in econometrics for handling such issues. This paper observes that the Bellman equation's structure inherently fits an IV interpretation: the current state-action features $\phi(s,a)$ act as endogenous variables, while the features mapped after environmental transitions, $\mathbb{E}[\phi(s',a')]$, serve as instruments. This framework provides a clear econometric explanation for LSTDQ's statistical behavior.

**2. Feature-Dynamics Coverage: Explicitly modeling how dynamics amplify coverage deficits**

A byproduct of the IV perspective is the new coverage parameter $C^\pi_{\mathrm{FD}}$. It measures coverage within a linear system induced by feature evolution dynamics: it considers not just how the behavior policy covers current features $\phi(s,a)$, but also how it covers "where the features evolve back into after environmental transitions." Intuitively, OPE difficulty depends on how environmental dynamics amplify weak coverage along trajectories; $C^\pi_{\mathrm{FD}}$ explicitly incorporates this amplification effect. It remains distribution-dependent but collapses back to standard recognized coverage in special cases.

**3. Unification and Error Bounds: One parameter to link all prior definitions**

The paper provides a main error bound for LSTDQ: the error grows polynomially with $C^\pi_{\mathrm{FD}}$ and feature dimension $d$, and decreases at a rate of $1/\sqrt{n}$ with high probability, denoted as $\text{Error} \lesssim \mathrm{poly}(C^\pi_{\mathrm{FD}},d) \cdot n^{-1/2} \cdot \sqrt{\log(1/\delta)}$. Its true value lies in its degradation behavior: under Bellman completeness, $C^\pi_{\mathrm{FD}}$ reduces to the standard linear coverage; under state abstraction, it covers the $\chi^2$ version of aggregated concentrability; and under minimal assumptions, it provides a tighter bound than previous candidates like $1/\sigma_{\min}(A)$. The paper highlights that $1/\sigma_{\min}(A)$ lacks scale invariance and fails to unify with other analyses.

## Key Experimental Results

### Main Results
While primarily a theoretical work, the paper includes numerical verifications.

| Setting | Metric | Key Findings |
|------|------|----------|
| Synthetic MDP (Linear Realizable) | MSE vs n | LSTDQ error follows the predicted $1/\sqrt{n}$ rate |
| Synthetic MDP (Bellman Complete) | MSE vs $C^\pi$ | Error relationship with feature-dynamics coverage fits theory |
| Diverse Coverage Data Distributions | Comparison of definitions | Feature-dynamics coverage is tighter than prior definitions |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Bellman Complete + Feature-Dynamics | Error Bound | Redundant with concentrability bound, verifying unification |
| Linear Realizable Only + Prior Definitions | Error Bound | Prior definitions yield looser bounds |
| Linear Realizable Only + Feature-Dynamics | Error Bound | Ours provides tighter bounds |
| Different Dimensions $d$ | Error Bound | Polynomial dependence is verified |

### Key Findings
- Feature-dynamics coverage is a more natural and tighter parameter than previously proposed definitions.
- Under Bellman completeness, it perfectly degrades to the known optimal coverage parameter.
- The LSTDQ algorithm has a clearer statistical explanation through the IV perspective.
- "Strange" algorithm-dependent coverage definitions in prior work are special cases of feature-dynamics coverage.

## Highlights & Insights
- **Elegance of Theoretical Unification**: It unifies multiple seemingly incompatible coverage definitions in linear OPE into a single concept.
- **Novelty of the IV Perspective**: Linking RL's OPE problem with econometric IV theory provides new analytical tools.
- **Understanding under Minimal Assumptions**: It provides tight analysis in the linear realizability setting where previous understanding was limited.
- **Conceptual Contribution**: The "induced dynamical system" interpretation suggests OPE difficulty is not just about data distribution but also how environmental dynamics "amplify" coverage gaps.

## Limitations & Future Work
- **Purely Theoretical**: Lacks experimental validation on large-scale real-world RL tasks.
- **Linear Setting**: Whether this framework extends to non-linear function approximation (e.g., neural networks) remains to be studied.
- **Evaluation vs. Learning**: Technical differences exist between OPE and Off-Policy Learning (OPL); it is unclear if this coverage concept applies to the latter.
- **Computational Feasibility**: It is unknown if feature-dynamics coverage can be efficiently estimated in practice.
- **Single Policy**: The work focuses on single policy evaluation rather than multi-policy or optimization scenarios.

## Related Work & Insights
- **Linear OPE Algorithms**: Re-analyzes classic LSTDQ in comparison to LSTD and FQE.
- **Coverage/Concentrability**: Provides a correct generalization of concentrability coefficients for the general linear setting.
- **Instrumental Variables**: Introduces econometric ideas to handle endogeneity in RL.
- **Interdisciplinary RL**: Contributes to the growing trend of causal inference and econometrics in RL theory.
- **Inspiration**: This suggests potential for designing adaptive data collection strategies that maximize OPE efficiency based on feature-dynamics coverage.

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
- [\[ICLR 2026\] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration](off-policy_safe_reinforcement_learning_with_cost-constrained_optimistic_explorat.md)
- [\[ICLR 2026\] Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses](primal-dual_policy_optimization_for_linear_cmdps_with_adversarial_losses.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](enhancing_generative_auto-bidding_with_offline_reward_evaluation_and_policy_sear.md)

</div>

<!-- RELATED:END -->
