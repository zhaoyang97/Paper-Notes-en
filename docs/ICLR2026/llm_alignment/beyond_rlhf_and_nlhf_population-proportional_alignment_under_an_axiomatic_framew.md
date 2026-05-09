---
title: >-
  [Paper Note] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework
description: >-
  [ICLR 2026][LLM Alignment][RLHF] This paper proposes a preference learning framework grounded in social choice theory axioms. It infers the feasible set of evaluator population distributions from pairwise comparison data and constructs policies satisfying two axioms: Population-Proportional Alignment (PPA) and Population-Bounded Manipulability (PBM).
tags:
  - ICLR 2026
  - LLM Alignment
  - RLHF
  - NLHF
  - preference learning
  - social choice theory
  - population-proportional alignment
  - axiomatic framework
date: 2026-05-08
content_hash: 4204910332dd3a1a
---

# Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework

**Conference**: ICLR 2026
**arXiv**: [2506.05619](https://arxiv.org/abs/2506.05619)
**Code**: Experimental code included in supplementary material
**Area**: AI Alignment / Social Choice Theory
**Keywords**: RLHF, NLHF, preference learning, social choice theory, population-proportional alignment, axiomatic framework

## TL;DR
This paper proposes a preference learning framework grounded in social choice theory axioms. It infers the feasible set of evaluator population distributions from pairwise comparison data and constructs policies satisfying two axioms: Population-Proportional Alignment (PPA) and Population-Bounded Manipulability (PBM).

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **State of the Field**: 1. RLHF relies on the Bradley-Terry model to compress preferences into scalar rewards, failing under inconsistent or cyclic preferences. 2. NLHF models preference learning as a game and seeks Nash equilibrium policies, but still fails to reflect the evaluator distribution proportionally. 3. **Core Problem**: When two groups of evaluators split nearly 50:50 over two options (e.g., 50+ε vs. 50−ε), both RLHF and NLHF output a deterministic policy selecting the slim majority, completely ignoring the minority group. 4. Existing pluralistic alignment methods (mixture-based, steerable models) typically require explicit evaluator group labels, which are difficult to obtain in practice. 5. Existing axiomatic approaches (e.g., Random Dictatorship) satisfy proportional alignment but cannot be implemented from pairwise comparison data alone. 6. **Paper Goals**: Achieve proportional alignment solely from pairwise comparison data, without requiring additional group-label information.

## Method

### Framework Design
- **Feasible Population Distribution Inference**: Define $u_i = \min_{y \neq y_i} P(y_i \succ y)$ as an upper bound on the population share for each option $y_i$, and construct a polyhedral outer approximation $\bar{\mathcal{W}}(P) = \{w \in \Delta(\mathcal{Y}) | w_i \leq u_i\}$.
- **Policy Construction**: Allocate probabilities proportional to the upper bounds as $\pi(y_i) = u_i / \sum_j u_j$, adopting a conservative policy that minimizes worst-case proportional mismatch.

### Four Axioms
1. **Monotonicity**: Improving the ranking of an option does not decrease its selection probability.
2. **Pareto Efficiency**: If all evaluators prefer $y$ over $y'$, the policy should favor $y$.
3. **α-PPA (Population-Proportional Alignment)**: $\pi(y_k)/w_k^\sigma \geq \alpha(\sigma)$, ensuring the policy is at least weakly proportional to the population share.
4. **γ-PBM (Population-Bounded Manipulability)**: The gain from manipulation is bounded by $\gamma_1 w_k^\sigma + \gamma_2$, preventing non-majority groups from attaining majority status through manipulation.

### Softmax Relaxation
- A parameter $\beta$ is introduced to control the trade-off between proportional alignment and Condorcet consistency: $\pi(y_i) = u_i \exp(\beta u_i) / \sum_j u_j \exp(\beta u_j)$.
- $\beta=0$ recovers the original $F^*$; $\beta \to \infty$ converges to the minimax Condorcet method.

## Key Experimental Results

### Tabular Experiment: MovieLens Movie Recommendation

### Main Results

| Method | Win Rate | PPA Level | PBM Gain |
|--------|----------|-----------|----------|
| RLHF | 0.7784 | 0 | 0.0611 |
| NLHF | 0.7712 | 0 | 0.0124 |
| $F^\beta$($\beta=1$) | ~0.60 | 0.4869 | 8.9e-4 |

- As $\beta$ increases, win rate improves but PPA decreases, empirically validating the theoretically predicted trade-off.
- The proposed method exhibits significantly better manipulation resistance than baselines when $\beta \leq 10$.

### LLM Experiment: Qwen2.5-3B-Instruct

### Ablation Study

| Dataset | β=0 PPA | DPO PPA |
|---------|---------|---------|
| Synthetic-Color | 0.0883 | 0.0000 |
| Alpaca-Expertise | 0.1428 | 0.1321 |
| Alpaca-Style | 0.5012 | 0.3786 |

- The trade-off is pronounced on synthetic data; results on Alpaca data are weaker due to annotation noise from GPT-4.1.
- Computational cost is comparable to RLHF and higher than DPO.

## Highlights & Insights
- Theoretical rigor: The paper formally proves that RLHF and NLHF violate PPA and PBM axioms at any strength.
- The feasible set of population distributions can be inferred from pairwise comparison data alone, without group labels.
- The softmax relaxation provides a tunable trade-off between proportional alignment and Condorcet consistency.
- Manipulation resistance is theoretically guaranteed: non-majority groups cannot attain majority status through strategic misreporting.

## Limitations & Future Work
- PPA only concerns the selection probability of each group's top-ranked option, ignoring lower-ranked preferences.
- Evaluating PPA levels in LLM settings remains an open problem (both logit estimation and group classification introduce noise).
- The two-stage function approximation approach incurs computational overhead no lower than RLHF; a direct policy optimization variant is needed.
- The outer approximation $\bar{\mathcal{W}}$ may become overly loose when the number of options is large.

## Related Work & Insights
- **RLHF / DPO**: Equivalent to maximizing the Borda rule, deterministically selecting the winner.
- **NLHF**: Equivalent to Maximal Lotteries, satisfying Pareto but not PPA.
- **Random Dictatorship**: Achieves perfect PPA but cannot be realized from pairwise comparison data.
- **Pluralistic Alignment** (Sorensen 2024, Chen 2024): Requires explicit group labels.
- **Manipulation-resistant mechanisms** (Buening 2025, Park 2024): Pursue strict strategy-proofness; this paper instead constrains population-level gains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](../../ACL2026/llm_alignment/beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](../../NeurIPS2025/llm_alignment/position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)

<!-- RELATED:END -->
