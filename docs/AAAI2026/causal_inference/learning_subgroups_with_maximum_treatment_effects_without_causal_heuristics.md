---
title: >-
  [Paper Note] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics
description: >-
  [AAAI 2026][Causal Inference][Treatment Effects] Under the SCM framework, the paper proves that the subgroup with maximum treatment effect must exhibit homogeneous pointwise effects (Theorem 1); under the partition model assumption, it proves that optimal subgroup discovery reduces to standard supervised learning (Theorem 2), achievable via CART with the Gini index. On 77 ACIC-2016 semi-synthetic datasets, the proposed method achieves a mean treatment effect of 10.54 (vs. 7.8…
tags:
  - "AAAI 2026"
  - "Causal Inference"
  - "Treatment Effects"
  - "Subgroup Discovery"
  - "CART"
  - "Partition Model"
date: 2026-05-08
content_hash: 4e60944e06f84115
---

# Learning Subgroups with Maximum Treatment Effects without Causal Heuristics

**Conference**: AAAI 2026
**arXiv**: [2511.20189](https://arxiv.org/abs/2511.20189)  
**Code**: [https://github.com/ylincen/causal-subgroup](https://github.com/ylincen/causal-subgroup)  
**Area**: Causal Inference / Subgroup Discovery
**Keywords**: Treatment Effects, Subgroup Discovery, CART, Partition Model, Causal Inference

## TL;DR
Under the SCM framework, the paper proves that the subgroup with maximum treatment effect must exhibit homogeneous pointwise effects (Theorem 1); under the partition model assumption, it proves that optimal subgroup discovery reduces to standard supervised learning (Theorem 2), achievable via CART with the Gini index. On 77 ACIC-2016 semi-synthetic datasets, the proposed method achieves a mean treatment effect of 10.54 (vs. 7.84 for the runner-up), ranking first on 51.9% of datasets.

## Background & Motivation

**Background**: Discovering subgroups with maximum average treatment effect is a central problem in causal inference. Existing methods (CausalTree, QUINT, SIDES, etc.) design specialized "causal heuristics" to construct splitting criteria.

**Limitations of Prior Work**: Causal heuristics are fragile — (1) imbalance between treatment and control groups degrades split quality during tree growth; (2) specialized splitting criteria lack theoretical optimality guarantees; (3) assumptions vary substantially across methods, leading to inconsistent results.

**Key Challenge**: Does causal subgroup discovery genuinely require specialized "causal" methods, or is standard supervised learning sufficient?

**Goal**: To prove that, under reasonable assumptions, maximum-effect subgroup discovery is equivalent to a standard classification/regression problem.

**Key Insight**: Theory-driven — first prove the homogeneity theorem (Theorem 1), then prove the reduction theorem under the partition model (Theorem 2), and finally implement the approach using the simplest possible CART.

**Core Idea**: The subgroup with maximum treatment effect must be one of the homogeneous partitions → learn partitions via CART → estimate treatment effect for each partition → select the one with the largest effect.

## Method

### Overall Architecture
(1) Learn data partitions using CART with Gini index (classification) or MSE (regression); (2) *Honest inference*: learn the tree on a training set, estimate treatment effects for each leaf on a held-out test set; (3) Select the leaf with the largest estimated effect as the target subgroup.

### Key Designs

1. **Theorem 1 (Homogeneity)**: The maximum-effect subgroup $Q^*$ must have homogeneous pointwise treatment effects — $Q' \subset Q$: if $A(Q') \leq A(Q)$, then $A(Q \setminus Q') \geq A(Q)$.

2. **Theorem 2 (Reduction)**: Under the partition model $Y = f_Y(T, \sum_i i \cdot \mathbf{1}_{K_i}(X), N_Y)$, the maximum-effect subgroup must correspond to some partition $K_i$ → standard supervised learning suffices to recover the partition.

3. **Theorem 3 (Extension)**: Results extend to settings with hidden confounders $U$.

4. **Honest Inference**: Training/evaluation split — tree structure is learned on one half of the data; treatment effects are estimated on the other half, preventing overfitting.

### Loss & Training
CART: Gini index (classification) or MSE (regression), with cost-complexity pruning and cross-validation.

## Key Experimental Results

### Main Results (77 ACIC-2016 Semi-Synthetic Datasets)

| Method | Mean Treatment Effect ↑ | Rank-1 Ratio |
|--------|------------------------|--------------|
| **Ours (CART)** | **10.540** | **51.9%** |
| CausalTree | 7.843 | 14.3% |
| CURLS | 7.410 | 18.0% |
| DistillTree | 7.451 | 13.0% |
| InteractionTree | 6.280 | 3.9% |
| QUINT | 5.135 | 0.0% |
| SIDES | 4.622 | 1.3% |

### Statistical Significance (Holm-corrected Wilcoxon)

| vs. Method | $p_{\text{holm}}$ |
|------------|-------------------|
| QUINT | 7.75e-14 |
| SIDES | 7.75e-14 |
| InteractionTree | 8.68e-12 |
| CausalTree | 1.21e-04 |
| CURLS | 1.30e-05 |

### Key Findings
- Standard CART ranks first on 51.9% of the 77 datasets, substantially outperforming all specialized causal methods.
- All improvements are statistically significant under Wilcoxon + Holm correction ($p < 0.001$).
- Causal heuristic methods are fragile under treatment/control imbalance — Gini/MSE is unaffected by this issue.
- Results generalize beyond the partition model assumption, remaining effective on semi-synthetic data.

## Highlights & Insights
- **The theoretical conclusion that "causal heuristics are unnecessary" is striking** — under reasonable assumptions, the simplest CART outperforms all specialized causal methods.
- **Honest inference** (training/evaluation separation) is a critical implementation detail — it prevents the bias introduced by selecting subgroups and estimating effects on the same data.

## Limitations & Future Work
- The partition model assumption may not fully hold in settings with continuous treatment effects.
- Axis-aligned splits in CART constrain the shape of discoverable subgroups.
- Performance in high-dimensional feature spaces (>50 features) has not been validated.

## Related Work & Insights
- **vs. CausalTree**: CausalTree employs causally-specific splitting criteria, yielding a mean effect of 7.84 vs. 10.54 for the proposed method.
- **vs. CURLS**: CURLS uses a regularized causal loss, yet still underperforms standard Gini.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical conclusion that causal heuristics are unnecessary overturns conventional wisdom.
- Experimental Thoroughness: ⭐⭐⭐⭐ 77 semi-synthetic + synthetic datasets, 7 baselines, and statistical significance testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theorem–experiment correspondence and rigorous logic.
- Value: ⭐⭐⭐⭐⭐ Offers fundamental methodological insights for causal inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[ICLR 2026\] Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](../../ICLR2026/causal_inference/matching_without_group_barrier_for_heterogeneous_treatment_effect_estimation.md)
- [\[ICML 2026\] Rank-Learner: Orthogonal Ranking of Treatment Effects](../../ICML2026/causal_inference/rank-learner_orthogonal_ranking_of_treatment_effects.md)
- [\[ICLR 2026\] Influence without Confounding: Causal Discovery from Temporal Data with Long-term Carry-over Effects](../../ICLR2026/causal_inference/influence_without_confounding_causal_discovery_from_temporal_data_with_long-term.md)
- [\[ICLR 2026\] Topological Causal Effects](../../ICLR2026/causal_inference/topological_causal_effects.md)

</div>

<!-- RELATED:END -->
