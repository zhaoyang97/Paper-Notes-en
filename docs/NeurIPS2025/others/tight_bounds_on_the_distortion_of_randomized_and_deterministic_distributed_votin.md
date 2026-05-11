---
title: >-
  [Paper Note] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting
description: >-
  [NeurIPS 2025][metric distortion] This paper studies metric distortion in the distributed voting model. For four cost objectives ($\text{avg-avg}$, $\text{avg-max}$, $\text{max-avg}$, $\text{max-max}$)…
tags:
  - "NeurIPS 2025"
  - "metric distortion"
  - "distributed voting"
  - "deterministic mechanisms"
  - "randomized mechanisms"
  - "social choice"
date: 2026-05-08
content_hash: 47c19d6d125f0c5e
---

# Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting

**Conference**: NeurIPS 2025
**arXiv**: [2509.17134](https://arxiv.org/abs/2509.17134)
**Code**: None
**Area**: Other
**Keywords**: metric distortion, distributed voting, deterministic mechanisms, randomized mechanisms, social choice

## TL;DR

This paper studies metric distortion in the distributed voting model. For four cost objectives ($\text{avg-avg}$, $\text{avg-max}$, $\text{max-avg}$, $\text{max-max}$), it establishes improved tight or near-tight bounds under both deterministic and randomized mechanisms, providing an almost complete characterization of distortion in this model.

## Background & Motivation

### Distributed Voting Model
The distributed voting model captures systems analogous to the U.S. presidential election: $n$ voters are partitioned into $k$ groups (e.g., states), each group elects a local representative, and a winner is then selected from these representatives (or from all candidates). Such two-stage mechanisms are prevalent in practice.

### Metric Distortion

**Core Setting**: Voters and candidates are embedded in a metric space, with voter preferences determined by distances to candidates. Each voter submits an ordinal ranking over candidates rather than exact distance values. **Distortion** measures the worst-case ratio between the social cost achieved by a voting mechanism — which has access only to ordinal preferences — and the optimal social cost.

### Four Cost Objectives
- $\text{avg-avg}$: average intra-group cost + average inter-group cost
- $\text{avg-max}$: average intra-group cost + maximum inter-group cost
- $\text{max-avg}$: maximum intra-group cost + average inter-group cost
- $\text{max-max}$: maximum intra-group cost + maximum inter-group cost

### Prior Work
Anshelevich et al. (2022) initiated the study of this problem but left several gaps between upper and lower bounds.

## Method

### Overall Architecture

This paper systematically addresses the distortion bounds for all four objectives under both deterministic and randomized mechanisms, employing the following methodology:
1. **Constructing new mechanisms** → proving upper bounds
2. **Constructing hard instances** → proving lower bounds
3. **Matching upper and lower bounds** → obtaining tight bounds

### Key Designs

#### Improvements for Deterministic Mechanisms

| Cost Objective | Prior UB | **This Work UB** | Prior LB | **This Work LB** |
|---|---|---|---|---|
| $\text{avg-avg}$ | 7 | 7 | 5 | 5 |
| $\text{avg-max}$ | 11 | **7** | 5 | 5 |
| $\text{max-avg}$ | 7 | 7 | $2+\sqrt{5}$ | **5** (tight) |
| $\text{max-max}$ | 5 | **3** | 3 | 3 (tight) |

Key improvements:
- **$\text{avg-max}$**: Upper bound reduced from 11 to 7 via a more refined two-stage selection mechanism.
- **$\text{max-avg}$**: Lower bound improved from $2+\sqrt{5} \approx 4.236$ to 5, matching the upper bound (tight).
- **$\text{max-max}$**: Upper bound reduced from 5 to 3, matching the lower bound (tight).

#### Randomized Mechanisms — Case (i): Randomization in Second Stage Only

After each group deterministically selects a representative, the final selection may be randomized:

| Cost Objective | Bound | Status |
|---|---|---|
| $\text{avg-avg}$ | $5 - 2/k$ | **Tight** |
| $\text{avg-max}$ | 3 | **Tight** |
| $\text{max-avg}$ | 5 | **Tight** |
| $\text{max-max}$ | 3 | **Tight** |

Tight bounds are obtained for all four objectives.

#### Randomized Mechanisms — Case (ii): Both Stages Randomized

Randomization is permitted in both stages:

| Cost Objective | LB | UB | Status |
|---|---|---|---|
| $\text{avg-avg}$ | $3 - 2/n$ | $3 - 2/(kn^*)$ | Near-tight |
| $\text{avg-max}$ | $3 - 2/n$ | 3 | Near-tight |
| $\text{max-avg}$ | 3 | 3 | **Tight** |
| $\text{max-max}$ | 3 | 3 | **Tight** |

Here $n^*$ denotes the size of the largest group.

### Proof Techniques

- **Upper bounds**: New voting rules are constructed, exploiting the triangle inequality in metric spaces and the median voter argument.
- **Lower bounds**: Adversarial configurations of candidates and voters in the metric space are carefully constructed to force high distortion for any mechanism.
- **Randomized lower bounds**: Yao's minimax lemma is applied to reduce lower bounds for randomized mechanisms to the analysis of deterministic mechanisms on random instances.

### Loss & Training

This is a purely theoretical work with no training procedure. The core optimization objective is minimizing the worst-case distortion ratio:
$$\text{Distortion}(f) = \sup_{\text{instances}} \frac{\text{SC}(f(\text{instance}))}{\text{SC}(\text{OPT})}$$
where SC denotes the social cost function.

## Key Experimental Results

### Main Results

This is a theoretical paper. The following table summarizes all main results:

| Cost Objective | Deterministic | Det. Status | Randomized (2nd only) | Randomized (both) |
|---|---|---|---|---|
| $\text{avg-avg}$ | [5, 7] | gap remains | $5-2/k$ (tight) | $[3-2/n, 3-2/(kn^*)]$ |
| $\text{avg-max}$ | [5, 7] | gap remains | 3 (tight) | $[3-2/n, 3]$ |
| $\text{max-avg}$ | **5 (tight)** | tight | 5 (tight) | **3 (tight)** |
| $\text{max-max}$ | **3 (tight)** | tight | 3 (tight) | **3 (tight)** |

Comparison of improvements over prior work:

| Result | Prior Best | This Work | Improvement |
|---|---|---|---|
| Det. $\text{avg-max}$ UB | 11 | 7 | −36% |
| Det. $\text{max-avg}$ LB | $2+\sqrt{5} \approx 4.24$ | 5 | +18% |
| Det. $\text{max-max}$ UB | 5 | 3 | −40% |
| Rand. (2nd only) $\text{avg-avg}$ | open | $5-2/k$ | new result |
| Rand. (both stages) all objectives | open | near-tight/tight | new results |

### Ablation Study

Effect of the number of groups $k$ on distortion under randomized mechanisms:

| Groups $k$ | $\text{avg-avg}$ distortion | $\text{avg-max}$ distortion |
|---|---|---|
| 2 | 4.0 | 3.0 |
| 5 | 4.6 | 3.0 |
| 10 | 4.8 | 3.0 |
| 50 | 4.96 | 3.0 |
| $\to \infty$ | 5.0 | 3.0 |

### Key Findings

1. **Distortion lies in the range 3–7**: Distributed voting inevitably introduces information loss, but this loss is bounded.
2. **Randomization provides substantial benefit**: For most objectives, distortion decreases from 5–7 under deterministic mechanisms to 3 under randomized ones.
3. **Full two-stage randomization outperforms second-stage-only randomization**: Most notably, $\text{max-avg}$ distortion decreases from 5 to 3.
4. **Limited room for further improvement**: Most results are tight or near-tight.

## Highlights & Insights

1. **Near-complete characterization**: Tight bounds are obtained for the majority of the 12 combinations of four objectives × three mechanism types.
2. **Practical relevance**: The results contribute to the theoretical understanding of distributed voting systems such as U.S.-style elections.
3. **Diverse proof techniques**: Constructive upper bound arguments and adversarial lower bound constructions complement each other throughout.
4. **Unified treatment**: All objectives and mechanism types are systematically addressed within a single paper.

## Limitations & Future Work

1. **Gap remains for deterministic $\text{avg-avg}$ and $\text{avg-max}$**: The interval [5, 7] has not been closed.
2. **Randomized (both stages) $\text{avg-avg}$ is near-tight but not tight**: The gap $[3-2/n,\, 3-2/(kn^*)]$ depends on problem parameters.
3. **Only ordinal information is considered**: Real-world voting systems may support richer preference representations.
4. **The metric space assumption is strong**: True preferences may not satisfy the triangle inequality.

## Related Work & Insights

- **Anshelevich et al. (2022)**: Introduced the distributed voting model and established initial results.
- **Metric distortion theory**: Classical results by Anshelevich & Postl in the non-distributed setting.
- **Approximate mechanism design**: The ideas developed here may extend to other mechanism design problems with information constraints.
- **Future directions**: Investigating non-metric preference spaces, weighted groups, and dynamic group partition settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematically improves bounds on an established problem.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Purely theoretical work with sophisticated proof techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ — No experiments by nature; results are illustrated via examples and tables.
- **Practical Impact**: ⭐⭐⭐ — Theoretical contribution with indirect implications for real-world voting system design.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with intuitive result tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)

</div>

<!-- RELATED:END -->
