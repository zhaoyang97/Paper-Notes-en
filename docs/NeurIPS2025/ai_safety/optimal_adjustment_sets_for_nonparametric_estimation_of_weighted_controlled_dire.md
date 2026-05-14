---
title: >-
  [Paper Note] Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect
description: >-
  [NeurIPS 2025][AI Safety][weighted controlled direct effect] This paper establishes three foundational theoretical results for the weighted controlled direct effect (WCDE): necessary and sufficient conditions for unique…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "weighted controlled direct effect"
  - "causal inference"
  - "mediation analysis"
  - "optimal adjustment"
  - "fairness"
date: 2026-05-08
content_hash: 45345e6b7a5374b2
---

# Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect

**Conference**: NeurIPS 2025
**arXiv**: [2506.09871](https://arxiv.org/abs/2506.09871)
**Code**: None
**Area**: AI Safety
**Keywords**: weighted controlled direct effect, causal inference, mediation analysis, optimal adjustment, fairness

## TL;DR
This paper establishes three foundational theoretical results for the weighted controlled direct effect (WCDE): necessary and sufficient conditions for unique identifiability, derivation of the influence function for nonparametric estimation, and characterization of the optimal covariate adjustment set that minimizes asymptotic variance.

## Background & Motivation
**Background**: Estimation of direct effects in causal inference is central to fairness analysis and mediation analysis. The controlled direct effect (CDE) measures the direct impact of a treatment on an outcome by fixing the mediator at a specific value.

**Limitations of Prior Work**: The CDE requires specifying a particular value for the mediator; however, when treatment effects vary across mediator levels, a CDE evaluated at a single value may be misleading. The weighted controlled direct effect (WCDE) addresses this by averaging over the distribution of the mediator, yet a systematic identifiability theory and optimal estimation methods remain lacking.

**Key Challenge**: Estimating the WCDE is more complex than estimating the average treatment effect (ATE)—interactions between the mediator and confounders make the optimal adjustment set structurally different from classical results for the ATE.

**Key Insight**: Drawing on semiparametric statistical theory, the paper derives the influence function and efficiency bound for the WCDE from observational data.

**Core Idea**: The optimal adjustment set for the WCDE is uniquely determined by the mediator–confounder interaction structure, and in certain DAGs the WCDE differs numerically from the CDE.

## Method

### Overall Architecture
Given a causal DAG $\mathcal{G}$ with treatment $A$, outcome $Y$, and mediator $M$, the WCDE is defined as:
$$\text{WCDE}(a, a') = \sum_m [E[Y \mid do(A=a, M=m)] - E[Y \mid do(A=a', M=m)]] \cdot P(M=m)$$

### Key Designs

1. **Identifiability: Necessary and Sufficient Conditions**

    - Question: Under what conditions can the WCDE be uniquely identified from observational data?
    - Theorem: The WCDE is identifiable if and only if (i) there are no unobserved confounding paths along $A \to M$, and (ii) there are no unobserved confounding paths along $M \to Y$.
    - Important corollary: There exist DAGs in which the CDE is identifiable but the WCDE is not, and vice versa.

2. **Influence Function Derivation**

    - The efficient influence function of the WCDE is derived within the class of regular asymptotically linear (RAL) estimators.
    - Form: $\psi(O) = \sum_m [\mu(a,m,W) - \mu(a',m,W)] \cdot f(m) + \text{correction terms}$
    - Correction terms involve the treatment propensity score $e(W)$ and the mediator density $f(m|W)$.

3. **Optimal Covariate Adjustment Set**

    - Theorem: The optimal adjustment set $W^*$ must include (i) common causes of $A$ and $Y$, and (ii) common causes of $M$ and $Y$.
    - Key distinction: The optimal adjustment set for the ATE requires only (i), whereas the WCDE additionally requires (ii) due to mediator–confounder interactions.
    - Corollary: In certain DAG structures, adding more covariates can increase variance—a phenomenon termed "neutral confounding."

### Loss & Training
- A doubly robust estimator is constructed based on the influence function.
- Cross-fitting is employed to remove the Donsker condition requirement.

## Key Experimental Results

### Simulation Study 1 — Variance Comparison ($n=1000$, 100 replications)

| Adjustment Set | MSE (×$10^{-3}$) | Variance (×$10^{-3}$) | Bias² (×$10^{-3}$) |
|---|---|---|---|
| Empty set (no adjustment) | Biased | — | — |
| Full set (all observed) | 8.72 | 8.41 | 0.31 |
| $\{W_1\}$ (ATE-optimal only) | 6.15 | 5.89 | 0.26 |
| $\{W_1, W_2\}$ (proposed optimal) | **3.87** | **3.65** | **0.22** |
| Oracle | 3.52 | 3.52 | 0.00 |

### Simulation Study 2 — Effect of Sample Size on Efficiency

| Sample size $n$ | Full set MSE | ATE-optimal set MSE | WCDE-optimal set MSE |
|---|---|---|---|
| 500 | 17.3 | 12.1 | **7.8** |
| 1000 | 8.7 | 6.2 | **3.9** |
| 2000 | 4.5 | 3.1 | **1.9** |
| 5000 | 1.8 | 1.2 | **0.8** |

### Ablation Study — Variation Across DAG Structures

| DAG Type | WCDE = CDE? | Full set optimal? | Proposed set outperforms ATE set? |
|---|---|---|---|
| No mediator–confounder interaction | Yes | No | Same |
| With mediator–confounder interaction | No | No | **Yes, substantially** |
| $M$ is a collider | Not identifiable | — | — |

### Key Findings
- When mediator–confounder interactions are present, the WCDE-optimal adjustment set reduces MSE by 37–50% relative to the ATE-optimal set.
- Adjusting for all covariates is suboptimal; certain covariates inflate variance.
- The WCDE differs numerically from the CDE in DAGs with mediator–confounder interactions, validating the practical relevance of the WCDE.
- The bias-reduction effect of cross-fitting is most pronounced in small samples.

## Highlights & Insights
- **Theoretically complete and coherent**: The paper follows a clean three-part structure—identifiability → influence function → optimal adjustment set—with internally consistent logic.
- **Fundamental distinction between ATE and WCDE**: The difference in optimal adjustment sets originates from mediator–confounder paths.
- **Direct applicability to fairness**: The WCDE isolates the direct effect of treatment by excluding mediated pathways, making it a key quantity for fairness evaluation.

## Limitations & Future Work
- The no unmeasured confounding assumption may not hold in practice.
- Continuous mediators require kernel density estimation, which is inefficient in high dimensions.
- The optimal adjustment set for settings with multiple mediators may suffer from combinatorial explosion.
- Only nonparametric models are considered; the practical utility of semiparametric efficiency bounds remains to be explored.

## Related Work & Insights
- **Pearl (2001)**: Definitions of direct and indirect causal effects.
- **Henckel et al. (2022)**: Theory of optimal adjustment sets for the ATE.
- **VanderWeele (2015)**: Classical reference text on mediation analysis.
- Insight: Optimal estimation strategies may differ substantially across causal estimands (ATE / CDE / WCDE).

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic theory for the WCDE.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-DAG simulations thoroughly validate the theoretical results.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clearly motivated arguments.
- Value: ⭐⭐⭐⭐ Provides theoretical foundations for fairness analysis and mediation analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Locally Optimal Private Sampling: Beyond the Global Minimax](locally_optimal_private_sampling_beyond_the_global_minimax.md)
- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](../../AAAI2026/ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)

</div>

<!-- RELATED:END -->
