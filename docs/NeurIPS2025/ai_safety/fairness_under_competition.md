---
title: >-
  [Paper Note] Fairness under Competition
description: >-
  [NeurIPS 2025][AI Safety][Algorithmic Fairness] This paper is the first to study the joint fairness of multiple fair classifiers operating in a competitive environment. It theoretically demonstrates that even when each individual classifier satisfies Equal Opportunity (EO), the ecosystem as a whole may remain unfair, and that applying fairness adjustments to a biased classifier can paradoxically reduce ecosystem-level fairness.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Algorithmic Fairness"
  - "Competitive Ecosystems"
  - "Equal Opportunity"
  - "Multi-Classifier Fairness"
  - "Ecosystem Fairness"
date: 2026-05-08
content_hash: 4ce017d0fba6509e
---

# Fairness under Competition

**Conference**: NeurIPS 2025
**arXiv**: [2505.16291](https://arxiv.org/abs/2505.16291)  
**Code**: [GitHub](https://github.com/eilamshapira/FairnessUnderCompetition)  
**Area**: AI Safety
**Keywords**: Algorithmic Fairness, Competitive Ecosystems, Equal Opportunity, Multi-Classifier Fairness, Ecosystem Fairness

## TL;DR
This paper is the first to study the joint fairness of multiple fair classifiers operating in a competitive environment. It theoretically demonstrates that even when each individual classifier satisfies Equal Opportunity (EO), the ecosystem as a whole may remain unfair, and that applying fairness adjustments to a biased classifier can paradoxically reduce ecosystem-level fairness.

## Background & Motivation

**Background**: Algorithmic fairness has become a central topic in ML, with constraints such as Equal Opportunity (EO) and Demographic Parity (DP) widely adopted to adjust classifiers to meet fairness requirements.

**Limitations of Prior Work**: Existing research almost exclusively focuses on the fairness of individual classifiers, overlooking real-world scenarios in which multiple entities (banks, employers, insurance companies) simultaneously deploy different classifiers to make decisions over the same population.

**Key Challenge**: Even when each classifier independently satisfies EO, differences in inter-classifier correlation and overlap in the populations served can produce systemic unfairness—one group may have "two chances" to obtain a loan while another has only "one chance."

**Goal**: (1) Formally define fairness under competition (EOC); (2) Quantify the extent to which EO classifiers can violate EOC; (3) Prove that fairness adjustments can backfire at the ecosystem level.

**Key Insight**: The analysis proceeds along two dimensions: Pearson correlation between classifiers and the degree of overlap in the populations they serve.

**Core Idea**: Individual fairness is neither a sufficient nor a necessary condition for ecosystem fairness; differences in inter-classifier correlation and differences in population coverage are the two fundamental driving forces.

## Method

### Overall Architecture

The framework involves two types of players: borrowers $(x, a, y) \in X \times A \times Y$ and a set of lenders $L$. Each lender $\ell$ employs a classifier $c_\ell: X \times A \mapsto \{0,1\}$ to decide whether to extend credit. $y=1$ denotes a "qualified borrower." The false negative rate of classifier $\ell$ is defined as $\beta_\ell = \Pr[c_\ell(X,A)=0|Y=1]$.

### Key Definitions

1. **Equal Opportunity (EO)**:
    - Function: Requires that a classifier's false negative rates be equal across both groups.
    - Core Definition: EO level $= |E[c_\ell(X,A)|Y=1,A=0] - E[c_\ell(X,A)|Y=1,A=1]|$; an EO level of 0 indicates that EO is satisfied.
    - Design Motivation: Ensures that qualified applicants from different groups receive equal acceptance probabilities.

2. **Equal Opportunity under Competition (EOC)**:
    - Function: Defines a fairness measure for settings with multiple competing classifiers.
    - Core Definition: Let $d(x,a) = \Pr[R(x,a) \geq 1]$ denote the probability of receiving at least one offer, where $R(x,a) = \sum_{\ell \in L} c_\ell(x,a)$. EOC level $= |E[d(X,A)|Y=1,A=0] - E[d(X,A)|Y=1,A=1]|$.
    - Design Motivation: In a competitive ecosystem, individuals care about whether they receive at least one offer, rather than whether any single classifier is fair.

### Force I: Differences in Inter-Classifier Correlation (Section 3.1)

Define two Bernoulli variables $B_\ell^a \equiv c_\ell(X,A)|(Y=1, A=a)$ with Pearson correlation $\rho^a$.

**Proposition 1**: For two EO classifiers with false negative rates $\beta_1, \beta_2$, the EOC level is:

$$\text{EOC level} = \sigma_1 \cdot \sigma_2 \cdot |\rho^0 - \rho^1|$$

where $\sigma_\ell = \sqrt{\beta_\ell(1-\beta_\ell)}$. The worst-case EOC level is $\min\{\beta_1, \beta_2\} - \max\{0, \beta_1+\beta_2-1\}$.

**Intuition**: If two classifiers are highly correlated for group 0 (e.g., both use the same model) but independent for group 1, qualified borrowers in group 1 have "two independent chances" to receive an offer while those in group 0 effectively have only "one."

**Corollary 1**: When $\beta_1 = \beta_2 = \beta \leq 1/2$, the worst-case EOC level is $\beta$—on the same order as the false negative rate.

### Force II: Differences in Population Coverage (Section 3.2)

When the borrower subsets $S_1, S_2$ served by the two classifiers do not fully overlap, let $\gamma^a$ denote the fraction of group $a$ covered by both classifiers.

**Proposition 4**: For two uncorrelated EO classifiers, the EOC level is:

$$|(\gamma_2^0 - \gamma_2^1)\beta_1 + (\gamma_1^0 - \gamma_1^1)\beta_2 + (\gamma^1 - \gamma^0)\beta_1\beta_2|$$

**Corollary 3**: When $\beta_1 = \beta_2 = \beta$, EOC level $= \beta(1-\beta)|\gamma^0 - \gamma^1|$; larger disparities in overlap rates lead to more severe EOC violations.

### Harmful Effects of Fairness Adjustment (Section 4)

Post-processing (Hardt et al., 2016) is applied to bring non-EO classifiers into EO compliance:

- **Example 3**: Two classifiers that violate EO but satisfy EOC before adjustment satisfy EO but no longer satisfy EOC after adjustment. The adjustment alters the correlation structure between classifiers across groups.
- **Example 4**: A perfect classifier, after adjustment, introduces a uniform false negative rate; due to asymmetric service coverage, this results in ecosystem-level unfairness.

### Extension: Multiple Classifiers and General Utility

- **Proposition 3**: The worst-case EOC level for $n$ EO classifiers is $\min_i \beta_i - \max\{0, \sum_j \beta_j - 1\}$.
- EOC worsens as the number of classifiers grows: with $n$ independent classifiers, EOC $= \beta - \beta^n$, which is increasing in $n$.

## Key Experimental Results

### Main Results: Lending Club Data

Experiments use approximately 890K loan records from Lending Club (2007–2015), with collateral status as the protected attribute.

| Experiment | Training Set Size | Probability of EOC Deterioration After Fairness Adjustment (95% CI) |
|---|---|---|
| Exp 1 (LR vs DT, same data) | 100K | [26.2%, 34.0%] |
| Exp 2 (LR vs LR, different data) | 100K | [12.6%, 19.0%] |
| Exp 3 (LR vs DT, different data) | 100K | [14.2%, 20.6%] |
| Exp 1 (LR vs DT, same data) | 300 | [75.0%, 82.2%] |
| Exp 2 (LR vs LR, different data) | 300 | [75.6%, 82.8%] |

### EOC Deterioration Factor

| Experiment | Training Set Size 100K | Description |
|---|---|---|
| Exp 1 | Average EOC deterioration 19× | Different model types, same data |
| Exp 2 | Average EOC deterioration 1.3× | Same model type, different data |
| Exp 3 | Average EOC deterioration 3.1× | Different model types + different data |

### Key Findings
- The probability of EOC deterioration following fairness adjustment reaches 75–82% on small training sets and remains 15–34% even on large training sets (100K).
- The EOC deterioration factor can reach 19× when classifiers differ in model type but share the same training data.
- As training set size increases, classifier accuracy improves, false negative rates decline, and EOC violations diminish correspondingly.

## Highlights & Insights
- **Highly novel problem formulation**: This is the first work to extend fairness research from individual classifiers to competitive ecosystems, identifying two fundamental forces—correlation disparity and coverage disparity—as drivers of unfairness.
- **Concise yet powerful theoretical results**: The finding that EOC level is on the same order as the false negative rate is highly intuitive and points toward "improving classifier accuracy" as a direction that simultaneously benefits both performance and fairness.
- **Counterintuitive finding**: Fairness adjustment via post-processing can harm ecosystem-level fairness, a result with important implications for policy design.

## Limitations & Future Work
- The theoretical analysis assumes simplifying conditions such as 0-1 preferences and two groups; real-world settings involve more complex group structures and utility functions.
- Experiments employ a simulated protected attribute (collateral status) rather than genuine demographic information, which may underestimate the severity of the problem in practice.
- The paper offers no concrete intervention mechanism at the ecosystem level—it diagnoses the problem without prescribing a solution.
- Strategic interactions among classifiers (e.g., fairness properties at Nash equilibrium) are not considered.

## Related Work & Insights
- **vs. Bower et al. (2017)**: Their work studies sequential composition of classifiers in a pipeline; this paper studies parallel competition, enabling the capture of inter-classifier correlation effects.
- **vs. Dwork & Ilvento (2019)**: They focus on individual fairness in subtasks within a platform and assume classifier independence; the correlation analysis in this paper is a key innovation.
- **vs. Liu et al. (2018)**: That work examines the long-term dynamic effects of fairness (temporal dimension), while this paper examines competitive/parallel effects (spatial dimension); the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to formally study fairness under competition; the problem formulation and identification of two fundamental forces are highly pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Theory and experiments complement each other well, though real demographic attributes and additional datasets are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematically rigorous, intuitively motivated, and clearly structured.
- Value: ⭐⭐⭐⭐ — Important implications for fairness policy, though concrete intervention strategies are absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cost Efficient Fairness Audit Under Partial Feedback](cost_efficient_fairness_audit_under_partial_feedback.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](../../ICML2026/ai_safety/optimal_transport_under_group_fairness_constraints.md)
- [\[NeurIPS 2025\] Matchings Under Biased and Correlated Evaluations](matchings_under_biased_and_correlated_evaluations.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)

</div>

<!-- RELATED:END -->
