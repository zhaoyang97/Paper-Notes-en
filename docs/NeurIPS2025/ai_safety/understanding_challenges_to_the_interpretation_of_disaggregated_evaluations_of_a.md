---
title: >-
  [Paper Note] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI
description: >-
  [NeurIPS 2025][AI Safety][algorithmic fairness] Through causal graphical modeling, this paper demonstrates that performance disparities across subgroups in disaggregated evaluations do not necessarily indicate unfairness…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "algorithmic fairness"
  - "disaggregated evaluation"
  - "causal inference"
  - "distribution shift"
  - "subgroup performance"
date: 2026-05-08
content_hash: 3fa5ad5ee37442ce
---

# Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI

**Conference**: NeurIPS 2025
**arXiv**: [2506.04193](https://arxiv.org/abs/2506.04193)  
**Code**: [GitHub](https://github.com/google-research/google-research/tree/master/causal_evaluation)  
**Area**: AI Safety
**Keywords**: algorithmic fairness, disaggregated evaluation, causal inference, distribution shift, subgroup performance

## TL;DR

Through causal graphical modeling, this paper demonstrates that performance disparities across subgroups in disaggregated evaluations do not necessarily indicate unfairness, but may instead reflect natural consequences of distributional differences in the data-generating process. The authors recommend supplementing standard disaggregated evaluations with causal assumptions and weighted evaluation methods.

## Background & Motivation

### State of the Field

**Background**: Disaggregated evaluation is the standard practice for assessing ML model fairness: model performance is broken down by subgroup (e.g., race, gender), and performance disparities are treated as evidence of fairness problems. However, this paper argues that such practice can be misleading:

### Limitations of Prior Work

**Limitations of Prior Work**: Even Bayes-optimal models (with zero estimation error) generally do not achieve equal performance across subgroups.

### Root Cause

**Key Challenge**: Performance disparities may be a natural consequence of distributional differences across subgroups, rather than a model deficiency.

### Solution Direction

**Solution Direction**: In the presence of selection bias, both disaggregated evaluation and conditional-independence-based alternatives may fail.

### Additional Remarks

**Additional Remarks**: Algorithmic strategies that enforce equal performance across subgroups may directly introduce harm.

These findings have broad implications for the design and interpretation of model evaluations.

## Method

### Overall Architecture

This paper uses causal directed acyclic graphs (DAGs) to characterize different data-generating processes underlying subgroup heterogeneity. It analyzes the fairness properties and stability of performance metrics of Bayes-optimal models under each setting, and subsequently proposes weighted evaluation as a method for controlling confounding.

### Key Designs

1. **Causal Graphs for Subgroup Heterogeneity**: Six settings are defined—three for the causal direction (X→Y) and three for the anti-causal direction (Y→X)—plus a selection bias setting:

    - **Causal direction**: covariate shift ($P(X|A)$ differs but $P(Y|X,A)=P(Y|X)$), outcome shift ($P(X|A)$ is the same but $P(Y|X,A) \neq P(Y|X)$), and compound shift.
    - **Anti-causal direction**: label shift ($P(Y|A)$ differs but $P(X|Y,A)=P(X|Y)$), manifestation shift ($P(Y|A)$ is the same but $P(X|Y,A) \neq P(X|Y)$), and compound shift.
    - Bidirectional edges represent the influence of unobserved confounders, rather than treating $A$ as a direct cause of $X$ or $Y$ (reflecting the indirect role of social structural determinants).

2. **Fairness Properties of Bayes-Optimal Models**:

    - **Sufficiency** criterion ($Y \perp A | R$): Subgroup-specific Bayes-optimal predictors always satisfy sufficiency, whereas the population-level Bayes-optimal predictor satisfies it only under $Y \perp A | X$ (covariate shift).
    - **Separation** criterion ($R \perp A | Y$): Satisfied only under label shift with a subgroup-agnostic predictor.
    - Key conclusion: $Y \perp A | X$ is the decisive condition—when it holds, the population-level optimum equals the subgroup-level optimum, and subgroup-aware modeling is unnecessary.

3. **Stability Analysis of Performance Metrics (Table 2)**: The core condition is $\{R,Y\} \perp A | V$ for control variable $V \in \{X, Y, R\}$:

    - Across all settings, $\{R,Y\} \perp A$ does not hold → equal performance across subgroups on any metric should not be expected.
    - Under covariate shift: conditioning on $X$ fully accounts for performance disparities (for subgroup-agnostic models).
    - Under label shift: conditioning on $Y$ fully accounts for performance disparities.
    - Under outcome shift, manifestation shift, and compound shift: neither $X$ nor $Y$ alone can account for the disparities.
    - For models satisfying sufficiency: conditioning on $R$ can account for disparities.

4. **Weighted Evaluation**: When disparities can be attributed to distributional differences in $X$, inverse probability weighting (analogous to propensity score methods) is used to construct weighted performance estimates that align the $X$ distributions across subgroups. This is equivalent to a class of configurable conditional independence tests.

### Loss & Training

This paper does not propose new training methods. The core contribution is an analytical framework and evaluation methodology. Weighted evaluation employs standard inverse probability weighting and propensity score techniques.

## Key Experimental Results

### Main Results (Verification of Theoretical Properties)

| Causal Setting | Y⊥A\|X? | Sufficiency(f*) | Separation(f*) | X Explains Disparity? |
|---|---|---|---|---|
| Covariate shift | ✓ | ✓ | ✗ | ✓ |
| Outcome shift | ✗ | ✗ (population) | ✗ | ✗ |
| Label shift | ✗ | ✗ (population) | ✓ (agnostic) | ✗ |
| Manifestation shift | ✗ | ✗ (population) | ✗ | ✗ |
| Compound shift | ✗ | ✗ (population) | ✗ | ✗ |

Subgroup-specific Bayes-optimal predictors satisfy sufficiency across all settings.

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Unadjusted disaggregated evaluation | Performance gap ≠ unfairness | Under covariate shift, disparity is entirely attributable to $X$ distribution |
| Weighted evaluation controlling for $X$ | Eliminates confounding | Valid only when $Y \perp A \mid X$ |
| Weighted evaluation controlling for $Y$ | Removes label distribution differences | Valid only under label shift |
| Subgroup-aware vs. agnostic | Agnostic is more stable | But sacrifices accuracy when $Y \not\perp A \mid X$ |
| Subgroup separability | Behaves like $Y \perp A \mid X$ | At extreme separation ratios, population-level ≈ subgroup-level |
| Selection bias extension | Evaluation may fail | Requires explicit assumptions about the bias mechanism |

### Key Findings

- Equal performance across subgroups is not a reliable measure of fairness: Bayes-optimal models inherently do not achieve equal performance in the presence of distributional differences.
- Enforcement constraints (e.g., equalized odds constrained optimization) may directly introduce harm rather than achieve fairness.
- Covariate shift is the only setting in which performance disparities can be fully accounted for by conditioning on $X$.
- Label shift is the only setting in which the separation criterion is naturally satisfied.
- When the causal structure is unknown, simple disaggregated evaluation is insufficient to support fairness judgments.

## Highlights & Insights

- The paper presents a systematic theoretical challenge to a widely adopted yet rarely questioned evaluation paradigm.
- Causal graphs are used to unify multiple types of distributional shift, enabling comparative analysis of properties across different scenarios.
- The comprehensive analysis in Table 2 provides highly valuable practical guidance, informing practitioners which control variables are appropriate under which causal assumptions.
- The paper emphasizes that fairness is not a property of the model but rather an effect of the deployment policy—a perspective of particular importance for policymakers.

## Limitations & Future Work

- The theoretical analysis is conducted primarily under the Bayes-optimality assumption; in practice, performance disparities may simultaneously reflect both estimation error and distributional differences.
- The choice of causal graph itself requires domain knowledge, whereas the true causal structure is often unknown in practice.
- Weighted evaluation relies on accurate propensity score estimation, which may be unreliable in high-dimensional covariate spaces.
- The analysis considers only binary labels and simple causal graphs; complex settings (multi-label, continuous outputs, multi-level causal structures) are not covered.
- Experiments are primarily based on synthetic and simple real-world data; validation on large-scale, complex ML systems is limited.

## Related Work & Insights

- Liu et al. (2019)'s impossibility theorem on the incompatibility of sufficiency, separation, and calibration serves as the direct theoretical foundation of this work.
- Cai et al.'s weighted evaluation method is further developed and given a causal interpretation in this paper.
- Mhasawade et al.'s use of causal models to analyze algorithmic fairness is a methodological predecessor.
- The paper has direct applied connections to the clinical prediction fairness literature (cardiovascular risk prediction, medical imaging diagnosis).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Presents a theoretical challenge to a widely used evaluation paradigm with a complete analytical framework.
- **Experimental Thoroughness**: ⭐⭐⭐ Primarily theoretical, with experiments serving mainly to validate theoretical properties.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorously argued; causal graphs effectively visualize complex problems.
- **Value**: ⭐⭐⭐⭐⭐ Has far-reaching implications for AI fairness evaluation practice; should be required reading for all practitioners conducting disaggregated evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D](ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)
- [\[NeurIPS 2025\] Matchings Under Biased and Correlated Evaluations](matchings_under_biased_and_correlated_evaluations.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)

</div>

<!-- RELATED:END -->
