---
title: >-
  [Paper Note] Matchings Under Biased and Correlated Evaluations
description: >-
  [NeurIPS 2025][AI Safety][stable matching] This paper introduces a correlation parameter $\gamma$ (the degree of alignment between institutional evaluations) into a two-institution stable matching model, and analyzes how bias $\beta$ and correlation $\gamma$ jointly affect the representation ratio of disadvantaged groups. It proves that even a slight loss of correlation can cause a sharp drop in representation, and characterizes the Pareto frontier of fairness interventions.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "stable matching"
  - "bias"
  - "correlation"
  - "fairness"
  - "representation ratio"
date: 2026-05-08
content_hash: c521b12a2f0effa3
---

# Matchings Under Biased and Correlated Evaluations

**Conference**: NeurIPS 2025
**arXiv**: [2510.23628](https://arxiv.org/abs/2510.23628)  
**Code**: None  
**Area**: Algorithmic Fairness / Matching Theory
**Keywords**: stable matching, bias, correlation, fairness, representation ratio

## TL;DR
This paper introduces a correlation parameter $\gamma$ (the degree of alignment between institutional evaluations) into a two-institution stable matching model, and analyzes how bias $\beta$ and correlation $\gamma$ jointly affect the representation ratio of disadvantaged groups. It proves that even a slight loss of correlation can cause a sharp drop in representation, and characterizes the Pareto frontier of fairness interventions.

## Background & Motivation
**Background**: Stable matching mechanisms are widely used in college admissions, labor markets, and digital platforms. Candidate evaluations typically rely on standardized tests, interviews, or AI scoring systems, which may exhibit group-dependent biases.

**Limitations of Prior Work**: Kleinberg & Raghavan (2018) analyzed the impact of bias in centralized matching under the assumption that all institutions use identical evaluations ($\gamma = 1$). In practice, however, institutions are decentralized and use overlapping but non-identical signals—for example, sharing standardized test scores while maintaining independent review processes.

**Key Challenge**: How do bias ($\beta$) and evaluation correlation ($\gamma$) jointly shape group-level representation? When inter-institutional evaluation alignment decreases, do minority groups suffer disproportionately?

**Key Insight**: The paper considers two institutions where each candidate has two independent attributes $v_{i1}, v_{i2}$; institution 1 uses $v_{i1}$, and institution 2 uses $\gamma v_{i1} + (1-\gamma)v_{i2}$. Scores for the disadvantaged group $G_2$ are scaled by a factor $\beta \in (0,1]$.

**Core Idea**: Closed-form equilibrium thresholds for stable matching are derived in the large-market limit. The 16 potential cases are compressed into 3 interpretable regimes based on $\gamma$, yielding a piecewise closed-form expression for the representation ratio $\mathcal{R}(\beta, \gamma)$.

## Method

### Overall Architecture
- **Model**: Two institutions with capacities $c_1 n, c_2 n$. Candidates belong to either the advantaged group $G_1$ or the disadvantaged group $G_2$, with sizes $\nu_1 n, \nu_2 n$.
- **Evaluations**: $u_{i1} = v_{i1}$, $u_{i2} = \gamma v_{i1} + (1-\gamma)v_{i2}$; scores for group $G_2$ are multiplied by $\beta$.
- **Preferences**: The main text assumes all candidates prefer institution 1 (reflecting prestige-driven preferences in practice).
- **Equilibrium**: In the large-market limit, the stable matching is determined by two deterministic thresholds $(s_1^*, s_2^*)$.

### Key Designs

1. **Regime Reduction Technique**

    - **Function**: Reduces the solution for $s_2^*$ from 16 possible cases to 3 interpretable regimes based on $\gamma$.
    - **Mechanism**: By analyzing derived threshold values $\gamma_1, \gamma_2, \gamma_3$, the regime of $\gamma$ is identified; the equilibrium equation takes a different form in each regime.
    - **Intuition**: Under different values of $\gamma$, the minimum admission threshold $s_2^*$ of institution 2 corresponds to different geometric configurations of the candidate distribution.

2. **Closed-Form Equilibrium Thresholds**

    - **Function**: Derives analytical expressions for $s_1^*$ and $s_2^*$ (Theorems 4.1, 4.2).
    - $s_1^*$ is relatively straightforward: $\nu_1(1 - s_1^*) + \nu_2 \max(1 - s_1^*/\beta, 0) = c_1$.
    - $s_2^*$ requires distinguishing 3 regimes in $\gamma$, each with a distinct closed-form expression.
    - $s_2^*$ exhibits unimodal behavior with respect to $\gamma$ (Theorem 8.1)—increasing correlation both improves evaluation alignment and intensifies competition with institution 1.

3. **Representation Ratio $\mathcal{R}(\beta, \gamma)$ and Normalized Variant $\mathcal{N}(\beta, \gamma)$**

    - **Function**: Defines and derives closed-form expressions for the ratio of admission rates between disadvantaged and advantaged groups.
    - $\mathcal{R}(\beta, \gamma)$ = number of admitted candidates from $G_2$ / number from $G_1$.
    - $\mathcal{N}(\beta, \gamma) = \mathcal{R}(\beta, \gamma) / \mathcal{R}(\beta, 1)$: a normalized variant that isolates the independent effect of evaluation misalignment.
    - **Key result**: $\mathcal{R}$ is monotonically increasing in both $\beta$ and $\gamma$, but the effect of $\gamma$ is nonlinear—a slight loss of correlation can lead to a sharp drop in representation.

4. **Pareto Frontier for Fairness Interventions**

    - **Function**: Given current parameters $(\beta_0, \gamma_0)$ and a fairness target $\tau$, identifies the minimum-cost intervention.
    - **Mechanism**: Plots the contour $\mathcal{N}(\beta, \gamma) \geq \tau$ in the $(\beta, \gamma)$ space, and determines the shortest path from $(\beta_0, \gamma_0)$ to the target region.
    - **Practical Relevance**: System designers can determine whether to prioritize bias reduction or improvement of evaluation alignment.

### Loss & Training
This paper is a purely theoretical study with no training procedure. The core equations are equilibrium conditions in the large-market limit (Equations 1–2); thresholds $s_1^*, s_2^*$ are solved analytically and used to compute all fairness metrics.

## Key Experimental Results

### Main Results (Visualization of Theoretical Findings)

| $(\beta, \gamma)$ | Behavior of $\mathcal{R}(\beta, \gamma)$ | Notes |
|-------------------|------------------------------------------|-------|
| $\gamma = 1$ (fully correlated) | $\mathcal{R} \propto \beta$ (linear) | Reproduces the result of Kleinberg & Raghavan |
| $\gamma$ slightly reduced | $\mathcal{R}$ drops sharply | Nonlinear effect—slight misalignment causes large representation loss |
| $\gamma = 0$ (fully independent) | $\mathcal{R}$ is minimized | Disadvantaged group suffers most when evaluations are completely different |

### Verification of Unimodality of $s_2^*$

| $\gamma$ range | Behavior of $s_2^*$ | Explanation |
|----------------|---------------------|-------------|
| $\gamma < \gamma_{\text{peak}}$ | $s_2^*$ increases with $\gamma$ | Improved alignment allows institution 2 to recruit stronger candidates |
| $\gamma > \gamma_{\text{peak}}$ | $s_2^*$ decreases with $\gamma$ | Intensified competition with institution 1 narrows institution 2's candidate pool |

### Key Findings
- **Nonlinear representation degradation**: Unlike the linear degradation in the fully correlated case, the representation ratio exhibits nonlinear decline under partial correlation.
- **Critical $\gamma$ thresholds**: Structural transitions in selection behavior occur at discrete values of $\gamma$.
- **Intervention priority**: In the low-$\gamma$ regime, improving evaluation alignment is more effective than reducing bias; in the high-$\gamma$ regime, the reverse holds.

## Highlights & Insights
- **First joint analysis of bias and correlation**: Prior work either considered bias alone ($\gamma = 1$) or correlation alone ($\beta = 1$); this paper provides a complete picture of their joint effects.
- **Regime reduction as an analytical technique**: Compressing 16 cases into 3 regimes is a key mathematical contribution that makes the piecewise closed-form solution tractable.
- **Elegant design of the normalized metric $\mathcal{N}(\beta, \gamma)$**: By comparing against the fully aligned case, the independent effect of evaluation misalignment is isolated, enabling meaningful cross-regime fairness comparisons.
- **Direct policy implications**: The Pareto frontier provides a quantitative basis for fairness-aware design of decentralized selection systems.

## Limitations & Future Work
- **Only two institutions**: Real-world scenarios (e.g., national college admissions) involve hundreds or thousands of institutions.
- **The $p = 1$ assumption (all candidates prefer institution 1)**: Although the appendix discusses general preference distributions, the main results are restricted to this simplified setting.
- **Multiplicative bias model**: Bias is modeled as a multiplicative factor $\beta$ applied to scores; real-world bias may be more complex (e.g., additive or nonlinear).
- **Uniform distribution assumption**: Attributes $v_{i1}, v_{i2}$ are assumed to be uniformly distributed; real-world distributions may exhibit more complex tail behavior.
- **Static analysis**: Dynamic effects are not considered, such as the evolution of bias over time or strategic behavior on the part of candidates.

## Related Work & Insights
- **vs. Kleinberg & Raghavan (2018)**: They analyze a centralized model ($\gamma = 1$) and show $\mathcal{R} \propto \beta$ (linear); this paper generalizes to $\gamma \in [0,1]$ and identifies nonlinear effects.
- **vs. Celis & Vishnoi (2020)**: They study the effect of correlation on statistical discrimination (without explicit bias); this paper jointly models explicit bias and correlation.
- **vs. Ashlagi et al. (2019)**: They examine welfare effects of shared vs. independent tie-breaking (corresponding to $\gamma = 1/0$); this paper provides a continuous parameterization over $\gamma$.

## Rating
- Novelty: ⭐⭐⭐⭐ The joint analysis of bias and correlation is a natural yet previously unaddressed problem.
- Experimental Thoroughness: ⭐⭐⭐ A purely theoretical work with numerical visualizations but no validation on real data.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorems are elegantly stated, and the presentation of the regime reduction is clear and intuitive.
- Value: ⭐⭐⭐⭐ Has direct theoretical and practical implications for algorithmic fairness and matching mechanism design.

## Additional Notes
- The theoretical framework and technical tools developed in this paper offer insights for adjacent research areas.
- The core contribution lies in a deepened theoretical understanding that provides a foundation for subsequent practical optimization.
- The paper is methodologically complementary to other NeurIPS 2025 papers published concurrently.
- The paper's treatment of problem motivation and technical approach is worth studying as a model of exposition.
- Readers are encouraged to consult the appendix for more complete experimental details and proofs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[NeurIPS 2025\] CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D](ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Cost Efficient Fairness Audit Under Partial Feedback](cost_efficient_fairness_audit_under_partial_feedback.md)
- [\[ICLR 2026\] Robust Optimization for Mitigating Reward Hacking with Correlated Proxies](../../ICLR2026/ai_safety/robust_optimization_for_mitigating_reward_hacking_with_correlated_proxies.md)

</div>

<!-- RELATED:END -->
