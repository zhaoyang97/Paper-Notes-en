---
title: >-
  [Paper Note] Alternative Fairness and Accuracy Optimization in Criminal Justice
description: >-
  [AAAI 2026][AI Safety][Algorithmic Fairness] This paper systematically reviews the three major dimensions of algorithmic fairness (group fairness, individual fairness, and procedural fairness), proposes an improved group fairness optimization formulation based on tolerance constraints, and constructs a "Three Pillars of Fairness" deployment framework tailored for public decision-making systems.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Algorithmic Fairness"
  - "Criminal Justice"
  - "Risk Assessment"
  - "Group Fairness"
  - "Individual Fairness"
date: 2026-05-08
content_hash: 953ad1062100725a
---

# Alternative Fairness and Accuracy Optimization in Criminal Justice

**Conference**: AAAI 2026  
**arXiv**: [2511.04505v4](https://arxiv.org/abs/2511.04505v4)  
**Code**: None  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: Algorithmic Fairness, Criminal Justice, Risk Assessment, Group Fairness, Individual Fairness  

## TL;DR

This paper systematically reviews the three major dimensions of algorithmic fairness (group fairness, individual fairness, and procedural fairness), proposes an improved group fairness optimization formulation based on tolerance constraints, and constructs a "Three Pillars of Fairness" deployment framework tailored for public decision-making systems.

## Background & Motivation

With the increasingly widespread application of algorithms in modern society, particularly in the field of criminal justice, the issue of fairness in algorithmic decision-making has become more prominent. Risk assessment tools such as COMPAS have been found to exhibit systematic discrimination against ethnic minorities, drawing widespread attention. However, there is still no consensus on the definition of "fairness" itself—inherent conflicts exist among different definitions such as group fairness, individual fairness, and procedural fairness, where satisfying one might compromise another. Existing strict equality constraints (requiring identical false negative rates across groups) often lead to infeasible optimization problems or severe degradation in accuracy in practice. Therefore, there is an urgent need for an approach that balances fairness with prediction accuracy, as well as a comprehensive framework to guide practical deployment.

## Core Problem

1. **Conflict in Fairness Definitions**: There is an irreconcilable tension among group fairness (demographic parity, equalized odds, equal opportunity, calibration), individual fairness (Lipschitz condition), and procedural fairness, making it impossible to satisfy all perfectly at the same time.
2. **Infeasibility of Strict Equality Constraints**: Traditional methods require error rates to be perfectly equal across protected groups. However, due to differences in base rates across groups, strict equality is often infeasible or causes significant degradation in accuracy.
3. **Lack of Guidance for Practical Deployment**: Algorithmic fairness optimization at the technical level lacks a comprehensive framework that integrates with legitimacy, transparency, and accountability.

## Method

### Overall Architecture

This paper makes two core contributions: (1) an improved mathematical optimization formulation for group fairness; (2) a "Three Pillars of Fairness" framework designed for the deployment of public decision-making systems.

### Key Designs

#### Improved Group Fairness Optimization Formulation

Traditional PAC settings require the false negative rates of different groups to be strictly equal:

$$FN(h, v_i) = FN(h, v_j), \quad \forall i \neq j$$

This work relaxes this formulation into a **tolerance constraint** form:

$$\min \sum_{i=1}^{n} \alpha \cdot FN(h, v_i) \cdot W_i$$

$$\text{s.t.} \quad |FN(h, v_i) - FN(h, v_j)| \leq \tau, \quad \forall i \neq j$$

Where:
- $\alpha$ and $\beta$ represent the error cost weights for false negatives and false positives, respectively.
- $W_i$ is the proportion of group $v_i$ in the overall population.
- $\tau$ is the tolerance parameter, controlling the upper bound of the difference in false negative rates between groups.

**Two Major Advantages**:
1. By adjusting $\tau$, it ensures that a feasible solution to the optimization problem always exists.
2. By relaxing the equality constraints, the overall prediction accuracy can be improved.

**Key Trade-off**: Setting $\tau$ involves ethical choices—an excessively large $\tau$ might further disadvantage marginalized groups, whereas intentionally favoring marginalized groups essentially constitutes race-based "affirmative action," which may trigger legal controversies.

#### Three Pillars of Fairness Framework

1. **Need-based Decisions**: The definition of fairness should vary across scenarios. If historical data is biased, different thresholds can be set for different groups to achieve group fairness; if the data is fair, a uniform threshold should be set to protect individual fairness.
2. **Transparency and Accountability**: The decision-making process should be open and transparent, clearly conveying which definition of fairness was chosen and what trade-offs were made, facilitating public understanding and oversight.
3. **Narrowly Tailored Definitions and Solutions**: Fairness definitions and solutions should be tailored to specific problems, avoiding a "one-size-fits-all" approach. This not only improves technical feasibility but also aligns with the "narrowly tailored" requirement in U.S. anti-discrimination law.

### Loss & Training

This study is not a traditional model-training paper, and its "loss function" is reflected in the design of the optimization framework:

- Objective function: Minimize the weighted sum of false negative rates, $\sum_i \alpha \cdot FN(h, v_i) \cdot W_i$.
- Constraint: The difference in false negative rates between any two groups must not exceed $\tau$.
- The ratio $\alpha/\beta$ reflects error cost preferences in different scenarios: in healthcare, $\alpha/\beta \gg 10$ (extremely high cost for false negatives); in credit scoring, $\alpha/\beta \ll 0.1$ (higher cost for false positives).

Additionally, the paper discusses multiple technical pathways to achieve group fairness:
- **Pre-processing**: Data suppression, label massaging, sample reweighting, and Disparate Impact (DI) remover.
- **In-processing**: Adding regularization terms to the loss function to penalize differences in prediction across groups.
- **Post-processing**: Equalized odds post-processing (flipping a portion of output labels) and Reject Option Classification (ROC).

## Key Experimental Results

This is a **theory- and framework-oriented paper** and does not contain traditional experimental comparisons. However, the paper cites the following key empirical findings:

| Data / Scenario | Key Findings | Source |
|-----------|---------|------|
| COMPAS Recidivism Prediction | Uniform threshold $\rightarrow$ individual fairness but group unfairness; differentiated thresholds $\rightarrow$ group fairness but degrades individual fairness and accuracy | Lagioia et al. |
| UK Crime API Data | Even when using completely randomized synthetic historical data, predictive policing algorithms still produce biased feedback loops | Chapman et al. |
| Adult Income Dataset | The larger the Wasserstein distance between target groups, the more individual fairness degrades after applying the DI remover; differences in mean have a greater impact than differences in variance | Žliobaitė et al. |

### Ablation Study

The paper conducts a theoretical analysis of the tolerance parameter $\tau$:
- $\tau = 0$: Degenerates into traditional strict equality constraints, which might be infeasible.
- $\tau > 0$: Guarantees the existence of a feasible solution; a larger $\tau$ yields higher accuracy but weaker fairness guarantees.
- Practical suggestion: For example, setting the difference in recidivism prediction rates among different racial groups to not exceed 5%.

## Highlights & Insights

1. **Mathematical Simplicity and Insightfulness**: Relaxing the strict equality constraints into tolerance constraints is simple in form yet effectively solves feasibility and accuracy issues.
2. **Interdisciplinary Perspective**: Concepts such as "legitimacy" and procedural justice from political science are introduced into the algorithmic fairness discussion.
3. **Practical Utility of the Three Pillars Framework**: Actionable deployment guidance is provided for government agencies and the public sector, rather than remaining purely at the technical level.
4. **Systematic Response to Three Major Criticisms**: Clear analyses are provided for each of the main criticisms: data bias, implicit affirmative action, and sub-group explosion.
5. **Making Ethical Choices Explicit**: The configuration of the $\alpha/\beta$ ratio and $\tau$ forces designers to explicitly articulate their error cost preferences, enhancing transparency.

## Limitations & Future Work

1. **Lack of Experimental Validation**: The optimization formulation proposed in this paper is not implemented or tested on real datasets, remaining at a theoretical level.
2. **Lack of Guidance on Selecting $\tau$**: Although tolerance constraints are proposed, no systematic methodology is provided for determining how $\tau$ should be specified based on concrete application scenarios.
3. **Focus Primarily on Binary Classification**: Most discussed methods and examples are based on binary protected attributes (e.g., Black/White), with insufficient discussion on multi-class scenarios.
4. **Optimization Formula Considers Only False Negative Rates**: The objective function only minimizes the weighted sum of false negative rates, leaving the false positive rate unintegrated into the optimization objectives.
5. **Macro-scale Nature of the Three Pillars Framework**: It lacks concrete quantitative metrics and evaluation workflows, still requiring substantial customization work during actual deployment.
6. **No Discussion on the Relation to Causal Fairness**: Research on fairness from the perspective of causal inference has evolved rapidly in recent years, but it is not addressed in this paper.

## Related Work & Insights

- **Dwork et al. (2012)**: Proposed the Lipschitz condition for individual fairness and "fair affirmative action" algorithms; this work analyzes the conflict conditions between individual and group fairness based on these concepts.
- **Hardt et al. (2016)**: Proposed the equalized odds post-processing method (flipping output labels); this work incorporates it into the discussion as one of the post-processing strategies.
- **Kearns et al. (2018)**: Proposed a polynomial-time algorithm for subgroup fairness; this work covers subgroup explosion as the third major criticism of group fairness.
- **Ho & Xiang (2020)**: Argued from a legal perspective that narrowly tailored algorithmic fairness adjustments have the highest legitimacy; this work develops this idea into the "Narrowly Tailored Definitions and Solutions" of the three pillars.
- **COMPAS System Analysis (Lagioia et al.)**: Revealed the incompatibility between calibration and group fairness, which serves as the motivation for this paper to propose tolerance constraints.

## Insights & Connections

1. **Generality of the Tolerance Constraint Concept**: Relaxing strict equality constraints into tolerance intervals can be generalized to other fairness definitions (e.g., equalized odds, calibration), offering potential applications in broader ML fairness problems.
2. **Scenario Dependency of Error Costs**: Different configurations of $\alpha/\beta$ (e.g., healthcare vs. credit scoring vs. criminal justice) inspire the explicit consideration of cost differences among different types of errors in any ML system.
3. **Complementary Nature of Procedural and Technical Fairness**: Even a technically perfectly fair algorithm may fail in deployment without transparency and public trust, offering valuable lessons for all AI safety research.
4. **Intersections with Adversarial Robustness Research**: Biased feedback loops share similar structures with adversarial attacks, indicating that defense methods might be transferable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The tolerance constraint formulation is a simple yet effective improvement over existing frameworks. The three-pillar framework integrates multidisciplinary perspectives, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐ As a theoretical/conceptual framework paper, it lacks its own experimental validation and relying solely on existing empirical results limits its persuasiveness.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The review section is systematic and comprehensive, mathematical notations are standardized, the arguments are logically clear, and interdisciplinary citations are rich.
- **Value**: ⭐⭐⭐⭐ The three-pillar framework holds practical guiding value for public sector algorithmic deployments. The concept of tolerance constraints is concise and practical, but the lack of experimental validation slightly diminishes the overall contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](../../NeurIPS2025/ai_safety/fairness-regularized_online_optimization_with_switching_costs.md)
- [\[CVPR 2026\] Decoupling Bias, Aligning Distributions: Synergistic Fairness Optimization for Deepfake Detection](../../CVPR2026/ai_safety/decoupling_bias_aligning_distributions_synergistic_fairness_optimization_for_dee.md)
- [\[ICLR 2026\] Wring Out the Bias: A Rotation-Based Alternative to Projection Debiasing](../../ICLR2026/ai_safety/wring_out_the_bias_a_rotation-based_alternative_to_projection_debiasing.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](../../CVPR2026/ai_safety/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)

</div>

<!-- RELATED:END -->
