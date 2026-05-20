---
title: >-
  [Paper Note] Alternative Fairness and Accuracy Optimization in Criminal Justice
description: >-
  [AAAI 2026][AI Safety][algorithmic fairness] This paper provides a systematic review of three dimensions of algorithmic fairness (group fairness, individual fairness, and procedural fairness)…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "algorithmic fairness"
  - "criminal justice"
  - "risk assessment"
  - "group fairness"
  - "individual fairness"
date: 2026-05-08
content_hash: e07648a5843eaf74
---

# Alternative Fairness and Accuracy Optimization in Criminal Justice

**Conference**: AAAI 2026
**arXiv**: [2511.04505v4](https://arxiv.org/abs/2511.04505v4)  
**Code**: None  
**Area**: AI Safety / Algorithmic Fairness
**Keywords**: algorithmic fairness, criminal justice, risk assessment, group fairness, individual fairness

## TL;DR

This paper provides a systematic review of three dimensions of algorithmic fairness (group fairness, individual fairness, and procedural fairness), proposes an improved group fairness optimization formulation based on tolerance constraints, and constructs a "Three Pillars of Fairness" deployment framework for public decision-making systems.

## Background & Motivation

As algorithms become increasingly prevalent in modern society—particularly in criminal justice—concerns over the fairness of algorithmic decision-making have intensified. Tools such as COMPAS have been found to exhibit systematic bias against minority groups, attracting widespread attention. However, no consensus definition of "fairness" has been reached: group fairness, individual fairness, and procedural fairness are inherently in conflict, and satisfying one notion may undermine another. Existing strict equality constraints (requiring exactly equal false negative rates across groups) frequently render optimization problems infeasible or cause substantial accuracy degradation in practice. There is therefore an urgent need for approaches that can guarantee fairness without excessively sacrificing predictive accuracy, as well as a holistic framework to guide real-world deployment.

## Core Problem

1. **Conflicting fairness definitions**: Group fairness (demographic parity, equalized odds, equal opportunity, calibration), individual fairness (Lipschitz condition), and procedural fairness exhibit irreconcilable tensions and cannot be perfectly satisfied simultaneously.
2. **Infeasibility of strict equality constraints**: Traditional methods require error rates to be exactly equal across protected groups, but due to differences in base rates, strict equality is often infeasible or results in significant accuracy loss.
3. **Lack of practical deployment guidance**: Technical fairness optimization lacks a complete framework that integrates legitimacy, transparency, and accountability.

## Method

### Overall Architecture

The paper makes two core contributions: (1) an improved mathematical optimization formulation for group fairness; and (2) a "Three Pillars of Fairness" framework for deploying public decision-making systems.

### Key Designs

#### Improved Group Fairness Optimization Formulation

Traditional PAC-style formulations require strict equality of false negative rates across groups:

$$FN(h, v_i) = FN(h, v_j), \quad \forall i \neq j$$

This paper relaxes the constraint to a **tolerance-based** form:

$$\min \sum_{i=1}^{n} \alpha \cdot FN(h, v_i) \cdot W_i$$

$$\text{s.t.} \quad |FN(h, v_i) - FN(h, v_j)| \leq \tau, \quad \forall i \neq j$$

where:
- $\alpha$ and $\beta$ are cost weights for false negatives and false positives, respectively
- $W_i$ is the population proportion of group $v_i$
- $\tau$ is the tolerance parameter controlling the upper bound on disparities in false negative rates across groups

**Two key advantages**:
1. By tuning $\tau$, feasibility of the optimization problem is always guaranteed
2. Relaxing the equality constraint allows overall predictive accuracy to improve

**Core trade-off**: The choice of $\tau$ involves an ethical decision—an excessively large $\tau$ may further disadvantage vulnerable groups, while deliberately favoring such groups constitutes race-based affirmative action in effect, which may invite legal challenges.

#### Three Pillars of Fairness Framework

1. **Need-based Decisions**: The definition of fairness should be context-dependent—if historical data are biased, different thresholds may be set for different groups to achieve group equity; if data are fair, a uniform threshold should be applied to ensure individual fairness.
2. **Transparency and Accountability**: Decision processes should be open and transparent, clearly communicating which fairness definition has been adopted and what trade-offs have been made, so that the public can understand and oversee the system.
3. **Narrowly Tailored Definitions and Solutions**: Fairness definitions and solutions should be tailored to specific problems rather than applied universally. This both improves technical feasibility and aligns with the "narrow tailoring" requirement under U.S. anti-discrimination law.

### Loss & Training

This paper is not a conventional model-training work; its "loss function" is embedded in the design of the optimization framework:

- Objective: minimize the weighted sum of false negative rates $\sum_i \alpha \cdot FN(h, v_i) \cdot W_i$
- Constraint: disparities in false negative rates across groups must not exceed $\tau$
- The $\alpha/\beta$ ratio reflects error cost preferences in different contexts: $\alpha/\beta \gg 10$ in medical settings (where false negatives are extremely costly) and $\alpha/\beta \ll 0.1$ in credit settings (where false positives are more costly)

The paper also discusses several technical approaches to achieving group fairness:
- **Pre-processing**: suppression, label massaging, instance reweighting, disparate impact remover (DI remover)
- **In-processing**: adding regularization terms to the loss function to penalize inter-group prediction disparities
- **Post-processing**: equalized odds post-processing (flipping a subset of output labels), reject option classification (ROC)

## Key Experimental Results

This is a **theoretical and framework-oriented paper** and does not include conventional experimental comparisons. However, the following key empirical findings are cited:

| Data / Setting | Key Finding | Source |
|---|---|---|
| COMPAS recidivism prediction | Uniform threshold → individual fairness but group unfairness; differentiated thresholds → group fairness but undermines individual fairness and accuracy | Lagioia et al. |
| UK Crime API data | Even with fully randomized synthetic historical data, predictive policing algorithms generate biased feedback loops | Chapman et al. |
| Adult income dataset | Larger Wasserstein distance between two groups leads to greater degradation of individual fairness after applying DI remover; mean differences have a larger impact than variance differences | Žliobaitė et al. |

### Ablation Study

The paper provides a theoretical analysis of the tolerance parameter $\tau$:
- $\tau = 0$: degenerates to the traditional strict equality constraint, which may be infeasible
- $\tau > 0$: guarantees the existence of a feasible solution; larger $\tau$ yields higher accuracy but weaker fairness guarantees
- Practical suggestion: e.g., constraining the disparity in recidivism prediction rates across racial groups to no more than 5%

## Highlights & Insights

1. **Mathematically concise yet insightful**: Relaxing strict equality constraints to tolerance constraints is simple in form yet effectively resolves issues of feasibility and accuracy
2. **Interdisciplinary perspective**: Concepts from political science such as "legitimacy" and procedural justice are introduced into the algorithmic fairness discourse
3. **Practical utility of the Three Pillars framework**: Provides actionable deployment guidance for government agencies and public sector organizations, rather than remaining purely technical
4. **Systematic response to three major critiques**: Data bias, implicit affirmative action, and the subgroup explosion problem are each addressed with clear analysis
5. **Making ethical choices explicit**: The settings of $\alpha/\beta$ and $\tau$ compel designers to articulate their error cost preferences explicitly, thereby enhancing transparency

## Limitations & Future Work

1. **Lack of empirical validation**: The proposed optimization formulation is not implemented or tested on real datasets, remaining at a theoretical level
2. **No systematic guidance for selecting $\tau$**: Although tolerance constraints are proposed, no systematic method is provided for determining $\tau$ based on specific contexts
3. **Focus primarily on binary classification**: Most methods and examples assume binary protected attributes (e.g., Black/White); multi-category settings are insufficiently addressed
4. **Optimization formulation only considers false negative rates**: The objective function minimizes only the weighted sum of false negative rates, without incorporating false positive rates into the optimization
5. **Three Pillars framework is relatively high-level**: Specific quantitative metrics and evaluation procedures are lacking, requiring substantial customization for deployment
6. **Causal fairness not discussed**: Despite rapid recent development in fairness research from a causal inference perspective, this paper does not engage with that literature

## Related Work & Insights

- **Dwork et al. (2012)**: Proposed the Lipschitz condition for individual fairness and the "fairness through awareness" algorithm; this paper builds on that work to analyze conditions under which individual and group fairness conflict
- **Hardt et al. (2016)**: Proposed equalized odds post-processing via output label flipping; this paper incorporates that method as one of its post-processing strategies
- **Kearns et al. (2018)**: Proposed a polynomial-time algorithm for subgroup fairness; this paper treats the subgroup explosion as the third major critique of group fairness
- **Ho & Xiang (2020)**: Argued from a legal perspective that narrowly tailored algorithmic fairness adjustments are most legitimate; this paper develops that idea into the "narrowly tailored definitions and solutions" pillar
- **COMPAS system analysis (Lagioia et al.)**: Revealed the incompatibility between calibration and group fairness, motivating the tolerance constraint formulation in this paper

### Further Connections

1. **Generality of tolerance constraints**: The approach of relaxing strict equality constraints to tolerance intervals can be extended to other fairness definitions (e.g., equalized odds, calibration) and has broader applicability in ML fairness research
2. **Context-dependence of error costs**: The varying settings of $\alpha/\beta$ across domains (medical vs. credit vs. criminal justice) suggest that any ML system should explicitly account for the differential costs of different error types
3. **Complementarity of procedural and technical fairness**: Even a technically perfect algorithm may fail in deployment if it lacks transparency and public trust—a lesson relevant to all AI safety research
4. **Potential connection to adversarial robustness**: Biased feedback loops share structural similarities with adversarial attacks, and defensive methods may transfer across these domains

## Rating

- **Novelty**: ⭐⭐⭐ The tolerance constraint formulation is a simple yet effective improvement over existing frameworks; the Three Pillars framework integrates multidisciplinary perspectives, though individual components are not entirely novel
- **Experimental Thoroughness**: ⭐⭐ As a theoretical/framework paper, it contains no original experiments, relying solely on cited empirical results, which limits persuasiveness
- **Writing Quality**: ⭐⭐⭐⭐ The survey portion is systematic and comprehensive, mathematical notation is rigorous, the argumentation is logically clear, and interdisciplinary references are rich
- **Value**: ⭐⭐⭐ The Three Pillars framework provides practical deployment guidance for public sector algorithmic systems, and the tolerance constraint idea is concise and useful, though the absence of empirical validation weakens the overall contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](../../NeurIPS2025/ai_safety/fairness-regularized_online_optimization_with_switching_costs.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)

</div>

<!-- RELATED:END -->
