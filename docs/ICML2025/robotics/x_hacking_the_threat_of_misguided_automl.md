---
title: >-
  [Paper Note] X-Hacking: The Threat of Misguided AutoML
description: >-
  [ICML2025][Robotics][AutoML] Reveals a new security threat in the Explainable AI (XAI) domain termed "X-hacking": by leveraging the pipeline search capabilities of AutoML, adversaries can find explanatory results within the Rashomon set of models that support predetermined conclusions, with Bayesian optimization running approximately 3 times faster than random search.
tags:
  - "ICML2025"
  - "Robotics"
  - "AutoML"
  - "X-hacking"
  - "Explainable AI"
  - "Rashomon set"
  - "Adversarial safety"
date: 2026-05-08
content_hash: 473146fa81968186
---

# X-Hacking: The Threat of Misguided AutoML

**Conference**: ICML2025  
**arXiv**: [2401.08513](https://arxiv.org/abs/2401.08513)  
**Code**: None  
**Area**: Robotics  
**Keywords**: AutoML, X-hacking, Explainable AI, Rashomon set, Adversarial safety

## TL;DR
Reveals a new security threat in the Explainable AI (XAI) domain termed "X-hacking": by leveraging the pipeline search capabilities of AutoML, adversaries can find explanatory results within the Rashomon set of models that support predetermined conclusions, with Bayesian optimization running approximately 3 times faster than random search.

## Background & Motivation

### From P-hacking to X-hacking

P-hacking is a well-known threat in traditional statistics, where researchers try multiple analysis methods to obtain desired statistically significant results. This paper generalizes this concept to the field of Explainable AI (XAI).

### Limitations of Prior Work

**Limitations of Prior Work**: There are numerous degrees of freedom in data science pipelines (feature engineering, model selection, hyperparameter tuning, etc.), and different pipeline configurations can lead to drastically different feature importance explanations. Adversaries can leverage AutoML to systematically search for pipeline configurations that support their "target narratives."

### Key Challenge

**Key Challenge**: The Rashomon set refers to the set of all models that achieve approximately optimal predictive performance on the same dataset. The key observation is that although these models exhibit similar predictive performance, their feature importance rankings can be completely different.

### Threat Scenarios

- Insurance companies: Searching for models that support the narrative that "age does not affect premiums" to evade discrimination charges.
- Loan approval: Choosing explanations that conceal the impact of sensitive features.
- Regulatory evasion: Manipulating feature importance to satisfy compliance requirements.

## Method

### Problem Formulation
Given a dataset $D$, the goal is to find a pipeline configuration $\pi^*$ such that:
1. Predictive performance is within an acceptable range: $\text{perf}(\pi^*) \geq \tau$
2. The feature importance ranking satisfies the adversary's target: $\text{rank}(x_j, \pi^*) \leq k$ (or $\geq k$)

### Strategy 1: Random Search (RS)
Randomly sample from the pipeline configuration space and filter configurations that simultaneously satisfy both the performance constraints and the explanatory goals. Simple but inefficient.

### Strategy 2: Bayesian Optimization (BO)
Formulate X-hacking as a black-box optimization problem:
- Objective function: Explanatory target score (e.g., the importance ranking of the target feature)
- Constraint: Predictive performance is not lower than a threshold
- Model the objective function using Gaussian Processes and guide the search via an acquisition function
- Finds configurations satisfying the conditions approximately 3 times faster than random search

### Pipeline Search Space
Covers all degrees of freedom in end-to-end data science pipelines:
- Feature engineering (normalization, missing value imputation, encoding methods)
- Model selection (RF, XGBoost, SVM, MLP, etc.)
- Hyperparameter settings
- Explanation methods (SHAP, Permutation Importance, etc.)

## Key Experimental Results

### Success Rate of X-hacking


### Main Results

| Dataset | Random Search Success Rate | BO Success Rate | BO Speedup |
|--------|-------------|---------|--------|
| Adult Income | High | Higher | ~3× |
| COMPAS | Medium | High | ~3× |
| German Credit | Medium | High | ~2.5%× |

### Dataset Vulnerability to X-hacking


### Ablation Study

| Feature | Vulnerable | Robust |
|------|---------|--------|
| Feature Information Redundancy | High Redundancy = Vulnerable | Low Redundancy = Robust |
| Rashomon Set Size | Large = Vulnerable | Small = Robust |
| Feature Independence | High Correlation = Vulnerable | Low Correlation = Robust |

### Key Findings
1. BO is 3 times more efficient than random search: adversaries can find target explanations faster.
2. Feature information redundancy is the primary determinant of vulnerability.
3. Even on powerful models like XGBoost, X-hacking can succeed.
4. Both SHAP and Permutation Importance can be manipulated equally.

## Highlights & Insights

1. First systematic study on the threat of X-hacking in end-to-end data science pipelines.
2. Elegantly generalizes the concept of p-hacking to the XAI domain, filling a gap in safety analysis.
3. Reveals that dataset redundancy determines X-hacking vulnerability, pointing towards detection and defense mechanisms.
4. The experimental design is close to real-world scenarios (insurance/loans/regulation), making it highly persuasive.

## Limitations & Future Work

1. Currently mainly focuses on traditional ML pipelines on tabular data; deep learning scenarios remain to be extended.
2. The defense mechanism is only preliminarily discussed, lacking a concrete, deployable detection scheme.
3. Assumes that adversaries have complete control over the pipeline configuration, whereas real-world constraints might be tighter.
4. Analysis of the impact of X-hacking under a causal inference framework is missing.

## Related Work & Insights

- **P-hacking Literature**: This work is a natural extension of traditional statistical biases to ML.
- **Rashomon Set Research**: Work on model multiplicity, such as Marx et al. (2020) and others.
- Insights:
  1. Pipeline auditing tools need to be developed to detect X-hacking behavior.
  2. Regulatory frameworks should require reporting the pipeline search space rather than a single explanation.
  3. "Multi-model consensus explanations" could be considered as a defense strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (5.0/5) — Discovery and systemization of a brand-new safety threat.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4.0/5) — Multi-dataset validation but lacking deep learning scenarios.
- Writing Quality: ⭐⭐⭐⭐☆ (4.0/5)
- Value: ⭐⭐⭐⭐⭐ (5.0/5) — Significant warning implications for AI safety and regulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](action-constrained_imitation_learning.md)
- [\[ICML 2025\] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)
- [\[ICML 2025\] Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts](sketch-plan-generalize_learning_and_planning_with_neuro-symbolic_programmatic_re.md)
- [\[ICML 2025\] Maximum Total Correlation Reinforcement Learning](maximum_total_correlation_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
