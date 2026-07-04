---
title: >-
  [Paper Note] Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption
description: >-
  [ICML2025][Causal Inference][Synthetic Data] This position paper argues that synthetic experiments are **indispensable** for the rigorous evaluation of causal machine learning (Causal ML) methods. However, current synthetic experimental designs suffer from bias and insufficient complexity. Adhering to a set of principles is required to improve experimental quality, thereby facilitating the broader adoption of Causal ML.
tags:
  - "ICML2025"
  - "Causal Inference"
  - "Synthetic Data"
  - "Experimental Evaluation"
  - "Benchmark Bias"
  - "Position Paper"
date: 2026-05-08
content_hash: 1a4edb0bc4af93b7
---

# Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption

**Conference**: ICML2025  
**arXiv**: [2508.08883](https://arxiv.org/abs/2508.08883)  
**Code**: [GitHub](https://github.com/panispani/causalml-needs-synth-eval)  
**Area**: Causal Inference  
**Keywords**: Causal Inference, Synthetic Data, Experimental Evaluation, Benchmark Bias, Position Paper

## TL;DR

This position paper argues that synthetic experiments are **indispensable** for the rigorous evaluation of causal machine learning (Causal ML) methods. However, current synthetic experimental designs suffer from bias and insufficient complexity. Adhering to a set of principles is required to improve experimental quality, thereby facilitating the broader adoption of Causal ML.

## Background & Motivation

### Key Challenge

Causal machine learning (Causal ML) aims to answer causal questions using ML algorithms and holds great potential in decision-making fields, but has not yet been widely adopted by the mainstream ML community. This is due to:

**The Fundamental Problem of Causal Inference**: For the same individual, one cannot observe both post-treatment and pre-treatment outcomes (counterfactuals are unobservable). This means that real-world data can almost never provide the ground truth for causal queries.

**Defects in Evaluation Practices**: Existing empirical evaluations fail to effectively demonstrate the reliability and robustness of causal methods.

**Community Bias**: The ML community is accustomed to assumption-free predictive methods and remains skeptical of causal methods that require strong assumptions.

### Author Position

In response to the community's criticism that Causal ML relies too heavily on synthetic experiments, the authors **counter-argue** that synthetic data itself is not the problem; **the issue lies in how synthetic experiments are designed**. Synthetic experiments are the only reliable way to precisely evaluate causal methods.

## Method

Instead of proposing a new algorithm, this paper systematically diagnoses the issues in current evaluation practices and proposes principles for improvement.

### Problem 1: Scarcity of Ground Truth Data

- Predictive ML can directly observe labels, whereas Causal ML cannot observe counterfactual outcomes.
- Sources of real causal datasets are limited: expert knowledge (expensive and subjective) and randomized controlled trials (RCTs) (costly and ethically restricted).
- A few studies providing observational + experimental data, such as LaLonde (1986) and Shadish et al. (2008), are repeatedly used, which is insufficient to support general conclusions.
- **Real-world datasets for counterfactual queries (PCH Layer 3) simply do not exist.**

### Problem 2: (Unconscious) Bias in Synthetic/Semi-Synthetic Data

Sources of bias:

1. **Researcher Design Bias**: Experiments are typically designed by the authors of the proposed method, tending to showcase the advantages of their own method.
2. **Modeling Limitation Bias**: Synthetic data can only contain features that researchers know how to model, leaving out "unknown unknowns."

Semi-synthetic data is also affected:

- Datasets for causal discovery fit CGMs from real-world data, inheriting bias from modeling assumptions.
- In semi-synthetic datasets for CATE estimation, if the data and assumptions do not satisfy the identifiability conditions, methods will still converge to an estimate—as if a unique solution exists.
- When generating artificial observational data from RCTs, the choice of sampling strategy directly introduces bias.

### Problem 3: Insufficient Complexity of Synthetic Experiments

- Over-simplified causal models: Much work still uses additive noise models (ANMs), or even restricts variables to quadratic or generalized linear mechanisms.
- Lack of sources of randomness: Simulation parameters (e.g., causal graphs, confounding levels) are typically fixed.
- Neglected robustness analysis: Methods are evaluated only on data that satisfy their own assumptions.

### Four Principles

**Principle 1: Synthetic data is necessary for drawing rigorous conclusions.**

Synthetic data is the only reliable source of ground truth for causal queries and allows complete control over the data-generating process, enabling systematic variation of noise, confounding, and other parameters. The authors emphasize that this is a **necessary** but not sufficient condition.

**Principle 2: Synthetic design choices must be explicitly stated to mitigate unconscious bias.**

Any experiment should specify at least five elements:

| Element | Description |
|------|------|
| (i) Set of causal models | Conditional expressions of the studied SCMs/CGMs |
| (ii) Set of causal queries | PCH layers, variables, and values of the queries |
| (iii) Set of training data | Dimensions, PCH layers, potential perturbations (measurement errors, selection bias, etc.) |
| (iv) Generating algorithm | Algorithm generating synthetic causal models, queries, and datasets |
| (v) Induced distribution | Distribution implied by the generating algorithm over the synthetic sample space |

**Principle 3: Go beyond aggregate accuracy within the identifiable domain to conduct comprehensive experiments**

- Evaluate not only within the method's identifiable domain but also test in scenarios where assumptions are violated.
- Evaluation dimensions should go beyond accuracy to include robustness, scalability, stability, and interpretability.
- Focus on insights rather than aggregate performance metrics.

**Principle 4 (Implicit): Methodological and systematic evaluation**

Systematically explore the performance of methods under different conditions by defining large synthetic experimental spaces.

## Key Experimental Results

### Experiment 1: Bias in the RealCause Semi-Synthetic Benchmark (Validates Problem 2)

**Setup**: RealCause (Neal et al., 2020) was used to generate semi-synthetic data on the IHDP dataset to evaluate the stability of ATE estimation error.

**Key Findings**:

| Experimental Condition | ATE Error | Description |
|----------|----------|------|
| Single replication of original paper | 0.17 | True $\text{ATE} = 4.02$ |
| Fixed realization, variable seed (20 seeds) | $0.38 \pm 0.39$ | Max 1.77 |
| Fixed seed, variable realization (100 realizations) | $0.95 \pm 1.36$ | Max 9.45 |
| Aggregation of 100 realizations $\times$ 20 seeds | Extreme case: $6.209 \pm 11.318$ | True $\text{ATE} = -0.604$ |

**Conclusion**: The RealCause benchmark exhibits high error and extreme variance, and method rankings are unstable across different seeds/realizations. **Benchmark results relying on a single seed run are fragile and potentially misleading.**

### Experiment 2: Performance of CausalNF under Assumption Violation (Validates Problem 3)

**Setup**: Causal Normalizing Flows (CausalNF) is used as an example to test its behavior when the diffeomorphism assumption is violated.

**Scenario A: Assumptions violated but performance is unaffected**
- Apply non-diffeomorphic transformations (piecewise linear functions, sine functions) to the noise of the TriangleNLIN SCM.
- Result: Counterfactual prediction RMSE did not increase significantly.
- Implications: Method-friendly synthetic designs (such as minimizing noise-parent interactions) can mask the impact of assumption violations.

**Scenario B: Assumptions violated leading to performance deterioration**
- Test on unidentifiable counterfactual examples: Two different SCMs share the same observational distribution but have different counterfactual distributions.
- Result: CausalNF always converges to one of the two structures, yielding a large error when the other is true.
- Implications: The method possesses a **systemic bias** that standard experiments alone would fail to detect.

## Highlights & Insights

1. **Counter-intuitive Position**: Instead of avoiding the issues associated with synthetic data, the paper argues that "making good use of synthetic data" is a necessary pathway to advance Causal ML.
2. **Highly Convincing RealCause Experiments**: Revealing the instability of a widely used benchmark through simple variations of seeds and realizations.
3. **Actionable Five-Element Framework**: Providing a clear checklist for the experimental sections of future Causal ML papers.
4. **CausalNF's Dual-Scenario Experiments**: Cleverly demonstrating that "assumption violations are not always catastrophic" yet "sometimes fatal"—underlining the critical importance of systematic robustness testing.
5. **Bridging** the identifiability theory of causal inference with the practical issues of ML experimental design.

## Limitations & Future Work

1. **Inherent Limitation of a Position Paper**: Proposing principles but falling short of providing complete automated tools or standardized frameworks to implement them.
2. **Narrow Experimental Coverage**: Focusing only on RealCause and CausalNF, while ignoring other mainstream methods (e.g., DoWhy, EconML).
3. **Questionable Operationalizability of Principles**: The "induced distribution of the generating algorithm over the synthetic space" in Principle 2 is difficult to describe precisely in practice.
4. **Lack of Discussion on Computational Cost**: Generating and evaluating large-scale synthetic experimental spaces requires substantial computing power.
5. **No "Good Enough" Standard Provided**: It remains unclear how many experiments, how many seeds, or how large a synthetic space is required to satisfy the principles.
6. **Lack of Domain-Specific Adaptation**: Causal inference characteristics vary widely across fields like medicine and economics, and whether these principles are universally applicable remains undiscussed.

## Related Work & Insights

- **Curth et al. (2021)**: First to demonstrate that synthetic outcome experiments can flip method rankings due to minor design choices.
- **Gentzel et al. (2019)**: Introduced the "unknown unknowns" bias of synthetic data.
- **Herrmann et al. (2024) / Karl et al. (2024)**: Reflections on current empirical practices in predictive ML.
- **Pearl & Mackenzie (2018)**: Pearl's Causal Hierarchy (PCH) three-layer framework—association, intervention, and counterfactuals.
- **Javaloy et al. (2023)**: CausalNF, a SOTA method using normalizing flows for counterfactual estimation.
- **Neal et al. (2020)**: RealCause, a widely used method for generating semi-synthetic causal benchmarks.

## Rating

- Novelty: ⭐⭐⭐ — As a position paper, it introduces no new methods, but the stance is clear and well-argued.
- Experimental Thoroughness: ⭐⭐⭐ — The two sets of experiments illustrate the points precisely, but the coverage is somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐⭐ — Very well-structured, with progressively deeper arguments, and a clear one-to-one mapping between the identified problems and the proposed principles.
- Value: ⭐⭐⭐⭐ — Provides an important reference for experimental guidelines within the Causal ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Designing Time Series Experiments in A/B Testing with Transformer Reinforcement Learning](../../ICLR2026/causal_inference/designing_time_series_experiments_in_ab_testing_with_transformer_reinforcement_l.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ICML 2026\] The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](../../ICML2026/causal_inference/the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)
- [\[ICML 2025\] RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals](rate_causal_explainability_of_reward_models_with_imperfect_counterfactuals.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)

</div>

<!-- RELATED:END -->
