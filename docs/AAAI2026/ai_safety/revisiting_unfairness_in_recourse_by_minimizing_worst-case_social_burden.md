---
title: >-
  [Paper Note] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden
description: >-
  [AAAI 2026][AI Safety][algorithmic recourse] This paper systematically analyzes three major limitations of fairness metrics in algorithmic recourse (neglecting classifier decision behavior, ignoring ground-truth labels, and disparity metrics concealing unfairness). It proposes a social burden-based fairness framework, MISOB, which reduces the social burden of all groups through a minimax-weighted training strategy, simultaneously improving fairness during both the prediction…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "algorithmic recourse"
  - "social burden"
  - "fairness"
  - "minimax optimization"
  - "counterfactual explanation"
date: 2026-05-08
content_hash: f65784181a9e0142
---

# Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden

**Conference**: AAAI 2026  
**arXiv**: [2509.04128](https://arxiv.org/abs/2509.04128)  
**Code**: [github](https://github.com/abarrainkua/MISOB)  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: algorithmic recourse, social burden, fairness, minimax optimization, counterfactual explanation

## TL;DR

This paper systematically analyzes three major limitations of fairness metrics in algorithmic recourse (neglecting classifier decision behavior, ignoring ground-truth labels, and disparity metrics concealing unfairness). It proposes a social burden-based fairness framework, MISOB, which reduces the social burden of all groups through a minimax-weighted training strategy, simultaneously improving fairness during both the prediction and recourse stages without requiring access to sensitive attributes.

## Background & Motivation

Automated decision-making systems are widely used in domains such as credit approval and public services. When a model yields a negative decision, it should provide actionable recourse recommendations (e.g., "increase monthly income by $500") to allow individuals to overturn the outcome. However, the recourse process itself may carry unfairness: different groups might receive identical recommendations but face drastically different implementation costs.

Existing research on algorithmic recourse fairness faces three core problems: (1) **Neglecting classifier decision behavior**—equalizing only the average recourse cost of rejected individuals ignores the fact that if a group has a higher rejection rate, it bears a heavier overall recourse burden; (2) **Ignoring ground-truth labels**—failing to distinguish between individuals who "should have been accepted but were incorrectly rejected" and those who "indeed deserved rejection," where forcing the former to change features constitutes unfairness caused by system errors; (3) **Disparity metrics may conceal unfairness**—a zero difference in social burden between two groups does not imply fairness, as both groups might be suffering from excessively high burdens.

Key Insight: This work shifts the focus of fairness from "disparity minimization" to a Rawlsian minimax perspective. By focusing on the social burden of the worst-off group, it introduces MISOB, a lightweight training method that does not require sensitive attributes.

## Method

### Overall Architecture

MISOB is an iterative training framework: (1) pre-training a base classifier $f^{(0)}$; (2) in each iteration, calculating the social burden of each training sample and then retraining the classifier using a burden-weighted loss function. High-burden samples receive higher weights, guiding the classifier to improve decisions on these samples.

### Key Designs

**1. Definition of Social Burden**

Conventional recourse cost focuses on the average cost of all rejected individuals, whereas social burden focuses on individuals who **should have been accepted but were incorrectly rejected**. For a sensitive group $s$:

$$B_{f,g}^s = \underbrace{\mathbb{E}[\delta((X,s), g_f(X))]}_{\text{假阴性的追索成本}} \cdot (1 - \underbrace{P(f(X)=1|S=s, Y=1)}_{\text{真阳性率 TPR}})$$

where $\delta$ is the recourse cost function, and $g_f$ is the recourse algorithm. Social burden simultaneously accounts for: the probability of being incorrectly rejected (related to TPR) and the cost of changing features. In an ideal scenario, the social burden should be zero.

The paper also defines the expected recourse cost $C_{f,g}^s$, formulated using the Acceptance Rate (AR) instead of the TPR:

$$C_{f,g}^s = \mathbb{E}[\delta((X,s), g_f(X))] \cdot (1 - P(f(X)=1|S=s))$$

**2. Burden-Aware Instance Weighting**

The core of MISOB is an instance weighting scheme. The weight of each training sample $x^i$ is defined as:

$$\phi(i, \mathcal{Q}, \alpha) = 1 + \alpha N \frac{b_{f,g_f}^i}{\sum_{j} b_{f,g_f}^j} \mathbb{1}\{\beta > 0\}$$

where $b_{f,g}^i = \delta(x^i, g_f(x^i)) \cdot \mathbb{1}\{y^i = 1\}$ represents the instance-level burden (burden is only incurred by positive class instances), and $\alpha$ is a hyperparameter balancing fairness and accuracy. High-burden samples receive greater weights, forcing the classifier to prioritize improving decisions for these samples.

**3. No Need for Sensitive Attributes**

MISOB completely avoids accessing sensitive attributes during training and inference. The computation of burden is based on ground-truth labels and recourse costs rather than group membership. This implies that: (a) collecting sensitive information is unnecessary, avoiding legal and ethical risks; (b) intersectional group fairness (e.g., "young minority females") is handled naturally, as groups can be defined post hoc during evaluation.

### Loss & Training

Weighted classification loss: $\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \phi(i, \mathcal{Q}, \alpha) \cdot \ell(f(x^i), y^i)$. First, the base classifier $f^{(0)}$ is pre-trained, followed by $T$ iterations of optimization. The recourse cost must be recomputed in each iteration. The overall computational complexity is $O(N^3)$, which can be improved using batching and parallelization.

## Key Experimental Results

### Main Results

Results on the Adult dataset with race as the sensitive attribute (average of 10 random splits):

| Recourse Method | Strategy | Accuracy ↑ | Worst Burden ↓ | Burden Δ ↓ | Worst TPR ↑ | TPR Δ ↓ | Worst Cost ↓ |
|---------|------|---------|-----------|---------|----------|---------|-----------|
| GS | None | 0.81 | 4.56 | 0.03 | 0.27 | 0.08 | 115.69 |
| GS | POSTPRO | 0.80 | 4.96 | 0.61 | 0.37 | 0.00 | 98.40 |
| GS | **MISOB** | **0.82** | **3.01** | **0.85** | **0.52** | **0.11** | **93.06** |
| WT | None | 0.81 | 1.28 | 0.01 | 0.27 | 0.08 | 38.27 |
| WT | POSTPRO | 0.80 | 1.55 | 0.01 | 0.37 | 0.00 | 39.71 |
| WT | **MISOB** | **0.82** | **0.79** | **0.16** | **0.59** | **0.02** | **30.77** |
| CCHVAE | None | 0.81 | 6.25 | 0.16 | 0.27 | 0.08 | 119.99 |
| CCHVAE | POSTPRO | 0.80 | 11.20 | 3.06 | 0.37 | 0.00 | 120.01 |
| CCHVAE | **MISOB** | **0.81** | **4.03** | **0.42** | **0.48** | **0.19** | **105.10** |

### Ablation Study

Impact of hyperparameter $\alpha$ on the fairness-accuracy trade-off (WT method, race as the sensitive attribute):

| $\alpha$ Range | Accuracy Trend | Worst Burden Trend | Worst TPR Trend |
|--------|-----------|------------|------------|
| 0.1-0.3 | Maintain / Slight Increase | Steady Decrease | Steady Increase |
| 0.3-0.5 | Maintain | Continuous Decrease | Continuous Increase |
| 0.5-1.0 | Start to Decrease | Tending to Stabilize | Tending to Stabilize |

Intersectional group (Race $\times$ Gender) analysis:

| Recourse Method | Strategy | Accuracy | Worst Burden ↓ | Worst TPR ↑ |
|---------|------|-------|-----------|----------|
| WT | None | 0.81 | 1.40 | 0.20 |
| WT | POSTPRO | 0.80 | 1.94 | 0.00 |
| WT | **MISOB** | **0.82** | **0.98** | **0.34** |

### Key Findings

- Although POSTPRO achieves TPR equality at the prediction level, it instead hurts fairness at the recourse level by increasing burden and cost (e.g., on CCHVAE, the worst burden increases from 6.25 to 11.20), confirming that "predictive fairness $\neq$ recourse fairness."
- MISOB systematically reduces the social burden of all groups while maintaining or even improving overall accuracy.
- MISOB requires only a single training run to evaluate arbitrary group partitions (single or intersectional attributes), whereas POSTPRO requires separate training for each partition.
- A small disparity metric $\Delta$ does not automatically equate to fairness; while MISOB sometimes yields a slightly larger $\Delta$, the absolute metrics for every single group are improved.

## Highlights & Insights

- The definition of social burden jointly considers the classifier's prediction error (TPR) and the recourse cost, revealing structural unfairness hidden by traditional "equal-cost" paradigms.
- Eliminating the need for sensitive attributes is a key practical advantage that aligns with privacy regulations such as GDPR.
- Solid theoretical contribution: it formalizes the relationship between recourse fairness and predictive fairness, proving that satisfying equality of opportunity does not guarantee equal social burden.
- The minimax perspective replaces disparity minimization, avoiding pseudo-fairness achieved by degrading the performance of privileged groups to "reach equality."

## Limitations & Future Work

- There is a lack of theoretical guarantees for the convergence of the algorithm, and the stability of iterative training relies heavily on the quality of pre-training.
- The recourse cost computation uses the $\ell_2$ distance, which may deviate from actual effort costs in real life.
- The framework is currently verified only in static settings; the dynamic evolution of social burden under distribution shifts over time remains unexplored.
- The $O(N^3)$ computational complexity might become a bottleneck for large-scale datasets.

## Related Work & Insights

- **vs Equal Cost Paradigm (von2022fairness, etc.)**: Traditional methods only focus on equalizing recourse costs for rejected individuals, ignoring group acceptance rate differences and the impact of misclassifications; MISOB defines a social burden metric that integrates both TPR and cost.
- **vs POSTPRO (Hardt et al.)**: POSTPRO achieves TPR equality through post-processing but can deteriorate recourse fairness; MISOB simultaneously improves both predictive and recourse fairness through instance weighting during the training stage.

## Rating

- Novelty: ⭐⭐⭐⭐ The formal definition of social burden and the minimax perspective constitute important theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Only validated on the Adult dataset, lacking evaluations on more diverse real-world datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivations are rigorous, the motivation progresses logically, and the problem analysis is in-depth.
- Value: ⭐⭐⭐⭐ Unveils fundamental problems in the field of recourse fairness with a framework that is both general and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Minimizing Inequity in Facility Location Games](minimizing_inequity_in_facility_location_games.md)
- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](../../ICML2026/ai_safety/position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[AAAI 2026\] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)

</div>

<!-- RELATED:END -->
