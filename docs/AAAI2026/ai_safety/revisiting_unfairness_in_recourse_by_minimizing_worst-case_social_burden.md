---
title: >-
  [Paper Note] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden
description: >-
  [AAAI 2026][AI Safety][algorithmic recourse] This paper systematically analyzes three fundamental limitations of existing fairness metrics in algorithmic recourse—neglecting classifier decision behavior…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "algorithmic recourse"
  - "social burden"
  - "fairness"
  - "minimax optimization"
  - "counterfactual explanation"
date: 2026-05-08
content_hash: b5d3236b60736c85
---

# Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden

**Conference**: AAAI 2026
**arXiv**: [2509.04128](https://arxiv.org/abs/2509.04128)  
**Code**: [github](https://github.com/abarrainkua/MISOB)  
**Area**: AI Safety / Algorithmic Fairness
**Keywords**: algorithmic recourse, social burden, fairness, minimax optimization, counterfactual explanation

## TL;DR

This paper systematically analyzes three fundamental limitations of existing fairness metrics in algorithmic recourse—neglecting classifier decision behavior, ignoring ground-truth labels, and the tendency of gap-based metrics to obscure unfairness—and proposes MISOB, a fairness framework grounded in social burden. Through a minimax-weighted training strategy, MISOB reduces social burden across all demographic groups without requiring access to sensitive attributes, simultaneously improving fairness at both the prediction and recourse stages.

## Background & Motivation

Automated decision-making systems are widely deployed in domains such as credit approval and public services. When a model issues a negative decision, it should provide actionable recourse suggestions (e.g., "increase monthly income by 500") to enable individuals to overturn the outcome. However, the recourse process itself can be unfair: different groups may receive identical recommendations yet face vastly different implementation costs.

Existing research on recourse fairness suffers from three core problems: (1) **Neglecting classifier decision behavior**—computing equal mean recourse costs only for rejected individuals ignores that groups with higher rejection rates bear a disproportionately greater overall recourse burden; (2) **Ignoring ground-truth labels**—failing to distinguish individuals who *should have been accepted but were incorrectly rejected* from those who were legitimately rejected, where the former are compelled to alter their features due to a systemic error; (3) **Gap metrics may mask unfairness**—a zero gap between two groups' social burdens does not imply fairness, as both groups may be suffering equally high burdens.

- **Key Insight**: Reframing fairness from gap minimization to a Rawlsian minimax perspective that focuses on the worst-off group's social burden, and proposing MISOB, a lightweight training method that requires no sensitive attributes.

## Method

### Overall Architecture

MISOB is an iterative training framework: (1) pretrain a base classifier $f^{(0)}$; (2) at each iteration, compute the social burden for each training instance and retrain the classifier using a burden-weighted loss. Instances with higher burden receive larger weights, guiding the classifier to prioritize improving decisions for these individuals.

### Key Designs

**1. Definition of Social Burden**

Traditional recourse cost focuses on the average cost over all rejected individuals, whereas social burden targets individuals who **should have been accepted but were incorrectly rejected**. For sensitive group $s$:

$$B_{f,g}^s = \underbrace{\mathbb{E}[\delta((X,s), g_f(X))]}_{\text{recourse cost for false negatives}} \cdot (1 - \underbrace{P(f(X)=1|S=s, Y=1)}_{\text{TPR}})$$

where $\delta$ is the recourse cost function and $g_f$ is the recourse algorithm. Social burden jointly accounts for the probability of being incorrectly rejected (related to TPR) and the cost of feature modification. Ideally, social burden should be zero.

The paper also defines the expected recourse cost $C_{f,g}^s$, replacing TPR with the acceptance rate (AR):

$$C_{f,g}^s = \mathbb{E}[\delta((X,s), g_f(X))] \cdot (1 - P(f(X)=1|S=s))$$

**2. Burden-Aware Instance Weighting**

The core of MISOB is an instance weighting scheme. The weight for training instance $x^i$ is:

$$\phi(i, \mathcal{Q}, \alpha) = 1 + \alpha N \frac{b_{f,g_f}^i}{\sum_{j} b_{f,g_f}^j} \mathbb{1}\{\beta > 0\}$$

where $b_{f,g}^i = \delta(x^i, g_f(x^i)) \cdot \mathbb{1}\{y^i = 1\}$ is the instance-level burden (only positive-class instances incur burden), and $\alpha$ is a hyperparameter balancing fairness and accuracy. Instances with higher burden receive larger weights, pushing the classifier to prioritize improvements for these cases.

**3. No Sensitive Attributes Required**

MISOB requires no access to sensitive attributes at either training or inference time. Burden computation is based on ground-truth labels and recourse costs rather than group membership. This entails: (a) no need to collect sensitive information, avoiding legal and ethical risks; (b) natural handling of intersectional fairness (e.g., "young minority women"), since group definitions can be specified post hoc at evaluation time.

### Loss & Training

Weighted classification loss: $\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \phi(i, \mathcal{Q}, \alpha) \cdot \ell(f(x^i), y^i)$. The procedure begins with pretraining $f^{(0)}$, followed by $T$ rounds of iterative optimization. Recourse costs must be recomputed at each round. The overall computational complexity is $O(N^3)$, which can be improved through batching and parallelization.

## Key Experimental Results

### Main Results

Results on the Adult dataset with race as the sensitive attribute (averaged over 10 random splits):

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

Effect of hyperparameter $\alpha$ on the fairness–accuracy trade-off (WT method, race as sensitive attribute):

| $\alpha$ Range | Accuracy Trend | Worst Burden Trend | Worst TPR Trend |
|--------|-----------|------------|------------|
| 0.1–0.3 | Stable / slight increase | Steady decrease | Steady increase |
| 0.3–0.5 | Stable | Continued decrease | Continued increase |
| 0.5–1.0 | Begins to decline | Plateaus | Plateaus |

Intersectional group (race × gender) analysis:

| Recourse Method | Strategy | Accuracy | Worst Burden ↓ | Worst TPR ↑ |
|---------|------|-------|-----------|----------|
| WT | None | 0.81 | 1.40 | 0.20 |
| WT | POSTPRO | 0.80 | 1.94 | 0.00 |
| WT | **MISOB** | **0.82** | **0.98** | **0.34** |

### Key Findings

- Although POSTPRO achieves TPR parity at the prediction level, it actually increases burden and cost at the recourse level (worst burden on CCHVAE rises from 6.25 to 11.20), confirming that **prediction fairness does not imply recourse fairness**.
- MISOB systematically reduces social burden across all groups without sacrificing—and often improving—overall accuracy.
- A single MISOB training run enables evaluation across arbitrary group definitions (single or intersectional attributes), whereas POSTPRO requires separate training for each partition.
- A small gap metric Δ does not imply fairness—MISOB sometimes yields a slightly larger Δ, yet all absolute per-group metrics are strictly better.

## Highlights & Insights

- The definition of social burden jointly accounts for the classifier's prediction error (TPR) and recourse cost, revealing structural unfairness concealed by the traditional "equal cost" paradigm.
- The absence of sensitive attribute requirements is a key practical advantage, aligning with privacy regulations such as GDPR.
- The theoretical contributions are rigorous: the paper formalizes the relationship between recourse fairness and predictive fairness, and proves that satisfying equal opportunity does not guarantee equal social burden.
- The minimax perspective replaces gap minimization, avoiding pseudo-fairness achieved by degrading the privileged group's performance.

## Limitations & Future Work

- No theoretical convergence guarantees are provided for the iterative training procedure; stability depends on the quality of pretraining.
- Recourse costs are computed using $\ell_2$ distance, which may diverge from the true effort costs in real-world settings.
- Validation is conducted only in a static setting; the dynamic evolution of social burden under distributional shift over time remains unexplored.
- Computational complexity of $O(N^3)$ may become a bottleneck for large-scale datasets.

## Related Work & Insights

- **vs. Equal Cost paradigm (von2022fairness et al.)**: Traditional methods focus solely on equalizing recourse costs among rejected individuals, ignoring group-level acceptance rate disparities and the effects of misclassification. MISOB defines a social burden metric that incorporates both TPR and cost.
- **vs. POSTPRO (Hardt et al.)**: POSTPRO achieves TPR parity via post-processing but may worsen recourse fairness. MISOB improves both predictive and recourse fairness simultaneously through instance weighting during training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The formal definition of social burden and the minimax perspective represent significant theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐ Validation is limited to the Adult dataset; experiments on additional real-world datasets are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, motivation is developed progressively, and problem analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Identifies fundamental issues in recourse fairness research; the framework is general and practically applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Minimizing Inequity in Facility Location Games](minimizing_inequity_in_facility_location_games.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[AAAI 2026\] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)
- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](alternative_fairness_and_accuracy_optimization_in_criminal_j.md)

</div>

<!-- RELATED:END -->
