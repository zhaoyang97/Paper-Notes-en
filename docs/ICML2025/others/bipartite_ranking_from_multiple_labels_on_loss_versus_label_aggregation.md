---
title: >-
  [Paper Note] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation
description: >-
  [ICML2025][bipartite ranking] This paper theoretically analyzes the Bayes-optimal solutions of two aggregation strategies in multi-label bipartite ranking—loss aggregation and label aggregation—revealing that loss aggregation suffers from a "label dictatorship" phenomenon (where a single label dominates the ranking due to marginal skewness), whereas label aggregation treats all labels in a more balanced manner.
tags:
  - "ICML2025"
  - "bipartite ranking"
  - "AUC"
  - "multi-label ranking"
  - "loss aggregation"
  - "label aggregation"
  - "Pareto optimality"
  - "label dictatorship"
date: 2026-05-08
content_hash: 109bd37d5437a10b
---

# Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation

**Conference**: ICML2025  
**arXiv**: [2504.11284](https://arxiv.org/abs/2504.11284)  
**Code**: None  
**Area**: Others/Learning Theory  
**Keywords**: bipartite ranking, AUC, multi-label ranking, loss aggregation, label aggregation, Pareto optimality, label dictatorship

## TL;DR

This paper theoretically analyzes the Bayes-optimal solutions of two aggregation strategies in multi-label bipartite ranking—loss aggregation and label aggregation—revealing that loss aggregation suffers from a "label dictatorship" phenomenon (where a single label dominates the ranking due to marginal skewness), whereas label aggregation treats all labels in a more balanced manner.

## Background & Motivation

- **Bipartite Ranking** is a fundamental problem in supervised learning, where the goal is to learn a scoring function $f: \mathcal{X} \to \mathbb{R}$ that ranks positive instances above negative ones to maximize the Area Under the ROC Curve (AUC).
- Traditional setups assume only a **single binary label**, but in practice, **multiple binary labels** often exist (e.g., relevance annotations and click-through rates in information retrieval, opinions from multiple experts in medical diagnosis, and click probability and diversity in recommendation systems).
- Dealing with multiple labels, how to generate a single, reasonable ranking? Two main paradigms exist in the literature:
    - **Loss Aggregation (LoA)**: Weighted sum of AUC objectives for each label operates as the optimization target.
    - **Label Aggregation (LaA)**: Consolidating multiple labels into a single aggregated label (e.g., sum) first, and then optimizing multi-partite AUC on this aggregated label.
- While both are widely used in practice, there is a lack of theoretical analysis guiding which one to choose. This paper fills this gap.

## Method

### Problem Formulation

Given $K$ binary labels $(\mathsf{Y}^{(1)}, \ldots, \mathsf{Y}^{(K)}) \in \{0,1\}^K$, with joint distribution $D^{\text{jnt}}$, the conditional probability for each label is $\eta^{(k)}(x) = P(\mathsf{Y}^{(k)}=1 \mid \mathsf{X}=x)$, and the marginal prior is $\pi^{(k)} = P(\mathsf{Y}^{(k)}=1)$.

### Loss Aggregation (LoA)

$$\text{AUC}_{\text{LoA}}(f) = \sum_{k \in [K]} a_k \cdot \text{AUC}(f; D^{(k)})$$

**Bayes-optimal Solution** (Proposition 5.2): The optimal scoring function is equivalent to ranking based on the following weighted sum:

$$\gamma(x) = \frac{1}{K} \sum_{k \in [K]} \frac{a_k}{\pi^{(k)} \cdot (1-\pi^{(k)})} \cdot \eta^{(k)}(x)$$

Key Observation: Even when weights are uniform ($a_k = 1$), the optimal solution still depends on the **marginal class priors** $\pi^{(k)}$. Labels with more skewed distributions receive larger weights ($1/[\pi^{(k)}(1-\pi^{(k)})]$ increases sharply when $\pi^{(k)}$ is far from 0.5).

### Label Aggregation (LaA)

First aggregate labels as $\bar{\mathsf{Y}} = \sum_k \mathsf{Y}^{(k)}$, then optimize multi-partite AUC:

$$\text{AUC}_{\text{LaA}}(f) = \mathbb{E}[c_{\bar{Y}\bar{Y}'} \cdot H(f(X)-f(X')) \mid \bar{Y} > \bar{Y}']$$

When the cost function is set to $c_{\bar{y}\bar{y}'} = |\bar{y} - \bar{y}'| \cdot \mathbf{1}(\bar{y} > \bar{y}')$, the **Bayes-optimal Solution** (Proposition 5.4) is:

$$\gamma(x) = \sum_{k \in [K]} \eta^{(k)}(x)$$

Key Advantage: The optimal solution is an **equally weighted sum** of the conditional probabilities of each label, completely independent of $\pi^{(k)}$, naturally avoiding label dictatorship.

### The "Label Dictatorship" Phenomenon

For deterministic label scenarios ($\eta^{(k)}(x) \in \{0,1\}$), Proposition 6.1 reveals:

- If $\alpha^{(1)} = a_1/[\pi^{(1)}(1-\pi^{(1)})] > \alpha^{(2)}$, then the ranking is **completely determined by label 1**;
- Otherwise, it is completely determined by label 2.

This means that labels with skewed marginal distributions of labels will "hijack" the overall ranking—even when both labels are of the same conditional quality. In contrast, label aggregation does not impose a partial order between $(1,0)$ and $(0,1)$, thus avoiding this issue.

## Key Experimental Results

### Synthetic Experiments

| Setting | Method | $\Delta_{\text{AUC}}$ (lower is better) |
|------|------|----------------------------------|
| $\pi^{(2)}$ skew increases | Loss Aggregation | Increases significantly with skewness |
| $\pi^{(2)}$ skew increases | Label Aggregation | Consistently remains low |

- As the skewness of label 2 increases, the difference in individual label AUCs for loss aggregation increases drastically (the dictatorship phenomenon), while label aggregation remains stable and balanced.

### Banking Dataset (UCI)

| Objective | AUC mortgage↑ | AUC loan↑ | Diff AUC↓ | Min AUC↑ |
|----------|---------------|-----------|-----------|----------|
| AUC(mortgage) | 0.637 | 0.523 | 0.113 | 0.523 |
| AUC(loan) | 0.550 | 0.573 | 0.023 | 0.550 |
| **AUC_LaA** | **0.616** | **0.562** | **0.054** | **0.562** |
| AUC_LoA(1,1) | 0.626 | 0.555 | 0.071 | 0.555 |

- Label aggregation performs best on Min AUC (0.562) with the lowest difference (0.054), yielding a more balanced result.

### MSLR Web30k Dataset

| Objective | AUC Click↑ | AUC Rel↑ | Diff AUC↓ | Min AUC↑ |
|----------|-----------|----------|-----------|----------|
| AUC(Click) | 0.74 | 0.61 | 0.13 | 0.61 |
| AUC(Rel) | 0.70 | 0.67 | 0.03 | 0.67 |
| **AUC_LaA** | **0.73** | **0.68** | **0.05** | **0.68** |
| AUC_LoA | 0.73 | 0.67 | 0.06 | 0.67 |

- Label aggregation Pareto-dominates loss aggregation (with equal weights), with Min AUC 0.68 > 0.67.

### HelpSteer Dataset

- Fixes the first label to coherence (most balanced), while varying the second label (with different skewness).
- The per-label AUC difference in loss aggregation is consistently higher than that of label aggregation across all settings.

## Highlights & Insights

- **Solid theoretical contribution**: This is the first work to fully characterize the Bayes-optimal solution forms for both aggregation methods, clearly bringing to light the hidden dependency on $\pi^{(k)}$ in loss aggregation.
- **Precise concept of "Label Dictatorship"**: Graphically demonstrates how loss aggregation forces a global preference between $(1,0)$ and $(0,1)$ using partial order diagrams (red arrows vs. unordered relations in label aggregation).
- **High practical guidance value**: For ranking scenarios with multiple label sources (information retrieval, recommendation systems, medical diagnosis), the authors recommend using label aggregation with a linear cost $c = |y - y'|$.
- **Choice of cost function is crucial**: Label aggregation with a uniform cost $c=1$ does not guarantee Pareto optimality (Proposition 5.3), whereas it is guaranteed with a linear cost $c=|y-y'|$ (Proposition 5.4).

## Limitations & Future Work

- **Limited to binary labels**: Although the aggregated label is multi-valued, the initial labels are restricted to $\{0,1\}$ and not extended to continuous ratings.
- **No new algorithms proposed**: This is more of a theoretical analysis; experiments directly employ existing AUC optimization methods (logistic/hinge surrogate losses).
- **Direct maximization of Min AUC is unexplored**: The authors note this as an interesting direction for future work.
- **Limited weight control between labels**: Label aggregation defaults to an equal-weight sum $\sum \eta^{(k)}$. Assigning non-uniform weights would require deliberately designing alternative cost functions.
- **Insufficient discussion on scalability**: When $K$ is large, the domain of the aggregated label $\{0,\ldots,K\}$ expands, and its empirical behavior has not been fully verified.

## Related Work & Insights

- **Multi-objective Ranking**: Svore et al. (2011) pioneered the multi-label ranking paradigm; Carmel et al. (2020) proposed applying label aggregation in search.
- **Multi-partite AUC Theory**: Uematsu & Lee (2015) established the Bayes-optimal solution conditions (scale condition) for multi-partite AUC, which this paper directly leverages.
- **Pareto Optimality and Linear Scalarization**: Ruchte & Grabocka (2021) showed that optimal solutions of linear scalarization are Pareto-optimal, but this paper reveals that Pareto-optimality does not guarantee practicality.
- **Connection to Fairness**: The Min AUC metric shares conceptual similarities with worst-group performance in fair learning (Sagawa et al., 2020).

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic analysis of the theoretical differences between two aggregation strategies; the concept of label dictatorship is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic + 3 real-world datasets, covering various levels of skewness.
- Writing Quality: ⭐⭐⭐⭐⭐ — Smooth transition of definitions, propositions, proofs, and experiments; charts are very intuitive.
- Value: ⭐⭐⭐⭐ — Provides solid theoretical guidance for practical choices in multi-label ranking tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICML 2025\] Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[ICML 2025\] LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection](lapsum_--_one_method_to_differentiate_them_all_ranking_sorting_and_top-k_selecti.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
