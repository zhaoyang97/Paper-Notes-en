---
title: >-
  [Paper Note] On the Impact of the Utility in Semivalue-based Data Valuation
description: >-
  [ICLR 2026][Data Valuation] This paper introduces a geometric representation termed *spatial signature* to unify the modeling of utility selection in data valuation as a directional rotation problem on the unit circle. I…
tags:
  - "ICLR 2026"
  - "Data Valuation"
  - "Semivalue"
  - "Shapley Value"
  - "Banzhaf Value"
  - "Robustness"
date: 2026-05-08
content_hash: 045ba11d1975d836
---

# On the Impact of the Utility in Semivalue-based Data Valuation

**Conference**: ICLR 2026
**arXiv**: [2502.06574](https://arxiv.org/abs/2502.06574)
**Code**: [https://github.com/taminemelissa/utility-impact](https://github.com/taminemelissa/utility-impact)
**Area**: Data Valuation / AI Theory
**Keywords**: Data Valuation, Semivalue, Shapley Value, Banzhaf Value, Robustness

## TL;DR

This paper introduces a geometric representation termed *spatial signature* to unify the modeling of utility selection in data valuation as a directional rotation problem on the unit circle. It further proposes a robustness metric $R_p$ and demonstrates that the Banzhaf value exhibits the highest ranking stability across different utility functions.

## Background & Motivation

**Background**: Semivalue-based data valuation is a mainstream approach to data quality assessment, employing solution concepts from cooperative game theory—such as the Shapley value, Beta Shapley, and Banzhaf value—to assign a value score to each data point that reflects its contribution to downstream ML tasks. These methods are widely used for identifying high-quality training samples, data cleansing, and fair data pricing.

**Limitations of Prior Work**: Computing semivalues requires a user-specified utility function, a choice that is inherently subjective. For instance, when training a cat-dog classifier, accuracy, precision, recall, F1, and AUROC are all plausible utilities, yet different choices may yield entirely different data rankings. Experiments conducted on 8 datasets reveal that, on the Titanic dataset under Shapley values, the rank correlation between accuracy and F1 is as low as $-0.19$, indicating severe ranking instability.

**Key Challenge**: Data valuation methods purport to objectively assess the importance of individual data points, yet their outputs are highly sensitive to the choice of utility function—a choice for which no uniquely correct answer exists. This leaves practitioners unable to determine whether their valuation results are trustworthy.

**Goal**: (1) How can the problem of utility variation's impact on rankings be modeled in a unified manner? (2) How can this robustness be quantified? (3) How large are the robustness differences between semivalues (Shapley vs. Banzhaf), and why?

**Key Insight**: The authors observe that, for any semivalue, the data value scores under all utility functions can be expressed as a linear functional in a low-dimensional space. This implies that ranking changes can be geometrized as variations in projection order induced by directional rotation on the unit circle—a concise and analytically tractable problem.

**Core Idea**: Each data point is embedded into a two-dimensional space (the spatial signature) determined by the semivalue weights and the base utilities, transforming the question of ranking stability under utility variation into a geometric problem that admits precise measurement and comparison.

## Method

### Overall Architecture

Given a dataset $\mathcal{D} = \{z_i\}_{i \in [n]}$, a semivalue weight vector $\omega$, and two base utilities $u_1, u_2$, the proposed method proceeds in three steps: (1) embed each data point $z_i$ into $\mathbb{R}^2$ to form its spatial signature; (2) analyze how rankings induced by projections along all directions $\bar{\alpha}$ on the unit circle $\mathcal{S}^1$ vary with rotation; (3) compute the robustness metric $R_p$ to measure ranking stability.

### Key Designs

1. **Unified Modeling of Two Scenarios**:

    - *Function*: Unifies the utility trade-off scenario and the multiple-valid-utility scenario within a single geometric framework.
    - *Mechanism*: In the utility trade-off scenario, $u_\nu = \nu u^A + (1-\nu) u^B$, where $\nu$ controls the trade-off between two objectives. In the multiple-valid-utility scenario, common classification metrics (accuracy, F1, precision, etc.) can each be approximated in the linear-fractional form $u(S) = \frac{c_0 + c_1\lambda(S) + c_2\gamma(S)}{d_0 + d_1\lambda(S) + d_2\gamma(S)}$, where $\lambda$ is the true-positive rate and $\gamma$ is the positive-prediction rate; after a first-order expansion, $u$ is approximately affine in $(\lambda, \gamma)$. Both scenarios thus reduce to the form $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$.
    - *Design Motivation*: A unified framework allows a single robustness metric to apply to both scenarios, substantially broadening the method's scope of applicability.

2. **Spatial Signature and Geometric Mapping**:

    - *Function*: Transforms the data valuation problem into a visualizable and analytically tractable geometric problem.
    - *Mechanism*: By Proposition 3.1, there exists a mapping $\psi_{\omega,\mathcal{D}}: \mathcal{D} \to \mathbb{R}^2$ such that for any utility $u_\alpha$, $\phi(z; \omega, u_\alpha) = \langle \psi_{\omega,\mathcal{D}}(z), \alpha \rangle$. Ranking stability is then equivalent to asking whether the projection order of all embedded points along direction $\alpha$ changes as the direction rotates. If all embedded points are approximately collinear, rotation has minimal effect on projection order, yielding maximal robustness.
    - *Design Motivation*: The linear inner-product structure directly links ranking changes to geometric angles, abstracting away the complexity of actual utility computation.

3. **Robustness Metric $R_p$**:

    - *Function*: Quantifies the stability of rankings under utility variation.
    - *Mechanism*: For each pair of data points $(z_i, z_j)$, a *cut direction* is defined as $H_{ij} = \{\alpha \in \mathcal{S}^1 : \langle \alpha, v_{ij} \rangle = 0\}$, where $v_{ij} = \psi(z_i) - \psi(z_j)$. All $\binom{n}{2}$ pairs produce $2N$ cut points that partition the unit circle into arcs of constant ranking. $\rho_p(\bar{\alpha}_0)$ denotes the minimum arc length from a starting direction $\bar{\alpha}_0$ required to produce $p$ pairwise rank swaps. The metric $R_p = \frac{\mathbb{E}[\rho_p]}{\pi/4}$ normalizes the result to $[0,1]$, with denominator $\pi/4$ corresponding to the maximum value attained when all points are collinear.
    - *Design Motivation*: $R_p$ can be computed exactly in $O(n^2 \log n)$ time and directly corresponds to the degree of Kendall rank correlation degradation.

### Loss & Training

This paper presents an analytical framework rather than a neural network training procedure. The central theoretical result, Proposition 3.3, shows that the Pearson correlation between semivalue score vectors under two base utilities decomposes as
$$\text{Corr}(\phi(u_1), \phi(u_2)) = \frac{\sum_j \omega_j^2 r_j}{\sqrt{\sum_j \omega_j^2 \text{Var}_j(u_1)} \sqrt{\sum_j \omega_j^2 \text{Var}_j(u_2)}},$$
where $r_j$ is the size-$j$ alignment factor. Banzhaf weights concentrate mass on the intermediate coalition sizes where $r_j$ tends to be largest, thereby systematically achieving higher correlation and robustness.

## Key Experimental Results

### Main Results: Kendall Rank Correlation Across Semivalues and Datasets

| Dataset | Shapley | (4,1)-Beta Shapley | Banzhaf |
|---------|---------|-------------------|---------|
| Breast | 0.95 ± 0.003 | 0.95 ± 0.003 | **0.97 ± 0.008** |
| Titanic | -0.19 ± 0.007 | -0.17 ± 0.01 | **0.94 ± 0.003** |
| Credit | -0.47 ± 0.01 | -0.44 ± 0.02 | **0.87 ± 0.01** |
| Heart | 0.64 ± 0.006 | 0.68 ± 0.004 | **0.96 ± 0.003** |
| Wind | 0.81 ± 0.008 | 0.82 ± 0.008 | **0.99 ± 0.002** |
| Cpu | 0.59 ± 0.02 | 0.62 ± 0.02 | **0.86 ± 0.007** |

Rank correlations between accuracy and F1 as utility functions. The Banzhaf value significantly outperforms Shapley and Beta Shapley across all datasets.

### Validation of the Robustness Metric $R_p$

| Dataset | Scenario | Shapley $R_p$ | Banzhaf $R_p$ | Consistency |
|---------|----------|-------------|-------------|-------------|
| Breast | Multiple utility | High | Highest | $R_p$ consistent with Kendall correlation |
| Titanic | Multiple utility | Very low | High | $R_p$ accurately reflects ranking instability |
| Diabetes | Utility trade-off | Moderate | Highest | Equally applicable to regression tasks |
| Digits | Utility trade-off | Moderate | Highest | Equally applicable to multi-class tasks |

### Key Findings

- **Geometric Explanation for Banzhaf's Consistent Advantage**: Banzhaf weights cause the spatial signature embeddings to become nearly collinear, which directly maximizes $R_p$. This occurs because Banzhaf weights $\omega_j = \binom{n-1}{j-1} / 2^{n-1}$ concentrate on intermediate coalition sizes, where the size-specific alignment factor $r_j$ is typically largest.
- **Consistency Between $R_p$ and Rank Correlation**: Across all experiments, the magnitude of $R_p$ strictly corresponds to Kendall rank correlation, validating the practical utility of the geometric framework.
- **Counterintuitive Finding**: On certain datasets (e.g., Titanic), rankings produced by Shapley and Beta Shapley under different utilities are even negatively correlated, indicating that these semivalues are entirely unreliable as data valuation tools in such settings.

## Highlights & Insights

- **Elegant Geometric Perspective**: Translating the abstract ranking stability problem from cooperative game theory into a two-dimensional projection ordering problem yields clear geometric intuition and precise mathematical correspondences. Such a bridge from algebraic to geometric reasoning is rare in ML theory.
- **High Practical Guidance Value**: The $R_p$ metric informs practitioners whether their data valuation results are trustworthy—a low $R_p$ indicates that rankings are unstable regardless of utility choice, signaling that semivalue methods should not be used in that setting.
- **Theoretical Explanation for Banzhaf Superiority**: While prior literature has empirically observed that the Banzhaf value is more stable, this paper provides the first theoretical explanation through the interaction between the weight distribution and alignment factors.

## Limitations & Future Work

- **Scope of the Linear-Fractional Approximation**: The analysis of the multiple-valid-utility scenario relies on a first-order linear approximation of utility with respect to $(\lambda, \gamma)$, which does not extend to highly nonlinear metrics such as negative log-loss.
- **Limited to Binary Classification and Certain Multi-Class Metrics**: Although regression utilities (e.g., MSE vs. MAE) are validated in the trade-off scenario, a unified linear-fractional derivation analogous to that for classification metrics is absent.
- **Computational Complexity**: Exact computation of $R_p$ requires $O(n^2 \log n)$ time, which may remain costly for very large-scale datasets.
- **Propagation of Utility Approximation Error**: The impact of errors introduced by the linear approximation on $R_p$ is not quantified.

## Related Work & Insights

- **vs. Data Shapley (Ghorbani & Zou, 2019)**: Data Shapley assigns uniform weights to all coalition sizes, making it susceptible to the high-variance marginal contributions of extreme-size coalitions and thus less robust. This paper explains why Banzhaf outperforms Shapley.
- **vs. Diehl & Wilson (2025)**: That work similarly identifies the unreliability and manipulability of semivalue-based valuations under ill-defined utilities, but only exposes the problem. The present paper goes further by providing tools to quantify fragility and guidance for selecting semivalues.
- **vs. Wang & Jia (2023)**: Data Banzhaf establishes robustness to the stochasticity of learning algorithms; this paper extends robustness analysis to the utility dimension.

## Rating

- Novelty: ⭐⭐⭐⭐ A genuinely novel perspective on geometric analysis of data valuation robustness, though the problem scope is relatively narrow.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, multiple semivalues, and both scenarios; strong agreement between theory and experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, complete logical chain, excellent figures, and tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Offers direct practical guidance for data valuation, though the primary audience is largely confined to the data valuation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)

</div>

<!-- RELATED:END -->
