---
title: >-
  [Paper Note] On the Impact of the Utility in Semivalue-based Data Valuation
description: >-
  [ICLR 2026][Others][Data Valuation] This paper introduces a geometric representation termed "spatial signature" to unify the problem of utility selection in data valuation as a directional rotation on the unit circle. It proposes a quantitative robustness metric $R_p$, revealing that the Banzhaf value exhibits the highest ranking stability across differe
tags:
  - ICLR 2026
  - Others
  - Data Valuation
  - Semivalue
  - Shapley Value
  - Banzhaf Value
  - Robustness
date: 2026-05-08
content_hash: e92c2e8b166d67ec
---
# On the Impact of the Utility in Semivalue-based Data Valuation

**Conference**: ICLR 2026  
**arXiv**: [2502.06574](https://arxiv.org/abs/2502.06574)  
**Code**: [https://github.com/taminemelissa/utility-impact](https://github.com/taminemelissa/utility-impact)  
**Area**: Data Valuation / AI Theory  
**Keywords**: Data Valuation, Semivalue, Shapley Value, Banzhaf Value, Robustness

## TL;DR

This paper introduces a geometric representation termed "spatial signature" to unify the problem of utility selection in data valuation as a directional rotation on the unit circle. It proposes a quantitative robustness metric $R_p$, revealing that the Banzhaf value exhibits the highest ranking stability across different utilities.

## Background & Motivation

**Background**: Semivalue-based data valuation is a predominant method for data quality assessment, utilizing solution concepts from cooperative game theory (such as Shapley Value, Beta Shapley, and Banzhaf Value) to assign value scores to each data point. These metrics measure contribution to downstream ML tasks and are widely used for identifying high-quality training samples, data cleaning, and fair data pricing.

**Limitations of Prior Work**: The calculation of semivalues relies on the user's choice of utility function, which is often subjective. For instance, when training a cat-dog classifier, accuracy, precision, recall, F1, and AUROC are all reasonable utilities, yet different utilities can lead to entirely different data rankings. Experiments across 8 datasets show that using the Shapley value on the Titanic dataset yields a ranking correlation of $-0.19$ between accuracy and F1, indicating extreme instability.

**Key Challenge**: Data valuation methods claim to objectively assess data point importance, yet results are highly dependent on utility choice—for which no unique "correct" answer exists. This makes it difficult for practitioners to judge the reliability of their data valuation results.

**Goal**: (1) How to unify the modeling of utility-driven ranking changes? (2) How to quantify this robustness? (3) What are the robustness differences between various semivalues (e.g., Shapley vs. Banzhaf), and why?

**Key Insight**: The authors observe that for any semivalue, data values under all utilities can be represented through a linear functional in a low-dimensional space. This allows ranking changes to be geometrized as changes in projection order when a direction rotates on a unit circle—a concise and analyzable problem.

**Core Idea**: Embed each data point into a 2D space determined by semivalue weights and base utilities (the spatial signature). This transforms the stability of rankings under utility changes into a geometric problem that can be precisely measured and compared.

## Method

### Overall Architecture

This work does not train a model but rather builds an analytical framework to address a practical question: will changing the utility significantly alter the data valuation ranking? Given a dataset $\mathcal{D} = \{z_i\}_{i \in [n]}$, a semivalue weight vector $\omega$, and two base utilities $u_1, u_2$, the paper simplifies the problem in four stages. First, two sources of utility uncertainty (explicit trade-offs and multiple valid metrics) are unified into a linear combination $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$, collapsing the choice of utility into a coefficient direction $\alpha$. Second, it is proven that each data point can be embedded into a 2D plane as its spatial signature, where the value score under any utility is exactly the projection of that signature onto the direction $\alpha$. Ranking stability is thus equivalent to the geometric question of whether projection orders flip as $\alpha$ rotates on the unit circle $\mathcal{S}^1$. Third, a robustness metric $R_p$ is defined to measure stability as a value in $[0,1]$, representing how much rotation is required to disrupt the ranking. Fourth, this geometric language is used to explain why the Banzhaf value remains the most stable across utilities.

### Key Designs

**1. Unified Modeling: Collapsing Utility Selection into a Linear Combination**

Uncertainty in utility selection stems from two scenarios, which the authors prove can be unified mathematically. One is utility trade-off, where users weight two objectives: $u_\nu = \nu u^A + (1-\nu) u^B$. The other is the "multiple-valid-utility" scenario, where accuracy, F1, and precision all seem reasonable for the same classifier. The key observation is that common classification metrics can be expressed as linear fractional functions of the true-positive rate $\lambda$ and positive-prediction rate $\gamma$: $u(S) = \frac{c_0 + c_1\lambda(S) + c_2\gamma(S)}{d_0 + d_1\lambda(S) + d_2\gamma(S)}$. A first-order expansion shows $u$ is approximately affine concerning $(\lambda, \gamma)$. Consequently, both scenarios collapse into $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$. Analyzing how $\alpha$ changes the ranking covers both trade-off and multiple-valid-utility problems.

**2. Spatial Signature: Mapping Stability to Unit Circle Projections**

Given $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$, Proposition 3.1 provides the fundamental geometric mapping: there exists $\psi_{\omega,\mathcal{D}}: \mathcal{D} \to \mathbb{R}^2$ such that the semivalue score of a point under any utility $u_\alpha$ is an inner product:

$$\phi(z; \omega, u_\alpha) = \langle \psi_{\omega,\mathcal{D}}(z), \alpha \rangle.$$

Each data point is embedded as a spatial signature in a 2D plane, and "choosing a utility" is equivalent to "projecting onto direction $\alpha$." Data ranking is the order of these 2D points projected onto $\alpha$. Stability is therefore determined by whether the projection order flips as $\alpha$ rotates on the unit circle. If the embedded points are approximately collinear, the ranking remains consistent regardless of the projection direction, indicating high robustness. The more dispersed the points, the more likely a small rotation will flip the ranking.

**3. Robustness Metric $R_p$: Quantifying Stability via Rotation Angles**

For every pair of data points $(z_i, z_j)$, let $v_{ij} = \psi(z_i) - \psi(z_j)$. The critical direction where their projection order flips is the "cutting angle" $H_{ij} = \{\alpha \in \mathcal{S}^1 : \langle \alpha, v_{ij} \rangle = 0\}$ orthogonal to $v_{ij}$. The $\binom{n}{2}$ pairs generate $2N$ cutting points, partitioning the unit circle into arcs within which the ranking is invariant. The definition of $\rho_p(\bar{\alpha}_0)$ is the minimum arc length to be swept from a starting direction $\bar{\alpha}_0$ to accumulate $p$ pairwise swaps. This angle is normalized to a metric in $[0,1]$ by taking its expectation over starting directions:

$$R_p = \frac{\mathbb{E}[\rho_p]}{\pi/4},$$

where $\pi/4$ is the maximum $\rho_p$ achievable when all points are perfectly collinear. $R_p$ can be calculated in $O(n^2 \log n)$ time, and its magnitude directly corresponds to the degradation of Kendall ranking correlation.

**4. Why Banzhaf is Most Robust: Alignment and Weight Concentration**

Proposition 3.3 decomposes the Pearson correlation between semivalue score vectors under two base utilities:

$$\text{Corr}(\phi(u_1), \phi(u_2)) = \frac{\sum_j \omega_j^2 r_j}{\sqrt{\sum_j \omega_j^2 \text{Var}_j(u_1)} \sqrt{\sum_j \omega_j^2 \text{Var}_j(u_2)}},$$

where $\omega_j$ is the semivalue weight for coalition size $j$, and $r_j$ is the alignment factor (measuring how aligned marginal contributions of two utilities are at size $j$). Higher correlation implies spatial signature points are closer to a single line, leading to a larger $R_p$. Empirically, $r_j$ is highest at moderate coalition sizes and decays at the extremes. Banzhaf weights $\omega_j = \binom{n-1}{j-1}/2^{n-1}$ are concentrated precisely in this moderate size region. Thus, Banzhaf systematically puts weight where $r_j$ is large, resulting in high correlation and robustness. Shapley, by contrast, distributes weight across all sizes and is hindered by high variance at the extremes, making it less stable.

## Key Experimental Results

### Main Results: Kendall Ranking Correlation across Semivalues and Datasets

| Dataset | Shapley | (4,1)-Beta Shapley | Banzhaf |
|--------|---------|-------------------|---------|
| Breast | 0.95 ± 0.003 | 0.95 ± 0.003 | **0.97 ± 0.008** |
| Titanic | -0.19 ± 0.007 | -0.17 ± 0.01 | **0.94 ± 0.003** |
| Credit | -0.47 ± 0.01 | -0.44 ± 0.02 | **0.87 ± 0.01** |
| Heart | 0.64 ± 0.006 | 0.68 ± 0.004 | **0.96 ± 0.003** |
| Wind | 0.81 ± 0.008 | 0.82 ± 0.008 | **0.99 ± 0.002** |
| Cpu | 0.59 ± 0.02 | 0.62 ± 0.02 | **0.86 ± 0.007** |

Ranking correlation between accuracy and F1 utility. Banzhaf significantly outperforms Shapley and Beta Shapley across all datasets.

### Robustness Metric $R_p$ Validation

| Dataset | Scenario | Shapley $R_p$ | Banzhaf $R_p$ | Consistency |
|--------|------|-------------|-------------|--------|
| Breast | Multi-utility | High | Highest | $R_p$ aligns with Kendall correlation |
| Titanic | Multi-utility | Extremely Low | High | $R_p$ reflects ranking instability |
| Diabetes | Utility trade-off | Medium | Highest | Applicable to regression tasks |
| Digits | Utility trade-off | Medium | Highest | Applicable to multi-class tasks |

### Key Findings

- **Geometric Explanation for Banzhaf's Consistency**: Banzhaf weights make spatial signature points nearly collinear, maximizing $R_p$. This is because Banzhaf weights align with the moderate coalition sizes where the alignment factor $r_j$ is typically highest.
- **Consistency between $R_p$ and Ranking Correlation**: In all experiments, $R_p$ magnitude corresponds strictly with Kendall correlation, validating the geometric framework.
- **Counter-intuitive Finding**: On certain datasets (e.g., Titanic), Shapley and Beta Shapley rankings under different utilities are even negatively correlated, suggesting these semivalues are unreliable tools for data valuation in such contexts.

## Highlights & Insights

- **Compelling Geometric Perspective**: Transforming abstract ranking stability in cooperative game theory into a projection problem in 2D space provides clear intuition and precise mathematical mapping, a rare bridge in ML theory.
- **Practical Guidance**: The $R_p$ metric provides practitioners with a way to check if their data valuation is trustworthy—if $R_p$ is low, the ranking is unstable regardless of utility, and semivalue methods should be avoided.
- **Theoretical Grounding of Banzhaf Superiority**: While previous literature identified Banzhaf's empirical stability, this paper provides the first theoretical explanation through the interaction of weight distribution and alignment factors.

## Limitations & Future Work

- **Scope of Linear Fractional Approximation**: The analysis of the multiple-valid-utility scenario relies on first-order linear approximations of $(\lambda, \gamma)$, which may not apply to non-linear metrics like negative log-loss.
- **Constraint to Binary and Specific Multi-class Metrics**: While regression utilities (MSE vs. MAE) were validated in trade-off scenarios, they lack a unified linear fractional derivation.
- **Computational Complexity**: Exact calculation of $R_p$ requires $O(n^2 \log n)$, which may remain expensive for extremely large datasets.
- **Utility Approximation Error**: The propagation of errors introduced by linear approximations into $R_p$ remains unquantified.

## Related Work & Insights

- **vs. Data Shapley (Ghorbani & Zou, 2019)**: Data Shapley weights all coalition sizes equally, making it susceptible to high-variance marginal contributions from extreme sizes. This paper explains why Banzhaf is more robust.
- **vs. Diehl & Wilson (2025)**: While that work highlights that semivalue valuation is unreliable and manipulable when utilities are under-defined, it only exposes the problem. This paper provides tools to quantify vulnerability and choose semivalues.
- **vs. Wang & Jia (2023)**: Data Banzhaf demonstrated robustness to learning algorithm randomness; this work extends robustness analysis to the dimension of utility choice.

## Rating

- Novelty: ⭐⭐⭐⭐ Geometrizing robustness in data valuation is a fresh perspective, though the problem setting is relatively specific.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, semivalues, and scenarios, with strong alignment between theory and experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, complete logical chain, and excellent visualizations with tight theory-experiment integration.
- Value: ⭐⭐⭐⭐ Directly impacts data valuation practice, though primarily targeted at the data valuation research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)

</div>

<!-- RELATED:END -->
