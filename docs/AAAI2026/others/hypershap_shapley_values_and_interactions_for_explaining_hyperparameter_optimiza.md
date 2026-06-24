---
title: >-
  [Paper Note] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization
description: >-
  [AAAI 2026][Hyperparameter Optimization] HyperSHAP proposes a game-theoretic framework based on Shapley values and Shapley interactions to explain hyperparameter optimization (HPO). By defining four categories of explanation games—ablation, sensitivity, tunability, and optimizer bias—it provides more actionable hyperparameter importance analysis than fANOVA.
tags:
  - "AAAI 2026"
  - "Hyperparameter Optimization"
  - "Shapley Values"
  - "Explainability"
  - "Game Theory"
  - "Hyperparameter Importance"
date: 2026-05-08
content_hash: da0bd3d234766cd8
---

# HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization

**Conference**: AAAI 2026  
**arXiv**: [2502.01276](https://arxiv.org/abs/2502.01276)  
**Code**: [GitHub](https://github.com/automl/HyperSHAP)  
**Area**: Explainable AI / Automated Machine Learning  
**Keywords**: Hyperparameter Optimization, Shapley Values, Explainability, Game Theory, Hyperparameter Importance

## TL;DR

HyperSHAP proposes a game-theoretic framework based on Shapley values and Shapley interactions to explain hyperparameter optimization (HPO). By defining four categories of explanation games—ablation, sensitivity, tunability, and optimizer bias—it provides more actionable hyperparameter importance analysis than fANOVA.

## Background & Motivation

Hyperparameter optimization (HPO) is a critical step in achieving strong predictive performance in machine learning. However, the influence of individual hyperparameters on model generalization is highly context-dependent—dataset characteristics, performance metrics, and search spaces all affect hyperparameter importance. This complexity typically necessitates opaque black-box optimization methods to identify optimal configurations.

A fundamental barrier facing existing HPO methods is **the lack of explainability**, which leads to low user trust and insufficient adoption. Prior research has shown that practitioners tend to tune hyperparameters manually rather than using HPO tools even in high-stakes scenarios, primarily because the reasoning behind optimization results remains opaque. Existing hyperparameter importance analysis methods are largely based on fANOVA (functional Analysis of Variance), which quantifies hyperparameter influence through variance decomposition. However, fANOVA has notable limitations: it measures the variance induced by varying hyperparameters rather than the performance gain from tuning, and is highly sensitive to the choice of configuration space and probability distribution.

The core idea of HyperSHAP is to formalize the HPO explanation problem as a cooperative game-theoretic problem, using Shapley values to attribute each hyperparameter's contribution to performance improvement, and Shapley interactions to reveal synergistic and redundant relationships among hyperparameters. Compared to fANOVA, HyperSHAP directly attributes performance contributions rather than decomposing variance, making it more actionable.

## Method

### Overall Architecture

HyperSHAP defines five explanation games covering four HPO analysis dimensions: Ablation Analysis, Sensitivity Analysis, Tunability Analysis, and Optimizer Bias Analysis. Each game defines a set function $\nu: 2^{\mathcal{N}} \to \mathbb{R}$ (where $\mathcal{N}$ is the set of hyperparameters), which is then additively decomposed via Shapley values (SV) and Shapley interactions (SI) to obtain importance scores and interaction effects for each hyperparameter.

### Key Designs

1. **Ablation Game**:

    - Function: Quantifies the contribution of each hyperparameter to the performance change from a baseline configuration $\bm{\lambda}^0$ (e.g., library defaults) to a target configuration $\bm{\lambda}^*$ (e.g., HPO result).
    - Mechanism: For a subset $S \subseteq \mathcal{N}$, an intermediate configuration $\bm{\lambda}^* \oplus_S \bm{\lambda}^0$ is constructed (hyperparameters in $S$ take target values; others take default values), and $\nu_{G_A}(S) = \text{Val}_u(\bm{\lambda}^* \oplus_S \bm{\lambda}^0, D)$ is evaluated. This is equivalent to baseline-filling in XAI.
    - Design Motivation: Traditional ablation studies (e.g., Fawcett et al.) replace hyperparameters one at a time along a single path, ignoring interactions. HyperSHAP enumerates all subset combinations and fairly captures interaction effects via Shapley values.

2. **Sensitivity Game**:

    - Function: Quantifies the performance variance induced by each hyperparameter when sampling over the configuration space $\bm{\Lambda}$.
    - Mechanism: $\nu_{G_V}(S) = \mathbb{V}_{\bm{\lambda} \sim p^*}[\text{Val}_u(\bm{\lambda} \oplus_S \bm{\lambda}^0, D)]$, i.e., the performance variance produced by varying only the hyperparameters in subset $S$. fANOVA implicitly relies on this game.
    - Design Motivation: Establishes a theoretical connection with fANOVA and reveals that the fANOVA decomposition corresponds to the Möbius interaction of the sensitivity game.

3. **Tunability Game**:

    - Function: Directly measures the maximum performance gain achievable by tuning the hyperparameters in subset $S$.
    - Mechanism: $\nu_{G_T}(S) = \max_{\bm{\lambda} \in \bm{\Lambda}} \text{Val}_u(\bm{\lambda} \oplus_S \bm{\lambda}^0, D)$, taking the optimal value rather than the variance. This game satisfies **monotonicity** ($S \subseteq T \Rightarrow \nu(S) \leq \nu(T)$), ensuring non-negative Shapley values and non-negative main effects.
    - Design Motivation: The variance decomposition of the sensitivity game is highly dependent on the distribution $p^*$ and the size of the configuration space (hyperparameters with larger domains receive lower scores), and does not distinguish between beneficial and detrimental changes. The tunability game directly quantifies tuning gain and better aligns with practical needs.

4. **Optimizer Bias Game**:

    - Function: Reveals systematic deficiencies of a specific HPO optimizer with respect to particular hyperparameters.
    - Mechanism: $\nu_{G_O}(S) = \text{Val}_u(\mathcal{O}(D, \bm{\Lambda}^S), D) - \nu_{G_T}(S)$, computing the gap between the optimizer's returned result and the theoretical optimum.
    - Design Motivation: Helps HPO researchers diagnose optimizer weaknesses, such as whether certain hyperparameter interactions are being neglected.

5. **Multi-Dataset Extension**:

    - Single-dataset games are extended to cross-dataset games via an aggregation operator (mean or quantile): $\nu_G^{\mathcal{D}}(S) = \bigoplus_{i=1}^M \nu_G^{D_i}(S)$.
    - Reveals which hyperparameters are generally worth tuning across multiple tasks.

### Approximation & Computation

- The $\max_{\bm{\lambda}}$ in the tunability game is approximated using a surrogate model (e.g., the surrogate from Bayesian optimization), with theoretical guarantees on the error: when the surrogate error is $\epsilon$, the Shapley value approximation error is bounded by $2\epsilon$.
- Optimizer bias analysis uses a virtual best optimizer (an ensemble of multiple optimizers and random search) to approximate the true optimum.
- All coalition evaluations are mutually independent and highly parallelizable.

## Key Experimental Results

### Main Results: Tunability vs. Sensitivity

| Game Type | Baseline $\bm{\lambda}^0 = (0,0)$ | Baseline $\bm{\lambda}^0 = \bm{\lambda}^*$ | Characteristics |
|---------|------|------|------|
| Sensitivity $\lambda_1$ | 1/4 | 1/4 | Invariant to baseline |
| Sensitivity $\lambda_2$ | $m/(m+1)^2$ | $m/(m+1)^2$ | Lower score for larger domain |
| Tunability $\lambda_1$ | 1 | 0 | Correctly reflects contribution |
| Tunability $\lambda_2$ | 1 | 0 | Equal to $\lambda_1$ |

In an illustrative example where two hyperparameters contribute equally to optimal performance, the tunability game correctly assigns equal weights, while the sensitivity game produces unfair scores due to differences in domain size.

### Ablation Study: HPO Subset Selection Task

| Method | Dataset 1 Performance | Dataset 2 Performance | Note |
|------|-------------|-------------|------|
| fANOVA (top-2 hyperparameters selected for tuning) | Lower | Lower | Variance decomposition does not directly correspond to tuning gain |
| HyperSHAP-Sensitivity | Medium | Medium | Sensitivity game |
| HyperSHAP-Tunability | **Highest** | **Highest** | Tunability game best suited for subset selection |

On the LCBench benchmark, subsequent HPO using the top-2 hyperparameters selected by HyperSHAP (tunability game) consistently outperforms subsets selected by fANOVA in terms of anytime performance.

### Key Findings

- For most HPO problems, explanatory power approximates the full game ($R^2 \approx 1$) at interaction order $k=3$, confirming that hyperparameter interactions are typically low-order.
- Optimizer bias analysis accurately detects artificially constructed biases: optimizers that tune each hyperparameter independently are detected as missing interactions, and optimizers that are forbidden from tuning a specific hyperparameter are detected as exhibiting a significant negative main effect for that parameter.
- By explaining SMAC's surrogate model, the dynamic evolution of the model's beliefs about hyperparameter importance during optimization can be observed.
- Runtime analysis: ablation games take 5 seconds to 2 minutes; tunability games take 6–15 minutes (for 4–7 hyperparameters), representing modest overhead relative to HPO runs that themselves take several hours.

## Highlights & Insights

- The paper systematically imports mature Shapley value theory from XAI into HPO explanation, establishing a complete theoretical framework.
- The monotonicity property of the tunability game guarantees non-negative Shapley values, making explanations more intuitive—"tuning this hyperparameter will at least not hurt performance."
- The theoretical contrast between sensitivity and tunability is highly instructive: the former measures variability while the latter measures optimization potential, and each has value in different scenarios.
- The optimizer bias analysis offers a novel perspective for auditing the capabilities of HPO methods, and can be used to guide their improvement.

## Limitations & Future Work

- The computational bottleneck lies in the tunability game, which requires simulating HPO for each coalition; with 10 hyperparameters, this can take up to 8.5 hours.
- Current approximation of optimal values relies solely on surrogate models; more efficient and unbiased approximation methods remain to be developed.
- The framework has not been extended to the explanation of machine learning pipeline optimization.
- Future work could leverage hyperparameter importance for warm-starting HPO across datasets.

## Related Work & Insights

- fANOVA is the most prevalent prior tool for hyperparameter importance analysis; HyperSHAP establishes a theoretical bridge to it via the sensitivity game while identifying its limitations.
- XAI methods such as SHAP (Lundberg & Lee, 2017) apply Shapley values to feature attribution; HyperSHAP transfers the same paradigm from "explaining model predictions" to "explaining hyperparameter optimization."
- The shapiq library provides an efficient implementation of Shapley interactions and serves as the computational foundation for the proposed method.
- This framework can inspire the future development of "interaction-aware" HPO methods—given that most interactions are low-order, more efficient search strategies may be designable.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/others/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](../../ACL2025/others/using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](../../ICML2025/others/prediction_via_shapley_value_regression.md)
- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)

</div>

<!-- RELATED:END -->
