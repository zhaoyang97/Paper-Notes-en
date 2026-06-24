---
title: >-
  [Paper Note] PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression
description: >-
  [ICLR 2026][Interpretability][Shapley Values] This paper proposes PolySHAP, which improves the estimation accuracy of Shapley values by extending the linear approximation of KernelSHAP to higher-order polynomial regression to capture non-linear feature interactions. It theoretically proves that paired sampling is equivalent to second-order PolySHAP, providing the first explanation for the superior performance of the paired sampling heuristic.
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Shapley Values"
  - "Explainable AI"
  - "Polynomial Regression"
  - "Feature Interactions"
  - "KernelSHAP"
date: 2026-05-08
content_hash: 0d50f5e11b2cfd93
---

# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

**Conference**: ICLR 2026  
**arXiv**: [2601.18608](https://arxiv.org/abs/2601.18608)  
**Code**: [GitHub](https://github.com/FFmgll/PolySHAP)  
**Area**: Explainability  
**Keywords**: Shapley Values, Explainable AI, Polynomial Regression, Feature Interactions, KernelSHAP

## TL;DR

This paper proposes PolySHAP, which improves the estimation accuracy of Shapley values by extending the linear approximation of KernelSHAP to higher-order polynomial regression to capture non-linear feature interactions. It theoretically proves that paired sampling is equivalent to second-order PolySHAP, providing the first explanation for the superior performance of the paired sampling heuristic.

## Background & Motivation

Shapley values are a central game-theoretical tool in explainable AI (XAI) for quantifying feature contributions to model predictions. However, for a model with $d$ features, exact calculation requires $2^d$ game evaluations, which is computationally prohibitive. KernelSHAP avoids exponential costs by approximating the game function $\nu$ as a linear function, but linear approximation inherently fails to capture non-linear interaction effects between features, limiting estimation accuracy.

Furthermore, paired sampling is a widely used heuristic that significantly improves KernelSHAP estimation quality, though the theoretical mechanism behind its performance has not been fully understood. This paper provides a unified theoretical framework and practical solution for these two issues from the perspective of polynomial regression.

## Method

### Overall Architecture

PolySHAP replaces the linear game approximation of KernelSHAP with a higher-order polynomial containing interaction terms. It first specifies an interaction frontier $\mathcal{I}$ (the set of feature combinations to be explicitly modeled), fits polynomial coefficients via weighted least squares, and then maps these coefficients back to efficiency-satisfying Shapley values using a closed-form formula. The method maintains the sampling-regression framework of KernelSHAP while extending the regression feature space from single features to interaction terms.

### Key Designs

**1. PolySHAP Interaction Representation: Replacing Linear with Polynomial Regression**

The bottleneck of KernelSHAP is its use of a linear function to fit the game $\nu$, discarding all feature interactions and capturing only "independent" contributions. PolySHAP extends the regression target to a polynomial $\phi^{\mathcal{I}} \in \mathbb{R}^{d'}$ (where $d' = d + |\mathcal{I}|$). It retains $d$ single-feature terms and explicitly incorporates feature combination terms specified in the interaction frontier $\mathcal{I}$, performing a constrained weighted least squares fit:

$$\phi^{\mathcal{I}}[\nu] := \arg\min_{\phi \in \mathbb{R}^{d'}: \langle\phi,\mathbf{1}\rangle = \nu(D)} \sum_{S \subseteq D} \mu(S)\left(\nu(S) - \sum_{T \in D \cup \mathcal{I}} \phi_T \prod_{j \in T} \mathbb{1}[j \in S]\right)^2$$

The fitted coefficients are not directly the Shapley values; the contribution of interaction terms $\phi_S^{\mathcal{I}}$ is blended. Following Theorem 4.3, these are distributed back to each participating feature: $\phi_i^{SV}[\nu] = \phi_i^{\mathcal{I}} + \sum_{S \in \mathcal{I}: i \in S} \frac{\phi_S^{\mathcal{I}}}{|S|}$. Since polynomials have higher expressive power than linear models and fit $\nu$ more accurately, the resulting Shapley estimates have lower error.

**2. Paired Sampling Equivalence Theorem: Explaining a Long-standing Heuristic**

Paired sampling (sampling a subset $S$ and its complement $D \setminus S$ simultaneously) has been an empirical trick for enhancing KernelSHAP. Theorem 5.1 provides the answer: under paired sampling, the output of KernelSHAP is **exactly equal** to 2-PolySHAP. This means "pairing" implicitly incorporates all second-order interactions into the estimate, equivalent to performing second-order polynomial regression. This theoretically identifies the source of paired sampling gains and implies that if paired sampling is already used, additional gains from PolySHAP only emerge starting from third-order interactions.

**3. $k$-additive Interaction Frontier: Adding Terms by Budget**

Interaction terms cannot be added without limit, as the number of $k$-th order interactions explodes as $\binom{d}{k}$. PolySHAP uses the interaction frontier $\mathcal{I}$ to define which combinations to model explicitly. It defines a $k$-additive frontier $\mathcal{I}_{\leq k} = \{S \subseteq D : 2 \leq |S| \leq k\}$, corresponding to $k$-PolySHAP. For $k=1$, it reduces to KernelSHAP; $k=2$ includes all second-order interactions. To handle high-dimensional cases, the paper introduces a **partial interaction frontier** $\mathcal{I}_\ell$, selectively including higher-order terms when the budget cannot support the full $k$-th order.

**4. Leverage Score Sampling: Guaranteeing Approximations in Expanded Spaces**

As the feature space grows from $d$ to $d'$, using standard Shapley weights for sampling can lead to poor condition numbers for the regression matrix. PolySHAP employs leverage score sampling, where sampling probabilities are determined by the influence of each subset on the least squares solution. It is proven that with a subset budget $m = O(d' \log(d'/\delta) + d'/({\epsilon\delta}))$, an $\epsilon$-level approximation quality can be guaranteed with probability $1-\delta$.

### Loss & Training

A constrained weighted least squares problem is solved, where the constraint $\langle\phi,\mathbf{1}\rangle = \nu(D)$ corresponds to the efficiency property of Shapley values. Implementation uses a projection matrix $\mathbf{P}_{d'}$ to convert the constrained problem into an unconstrained closed-form solution. A "border trick" is used to enumerate very small/large subsets directly rather than sampling, reducing estimation variance on small subsets.

## Key Experimental Results

### Main Results

Evaluations across 15 different attribution games (tabular, image, language; $d$ from 8 to 101) compare PolySHAP against several baselines.

| Dataset/Game | Metric | PolySHAP (3rd order) | KernelSHAP | Gain |
|-------------|------|---------------|------------|------|
| Housing ($d=8$) | MSE | Best | Baseline | Significant reduction in MSE |
| Adult ($d=14$) | MSE | Best | Baseline | Significant reduction in MSE |
| Estate ($d=15$) | MSE | Best | Baseline | Significant reduction in MSE |
| Cancer ($d=30$) | MSE | Best | Baseline | Significant reduction in MSE |
| CG60 ($d=60$) | MSE | Slight Improvement | Baseline | Small gain (High-dim limit) |

### Ablation Study

| Configuration | Key Metric (MSE) | Description |
|------|---------------|------|
| 1-PolySHAP (= KernelSHAP) | Baseline | No interaction terms |
| 2-PolySHAP | Significant Improvement | All 2nd-order interactions included |
| 2-PolySHAP (50%) | Medium Improvement | 50% of 2nd-order interactions included |
| 3-PolySHAP | Best | 3rd-order added; highest gain in low-dim |
| Paired KernelSHAP vs Paired 2-PolySHAP | Identical | Empirical validation of Theorem 5.1 |
| Paired 3-PolySHAP vs Paired 4-PolySHAP | Almost Identical | Suggests higher-order equivalence relations |

### Key Findings

- Inclusion of any number of interaction terms improves Shapley value approximation quality.
- Under paired sampling, KernelSHAP automatically achieves 2-PolySHAP performance; thus, gains from PolySHAP in practice begin with third-order interactions.
- In high-dimensional scenarios ($d \geq 60$), the number of third-order terms is limited, leading to smaller gains.
- RegressionMSR is the only baseline comparable to PolySHAP but relies on XGBoost and shows instability in certain games.

## Highlights & Insights

- **Major Theoretical Contribution**: The discovery that paired sampling is equivalent to 2-PolySHAP is an elegant result that resolves long-standing practical confusion.
- **Natural and Elegant Method**: The transition from linear to polynomial regression is intuitive and maintains consistency guarantees.
- **Unified Perspective**: Incorporates methods like KernelSHAP, Faith-SHAP, and $k_{ADD}$-SHAP into a single framework.
- **Projection Lemma**: The technical projection lemma (Lemma A.1) plays a critical role in proving multiple theorems.

## Limitations & Future Work

- Combinatorial explosion ($\binom{d}{3}$) in high-dimensional settings limits the number of inclusion terms.
- Conjecture that paired $k$-PolySHAP is equivalent to $(k+1)$-PolySHAP for odd $k$ remains unproven.
- Selection of the interaction frontier is handled generally; it does not yet utilize problem-specific interaction structures.
- Practical efficiency in large-scale applications requires further validation beyond theoretical time analysis.

## Related Work & Insights

- Comparison with RegressionMSR (Witter et al., 2025): PolySHAP maintains consistency without extra regression adjustment steps.
- Relationship with $k_{ADD}$-SHAP (Pelegrina et al., 2023): PolySHAP simplifies and generalizes its convergence proofs.
- Insight: Future work could combine interaction detection methods (e.g., Tsang et al., 2020) or graph structures to build smarter interaction frontiers.

## Rating

- Novelty: ⭐⭐⭐⭐ The polynomial extension is natural; the paired sampling equivalence theorem is the highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 games across tabular/image/language with comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations, intuitive charts, and smooth narrative.
- Value: ⭐⭐⭐⭐ Substantially advances Shapley estimation in XAI; deep significance in explaining paired sampling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Joint Distribution–Informed Shapley Values for Sparse Counterfactual Explanations](joint_distributioninformed_shapley_values_for_sparse_counterfactual_explanations.md)
- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](../../ICML2026/interpretability/a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)

</div>

<!-- RELATED:END -->
