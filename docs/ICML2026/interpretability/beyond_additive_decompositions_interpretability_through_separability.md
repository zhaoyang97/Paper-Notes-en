---
title: >-
  [Paper Note] Beyond Additive Decompositions: Interpretability Through Separability
description: >-
  [ICML2026][Interpretability][Interpretable Machine Learning] Ours proposes Tensor Separable Learning (TSL), a stagewise greedy regression method that models the conditional mean as the difference between positive rank-1 separable products. By utilizing a separable structure, it avoids signal cancellation and interaction masking issues inherent in additive decompositions under strong interactions, while its partial dependence functions can precisely recover the shapes of the f…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Interpretable Machine Learning"
  - "Tensor Separable Learning"
  - "Separable Models"
  - "Partial Dependence Analysis"
  - "Glass-box Models"
date: 2026-05-08
content_hash: d026516b54ee4fbe
---

# Beyond Additive Decompositions: Interpretability Through Separability

**Conference**: ICML2026  
**arXiv**: [2605.31200](https://arxiv.org/abs/2605.31200)  
**Code**: https://github.com/jyliuu/TSL  
**Area**: Interpretability  
**Keywords**: Interpretable Machine Learning, Tensor Separable Learning, Separable Models, Partial Dependence Analysis, Glass-box Models

## TL;DR

Ours proposes Tensor Separable Learning (TSL), a stagewise greedy regression method that models the conditional mean as the difference between positive rank-1 separable products. By utilizing a separable structure, it avoids signal cancellation and interaction masking issues inherent in additive decompositions under strong interactions, while its partial dependence functions can precisely recover the shapes of the fitting factors.

## Background & Motivation

**Background**: Mainstream interpretable machine learning relies on additive decomposition to provide readable model structures. Post-hoc explanation methods like SHAP decompose predictions into a sum of feature contributions; glass-box models like GAMs restrict predictions to a sum of univariate functions; and functional ANOVA provides a global additive perspective.

**Limitations of Prior Work**: Additive decomposition faces three fundamental failure modes when strong interactions exist: (1) In SHAP's local attribution, main effects and interaction effects may cancel each other out, leading to $\phi_k(\mathbf{x})=0$ even when feature $k$ is highly active; (2) Partial Dependence (PD) plots only recover main effects $\text{PD}_1(x_1) = m_\emptyset + m_1(x_1)$, remaining entirely blind to high-order interactions; (3) When features are correlated, marginal averaging produces extrapolation artifacts in low-density regions.

**Key Challenge**: Additive decomposition requires "projecting" high-dimensional interactions onto low-dimensional additive components, a process that inevitably loses interaction information. To maintain interpretability, one must sacrifice interaction modeling capability—a structural contradiction between accuracy and interpretability.

**Goal**: Design a glass-box regression method that both captures arbitrary-order interactions and allows for the precise reconstruction of model structures from one-dimensional partial dependence curves.

**Key Insight**: For a separable product $h(\mathbf{x}) = \prod_j h_j(x_j)$, its partial dependence function satisfies $\text{PD}_j[h](x_j) = c_j h_j(x_j)$. That is, the 1D PD precisely recovers the factor shape rather than degenerating into a main effect. This property makes the separable structure a natural choice for being "interaction-friendly."

**Core Idea**: Replace additive decomposition with a stagewise superposition of differences of positive rank-1 separable products, allowing the model to natively preserve interaction structures and be precisely reconstructed from 1D PDs.

## Method

### Overall Architecture

TSL addresses the contradiction between "modeling arbitrary-order interactions" and "ensuring 1D partial dependence curves are faithfully readable." It expresses the conditional mean $m(\mathbf{x}) = \mathbb{E}[Y|X=\mathbf{x}]$ as a superposition of $R$ stages. Each stage is not a sum of univariate functions (as in additive decomposition) but the difference between two "positive separable products." For training, a stagewise greedy approach is used: each stage fits the residuals left by the previous stage, followed by an orthogonal refitting of the coefficients of all existing stages to prevent inter-stage interference.

### Key Designs

**1. Difference of Positive Separable Products: Replacing Additive Separation with Multiplicative Separation**

The blindness of additive decomposition to strong interactions stems from its forced projection of high-dimensional interactions onto low-dimensional additive components, leading to information loss. TSL adopts a product structure $\hat{m}(\mathbf{x}) = \sum_{\ell=1}^{R}\big(\lambda_+^{(\ell)}\prod_j \hat{m}_{+,j}^{(\ell)}(x_j) - \lambda_-^{(\ell)}\prod_j \hat{m}_{-,j}^{(\ell)}(x_j)\big)$, where each univariate component $\hat{m}_{\pm,j}^{(\ell)} > 0$ is constrained to be strictly positive. Products naturally express interactions, while the positivity constraint eliminates sign ambiguity—in unconstrained products, a positive component does not guarantee a positive contribution to the whole product (as other components might flip signs). Positivity ensures that "increasing a component necessarily amplifies the product." Since pure positive products cannot express negative effects, a difference structure of "subtracting two positive products" is used to restore the ability to represent negative values. This positivity constraint also solves the identifiability issue for subsequent bagging: unconstrained products have multiple equivalent representations via sign flips on non-rectangular supports, whereas positivity pins them to a unique form, allowing independently fitted components to be safely averaged.

**2. Backbone/Tilt Reparameterization: Decoupling "Activity" and "Direction"**

To make the positive/negative component pairs more intuitive, TSL reparameterizes each pair as $\hat{m}_{\pm,j}^{(\ell)}(x_j) = b_j^{(\ell)}(x_j)\, e^{\pm d_j^{(\ell)}(x_j)}$. The backbone $b_j^{(\ell)} > 0$ encodes the magnitude shared by the positive and negative branches, while the tilt $d_j^{(\ell)} \in \mathbb{R}$ encodes the imbalance between them. Substituting this into the model yields $\hat{m}(\mathbf{x}) = 2\sum_\ell b^{(\ell)}(\mathbf{x})\, \sinh\!\big(d^{(\ell)}(\mathbf{x})\big)$. Consequently, the backbone product acts as an "activity gate" (if a feature's backbone approaches zero, the entire stage is turned off), and the sum of tilts determines the output direction through the $\sinh$ function. This separation specifically addresses the "interaction masking" of additive decomposition—even if the signed partial dependence happens to cancel to zero, the backbone still retains the magnitude information of the feature; meanwhile, the tilt, as an additive imbalance score, provides a clear directional interpretation.

**3. Grid Tensor Refinement and Bagging Aggregation: Fitting Components and Reducing Variance**

Each univariate component is represented as a piecewise constant function, using CART-style greedy splitting to refine partitions. Each candidate split is scored using regularized least squares $\mathcal{L}_S(u_+^S, u_-^S) = \sum_i w_i\big(R_i - (u_+^S \hat{m}_+^{(i)} - u_-^S \hat{m}_-^{(i)})\big)^2 + \alpha\big((u_+^S-1)^2 + (u_-^S-1)^2\big)$. This objective has a closed-form $2\times2$ linear system solution for $u_+^S, u_-^S$, making refinement rapid. To reduce variance, multiple grid tensors are fitted in parallel on bootstrap samples. They are then aggregated in the backbone/tilt space via "Normalization → Anchoring → Similarity Filtering → Averaging." Similarity filtering discards bags that are inconsistent due to the non-identifiability of separability, and the remaining consistent bags are averaged. Thus, bagging reduces variance without being contaminated by ambiguous solutions.

## Key Experimental Results

### Main Results

Evaluation was conducted on 27 datasets from the OpenML CTR 23 regression benchmark, comparing with EBM, SepALS, XGBoost, LightGBM, and Random Forest:

| Dataset | TSL (R≤10) | EBM | SepALS (r≤10) | XGBoost (Blackbox) | Best Group |
|--------|-----------|-----|---------------|---------------|-------|
| brazilian_houses | **2398.68** | 3327.29 | 3996.05 | 4289.10 | TSL |
| auction_verification | **624.36** | 1738.17 | 682.20 | 369.34 | TSL (Best Interpretable) |
| socmob | **9.48** | 20.21 | 22.88 | 17.65 | TSL |
| california_housing | 49376.09 | **48866.28** | 62162.22 | 44971.31 | EBM (Best Interpretable) |
| cpu_activity | **2.3076** | 2.3546 | 2.8475 | 2.1945 | TSL |
| miami_housing | **89692.96** | 91777.25 | 99426.86 | 82325.09 | TSL |

Among 27 datasets, TSL (R≤2 or R≤10) ranked in the top three for the interpretable group in 17 cases and was the best interpretable model in 5 cases.

### Ablation Study

| Configuration | socmob | naval_prop. | auction_ver. | Description |
|------|--------|-------------|-------------|------|
| TSL (R≤2, Diff. Positive Prod.) | 9.87 | 0.0013 | 1135.80 | Full Model |
| TSL (1-product, No Positivity) | 10.58 | 0.0027 | 1336.51 | Significant degradation without positivity + difference |
| SepALS (r≤2) | 7.73 | 0.0004 | 997.08 | SepALS is better on smooth data |

Under matched total separation rank (≤4), the positivity constraint + difference structure brought significant improvements (socmob: 10.58→9.87, naval: 0.0027→0.0013).

### Key Findings

- TSL performs best on data where signals have a low-rank separable structure (e.g., socmob has a known log-additive structure, TSL RMSE 9.48 vs EBM 20.21).
- SepALS may perform better on smooth data (e.g., naval_propulsion_plant) but over-smooths on sharp features; TSL’s adaptive grid splitting captures more abrupt patterns.
- In a California Housing interpretability case study, TSL's two-stage model clearly demonstrated separable spatial gating mechanisms for "Coastal Premium" (Stage 1) and "Inland Desert Correction" (Stage 2).
- Synthetic experiments validated the interaction masking problem: when $\mathbb{E}[1+X_3]=0$, the 1D PD of all methods becomes zero, but TSL’s backbone still preserves the quadratic effect magnitude of $x_1$.

## Highlights & Insights

- **Partial Dependence Precision of Separable Structures**: For separable products, partial dependence functions precisely recover factor shapes (rather than degenerating into main effects). This is a core theoretical insight—replacing additive structures with multiplicative ones allows for both interaction modeling and faithfulness in 1D visualization.
- **Backbone/Tilt Decoupling**: Decoupling "where the model is active" (backbone gating) from "which direction the model goes" (tilt direction) provides a novel interpretability perspective. This idea can be migrated to any scenario requiring the separation of magnitude and directional information.
- **Multiple Roles of the Positivity Constraint**: A simple positivity constraint simultaneously solves three problems (sign ambiguity, bagging stability, and interpretable directionality), representing an elegant "three birds with one stone" design.

## Limitations & Future Work

- Theoretical guarantees only cover approximation rates of $O(1/\sqrt{r})$ and do not provide finite-sample learning rates or consistency guarantees.
- The non-identifiability of separable representations is a core limitation: on non-rectangular supports, different factorizations can produce the same prediction, and bagging aggregation may introduce high variance.
- On problems dominated by additive structures, EBM remains a stronger interpretable baseline (e.g., QSAR_fish_toxicity, red_wine).
- Currently, univariate components in backbone/tilt are modeled using piecewise constant functions; future work could replace these with more flexible parameterizations like neural networks or splines.
- The aggregation strategy currently discards bagged grids that fail similarity filtering; all bags could potentially be aligned via Riemannian optimization on the positive rank-1 manifold.

## Related Work & Insights

- Comparison with GAM/EBM: GAM and its extensions (GA²M, NAM, NODE-GAM) are essentially additive decompositions. TSL replaces additive separation with multiplicative separation, serving as an orthogonal modeling paradigm.
- Comparison with Classical Tensor Decomposition (CP/PARAFAC): They share a separable form but have different goals—classical methods recover latent factors, while TSL learns an interpretable supervised predictor.
- Comparison with SepALS: TSL’s stagewise residual fitting replaces fixed-rank joint optimization, and its positive difference structure replaces unconstrained products.
- The OGA framework provides an $O(1/\sqrt{r})$ approximation rate independent of dimension (though the target class tightens with dimension), providing a powerful tool for the theoretical analysis of separable models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Circuit Insights: Towards Interpretability Beyond Activations](../../ICLR2026/interpretability/circuit_insights_towards_interpretability_beyond_activations.md)
- [\[CVPR 2026\] Beyond Top Activations: Efficient and Reliable Crowdsourced Evaluation of Automated Interpretability](../../CVPR2026/interpretability/beyond_top_activations_efficient_and_reliable_crowdsourced_evaluation_of_automat.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](../../ACL2026/interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)

</div>

<!-- RELATED:END -->
