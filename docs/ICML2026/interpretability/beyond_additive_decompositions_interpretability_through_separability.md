---
title: >-
  [Paper Note] Beyond Additive Decompositions: Interpretability Through Separability
description: >-
  [ICML2026][Interpretability][Interpretable Machine Learning] Tensor Separable Learning (TSL) is proposed, a stage-wise greedy regression method that models the conditional mean as a difference of positive rank-1 separabl…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Interpretable Machine Learning"
  - "Tensor Separable Learning"
  - "Separable Models"
  - "Partial Dependence Analysis"
  - "Glass-box Models"
date: 2026-05-08
content_hash: 07f65f507ce12c10
---

# Beyond Additive Decompositions: Interpretability Through Separability

**Conference**: ICML2026  
**arXiv**: [2605.31200](https://arxiv.org/abs/2605.31200)  
**Code**: https://github.com/jyliuu/TSL  
**Area**: Interpretability  
**Keywords**: Interpretable Machine Learning, Tensor Separable Learning, Separable Models, Partial Dependence Analysis, Glass-box Models  

## TL;DR

Tensor Separable Learning (TSL) is proposed, a stage-wise greedy regression method that models the conditional mean as a difference of positive rank-1 separable products. By utilizing separable structures, it avoids signal cancellation and interaction masking issues inherent in additive decompositions under strong interactions, while its partial dependence functions accurately recover the shapes of fitting factors.

## Background & Motivation

**Background**: Mainstream interpretable machine learning relies on additive decomposition to provide readable model structures. Post-hoc explanation methods like SHAP decompose predictions into a sum of feature contributions; glass-box models like GAM restrict predictions to a sum of univariate functions; functional ANOVA provides a global additive perspective.

**Limitations of Prior Work**: Additive decomposition faces three fundamental failure modes in the presence of strong interactions: (1) Main effects and interaction effects in SHAP local attributions may cancel each other out, resulting in $\phi_k(\mathbf{x})=0$ even when feature $k$ is highly active; (2) Partial Dependence (PD) plots only recover the main effect $\text{PD}_1(x_1) = m_\emptyset + m_1(x_1)$ and are completely blind to high-order interactions; (3) When features are correlated, marginal averaging produces extrapolation artifacts in low-density regions.

**Key Challenge**: Additive decomposition requires "projecting" high-dimensional interactions onto low-dimensional additive components, a process that inevitably loses interaction information. To maintain interpretability, one must sacrifice interaction modeling capability, which constitutes a structural contradiction between accuracy and interpretability.

**Goal**: Design a glass-box regression method that both captures arbitrary-order interactions and allows for accurate reconstruction of the model structure from one-dimensional partial dependence curves.

**Key Insight**: For a separable product $h(\mathbf{x}) = \prod_j h_j(x_j)$, its partial dependence function satisfies $\text{PD}_j[h](x_j) = c_j h_j(x_j)$. That is, the 1D PD exactly recovers the shape of the factor rather than degrading into a main effect. This property makes separable structures a natural "interaction-friendly" choice.

**Core Idea**: Replace additive decomposition with a stage-wise superposition of differences of positive rank-1 separable products, enabling the model to natively preserve interaction structures and be accurately reconstructed from 1D PD.

## Method

### Overall Architecture

TSL models the conditional mean $m(\mathbf{x}) = \mathbb{E}[Y|X=\mathbf{x}]$ as a superposition of $R$ stages, where each stage is the difference between two positive separable products. The input is $p$-dimensional features $\mathbf{x}$, and the output is a scalar regression prediction. Training adopts a stage-wise greedy strategy: each stage fits the current residual, followed by an orthogonal refitting of coefficients for all existing stages once fitting is complete.

### Key Designs

1.  **Positive Separable Product Difference Structure**:

    - **Function**: Construct a regression model that captures arbitrary interactions and can be accurately reconstructed from 1D PD.
    - **Mechanism**: The model form is $\hat{m}(\mathbf{x}) = \sum_{\ell=1}^{R}(\lambda_+^{(\ell)}\prod_j \hat{m}_{+,j}^{(\ell)}(x_j) - \lambda_-^{(\ell)}\prod_j \hat{m}_{-,j}^{(\ell)}(x_j))$, where each component function $\hat{m}_{\pm,j}^{(\ell)} > 0$ is strictly positive. Positivity constraints eliminate sign ambiguity: in unconstrained products, a positive component does not guarantee a positive contribution to the product (as other components might be negative), whereas positivity ensures "increasing means amplifying." The difference structure (difference between two positive products) restores the ability to express negative values.
    - **Design Motivation**: Positivity also addresses identifiability issues in bagging aggregation—unconstrained products can have multiple equivalent representations (sign flips) on non-rectangular supports, whereas positivity eliminates sign ambiguity, allowing independently fitted components to be safely averaged.

2.  **Backbone/Tilt Reparameterization**:

    - **Function**: Decouple the activation level of each stage from its directionality, providing an intuitive interpretable perspective.
    - **Mechanism**: Reparameterize the positive and negative component pairs as $\hat{m}_{\pm,j}^{(\ell)}(x_j) = b_j^{(\ell)}(x_j) e^{\pm d_j^{(\ell)}(x_j)}$, where the backbone $b_j^{(\ell)} > 0$ encodes shared magnitude and the tilt $d_j^{(\ell)} \in \mathbb{R}$ encodes sign imbalance. The entire model becomes $\hat{m}(\mathbf{x}) = 2\sum_\ell b^{(\ell)}(\mathbf{x}) \sinh(d^{(\ell)}(\mathbf{x}))$. The backbone product acts as an active gate (if a feature's backbone is near zero, the stage is closed), and the sum of tilts determines the direction via $\sinh$.
    - **Design Motivation**: The backbone retains magnitude information even when the signed partial dependence is near zero, avoiding interaction masking; the tilt acts as an additive imbalance score with a clear directional interpretation.

3.  **Stepwise Refinement of Grid Tensors and Bagging Aggregation**:

    - **Function**: Efficiently fit the separable products of each stage and reduce variance.
    - **Mechanism**: Each univariate component is modeled as a piecewise constant function, with partitions refined step-by-step through CART-style greedy splits. The evaluation objective for each split is regularized least squares $\mathcal{L}_S(u_+^S, u_-^S) = \sum w_i(R_i - (u_+^S \hat{m}_+^{(i)} - u_-^S \hat{m}_-^{(i)}))^2 + \alpha((u_+^S-1)^2 + (u_-^S-1)^2)$, which has a closed-form $2\times2$ linear system solution. Multiple grid tensors are fitted in parallel on bootstrap samples, followed by aggregation in the backbone/tilt space through normalization → anchoring → similarity filtering → averaging.
    - **Design Motivation**: Bagging reduces variance, and similarity filtering resolves identifiability ambiguities of separable models on non-rectangular supports.

## Key Experimental Results

### Main Results

Evaluated on 27 datasets from the OpenML CTR 23 regression benchmark, compared against EBM, SepALS, XGBoost, LightGBM, and Random Forest:

| Dataset | TSL ($R\le10$) | EBM | SepALS ($r\le10$) | XGBoost (Black-box) | Best Group |
| :--- | :--- | :--- | :--- | :--- | :--- |
| brazilian_houses | **2398.68** | 3327.29 | 3996.05 | 4289.10 | TSL |
| auction_verification | **624.36** | 1738.17 | 682.20 | 369.34 | TSL (Best Interpretable) |
| socmob | **9.48** | 20.21 | 22.88 | 17.65 | TSL |
| california_housing | 49376.09 | **48866.28** | 62162.22 | 44971.31 | EBM (Best Interpretable) |
| cpu_activity | **2.3076** | 2.3546 | 2.8475 | 2.1945 | TSL |
| miami_housing | **89692.96** | 91777.25 | 99426.86 | 82325.09 | TSL |

Among the 27 datasets, TSL ($R\le2$ or $R\le10$) ranked in the top three of the interpretable group in 17 cases and was the best interpretable model in 5.

### Ablation Study

| Configuration | socmob | naval_prop. | auction_ver. | Description |
| :--- | :--- | :--- | :--- | :--- |
| TSL ($R\le2$, diff. pos. products) | 9.87 | 0.0013 | 1135.80 | Full model |
| TSL (1-product, no positivity) | 10.58 | 0.0027 | 1336.51 | Significant degradation after removing positivity + difference |
| SepALS ($r\le2$) | 7.73 | 0.0004 | 997.08 | SepALS is better on smooth data |

Under matched total separation rank ($\le4$), the positivity constraint + difference structure brought significant Gains (socmob: 10.58 → 9.87, naval: 0.0027 → 0.0013).

### Key Findings

- TSL shows the greatest advantage on data with low-rank separable structures (e.g., socmob has a known log-additive structure, TSL RMSE 9.48 vs EBM 20.21).
- SepALS may perform better on smooth data (e.g., naval_propulsion_plant) but over-smooths on sharp features; TSL's adaptive grid splitting captures more abrupt patterns.
- In the California Housing interpretability case study, TSL's two-stage model clearly demonstrates separable spatial gating mechanisms like "coastal premium" (Stage 1) and "inland desert correction" (Stage 2).
- Synthetic experiments validated the mitigation of interaction masking: when $\mathbb{E}[1+X_3]=0$, the 1D PD of all methods is zero, but TSL's backbone still preserves the magnitude of the quadratic effect of $x_1$.

## Highlights & Insights

- **Partial Dependence Accuracy of Separable Structure**: For separable products, partial dependence functions exactly recover factor shapes (rather than degrading to main effects). This is the core theoretical insight—replacing additive structures with multiplicative ones allows for simultaneous interaction modeling and fidelity in 1D visualization.
- **Backbone/Tilt Separation**: Decoupling "where the model is active" (backbone gating) from "which direction the model moves" (tilt direction) provides a brand-new perspective on interpretability. This idea can be migrated to any scenario requiring the separation of magnitude and directional information.
- **Triple Role of Positivity Constraint**: A simple positivity constraint simultaneously addresses three issues (sign ambiguity, bagging stability, and interpretable directionality), making it an elegant "three birds with one stone" design.

## Limitations & Future Work

- Theoretical guarantees only cover an approximation rate of $O(1/\sqrt{r})$, with no finite-sample learning rate or consistency guarantees provided.
- Non-identifiability of separable representations remains a core limitation: on non-rectangular supports, different factorizations can yield identical predictions, and bagging aggregation might introduce high variance.
- On problems dominated by additive structures, EBM remains a stronger interpretable baseline (e.g., QSAR_fish_toxicity, red_wine).
- Currently, univariate components in backbone/tilt are modeled with piecewise constant functions; future work could replace these with more flexible parameterizations like neural networks or splines.
- The aggregation strategy currently discards bagged grids that do not pass similarity filtering; this could be improved by aligning all bags through Riemannian optimization on the positive rank-1 manifold.

## Related Work & Insights

- Comparison with GAM/EBM: GAM and its extensions (GA²M, NAM, NODE-GAM) are essentially additive decompositions. TSL replaces additive separation with multiplicative separation, representing an orthogonal modeling paradigm.
- Comparison with Classical Tensor Decomposition (CP/PARAFAC): They share separable forms but differ in objectives—classical methods recover latent factors, while TSL learns interpretable supervised predictors.
- Comparison with SepALS: TSL's stage-wise residual fitting replaces fixed-rank joint optimization, and its positive difference structure replaces unconstrained products.
- The $O(1/\sqrt{r})$ approximation rate provided by the OGA framework is dimension-independent (though the target class tightens with dimension), providing a powerful tool for the theoretical analysis of separable models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](../../ACL2026/interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->
