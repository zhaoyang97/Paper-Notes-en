---
title: >-
  [Paper Note] TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models
description: >-
  [AAAI 2026][local attribution] Under the Taylor expansion framework, three postulates—precision, federation, and zero-discrepancy—are proposed to regulate feature attribution. An adaptation property is further introduced…
tags:
  - "AAAI 2026"
  - "local attribution"
  - "Taylor expansion"
  - "post-hoc explainability"
  - "feature interaction"
  - "AUP optimization"
date: 2026-05-08
content_hash: 75d69e51fc426a80
---

# TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models

**Conference**: AAAI 2026
**arXiv**: [2507.10643](https://arxiv.org/abs/2507.10643)
**Code**: Provided in Appendix
**Area**: Other
**Keywords**: local attribution, Taylor expansion, post-hoc explainability, feature interaction, AUP optimization

## TL;DR
Under the Taylor expansion framework, three postulates—precision, federation, and zero-discrepancy—are proposed to regulate feature attribution. An adaptation property is further introduced to optimize the allocation weights of interaction effects via an AUP objective, making TaylorPODA the only post-hoc, model-agnostic attribution method that simultaneously satisfies all postulates and properties.

## Background & Motivation
**Background**: In post-hoc explainable AI (XAI), local attribution is the dominant paradigm, with LIME and SHAP being widely adopted. Deng et al. (2024) proposed a unified analytical framework for various attribution methods based on Taylor expansion.

**Limitations of Prior Work**: Existing methods suffer from two core issues—(F1) incorrectly attributing irrelevant Taylor terms to target features; and (F2) incomplete or overlapping allocation of Taylor terms. Moreover, the allocation of interaction effects is typically governed by fixed, predefined schemes (e.g., SHAP assumes equal splitting among involved features), lacking task adaptivity.

**Key Challenge**: In post-hoc, model-agnostic settings where ground-truth explanations are unavailable, fixed predefined interaction allocation schemes may yield arbitrary attribution results that deviate from the true feature importance ordering for the instance under analysis.

**Goal**: To establish a theoretically rigorous set of attribution criteria under the Taylor expansion framework, and to design an attribution method that satisfies all criteria while supporting adaptive optimization.

**Key Insight**: Beginning from the independent and interaction effects in Taylor expansion, the attribution process is regulated axiomatically, with tunable parameters enabling task-oriented adaptive allocation.

**Core Idea**: Three postulates are used to constrain the assignment of Taylor terms, and Dirichlet-sampled, AUP optimization-driven tuning is employed to adaptively allocate interaction effect weights.

## Method

### Overall Architecture
TaylorPODA performs a Taylor expansion of the model output $f(\mathbf{x})$ around a baseline point $\boldsymbol{\beta}$, decomposing the expansion terms into independent effects $\lambda(\boldsymbol{\phi})$ and interaction effects $\mu(\boldsymbol{\psi})$. Three postulates constrain the attribution rules, and tunable coefficients $\xi_{i,S}$ are introduced to enable adaptive allocation of interaction effects.

### Key Designs
1. **Three Postulates**

   - **Precision**: The Taylor independent effect of the $i$-th feature must be attributed solely to the $i$-th feature, with $\tau_{i,j}=1$ if $i=j$ and $0$ otherwise. This addresses the erroneous attribution of independent effects in F1.
   - **Federation**: The Taylor interaction effect of a feature set $S$ may only be attributed to features within $S$, with $\zeta_{i,\psi}=0$ when $i \notin S$. This addresses the erroneous attribution of interaction effects to irrelevant features in F1.
   - **Zero-Discrepancy**: The sum of all attribution values plus the baseline value exactly equals the model output, $f(\boldsymbol{\beta})+\sum_i a_i = f(\mathbf{x})$. This addresses incomplete or redundant allocation in F2.

2. **Adaptation Property**

   - **Function**: Allows the allocation weights $\xi_{i,S}$ of interaction effects among involved features to be tunable, with $\xi_{i,S} \in (0,1)$ and $\sum_{i \in S} \xi_{i,S} = 1$.
   - **Mechanism**: Unlike SHAP's fixed equal splitting ($1/|S|$), TaylorPODA allows allocation ratios to be optimized toward downstream objectives.
   - **Design Motivation**: In post-hoc settings without ground-truth explanations, fixed allocation may deviate from the true feature importance.

3. **AUP Optimization Strategy**

   - **Function**: Uses the Area Under Prediction recovery curve (AUP) as the optimization objective, and performs random search over optimal combinations of $\xi_{i,S}$ via Dirichlet distribution sampling.
   - **Mechanism**: $\text{AUP}(\mathbf{a};\mathbf{x},f) = \sum_{m=1}^d |f(\mathbf{x}) - \mathbb{E}[f(X)|X_{\mathcal{I}(m)}=\mathbf{x}_{\mathcal{I}(m)}]|$; minimizing AUP means that features ranked by attribution magnitude recover the prediction most rapidly.
   - **Design Motivation**: The Dirichlet distribution naturally satisfies the normalization constraint, ensuring the zero-discrepancy postulate.

### Attribution Formula
$$a_i^{(\text{TaylorPODA})} = f(\mathbf{x}) - f_{G\setminus\{i\}}(\mathbf{x}) - \sum_{\substack{S \subseteq G, |S|>1 \\ i \in S}} (1-\xi_{i,S}) H(S)$$

where $H(S) = \sum_{T \subseteq S} (-1)^{|T|-|S|} f_T(\mathbf{x})$ is the Harsanyi dividend.

## Key Experimental Results

### Main Results — Tabular Data Feature Importance Alignment

| Method | Cancer AUP↓ | Rice AUP↓ | Titanic AUP↓ | Abalone AUP↓ | Concrete AUP↓ |
|--------|-------------|-----------|--------------|--------------|---------------|
| OCC-1 | 0.672 | 0.595 | 0.530 | 0.152 | 0.373 |
| LIME | 0.790 | 0.694 | 0.625 | 0.140 | 0.343 |
| SHAP | 0.874 | 0.668 | 0.516 | 0.161 | 0.274 |
| WeightedSHAP | **0.519** | **0.470** | **0.392** | 0.104 | 0.226 |
| TaylorPODA | 0.601 | 0.493 | 0.444 | **0.092** | **0.221** |

### Postulate Satisfaction Comparison

| Method | Precision | Federation | Zero-Discrepancy | Adaptation |
|--------|-----------|------------|------------------|------------|
| OCC-1 | ✓ | ✓ | ✗ | ✗ |
| LIME | — | — | — | — |
| SHAP | ✓ | ✓ | ✓ | ✗ |
| WeightedSHAP | ✗ | ✓ | ✗ | ✓ |
| TaylorPODA | ✓ | ✓ | ✓ | ✓ |

### Key Findings
- TaylorPODA is the only method that simultaneously satisfies all three postulates and the adaptation property.
- On the AUP metric, TaylorPODA and WeightedSHAP alternate as the best performer—WeightedSHAP slightly outperforms on classification tasks, while TaylorPODA performs better on regression tasks.
- TaylorPODA and SHAP consistently maintain zero-discrepancy across all test samples (Figure 2 violin plots), while discrepancy distributions of other methods are unstable.
- On MNIST image data, TaylorPODA's attribution visualizations are highly consistent with SHAP and intuitively highlight discriminative features.
- OCC-1 performs worst on multiple datasets due to its violation of the zero-discrepancy postulate, resulting in incomplete attributions.
- Experiments on non-differentiable models (Appendix C) yield consistent conclusions, demonstrating practical generalizability (e.g., the open region of the digit "8").

## Highlights & Insights
- **Theoretical Rigor**: The axiomatic approach explicitly defines what constitutes a "good" Taylor attribution, providing verifiable theoretical guarantees for XAI.
- **Explicit Characterization of SHAP's Limitations**: The paper formally demonstrates that SHAP satisfies precision, federation, and zero-discrepancy but lacks adaptivity; WeightedSHAP introduces adaptivity at the cost of precision and zero-discrepancy.
- **SHAP-Style Visualization**: TaylorPODA's satisfaction of zero-discrepancy enables contribution-summing bar chart visualizations analogous to SHAP.
- **Elegance of Dirichlet Search**: Leveraging the normalization property of the Dirichlet distribution naturally satisfies the allocation constraint, avoiding complex constrained optimization.
- **Value of Table 2**: The postulate/property satisfaction comparison table is a core contribution of the paper, presenting TaylorPODA's advantages at a glance.

## Limitations & Future Work
- **Computational Efficiency**: The full version requires $2^{|G|-1}$ Harsanyi dividend computations, each involving $2^{|S|}$ masked output queries, making it infeasible in high-dimensional feature spaces.
- The paper employs a truncation approximation (limiting $|S| \leq c$), but error analysis is insufficiently developed.
- Dirichlet random search does not guarantee a global optimum; more efficient optimization algorithms (e.g., Bayesian optimization, gradient-based methods) warrant exploration.
- All opaque models in the experiments use MLP with tanh/logistic activations; validation on more complex architectures (e.g., Transformers) is absent.
- The suitability of AUP as the sole optimization objective requires validation across more diverse task types.
- The MNIST experiment uses a heuristic approximation ($|S|\leq c$), which deviates from the theoretically complete formulation.
- No comparison is made with gradient-based attribution methods (e.g., Integrated Gradients), as the latter require internal model access.
- The estimation approach for masked outputs (conditional expectation) may itself introduce bias.

## Related Work & Insights
- **Deng et al. (2024)**: Unified 14 attribution methods via Taylor expansion → TaylorPODA further axiomatizes and optimizes within this framework.
- **WeightedSHAP** (Kwon & Zou 2022): Introduced the AUP metric and adaptive weights → TaylorPODA achieves adaptation while satisfying a broader set of postulates.
- **Shapley-Taylor Interaction Index** (Sundararajan et al. 2020): Addresses feature subset-level attribution → TaylorPODA focuses on the more practical single-feature-level attribution.
- **LIME** (Ribeiro et al. 2016): Local linear approximation that is not decomposable under the Taylor framework → TaylorPODA provides stricter attribution semantics.
- **Harsanyi Dividend**: TaylorPODA cleverly borrows the Harsanyi dividend from cooperative game theory as the operator for interaction effects.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The axiomatic Taylor attribution analysis makes a clear theoretical contribution; the precision–federation–zero-discrepancy postulate formulation is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across tabular and image data, classification and regression tasks, and both quantitative metrics and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, and the comparative summary table is clearly presented (Table 2 is a highlight).
- **Value**: ⭐⭐⭐⭐ Offers a theoretically better-grounded alternative for post-hoc attribution, though computational efficiency limits large-scale practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](cost-free_neutrality_for_the_river_method.md)
- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](../../NeurIPS2025/others/an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[AAAI 2026\] DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)

</div>

<!-- RELATED:END -->
