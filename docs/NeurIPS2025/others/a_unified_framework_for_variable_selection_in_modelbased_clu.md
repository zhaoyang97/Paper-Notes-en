---
title: >-
  [Paper Note] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random
description: >-
  [NeurIPS 2025][Model-based clustering] Within a Gaussian mixture model clustering framework, this paper jointly addresses variable selection (distinguishing signal, redundant…
tags:
  - "NeurIPS 2025"
  - "Model-based clustering"
  - "variable selection"
  - "MNAR"
  - "Gaussian mixture model"
  - "LASSO penalty"
  - "BIC consistency"
date: 2026-05-08
content_hash: 626df3a437e21727
---

# A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random

**Conference**: NeurIPS 2025
**arXiv**: [2505.19093](https://arxiv.org/abs/2505.19093)  
**Code**: None  
**Area**: Statistical Learning / Clustering / Missing Data Handling
**Keywords**: Model-based clustering, variable selection, MNAR, Gaussian mixture model, LASSO penalty, BIC consistency

## TL;DR
Within a Gaussian mixture model clustering framework, this paper jointly addresses variable selection (distinguishing signal, redundant, and noise variables) and MNAR missing data modeling. A two-stage strategy—LASSO-penalized ranking followed by BIC-based role assignment—combined with spectral-distance adaptive penalty weights enables efficient inference in high-dimensional settings. Identifiability and asymptotic selection consistency are established theoretically.

## Background & Motivation

**Background**: Model-based clustering performs probabilistic clustering by assuming data arise from a finite mixture model, with the Gaussian mixture model (GMM) as the canonical instance. In high-dimensional data, many variables may be irrelevant or redundant to the cluster structure, motivating variable selection to improve clustering performance and interpretability. Maugis et al. proposed the SRUW model, which partitions variables into four roles: clustering-relevant (S), redundant variables linearly explained by S (U), the regression set linking U to S (R), and independent noise (W).

**Limitations of Prior Work**: (1) Existing variable selection methods (e.g., Clustvarsel, Selvar) assume complete data or handle only MAR missingness, whereas practical settings such as transcriptomics often exhibit MNAR mechanisms (missingness depends on unobserved values), and ignoring this leads to errors in both clustering and variable selection. (2) Prior work either addresses variable selection without handling missingness, or handles missingness without variable selection—no unified framework exists. (3) The two-stage ranking strategy of Celeux et al. is computationally efficient but lacks theoretical guarantees.

**Key Challenge**: MNAR missingness violates the ignorability assumption required by standard EM, while variable selection depends on accurate parameter estimation—the two problems are mutually entangled and must be solved jointly.

**Goal**: To simultaneously perform variable role identification (S/R/U/W) and clustering parameter estimation in the presence of MNAR missing data, with formal theoretical guarantees.

**Key Insight**: Under the MNARz mechanism—where missingness probability depends only on latent class membership rather than observed values—the MNAR problem can be reformulated as a MAR problem on augmented data (original data plus the missingness indicator matrix), thereby preserving the tractability of the EM algorithm.

**Core Idea**: Jointly model the SRUW variable selection framework and the MNARz missingness mechanism, and achieve efficient inference via a two-stage strategy with spectral-distance adaptive penalty weights, while establishing asymptotic selection consistency.

## Method

### Overall Architecture
The input is a data matrix $\mathbf{Y} \in \mathbb{R}^{N \times D}$ with missing values and a missingness mask $\mathbf{C}$. The procedure proceeds in two stages: Stage A (ranking) fits a LASSO-penalized GMM on quickly imputed data to rank variables; Stage B (role assignment) traverses the ranked list sequentially, using BIC on the original incomplete data to assign each variable a role (S/U/W), while estimating parameters via an EM algorithm under the MNARz mechanism. The output is a variable partition $(\hat{\mathbb{S}}, \hat{\mathbb{R}}, \hat{\mathbb{U}}, \hat{\mathbb{W}})$, the estimated number of clusters $\hat{K}$, and parameter estimates.

### Key Designs

1. **SRUW-MNARz Joint Model**:

    - **Function**: Unified modeling of variable roles and the missingness mechanism.
    - **Mechanism**: The complete-data density is decomposed as a product of three components—a clustering component $f_{\text{clust}}(\mathbf{y}^{\mathbb{S}}|\alpha_k)$, a regression component $f_{\text{reg}}(\mathbf{y}^{\mathbb{U}}|\mathbf{y}^{\mathbb{R}})$, and an independent component $f_{\text{indep}}(\mathbf{y}^{\mathbb{W}})$—multiplied by the MNARz missingness component $f_{\text{MNARz}}(\mathbf{c}_n|z_{nk}=1;\psi_k)$. Because MNARz assumes that missingness probability depends only on the class label $k$, the missingness term factors out of the integral, preserving ignorability.
    - **Design Motivation**: The key advantage of the MNARz mechanism is that it transforms MNAR into MAR on augmented data—inference on $(Y, C)$ is equivalent to MAR inference. This avoids the non-identifiability issues of general MNAR while allowing the missingness pattern itself to carry clustering information.

2. **Spectral-Distance Adaptive Penalty Weights**:

    - **Function**: Replaces traditional inverse partial correlations to construct the penalty matrix $\mathbf{P}_k$.
    - **Mechanism**: An undirected graph $\mathcal{G}^{(k)}$ is constructed from an initial precision matrix estimate $\hat{\Psi}_k^{(0)}$; the symmetric normalized Laplacian $\mathbf{L}_{sym}^{(k)}$ is computed with the goal of shrinking $\Psi_k$ toward a diagonal matrix (empty graph). The spectral distance is $D_{\text{LS}} = \|\text{spec}(\mathbf{L}_{sym}^k)\|_2$, and the adaptive weight is $\mathbf{P}_{k,ij} = (D_{\text{LS}} + \epsilon)^{-1}$.
    - **Design Motivation**: The spectral distance provides a global measure of graph structural complexity that is more stable than element-wise partial correlations, particularly in high-dimensional, small-sample settings with missing data.

3. **Two-Stage Selection Strategy**:

    - **Function**: Efficiently performs variable ranking and role assignment.
    - **Mechanism**: Stage A fits a penalized GMM on quickly imputed data along a $(\lambda, \rho)$ regularization path and computes a ranking score $\mathcal{O}_K(d)$ for each variable based on how frequently its mean is shrunk to zero. Stage B performs a single pass through the ranked list, fitting an unpenalized SRUW-MNARz model on the original incomplete data each time a new variable is added, and assigns roles via BIC.
    - **Design Motivation**: Stage A uses fast imputation and a penalized model for low-cost ranking (avoiding combinatorial explosion); Stage B performs exact inference on the original data (avoiding error propagation from imputation). Decoupling the two stages reduces computational complexity from exponential to linear.

### Loss & Training
Stage A uses a penalized log-likelihood with an $\ell_1$ mean penalty and an adaptive graphical lasso penalty on the precision matrix, searching along a data-driven geometric path over $(\lambda, \rho)$. Stage B maximizes the BIC criterion for SRUW-MNARz via standard EM. Initialization employs K-means++ / hierarchical clustering combined with a soft E-step to estimate missingness rates.

## Key Experimental Results

### Main Results (Synthetic Data, 50% MNAR Missingness Rate)

| Model | Mean ARI | Std. Dev. | vs. SelvarMNARz | p-value |
|-------|----------|-----------|-----------------|---------|
| **SelvarMNARz** | **0.511** | 0.052 | — | — |
| Clustvarsel | 0.363 | 0.088 | Significantly worse | <0.001 |
| Selvar | 0.348 | 0.108 | Significantly worse | <0.001 |
| VarSelLCM | 0.344 | 0.101 | Significantly worse | <0.001 |

### Ablation Study

| Missingness Rate | Missingness Type | SelvarMNARz ARI | Best Baseline ARI | Notes |
|------------------|-----------------|-----------------|-------------------|-------|
| 5% | MAR | ~0.85 | ~0.82 | Small gap at low missingness |
| 20% | MAR | ~0.75 | ~0.60 | Baselines begin to degrade |
| 50% | MNAR | 0.511 | 0.363 | Baselines severely degrade; proposed method robust |
| 50% | MAR | ~0.55 | ~0.40 | MNAR more challenging; proposed method still best |

### Key Findings
- **SelvarMNARz achieves the best performance across all missingness rates and mechanisms**, with the most gradual degradation as missingness increases—ARI remains 0.51 at 50% MNAR, while baseline methods drop to 0.34–0.36.
- **Variable selection accuracy**: SelvarMNARz correctly recovers the clustering variable set and the number of clusters $K$ across all configurations, whereas impute-then-cluster pipelines begin misclassifying variable roles when missingness exceeds 20%.
- **Transcriptomics experiment**: On an Arabidopsis dataset with 1,267 genes, SelvarMNARz identifies 18 clusters and reclassifies variables P6/P7 from "signal variables" (as identified in prior analyses) to "redundant variables" (explainable by P1–P4), demonstrating how correctly handling the missingness mechanism can alter biological conclusions.

## Highlights & Insights
- The **necessity of a unified framework** is convincingly argued: rather than simply combining variable selection and missing data handling, the paper theoretically demonstrates that joint treatment is required—separate handling leads to selection inconsistency.
- The **MNARz-to-MAR transformation** is the key theoretical contribution: by incorporating the missingness indicator matrix into augmented data, the intractability of MNAR is converted into standard MAR inference. This trick has broad applicability to any problem involving latent variables and MNAR missingness.
- The **proof strategy for the selection consistency theorem (Theorem 3)**—from RSC conditions to ranking consistency to BIC consistency—provides new tools for the theoretical analysis of penalized likelihood methods in mixture models.

## Limitations & Future Work
- The current framework is restricted to continuous data under a Gaussian mixture model and cannot handle categorical or mixed-type variables; the authors acknowledge that extension to discrete data (e.g., latent class models) is a natural next step.
- The MNARz assumption (missingness depends only on class membership), while more tractable than general MNAR, may be too strong in certain practical scenarios—for example, gene expression values are more likely to be missing when extremely low, which corresponds to general MNAR.
- The fast imputation in Stage A of the two-stage strategy may introduce bias at high missingness rates; although Stage B corrects this on the original data, initial ranking bias may lead to suboptimal results.
- The computational complexity analysis is insufficiently detailed, and scalability in truly large-scale, high-dimensional settings ($D > 10{,}000$) remains unclear.

## Related Work & Insights
- **vs. Maugis et al. (SRUW)**: Introduced the SRUW variable partition framework but addressed only MAR/complete data; the present paper extends it to MNARz and provides theoretical guarantees.
- **vs. Celeux et al. (Selvar)**: Proposed the two-stage ranking strategy without theoretical guarantees; the present paper establishes selection consistency of this strategy under MNARz.
- **vs. Sportisse et al. (MNARz)**: Provided identifiability analysis for the MNARz mechanism; the present paper builds on this foundation and integrates it with SRUW variable selection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified framework and theoretical guarantees represent solid contributions, though each individual component (SRUW, MNARz, two-stage strategy) has prior precedent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic experiments are systematic (multiple missingness rates, multiple mechanisms, statistical testing); real transcriptomics data include biological interpretation.
- **Writing Quality**: ⭐⭐⭐ The mathematical notation is heavy; informal statements of theorems aid comprehension, but technical details require consulting the supplementary material.
- **Value**: ⭐⭐⭐⭐ Directly applicable to high-dimensional missing-data clustering settings such as omics; theoretical guarantees enhance methodological credibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)
- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[ICML 2026\] Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](../../ICML2026/others/local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)
- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)

</div>

<!-- RELATED:END -->
