---
title: >-
  [Paper Note] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces
description: >-
  [NeurIPS 2025][feature selection] This paper proposes TangledFeatures, a selection framework centered on **feature stability**, implementing a three-stage pipeline of correlation-graph clustering → ensemble representative selection → random forest refinement. The framework achieves highly reproducible, domain-knowledge-consistent feature subsets across resampling in highly correlated feature spaces, validated on alanine dipeptide backbone torsion angle prediction.
tags:
  - NeurIPS 2025
  - feature selection
  - stability
  - correlation redundancy
  - random forests
  - structural biology
date: 2026-05-08
content_hash: d8b6019343136a38
---

# TangledFeatures: Robust Feature Selection in Highly Correlated Spaces

**Conference**: NeurIPS 2025
**arXiv**: [2510.15005](https://arxiv.org/abs/2510.15005)
**Code**: R package TangledFeatures (GitHub, pending CRAN submission)
**Area**: Interpretability / Feature Selection
**Keywords**: feature selection, stability, correlation redundancy, random forests, structural biology

## TL;DR

This paper proposes TangledFeatures, a selection framework centered on **feature stability**, implementing a three-stage pipeline of correlation-graph clustering → ensemble representative selection → random forest refinement. The framework achieves highly reproducible, domain-knowledge-consistent feature subsets across resampling in highly correlated feature spaces, validated on alanine dipeptide backbone torsion angle prediction.

## Background & Motivation

**State of the Field**: Feature selection is a foundational step in predictive modeling, directly affecting model performance and interpretability. In structural biology, deep learning models such as AlphaFold achieve extremely high predictive accuracy, yet lack transparent understanding of the specific structural factors (residues, motifs, interatomic interactions) driving predictions. Post-hoc explanation methods such as SHAP and Integrated Gradients are widely used, but face a fundamental problem.

**Limitations of Prior Work**: When predictive features are highly correlated, most feature selection methods behave unpredictably. (1) **Instability**: LASSO arbitrarily selects one feature among correlated ones, potentially selecting entirely different feature subsets under different data splits; (2) **Redundancy**: Methods such as Boruta and RFE may retain large numbers of redundant features; (3) **Irreproducibility**: Different analysis runs yield different "important features," making scientific findings difficult to trust. Post-hoc methods such as SHAP are equally unstable in high-correlation settings.

**Root Cause**: For biologically actionable insights (e.g., guiding mutation experiments or protein design), feature selection results must simultaneously satisfy two difficult-to-achieve objectives: (1) biological interpretability—mapping to known structural or functional elements; and (2) cross-analysis reproducibility—yielding consistent feature sets across different data subsets. Existing methods focus solely on predictive accuracy and cannot guarantee both simultaneously.

**Paper Goals**: (1) Extract non-redundant representatives from correlated feature groups; (2) ensure selection results are highly stable under data perturbation; (3) validate whether selected features correspond to known biological drivers.

**Starting Point**: The authors elevate stability to equal importance with predictive accuracy—explicitly accepting minor sacrifices in predictive accuracy in exchange for highly reproducible and interpretable feature subsets.

**Core Idea**: First use a correlation graph to group and de-redundify entangled features, then use ensemble selection to ensure stability, and finally apply importance-based filtering to ensure parsimony.

## Method

### Overall Architecture

TangledFeatures is a three-stage pipeline: input feature matrix $D \in \mathbb{R}^{n \times m}$ → correlation module $c_\alpha$ constructs a correlation graph and clusters features → selection module $s_\beta$ identifies stable representative features from each cluster → refinement module $r_\gamma$ truncates by cumulative importance to produce the final subset $d'$ → downstream prediction $f(d') \to (\phi, \psi)$. The three modules execute sequentially with clear conceptual roles, each serving an explicit de-redundification or refinement function.

### Key Designs

1. **Correlation Module $c_\alpha$ — Graph Clustering for De-redundification**:

    - Function: Automatically groups highly correlated features and identifies redundant clusters.
    - Mechanism: Computes the Pearson correlation matrix $\Sigma \in \mathbb{R}^{m \times m}$ over the full feature matrix and constructs an undirected graph $G = (V, E)$, where $E = \{(i,j): |\Sigma_{ij}| \geq \tau\}$ and $\tau$ is a user-specified threshold. The **connected components** of the graph naturally define correlated feature clusters—features within the same cluster convey approximately redundant information. Prediction targets (torsion angles $\phi, \psi$) are represented in cosine-sine form $(\cos\phi, \sin\phi, \cos\psi, \sin\psi)$ to eliminate discontinuities at angular boundaries.
    - Design Motivation: Simpler and more intuitive than hierarchical clustering or PCA, and connected components naturally enforce transitivity—if $A$ is highly correlated with $B$ and $B$ is highly correlated with $C$, then $A$, $B$, and $C$ are placed in the same cluster.

2. **Selection Module $s_\beta$ — Ensemble Stability Selection**:

    - Function: Identifies the most representative and stable single feature from each correlated cluster.
    - Mechanism: Performs $R$ rounds of random forest training. In each round, one candidate feature is randomly sampled from each cluster and combined with all unclustered features to predict torsion angles. The cross-round average importance $\hat{I}(d_{ij}) = \frac{1}{R}\sum_{r=1}^{R}I_r(d_{ij})$ is computed, and the feature with the highest $\hat{I}$ in each cluster is retained as its representative. This ensemble approach ensures selections are not contingent on the randomness of any single run, consistent in spirit with stability selection and Boruta.
    - Design Motivation: Using a single-run feature importance ranking produces a "coin-flip effect"—correlated features have similar importances but only one is selected per run, leading to different features being chosen across runs. Repeated sampling and averaging identifies the feature that is "most consistently important."

3. **Refinement Module $r_\gamma$ — Cumulative Importance Truncation**:

    - Function: Further reduces the set of representative features by removing those with negligible contributions.
    - Mechanism: Trains a random forest on the representative features, ranks them in descending order of importance, and retains features until cumulative importance reaches a threshold of 0.99 (i.e., capturing 99% of the predictive signal). This step is analogous to retaining principal components that explain 99% of variance in PCA.
    - Design Motivation: Even after de-redundification, some cluster representatives may contribute marginally to prediction; retaining them adds noise without adding information.

### Loss & Training

TangledFeatures is a nonparametric method and involves no gradient optimization or loss functions. The core component is random forest feature importance (based on impurity reduction or permutation importance), used in both the selection and refinement modules.

## Key Experimental Results

### Main Results

| Method | OLS-$\phi$ | RF-$\phi$ | SVM-$\phi$ | OLS-$\psi$ | RF-$\psi$ | SVM-$\psi$ |
|------|-----------|----------|-----------|-----------|----------|-----------|
| No Feature | 0.19/0.93 | 0.29/0.83 | 0.06/0.99 | 0.65/0.71 | 0.69/0.80 | 0.83/0.69 |
| LASSO | 0.21/0.92 | 0.09/0.98 | 0.05/0.99 | 0.89/0.64 | 0.69/0.79 | 0.84/0.70 |
| Elastic Net | 0.20/0.92 | 0.07/0.98 | **0.05/0.99** | 0.90/0.64 | 0.66/0.81 | **0.87/0.89** |
| Boruta | 0.22/0.91 | 0.07/0.98 | 0.05/0.99 | 0.91/0.64 | 0.65/0.82 | 0.84/0.70 |
| TangledFeatures | 0.26/0.87 | 0.09/0.98 | 0.09/0.98 | 0.97/0.61 | 0.67/0.81 | 0.86/0.75 |

(Format: RMSE / $R^2$; best results in bold)

### Ablation Study: Stability Comparison

| Metric | TangledFeatures | ENR | RFE | LASSO | Boruta |
|------|----------------|-----|-----|-------|--------|
| Kuncheva Index ($\phi$) | **Near maximum, flat** | Sharp decline | Low overlap | Medium | Medium |
| Spearman Rank Correlation ($\phi$) | **Near 1.0** | Strong fluctuation | Strong fluctuation | Medium | Medium |
| Kuncheva Index ($\psi$) | **Highest and stable** | Declining | Low | Medium | Medium |
| Spearman Rank Correlation ($\psi$) | **Near perfect** | Fluctuating | Fluctuating | Medium | Medium |

### Key Findings

- **Stability dominates all baselines**: TangledFeatures' Kuncheva Index and Spearman rank correlation remain nearly constant across 10 bootstrap resampling runs, while ENR and RFE exhibit large fluctuations in the presence of correlated features. This is the paper's most central experimental finding.
- **Predictive accuracy remains competitive**: $R^2 > 0.97$ for RF/XGBoost prediction of $\phi$; although SVM with full features achieves $R^2 = 0.99$, the latter relies on additional signal from redundant features.
- **Selected features align with biological knowledge**: Features driving $\phi$ prediction concentrate on backbone and near-backbone distances such as ACE1-CH3↔ALA2-CB (flexibility/cap-related), which are known determinants of torsion angle variation. In contrast, LASSO selects similarly sparse features but includes redundant distances with weaker chemical interpretability.
- **The cost of high SVM accuracy**: SVM achieves the highest accuracy when using redundant features, as it can exploit complementary information across correlated features. TangledFeatures trades a small amount of accuracy for stability and interpretability—an explicit trade-off.

## Highlights & Insights

- **Elevating stability to a first-class objective**: In a field dominated by "accuracy first" thinking, explicitly positioning stability as equally important to accuracy is itself a valuable contribution. This perspective can be directly adopted in any application requiring reproducible explanations (medicine, law, scientific discovery).
- **Elegant design of graph connected components for clustering**: Requires no hyperparameter tuning (beyond threshold $\tau$), no specification of the number of clusters, and automatically handles transitive dependencies. Conceptually simple yet sufficiently effective against correlation redundancy.
- **The "most consistently important" logic of ensemble selection**: Repeated random sampling and averaging eliminate the "coin-flip effect," identifying the most reliable representative within each cluster—a technique directly transferable to any scenario requiring representative selection from correlated groups.

## Limitations & Future Work

- **Validated on alanine dipeptide only**: This is the simplest peptide model (only 10 heavy atoms, approximately 45 atomic pair distances); the method's scalability to high-dimensional proteins (thousands of residues) is unknown.
- **Threshold $\tau$ requires manual selection**: Different thresholds yield different clustering results; the paper does not discuss sensitivity analysis or adaptive selection strategies.
- **Relies solely on Pearson correlation**: Linear correlation may miss nonlinear dependencies (e.g., mutual information, distance correlation).
- **Accuracy trade-offs may be unacceptable for some applications**: $R^2$ drops from 0.93 to 0.87 for $\phi$ under OLS, which is not a negligible gap in accuracy-sensitive settings.
- **No direct comparison with PCA**: PCA also removes redundancy via linear transformation but sacrifices interpretability—including this baseline would strengthen the argument.

## Related Work & Insights

- **vs. Boruta (Kursa et al., 2010)**: Boruta performs global importance filtering via shadow features but does not address correlation—it may retain multiple redundant features from the same cluster. TangledFeatures first de-redundifies and then filters, making it better suited for high-correlation settings.
- **vs. Stability Selection (Meinshausen & Bühlmann, 2010)**: Stability selection evaluates feature inclusion frequency via resampling combined with LASSO, but remains subject to LASSO's arbitrary selection among correlated features. TangledFeatures addresses redundancy at the root through explicit clustering.
- **vs. PCA**: PCA eliminates correlation via orthogonal transformation, but the resulting principal components are difficult to trace back to the physical meaning of original features. TangledFeatures operates entirely in the original feature space, maintaining full interpretability.

## Rating

- Novelty: ⭐⭐⭐⭐ The stability-first positioning is well-defined; the three-module design combines existing techniques in a well-motivated manner.
- Experimental Thoroughness: ⭐⭐⭐ Single application domain (alanine dipeptide), but comprehensively evaluated along the three axes of accuracy, stability, and interpretability.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise, with intuitive figures and well-organized pipeline description.
- Value: ⭐⭐⭐ Methodology is general but validation scale is limited; R package release facilitates community adoption.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](../../AAAI2026/interpretability/unsupervised_feature_selection_through_group_discovery.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[NeurIPS 2025\] OrdShap: Feature Position Importance for Sequential Black-Box Models](ordshap_feature_position_importance_for_sequential_black-box_models.md)
- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](../../ICLR2026/interpretability/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[CVPR 2026\] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](../../CVPR2026/interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)

<!-- RELATED:END -->
