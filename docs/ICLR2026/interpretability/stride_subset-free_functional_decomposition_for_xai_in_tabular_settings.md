---
title: >-
  [Paper Note] STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings
description: >-
  [ICLR 2026][Functional Decomposition] STRIDE reformulates model explanation as an orthogonal functional decomposition problem in RKHS. By recursively centering kernel functions, it analytically computes orthogonal functional components $f_S(x_S)$ without enumerating $2^d$ subsets. The method not only produces scalar importance scores but also reveals how features synergistically or redundantly influence predictions, achieving 3× speedup over TreeSHAP with $R^2 = 0.93$ on tabular data.
tags:
  - ICLR 2026
  - Functional Decomposition
  - RKHS
  - Centered Kernels
  - Feature Interaction
  - Component Surgery
date: 2026-05-08
content_hash: 698cc6636dd5049b
---

# STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings

**Conference**: ICLR 2026
**arXiv**: [2509.09070](https://arxiv.org/abs/2509.09070)
**Code**: None
**Area**: Explainable AI / Kernel Methods
**Keywords**: Functional Decomposition, RKHS, Centered Kernels, Feature Interaction, Component Surgery

## TL;DR

STRIDE reformulates model explanation as an orthogonal functional decomposition problem in RKHS. By recursively centering kernel functions, it analytically computes orthogonal functional components $f_S(x_S)$ without enumerating $2^d$ subsets. The method not only produces scalar importance scores but also reveals how features synergistically or redundantly influence predictions, achieving 3× speedup over TreeSHAP with $R^2 = 0.93$ on tabular data.

## Background & Motivation

**Background**: Mainstream XAI methods (e.g., SHAP, LIME, IG) compress each feature's influence into a single scalar $\phi_i$. The Shapley value framework provides axiomatically fair attribution, and optimized algorithms such as TreeSHAP enable efficient computation for specific model families, establishing the de facto standard in practice.

**Limitations of Prior Work**: Scalar attribution suffers from two fundamental flaws. First, **limited expressiveness**: compressing complex nonlinear feature effects into a single number cannot answer "how does a feature influence predictions"—only "which feature matters." Synergy and redundancy among features are entirely obscured. Second, **high computational cost**: exact Shapley value computation requires enumerating $2^d$ feature subsets, leading to exponential complexity; practitioners must rely on approximations.

**Key Challenge**: The XAI community faces a fundamental dilemma—either use scalar attribution for efficient but coarse explanations (answering "what matters"), or use functional decomposition for detailed but expensive explanations (answering "how it matters"). No prior framework simultaneously provides both at practical computational efficiency.

**Goal**: (1) How to efficiently compute functional decompositions without enumerating $2^d$ subsets? (2) How to recover orthogonal functional components while remaining model-agnostic? (3) How to quantitatively demonstrate the "functional necessity" of interaction effects rather than merely observing correlations?

**Key Insight**: The authors observe that Hoeffding's functional ANOVA decomposition and reproducing kernel Hilbert space (RKHS) theory can be combined—by defining recursively centered kernel functions, orthogonal subspaces corresponding to each feature subset can be constructed, from which functional components are recovered via analytic projection. The key insight is that the Möbius inversion structure of centered kernels enables decomposition without subset enumeration.

**Core Idea**: Construct orthogonal subspaces via recursively centered kernels in RKHS, upgrading model explanation from scalar attribution to orthogonal functional decomposition without subset enumeration.

## Method

### Overall Architecture

The STRIDE pipeline takes an arbitrary black-box model $f: \mathcal{X} \rightarrow \mathbb{R}$ and a dataset as input. (1) **Kernel Construction**: Select base kernels $k_i$, normalize them, and build product kernels $K_S$; (2) **Recursive Centering**: Recursively center product kernels to obtain $K_S^{(c)}$, constructing orthogonal subspaces $\mathcal{H}_S$; (3) **Analytic Projection**: Project $f$ onto each subspace to recover orthogonal components $f_S(x_S)$; (4) **Aggregation** (optional): Aggregate functional components into scalar attributions $\phi_i$ compatible with Shapley axioms. The output is a complete functional decomposition $f = \sum_S f_S$ and/or scalar attributions.

### Key Designs

1. **Recursively Centered Kernels**:

    - **Function**: Construct centered kernels $K_S^{(c)}$ corresponding to each feature subset $S$ such that kernels for different subsets are orthogonal in $L^2(\mu)$.
    - **Mechanism**: Define the recursive relation $K_S^{(c)} := K_S - \sum_{R \subsetneq S} K_R^{(c)}$, with base case $K_\emptyset^{(c)} := 1$. Via Möbius inversion, an explicit formula follows: $K_S^{(c)} = \sum_{R \subseteq S}(-1)^{|S|-|R|}K_R$. The central property is "partial zero-mean"—integrating over any dimension $i \in S$ yields zero: $\int_{T_i} K_S^{(c)}(x_S, t_S)\, d\mu_i(t_i) = 0$. This directly implies orthogonality between centered kernels of different subsets (Theorem 1).
    - **Design Motivation**: Orthogonality guarantees the uniqueness and interpretability of the decomposition—each component $f_S$ precisely captures the "pure interaction effect" of subset $S$, free from contamination by lower- or higher-order effects.

2. **Analytic Projection and Shapley-Compatible Aggregation**:

    - **Function**: Recover functional components via orthogonal projection, with optional aggregation into scalar attributions satisfying Shapley axioms.
    - **Mechanism**: By the orthogonal projection theorem, $f_S(x_S) = \langle f, K_S^{(c)}(\cdot_S, x_S) \rangle$, i.e., the projection of $f$ onto subspace $\mathcal{H}_S$. In practice, integrals are approximated by empirical sample averages, with low-rank approximation and regularization for numerical stability. Scalar attributions are defined as $\phi_i(x) := \sum_{S \ni i} \frac{1}{|S|} f_S(x_S)$, satisfying the efficiency, symmetry, dummy, and linearity axioms.
    - **Design Motivation**: This preserves the axiomatic advantages of Shapley values (familiar to practitioners) while providing richer functional component information—the latter is a strict superset of the former.

3. **Component Surgery**:

    - **Function**: Quantify the direct impact of individual interaction components on model performance, validating their "functional necessity."
    - **Mechanism**: STRIDE decomposition first identifies the most critical interaction components (e.g., the highest-impact $f_{ij}$), then surgically removes that component from model predictions—replacing $f(x)$ with $f(x) - f_{ij}(x_{ij})$—and measures the resulting performance drop. On the California Housing dataset, removing the single most important interaction component causes a test $R^2$ drop of $0.023 \pm 0.004$.
    - **Design Motivation**: This bridges the gap from the qualitative claim "this interaction appears important" to the quantitative statement "removing it costs this much performance," providing causal-level evidence rather than mere statistical observation.

### Loss & Training

STRIDE is a post-hoc method and involves no model training. Kernel hyperparameters (e.g., RBF bandwidth) are set using the median heuristic. The rank of the low-rank approximation and regularization strength are treated as hyperparameters. Interaction order is selected by filtering via feature dependency scores, modeling only the most relevant feature pairs.

## Key Experimental Results

### Main Results: Comparison with TreeSHAP

| Dataset | Dim $d$ | STRIDE Time (s) | TreeSHAP Time (s) | Speedup | $R^2$ | Spearman $\rho$ |
|--------|---------|----------------|-------------------|--------|-------|-----------------|
| California Housing | 8 | 0.550 | 5.331 | **9.7×** | 0.932 | 0.955 |
| Credit Default | 23 | 1.609 | 11.679 | **7.3×** | 0.988 | 0.945 |
| Online Shoppers | 25 | 1.039 | 3.914 | **3.8×** | 0.965 | 0.898 |
| YearPredictionMSD | 90 | 72.613 | 168.976 | **2.3×** | 0.808 | 0.553 |
| Breast Cancer | 30 | 0.069 | 0.038 | 0.6× | 0.999 | 0.736 |

Median speedup is approximately 3.0×, mean $R^2 = 0.93$, and Spearman correlation exceeds 0.8 on most datasets.

### Component Surgery and XAI Framework Comparison

| Method | Target Model | Explanation Unit | High-Order Interactions | Computational Efficiency |
|------|---------|---------|---------|---------|
| LIME | Any | $\phi_i$ | ✗ | High |
| KernelSHAP | Any | $\phi_i$ | ✗ | Very High |
| TreeSHAP | Tree Models | $\phi_i$ | ✓ (partial) | Low |
| IG / DeepLIFT | Neural Networks | $\phi_i$ | ✗ | Very Low |
| KAN | Neural Networks | Function | (implicit) | Medium |
| **STRIDE** | **Any*** | **$\phi_i$ + $f_S$** | **✓** | **Low–Medium** |

Component surgery result: removing the single most important interaction on California Housing → test $R^2$ drop of $0.0232 \pm 0.0035$, demonstrating functional necessity of the interaction.

### Key Findings

- **Functional Components Reveal Domain Knowledge**: On California Housing, STRIDE automatically discovers redundancy between Latitude and Longitude, as well as positive synergy between Longitude and Population (corresponding to high-density, high-value coastal California areas), in perfect agreement with domain knowledge.
- **What-If Analysis Capability**: Simulating an increase in MedInc substantially reduces the model's reliance on proxy features (Latitude, Longitude), demonstrating that the model correctly learned "location is a proxy for income."
- **Scalability to High Dimensions**: STRIDE maintains a 2.3× speedup on YearPredictionMSD ($d=90$), though $R^2$ drops to 0.808.
- **TreeSHAP Faster on Small Tasks**: On Breast Cancer ($d=30$), STRIDE is slower than TreeSHAP, indicating limited optimization gains on small-scale simple tasks.

## Highlights & Insights

- **Paradigm Shift from "What" to "How"**: STRIDE not only identifies which features matter but reveals how features work synergistically or redundantly. This functional perspective is strictly more expressive than scalar attribution—scalar attributions can be aggregated from functional components, but not vice versa.
- **Novelty of Component Surgery**: This is the first method to enable quantitative causal validation of individual high-order interactions in black-box models. The approach can be directly transferred to any XAI framework offering functional decomposition, serving as a standard tool for verifying explanation reliability.
- **Theoretical Elegance**: The mathematical chain—recursively centered kernels → Möbius inversion → orthogonal subspaces → analytic projection—is remarkably clean, elegantly reducing a complex combinatorial problem to linear algebraic operations.

## Limitations & Future Work

- **Validation Limited to Tabular Data**: Current experiments are restricted to random forests on tabular data; applicability to deep learning models (e.g., Transformers) and non-tabular data (images, text) remains unvalidated.
- **Scalability of High-Order Interactions**: Although $2^d$ enumeration is avoided, modeling high-order interactions may still incur substantial cost; the current implementation models only a subset of feature pairs.
- **Sensitivity to Kernel Choice**: Decomposition results depend on the choice of kernel function and bandwidth; different kernels may yield different decompositions.
- **Degradation Under Weak Signals**: On YearPredictionMSD ($\rho \approx 0.55$), ranking agreement with TreeSHAP is low, suggesting that the two methods diverge when feature effects are weak.

## Related Work & Insights

- **vs TreeSHAP**: TreeSHAP is model-specific (tree models only) and provides only scalar attributions. STRIDE is model-agnostic and delivers complete functional components, though currently validated only in tabular settings.
- **vs RKHS-SHAP (Chau et al., 2022)**: Also RKHS-based but still focused on scalar attribution aggregation; STRIDE retains the richer intermediate result of functional components.
- **vs KAN**: KAN provides interpretability through its architecture (spline activations) and is model-intrinsic rather than post-hoc. As a post-hoc method, STRIDE is applicable to any pre-trained model, offering greater flexibility.
- **vs Functional ANOVA**: Classical ANOVA requires marginal integration and is typically a global analysis; STRIDE provides instance-level local decomposition.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing RKHS orthogonal decomposition into XAI is a novel idea; component surgery is a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10 datasets, multiple baselines, and complete ablation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear; experimental narration is fluent; the motivation–theory–experiment logical chain is complete.
- Value: ⭐⭐⭐⭐ — Offers a meaningful tool upgrade for tabular XAI, though the scope of application awaits extension.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](../../CVPR2026/interpretability/from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)
- [\[ACL 2026\] TabReX: Tabular Referenceless eXplainable Evaluation](../../ACL2026/interpretability/tabrex_tabular_referenceless_explainable_evaluation.md)
- [\[ICLR 2026\] Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)
- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](../../CVPR2026/interpretability/cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/interpretability/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)

<!-- RELATED:END -->
