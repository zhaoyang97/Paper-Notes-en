---
title: >-
  [Paper Note] Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p
description: >-
  [ICML2025][Interpretability][Symbolic Regression] Proposed the PAN+SR framework, which reduces high-dimensional symbolic regression problems to low-dimensional subspaces through BART-based nonparametric variable pre-screening, achieving significant performance improvements for 19 existing SR methods in high-dimensional scenarios.
tags:
  - "ICML2025"
  - "Interpretability"
  - "Symbolic Regression"
  - "Variable Selection"
  - "Nonparametric Methods"
  - "BART"
  - "High-Dimensional Data"
date: 2026-05-08
content_hash: 10613c6e8ce99729
---

# Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p

**Conference**: ICML2025  
**arXiv**: [2410.13681](https://arxiv.org/abs/2410.13681)  
**Code**: [GitHub - PAN_SR](https://github.com/mattsheng/PAN_SR)  
**Area**: Interpretability  
**Keywords**: Symbolic Regression, Variable Selection, Nonparametric Methods, BART, High-Dimensional Data

## TL;DR
Proposed the PAN+SR framework, which reduces high-dimensional symbolic regression problems to low-dimensional subspaces through BART-based nonparametric variable pre-screening, achieving significant performance improvements for 19 existing SR methods in high-dimensional scenarios.

## Background & Motivation

### Key Challenge

**Key Challenge**: Symbolic Regression (SR) aims to discover interpretable mathematical expressions, but the search space grows doubly exponentially with the number of features p. Almost all existing SR methods work well only when p≤10, failing to scale to high-dimensional datasets in modern science (p=102~459).

### Solution Approach

**Goal**: Unlike traditional variable selection, SR requires a **false negative rate (FNR) close to zero**—missing any relevant variable prevents the recovery of the true function. In contrast, false positives only increase the computational burden without affecting correctness. This asymmetric requirement renders standard FDR control methods inapplicable.

### Failure of PySR's Built-in Pre-selection Method

The random forest feature pre-selection in PySR performs poorly, and its documentation directly states that this option is "hardly used".

## Method

### Overall Architecture: PAN+SR (Two-Stage)
1. **PAN Pre-screening**: Identify the relevant variable subset Ŝ from high-dimensional data
2. **SR Solving**: Feed the dimension-reduced data into any SR method

### Key Designs 1: BART-Based Variable Importance Ranking
Independently run BART for K=20 times, and calculate the Variable Inclusion Proportion (VIP) as splitting variables for each feature across every posterior-sampled tree.

### Key Designs 2: Rank Aggregation Strategy
Utilize the **average rank** across K runs (rather than the raw VIP values):
- Relevant features cluster: ranks concentrate at the front
- Irrelevant features cluster: ranks concentrate at the back
- Use Agglomerative Hierarchical Clustering (AHC) to automatically split the two groups

### Design Advantages
- No need to know the sparsity $p_0$
- No tunable thresholds, completely data-driven
- Extremely low computational overhead for PAN (averaging only 74.14 seconds)

## Key Experimental Results

### Black-Box Regression Problems (35 High-Dimensional Real Datasets)

| Metric | PAN+SR Performance |
|------|-----------|
| R² Improvement | 18/19 methods improved |
| Training Time | AIFeynman 5x speedup, uDSR 3x speedup |
| Model Complexity | Unchanged or reduced |
| PAN Overhead | 74.14 seconds on average |

### Synthetic Regression (100 High-Dimensional Feynman Equations, p=102~459)

| SR Method | Standalone Solve Rate (%) | PAN+SR Solve Rate (%) |
|---------|-----------|-------------|
| uDSR | 36.6 | **71.8** |
| AIFeynman | 0 (OOM) | **Execution Restored** |
| Operon | 18.1 | **27.4** |
| DSR | 8.9 | **25.8** |
| GP-GOMEA | 18.2 | **24.1** |

### Ablation: Comparison of Variable Selection Methods

| Method | TPR | PAN Criterion Compliance |
|------|-----|-------------|
| PAN (Ours) | **~99%** | ✅ |
| BART-G.SE | Insufficient | ❌ |
| Random Forest | Poor | ❌ |

## Highlights & Insights

1. Accurately identifies the asymmetric requirements of FNR/FPR in SR variable selection, laying the foundation for future research.
2. Minimalist approach: a three-step pipeline of "multiple BART runs → average ranking → hierarchical clustering", with no hyperparameters or thresholds.
3. Unprecedented scale of evaluation: 138K core-hours of computation, 19 methods × 135 datasets × multiple SNRs.
4. The extended high-dimensional SRBench (102–459 dimensions) fills the benchmark gap.

## Limitations & Future Work

1. Under extreme noise (SNR=0.5, n=500), FNR > 5%, and the PAN criterion is no longer satisfied.
2. The capability to handle highly correlated feature sets still needs improvement.
3. The method relies on the sparsity assumption $p_0 \ll p$, and non-sparse scenarios are not discussed.
4. The efficiency of BART may decrease when p > 1000.

## Related Work & Insights

- **SRBench**: The standard SR benchmark, on which ours extends high-dimensional problems.
- **iBART**: Iterative BART variable selection, which is the source of inspiration for the design of PAN.
- Insight: The "screen-then-search" paradigm can be generalized to combinatorial explosion problems such as NAS and program synthesis.

## Supplementary Analysis

### Sensitivity to SNR and Sample Size
- n=1000, SNR=Layout: FNR≈0%, FPR≈0%, optimal conditions
- n=1000, SNR=10: FNR≈0% but FPR begins to increase, and the solve rate drops to 0% due to noise limitations
- n=500, SNR=0.5: FNR > 5%, and the PAN criterion fails under extreme noise
- The effect of sample size is small, and SNR is the dominant factor

### Computational Overhead Comparison
The PAN pre-screening averages only 74s (black-box) and 325s (synthetic), and K=20 runs of BART can be fully parallelized. Compared to downstream SR (AIFeynman 71250s → PAN+AIFeynman 13997s = 5x speedup), the pre-screening overhead is negligible.

### Extending the SRBench Benchmark
For each relevant feature in the Feynman equations, s=50 independent and identically distributed (i.i.d.) irrelevant features are generated, resulting in a total dimension of p=51·p₀ (102 to 459 dimensions), with 8 signal-to-noise ratios added.

## Rating
- Novelty: ⭐⭐⭐⭐☆ (4.0/5)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5.0/5) — The largest systematic evaluation in the SR field to date
- Writing Quality: ⭐⭐⭐⭐⭐ (5.0/5)
- Value: ⭐⭐⭐⭐☆ (4.0/5) — Direct practical value to the SR community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[NeurIPS 2025\] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](../../NeurIPS2025/interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)

</div>

<!-- RELATED:END -->
