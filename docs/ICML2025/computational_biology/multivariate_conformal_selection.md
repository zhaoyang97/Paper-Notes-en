---
title: >-
  [Paper Note] Multivariate Conformal Selection
description: >-
  [ICML2025][Computational Biology][Conformal Selection] Extends Conformal Selection from univariate responses to multivariate settings, introduces the concept of Regional Monotonicity, designs distance-based (mCS-dist) and learning-based (mCS-learn) nonconformity scores, and guarantees finite-sample FDR control while improving selection power.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Conformal Selection"
  - "Multivariate Response"
  - "FDR Control"
  - "Nonconformity Score"
  - "BH Procedure"
  - "Regional Monotonicity"
  - "Differentiable Sorting"
date: 2026-05-08
content_hash: e173647f73e56361
---

# Multivariate Conformal Selection

**Conference**: ICML2025  
**arXiv**: [2505.00917](https://arxiv.org/abs/2505.00917)  
**Code**: None  
**Area**: Optimization/Theory  
**Keywords**: Conformal Selection, Multivariate Response, FDR Control, Nonconformity Score, BH Procedure, Regional Monotonicity, Differentiable Sorting

## TL;DR

Extends Conformal Selection from univariate responses to multivariate settings, introduces the concept of Regional Monotonicity, designs distance-based (mCS-dist) and learning-based (mCS-learn) nonconformity scores, and guarantees finite-sample FDR control while improving selection power.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Ubiquity of selection problems: Drug discovery (screening compounds with high binding affinity), precision medicine (identifying positive treatment effects), and LLM output certification (filtering trustworthy generated content) all require selecting a subset of candidates that meet specific criteria.

### Background

**Background**: Limitations of existing CS: Conformal Selection (Jin & Candès, 2023) only supports threshold selection for univariate responses $y > c$, failing to handle multi-dimensional criteria (e.g., LLM outputs satisfying fairness, safety, and correctness simultaneously).

### Proposed Approach

**Proposed Approach**: Multivariate CP is not directly applicable: Confidence sets constructed by multivariate conformal prediction might be incompatible with the shape of the pre-defined target region $R$, and they only control PCER instead of FDR.

### Key Challenge

**Key Challenge**: Goal: To construct a selection framework under the multivariate response setting that simultaneously satisfies: (1) finite-sample FDR control, (2) maximized selection power, and (3) being model-agnostic.

## Method

### Overall Architecture (Algorithm 1)

1. **Training**: Construct a multivariate predictive model $\hat{\mu}$.
2. **Calibration**: Compute regionally monotonic nonconformity scores $V_i = V(\bm{x}_i, \bm{y}_i)$ and construct conformal p-values.
3. **Thresholding**: Apply the BH procedure for multiple hypothesis testing correction to output the selection set $\mathcal{S}$.

### Key Design 1: Regional Monotonicity (Definition 3.1)

$$V(\bm{x}, \bm{y}') \leq V(\bm{x}, \bm{y}), \quad \forall \bm{y}' \in R^c, \bm{y} \in R$$

This guarantees the conservativeness of conformal p-values (Proposition 3.2), thereby ensuring FDR control (Theorem 3.5).

### Key Design 2: mCS-dist (Distance-based Score)

$$V(\bm{x}, \bm{y}) = D_1(\bm{y}, R^c) - D_2(\hat{\mu}(\bm{x}), R^c)$$

- **Regular score**: $D_1 = D_2 = \inf_{\bm{s} \in R^c} \|\cdot - \bm{s}\|_p$
- **Clipped score (Superior)**: $D_1 = M \cdot \mathbb{1}\{\bm{y} \notin R^c \cup \partial R\}$. Theorem 4.1 proves that the clipped score outperforms the regular score in terms of asymptotic power.

### Key Design 3: mCS-learn (Learning-based Score)

$$V^\theta(\bm{x}, \bm{y}) = M \cdot \mathbb{1}\{\bm{y} \notin R^c \cup \partial R\} - f_\theta(\bm{x}, \bm{y}; R)$$

- Uses differentiable sorting (soft-rank) to approximate conformal p-values, optimizing $f_\theta$ via backpropagation.
- Loss function $L_2$: Directly penalizes p-values, minimizing the p-values for samples inside the target region and increasing them for samples outside the region.
- Proposition 4.2 proves that this family of scores contains the optimal nonconformity score.

## Key Experimental Results

### Simulated Data

- Tested on 2D/5D/10D Gaussian mixtures and various target regions (convex/non-convex/irregular).
- Both mCS-dist and mCS-learn maintain FDR $\leq q$ under all settings, outperforming baseline methods by a significant margin.
- mCS-learn shows the most pronounced advantages in non-convex regions and high-dimensional scenarios.
- When dimensions increase from 2 to 10, the power of mCS-dist decreases but still maintains FDR control, whereas the performance drop of mCS-learn is much smaller.

### Real-world Data

- Drug discovery datasets: mCS achieves the highest selection power under FDR control.
- LLM alignment: In selection tasks with multi-dimensional alignment scores, mCS successfully screens outputs that meet multi-dimensional criteria simultaneously.

### Baseline Comparison

- **Marginal CS (per-dimension independent CS + Bonferroni correction)**: Very low power due to overly conservative multiple testing correction.
- **CP-based selection**: Only controls PCER, potentially leading to FDR inflation.
- **Oracle selection**: Serves as an upper-bound reference where true responses are known.
- Both mCS-dist and mCS-learn significantly outperform Marginal CS, approaching Oracle selection.

## Highlights & Insights

1. **Regional monotonicity** is the core innovation, elegantly generalizing univariate monotonicity to arbitrary dimensions and target regions.
2. **Expressive power of mCS-learn**: Proposition 4.2 theoretically guarantees that the optimal score is covered within the learnable family.
3. **Practical value**: Provides a general uncertainty quantification framework covering drug discovery to LLM certification.
4. **Modular design**: The pre-trained model $\hat{\mu}$ can be separated from the selection process, allowing flexible integration.

## Limitations & Future Work

- Splitting a calibration set from the training set is required, which reduces the data available for model training.
- mCS-learn requires an additional three-way split (train-validate-calibrate), leading to low data efficiency.
- The target region $R$ needs to be pre-defined; exploring adaptive target regions remains future work.
- Computing $\inf_{\bm{s} \in R^c} \|\cdot\|$ can be computationally expensive on complex regions.
- The power may be suboptimal when $|R| $ is extremely small or extremely large.
- Combining conditional density estimators with conformal p-values has not been explored.
- Robustness to out-of-distribution (OOD) test data is not discussed.

## Related Work & Insights

- **Conformal Selection (Jin & Candès, 2023)**: Direct generalization of this work, extending from $y > c$ to $\bm{y} \in R$.
- **Multivariate Extensions of Conformal Prediction**: Multivariate CP methods such as Bates et al. (2021) and Feldman et al. (2023) focus on constructing prediction sets rather than selection.
- **BH Procedure (Benjamini & Hochberg, 1995)**: The foundation of multiple testing correction for FDR control.
- **Differentiable Sorting (Blondel et al., 2020)**: Technical foundation for soft-rank in mCS-learn.
- **Insights**: The concept of learning-based scores could be generalized to conditional density estimation or Bayesian non-parametric frameworks; learning adaptive target regions can also be explored.

## Rating

- Novelty: ⭐⭐⭐⭐ — Regional monotonicity is a simple yet powerful generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough validation across both simulation and real-world data.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear theoretical and algorithmic descriptions.
- Value: ⭐⭐⭐⭐ — Provides a framework with rigorous statistical guarantees for multi-criteria selection problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)
- [\[ICLR 2026\] ConfHit: Conformal Generative Design with Oracle Free Guarantees](../../ICLR2026/computational_biology/confhit_conformal_generative_design_with_oracle_free_guarantees.md)
- [\[NeurIPS 2025\] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](../../NeurIPS2025/computational_biology/a_unified_framework_for_variable_selection_in_modelbased_clu.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/computational_biology/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2026\] Active Timepoint Selection for Learning Measure-Valued Trajectories](../../ICML2026/computational_biology/active_timepoint_selection_for_learning_measure-valued_trajectories.md)

</div>

<!-- RELATED:END -->
