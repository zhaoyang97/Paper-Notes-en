---
title: >-
  [Paper Note] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings
description: >-
  [ICML2025][Self-Supervised Learning][Contrastive Learning] This paper establishes the first generalization bounds for supervised contrastive representation learning (CRL) under non-independent and identically distributed (non-IID) settings. By leveraging U-statistics decomposition techniques to handle the dependency issue arising from overlapping samples in training tuples, it provides the convergence rate of the excess risk with respect to the number of labeled samples $N$.
tags:
  - "ICML2025"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "Generalization Bounds"
  - "U-Statistics"
  - "Non-IID"
  - "Representation Learning"
date: 2026-05-08
content_hash: d641ca4c535bab34
---

# Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings

**Conference**: ICML2025  
**arXiv**: [2505.04937](https://arxiv.org/abs/2505.04937)  
**Code**: Not provided  
**Area**: Self-Supervised  
**Keywords**: Contrastive Learning, Generalization Bounds, U-Statistics, Non-IID, Representation Learning

## TL;DR

This paper establishes the first generalization bounds for supervised contrastive representation learning (CRL) under non-independent and identically distributed (non-IID) settings. By leveraging U-statistics decomposition techniques to handle the dependency issue arising from overlapping samples in training tuples, it provides the convergence rate of the excess risk with respect to the number of labeled samples $N$.

## Background & Motivation

### Background

**Background**: **Limited theoretical understanding of contrastive learning**: CRL has achieved widespread success in domains such as vision, graphs, and NLP, but theoretical generalization analyses have been limited by the assumption of i.i.d. tuples.

### Limitations of Prior Work

**Limitations of Prior Work**: **Tuples are not independent in practice**: CRL training typically constructs tuples from a fixed pool of labeled samples $\mathcal{S}=\{(\mathbf{x}_j,\mathbf{y}_j)\}_{j=1}^N$ by "recycling" the same data points, leading to overlapping data between tuples and violating the assumption of independence.

### Key Challenge

**Key Challenge**: **Gap in existing work**: Previous works such as Arora et al. (2019), Lei et al. (2023), and Hieu et al. (2024) all assume i.i.d. tuples.

### Goal

**Goal**: To propose a corrected, practical theoretical framework for deriving excess risk generalization bounds under non-IID settings.

## Method

### Corrected Theoretical Framework

Constructing a sub-sampled tuple set $\mathcal{T}_{sub}$ from a fixed pool of labeled samples $\mathcal{S}$:
1. Select the class of the anchor-positive pair according to the empirical class probability $\hat{\rho}(c)=N_c^+/N$.
2. Sample two positive samples from the class without replacement.
3. Sample $k$ negative samples from non-target classes without replacement.
4. Repeat independently $M$ times.

### U-Statistics Formulation

Construct a class-conditional U-statistic $\mathcal{U}_N(f|c)$ for each class $c$, and sum them with weights to obtain the overall estimator:

$$\mathcal{U}_N(f)=\sum_{c\in\mathcal{C}}\frac{N_c^+}{N}\mathcal{U}_N(f|c)$$

where $N_c=\min(\lfloor N_c^+/2\rfloor,\lfloor(N-N_c^+)/k\rfloor)$.

### Proof Strategy: Two-Step Decomposition

$$\frac{1}{2}[L_{un}(\hat{f}_{sub})-\inf_f L_{un}(f)] \leq \sup_f|\hat{\mathcal{L}}(f;\mathcal{T}_{sub})-\mathcal{U}_N(f)| + \sup_f|\mathcal{U}_N(f)-L_{un}(f)|$$

- **First term**: Deviation between sub-sampled empirical risk and the U-statistic $\rightarrow$ Rademacher complexity bound (since tuples in $\mathcal{T}_{sub}$ are sampled i.i.d.).
- **Second term**: Deviation between the U-statistic and the population risk $\rightarrow$ U-statistic decoupling + McDiarmid's inequality.

### Contrast Loss

Standard logistic (N-pair) loss: $\ell(\mathbf{v})=\log(1+\sum_{i=1}^k\exp(-\mathbf{v}_i))$

### Core Theorems

**Theorem 5.1** (U-statistic Minimizer): With probability $\geq 1-\delta$,

$$ER_{un}(\hat{f}_{\mathcal{U}}) \leq \mathcal{O}\left[\sum_c\rho(c)\frac{K_{\mathcal{F},c}}{\sqrt{\tilde{N}}}+\mathcal{M}\sqrt{\frac{\ln|\mathcal{C}|/\delta}{\tilde{N}}}\right]$$

where $\tilde{N}=N\min(\frac{\min_c\rho(c)}{2},\frac{1-\max_c\rho(c)}{k})$.

**Theorem 5.2** (Sub-sampled Minimizer): With an additional $\mathcal{O}(1/\sqrt{M})$ term.

## Key Experimental Results

### Main Results

| Representation Function Class | Estimator | Generalization Bound (Balanced Classes, Large $k$) |
|---|---|---|
| Linear Mapping | $\hat{f}_{\mathcal{U}}$ | $\tilde{\mathcal{O}}(\eta sab^2/\sqrt{\tilde{N}})$ |
| Neural Network | $\hat{f}_{\mathcal{U}}$ | $\tilde{\mathcal{O}}(\mathcal{M}\mathcal{W}^{1/2}/\sqrt{\tilde{N}})$ |
| Linear Mapping | $\hat{f}_{sub}$ | Additional $1/\sqrt{M}$ term |
| Neural Network | $\hat{f}_{sub}$ | Additional $1/\sqrt{M}$ term |

- MNIST experiments: As the number of sub-sampled tuples $M$ increases, the performance of $\hat{f}_{sub}$ converges to $\hat{f}_{\mathcal{U}}$.
- Synthetic data validation of the relationship between $|\mathcal{C}|$ and $k$ with sample complexity.
- Under balanced class distributions, the bound is $\mathcal{O}(\sqrt{|\mathcal{C}|/N})$ for small $k$ and $\mathcal{O}(\sqrt{k/N})$ for large $k$.

## Highlights & Insights

1. **First to Solve Non-IID Generalization Analysis of CRL**: The class-level U-statistics avoid the difficult construction of $(k+2)$-order U-statistics.
2. **Consistent with Previous IID Results**: Under the $N=nk$ substitution, the bound aligns with the order of magnitude in Lei et al.
3. **Practical Guidance**: When it is impossible to traverse all tuples, increasing $M$ is sufficient to approach the theoretical optimum.

## Limitations & Future Work

- The bound contains $\sqrt{|\mathcal{C}|}$ under small $k$, which stems from class-independent analysis.
- Only supervised CRL is considered, without extension to self-supervised learning.
- Experiments are only validated on MNIST and synthetic data.

## Related Work & Insights

- Arora et al. (2019): Foundations of theoretical framework for CRL, i.i.d. tuple excess risk bound.
- Lei et al. (2023): Improved $k$-dependency to logarithmic order.
- Clémençon et al. (2008): Decoupling of U-statistics for ranking.
- Hieu et al. (2024): Generalization of CRL in DNNs.
- The methodology of this paper can inspire generalization analyses for non-i.i.d. sampling scenarios such as online learning and active learning.

## Rating

⭐⭐⭐⭐ — Solid theoretical work that fills the gap in non-IID generalization analysis for CRL, offering significant reference value for the contrastive learning theory community.


## Rating
- **Novelty**: To be evaluated
- **Experimental Thoroughness**: To be evaluated
- **Writing Quality**: To be evaluated
- **Value**: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](../../ICML2026/self_supervised/a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](../../ICML2026/self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](../../ICLR2026/self_supervised/understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)

</div>

<!-- RELATED:END -->
