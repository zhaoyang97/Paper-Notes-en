---
title: >-
  [Paper Note] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification
description: >-
  [NeurIPS 2025][Time Series / Generative Model Theory][Diffusion models] This paper analyzes the sample complexity and uncertainty quantification performance of conditional diffusion Transformers (DiT) for time series imp…
tags:
  - "NeurIPS 2025"
  - "Time Series / Generative Model Theory"
  - "Diffusion models"
  - "Transformer"
  - "time series imputation"
  - "uncertainty quantification"
  - "statistical learning theory"
date: 2026-05-08
content_hash: a7e0967cabe11c5a
---

# Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification

**Conference**: NeurIPS 2025
**arXiv**: [2510.02216](https://arxiv.org/abs/2510.02216)  
**Code**: [Available](https://github.com/liamyzq/DiT_time_series_imputation)  
**Area**: Time Series / Generative Model Theory
**Keywords**: Diffusion models, Transformer, time series imputation, uncertainty quantification, statistical learning theory

## TL;DR

This paper analyzes the sample complexity and uncertainty quantification performance of conditional diffusion Transformers (DiT) for time series imputation from a statistical learning perspective, and proposes a mixed-masking training strategy to improve imputation quality.

## Background & Motivation

Time series data are ubiquitous in finance, healthcare, transportation, and meteorology, yet frequently contain large proportions of missing values due to sensor failures, data transmission errors, and similar causes. Missing values significantly degrade downstream task performance, making accurate imputation critical.

Traditional statistical methods (mean imputation, interpolation, Kalman filtering, ARIMA, etc.) rely on strong assumptions such as linearity and stationarity, and struggle with complex nonlinear data. Recent diffusion-based generative imputation methods (e.g., CSDI) have demonstrated superior empirical performance, yet two key issues remain:

1. Diffusion models exhibit large performance variance across datasets, leading to instability.
2. Imputation quality is heavily influenced by the missing pattern.

The core problem of this paper: **How well can diffusion models capture the conditional distribution of missing values? How does the missing pattern affect imputation performance?**

## Method

### Overall Architecture

The paper uses Gaussian process (GP) data as the theoretical object of analysis and studies the statistical efficiency of DiT for imputation. The core mechanism frames imputation as conditional distribution estimation: given the observed sequence $x_{\text{obs}}$, estimate the conditional distribution $P(x_{\text{miss}} | x_{\text{obs}})$ of the missing values.

**Data model**: A $d$-dimensional Gaussian process with sequence length $H$ is considered, with joint distribution $\mathcal{N}(\mu, \Gamma \otimes \Lambda)$, where $\Gamma$ encodes temporal correlation and $\Lambda$ encodes spatial dependence.

### Key Designs

**1. Transformer Approximation Theory for Conditional Score Functions (Theorem 1)**

The authors present a novel constructive proof for DiT, employing algorithm unrolling to demonstrate that Transformers can effectively approximate the conditional score function of Gaussian processes. Key steps include:
- Decomposing the conditional score function via orthogonal bases
- Capturing positional embeddings and temporal dependencies through attention mechanisms
- Realizing nonlinear transformations via MLP layers

**2. Statistical Sample Complexity (Theorem 2)**

An upper bound on the sample complexity for DiT to learn the conditional distribution is established:
$$\tilde{O}\left(\frac{\sqrt{Hd^2\kappa^5}}{\sqrt{n}}\right)$$
where $n$ is the number of training samples, $H$ is the sequence length, $d$ is the dimensionality, and $\kappa$ is the condition number of the conditional covariance matrix determined by the missing pattern. Key findings:
- The convergence rate is $n^{-1/2}$, with only mild polynomial dependence on sequence length $H$.
- The condition number $\kappa$ directly characterizes the difficulty of imputation induced by the missing pattern.

**3. Uncertainty Quantification (Corollary 1)**

The trained DiT is used to generate a large number of samples for missing values, from which confidence regions (CRs) are constructed. The coverage probability is shown to converge to the nominal level at rate $\tilde{O}(n^{-1/2})$.

### Loss & Training

**Mixed-Masking Training Strategy**

Motivated by the theoretical analysis, a training strategy that mixes diverse missing patterns is proposed:
- **S1**: 100% random missing (16×1)
- **S2**: 50% random + 50% weakly grouped (8×2)
- **S3**: 33% random + 33% weakly grouped + 33% moderately grouped (4×4)
- **S4**: 25% random + 25% weakly grouped + 25% moderately grouped + 25% strongly grouped (1×16)

Core Idea: Introducing a curriculum of missing patterns ranging from easy to hard during training reduces the distribution shift between training and test distributions.

## Key Experimental Results

### Main Results

**Confidence Region Coverage on Gaussian Process Data (Tables 1–2)**

| Sequence Length H | 16 | 32 | 64 | 96 | 128 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CR Coverage (%) | 92.67 | 88.63 | 82.14 | 80.25 | 77.81 |

CR coverage (%) under different training strategies and missing patterns:

| Strategy | P1 (κ=415) | P2 (κ=30) | P3 (κ=9.5) | P4 (κ=3.0) |
|:---:|:---:|:---:|:---:|:---:|
| S1 (pure random) | 34.58 | 58.46 | 72.42 | 80.25 |
| S4 (mixed) | 57.27 | 79.00 | 74.38 | 82.74 |

**MSE Comparison on Latent Gaussian Processes (Table 3)**

| Model | P1-S4 | P2-S4 | P3-S4 | P4-S4 |
|:---:|:---:|:---:|:---:|:---:|
| DiT | **0.67** | **0.62** | **0.58** | **0.53** |
| CSDI | 0.68 | 0.63 | 0.61 | 0.58 |
| GPVAE | 5.28 | 4.84 | 4.59 | 4.45 |

**MAE on Real Datasets (Table 6, Appendix)**

| Model | ETT_m1 10% | ETT_m1 50% | BeijingAir 10% | BeijingAir 50% |
|:---:|:---:|:---:|:---:|:---:|
| DiT | **0.1269** | **0.1543** | **0.1753** | **0.2057** |
| CSDI | 0.1448 | 0.1650 | 0.1780 | 0.2141 |
| GP-VAE | 0.2786 | 0.4666 | 0.4152 | 0.5265 |

### Ablation Study

Ablation of the mixed-masking strategy:
- Using any single pattern alone (8×2, 4×4, or 1×16) consistently underperforms the mixed strategy.
- Distribution shift coefficient analysis: S4 reduces the distribution shift coefficient by approximately **47.93×** compared to S1, providing strong theoretical support.

### Key Findings

1. **Condition number is the key metric**: A lower $\kappa$ (missing points spaced farther apart) implies easier imputation and lower sample requirements.
2. **Consistent advantage of mixed-masking training**: The mixed strategy outperforms pure random masking across all missing patterns.
3. **DiT consistently outperforms CSDI**: DiT leads on both MSE and CR coverage, indicating that the Transformer architecture is better suited to this task.
4. **Theory aligns with experiment**: Increasing sequence length degrades coverage, and patterns with lower condition numbers are easier to estimate, both consistent with theoretical predictions.

## Highlights & Insights

1. **First end-to-end statistical guarantees for diffusion Transformer imputation**: The analysis covers not only distribution estimation but also uncertainty quantification.
2. **Theory-driven method design**: The mixed-masking strategy is directly motivated by theoretical results on distribution shift, rather than empirical tuning.
3. **Algorithm unrolling for constructive proof**: Algorithm unrolling is innovatively employed to construct a Transformer that approximates the conditional score function.
4. **Natural confidence interval construction**: The sampling capability of the generative model is directly leveraged to build confidence regions, yielding a concise and effective approach.

## Limitations & Future Work

1. The theoretical analysis is restricted to Gaussian process data; applicability to heavy-tailed distributions (e.g., financial data) remains to be studied.
2. The optimal mixture ratio for the masking strategy is instance-dependent, and no adaptive selection method is currently available.
3. Experiments are conducted primarily on synthetic data and small-scale real datasets; validation in large-scale real-world scenarios is insufficient.
4. Only the block-missing setting is considered; analysis of random point-wise missing is not addressed.

## Related Work & Insights

- **CSDI** [Tashiro et al., 2021]: The first conditional diffusion method for time series imputation and the primary baseline in this paper.
- **DiT** [Peebles and Xie, 2022]: The diffusion Transformer architecture serving as the backbone network in this work.
- **GPVAE** [Fortuin et al., 2020]: A VAE-based generative imputation method; experiments show it falls substantially short of diffusion-based approaches.
- Diffusion model theory [Chen et al., 2023; Fu et al., 2024]: Provides the theoretical foundations for the analysis in this paper.

## Rating

- **Novelty**: ★★★★☆ — Solid theoretical contributions; the mixed-masking strategy is simple yet theoretically grounded.
- **Technical Depth**: ★★★★★ — Involves advanced statistical learning theory with high technical rigor.
- **Experimental Thoroughness**: ★★★☆☆ — Synthetic data experiments are thorough; real-data experiments are limited.
- **Writing Quality**: ★★★★☆ — Clear paper structure with a good balance between theory and experiments.
- **Value**: ★★★☆☆ — Primarily theoretical; the mixed-masking strategy offers moderate practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](../../ICML2026/learning_theory/mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[NeurIPS 2025\] On Agnostic PAC Learning in the Small Error Regime](on_agnostic_pac_learning_in_the_small_error_regime.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)

</div>

<!-- RELATED:END -->
