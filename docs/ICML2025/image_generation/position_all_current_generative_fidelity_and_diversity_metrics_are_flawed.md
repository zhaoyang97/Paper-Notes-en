---
title: >-
  [Paper Note] Position: All Current Generative Fidelity and Diversity Metrics are Flawed
description: >-
  [ICML2025][Image Generation][Generative Model Evaluation] Position paper: This work systematically demonstrates that all existing generative model fidelity and diversity metrics (including six pairs of metrics such as Improved Precision/Recall, Density/Coverage, and α-precision/β-recall) suffer from extensive failures in carefully designed sanity checks, urging the community to invest more effort in developing more reliable evaluation metrics.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Generative Model Evaluation"
  - "fidelity/diversity metrics"
  - "precision/recall"
  - "synthetic data quality"
  - "sanity check"
date: 2026-05-08
content_hash: d88b75bf73dbba8d
---

# Position: All Current Generative Fidelity and Diversity Metrics are Flawed

**Conference**: ICML2025  
**arXiv**: [2505.22450](https://arxiv.org/abs/2505.22450)  
**Code**: [vanderschaarlab/position-fidelity-diversity-metrics-flawed](https://github.com/vanderschaarlab/position-fidelity-diversity-metrics-flawed)  
**Area**: Image Generation  
**Keywords**: Generative Model Evaluation, fidelity/diversity metrics, precision/recall, synthetic data quality, sanity check

## TL;DR

Position paper: This work systematically demonstrates that all existing generative model fidelity and diversity metrics (including six pairs of metrics such as Improved Precision/Recall, Density/Coverage, and α-precision/β-recall) suffer from extensive failures in carefully designed sanity checks, urging the community to invest more effort in developing more reliable evaluation metrics.

## Background & Motivation

The rapid development of generative models (such as GANs, diffusion models, and LLM-generated tabular data) relies heavily on reliable evaluation metrics. Traditional metrics like FID can only provide an overall quality score, failing to distinguish between different dimensions of generation quality. To address this, the community has proposed precision/recall-type metrics, splitting the evaluation into two dimensions:

- **Fidelity**: Whether the generated samples are realistic (specifically, whether synthetic samples lie within the real distribution).
- **Diversity**: Whether the generated distribution covers all the modes of the real distribution.

The current mainstream pairs of fidelity/diversity metrics include:

| Paper | Fidelity Metric | Diversity Metric |
|------|-------------|---------------|
| Kynkäänniemi et al., 2019 | Improved Precision (I-Prec) | Improved Recall (I-Rec) |
| Naeem et al., 2020 | Density | Coverage |
| Alaa et al., 2022 | Integrated α-precision (IAP) | Integrated β-recall (IBR) |
| Cheema & Urner, 2023 | Precision Cover (C-Prec) | Recall Cover (C-Rec) |
| Khayatkhoei & Abdalmageed, 2023 | Symmetric Precision (symPrec) | Symmetric Recall (symRec) |
| Park & Kim, 2023 | Probabilistic Precision (P-Prec) | Probabilistic Recall (P-Rec) |

Some existing studies have identified individual failures in these metrics (such as a lack of robustness to outliers, unclear upper and lower bounds, etc.), but each work only focuses on and patches a few issues, lacking a comprehensive and systematic evaluation. **Core Problem**: When existing metrics are compiled together and comprehensively tested under a unified set of standards, can any metric pass all the tests?

## Core Idea

This paper presents three main contributions:

1. **Six Desiderata**: Definitively establishes six criteria that synthetic data evaluation metrics should satisfy.
2. **14 Sanity Checks**: Translates reported failure cases in the literature into automated, simple tests.
3. **Systematic Evaluation**: Evaluates 6 pairs of metrics (12 metrics in total) across all sanity checks.

Core Position: **All current fidelity and diversity metrics are flawed**, and many fail to reliably measure the most fundamental properties they are designed to evaluate.

## Method

### Six Desiderata

| ID | Name | Requirement |
|------|------|------|
| D1 | Purpose | Measures quantities of direct practical value (D1a), provides interpretable information on distributional differences (D1b), or serves as a reliable proxy metric (D1c). |
| D2 | Hyperparameters | Minimizes the number of hyperparameters, with clear and controllable impacts. |
| D3 | Data | The required amount of real data is less than the practically available data, with a threshold set at 1000. |
| D4 | Bounds | Clear upper and lower bounds to allow absolute evaluation rather than only relative comparison. |
| D5 | Invariance | Remains invariant to transformations that do not affect data quality (such as scaling, permutation of categorical variables, etc.). |
| D6 | Computation | Can be computed within a reasonable timeframe. |

### Embedding Strategies

All metrics first embed data into a space more suitable for measuring geometric relationships:

- **Image Data**: Uses pretrained neural networks (such as InceptionV3).
- **Tabular Data**: One-hot encodes categorical variables and standardizes numerical variables to zero mean and unit variance. This simple embedding satisfies D5 (scaling invariance, category permutation invariance) without requiring extra hyperparameters.

### Design of 14 Sanity Checks

Each test utilizes artificially constructed real/synthetic distributions, focusing on a single potential issue with clear pass/fail criteria:

**Gaussian Tests (5 tests)**:

- **Gaussian Mean Difference**: Two Gaussian distributions differ only in their means, testing whether metrics can detect distribution shifts.
- **Gaussian Mean Diff + Outlier**: Introduces outliers to test robustness to outliers.
- **Gaussian Std Deviation & Difference**: Only standard deviations differ, testing sensitivity to differences in distribution width.
- **One Disjoint Dim + Many Identical Dim**: Disjoint in only one dimension while remaining identical in the rest, testing detection capability in high-dimensional settings.
- **Scaling One Dimension**: Applies scale transformation to one dimension to test D5 invariance.

**Gaussian Mixture Tests (3 tests)**:

- **Mode Collapse**: Two-mode real distribution vs. a single wide-mode synthetic distribution.
- **Mode Dropping + Invention**: The synthetic distribution gradually increases the number of modes, first covering real modes and then inventing new ones.
- **Sequential / Simultaneous Mode Dropping**: Dropping modes one by one or simultaneously decreasing weights among 10 modes.

**Hypercube/Hypersphere Tests (3 tests)**:

- **Hypercube, Varying Sample Size**: Fixed distributions with varying sample sizes to test D3.
- **Hypercube, Varying Syn. Size**: Fixed number of real samples with varying synthetic sample sizes to test D2.
- **Hypersphere Surface**: Uniform distribution on hypersphere surfaces of different radii, testing correctness in high-dimensional environments.

**Geometric/Tabular Tests (3 tests)**:

- **Sphere vs. Torus**: Non-overlapping distributions between a sphere and a torus.
- **Discrete Num. vs. Continuous Num.**: Gaussian distribution vs. its discretized (rounded) counterpart (a common scenario in tabular data).
- **Gaussian Mean Diff + Pareto**: Incorporates an additional dimension with a heavy-tailed Pareto distribution (common in tabular data).

### Pass/Fail Criteria

Each test is associated with one or more desiderata:
- **D1b**: The overall behavior of the metric is correct (trends in the correct direction).
- **D4**: The metric approaches the theoretical upper/lower bounds ($0$ or $1$) in extreme cases.
- **D3**: The metric converges stably when the sample size exceeds 1000.
- **D5**: Invariance under scaling transformations.

For diversity metrics, a **High/Low distinction** is introduced: when the synthetic distribution fully covers but is much wider than the real distribution, whether diversity is high or low depends on the definition of "coverage," allowing metrics to consistently choose either interpretation.

## Key Experimental Results

### Fidelity Metrics Results (Selected from Table 3)

| Sanity Check | I-Prec | Density | IAP | C-Prec | symPrec | P-Prec |
|---|---|---|---|---|---|---|
| Gaussian Mean Diff (D1b) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| + Outlier (D1b) | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Gaussian Std Diff (D1b) | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Hypercube Vary Size (D1b)| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Hypersphere Surface (D1b)| ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| Mode Drop+Invention (D1b)| ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 1 Disjoint + Many Ident (D1b) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Discrete vs Continuous (D1b)  | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Scaling One Dim (D5)     | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hypercube Vary Size (D3) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Key Findings**:

- **No single fidelity metric passes all tests**.
- **D3 (Data requirements) fails completely**: All metrics are unstable as the sample size changes.
- **D1b high-dimensional tests fail completely**: All metrics fail in the setup where "one dimension differs while multiple dimensions are identical."
- **Discrete vs Continuous fails completely**: No metric can distinguish between discrete and continuous numerical distributions.
- Density and P-Prec perform relatively well (passing more D1b tests) but still exhibit a large number of failures.
- I-Prec lacks robustness to outliers and scaling invariance.

### Diversity Metrics Results

Diversity metrics also exhibit widespread failures:
- All metrics fail on Hypercube Varying Sample Size (D3).
- The capability to distinguish between discrete and continuous distributions is universally insufficient.
- Coverage performs well on several D4 (upper/lower bounds) tests, but also faces significant failures on D1b.

### Core Conclusions & Practical Advice

1. **All metrics are flawed**—there is no "gold standard" metric that can be used blindly.
2. **Advice for practitioners**: When using these metrics, one must be aware of their limitations; a high score on a specific metric should not be interpreted as unconditionally good generation quality.
3. **Appeal to researchers**: The community should invest more effort in developing new evaluation metrics rather than just new models, and new metrics must be validated through extensive sanity checks.

## Highlights & Insights

1. **Extremely high methodological value**: Unifies scattered failure cases from various papers into a reproducible programmatic test suite, establishing a standardized benchmark.
2. **Comprehensive Desiderata framework**: The six criteria capture the core needs of metric design and can serve as a reference standard for future metric designs.
3. **High/Low diversity distinction**: Provides a reasonable handling of the ambiguity in diversity metrics, avoiding unfair evaluations.
4. **Inclusion of tabular data**: Incorporates test scenarios unique to tabular data (heavy-tailed distributions, discrete vs. continuous), filling a gap in previous image-centric evaluations.
5. **Open-source code**: All sanity check code is open-sourced, facilitating replication and expansion by subsequent researchers.

## Limitations & Future Work

1. **Identification without remedy**: As a position paper, it only exposes problems without proposing concrete alternative metrics.
2. **Use of artificial distributions for sanity checks**: All tests are based on synthetic, simple distributions (Gaussian, hypercube, etc.), which may behave differently from real-world data (natural images, complex tables).
3. **Lack of deep analysis on embedding in the image domain**: No systematic evaluation of the biases introduced by pretrained embeddings such as InceptionV3.
4. **Exclusion of curve-valued metrics**: Metrics that return curves rather than scalar values (e.g., Sajjadi et al. 2018) are excluded.
5. **Certain metrics excluded due to computational cost**: Topological metrics like Kim et al. 2023 were excluded, though they might have unique advantages.
6. **Lack of discussion on evaluation scenarios for modern diffusion models**: The evaluation needs of modern large-scale diffusion models may differ from traditional GAN evaluations.

## Related Work & Insights

- **Borji (2019, 2022); Xu et al. (2018)**: Early GAN evaluation metric surveys proposed partially overlapping desiderata, but required a single metric to simultaneously measure multiple aspects.
- **Theis et al. (2016)**: Discovered that classical metrics could yield contradictory evaluations.
- **Theis (2024)**: Theoretically explored the properties that fidelity ("realism") metrics should possess.
- **Sajjadi et al. (2018) $\rightarrow$ Kynkäänniemi et al. (2019)**: Pioneering and improvements of precision/recall metrics.
- **Insights for future metric design**: There is a need to establish systematic sanity check validation workflows from the very beginning of the design phase, rather than patching issues after they are discovered post-hoc.

## Rating
- Novelty: ⭐⭐⭐ (As a position paper, it does not propose a new method, but the systematic evaluation framework is innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive coverage with 14 sanity checks × 12 metrics)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous desiderata-check-result logic)
- Value: ⭐⭐⭐⭐ (Serves as a wake-up call for the community, and the sanity check suite can become a standard validation tool)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials](all-atom_diffusion_transformers_unified_generative_modelling_of_molecules_and_ma.md)
- [\[NeurIPS 2025\] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](../../NeurIPS2025/image_generation/continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](../../ICML2026/image_generation/balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](../../NeurIPS2025/image_generation/pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)

</div>

<!-- RELATED:END -->
