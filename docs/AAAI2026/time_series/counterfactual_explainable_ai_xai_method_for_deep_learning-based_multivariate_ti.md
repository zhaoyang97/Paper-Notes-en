---
title: >-
  [Paper Note] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification
description: >-
  [AAAI 2026][Time Series][Counterfactual Explanation] This paper proposes CONFETTI, a multi-objective counterfactual explanation method for multivariate time series (MTS) classification. By combining Class Activation Map (CAM)-guided subsequence extraction with NSGA-III multi-objective optimization, CONFETTI achieves an optimal balance among prediction confidence, sparsity, and proximity, outperforming existing methods across 7 UEA benchmark datasets.
tags:
  - AAAI 2026
  - Time Series
  - Counterfactual Explanation
  - Multivariate Time Series
  - Multi-Objective Optimization
  - Explainable AI
  - NSGA-III
date: 2026-05-08
content_hash: 7c6d4d30ac03c819
---

# Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification

**Conference**: AAAI 2026
**arXiv**: [2511.13237](https://arxiv.org/abs/2511.13237)
**Code**: [https://github.com/serval-uni-lu/confetti](https://github.com/serval-uni-lu/confetti)
**Area**: Time Series / Explainable AI
**Keywords**: Counterfactual Explanation, Multivariate Time Series, Multi-Objective Optimization, Explainable AI, NSGA-III

## TL;DR
This paper proposes CONFETTI, a multi-objective counterfactual explanation method for multivariate time series (MTS) classification. By combining Class Activation Map (CAM)-guided subsequence extraction with NSGA-III multi-objective optimization, CONFETTI achieves an optimal balance among prediction confidence, sparsity, and proximity, outperforming existing methods across 7 UEA benchmark datasets.

## Background & Motivation

Deep learning models (CNNs, RNNs, Transformers, etc.) have achieved strong performance on MTS classification tasks, yet their "black-box" nature severely hinders decision-makers' ability to understand and trust model predictions. While existing XAI methods offer partial insights, they fall short in revealing the full decision space. Counterfactual Explanations (CE) address this gap by showing "what minimal changes to the input would alter the prediction," but current MTS counterfactual approaches exhibit a fundamental **Key Challenge**:

- **CoMTE / AB-CE**: Focus on maximizing prediction confidence but may require large modifications to the original time series.
- **SETS / LASTS**: Emphasize proximity but may generate out-of-distribution instances.
- **TSEvo**: Although multi-objective, it relies on prior-free population search, which is computationally expensive and inefficient for high-dimensional or long time series.

**Core Idea**: CONFETTI introduces CAM weights as prior knowledge to guide the search process through a four-step pipeline — finding the Nearest Unlike Neighbor (NUN) → extracting the most influential subsequence → generating seed CEs via naive substitution → multi-objective optimization with NSGA-III — simultaneously optimizing for confidence, sparsity, and proximity, while guaranteeing validity and plausibility by design.

## Method

### Overall Architecture
CONFETTI consists of four stages:
1. **NUN Retrieval**: Identifies the nearest neighbor instance with a different predicted class from the instance to be explained.
2. **Subsequence Extraction**: Leverages CAM weights to locate the most influential subsequence.
3. **Naive Stage**: Replaces the original subsequence with the corresponding NUN subsequence to generate an initial CE.
4. **Optimization Stage**: Applies NSGA-III multi-objective optimization to balance the three objectives.

### Key Designs

1. **Nearest Unlike Neighbor (NUN) Retrieval**:

    - **Function**: Finds the nearest neighbor in reference set $R$ whose predicted class differs from query instance $X_i$.
    - **Mechanism**: Filters candidates by class, performs k-NN search, and retains candidates whose classifier confidence exceeds threshold $\theta$.
    - **Design Motivation**: Using instances from the real data distribution as counterfactual targets naturally guarantees the plausibility of generated CEs.

2. **CAM-Guided Subsequence Extraction**:

    - **Function**: Uses the NUN's CAM weights to identify the subsequence of length $\ell$ with the maximum cumulative weight via a sliding window.
    - **Mechanism**: Averages CAM weights across channels, then performs a linear scan to locate the most important contiguous time segment.
    - **Design Motivation**: Restricting modifications to regions the model attends to most avoids indiscriminate replacement of the entire sequence, thereby improving sparsity.

3. **NSGA-III Multi-Objective Optimization**:

    - **Function**: Refines the initial CE through evolutionary search to simultaneously optimize the three objectives.
    - **Mechanism**: Employs binary search to narrow the time window, Das-Dennis reference point generation, two-point crossover, and bit-flip mutation to evolve the population.
    - **Design Motivation**: Enforces the validity constraint $P(f(C_j)=c) \geq \theta$ while discovering the Pareto front between sparsity and proximity.

### Loss & Training

CONFETTI frames optimization as a three-objective problem (not a deep learning loss):
- **$m_1$ (maximize)**: Sum of prediction confidences, measuring the degree to which counterfactuals are accepted by the target class.
- **$m_2$ (minimize)**: Normalized Hamming distance, measuring the proportion of modified elements (sparsity).
- **$m_3$ (minimize)**: $L_1$/$L_2$/DTW distance, measuring the magnitude of modifications (proximity).

Constraint: Each CE must achieve target-class confidence no lower than threshold $\theta$. Users can adjust preference between confidence and sparsity via weight parameter $\alpha \in [0,1]$.

## Key Experimental Results

### Main Results

Experiments are conducted on 7 UEA datasets with 2 model architectures (FCN and ResNet), comparing against three baselines: CoMTE, SETS, and TSEvo.

**Confidence Comparison** ($\theta=0.95$, averaged across models):

| Dataset | CoMTE | SETS | CONFETTI ($\theta$=0.95) |
|--------|-------|------|-------------------|
| AWR | 0.953 | 0.940 | **0.978** |
| BasicMotions | 0.917 | 0.487 | **0.965** |
| ERing | 0.701 | 0.766 | **0.981** |
| NATOPS | 0.755 | * | **0.976** |
| Average | 0.86 | - | **0.98** |

**Sparsity Comparison** ($\alpha=0.0$):

| Dataset | CoMTE | TSEvo | CONFETTI ($\alpha$=0.0) |
|--------|-------|-------|-------------------|
| AWR | 0.731 | 0.002 | **0.926** |
| BasicMotions | 0.486 | 0.003 | **0.822** |
| Epilepsy | 0.461 | 0.011 | **0.822** |
| Average | 0.56 | 0.01 | **0.88** |

### Ablation Study

| Configuration | Key Metrics | Notes |
|------|---------|------|
| FCN full metrics ($\alpha$=0.5, $\theta$=0.95) | COV=100%, VAL=1.00, CONF=0.97, SPA=0.81 | Best trade-off with full method |
| FCN ($\alpha$=0.0, $\theta$=0.51) | SPA=0.88, CONF=0.59 | Maximizing sparsity |
| FCN ($\alpha$=0.5, $\theta$=0.51) | SPA=0.85, CONF=0.69 | Balanced setting |
| Without CAM weights | Subsequence extraction and naive stage skipped | Performance degrades but still operates as a model-agnostic method |

### Key Findings

- CONFETTI is the only method achieving 100% coverage and 100% validity across all datasets and models.
- Sparsity exceeds CoMTE by an average of 32 percentage points and TSEvo by 87 percentage points.
- The $\theta$ parameter allows users to flexibly switch between high-confidence ($\theta=0.95$) and high-sparsity ($\theta=0.51$) scenarios.
- All methods achieve a yNN score of 0.99, indicating that all generated CEs exhibit strong plausibility.

## Highlights & Insights

- **Introduction of CAM priors** is the most critical contribution: it compresses the originally prior-free search space to the subsequences most attended to by the model, substantially improving efficiency and sparsity.
- **Theoretical guarantee**: Theorem 1 proves that the Hamming distance of all CEs produced in the optimization stage is no greater than that of the initial CE, ensuring the optimization process only improves or maintains sparsity.
- The **binary search strategy** for time window length is particularly elegant, avoiding exhaustive search.
- The two parameters $\alpha$ and $\theta$ provide flexibility across diverse application scenarios.

## Limitations & Future Work

- Reliance on CAM extraction restricts applicability to model architectures that include global average pooling (e.g., FCN, ResNet); arbitrary models are not supported.
- Without CAM, the method degrades to a model-agnostic mode, though the overall framework handles this gracefully by design.
- Validation is limited to UEA datasets (maximum 207 time steps); scalability to very long time series remains to be verified.
- Human-interpretability evaluation of the counterfactual explanations is absent, as no user study was conducted.

## Related Work & Insights

- CoMTE is the first MTS counterfactual method, but its channel-wise replacement strategy leads to poor sparsity.
- TSEvo performs prior-free search via NSGA-II, resulting in high computational cost and unstable solution quality.
- The CAM-guided search paradigm proposed in this paper can be generalized to counterfactual explanation in other modalities (images, text).
- The application of NSGA-III in modular multi-objective optimization offers valuable methodological reference.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[AAAI 2026\] HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
