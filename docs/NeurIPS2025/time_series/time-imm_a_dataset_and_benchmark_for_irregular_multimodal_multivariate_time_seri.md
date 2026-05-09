---
title: >-
  [Paper Note] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series
description: >-
  [NeurIPS 2025][Time Series][irregular time series] This work constructs Time-IMM — the first multimodal multivariate time series benchmark that categorizes irregularity according to causal mechanisms (9 irregularity types organized into three classes: Trigger, Constraint, and Artifact, spanning 9 datasets). An accompanying forecasting library, IMM-TSF, supports asynchronous multimodal fusion. Experiments demonstrate that explicitly modeling multimodal information reduces MSE by 6.71% on average across irregular time series settings, with a maximum improvement of 38.38%.
tags:
  - NeurIPS 2025
  - Time Series
  - irregular time series
  - multimodal fusion
  - causally-driven irregularity
  - time series forecasting
  - benchmark
date: 2026-05-08
content_hash: dcaa22eb6e8f281e
---

# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series

**Conference**: NeurIPS 2025
**arXiv**: [2506.10412](https://arxiv.org/abs/2506.10412)
**Code**: [https://github.com/blacksnail789521/Time-IMM](https://github.com/blacksnail789521/Time-IMM)
**Area**: Time Series / Multimodal
**Keywords**: irregular time series, multimodal fusion, causally-driven irregularity, time series forecasting, benchmark

## TL;DR
This work constructs Time-IMM — the first multimodal multivariate time series benchmark that categorizes irregularity according to causal mechanisms (9 irregularity types organized into three classes: Trigger, Constraint, and Artifact, spanning 9 datasets). An accompanying forecasting library, IMM-TSF, supports asynchronous multimodal fusion. Experiments demonstrate that explicitly modeling multimodal information reduces MSE by 6.71% on average across irregular time series settings, with a maximum improvement of 38.38%.

## Background & Motivation

**Background**: Time series analysis is widely applied in domains such as healthcare, climate science, and finance. Existing benchmarks (UCR / M4 / Time-MMD) assume clean, uniformly sampled, unimodal data.

**Limitations of Prior Work**: (a) Real-world time series data is inherently irregular (variable sampling rates, asynchronous modalities, pervasive missingness); (b) existing irregular time series methods (GRU-D / Raindrop / mTAND) handle only unimodal numerical sequences; (c) no systematic taxonomy of irregularity causes exists — yet different causes require different modeling strategies.

**Key Challenge**: Models must simultaneously handle irregular sampling and cross-modal asynchronous fusion, yet no data or evaluation tools reflecting real-world complexity are available.

**Goal**: Construct the first multimodal time series benchmark that classifies irregularity by causal mechanism, together with a companion forecasting library.

**Key Insight**: Irregularity is categorized by cause into three major classes and nine subtypes (Trigger / Constraint / Artifact), each paired with a real-world dataset, and each time series is paired with a text modality.

**Core Idea**: A causally-driven irregularity taxonomy combined with multimodal (numerical + text) asynchronous fusion drives time series analysis from idealized settings toward real-world complexity.

## Method

### Overall Architecture
Time-IMM comprises 9 datasets: Trigger-type (GDELT event-driven / RepoHealth adaptive / MIMIC clinician-recorded), Constraint-type (FNSPID trading window / ClusterTrace resource / StudentLife human routine), and Artifact-type (ILINet missingness / CESNET delay / EPA-Air asynchrony). Each dataset is paired with a text modality generated via GPT-4.1 Nano summarization with semantic filtering.

### Key Designs

1. **Irregularity Taxonomy**:

    - Function: Groups 9 irregularity subtypes into three classes based on causal mechanism.
    - Mechanism: Trigger — data collection is event-driven (external events / system responses / clinician judgment); Constraint — collection is restricted by operational windows / resource availability / human availability; Artifact — unintended missingness / delays / multi-source asynchrony.
    - Design Motivation: Different causal origins of irregularity require different modeling strategies (e.g., operational windows can exploit cyclic masks, whereas clinician-recorded data requires time-aware modeling).

2. **IMM-TSF Forecasting Library**:

    - Function: A plug-and-play benchmark library supporting asynchronous multimodal time series forecasting.
    - Mechanism: Text encoding (a timestamp-to-text fusion module converts timestamps into textual descriptions) → multimodal fusion (recency-aware averaging or attention-based strategy) → standard time series forecasting backbone.
    - Design Motivation: Existing time series forecasting libraries do not support integration of asynchronous text modalities.

3. **Irregularity Metrics**:

    - Function: Quantify the degree of irregularity in time series data.
    - Mechanism: Feature observability entropy (distribution of missingness across features), temporal observability entropy (distribution of observations over time), and mean inter-observation interval.
    - Design Motivation: Provide objective measures rather than subjective descriptions.

### Loss & Training
- Standard time series forecasting losses (MSE / MAE).
- Text modalities are generated as five-sentence summaries via GPT-4.1 Nano; only semantically relevant documents are retained.

## Key Experimental Results

### Main Results (Multimodal vs. Unimodal Forecasting)

| Dataset | Irregularity Type | Unimodal MSE | +Text Modality MSE | Gain |
|--------|----------|-----------|-------------|------|
| GDELT | Event-triggered | Baseline | **−38.38%** | Highest text informativeness |
| MIMIC | Clinician-recorded | Baseline | Significant reduction | Clinical notes are critical |
| Overall Average | Mixed | Baseline | **−6.71%** | Consistent improvement |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| w/o text modality | MSE increases | Text provides context not captured by numerical data |
| Recency-aware vs. Attention fusion | Comparable / case-dependent | Depends on text information density |
| w/o timestamp-to-text fusion | Performance drops | Temporal alignment is important for asynchronous fusion |

### Key Findings
- **Datasets with higher text informativeness yield larger improvements** (GDELT 38.38% vs. smaller gains on ILINet).
- **Different irregularity types pose distinct challenges**: Artifact-type (missingness / delay) impacts all models most severely.
- **Existing time series models perform substantially worse on Time-IMM than on standard benchmarks** — confirming the challenge posed by real-world irregularity.
- **Text modalities are most valuable for Trigger-type datasets** — event descriptions directly explain the triggers behind sampling.

## Highlights & Insights
- The insight that **"irregularity has causal structure"** is profound — different causes produce different patterns and require different strategies.
- This work is the first to systematically pair text modalities with a time series benchmark, advancing multimodal time series research.
- The 9 datasets span 6 domains (geopolitics / healthcare / finance / cloud computing / education / environment), providing exceptional diversity.

## Limitations & Future Work
- Text summaries are generated by GPT-4.1 Nano rather than sourced from original documents (due to copyright restrictions).
- Only forecasting tasks are evaluated; classification and anomaly detection remain untested.
- Text modality informativeness is limited in certain datasets (e.g., ILINet contains only 650 text entries).
- Other modalities such as images are not explored.

## Related Work & Insights
- **vs. Time-MMD**: Time-MMD focuses on domain diversity but assumes regular sampling. Time-IMM places irregularity at its core.
- **vs. GRU-D / Raindrop / mTAND**: These methods handle only unimodal irregular time series. IMM-TSF adds text fusion.
- **vs. UCR / M4**: Classical benchmarks entirely disregard irregularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First causally-driven irregularity taxonomy + multimodal time series benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 datasets × multiple models × complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Taxonomy is clearly articulated; dataset construction pipeline is transparent.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in real-world time series benchmarking.

### Additional Technical Details
- The 9 datasets cover geopolitics (GDELT), open-source software (RepoHealth), clinical healthcare (MIMIC), finance (FNSPID), cloud computing (ClusterTrace), education (StudentLife), epidemiology (ILINet), networking (CESNET), and environmental monitoring (EPA-Air).
- Text modalities are five-sentence summaries generated by GPT-4.1 Nano; only semantically relevant documents are retained.
- IMM-TSF supports two fusion strategies: recency-aware averaging (time-weighted averaging) and attention-based integration.
- Three irregularity metrics are provided: feature observability entropy, temporal observability entropy, and mean inter-observation interval.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)

<!-- RELATED:END -->
