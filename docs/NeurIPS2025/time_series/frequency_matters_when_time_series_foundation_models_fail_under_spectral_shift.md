---
title: >-
  [Paper Note] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift
description: >-
  [NeurIPS 2025][Time Series][Time Series Foundation Models] This paper identifies spectral shift—a mismatch between the dominant frequencies of downstream data and those covered by pretraining data—as the key reason for g…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Time Series Foundation Models"
  - "Spectral Shift"
  - "Generalization Failure"
  - "MOMENT"
  - "Player Engagement Prediction"
date: 2026-05-08
content_hash: 7cda7f483799533a
---

# Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift

**Conference**: NeurIPS 2025
**arXiv**: [2511.05619](https://arxiv.org/abs/2511.05619)  
**Code**: None  
**Area**: Time Series / Foundation Models
**Keywords**: Time Series Foundation Models, Spectral Shift, Generalization Failure, MOMENT, Player Engagement Prediction

## TL;DR
This paper identifies spectral shift—a mismatch between the dominant frequencies of downstream data and those covered by pretraining data—as the key reason for generalization failure of Time Series Foundation Models (TSFMs) in industrial settings. The hypothesis is validated through an industrial-scale mobile game player engagement prediction task and controlled synthetic experiments.

## Background & Motivation

**Background**: Time series foundation models (e.g., MOMENT, Chronos, TimesFM) achieve strong performance on public benchmarks and have been likened to the "BERT moment" for time series. These models are pretrained via self-supervised learning on large-scale heterogeneous time series corpora.

**Limitations of Prior Work**: In real-world industrial applications (e.g., mobile game player behavior prediction), TSFMs fall far short of domain-adapted fully supervised baselines (PatchTST) and even traditional methods (XGBoost). Success on public benchmarks does not imply industrial transferability.

**Key Challenge**: Unlike NLP, where linguistic structure and semantics exhibit strong cross-domain sharing patterns, time series data vary greatly across domains in sampling rate, periodicity, and non-stationarity. Even small shifts in dominant frequency can produce entirely different signal characteristics.

**Goal**: Why do TSFMs fail in industrial settings? Is spectral misalignment the core reason?

**Key Insight**: The paper conducts frequency-domain analysis to compare dominant frequency distributions between pretraining and downstream data, proposes the "spectral shift" hypothesis, and validates it through controlled experiments.

**Core Idea**: The primary cause of TSFM generalization failure is that the dominant frequencies of downstream data fall outside the spectral coverage of pretraining data—models memorize specific frequency bands rather than learning general temporal representations.

## Method

### Overall Architecture
Rather than proposing a new model, this paper validates the spectral shift hypothesis through three experimental tiers: (1) an industrial-scale Player Engagement Prediction (PEP) task to demonstrate TSFM failure in practice; (2) spectral analysis comparing pretraining vs. downstream data to provide evidence; and (3) controlled synthetic experiments to isolate the causal effect of spectral shift.

### Key Designs

1. **Industrial Case Study: Player Engagement Prediction (PEP)**:

    - **Function**: Predict player purchase behavior (binary classification) and engagement level (regression) within 30 days in the mobile game Candy Crush Saga.
    - **Data Setup**: 824,208 multivariate time series samples, each up to 512 game rounds in length, with 32 feature dimensions (progress, gameplay, resources, strategy, context). Time span: 226 days.
    - **Evaluation Protocol**: Player-holdout (different players in the same period) and temporal-holdout (zero-shot across time periods).
    - **Design Motivation**: A genuine industrial scenario with sampling rates and periodicity entirely distinct from public datasets.

2. **Spectral Analysis**:

    - **Function**: Compare the dominant frequency distributions of downstream datasets with MOMENT's pretraining datasets (FordA, FaultDetectionA, etc.).
    - **Mechanism**: FFT is computed for each time series to extract the top-5 dominant frequencies. The analysis reveals that the dominant frequencies of the downstream game data and the pretraining data occupy almost non-overlapping frequency ranges.
    - **Design Motivation**: To provide intuitive empirical evidence for the spectral shift hypothesis.

3. **Controlled Synthetic Experiments**:

    - **Function**: Construct "seen" and "unseen" frequency-band synthetic datasets to compare downstream performance of frozen TSFM encoders.
    - **Mechanism**: FFT is applied to each sequence in the pretraining dataset to extract the dominant frequency range $[f^{\text{low}}, f^{\text{high}}]$. Two groups of synthetic signals are generated: the seen group samples frequencies from $[f^{\text{low}}, f^{\text{high}}]$; the unseen group samples from $[f^{\text{low}}+\delta, f^{\text{high}}+\delta]$ (disjoint intervals). Synthetic signals are composed as superimposed sinusoids: $x(t) = \sum_j A_j \sin(2\pi f_j t + \phi_j) + \text{noise}$. Regression labels are the z-score normalization of the sum of frequencies: $\tilde{y} = (y - \mu_y) / \sigma_y$.
    - **Design Motivation**: To eliminate confounding non-frequency factors and isolate the causal impact of spectral shift on representation quality. The backbone is frozen and only a lightweight head is trained, so the evaluation targets the encoder's representational capacity rather than end-to-end performance.

### Loss & Training
- Synthetic experiments use the Adam optimizer (lr=$10^{-3}$) for 50 epochs.
- Cross-entropy loss for binary classification; MSE loss for regression.
- Model selection is based on validation MSE; all experiments are run 3 times with mean and standard deviation reported.

## Key Experimental Results

### Main Results: PEP Industrial Task

| Model | Accuracy↑ (player) | AUC↑ (player) | MSE↓ (player) | MAE↓ (player) |
|------|---------------------|---------------|---------------|---------------|
| XGBoost | 0.841 | 0.915 | 1.200 | 0.780 |
| TabNet | 0.836 | 0.911 | 1.304 | 0.852 |
| **PatchTST** | **0.939** | **0.982** | **0.518** | **0.489** |
| MOMENT-small | 0.758 | 0.791 | 2.250 | 1.151 |

MOMENT substantially underperforms the fully supervised PatchTST on all metrics (Accuracy gap: 18.1%, AUC gap: 19.1%) and even falls below XGBoost.

### Ablation Study: Regression Performance under Spectral Shift

| Pretraining Dataset | Seen MSE↓ | Unseen MSE↓ | Seen MAE↓ | Unseen MAE↓ |
|-------------|-----------|-------------|-----------|-------------|
| FordA | 0.333±0.010 | 0.366±0.005 | 0.439±0.005 | 0.457±0.005 |
| ElectricDevices | 0.644±0.002 | **0.952±0.003** | 0.559±0.001 | **0.791±0.004** |
| FaultDetectionA | 0.689±0.001 | **0.942±0.004** | 0.666±0.001 | **0.779±0.001** |
| FaultDetectionB | 1.129±0.172 | **2.005±0.266** | 0.875±0.084 | **1.140±0.034** |

Across all datasets, MSE and MAE on unseen frequency bands are consistently higher than on seen bands. MSE degradation reaches **47.8%** on ElectricDevices and **77.6%** on FaultDetectionB.

### Key Findings
- The spectral shift effect holds for classification tasks as well (e.g., AUC on ElectricDevices drops from 0.890 to 0.716).
- The degree of degradation correlates positively with the degree of spectral non-overlap.
- Even after z-score normalization of labels to eliminate scale differences, models still perform worse on unseen frequency bands, indicating that the problem lies in the quality of temporal representations rather than label distribution.

## Highlights & Insights
- **Spectral Shift Hypothesis**: A simple yet explanatory framework that provides an actionable diagnostic for TSFM generalization failure. In practice, spectral overlap analysis can be conducted before deployment.
- **Dual Validation (Industrial + Synthetic)**: The paper first identifies the problem in a real industrial scenario, then isolates the causal mechanism via controlled synthetic experiments—a methodological paradigm worth emulating.
- Practical recommendations for the TSFM community: (a) quantify spectral overlap between pretraining and downstream data; (b) introduce frequency-aware data augmentation or pretraining strategies; (c) benchmarks should explicitly evaluate spectral diversity.

## Limitations & Future Work
- Validation is limited to a single TSFM (MOMENT-small) and a single industrial domain (mobile gaming); broader generalization requires further verification.
- Synthetic experiments use sinusoidal signals, which cannot fully capture irregular sampling, bursty patterns, and regime shifts in real-world data.
- No concrete solutions for addressing spectral shift (e.g., frequency-aware pretraining or fine-tuning strategies) are proposed.
- The paper is relatively short (4-page main body + appendix), with limited technical depth.
- Other TSFMs (Chronos, TimesFM, Moirai) are not compared, leaving the generality of the conclusions open to question.

## Related Work & Insights
- **vs. Chronos/TimesFM/Moirai**: These TSFMs similarly excel on public benchmarks, but their industrial generalization remains unknown. The findings of this paper may apply to them as well.
- **vs. PatchTST**: Fully supervised in-domain training avoids spectral shift entirely, which explains its substantial performance advantage over TSFMs.
- The hypothesis can inspire new pretraining data collection strategies: ensuring that training corpora cover a broader spectral range.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The spectral shift hypothesis offers a novel and insightful perspective, though the experimental design is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to a single TSFM, one industrial case study, and synthetic experiments; coverage is narrow.
- **Writing Quality**: ⭐⭐⭐⭐ Logical and well-structured; the narrative arc from industrial observation to hypothesis to validation is smooth.
- **Value**: ⭐⭐⭐⭐ Offers significant practical guidance for TSFM deployment and serves as a timely warning to the community that "benchmark performance ≠ real-world performance."

## Related Papers

- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[ICML 2025\] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](../../ICML2025/time_series/when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
