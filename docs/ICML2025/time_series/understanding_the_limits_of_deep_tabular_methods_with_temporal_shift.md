---
title: >-
  [Paper Note] Understanding the Limits of Deep Tabular Methods with Temporal Shift
description: >-
  [ICML 2025][Time Series][Tabular Data] This paper reveals the root causes of deep tabular models failing under temporal distribution shifts—namely, model selection failure caused by training lag and validation bias, and the loss of periodic/trend information in model representations. It proposes an improved temporal splitting strategy and a plug-and-play temporal embedding method based on Fourier series.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Tabular Data"
  - "Temporal Distribution Shift"
  - "Data Splitting Strategy"
  - "Temporal Embedding"
  - "Fourier Series"
date: 2026-05-08
content_hash: 31e3a026c9a622c9
---

# Understanding the Limits of Deep Tabular Methods with Temporal Shift

**Conference**: ICML 2025  
**arXiv**: [2502.20260](https://arxiv.org/abs/2502.20260)  
**Code**: [Available](https://github.com/LAMDA-Tabular/Tabular-Temporal-Shift)  
**Area**: Time Series  
**Keywords**: Tabular Data, Temporal Distribution Shift, Data Splitting Strategy, Temporal Embedding, Fourier Series

## TL;DR

This paper reveals the root causes of deep tabular models failing under temporal distribution shifts—namely, model selection failure caused by training lag and validation bias, and the loss of periodic/trend information in model representations. It proposes an improved temporal splitting strategy and a plug-and-play temporal embedding method based on Fourier series.

## Background & Motivation

**Background**: Deep tabular models (MLPs, Transformers, retrieval-based methods, etc.) have made significant progress on i.i.d. data, with some models even outperforming tree models on standard benchmarks. The TabReD benchmark introduces the concept of "temporal distribution shift," highlighting that real-world tabular data is inherently temporal in nature.

**Limitations of Prior Work**: When data exhibits temporal distribution shifts (trends, periodic variations), the performance of deep tabular models severely degrades. Specifically, on the TabReD benchmark, retrieval-based methods (e.g., TabR, ModernNCA) experience a sharp decline in performance under temporal splits, ranking far below tree-based models and MLP-PLR.

**Key Challenge**: TabReD employs temporal splits to simulate real-world deployment scenarios, but experiments reveal that even a simple random split can significantly boost performance. This counterintuitive phenomenon suggests that the issue lies not only in the model architectures themselves but also in the training protocols.

**Ours**: To systematically address this issue from two levels: (1) analyze the root causes of temporal split failure (training lag and validation bias) and propose an improved data splitting strategy, and (2) identify the loss of periodic and trend information in model representations and propose a temporal embedding method based on Fourier series.

**Key Insight**: Instead of designing new model architectures, the authors approach the problem from two practical perspectives—training protocol and feature engineering—to propose general, plug-and-play solutions.

## Method

### Overall Architecture

The proposed framework consists of two parts: an improved data splitting strategy (training protocol) and a temporal embedding method. These two components are orthogonal, can be stacked together, and are applicable to any deep tabular model.

### Key Designs

#### 1. Analyzing and Eliminating Training Lag

**Function**: Analyze and resolve the issue of the time interval (training lag) between the training set and the test set in temporal splitting.

**Mechanism**: In the original temporal split of TabReD, data close to the test time $T_{\text{train}}$ is used as the validation set rather than the training set, resulting in a time gap between training and testing data. However, samples closer to the testing window typically feature more similar distributions and are more valuable for training than for validation.

By designing a controlled experiment (split (a) vs. (b) in Figure 3) that keeps the validation and test sets fixed while only varying the training lag, results show that eliminating training lag yields an average performance improvement of 1.62%. The retrieval-based method ModernNCA benefits the most (+2.19%) because retrieval methods heavily rely on the quality of candidate samples, where a lag-free candidate pool better reflects the test-time distribution.

**Design Motivation**: Intuitively, data closer to the test time is more representative and should be prioritized for training rather than being wasted only on validation.

#### 2. Analyzing and Alleviating Validation Bias

**Function**: Analyze the impact of distribution shift differences between the validation set and the test set (validation bias) on model selection.

**Mechanism**: In temporal splitting, the training-validation gap is typically much smaller than the training-test gap, causing the degree of distribution shift in the validation set to mismatch the actual test-time shift. Consequently, model selection (early stopping, hyperparameter tuning) is guided by a less-shifted validation set, failing to reflect the challenges encountered during testing.

Through a controlled experiment (split (a) vs. (c)), keeping the training and test sets identical while only changing the degree of validation bias, reducing validation bias yields an average improvement of 0.59%. The ensemble method TabM demonstrates the most significant gain (+0.83%) because ensemble methods are naturally robust to training data quality (reducing variance) but remain sensitive to validation bias.

**Design Motivation**: Model selection heavily relies on the representativeness of the validation set; thus, the shift level of the validation set should align with the actual test-time shift.

#### 3. Temporal Directional Equivalence of Validation Sets

**Function**: Verify whether data from the opposite temporal direction can also serve as an effective validation set.

**Mechanism**: Visualizing the distribution distances between different time slices via MMD heatmaps reveals a regular diagonal stripe pattern, indicating that the distribution shift across equal time intervals is roughly uniform. This implies that taking data from the opposite temporal direction with the same time split interval as the test set can serve as an approximately equivalent validation set.

Experiments (split (b) vs. (d)) show that using an opposite-direction validation set leads to a minor 0.91% performance drop, which is significantly smaller than the gain of 1.62% from eliminating training lag. This indicates that adopting this strategy to maximize training data utilization is overall beneficial.

#### 4. Proposed Temporal Splitting Strategy

Based on the three findings above, the final splitting strategy is proposed:
- Minimize training lag to zero (symmetrically splitting on both sides of $T_{\text{train}}$).
- Take validation data from the opposite temporal direction with equivalent shift severity as the test set.
- Achieves comparable performance gains to random splitting (2.18% vs. 2.17% average gain) but with vastly superior stability (standard deviation increases by 16.69% vs. 153.81%).

#### 5. Temporal Embedding Method

**Function**: Design a plug-and-play embedding method for timestamps to inject temporal information into the models.

**Mechanism**: By comparing the MMD heatmaps of raw data and model-learned representations, it is observed that periodic and trend information is severely lost in model representations. To this end, a Fourier series expansion-based temporal embedding is designed:

$$\psi(t) = [\text{ReLU}(\text{Linear}(\text{Periodic}(t))), \text{Trend}(t)]$$

where the periodic part is a concatenation of multi-scale Fourier embeddings:

$$\text{Periodic}(t) = [\text{Fourier}(t, T_1), \ldots, \text{Fourier}(t, T_m)]$$

with each Fourier embedding defined as a $K$-th order expansion:

$$\text{Fourier}(t, T) = [\sin(\frac{2\pi k t}{T}), \cos(\frac{2\pi k t}{T})] \in \mathbb{R}^{2K}, \quad k \in \{1, \ldots, K\}$$

And the trend part is the z-score normalized timestamp: $\text{Trend}(t) = \text{z-score}(t)$

**Design Motivation**:
- Use predefined periodic priors (year, month, week, day) rather than learnable frequencies. Under temporal shifts, validation sets might not be perfectly accurate, making a fixed prior more stable (fixed prior +0.30% vs. learnable frequencies -2.20%).
- Aggregate Fourier coefficients via a learnable linear layer, utilizing ReLU to enhance sparsity.
- The trend term captures linear temporal shifts.
- This embedding implicitly enables adaptation: the model can learn stage-specific knowledge over time, automatically adjusting the mapping $f_t = g_t \circ h_t$ based on current timestamps after deployment.

### Loss & Training

- Use standard classification/regression losses, with no additional loss design.
- Hyperparameter search is conducted using Optuna with 100 trials (25 trials for FT-Transformer and TabR).
- Results are averaged over 15 random seeds for each set of experiments.
- For random splits, models are additionally run on 3 different random splits with 15 seeds each to average (total of 45 runs).
- Hyperparameters for the temporal embedding include the Fourier orders of each period ($2^1$ to $2^7$ or 0), the trend flag, and the embedding dimension.

## Key Experimental Results

### Main Results

Evaluation on 8 datasets (3 classification + 5 regression) from the TabReD benchmark:

| Method | Metric | Ours Split (Avg. Imp.) | Random Split (Avg. Imp.) | Stability Comparison (Std↓) |
|------|------|---------------------|---------------------|-------------------|
| MLP | AUC/RMSE | +3.50% | +4.30% | Ours Better |
| MLP-PLR | AUC/RMSE | +0.75% | +0.73% | Ours Better |
| FT-Transformer | AUC/RMSE | +2.78% | +3.76% | Ours Better |
| TabR | AUC/RMSE | +2.20% | +2.00% | Ours Better |
| ModernNCA | AUC/RMSE | +2.49% | +2.53% | Ours Better |
| TabM | AUC/RMSE | +1.25% | +1.51% | Ours Better |
| XGBoost | AUC/RMSE | +2.06% | +1.79% | Ours Better |
| CatBoost | AUC/RMSE | +2.37% | +2.09% | Ours Better |
| Overall Average | - | +2.18% | +2.17% | 16.7% vs. 154% |

Further improvement of temporal embedding on top of the proposed split:

| Embedding Type | MLP | MLP-PLR | ModernNCA | Average |
|---------|-----|---------|-----------|------|
| Num (Numerical) | -0.04% | -0.06% | -0.04% | -0.05% |
| Time (Time Decomposition) | -0.70% | -0.15% | -0.32% | -0.39% |
| PLR Embedding | +0.70% | +0.01% | +0.02% | +0.25% |
| Ours Temporal Embedding | +1.31% | +0.01% | +0.30% | +0.54% |

### Ablation Study

| Configuration | Key Metric (Avg. Imp.) | Description |
|------|---------------------|------|
| Eliminating training lag (b vs. a) | +1.62% | Data closer to the test time is more effective for training |
| Reducing validation bias (a vs. c) | +0.59% | Aligning validation shift severity improves generalization |
| Opposite-direction validation set (d vs. b) | -0.91% | The loss is far smaller than the grain from eliminating lag |
| Fixed period vs. Learnable period | +0.30% vs. -2.20% | Fixed priors are more reliable in shift scenarios |
| Embedding directly into backbone vs. Via numerical encoding | +0.41% vs. +0.30% | Incompatibility exists between temporal embedding and PLR encoding |

### Key Findings

1. **Random splitting is not always superior to temporal splitting**: Although random splitting shows comparable average performance, its standard deviation increases by 154%, indicating severe instability. When evaluated using the robustness score ($RS_k = \mu - k\sigma$), the proposed method shows more obvious advantages under high penalties.
2. **Retrieval-based methods are most sensitive to temporal shifts**: ModernNCA benefits the most from splitting improvements (+2.63%), because candidate set quality directly dictates retrieval performance.
3. **Model representations lose temporal information**: MMD heatmaps indicate that representations learned by MLPs retain only coarse-grained patterns (e.g., weekdays vs. weekends) while completely losing long-term periodic and trend information.
4. **Temporal embeddings restore temporal structures**: Incorporating temporal embeddings allows the MMD heatmaps of model representations to recover diagonal stripe patterns consistent with the raw data.
5. **Method is equally effective for new model families**: The effectiveness of the training protocol is also validated on Mambular (an autoregressive method) and TabPFN v2 (a general tabular foundation model).

## Highlights & Insights

1. **Precise problem isolation**: Rather than rushing to design a new model, the paper systematically dissects why current training protocols (data splitting) fail. It strictly isolates the impacts of training lag and validation bias through controlled variable experiments.
2. **Validation set directional equivalence**: The discovery that data in the opposite temporal direction can serve as an approximately equivalent validation set is both novel and practical, permitting the training set to fully exploit the most recent data close to the test window.
3. **Visual analysis of representation layers**: MMD heatmaps reveal the loss of temporal information from a representation-learning standpoint, providing direct motivation for the temporal embedding design.
4. **Practically-oriented design philosophy**: The temporal embedding utilizes fixed periodic priors instead of learnable frequencies. This choice proves more stable under temporal shifts—since the validation set itself is not fully reliable, overfitting to it can be detrimental.
5. **Plug-and-play**: The overall scheme (splitting strategy + temporal embedding) does not modify model architectures and can be directly applied to any tabular learning method.

## Limitations & Future Work

1. **Assumption of uniform sampling**: The directional equivalence of validation sets relies on the assumption that temporal shifts are approximately uniform across time slices. Performance is limited when the sample distribution is highly non-uniform (e.g., on the SH and EO datasets).
2. **Dependency of periodic priors on domain knowledge**: Periods within the temporal embedding (year/month/week/day) must be manually specified and may require tuning for data with non-standard periodic patterns.
3. **Limited to data with explicit timestamps**: The method requires explicit timestamp information, rendering it inapplicable to tabular data with implicit temporality but no timestamps.
4. **Neglecting non-linear trends**: The trend component is linearly represented using only z-scores, which might insufficiently capture complex, non-linear trends.
5. **Compatibility with numerical encodings**: Temporal embeddings encounter compatibility issues with numerical encodings such as PLR. Although alleviated by inputting directly into the backbone, this increases deployment complexity.

## Related Work & Insights

- **TabReD** (Rubachev et al., 2024): First to systematically propose temporal shift problems and benchmarks for tabular data; this work conducts an in-depth analysis and improvement upon it.
- **TabM** (Gorishniy et al., 2024): Ensemble methods demonstrate robustness under temporal shifts, inspiring the robustness analysis of different methods in this work.
- **PLR Embedding** (Gorishniy, 2022): A periodic embedding method for numerical features; this work uncovers its incompatibility with temporal embeddings.
- **Fourier Features** (Tancik et al., 2020; Li et al., 2021): The successful application of random Fourier features in positional encoding inspired the temporal embedding design in this work.
- **Wild-Time** (Yao et al., 2022): Explores real-world temporal shifts but formulates it as a domain-to-domain setup, neglecting temporal continuity.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the improved splitting strategy and temporal embedding are not entirely brand-new concepts, the systematic analytical framework and experimental design are highly convincing, and the insight into validation set directional equivalence is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 8 datasets, 11+ methods, rigorous control experiments, 15 random seeds, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly logical, progressing systematically from problem discovery $\rightarrow$ cause analysis $\rightarrow$ solution design, with exquisite and highly informative figures.
- Value: ⭐⭐⭐⭐ Imparts direct guidance to practical tabular learning, offering plug-and-play solutions that lower the barrier of entry, though limited to tabular data scenarios with explicit timestamps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](../../NeurIPS2025/time_series/selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](../../NeurIPS2025/time_series/frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[ICLR 2026\] The Forecast After the Forecast: A Post-Processing Shift in Time Series](../../ICLR2026/time_series/the_forecast_after_the_forecast_a_post-processing_shift_in_time_series.md)

</div>

<!-- RELATED:END -->
