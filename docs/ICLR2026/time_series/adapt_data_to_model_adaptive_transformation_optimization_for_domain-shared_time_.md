---
title: >-
  [Paper Note] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models
description: >-
  [ICLR 2026][Time Series][Time series foundation models] This paper proposes TATO, a framework that automatically optimizes data preprocessing pipelines (including context trimming, scale normalization…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Time series foundation models"
  - "data transformation optimization"
  - "zero-shot forecasting"
  - "frozen model inference"
  - "Bayesian optimization"
date: 2026-05-08
content_hash: ff3d74fb80039618
---

# Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2603.00629](https://arxiv.org/abs/2603.00629)
**Code**: [https://github.com/thulab/TATO](https://github.com/thulab/TATO)
**Area**: Self-supervised Learning / Time Series Forecasting
**Keywords**: Time series foundation models, data transformation optimization, zero-shot forecasting, frozen model inference, Bayesian optimization

## TL;DR
This paper proposes TATO, a framework that automatically optimizes data preprocessing pipelines (including context trimming, scale normalization, and outlier correction) to adapt frozen large time-series models (LTMs) to diverse downstream domains without fine-tuning, achieving an average MSE reduction of 13.6% and up to 65.4%.

## Background & Motivation

**Background**: Large time-series models (LTMs) such as Timer, Moirai, and Chronos have demonstrated zero-shot forecasting capabilities; however, their generalization performance under distribution shifts across different domains remains limited when deployed as frozen models.

**Limitations of Prior Work**: The conventional approach of fine-tuning models for each new domain causes the number of model instances to grow linearly with the number of domains, incurring high computational costs and compromising generalization.

**Key Challenge**: LTMs must simultaneously satisfy two conflicting requirements—cross-domain generality and domain-specific accuracy. Fine-tuning improves domain-specific performance at the expense of general applicability.

**Goal**: Can one adapt LTMs to different domains solely by optimizing input data transformations, without modifying any model parameters?

**Key Insight**: The authors observe that simple data transformations (e.g., downsampling, outlier interpolation, differencing) can substantially improve LTM forecasting quality (see Figure 1 for three illustrative examples), suggesting that the bottleneck lies in the mismatch between data and model rather than insufficient model capacity.

**Core Idea**: The discovery of effective data transformations is formalized as a hyperparameter optimization problem, where Bayesian search automatically identifies the optimal preprocessing pipeline, enabling a single frozen model to adapt across multiple domains.

## Method

### Overall Architecture
TATO (Time-series Adaptive Transformation Optimization) defines a new paradigm termed *FrozenForecasting*—the model is kept entirely frozen while only the data transformation pipeline is optimized. The framework consists of three stages: (1) data preparation (diversity augmentation), (2) transformation pipeline optimization (searching for the optimal preprocessing configuration), and (3) two-stage ranking to select the best pipeline. Given historical time-series data $D_{\text{history}}$, the framework outputs the optimal transformation configuration $h^*$, with the optimization objective:

$$h^* = \arg\min_{h \in \mathcal{H}} \mathcal{L}(M, D_{\text{history}}, h)$$

### Key Designs

1. **Data Preparation**

   - **Function**: Augments the diversity of historical samples during the optimization stage via multiple augmentation strategies.
   - **Mechanism**: Eight augmentation methods are employed—magnitude/time flip, magnitude/time warp, noise injection (EWMA smoothing, jitter), translation, and slope addition—ensuring that the discovered transformation pipelines remain robust under distributional shifts.
   - **Design Motivation**: A distribution gap may exist between the $D_{\text{history}}$ used during optimization and $D_{\text{future}}$ encountered at test time. Augmentation exposes candidate pipelines to more extreme scenarios, reducing the risk of overfitting to the optimization set.

2. **Transformation Pipeline Search Space**

   - **Function**: Defines nine tunable operators organized into three categories: (a) context transformations (Trimmer for input length cropping, Downsampler, Differencer); (b) normalization transformations (Scaler, e.g., Z-score / MinMax / BoxCox); and (c) outlier transformations (Denoiser, e.g., $k$-sigma detection with interpolation, IQR filtering).
   - **Mechanism**: Each operator comprises a forward transformation (applied before feeding data to the LTM) and an inverse transformation (applied to the LTM output). The hyperparameters of all operators constitute the search space $\mathcal{H}$, which is explored via TPE (Tree-structured Parzen Estimator) Bayesian optimization.
   - **Design Motivation**: The three operator categories address three distinct failure modes: input length mismatch, scale/distribution mismatch, and outlier interference, respectively. Operator ordering is predefined by heuristic rules (Trimmer is placed first to reduce downstream computation; outlier handling precedes normalization to prevent outliers from distorting normalization statistics).

3. **Two-stage Pareto Ranking**

   - **Function**: Selects the single best pipeline from hundreds of candidates.
   - **Mechanism**: **Stage 1**—Pareto filtering is applied across all augmented samples to eliminate pipelines that are dominated on any combination of metrics, retaining 16 candidates. **Stage 2**—The retained candidates are evaluated exclusively on the original (non-augmented) samples and ranked by a weighted multi-metric score (MSE, MAE, RMSE, MAPE, MSPE); the highest-scoring pipeline is selected.
   - **Design Motivation**: Selecting by a single metric risks favoring pipelines that excel on one criterion while underperforming on others. Pareto filtering ensures robustness, while the second stage focuses on the original distribution to guarantee optimality in real-world conditions.

### Loss & Training
- Optimization requires only 500 historical samples (less than 2% of the full training set).
- 500 search trials are conducted with TPE Bayesian optimization.
- The entire optimization process typically completes within 2 minutes.
- Inference overhead is less than 3 milliseconds (at batch size = 1).

## Key Experimental Results

### Main Results
Evaluation covers 8 datasets × 4 forecasting horizons × 6 LTM variants = 192 scenarios:

| Model | Avg. MSE↓ (vanilla) | Avg. MSE↓ (TATO) | MSE Improvement % |
|---|---|---|---|
| Timer-UTSD | 0.3431 | 0.3225 | 6.0% |
| Timer-LOTSA | 0.3923 | 0.2950 | **24.8%** |
| Moirai-small | 0.4185 | 0.3799 | 9.2% |
| Moirai-base | 0.4066 | 0.3568 | 12.2% |
| Moirai-large | 0.4031 | 0.3465 | 14.0% |
| Chronos-tiny | 0.3770 | 0.3225 | 14.5% |
| **Average** | **0.3901** | **0.3372** | **13.6%** |

The largest improvement occurs on the Exchange dataset with Timer-LOTSA, where MSE is reduced by 65.4%.

### Ablation Study

| Configuration | Avg. MSE Improvement % | Notes |
|---|---|---|
| Full TATO | Best (baseline) | Complete framework |
| w/o Trimmer | Significant degradation | Context trimming is critical for matching model input requirements |
| w/o Scaler | Significant degradation | Normalization is essential for cross-domain adaptation |
| w/o Denoiser | Slightly higher mean, larger variance | Denoising benefits specific samples but reduces robustness |
| w/o TwoStageRank | Slightly higher mean, lower median | Pareto filtering ensures consistency |

### Key Findings
- Trimmer and Scaler are the two most critical operators; removing either leads to substantial performance degradation.
- Removing Denoiser or TwoStageRank yields marginally better average performance but degrades variance and median metrics, indicating that their role is to ensure robustness rather than to boost peak performance.
- TATO yields the largest gains in scenarios where the base LTM performs poorly (e.g., Timer-LOTSA on Exchange: MSE 0.83 → 0.29), while improvements are smaller where the model already performs well (e.g., Timer-UTSD on Traffic at MSE ~0.06).
- TATO is complementary to fine-tuning: applying TATO on top of a universally fine-tuned model achieves an additional MSE reduction of 7.3%.

## Highlights & Insights
- **Data-centric vs. model-centric paradigm shift**: At a time when the prevailing approach is to fine-tune large models, this work takes the opposite stance—freezing the model and optimizing the data instead. Optimization completes within 2 minutes, making the framework well-suited for production deployment.
- **The two-stage Pareto selection mechanism** is an elegant design: Pareto filtering over augmented samples ensures robustness, while final ranking on original samples preserves fidelity to real-world performance. This design is transferable to any hyperparameter search scenario.
- **Refined transformation search space**: Nine operators across three categories suffice to cover the core requirements of time-series preprocessing. The search space is compact yet expressive, offering a valuable reference for future work.

## Limitations & Future Work
- The current framework supports only **univariate** time-series forecasting; multivariate settings are explicitly identified as future work.
- On the Weather dataset, TATO degrades performance for some models (e.g., Timer-UTSD MSE worsens by 22.5%), indicating that data transformations can have adverse effects on certain distributional characteristics.
- The search space is manually designed, lacking a mechanism for adaptive operator discovery.
- Although the configuration of 500 trials and 500 samples is efficient, total optimization time may become a bottleneck when the number of domains is very large (e.g., hundreds of IoT domains).
- Comparison with test-time adaptation (TTA) methods is insufficiently thorough.

## Related Work & Insights
- **vs. Model Fine-tuning**: TATO does not modify model parameters, offering better generality; however, its effectiveness is limited in scenarios where the model is already well-adapted (e.g., Timer-UTSD on Weather).
- **vs. TTA methods (e.g., NewNorm)**: TATO covers a broader scope (not limited to normalization) and requires no self-supervised training steps, making it more lightweight.
- **vs. AutoML / HPO**: TATO is essentially an automated search over data preprocessing transformations treated as hyperparameters—a paradigm consistent with AutoML frameworks such as Auto-sklearn.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The data-centric perspective is not entirely new, but it carries fresh significance in the LTM era; the FrozenForecasting paradigm merits broader adoption.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The 192-scenario evaluation, ablation studies, efficiency analysis, and complementarity-with-fine-tuning experiments are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The presentation is logically clear; the three motivating examples in Figure 1 are intuitive and compelling.
- **Value**: ⭐⭐⭐⭐ Highly practical, though applicability is currently constrained to the univariate setting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](../../NeurIPS2025/time_series/how_foundational_are_foundation_models_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
