---
title: >-
  [Paper Note] Towards Self-Supervised Foundation Models for Critical Care Time Series
description: >-
  [NeurIPS 2025][Medical Imaging][Self-Supervised Learning] A self-supervised foundation model for critical care time series is constructed by pre-training a Biaxial Transformer (BAT) architecture on multiple ICU datasets…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Self-Supervised Learning"
  - "Foundation Models"
  - "Critical Care"
  - "Time Series"
  - "Transfer Learning"
date: 2026-05-08
content_hash: a39938ce16b08c10
---

# Towards Self-Supervised Foundation Models for Critical Care Time Series

**Conference**: NeurIPS 2025
**arXiv**: [2509.19885](https://arxiv.org/abs/2509.19885)  
**Code**: [GitHub](https://github.com/Katja-Jagd/YAIB)  
**Area**: Medical Imaging
**Keywords**: Self-Supervised Learning, Foundation Models, Critical Care, Time Series, Transfer Learning

## TL;DR

A self-supervised foundation model for critical care time series is constructed by pre-training a Biaxial Transformer (BAT) architecture on multiple ICU datasets, substantially outperforming supervised baselines in low-data regimes.

## Background & Motivation

Foundation models have advanced significantly in NLP and medical imaging, yet the **critical care time series** domain remains severely underserved. The root causes are as follows:

**Data scarcity and heterogeneity**: Publicly available ICU datasets are limited in number and vary in format, with substantial differences in monitoring equipment and sampling frequencies across hospitals.

**Poor reproducibility**: Existing models are typically trained on a single dataset for a single supervised task, making results difficult to reproduce in new clinical environments.

**Weak transferability**: Models trained on one dataset suffer marked performance degradation when transferred directly to another (e.g., AUC-ROC drops of 4–5 points).

**Closed-source prior work**: The first ICU foundation model, ICareFM, has released neither code nor model weights.

This paper aims to build the first **open-source** critical care time series foundation model within a transparent and reproducible framework (YAIB), by pooling multiple ICU datasets for self-supervised pre-training, with particular emphasis on performance in resource-constrained settings (limited labeled data).

## Method

### Overall Architecture

The overall approach follows a two-stage pipeline: **self-supervised pre-training → supervised fine-tuning**.

- **Pre-training stage**: Multiple ICU datasets (e.g., eICU + MIMIC-IV) are pooled, and a time series **forecasting task** is used for self-supervised learning.
- **Fine-tuning stage**: Binary classification for mortality prediction is performed on a held-out dataset (e.g., MIMIC-III) not seen during pre-training.

### Key Designs

1. **Biaxial Transformer (BAT) Architecture**: BAT performs attention along both the temporal and feature axes, jointly capturing temporal dependencies and cross-feature relationships. The input embedding comprises three components: observed values (with missingness indicators), learned feature identity embeddings, and continuous-time positional encodings. Outputs are pooled and concatenated with static features (age, sex, etc.). The key advantage of BAT is its **native handling of missing values** and **irregular sampling**, enabling it to model informative missingness without mean imputation.

2. **Dynamic Window Sampling Strategy**: Observation and prediction windows are constructed dynamically during training. For each batch, patients and time indices are sampled randomly, subject to three validity constraints: (a) a sparsity check ensuring at least one observation exists within the window; (b) a minimum observation length of $L=12$ hours; and (c) availability of a prediction horizon of $H=2$ hours. This strategy exposes the model to diverse temporal contexts, enhancing generalization.

3. **Dual-Head Design**: BAT is adapted to support two prediction heads simultaneously — a forecasting head for self-supervised pre-training, outputting $\hat{\mathbf{X}}^{\text{for}} \in \mathbb{R}^{T^{\text{for}} \times D}$, and a binary classification head for supervised fine-tuning, outputting $\hat{y} \in \{0, 1\}$. During fine-tuning, one may opt for head-only fine-tuning or full model fine-tuning.

### Loss & Training

**Pre-training loss**: Masked MSE loss computed only at observed positions within the prediction window:

$$\mathcal{L}^{\text{Pre}} = \frac{1}{\sum_{k=2}^{K} N_k} \sum_{k=2}^{K} \sum_{i=1}^{N_k} \left\| \mathbf{M}_i^{\text{for}} \odot (\hat{\mathbf{X}}_i^{\text{for}} - \mathbf{X}_i^{\text{for}}) \right\|_F^2$$

**Fine-tuning loss**: Standard binary cross-entropy:

$$\mathcal{L}^{\text{Fine}} = -\frac{1}{N_1} \sum_{i=1}^{N_1} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

Pre-training employs Bayesian hyperparameter optimization via the YAIB framework, with learning rates of approximately 3–8e-4, batch size 64, up to 200 epochs, and early stopping with patience 10–15. Weighted loss is used to address class imbalance.

## Key Experimental Results

### Main Results

Experiments use three datasets — MIMIC-III, MIMIC-IV, and eICU — with 52 clinical features (4 static + 48 time-varying), evaluated under a leave-one-dataset-out pre-training and fine-tuning protocol.

| Fine-tuning Dataset | Data Size | BAT (Full FT) | BAT (Head FT) | BAT (Scratch) | Transformer (Scratch) |
|---|---|---|---|---|---|
| MIMIC-III | 1000 | **36.24±1.63** | 33.98±1.32 | 27.63±2.23 | 21.34±4.58 |
| MIMIC-III | 5000 | **41.89±1.31** | 38.99±0.96 | 36.30±0.31 | 26.14±0.40 |
| MIMIC-III | 9000 | **43.57±0.94** | 40.14±0.70 | 37.09±1.06 | 27.13±0.41 |
| MIMIC-IV | 1000 | **28.98±0.85** | 26.97±1.71 | 26.12±1.95 | 13.06±1.56 |
| MIMIC-IV | 5000 | **38.10±1.36** | 31.31±1.17 | 34.97±1.03 | 18.00±1.11 |
| eICU | 1000 | **28.37±1.13** | 25.39±1.56 | 20.86±3.31 | 6.58±4.00 |
| eICU | 5000 | **33.89±0.49** | 29.09±0.62 | 30.13±1.37 | 14.41±0.42 |

*Metric: AUC-PR (%). Pre-training uses the two datasets not included in fine-tuning.*

### Ablation Study

| Configuration | Key Observation | Notes |
|---|---|---|
| Full fine-tuning vs. head-only fine-tuning | Full FT gains 2–6 AUC-PR points | Full fine-tuning consistently superior |
| Pre-training data: 273K vs. 100K | Larger pre-training set yields better transfer | eICU+MIMIC-IV (273K) achieves best performance |
| <5000 vs. ≥5000 fine-tuning samples | Advantage more pronounced in low-data regime | Pre-training most beneficial with limited labels |
| Direct cross-dataset inference | AUC-ROC drops 1–5 points | Confirms poor direct transferability |

### Key Findings

- Pre-trained models outperform from-scratch baselines in **all settings with >500 samples**, with the largest gains observed when **<5000 samples** are available for fine-tuning.
- The largest pre-training corpus (MIMIC-IV + eICU, 273K samples) yields the strongest transferable representations.
- Head-only fine-tuning approaches the performance of full model fine-tuning, indicating that pre-training has learned generalizable representations.

## Highlights & Insights

- **First open-source and reproducible ICU time series foundation model**, with all code and experiments implemented within the YAIB framework.
- The masked loss function elegantly handles irregular sampling and sparse data, eliminating dependence on missing value imputation.
- Strong potential in resource-constrained clinical settings: the pre-training advantage is most pronounced when labeled data are scarce.

## Limitations & Future Work

- Only three U.S.-based ICU datasets are used, limiting data diversity.
- The model contains approximately 1M parameters, far smaller in scale than foundation models in NLP or computer vision.
- The potential benefit of cross-domain time series data (e.g., meteorological, power grid) for pre-training remains unexplored.
- In large labeled-data regimes (≥5000 samples), from-scratch BAT training occasionally matches pre-trained model performance.

## Related Work & Insights

- **ICareFM**: The first ICU foundation model, but closed-source; this paper provides an open-source alternative.
- **YAIB framework**: Offers end-to-end benchmarking for reproducible clinical machine learning.
- Insight: Foundation model development for medical time series remains in its early stages, and data scale constitutes a critical bottleneck.

## Rating

- Novelty: ⭐⭐⭐ The methodological framework (pre-train + fine-tune) is relatively straightforward, though this represents the first open-source work in the ICU time series domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple data scales, diverse baselines, and cross-validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with comprehensive appendices.
- Value: ⭐⭐⭐⭐ An open-source and reproducible ICU foundation model offers significant value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing](generating_multi-table_time_series_ehr_from_latent_space_with_minimal_preprocess.md)

</div>

<!-- RELATED:END -->
