---
title: >-
  [Paper Note] TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series
description: >-
  [ICLR 2026][Time Series][Time Series Pre-trained Model] This paper proposes TSPulse, an ultra-lightweight time series pre-trained model with only 1M parameters, which surpasses models 10–100× larger on four tasks — classification (+5–16%), anomaly detection (+20%), imputation (+50%), and similarity retrieval (+25%) — through dual-space masked reconstruction and dual-embedding disentanglement.
tags:
  - ICLR 2026
  - Time Series
  - Time Series Pre-trained Model
  - Disentangled Representations
  - Dual-Space Reconstruction
  - Anomaly Detection
  - Tiny Model
date: 2026-05-08
content_hash: a138663a64cceb0c
---

# TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series

**Conference**: ICLR 2026
**arXiv**: [2505.13033](https://arxiv.org/abs/2505.13033)
**Code**: [https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
**Area**: Time Series
**Keywords**: Time Series Pre-trained Model, Disentangled Representations, Dual-Space Reconstruction, Anomaly Detection, Tiny Model

## TL;DR

This paper proposes TSPulse, an ultra-lightweight time series pre-trained model with only 1M parameters, which surpasses models 10–100× larger on four tasks — classification (+5–16%), anomaly detection (+20%), imputation (+50%), and similarity retrieval (+25%) — through dual-space masked reconstruction and dual-embedding disentanglement.

## Background & Motivation

Time series analysis encompasses diverse downstream tasks including forecasting, anomaly detection, imputation, classification, and retrieval. Inspired by successes in NLP and CV, the time series community has increasingly explored large-scale pre-trained models:

**Task-specific models**: TimesFM, Chronos, and Moirai focus on forecasting tasks.

**General-purpose models**: Moment and UniTS extend to classification, anomaly detection, and imputation.

**Cross-domain models**: Time-LLM and GPT4TS attempt to adapt LLMs to time series.

Core problem: **Existing pre-trained models have massive parameter counts** (hundreds of millions to billions), leading to prohibitively high deployment and fine-tuning costs. TTM demonstrates that compact models with 1–5M parameters can achieve competitive performance on forecasting, but this advantage is limited to that task.

Research gap: **Can a ~1M-parameter pre-trained model be constructed that simultaneously achieves state-of-the-art performance across multiple non-forecasting diagnostic tasks?**

## Method

### Overall Architecture

TSPulse is built on the lightweight TSMixer architecture. The core pipeline is:
- Input $\mathbf{X} \in \mathbb{R}^{S \times C}$ → masking → dual-space encoding (time domain + frequency domain) → TSMixer backbone → mini decoder → multi-objective output heads

### Key Designs

1. **Dual-Space Masked Reconstruction**

    - Reconstruction of masked inputs is performed simultaneously in both the time domain and frequency domain.
    - Core intuition: certain patterns are more readily detected in the time domain (e.g., spikes), while others are more prominent in the frequency domain (e.g., periodicity).
    - The time-domain masked input $\mathbf{X}_m$ is transformed via FFT to obtain the frequency-domain representation $\mathbf{X}^f_m$.
    - Key design: **the frequency domain is not explicitly masked**; instead, the time-domain masked signal is directly fed into FFT, naturally propagating masking into the frequency domain.
    - After encoding, representations are concatenated: $\mathbf{Input}_E = [\mathbf{Time}_E; \mathbf{FFT}_E; \mathbf{Reg}_E] \in \mathbb{R}^{C \times K \times D}$

2. **Dual-Embedding Disentanglement**

    - **Detailed embeddings** (first $2N$ patch embeddings): used for full-signal reconstruction, capturing fine-grained time-domain and frequency-domain patterns.
    - **Semantic embeddings** (last $R$ register embeddings): used for high-level semantic reconstruction, encoding global features.
    - Semantic embeddings are supervised through two tasks:
        - Frequency signature prediction: $\mathcal{L}_{prob} = \text{CE}(\mathbf{X}^f_{prob}, \mathbf{Y}^f_{prob})$ (softmax distribution over log-amplitude spectra)
        - Short-term forecasting: $\mathcal{L}_{future} = \text{MSE}(\mathbf{X}_{future}, \mathbf{Y}_{future})$
    - Design motivation: different downstream tasks require different levels of information — classification benefits from semantic embeddings, while imputation relies on detailed embeddings.

3. **TSLens (Classification Fine-Tuning Component)**

    - A learned mechanism replacing standard pooling that adaptively extracts relevant features from dual embeddings.
    - Pipeline: mini decoder (initialized from pre-trained weights + channel mixing) → dimensionality-reduction projection → flatten → linear classification head.
    - Dynamically focuses on the most informative features in both local and global representations.

4. **Multi-Head Triangulation for Anomaly Detection**

    - Three prediction heads detect anomalies from complementary perspectives:
        - $\text{Head}_{time}$: time-domain reconstruction deviation → detects spike anomalies
        - $\text{Head}_{fft}$: frequency-domain reconstruction deviation → detects periodic anomalies
        - $\text{Head}_{future}$: short-term prediction deviation → detects trend anomalies
    - Fusion strategies: $\text{Head}_{ensemble}$ (max-value fusion) or $\text{Head}_{triang.}$ (selecting the best head based on a small validation set).
    - This represents the first pre-trained model to unify multi-space outputs for triangulation within a single lightweight framework.

5. **Hybrid Masking Strategy**

    - Conventional block masking is ill-suited for real-world imputation, where missing values are irregular.
    - Hybrid strategy: simultaneously masks complete patches and partial point-level positions.
    - Key design: the mask token $\mathbf{M} \in \mathbb{R}^{1 \times pl}$ is defined at the original patch level (not in embedding space), enabling flexible partial masking.
    - Ablations show that removing hybrid pre-training causes a 79% performance drop under hybrid-masking evaluation.

6. **Identity Initialization for Channel Mixing**

    - Pre-training employs a channel-independent mode.
    - Channel mixing is enabled during fine-tuning, but newly added mixing layers are initialized with **identity weights**.
    - This prevents random initialization from causing activation discontinuities and gradient instability between pre-trained layers.

### Loss & Training

Joint minimization of a multi-objective weighted loss:
- $\mathcal{L}_{time1} = \text{MSE}(\mathbf{X}, \mathbf{Y})$: time-domain reconstruction (masked positions only)
- $\mathcal{L}_{time2} = \text{MSE}(\mathbf{X}, \mathbf{Y}')$: time-domain reconstruction via inverse FFT from frequency space
- $\mathcal{L}_{fft} = \text{MSE}(\mathbf{X}^f, \mathbf{Y}^f)$: frequency-domain reconstruction
- $\mathcal{L}_{prob} = \text{CE}(\mathbf{X}^f_{prob}, \mathbf{Y}^f_{prob})$: frequency signature
- $\mathcal{L}_{future} = \text{MSE}(\mathbf{X}_{future}, \mathbf{Y}_{future})$: short-term forecasting

Task-specialized pre-training is achieved by re-weighting loss head priorities (e.g., AD retains all heads; classification emphasizes the time-domain and probability heads).

Pre-training covers ~1B time series samples and requires only one day on 8×A100 GPUs.

## Key Experimental Results

### Anomaly Detection (TSB-AD Leaderboard, Figure 4)

| Method | Univariate VUS-PR | Multivariate VUS-PR |
|--------|------------------|---------------------|
| Sub-PCA (Prev. SOTA) | 0.42 | - |
| CNN (Prev. SOTA) | - | 0.31* |
| MOMENT (ZS) | 0.38 | - |
| **TSPulse (ZS)** | **0.48** (+14%) | **0.36** (+16%) |
| **TSPulse (FT)** | **0.52** (+24%) | **0.36** (+26%*) |

*TSPulse simultaneously ranks first on both univariate and multivariate TSB-AD leaderboards.

### Classification (UEA 29 Datasets, Figure 5)

| Method | Parameters | Mean Accuracy |
|--------|-----------|--------------|
| VQShape | ~37M | 0.701 |
| MOMENT | ~110M | 0.675 |
| UniTS | ~10M | 0.634 |
| **TSPulse** | **~1M** | **0.733** (+5–16%) |

### Imputation (6 LTSF Benchmarks, Figure 6 — Hybrid Masking)

| Method | Setting | Mean MSE↓ |
|--------|---------|-----------|
| MOMENT | ZS | 0.276 |
| UniTS (PMT) | PMT | 0.170 |
| **TSPulse** | **ZS** | **0.074** (+56–73%) |
| TimesNet | FT | 0.080 |
| **TSPulse** | **FT** | **0.039** (+49–51%) |

### Ablation Study (Table 1)

**Classification Ablation**:

| Variant | Accuracy | Drop |
|---------|----------|------|
| TSPulse (Full) | 0.747 | - |
| w/o Short Embedding | 0.689 | -8% |
| w/o Long Embedding | 0.681 | -10% |
| w/o Masking | 0.691 | -8% |
| w/o CM Identity Init | 0.685 | -9% |
| w/o TSLens (Avg-Pool) | 0.675 | -11% |
| w/o TSLens (Max-Pool) | 0.645 | -16% |
| w/o Dual-space | 0.696 | -7% |

### Efficiency Comparison (Table 23)

| Model | Params (M) | GPU Inference (ms) | CPU Inference (s) | Memory (GB) |
|-------|-----------|-------------------|-------------------|-------------|
| **TSPulse** | **1.06** | **7.16** | **0.06** | **0.39** |
| MOMENT (small) | 35.34 (33×) | 32.57 (5×) | 2.74 (46×) | 0.56 |
| MOMENT (large) | 341.24 (322×) | 405.42 (57×) | 21.98 (366×) | 2.30 |
| Chronos (tiny) | 8.39 (8×) | 39.81 (6×) | 66.15 (1103×) | 2.91 |

### Key Findings

1. **1M parameters outperforms models 10–100× larger**: model size is not the sole determinant of performance; architectural design is equally critical.
2. **Dual-space learning is essential**: removing the frequency-domain branch causes a 7% drop in classification and an 8% drop in imputation.
3. **Hybrid masking pre-training is the key to imputation performance**: pure block masking leads to a 79% collapse under hybrid-masking evaluation.
4. **TSLens significantly outperforms standard pooling**: drops of -11% (avg-pool) and -16% (max-pool) validate the value of learned attention.
5. **Semantic embeddings from register tokens are robust to distortion**: insensitive to noise, amplitude variation, and temporal shift, while remaining sensitive to frequency and shape.

## Highlights & Insights

- **"Small but mighty" philosophy**: 1M parameters suffices — the key lies in elegant architectural design (dual-space, dual-embedding, multi-head triangulation).
- **Value of disentangled representations**: the separation of fine-grained embeddings from semantic embeddings allows different tasks to select the most suitable representations.
- **Elegance of multi-head triangulation**: different reconstruction heads are naturally suited to detecting different anomaly types; their fusion outperforms any single perspective.
- **Zero-shot surpasses trained models**: TSPulse's zero-shot anomaly detection outperforms all models trained on the target data.
- **CPU-friendly**: a 0.06-second CPU inference time enables GPU-free deployment.
- **IBM Granite series**: open-sourced on HuggingFace, offering strong practical utility.

## Limitations & Future Work

1. Forecasting tasks are currently not addressed, though the viability of compact models for forecasting has been established by TTM.
2. Pre-training data primarily covers specific domains (energy, transportation, etc.); transferability to other domains remains to be validated.
3. The two-stage design of univariate pre-training followed by multivariate fine-tuning may not be optimal.
4. Incremental learning capability is absent: the model cannot continuously update without forgetting prior knowledge.
5. Few-shot classification capability warrants further exploration.
6. Cross-modal fusion (e.g., time series + text) is a promising direction for future work.

## Related Work & Insights

- **TTM (Tiny Time Mixers)**: a pioneer in compact time series pre-trained models, but limited to forecasting tasks.
- **MOMENT**: a general-purpose time series foundation model based on a T5-encoder architecture, with 35–341M parameters.
- **Chronos**: a T5-style encoder-decoder focused on forecasting, ranging from 0.06M to 709M parameters.
- **UniTS**: a prompt-tuned multi-task model.
- **TSMixer**: the backbone network of TSPulse, adopting the MLP-Mixer paradigm as an alternative to Transformers.
- **Insight**: compact models + task-specialized pre-training + carefully designed post-processing components constitute an efficient and powerful foundation model design paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (dual-space dual-embedding disentanglement + multi-head triangulation + hybrid masking — a combination of multiple innovations)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (75+ datasets, 4 major tasks, comprehensive ablations, efficiency analysis, embedding sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (thorough content, clear logic, exceptionally rich appendix)
- Value: ⭐⭐⭐⭐⭐ (1M-parameter model surpasses models 100× larger; open-sourced and deployment-friendly)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
