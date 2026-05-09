---
title: >-
  [Paper Note] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data
description: >-
  [ICLR2026][Time Series][time-series foundation model] This paper proposes GTM, a general time-series foundation model that captures temporally granularity-aware features via a frequency-domain attention mechanism. Combined with a hybrid masking pre-training strategy, GTM is the first model to support all generative time-series tasks without any task-specific architectural modifications.
tags:
  - ICLR2026
  - Time Series
  - time-series foundation model
  - frequency-domain attention
  - representation learning
  - hybrid masking
  - multi-task
date: 2026-05-08
content_hash: 30ef1c945a8cf9c7
---

# GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data

**Conference**: ICLR2026
**arXiv**: [2502.03264](https://arxiv.org/abs/2502.03264)
**Code**: [GitHub](https://github.com/MMTS4All/GTM)
**Area**: Time Series
**Keywords**: time-series foundation model, frequency-domain attention, representation learning, hybrid masking, multi-task

## TL;DR

This paper proposes GTM, a general time-series foundation model that captures temporally granularity-aware features via a frequency-domain attention mechanism. Combined with a hybrid masking pre-training strategy, GTM is the first model to support all generative time-series tasks without any task-specific architectural modifications.

## Background & Motivation

Time-series foundation models (TSFMs) have made notable progress in recent years, yet two key challenges remain: (1) limited representational capacity of scalar time-series sequences — most existing models operate solely in the time domain and do not fully exploit frequency-domain information; (2) the wide variety of downstream tasks (forecasting, imputation, anomaly detection, classification, etc.) typically requires task-specific modifications, such as replacing tokenization strategies, adjusting pre-training objectives, or swapping projection heads.

Through FFT combined with 2D kernel density estimation on large-scale multi-domain time-series data, the authors observe that time-series data at different temporal granularities (second-level, minute-level, hour-level, etc.) exhibit distinct joint distributions of amplitude–frequency and phase–frequency in the frequency domain. This key observation directly motivates the model design — a frequency-domain module is required to capture multi-granularity representations.

## Core Problem

1. How to explicitly model the differences in frequency-domain distributions across temporal granularities within the model, so as to enhance representation learning quality?
2. How to design a unified pre-training framework that allows a single model to support all generative downstream tasks without any task-specific modifications?

## Method

### Overall Architecture

GTM adopts a Decoder-only Transformer architecture consisting of three core modules:

- **Input Embedding**: Applies Reversible Instance Normalization, Channel Independence (CI), patching, and masking to the raw time series, converting it into a univariate masked token sequence, followed by linear embedding and positional encoding.
- **N-stack Decoder Backbone**: Each decoder block contains a standard time-domain self-attention layer followed by a Fourier Attention module.
- **Output Projection**: A unified linear projection layer with denormalization that generates outputs in an autoregressive manner.

### Fourier Attention Mechanism

This is the core innovation of the paper. The time-domain self-attention output $\mathbf{H}_{\text{TemAttOut}}$ is transformed to the frequency domain via column-wise FFT, and then processed as follows:

1. **Temporal Granularity Encoding**: The sampling granularity is represented as a 5-tuple (day, hour, minute, second, millisecond); for example, the ETTm dataset is encoded as [0, 0, 15, 0, 0].
2. **Granularity-Aware Attention**: Five learnable key embeddings corresponding to five typical granularities are introduced; attention weights $\alpha$ are computed via dot-product between queries and keys followed by softmax.
3. **Low-Rank Frequency Module**: Five groups of low-rank matrices $\{A_i, B_i\}$ correspond to the five granularities and are aggregated by weighted summation using the attention weights.
4. **Global Frequency Module**: A fully connected matrix $W_{\text{full}}$ operates in parallel to capture granularity-agnostic frequency patterns.
5. **Inverse FFT**: The frequency-domain output is transformed back to the time domain.

The final output is:
$$\mathbf{H}_{\text{out}} = \text{iFFT}\left(\sum_{i=1}^{5} \alpha_i (A_i B_i) \mathbf{H}_{\text{FFT}} + W_{\text{full}} \mathbf{H}_{\text{FFT}}\right)$$

### Pre-training Framework: Hybrid Masking Strategy

The framework unifies reconstruction and autoregressive objectives. Key designs include:

- **Hybrid Masking**: A hyperparameter `pred_ratio` controls the probability of applying tail-contiguous masking. A sample $r \sim \mathcal{U}(0,1)$ is drawn; if $r \leq$ `pred_ratio`, tail-contiguous masking is applied (simulating forecasting); otherwise, random masking is used (simulating reconstruction).
- **Span Shuffling**: Multiple contiguous patch spans are randomly sampled, randomly permuted, and concatenated as the target sequence.
- **2D Positional Encoding**: Enables the model to perceive the length information of masked spans.
- **Attention Mechanism**: Full attention is applied to masked reconstruction portions, and causal attention is applied to autoregressive generation portions to prevent information leakage.
- **Pre-training Data**: The large-scale public dataset UTSD-12G is used, covering multiple domains and temporal granularities.

### Downstream Task Adaptation

Thanks to the unified architecture and pre-training strategy, GTM requires only minor preprocessing adjustments (removing masking and 2D positional encoding) for generative tasks, with no structural modifications to the model. For discriminative tasks such as classification, only the output projection layer is replaced.

## Key Experimental Results

### Long-term Forecasting

Across five datasets — ETTh1, ETTm1, Weather, Traffic, and Electricity — GTM achieves the best average performance over prediction horizons $T \in \{96, 192, 336, 720\}$. Representative results:

| Dataset | GTM (MSE/MAE) | PatchTST | TimesNet |
|---------|--------------|----------|----------|
| ETTh1 | 0.404/0.429 | 0.413/0.434 | 0.458/0.450 |
| Weather | 0.225/0.266 | 0.225/0.263 | 0.259/0.287 |
| Electricity | 0.161/0.254 | 0.159/0.252 | 0.192/0.295 |

### Imputation

On ETTh1, GTM reduces MSE by 23.1% and MAE by 12.1% over the second-best model; on ETTm1, MSE is reduced by 25.0% and MAE by 8.6%.

### Anomaly Detection

GTM achieves an average F1 of 87.01% across five datasets (MSL, SMAP, SWaT, SMD, PSM), surpassing all baselines (second-best GPT4TS: 86.72%).

### Classification

GTM achieves the best performance on 5 out of 10 classification datasets and second-best on 4, outperforming multi-task TSFMs including UniTS, GPT4TS, and TimesNet.

### Zero-shot Forecasting

Compared with Timer-1B, MOIRAI-S, MOMENT, TimesFM, and Chronos-S1, GTM achieves the lowest average MSE across five datasets (0.380 vs. Timer's 0.392 as second-best).

### Pre-training Effectiveness

Pre-trained GTM vs. randomly initialized GTM: forecasting MSE decreases by 0.5%–7.8%, imputation MSE decreases by 1.2%–11.7%, and anomaly detection F1 improves by 1.2%.

### Scalability

GTM follows the scaling law: deeper and wider models, as well as larger pre-training datasets, consistently yield performance gains. However, when model depth is insufficient, increasing width alone may not improve performance.

### Computational Efficiency

GTM has 35.73M parameters, a training speed of 0.290s/iter, inference memory of 1.25GB, and a univariate inference latency of only 0.043s/item on an A100 GPU. The additional overhead introduced by FFT/iFFT operations is negligible.

## Highlights & Insights

1. **Frequency-Domain Attention**: The first TSFM to introduce temporally granularity-aware frequency-domain modeling; the combination of low-rank decomposition and a global module is both efficient and effective.
2. **Task-Agnostic Generative Design**: The first foundation model to support all generative time-series tasks without any task-specific modifications.
3. **Hybrid Masking Pre-training**: Elegantly unifies reconstruction and autoregressive objectives into a single pre-training framework via probabilistic control.
4. **Comprehensive Evaluation**: Covers forecasting, imputation, anomaly detection, classification, zero-shot, few-shot, ablation, scalability, and computational efficiency across multiple dimensions.

## Limitations & Future Work

1. Frequency-domain granularity is encoded as only 5 types, which may lack flexibility for irregularly sampled or mixed-granularity time series.
2. Classification tasks still require replacing the projection layer, so full task-agnosticism has not been achieved.
3. The Channel Independence strategy may discard inter-channel correlation information in multivariate settings.
4. Zero-shot performance on certain datasets (e.g., ETTm1, Traffic) falls short of MOIRAI-S or Timer, leaving room for improved generalization.
5. The impact of domain coverage and data quality of the pre-training dataset UTSD-12G on model performance is not thoroughly discussed.

## Related Work & Insights

| Model | Characteristics | GTM Advantage |
|-------|----------------|---------------|
| Timer | Autoregressive pre-training, requires task-specific modifications | GTM uses unified pre-training without modification |
| UniTS | Dual-tower Transformer + task tokenization | GTM is simpler, requires no task tokens |
| UP2ME | MAE pre-training + Graph Transformer fine-tuning | GTM achieves the same with a single architecture |
| PatchTST | Forecasting only, CI + patching | GTM extends this with frequency-domain modeling |
| MOIRAI | Cross-frequency learning but forecasting only | GTM supports multiple tasks |
| Time-MOE | MOE design, multi-resolution forecasting | GTM achieves granularity awareness via frequency-domain attention |

### Takeaways

- The low-rank design of the Fourier Attention is worth borrowing: different granularities correspond to different low-rank subspaces, which is highly consistent with signal processing intuition.
- The hybrid masking strategy unifies BERT-style reconstruction and GPT-style autoregression into a single framework — this idea is generalizable to foundation models in other modalities.
- Encoding temporal granularity as prior knowledge into the model provides a useful reference for handling multi-source heterogeneous time-series data.
- Combining frequency-domain attention with state space models (e.g., Mamba) could further reduce computational complexity.

## Rating
- **Novelty**: 8/10 — The combination of frequency-domain attention and hybrid masking pre-training represents significant innovation.
- **Experimental Thoroughness**: 9/10 — Broad coverage with detailed ablation and efficiency analysis.
- **Writing Quality**: 7/10 — Structure is clear, though some notation is dense in places.
- **Value**: 8/10 — The first task-agnostic generative TSFM with strong potential for industrial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

<!-- RELATED:END -->
