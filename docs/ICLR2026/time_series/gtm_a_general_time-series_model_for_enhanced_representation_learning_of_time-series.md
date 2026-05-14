---
title: >-
  [Paper Note] GTM: A General Time-series Model for Enhanced Representation Learning
description: >-
  [ICLR 2026][Time Series][Time series foundation model] GTM is a general time-series foundation model that captures temporal granularity-aware features via a Fourier attention mechanism and unifies reconstruction and auto…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Time series foundation model"
  - "frequency-domain attention"
  - "hybrid mask pre-training"
  - "multi-task"
  - "temporal granularity awareness"
date: 2026-05-08
content_hash: 7aced6b83910bd1d
---

# GTM: A General Time-series Model for Enhanced Representation Learning

**Conference**: ICLR 2026
**arXiv**: [2502.03264](https://arxiv.org/abs/2502.03264)
**Code**: [https://github.com/MMTS4All/GTM](https://github.com/MMTS4All/GTM)
**Area**: Time Series
**Keywords**: Time series foundation model, frequency-domain attention, hybrid mask pre-training, multi-task, temporal granularity awareness

## TL;DR
GTM is a general time-series foundation model that captures temporal granularity-aware features via a Fourier attention mechanism and unifies reconstruction and autoregressive pre-training objectives through hybrid masking, achieving state-of-the-art performance across forecasting, imputation, anomaly detection, and classification tasks.

## Background & Motivation

1. **Background**: Time series foundation models (TSFMs) fall into two categories — forecasting-specialized models (e.g., TimesFM, Lag-Llama) and multi-task models (e.g., Timer, UniTS). The former are optimized for forecasting, while the latter aim to cover diverse downstream tasks.
2. **Limitations of Prior Work**: (a) Existing models primarily extract features in the time domain, neglecting frequency-domain distributional differences associated with temporal granularity; (b) multi-task TSFMs typically require task-specific modifications to tokenization, pre-training strategies, or projection heads, making them insufficiently task-agnostic.
3. **Key Challenge**: How can a unified architecture and pre-training framework simultaneously learn rich time-series representations (including frequency-domain information) while seamlessly adapting to all generative downstream tasks?
4. **Goal**: (a) Design a frequency-domain attention mechanism to capture distributional differences across temporal granularities (second/minute/hour/day); (b) unify reconstruction and autoregressive pre-training objectives so that the model requires no task-specific modifications across generative tasks.
5. **Key Insight**: Through FFT analysis and 2D kernel density estimation on large-scale time-series data, the authors identify significant differences in the joint amplitude-frequency and phase-frequency distributions across temporal granularities — a critical but previously overlooked dimension that directly motivates the model design.
6. **Core Idea**: Employ Fourier attention to capture temporal granularity-aware frequency features, and apply hybrid masking (random + tail-continuous) to unify reconstruction and autoregressive objectives, yielding the first generative task-agnostic time-series foundation model.

## Method

### Overall Architecture
GTM adopts a decoder-only Transformer architecture. The input is a univariate token sequence processed through RevIN normalization, channel-independent (CI) patching, and masking. The sequence passes through $N$ stacked decoder blocks — each comprising temporal self-attention followed by Fourier attention — and is then projected by a unified linear head to autoregressively generate outputs. Pre-training is conducted on the UTSD-12G large-scale dataset.

### Key Designs

1. **Fourier Attention Module**:
    - **Function**: Transforms temporal features in the frequency domain with temporal granularity awareness, capturing distributional differences across sampling rates.
    - **Mechanism**: Column-wise FFT is first applied to the output of temporal self-attention: $\mathbf{H}_{\text{FFT}} = \text{FFT}(\mathbf{H}_{\text{TemAttOut}})$. Five low-rank frequency-domain learning matrices $\{(\mathbf{A}_i, \mathbf{B}_i)\}_{i=1}^5$ are designed to correspond to five temporal granularities (day/hour/minute/second/millisecond), supplemented by one fully connected global module. Temporal granularity is encoded as a five-tuple query; attention weights $\alpha$ are computed via softmax over five learnable keys and used for weighted aggregation: $\mathbf{H}_{\text{FourierAtt}} = \sum_{i=1}^{5} \alpha_i (\mathbf{A}_i \mathbf{B}_i) \mathbf{H}_{\text{FFT}} + \mathbf{W}_{\text{full}} \mathbf{H}_{\text{FFT}}$. An inverse FFT then maps the result back to the time domain.
    - **Design Motivation**: Empirical observation reveals significant frequency distribution differences across granularities. Low-rank decomposition controls parameter count, while the global module captures granularity-agnostic universal frequency patterns.

2. **Hybrid Mask Pre-training Strategy**:
    - **Function**: Simultaneously learns reconstruction and autoregressive capabilities under a single unified objective.
    - **Mechanism**: A hyperparameter $pred\_ratio$ controls the probability of applying tail-continuous masking. Each training sample undergoes tail-continuous masking (autoregressive prediction) with probability $p$, or random span masking (reconstruction) with probability $1-p$. Masked spans are randomly permuted and concatenated with [START]/[END] tokens. Full attention is used for reconstruction and causal attention for autoregression to prevent information leakage.
    - **Design Motivation**: Pure reconstruction pre-training excels at representation learning but underperforms in forecasting; pure autoregressive pre-training favors forecasting but yields limited representation quality. The hybrid strategy endows the model with both capabilities, enabling it to be truly task-agnostic.

3. **2D Positional Encoding + Span Shuffling**:
    - **Function**: Incorporates 1D and 2D positional encodings into the input embedding and applies random permutation to masked spans.
    - **Mechanism**: $\mathbf{H}_{in} = \mathbf{W}_{emb} \mathbf{X}_{in} + \mathbf{W}_{1D\_pos} + \mathbf{W}_{2D\_pos}$. The 2D positional encoding enables the model to perceive the length of masked spans, and span shuffling improves pre-training robustness.
    - **Design Motivation**: Inspired by GLM, this design ensures the model is aware of the span length to be filled during generation.

### Loss & Training
- MSE loss: $\text{Loss} = \frac{1}{|\mathbf{Y}|} \sum_i \|\mathbf{X}_{out_i} - \mathbf{y}_i\|^2$
- Autoregressive generation: $\mathbb{P}(\mathbf{X}_{out}) = \prod_i \mathbb{P}(\mathbf{X}_{out_i} | \mathbf{X}_{P_{crpt}}, \mathbf{S}_{\sigma(j \leq i)})$
- Pre-training data: UTSD-12G, with no downstream data leakage
- Downstream tasks require only fine-tuning; forecasting, imputation, and anomaly detection involve no architectural modification, while classification replaces only the projection head.

## Key Experimental Results

### Main Results — Long-term Forecasting
Average MSE/MAE over prediction horizons $T \in \{96, 192, 336, 720\}$:

| Dataset | GTM MSE | PatchTST MSE | TimesNet MSE | GPT4TS MSE | Gain |
|---------|---------|-------------|-------------|-----------|------|
| ETTh1 | **0.404** | 0.413 | 0.458 | 0.427 | vs PatchTST: −2.2% |
| ETTm1 | **0.339** | 0.352 | 0.400 | 0.352 | vs PatchTST: −3.7% |
| Weather | **0.225** | 0.225 | 0.259 | 0.237 | On par with PatchTST |
| Traffic | **0.385** | 0.390 | 0.620 | 0.414 | vs PatchTST: −1.3% |
| Electricity | **0.161** | 0.159 | 0.192 | 0.167 | Slightly below PatchTST |

### Ablation Study

| Configuration | ETTh1 MSE | ETTm1 MSE | Weather MSE | Note |
|--------------|-----------|-----------|------------|------|
| GTM (full) | 0.404 | 0.339 | 0.225 | Full model |
| w/o frequency module | ~0.415+ | ~0.345+ | ~0.230+ | Baseline variant |
| w/o granularity-aware module | ~0.410+ | ~0.342+ | ~0.228+ | Advanced variant without granularity |
| w/o pre-training | 0.435 | 0.351 | 0.244 | MSE increases by 0.5%–7.8% |

### Other Task Performance
- **Imputation**: ETTh1 MSE 0.053 (vs GPT4TS 0.069, +23.1%); ETTm1 MSE 0.021 (vs TimesNet 0.027, +25.0%)
- **Anomaly Detection**: Average F1 87.01% (vs GPT4TS 86.72%)
- **Classification**: Best on 5 and second-best on 4 out of 10 datasets
- **Zero-shot Forecasting**: Average MSE 0.380 (vs Timer-1B 0.392, MOIRAI-S 0.405)

### Key Findings
- The **temporal granularity-aware module** within Fourier attention contributes most significantly — removing it degrades performance across all datasets.
- Pre-training yields consistent improvements: forecasting MSE decreases by 0.5%–7.8%, imputation MSE by 1.2%–11.7%, and anomaly detection F1 increases by 1.2%.
- The model follows a scaling law: increasing layer depth, model dimension, and pre-training data volume all lead to sustained performance gains.
- Fine-tuning on only 10% of data surpasses the few-shot performance of TimesFM.

## Highlights & Insights
- **Temporal granularity-aware modeling in the frequency domain** is a particularly novel contribution: empirical analysis first identifies distributional differences across granularities, which are then incorporated into the model through low-rank matrices and adaptive attention weighting — an elegant way to embed domain priors. This technique is transferable to any scenario involving multi-scale temporal data (e.g., multi-resolution remote sensing, audio processing).
- **Hybrid mask pre-training** addresses a long-standing tension between two pre-training paradigms: reconstruction and autoregression. GTM unifies both via a probabilistic switch, enabling the same pre-trained model to seamlessly handle both imputation (reconstruction) and forecasting (autoregression).
- **Truly generative task-agnostic design**: forecasting, imputation, and anomaly detection all share an identical architecture without modification — a first in the TSFM literature.

## Limitations & Future Work
- The channel-independent (CI) strategy completely ignores cross-channel dependencies in multivariate settings — spatial modules (e.g., CPiRi-style methods) could be incorporated.
- Temporal granularity encoding relies on a manually defined five-tuple, which requires domain prior knowledge — automatic learning of granularity representations from data would be preferable.
- Classification still requires replacing the projection head (not fully task-agnostic) — prompt-based or in-context learning approaches could address this.
- The domain coverage of the UTSD-12G pre-training dataset may limit zero-shot generalization — on some datasets (e.g., Traffic), zero-shot performance falls short of Timer-1B.

## Related Work & Insights
- **vs Timer**: Timer employs pure autoregressive pre-training and requires strategy switching across tasks; GTM unifies objectives via hybrid masking.
- **vs PatchTST**: PatchTST pioneered CI + patch modeling but operates exclusively in the time domain; GTM extends this with frequency-domain analysis.
- **vs MOIRAI**: MOIRAI also addresses cross-frequency learning but uses a masked Transformer architecture; GTM's Fourier attention more explicitly models granularity differences.
- **vs UniTS**: UniTS supports multi-task learning via task tokenization, requiring task-specific tokens; GTM requires none.

## Rating
- Novelty: ⭐⭐⭐⭐ Fourier attention and hybrid mask pre-training represent meaningful innovations, though the overall framework builds on existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers forecasting, imputation, anomaly detection, classification, zero-shot, few-shot, ablation, and scaling law experiments comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though certain details (e.g., computational complexity analysis of Fourier attention) could be elaborated further.
- Value: ⭐⭐⭐⭐ The first generative task-agnostic TSFM; highly practical and suitable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](../../AAAI2026/time_series/mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)
- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](../../AAAI2026/time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)

</div>

<!-- RELATED:END -->
