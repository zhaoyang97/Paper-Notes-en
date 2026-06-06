---
title: >-
  [Paper Note] Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative
description: >-
  [ICLR2026][Time Series][multimodal time series] This paper identifies that time-series-paired texts exhibit periodicity analogous to that of time series (Chronological Textual Resonance), and proposes the TaTS framework…
tags:
  - "ICLR2026"
  - "Time Series"
  - "multimodal time series"
  - "text-augmented forecasting"
  - "Chronological Textual Resonance"
  - "plug-and-play framework"
date: 2026-05-08
content_hash: ce4aed8d69c0504c
---

# Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative

**Conference**: ICLR2026  
**arXiv**: [2502.08942](https://arxiv.org/abs/2502.08942)  
**Code**: [iDEA-iSAIL-Lab-UIUC/TaTS](https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS)  
**Area**: Time Series  
**Keywords**: multimodal time series, text-augmented forecasting, Chronological Textual Resonance, plug-and-play framework  

## TL;DR
This paper identifies that time-series-paired texts exhibit periodicity analogous to that of time series (Chronological Textual Resonance), and proposes the TaTS framework, which transforms text representations into auxiliary variables to enhance the forecasting and imputation performance of arbitrary existing time series models in a plug-and-play manner.

## Background & Motivation

### State of the Field

**Background**: Real-world time series data are frequently accompanied by textual information (e.g., infection rates paired with government announcements during a pandemic, or economic indicators paired with news reports), yet most existing models rely solely on numerical data, neglecting the complementary information contained in text.

### Root Cause

**Key Challenge**: The current best-performing multimodal method (MM-TSFLib), while incorporating text, overlooks the positional characteristics and periodicity inherent to time-series-paired texts.

### Limitations of Prior Work

**Limitations of Prior Work**: Motivated by the Platonic Representation Hypothesis (PRH)—which posits that representations of the same entity across different modalities converge toward a shared space—if time series and paired texts describe the same evolving event, both should exhibit similar periodicity.

### Starting Point

**Goal**: What unique properties do time-series-paired texts possess? How can such textual information be systematically integrated to improve time series modeling?

## Method

### 1. Chronological Textual Resonance (CTR)
- Frequency-domain analysis is conducted on three categories of real-world datasets (economic, social welfare, and traffic), revealing that the lag-similarity of paired text embeddings exhibits periodicity highly consistent with that of the corresponding time series.
- Specifically, a text encoder maps each time-step text $s_t$ to an embedding $e_t$; the lag similarity $d_l = \sum_t \cos(e_t, e_{t+L})$ is computed, and FFT is applied to $d_l$ to identify dominant frequencies.
- Three factors explain CTR: (1) shared external drivers (seasonal cycles, economic cycles, etc.); (2) text reflects time series trends; (3) text contains additional variables with aligned periodicity.

### 2. TT-Wasserstein Metric
- A TT-Wasserstein metric is proposed to quantify the degree of CTR: the Wasserstein distance between the normalized spectral distributions of the time series and the text is computed.
- Lower values indicate higher alignment between text and time series.
- Validation: timestamp shuffling on the Time-MMD dataset causes a significant increase in TT-Wasserstein, confirming that the metric captures cross-modal alignment quality.

### 3. Texts as Time Series (TaTS) Framework
The overall pipeline consists of three steps:

**Step 1: Text Encoding**  
A pretrained language model (GPT-2 by default) encodes the text at each time step: $e_t = \mathcal{H}_{\text{text}}(s_t) \in \mathbb{R}^{d_{\text{text}}}$

**Step 2: Dimensionality Mapping**  
A three-layer MLP projects the high-dimensional text embeddings into a lower-dimensional space: $z_t = \text{MLP}(e_t) \in \mathbb{R}^{d_{\text{mapped}}}$

**Step 3: Concatenation and Modeling**  
The mapped text representations are concatenated with the original time series as auxiliary variables: $U = [X; Z^\top] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$, which is then fed into any existing time series model; the MLP and time series model parameters are jointly optimized.

- Key design: text is treated as auxiliary variables of the time series without modifying the downstream model architecture.
- At inference, only the first $N$ variables are retained as output; the auxiliary variable dimensions are discarded.

## Key Experimental Results

### Forecasting Task (9 Time-MMD Datasets × 9 Models)
- TaTS outperforms both uni-modal baselines and MM-TSFLib across all datasets.
- Average improvement exceeds 5% on 6 out of 9 datasets, with the largest dataset (Environment) showing improvements exceeding 30%.
- The Economy dataset yields the most significant gains: iTransformer MSE drops from 0.014 to 0.008 (↓ 42.9%), and Transformer MSE drops from 0.584 to 0.079 (↓ 86.5%).

### Imputation Task (Climate/Economy/Traffic)
- Maximum improvement of 67.2% (Economy dataset, PatchTST MAE).

### Comparison with Other Baselines (Table 4)
- Significantly outperforms covariate/convolutional methods such as N-BEATS, N-HiTS, and TCN.
- Outperforms ChatTime (zero-shot multimodal foundation model) and GPT4MTS.

### Relationship Between TT-Wasserstein and Performance Gain
- Lower original-to-shuffled TT-Wasserstein ratios correspond to larger TaTS improvements (e.g., Economy ratio 22.3%, improvement 64.8%).

### Efficiency
- The MLP adds approximately 1% additional parameters and increases training time by approximately 8%, while delivering an average performance improvement of approximately 14%.

## Highlights & Insights
- **Novel Insight**: The CTR phenomenon is identified and formalized for the first time, providing a theoretical perspective for multimodal time series research.
- **Simple yet Effective**: No modification to any downstream model architecture is required; a lightweight MLP suffices for plug-and-play integration.
- **Strong Generalizability**: Compatible with 9 mainstream time series models (Transformer/Linear/Frequency-based) and supports both forecasting and imputation tasks.
- **Practical Metric**: TT-Wasserstein can predict the potential gain from text-augmented modeling, guiding practical application decisions.

## Limitations & Future Work
- The text encoder is fixed as a pretrained LM (GPT-2); end-to-end fine-tuning of the text encoder remains unexplored.
- The MLP mapping dimension $d_{\text{mapped}}$ requires manual specification; although experiments suggest low sensitivity, an automatic selection mechanism is absent.
- When text quality is very low (e.g., randomly shuffled), TaTS may slightly underperform pure numerical models; while a mitigation strategy of discarding low-quality text is provided, automatic detection has not been implemented.
- Only timestamp-level paired texts are considered; irregular or asynchronous text–time series pairing scenarios are not addressed.
- The fusion approach is evaluated only with MLP, gated residual, and cross-attention mechanisms; more sophisticated fusion strategies may yield further improvements.

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|----------------|-------------|
| MM-TSFLib | First multimodal time series library | Ignores positional properties of text |
| ChatTime | Zero-shot multimodal reasoning | Underperforms supervised TaTS |
| N-BEATS/N-HiTS | Covariate modeling | Not designed for text; poor performance |
| StockNet/Dandelion | Text fusion in financial domain | Not timestamp-aligned; poor generalizability |
| **TaTS (Ours)** | Plug-and-play; treats text as auxiliary variables | Requires timestamp-aligned paired text |

## Related Work & Insights
- The CTR phenomenon is essentially a concrete instantiation of the PRH in the multimodal time series setting, offering a framework for exploring the alignment of other modalities (e.g., images, audio) with time series.
- TT-Wasserstein as a data quality metric can be generalized to other multimodal settings for evaluating cross-modal alignment.
- The paradigm of "text as auxiliary variables" may be applicable to other heterogeneous data fusion scenarios (e.g., using knowledge graph embeddings as auxiliary variables for time series).
- Performance improves incrementally with larger LMs (BERT → GPT-2 → LLaMA2), suggesting that stronger text encoders could further unlock multimodal potential.

## Rating
- Novelty: ⭐⭐⭐⭐ — The CTR phenomenon and TT-Wasserstein metric are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 18 datasets, 9 models, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and rich figures and tables.
- Value: ⭐⭐⭐⭐ — The plug-and-play design is highly practical, though applicability is limited by the availability of paired text.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

</div>

<!-- RELATED:END -->
