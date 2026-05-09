---
title: >-
  [Paper Note] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression
description: >-
  [NeurIPS 2025][Time Series][KV cache compression] AttentionPredictor is the first learning-based method that directly predicts attention patterns for KV cache compression and critical token identification. By leveraging a lightweight CNN to capture spatiotemporal patterns in attention scores, it achieves 13× KV cache compression and 5.6× inference speedup, with a unified prediction model of only 21 KB shared across all Transformer layers.
tags:
  - NeurIPS 2025
  - Time Series
  - KV cache compression
  - attention prediction
  - temporal patterns
  - LLM inference acceleration
  - cache prefetching
date: 2026-05-08
content_hash: 0705173784ef2ff5
---

# AttentionPredictor: Temporal Patterns Matter for KV Cache Compression

**Conference**: NeurIPS 2025
**arXiv**: [2502.04077](https://arxiv.org/abs/2502.04077)
**Code**: [GitHub](https://github.com/MIRALab-USTC/LLM-AttentionPredictor)
**Area**: Time Series / Efficient Inference
**Keywords**: KV cache compression, attention prediction, temporal patterns, LLM inference acceleration, cache prefetching

## TL;DR
AttentionPredictor is the first learning-based method that directly predicts attention patterns for KV cache compression and critical token identification. By leveraging a lightweight CNN to capture spatiotemporal patterns in attention scores, it achieves 13× KV cache compression and 5.6× inference speedup, with a unified prediction model of only 21 KB shared across all Transformer layers.

## Background & Motivation

**Background** The KV cache is the primary memory bottleneck in long-context LLM inference (a 7B model requires 72 GB at 128K context length). Sparse attention methods compress the cache by retaining only "critical tokens."

**Limitations of Prior Work** (1) Heuristic methods (H2O, SnapKV) use static rules to assess token importance and fail to capture dynamic attention changes; (2) learning-based methods (SeerAttention) require training a separate model per layer (101 MB in total) and do not directly model the attention score distribution.

**Key Challenge** Attention patterns exhibit complex dynamic temporal characteristics (re-access, sequential, seasonal), yet existing approaches either rely on simple heuristics (insufficient accuracy) or learning methods that encode only keys/hidden states (insufficiently direct).

**Goal** Directly predict the attention score distribution at the next step to precisely identify critical tokens and achieve high compression ratios.

**Key Insight** Attention scores exhibit predictable spatiotemporal regularities along the time axis—arising from query self-similarity and the intrinsic properties of positional encodings—and can be formulated as a 2D time series prediction problem.

**Core Idea** Formalize KV cache compression as a 2D temporal prediction problem over attention scores, replacing heuristic rules with a lightweight CNN predictor to identify critical tokens.

## Method

### Overall Architecture
The core pipeline of AttentionPredictor: (1) during the prefill phase, prepare the attention history sequence $\mathcal{A}_H$; (2) at each decoding step: apply block-wise max-pooling to compress the current attention scores → update the history sequence → use the pretrained CNN to predict the next-step attention → select Top-K critical token positions → expand into the final index set $S$.

### Key Designs

1. **Discovery and Theoretical Analysis of Temporal Attention Patterns**:
    - Function: Identify three predictable temporal patterns in attention scores.
    - Mechanism: (1) **Re-access** $(\delta_t=1,\delta_i=0)$: sustained focus on fixed tokens; (2) **Sequential** $(\delta_t=1,\delta_i=1)$: attention advances token by token (driven by RoPE's relative position dependency); (3) **Seasonal** $(\delta_t>1,\delta_i=0)$: periodic revisiting of fixed positions. Unified description: $a_{t,i} \approx a_{t+\delta_t, i+\delta_i}$.
    - Design Motivation: Theoretical analysis demonstrates that high query self-similarity (cosine autocorrelation of 0.87) and RoPE positional encoding are the intrinsic sources of these patterns, making temporal prediction feasible.

2. **Lightweight Spatiotemporal CNN Predictor**:
    - Function: Predict the next-step attention distribution from attention history.
    - Mechanism: Two layers of 2D convolution capture multi-scale spatiotemporal features, followed by a 1D convolution focusing on the temporal dimension. The 1D convolution replaces fully connected layers to accommodate variable spatial dimensions. The entire model is only **21 KB** (one-millionth of LLaMA-3.1-8B) and **a single predictor is shared across all layers and heads**.
    - Design Motivation: Temporal attention patterns are intrinsic properties of LLMs (query similarity + positional encoding) that do not vary across layers, heads, or datasets, enabling a single lightweight model to capture them universally.

3. **Cross-Token KV Cache Prefetching Framework**:
    - Function: Hide prediction and cache transfer latency to accelerate decoding.
    - Mechanism: Unlike existing cross-layer prefetching, AttentionPredictor uses the inference time of the current token to predict critical cache indices for the next token and asynchronously prefetches the corresponding KV pairs from CPU—cross-token prefetching exploits a longer transfer time window.
    - Design Motivation: The prefetching window in cross-layer methods spans only a single-layer inference time, which is insufficient to hide large-scale transfers; the cross-token window spans the full single-token inference time.

### Additional Techniques
- **Block-wise Compression**: Max-pooling compresses the attention vector by a factor of $1/b$ ($b=16$), reducing prediction overhead.
- **Distribution Error Calibration**: Full attention is computed every $M=5$ steps to correct the distribution shift introduced by sparse computation.

## Key Experimental Results

### Main Results — LongBench (LLaMA-3.1-8B, 1K/2K/4K budget)

| Method | 1K Avg. | 2K Avg. | 4K Avg. |
|--------|---------|---------|---------|
| Full cache | 48.17 | 48.17 | 48.17 |
| StreamingLLM | 42.35 | 41.64 | - |
| H2O+ | 44.25 | 46.34 | - |
| SnapKV | 46.37 | 47.44 | - |
| Quest | 46.92 | - | - |
| **AttentionPredictor** | **48.58** | **48.82** | - |

### Performance Metrics

| Metric | Value |
|--------|-------|
| KV cache compression ratio | 13× |
| Speedup in cache offloading scenario | 5.6× |
| Prediction model size | **21 KB** (one-millionth of the LLM) |
| SeerAttention model size | 101 MB (0.02% of AttentionPredictor's) |

### Ablation Study

| Configuration | Attention Recall | Notes |
|---------------|-----------------|-------|
| H2O (heuristic) | Lower | Biased cumulative scores |
| Quest (compressed retrieval) | Medium | Degrades with large page size |
| AttentionPredictor | **Highest** | Most accurate via direct prediction |
| + Calibration ($M=5$) | Further improved | Corrects sparse distribution shift |

### Key Findings
- Under extreme compression at a 1K budget, AttentionPredictor is nearly lossless (48.58 vs. Full 48.17).
- Prediction accuracy exceeds heuristic methods by 5%+ and also surpasses fixed-template methods (MInference) by 5%+.
- Training on only ~3% of the attention data generalizes to the full dataset (trained on LongBench → generalized to GSM8K).

## Highlights & Insights
- The perspective of framing "attention prediction as time series forecasting" is novel and well-grounded theoretically (query self-similarity + RoPE).
- The 21 KB cross-layer shared model represents an extremely lightweight design; the contrast with SeerAttention's 101 MB highlights the paradigm's advantage.
- The cross-token prefetching framework provides a larger latency-hiding window than cross-layer prefetching, constituting a practically valuable system optimization.

## Limitations & Future Work
- Block-wise compression may sacrifice accuracy on tasks requiring fine-grained token-level discrimination.
- The calibration step requires full attention computation every $M$ steps, incurring a fixed cost on very long sequences.
- Validation is currently limited to 7B–8B models; performance on larger models remains to be confirmed.

## Related Work & Insights
- **vs. H2O/SnapKV**: These rely on heuristic cumulative scores; AttentionPredictor employs a learned predictor to capture dynamic changes.
- **vs. SeerAttention**: That method trains a per-layer model and indirectly encodes keys; AttentionPredictor uses a single 21 KB model to directly predict attention.
- **vs. InfiniGen**: Cross-layer prefetching has a short window; AttentionPredictor's cross-token prefetching provides a larger time budget.

## Rating
- Novelty: ⭐⭐⭐⭐ First learning-based compression method to directly predict attention patterns, with rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across LongBench, InfiniteBench, AIME, GSM8K, and MMLU.
- Writing Quality: ⭐⭐⭐⭐ Theoretical analysis and system design are well integrated.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for long-context LLM inference.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] How Patterns Dictate Learnability in Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)

<!-- RELATED:END -->
