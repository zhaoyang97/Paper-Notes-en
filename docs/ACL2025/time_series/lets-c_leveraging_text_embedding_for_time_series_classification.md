---
title: >-
  [Paper Note] LETS-C: Leveraging Text Embedding for Time Series Classification
description: >-
  [ACL 2025][Time Series][time series classification] LETS-C is proposed: it digitizes time series into text strings, encodes them using a text embedding model, merges them with the original time series via element-wise addition, and feeds them into a lightweight CNN-MLP classification head. With only 14.5% of the trainable parameters, it achieves SOTA, outperforming 27 baselines including OneFitsAll (fine-tuned GPT-2) on 10 UEA multivariate time series datasets.
tags:
  - "ACL 2025"
  - "Time Series"
  - "time series classification"
  - "text embedding"
  - "cross-modal transfer"
  - "lightweight model"
  - "CNN-MLP"
date: 2026-05-08
content_hash: 6c306a7d1ce919d3
---

# LETS-C: Leveraging Text Embedding for Time Series Classification

**Conference**: ACL 2025  
**arXiv**: [2407.06533](https://arxiv.org/abs/2407.06533)  
**Code**: None (uses OpenAI API)  
**Area**: Time Series  
**Keywords**: time series classification, text embedding, cross-modal transfer, lightweight model, CNN-MLP

## TL;DR
LETS-C is proposed: it digitizes time series into text strings, encodes them using a text embedding model, merges them with the original time series via element-wise addition, and feeds them into a lightweight CNN-MLP classification head. With only 14.5% of the trainable parameters, it achieves SOTA, outperforming 27 baselines including OneFitsAll (fine-tuned GPT-2) on 10 UEA multivariate time series datasets.

## Background & Motivation

**Background**: Time series classification is a crucial task in domains such as finance, healthcare, and activity recognition. Recently, LLM fine-tuning methods (such as OneFitsAll based on GPT-2) have achieved SOTA performance on standard benchmarks, but they require millions of trainable parameters.

**Limitations of Prior Work**: LLM fine-tuning methods (a) feature huge model sizes (requiring millions of trainable parameters even when freezing most parameters); (b) involve high inference costs; (c) are impractical for resource-constrained scenarios.

**Key Challenge**: Can the success of language models in sequence modeling be harnessed while avoiding the high cost of fine-tuning LLMs?

**Goal**: Encode time series using a text embedding model (rather than LLM fine-tuning) coupled with a lightweight classification head to achieve efficient classification.

**Key Insight**: Text embedding models have demonstrated powerful sequence representation capabilities on MTEB, and their inference is one-time (which can be precomputed and stored), making them far more efficient than LLM fine-tuning.

**Core Idea**: Time series $\rightarrow$ Digit strings $\rightarrow$ Text embedding $\rightarrow$ Fusion with original sequence $\rightarrow$ CNN-MLP classification.

## Method

### Overall Architecture
Input: Multivariate time series $\mathbf{x}_i \in \mathbb{R}^{d \times l_x}$. Steps: (1) min-max normalization $\rightarrow$ (2) format each dimension into a digit-space text string $\rightarrow$ (3) encode into $\mathbf{e}_i \in \mathbb{R}^{d \times l_e}$ using text-embedding-3-large $\rightarrow$ (4) element-wise addition of the embedding and original time series (aligned with zero-padding) $\rightarrow$ (5) 1D CNN + MLP classification.

### Key Designs

1. **Digit-Space Tokenization**

    - **Function**: Convert floating-point numbers into a string where each digit is tokenized independently.
    - **Mechanism**: `0.645, 6.45` $\rightarrow$ `"6 4 , 6 4 5"`, where commas separate time steps and spaces separate digits.
    - **Design Motivation**: Subword tokenizers like BPE arbitrarily split digits (Gruver et al., 2024), resulting in large representation discrepancies for similar numbers. Digit-space ensures each digit is tokenized independently, preserving numerical integrity.

2. **Text Embedding Model Encoding**

    - **Function**: Encode the formatted text string using OpenAI text-embedding-3-large (3072 dimensions).
    - **Mechanism**: Each dimension is encoded independently (channel-wise), yielding a $d \times l_e$ embedding matrix. **Embeddings are precomputed in a one-time manner**, allowing storage and reuse.
    - **Design Motivation**: Text embedding models are trained on large-scale text and possess an inherent understanding of sequence patterns; furthermore, inference is an API call requiring no local GPUs.

3. **Element-wise Addition Fusion**

    - **Function**: $\text{fused} = \text{embedding} + \text{timeseries}$ (aligned using zero-padding).
    - **Design Motivation**: Similar to the shortcut connection in ResNet—the embedding provides high-level semantic features while the original time series retains precise numerical information. Experiments demonstrate that addition outperforms concatenation or attention-based fusion.

4. **Lightweight CNN+MLP Classification Head**

    - **Function**: 1D CNN extracts local patterns $\rightarrow$ flatten $\rightarrow$ MLP + softmax classification.
    - **Design Motivation**: **Intentionally simple**—to verify that the text embedding itself already provides sufficiently strong feature representations without needing complex models.

## Key Experimental Results

### Main Results (Average on 10 UEA Datasets)

| Method | Average Accuracy | Trainable Parameters | AvgWins% |
|------|-----------|-----------|---------|
| DTW | 66.97% | - | 0% |
| TimesNet | 73.60% | ~1.5M | 10% |
| PatchTST | 74.33% | ~1.2M | 20% |
| OneFitsAll (GPT-2) | 75.20% | ~1.0M | 40% |
| MOMENT | 76.50% | ~0.8M | 30% |
| **LETS-C** | **78.56%** | **~0.14M** | **90%** |

### Parameter Efficiency Comparison

| Method | Trainable Parameters | vs LETS-C |
|------|-----------|----------|
| OneFitsAll | ~1.0M | 6.9× |
| TimesNet | ~1.5M | 10.3× |
| **LETS-C** | **~0.14M** | **1×** |

### Ablation Study

| Configuration | Average Accuracy |
|------|-----------|
| Text embedding only (no original TS) | 73.2% |
| Original TS only (no embedding) | 71.8% |
| Concatenation fusion | 76.1% |
| **Additive fusion (LETS-C)** | **78.56%** |

### Key Findings
- **Intra-class cohesion of text embeddings**: The cosine similarity of text embeddings for time series of the same class is significantly higher than those of different classes, indicating that text embeddings inherently possess discriminative capacity for time series.
- **LETS-C ranks in the top-2 for 9 out of 10 datasets**, showing highly robust generalization.
- **With only 14.5% of SOTA's parameter count**, it holds direct deployment value in industrial scenarios like JPMorgan.
- **Multiple text embedding models are effective**: Beyond OpenAI, open-source models such as GTE and Mistral-embed also achieve performance outperforming OneFitsAll.
- **Friendly to model compression**: Truncating the embedding dimension from 3072 to 768 leads to a loss of less than 2% in accuracy.

## Highlights & Insights
- **The surprising finding that "text embedding understands time series"**: Text encoders have never seen time series data, yet embeddings of digit-space formatted numerical strings exhibit strong discriminative power. This suggests that language models possess an implicit comprehension capability for numerical sequence patterns.
- **Extremely simple design**: The method essentially converts numbers to strings $\rightarrow$ calls an API $\rightarrow$ adds them back $\rightarrow$ runs a small network, yet it surpasses all complex approaches.
- **One-time precomputation of embeddings**: No GPU is required to train the embedding; once called via API, they can be stored and reused, leading to extremely simple deployment.
- **Challenging the "necessity of LLM fine-tuning"**: It is unnecessary to access the internal layers of a Transformer; leveraging only the embedding interface is sufficient.

## Limitations & Future Work
- **Reliance on commercial APIs**: OpenAI's text-embedding-3-large is a paid API, limiting its cost-efficiency and controllability.
- **Limited to multivariate classification**: It has not been extended to other time series tasks such as forecasting or anomaly detection.
- **No validation set**: The benchmark setup is consistent with prior work (train/test split only), thus reporting the upper bound.
- **Why do text embeddings work for time series?**: There is a lack of mechanistic explanation—are patterns of numerical sequences implicitly learned during BPE training?
- **Constrained by long time-series**: Limited by the token limit (8191 tokens).

## Related Work & Insights
- **vs OneFitsAll (Zhou et al., 2024)**: Fine-tunes all parameters of GPT-2, yielding lower accuracy than LETS-C while requiring 7 times more parameters.
- **vs MOMENT (Goswami et al., 2024)**: A pre-trained time-series foundation model that learns representations via self-supervised learning; in contrast, LETS-C obtains representations through text embeddings with zero training.
- **vs TS2Vec/TNC**: Contrastive learning methods require pre-training on domain-specific data, whereas LETS-C directly exploits the "cross-domain" representation capabilities of language models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first to apply text embedding to time series classification; the idea is exceptionally simple yet yields striking results.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 27 baselines + 10 datasets + multiple embedding models + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the method; Figure 1 is highly intuitive.
- Value: ⭐⭐⭐⭐⭐ Initiates a brand-new paradigm of "using language models as feature extractors" for time series analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] QuITE: Query-based Irregular Time Series Embedding](../../ICML2026/time_series/quite_query-based_irregular_time_series_embedding.md)
- [\[ICML 2025\] Learning Soft Sparse Shapes for Efficient Time-Series Classification](../../ICML2025/time_series/learning_soft_sparse_shapes_for_efficient_time-series_classification.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ICLR 2026\] MambaSL: Exploring Single-Layer Mamba for Time Series Classification](../../ICLR2026/time_series/mambasl_exploring_single-layer_mamba_for_time_series_classification.md)
- [\[ICLR 2026\] Unlocking the Value of Text: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)

</div>

<!-- RELATED:END -->
