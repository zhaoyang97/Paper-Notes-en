---
title: >-
  [Paper Note] Scaling Embedding Layers in Language Models
description: >-
  [NeurIPS 2025][LLM Pretraining][embedding scaling] This paper proposes Scone, a method that learns contextualized embeddings for high-frequency n-grams using a separate Transformer model, and offloads these embeddings to main memory/SSD at inference time. This enables a new scaling paradigm in which additional compute is consumed during training without increasing accelerator resource usage at inference. A 1B-parameter Scone model surpasses a 1.9B baseline.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - embedding scaling
  - n-gram embeddings
  - reasoning efficiency
  - offloading
  - Scone
date: 2026-05-08
content_hash: d14505a40fc0ce78
---

# Scaling Embedding Layers in Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.01637](https://arxiv.org/abs/2502.01637)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: embedding scaling, n-gram embeddings, reasoning efficiency, offloading, Scone

## TL;DR
This paper proposes Scone, a method that learns contextualized embeddings for high-frequency n-grams using a separate Transformer model, and offloads these embeddings to main memory/SSD at inference time. This enables a new scaling paradigm in which additional compute is consumed during training without increasing accelerator resource usage at inference. A 1B-parameter Scone model surpasses a 1.9B baseline.

## Background & Motivation

### State of the Field

**Background**: The conventional scaling approach is to increase model parameters — but this simultaneously increases inference-time FLOPs and accelerator memory.

**Limitations of Prior Work**: Expanding embeddings by enlarging the vocabulary introduces two problems: (1) the output layer grows in tandem, causing decoding costs to escalate; (2) tail tokens receive insufficient training.

**Key Challenge**: Inference costs often far exceed training costs (models are queried billions of times), yet conventional scaling tightly couples inference cost with training compute.

**Goal**: Identify a new scaling paradigm in which more compute can be consumed during training while accelerator resource requirements at inference remain unchanged.

**Key Insight**: Embedding lookup is fundamentally a memory-read operation (compute-free) and can be offloaded to main memory/SSD with negligible latency impact. Contextualized embeddings for high-frequency n-grams can be precomputed and cached.

**Core Idea**: Train a separate Transformer to learn contextualized embeddings for high-frequency n-grams, precompute and offload them at inference time, thereby decoupling training-time scaling from inference cost.

## Method

### Overall Architecture
Construct a high-frequency n-gram set (f-grams) → Train a separate f-gram Transformer to learn contextualized embeddings → Precompute all f-gram embeddings before inference and store them in main memory/SSD → At inference, match each input token sequence to the longest f-gram, substitute the cached embedding for the original token embedding → Feed into the main model.

### Key Designs

1. **F-gram Selection**:

    - Function: Select the most frequent n-grams (n = 2 to K) from the training corpus.
    - Mechanism: A greedy merging strategy inspired by BPE; K−1 linear passes over the corpus to select the highest-frequency candidates.
    - Design Motivation: High-frequency n-grams cover the majority of token occurrences; low-frequency ones are undertrained and not worth including.

2. **F-gram Transformer Model**:

    - Function: A standalone small Transformer that takes the token embedding sequence of an n-gram as input and outputs a single contextualized embedding vector.
    - Mechanism: $e_i = \mathcal{A}_{f\text{-}gram}(\mathcal{T}(\sigma_j), ..., \mathcal{T}(\sigma_i))$; trained end-to-end jointly with the main model.
    - Design Motivation: More expressive than a lookup table — capable of compositionally capturing n-gram semantics, and the f-gram model can be scaled independently.

3. **Inference-Time Offloading**:

    - Function: After training, precompute all f-gram embeddings and store them in main memory or NVMe SSD.
    - Mechanism: Embedding lookups at inference are served from main memory/SSD, consuming no accelerator resources. Main memory latency is negligible; NVMe introduces minimal overhead and does not become a bottleneck.
    - Design Motivation: Embedding lookup is an O(1) memory-read operation, making it inherently suitable for offloading.

### Two New Scaling Dimensions
1. **Increasing the number of f-grams**: More n-grams → more contextualized embeddings → richer input representations (requires only more main memory).
2. **Scaling up the f-gram model**: A larger Transformer learns higher-quality embeddings (requires only more training compute).

## Key Experimental Results

### Main Results

| Model | Accelerator Parameters | Inference FLOPs | Perplexity |
|-------|----------------------|-----------------|------------|
| Baseline 1.9B | 1.9B | ~2x | Baseline |
| Scone 1B + 10M f-grams | 1B | ~1x | Matches 1.9B |
| Scone 1B + 1B f-grams | 1B | ~1x | **Surpasses 1.9B** |

### Key Findings
- 10M f-grams suffice to bring a 1.3B model on par with the 1.9B baseline.
- 1B f-grams allow a 1B model to surpass the 1.9B baseline with roughly half the inference FLOPs and memory.
- Storing f-gram embeddings in main memory introduces virtually no latency increase.
- NVMe storage incurs minor latency but does not constitute a bottleneck.
- Scaling the f-gram model to larger sizes yields consistent gains.

## Highlights & Insights
- **New Scaling Paradigm**: Challenges the assumption that better models necessarily require more inference compute. Offloading embeddings achieves "scale at training time, free at inference time."
- **High Practical Utility**: Main memory is 10–100× cheaper than GPU memory; storing 1B embeddings requires only tens of gigabytes.
- **Elegant Connection to BPE**: The f-gram selection strategy is inspired by BPE but does not modify the tokenizer — thereby avoiding the output-layer cost increase associated with vocabulary expansion.

## Limitations & Future Work
- The longest-match strategy for f-grams may be suboptimal; shorter n-gram embeddings may sometimes be preferable.
- Storage cost for precomputed f-gram embeddings scales linearly with the number of f-grams.
- Validation is limited to decoder-only architectures.
- Joint training of the f-gram model and the main model may increase training complexity.

## Related Work & Insights
- **vs. vocabulary expansion**: Directly enlarging the vocabulary increases logit computation cost; Scone decouples the input and output sides.
- **vs. MoE**: MoE models also avoid activating all parameters at inference, but the inactive parameters still reside on the accelerator. Scone moves the additional parameters entirely off the accelerator.
- **Core Insight**: The embedding layer is the least computationally expensive operation in an LLM, and its scaling potential has been severely underestimated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — An entirely new scaling paradigm, elegant and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-scale model comparisons with offloading latency benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation and concise method description.
- Value: ⭐⭐⭐⭐⭐ — Significant practical implications for inference efficiency optimization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](retrospective_incontext_learning_for_temporal_credit_assignm.md)

<!-- RELATED:END -->
