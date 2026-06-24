---
title: >-
  [Paper Note] Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity
description: >-
  [ACL 2025][embedding capacity] By compressing text into trainable [mem] vectors via a per-sample optimization method, it is discovered that Llama-3.1-8B can losslessly compress 1568 tokens into a single input vector. This reveals a gap of two orders of magnitude between existing methods (approx. x10 compression ratio) and the practically achievable limit (x1500+).
tags:
  - "ACL 2025"
  - "embedding capacity"
  - "token compression"
  - "lossless compression"
  - "trainable memory vector"
  - "LLM input representation"
date: 2026-05-08
content_hash: e6cede6c6c300641
---

# Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity

**Conference**: ACL 2025  
**arXiv**: [2502.13063](https://arxiv.org/abs/2502.13063)  
**Code**: GitHub (Yes, mentioned in the paper)  
**Area**: Others  
**Keywords**: embedding capacity, token compression, lossless compression, trainable memory vector, LLM input representation

## TL;DR

By compressing text into trainable [mem] vectors via a per-sample optimization method, it is discovered that Llama-3.1-8B can losslessly compress 1568 tokens into a single input vector. This reveals a gap of two orders of magnitude between existing methods (approx. x10 compression ratio) and the practically achievable limit (x1500+).

## Background & Motivation

**Background**: LLMs are based on the Transformer architecture where the input embedding dimension increases with model size (from 2,048 to 16,384 for the Llama series), yet each embedding still represents only a single token. However, a 2048-dimensional 16-bit vector can theoretically encode about 32,768 bits, which is sufficient to represent approximately 1,931 tokens.

**Limitations of Prior Work**: Existing context compression methods (e.g., AutoCompressors, ICAE, 500xCompressor) rely on encoder models, where the maximum lossless compression ratio typically does not exceed x10, while lossy compression can reach x500 but with severe quality degradation.

**Key Challenge**: The theoretical information capacity of input vectors is far higher than the currently achievable practical compression ratio, indicating a massive utilization gap.

**Goal**: Explore the compression limit of LLM input representations—exactly how many tokens can be crammed into a single vector?

**Key Insight**: Abandon the encoder (the encoder itself might be a bottleneck) and adopt per-sample optimization—directly optimize a set of [mem] vectors for each text segment, using a frozen LLM for decoding.

**Core Idea**: By optimizing trainable input vectors on a per-sample basis, directly explore the information capacity limit of the LLM input embedding space.

## Method

### Overall Architecture

Prefix the text to be compressed $[t_1, ..., t_N]$ with a set of trainable [mem] vectors $[m_1, ..., m_K]$. Freeze all LLM parameters and only optimize the [mem] vectors to minimize the standard next-token prediction cross-entropy loss. During inference, generation starts from the learned [mem] vectors, letting the LM autoregressively decode the original text.

### Key Designs

1. **Per-sample Optimization**: Instead of training a general encoder, separate [mem] vectors are optimized independently for each text segment. Although computationally expensive, this reveals the theoretical upper bound of capacity. The optimization process is similar to Prompt Tuning, but the objective is to precisely reconstruct the original text.
2. **Three-metric Evaluation System**:

    - Decoding Capacity (in tokens): The maximum text length that can be perfectly reconstructed (>99% accuracy)
    - Token Gain: The number of additional tokens correctly decoded by the [mem] vector (subtracting those predictable by the LLM itself)
    - Information Gain: The reduction in cross-entropy (in bits), measuring the ability of the [mem] vector to reduce uncertainty
3. **Cross-domain Validation**: Verification is performed across three types of texts: natural text (PG-19 books), newly published text (fanfiction post-2024), and random word sequences, proving that the capacity does not depend on text predictability or training data memorization.

### Loss & Training

- Standard next-token prediction cross-entropy loss
- Only optimize [mem] vectors ($K \times d_{model}$ parameters), while the LLM parameters are completely frozen
- Independent optimization for each text segment, similar to Prompt Tuning but with a different objective

## Key Experimental Results

### Main Results

Decoding capacity of a single [mem] vector (PG-19 natural text):

| Model | Decoding Capacity (tokens) | Token Gain | Information Gain (bits) |
|------|----------------|----------|--------------|
| Pythia-160M | 80 | 70.9 | 396.4 |
| Pythia-1.4B | 160 | 158.0 | 792.8 |
| Llama-3.2-1B | 512 | 426.2 | 2119.9 |
| Llama-3.2-3B | 1024 | 720.3 | 3292.2 |
| **Llama-3.1-8B** | **1568** | **1094.1** | **4865.7** |

Results on random text (no language model predictability):

| Model | Decoding Capacity | Token Gain | Information Gain |
|------|---------|----------|---------|
| Llama-3.2-1B | 316 | 294.9 | 2265.2 |
| Llama-3.1-8B | 792 | 623.2 | 4541.2 |

### Ablation Study

Linear scaling of multiple vectors:

| Model | Number of [mem] | Perfectly Decodable Length |
|------|----------|-------------|
| Pythia-160M | 1 | 80 |
| Pythia-160M | 32 | 2016 (close to the context upper limit of 2048) |
| Llama-3.2-1B | 1 | 512 |
| Llama-3.2-1B | 16 | **7168** |

### Key Findings

- **Capacity limit is determined by cross-entropy**: Regardless of text length or domain, as long as the total cross-entropy of the text is below the model-specific information gain threshold, perfect reconstruction can be achieved. This threshold is an intrinsic property of the model.
- **PG-19 ≈ Fanfiction**: Compression performance on PG-19 (which might have been seen during training) and on fanfiction unseen by the model is almost identical, showing that capacity does not rely on training data memorization.
- **Linear scaling of capacity**: The capacity of multiple [mem] vectors grows approximately linearly.
- **Architecture-independent**: Mamba (a state space model) also demonstrates similar compression behavior.
- **Capacity utilization**: Newer models (Llama, OLMo) achieve higher utilization than older models (Pythia, OPT), which might reflect pre-training quality.

## Highlights & Insights

- Reveals a striking gap of two orders of magnitude—existing encoder methods (x10) vs. the achievable limit (x1500), pointing out a massive optimization space for future research.
- The linear relationship between information gain and text cross-entropy is highly elegant—directly connecting the compression limit to information theory.
- Discovers that "capacity utilization" might serve as a new metric for measuring pre-training quality.
- An interesting corollary: *The Hobbit* (~120,000 tokens) could be compressed into just 128 input vectors using Llama-3.1-8B.

## Limitations & Future Work

- Per-sample optimization is computationally extremely expensive and not suitable for practical applications at present—its core value lies in revealing the theoretical upper bound.
- The semantic structure and representation properties of the learned [mem] vectors remain unclear.
- The largest model tested is only 8B, and the scaling laws for larger models remain unknown.
- Random texts use dictionary words instead of directly sampling from the tokenizer, which might slightly overestimate the capacity.
- The direct usability of the compressed vectors in downstream tasks has not been explored.

## Related Work & Insights

- Complementary to methods like ICAE (x4 compression) and 500xCompressor (x480 lossy)—this paper provides a reference for the theoretical upper bound.
- Methodologically related to Memory Transformer and Prompt Tuning, but with different goals—exploring information capacity rather than adapting to new tasks.
- Highly significant for latent space reasoning—the input embedding capacity determines the upper bound for reasoning in the latent space.
- Insight: Future compression methods have space for improvement of at least two orders of magnitude.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The approach of using per-sample optimization to explore the upper bound of capacity is highly novel, and the findings are profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, multiple data sources, and multiple metrics are included, but it lacks larger scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous information theory framework, clear definitions of metrics, and excellent visualization.
- Value: ⭐⭐⭐⭐⭐ Redefines the pursuit targets for the entire context compression field and indicates a massive optimization space.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [\[ICLR 2026\] Exploring State-Space Models for Data-Specific Neural Representations](../../ICLR2026/others/exploring_state-space_models_for_data-specific_neural_representations.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](../../ICLR2026/others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)

</div>

<!-- RELATED:END -->
