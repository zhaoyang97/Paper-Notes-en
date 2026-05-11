---
title: >-
  [Paper Note] Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis
description: >-
  [ICLR 2026][Model Compression][attention patterns] This paper proposes the TAPPA framework, which explains the formation mechanisms of various attention patterns in LLMs (attention sink, diagonal, periodic, etc.) from a temporal continuity perspective in a unified manner, and leverages query self-similarity (q-similarity) as a metric to guide KV cache compression and model pruning tasks.
tags:
  - ICLR 2026
  - Model Compression
  - attention patterns
  - temporal analysis
  - RoPE
  - query self-similarity
  - KV cache compression
  - LLM pruning
date: 2026-05-08
content_hash: 89e6ff4e6e0e51e2
---

# Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis

**Conference**: ICLR 2026
**arXiv**: [2601.21709](https://arxiv.org/abs/2601.21709)
**Code**: [GitHub](https://github.com/MIRALab-USTC/LLM-TAPPA)
**Area**: Model Compression / Attention Mechanism Analysis / LLM Inference Acceleration
**Keywords**: attention patterns, temporal analysis, RoPE, query self-similarity, KV cache compression, LLM pruning

## TL;DR

This paper proposes the TAPPA framework, which explains the formation mechanisms of various attention patterns in LLMs (attention sink, diagonal, periodic, etc.) from a temporal continuity perspective in a unified manner, and leverages query self-similarity (q-similarity) as a metric to guide KV cache compression and model pruning tasks.

## Background & Motivation

Attention heads in LLMs exhibit diverse structured patterns:
- **Attention sinks**: The first token receives anomalously high attention.
- **Diagonal patterns**: Attention is focused on neighboring tokens.
- **Retrieval heads**: Global scanning of the context.
- **Periodic patterns**: Attention recurs at regular intervals.

Prior work typically analyzes individual patterns in isolation, lacking a unified explanation. The core question is: **given the same attention formulation, what factors determine which attention pattern a given head adopts?**

## Method

### Overall Architecture: TAPPA

TAPPA (Temporal Attention Pattern Predictability Analysis) categorizes attention patterns into two classes:
- **Predictable patterns**: Exhibit temporal continuity, with attention metrics evolving smoothly across decoding steps.
- **Unpredictable patterns**: Exhibit irregular jumps and lack temporal consistency (e.g., retrieval heads).

The key discriminating factor is **query self-similarity (q-similarity)**.

### Key Design 1: Predictable vs. Unpredictable Patterns

**Proposition 4.1**: If the difference between consecutive queries $\|q_{t+1} - q_t\|$ is large and not orthogonal to the rotated keys, the difference in attention logits must also be large:

$$\|a_{t+1} - a_t\|_\infty \geq c_1 \|q_{t+1} - q_t\| - c_2$$

That is, low q-similarity leads to random patterns, while high q-similarity is a necessary condition for predictable patterns.

### Key Design 2: Re-access Patterns (Attention Sink)

**Theorem 5.1** (Vertical Stability of Attention): When queries are highly self-similar and a dominant low-frequency RoPE channel exists, the attention logits are vertically stable over time:

$$|a_{t+1,i} - a_{t,i}| \leq \text{small quantity}$$

When the angle $\phi_{t,i}^{(m)}$ between $q$ and $k_i$ is small, the cosine term approaches 1, which explains the attention sink phenomenon.

### Key Design 3: Sequential Patterns (Diagonal)

**Theorem 5.2**: When both queries and keys exhibit high self-similarity ($\|q_{t+1} - q_t\| \leq \varepsilon$, $\|k_{i+1} - k_i\| \leq \varepsilon$):

$$|a_{t+1,i+1} - a_{t,i}| \leq C\varepsilon$$

RoPE's relative positional encoding preserves query–key interactions under synchronized position shifts, giving rise to diagonal patterns.

### Key Design 4: Periodic Sequential Patterns

**Theorem 5.3**: When a dominant RoPE channel $m^\star$ exists, the diagonal spacing is:

$$T = \frac{2\pi}{\theta_{m^\star}} = 2\pi c^{2m^\star/d}$$

This is verified experimentally: relocating the dominant channel to a low index (high-frequency) position induces the theoretically predicted periodic diagonals, and adjusting the RoPE base $c$ controls the spacing accordingly.

### Key Design 5: Seasonal Patterns

**Theorem 5.4**: When queries and keys are approximately periodic with period $L$ and resonate with the dominant RoPE frequency:

$$|a_{t+L,i} - a_{t,i}| \leq C_1(\varepsilon_q + \varepsilon_k) + C_2\delta$$

This produces seasonal attention patterns with period $L$.

### Downstream Applications

Q-similarity is used as a simple metric to guide:
- **KV cache compression**: Heads with high q-similarity can be safely compressed.
- **LLM pruning**: Identifying redundant heads suitable for pruning.

## Key Experimental Results

### KV Cache Compression (LongBench)

| Method | Budget=512 | Avg. Score |
|--------|-----------|------------|
| StreamingLLM | — | 41.75 |
| H2O | — | 44.39 |
| SnapKV | — | 46.92 |
| CAKE | — | 47.19 |
| **TAPPA** | — | **47.55** |
| Full cache | — | 49.06 |

TAPPA's simple q-similarity-based metric consistently outperforms all baseline methods.

### LLM Pruning

On Llama-3.1-8B and Qwen-2.5-7B:
- Q-similarity-guided pruning outperforms uniform pruning without guidance.
- Pruning heads with high q-similarity yields smaller performance degradation.

### Theoretical Validation Experiments

1. **Dominant channel relocation**: Relocating the dominant channel at index 124 in Qwen2.5 to indices 2/3/5 successfully induces the theoretically predicted periodic diagonal patterns.
2. **RoPE base adjustment**: Reducing $c$ from $1{,}000{,}000$ to $100{,}000$ shortens the diagonal spacing, consistent with the theoretical prediction $T = 2\pi / \theta_{m^\star}$.
3. **Q-similarity distribution**: Analysis across layers, heads, models, and datasets confirms the ubiquitous presence of both high- and low-continuity heads.

### Key Findings

1. Q-similarity is the key factor distinguishing predictable from unpredictable attention patterns.
2. Re-access patterns require high q-similarity combined with a dominant low-frequency RoPE channel.
3. Sequential patterns require high q-similarity combined with high k-similarity.
4. The spacing of periodic diagonals is determined by the frequency of the dominant RoPE channel.
5. Q-similarity serves as a simple yet consistently effective metric for downstream tasks.

## Highlights & Insights

- First work to provide a unified explanation of diverse attention patterns from a temporal continuity perspective.
- Four theorems offer rigorous mathematical analysis.
- The q-similarity metric is extremely simple yet consistently effective.
- Controlled experiments (channel relocation and RoPE base adjustment) precisely validate the theoretical claims.

## Limitations & Future Work

- The theoretical analysis assumes that query/key self-similarity is measurable, whereas in practice these quantities vary with context.
- Analysis of unpredictable patterns (e.g., retrieval heads) remains relatively limited.
- Seasonal patterns require RoPE resonance conditions, which may have limited applicability in practice.
- Downstream task improvements, while consistent, are modest in magnitude (~0.5–1 point).

## Related Work & Insights

- **Attention Patterns**: Attention sinks by Xiao et al. (2023); retrieval heads by Wu et al. (2024).
- **RoPE Analysis**: Barbero et al. (2025) attribute diagonal patterns to high-frequency RoPE components.
- **KV Cache Compression**: H2O, SnapKV, PyramidKV, MInference.
- **Input Dynamics**: AttentionPredictor (Yang et al., 2025); Lee et al. (2024).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The unified theoretical framework is a significant contribution.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Four theorems with rigorous derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Theoretical validation is impressive; downstream tasks are somewhat limited.
- Value: ⭐⭐⭐⭐ — Q-similarity is simple and practical.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and elegant, with outstanding visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FASA: Frequency-aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)

</div>

<!-- RELATED:END -->
