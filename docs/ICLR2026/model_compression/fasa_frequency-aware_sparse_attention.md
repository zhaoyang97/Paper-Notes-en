---
title: >-
  [Paper Note] FASA: Frequency-aware Sparse Attention
description: >-
  [ICLR2026][Model Compression][KV Cache Compression] This paper identifies a functional sparsity in RoPE attention at the frequency chunk (FC) level—fewer than 1% of "dominant FCs" suffice to approximate the token selecti…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "KV Cache Compression"
  - "Sparse Attention"
  - "RoPE"
  - "Frequency Chunks"
  - "Long-context Inference"
date: 2026-05-08
content_hash: 2d126b0701736dfa
---

# FASA: Frequency-aware Sparse Attention

**Conference**: ICLR2026  
**arXiv**: [2602.03152](https://arxiv.org/abs/2602.03152)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: KV Cache Compression, Sparse Attention, RoPE, Frequency Chunks, Long-context Inference  

## TL;DR
This paper identifies a functional sparsity in RoPE attention at the frequency chunk (FC) level—fewer than 1% of "dominant FCs" suffice to approximate the token selection behavior of full attention heads. Building on this finding, the authors propose FASA, a training-free framework that employs a two-stage strategy (dominant FCs predict token importance → full attention is computed only over important tokens), achieving 8× memory compression and 2.6× inference speedup with negligible quality loss.

## Background & Motivation
- The core bottleneck of long-context LLM inference is that the KV cache grows linearly with sequence length, incurring enormous memory and computational overhead. At 32K context, the decoding phase accounts for 90% of total latency.
- Existing sparse attention methods (StreamingLLM, H2O, SnapKV, etc.) typically discard KV pairs based on token-level importance scores, yet evaluating token importance itself requires computing full attention—creating a chicken-and-egg problem.
- RoPE (Rotary Position Embedding) decomposes attention into a sum over multiple frequency chunks (FCs), each corresponding to a distinct rotation frequency $\theta_i = B^{-2(i-1)/d}$.
- **Key finding**: The vast majority of FCs contribute negligibly to the final attention (CA < 0.05), while a small subset of "dominant FCs" (< 1% of all FCs) is sufficient to approximate the token selection behavior of full attention—a structured sparsity that has not been previously identified or exploited.

## Method

### Overall Architecture
Inference proceeds in two stages: (1) **Token Importance Predictor (TIP)**: a pre-calibrated subset of dominant FCs is used to efficiently estimate per-token importance scores, selecting the top-$N_{fac}$ critical tokens; (2) **Focused Attention Computation (FAC)**: full-dimensional attention is computed exclusively over the selected critical tokens to generate the next token. Dominant FC identification is a one-time offline process that generalizes across tasks.

### Key Designs
1. **Frequency Chunk (FC) Decomposition**:

    - RoPE attention $\mathbf{A}_{t_1,t_2} = \mathbf{q}_{t_1}\mathbf{R}_{\Delta t}\mathbf{k}_{t_2}^T$ can be exactly decomposed into a sum of $d/2$ frequency chunks.
    - Each FC is a 2D subspace associated with a distinct rotation frequency $\theta_i = B^{-2(i-1)/d}$.
    - Low-dimensional FCs correspond to high-frequency rotations (primarily encoding positional information), while high-dimensional FCs correspond to low-frequency rotations (primarily encoding semantic information).

2. **Contextual Agreement (CA) Metric**:

    - Defined as the normalized intersection between the top-K token set of a single FC and that of the full attention head.
    - A small number of "dominant FCs" (< 1% of all FCs) exhibit CA values substantially higher than the remaining FCs (> 0.15 vs. < 0.05).
    - Three key properties of dominant FCs: **sparsity** (1–3 FCs suffice), **cross-model generality** (holds for LLaMA, Mistral, and Qwen), and **cross-task invariance** (overlap rate of dominant FCs across different calibration datasets exceeds 70%).

3. **FASA-M (Memory-optimized Variant)**:

    - Offloads value cache and non-dominant key cache to CPU memory.
    - Retains only dominant FC keys on GPU for TIP.
    - Achieves 8× KV cache compression.

4. **FASA-C (Compute-optimized Variant)**:

    - Retains the full cache on GPU.
    - During TIP, only the dominant FC key subset is accessed (sparse memory access).
    - Achieves 2.6× inference speedup.

### Loss & Training
FASA is entirely training-free. Dominant FCs are identified once through offline calibration (computing CA scores on a small number of samples) and are directly applicable to all downstream tasks without modifying model weights.

## Key Experimental Results

### Main Results

| Task | Metric | FASA | Full-KV | SnapKV | H2O | Stream |
|------|--------|------|---------|--------|-----|--------|
| LongBench-V1 | Performance recovery | ~100% | 100% | ~85% | ~75% | ~70% |
| AIME24 | Speedup | 2.56× | 1.0× | - | - | - |
| Language modeling | PPL | Near Full-KV | Baseline | Slightly higher | Notably higher | Highest |

### Ablation Study (Compound CA scores, K=256)

| Dominant FC count F | CA Score | vs. SnapKV | Note |
|--------------------|----------|------------|------|
| F=8 (1/8) | 49.4% | +8.5% | Only 1/8 of dimensions |
| F=12 | 54.7% | +13.8% | Sweet-spot configuration |
| F=16 (1/4) | 59.7% | +18.8% | 25% of FCs |
| Random | 3.6% | −37.3% | Random FC selection is ineffective |

### Key Findings
- FASA recovers approximately 100% of Full-KV performance while retaining only 256 KV cache tokens.
- Dominant FCs account for less than 1% of all FCs, yet their CA scores substantially exceed those of baselines such as SnapKV.
- FASA is orthogonally compatible with layer-wise budget allocation methods such as PyramidKV, enabling further performance gains.
- FASA's advantage is particularly pronounced on LongCoT (long chain-of-thought) tasks, where conventional methods collapse due to eviction of intermediate reasoning tokens.

## Highlights & Insights
- This is the first work to analyze the sparsity of RoPE attention from a frequency-domain perspective, revealing an elegant structural prior: high-frequency FCs encode positional information while low-frequency FCs encode semantic information.
- The paper presents a complete logical chain from discovering sparsity → quantifying sparsity → exploiting sparsity, with the CA metric being concise and effective.
- The training-free, plug-and-play design substantially lowers the barrier to practical deployment—no modification of model weights and no additional training are required.
- FASA-M and FASA-C address memory and compute bottlenecks respectively, forming a complementary pair of solutions.
- Orthogonal compatibility with layer-wise budget allocation schemes such as PyramidKV enables compositional use.
- The advantage on LongCoT inference tasks is especially notable, as conventional token eviction methods collapse by discarding intermediate reasoning tokens.

## Limitations & Future Work
- The current selection of dominant FCs follows a static strategy at the layer/head granularity; dynamic adaptive selection (e.g., conditioned on the characteristics of the current query) may yield further improvements.
- Validation is limited to decoder-only architectures (LLaMA, Mistral, Qwen); applicability to encoder-decoder architectures (e.g., T5) remains unexplored.
- Integration with system-level optimizations such as FlashAttention warrants further investigation, as FASA's sparse access patterns may conflict with FlashAttention's tiling strategy.
- The stability of dominant FCs under extremely long contexts (> 256K) requires further verification.
- In multi-turn dialogue settings, the token importance distribution may shift significantly across turns; whether one-time calibration suffices remains an open question.
- The interaction with Grouped Query Attention (GQA) has not been tested—key sharing in GQA may affect FC sparsity.

## Related Work & Insights
- **vs. H2O / SnapKV / StreamingLLM**: These methods require computing full attention first to select tokens (chicken-and-egg problem). FASA circumvents this by using dominant FCs—less than 25% of dimensions—to accurately predict token importance.
- **vs. SparQ / LoKi**: SparQ selects key dimensions based on query magnitude (head-agnostic); LoKi employs PCA projection (requiring stored projection matrices). FASA directly exploits the intrinsic structure of RoPE with zero additional memory overhead.
- **vs. YaRN / NTK-aware scaling**: These methods extend context length from a frequency perspective; FASA performs sparse attention from a frequency perspective. The two approaches offer complementary understandings of RoPE.
- **Inspiration**: FC-level sparsity may extend beyond inference acceleration to attention visualization (identifying which FCs encode semantics vs. position) and model compression (pruning parameters of non-dominant FCs).

## Rating
- Novelty: ⭐⭐⭐⭐ Frequency chunk sparsity is a genuinely novel perspective; the CA metric is elegantly defined.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-benchmark evaluation (LongBench, AIME, language modeling) with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from observation → hypothesis → validation → method → experiments is coherent and well-structured.
- Value: ⭐⭐⭐⭐⭐ Training-free and plug-and-play; highly practical with direct impact on KV cache optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](../../NeurIPS2025/model_compression/specattn_speculating_sparse_attention.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [\[ICLR 2026\] Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis](why_attention_patterns_exist_a_unifying_temporal_perspective_analysis.md)

</div>

<!-- RELATED:END -->
