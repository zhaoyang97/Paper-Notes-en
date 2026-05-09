---
title: >-
  [Paper Note] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels
description: >-
  [NeurIPS 2025][LLM Efficiency][Tiled Flash Linear Attention] This paper proposes TFLA (Tiled Flash Linear Attention), which achieves efficient linear RNN/mLSTM kernels through two-level sequence parallelism and tiling optimization, delivering significant wall-clock speedups over FlashAttention 3 and Mamba 2 (>2× in training vs. Mamba 2) while maintaining equivalent model accuracy.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Tiled Flash Linear Attention
  - mLSTM
  - xLSTM
  - Linear RNN
  - CUDA Kernel
  - Chunked Parallelism
date: 2026-05-08
content_hash: 6070190080794979
---

# Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels

**Conference**: NeurIPS 2025
**arXiv**: [2503.14376](https://arxiv.org/abs/2503.14376)
**Code**: [github](https://github.com/NX-AI/mlstm_kernels)
**Area**: LLM Efficiency / Efficient Attention Mechanisms
**Keywords**: Tiled Flash Linear Attention, mLSTM, xLSTM, Linear RNN, CUDA Kernel, Chunked Parallelism

## TL;DR
This paper proposes TFLA (Tiled Flash Linear Attention), which achieves efficient linear RNN/mLSTM kernels through two-level sequence parallelism and tiling optimization, delivering significant wall-clock speedups over FlashAttention 3 and Mamba 2 (>2× in training vs. Mamba 2) while maintaining equivalent model accuracy.

## Background & Motivation

**Background**: Linear RNNs and linear attention mechanisms (RetNet, Mamba, GLA, mLSTM) offer $O(T)$ computational complexity, theoretically superior to the $O(T^2)$ cost of Transformers. The FlashAttention series has established IO-aware algorithms as the standard for attention implementation.

**Limitations of Prior Work**:
- Despite their theoretical efficiency, linear RNNs lack well-optimized CUDA kernels, making practical speed advantages difficult to realize.
- Flash Linear Attention (FLA) is constrained by GPU SRAM capacity to a maximum chunk size of $L=64$, forcing large numbers of intermediate states to be materialized to HBM.
- Small chunk sizes lead to low arithmetic intensity, high memory IO cost, and underutilization of GPU compute.

**Key Challenge**: The chunk-size dilemma — small chunks cause IO bottlenecks, while large chunks exceed SRAM capacity.

**Goal**: Break the SRAM constraint on chunk size to enable efficient linear RNN kernels with arbitrarily large chunks.

**Key Insight**: Building upon FLA's single-level sequence parallelism (inter-chunk), this work introduces a second level of sequence parallelism (intra-chunk tiling), analogous to the tiling strategy in FlashAttention 2.

**Core Idea**: TFLA = Flash Linear Attention (inter-chunk parallelism) + FlashAttention 2 tiling (intra-chunk parallelism).

## Method

### Overall Architecture
TFLA comprises two core kernels:
- **Recurrent Kernel**: Computes intermediate states $C_k$ chunk by chunk, handling inter-chunk dependencies.
- **Parallel Kernel**: Computes intra-chunk outputs $H^{(k)}$ across all chunks in parallel, supporting two-level parallelism.

### Key Designs

1. **Inter-chunk Recurrence**:

   - Function: Recursively computes chunk-level memory states $C_k = \bar{g}_k C_{k-1} + (\bar{a}_k \odot K^{(k)})^T V^{(k)}$.
   - Mechanism: Compresses the $T$-step recurrence into $N_c = \lceil T/L \rceil$ chunk-level recurrences, substantially reducing the number of intermediate states that must be materialized.
   - Design Motivation: Reduces intermediate states from $O(T)$ to $O(T/L)$; larger $L$ yields greater savings.

2. **Intra-chunk Parallelism and Tiling**:

   - Function: Fuses and tiles matrix multiplications within each chunk, supporting arbitrarily large chunk sizes.
   - Mechanism: Decomposes the intra-chunk output as $H^{(k)} = (\tilde{Q}^{(k)} K^{(k)T} V^{(k)}) + (Q^{(k)} C_{k-1})$, introducing block sizes $B_{L_{hq}}$ (sequence dimension) and $B_{d_{hv}}$ (embedding dimension) to parallelize over these two dimensions while looping over $L_{kv}$ and $d_{qk}$.
   - Design Motivation: Large chunk sizes can exceed SRAM limits (FLA's $L=64$ → TFLA's $L=256+$), with tiling automatically decomposing operations into SRAM-friendly units. Arithmetic intensity increases with chunk size.

3. **mLSTMsig: Sigmoid Input Gate Variant**:

   - Function: Replaces the exponential input gate in mLSTM with a sigmoid: $C_t = \sigma(\tilde{f}_t) C_{t-1} + \sigma(\tilde{i}_t) k_t v_t^T$.
   - Mechanism: The sigmoid gate naturally enforces upper and lower bounds, eliminating the need to track max states and normalization states.
   - Design Motivation: Simplifies kernel implementation by removing rescaling logic, achieving >30% faster forward passes. Transfer behavior analysis demonstrates equivalence with mLSTMexp.

### Loss & Training
- **Input Gate Bias Initialization**: The bias is initialized to $-10$ (rather than the default $0$) to keep pre-activations negative in early training, reducing gradient norm spikes and improving training stability.
- **Information Gating via RMS Norm**: The normalization layer is found to play a role not only in stabilization but also in information routing — when $\|C_t^T q\|^2$ approaches epsilon, the output is suppressed toward zero.

## Key Experimental Results

### Main Results: Language Modeling Performance

| Model Size | Llama2 | mLSTMexp (FLA) | mLSTMexp (TFLA) | mLSTMsig (TFLA) |
|-----------|--------|----------------|-----------------|-----------------|
| 160M | 21.03 | 21.18 | 21.03 | 21.06 |
| 400M | 16.66 | 16.66 | 16.60 | 16.61 |
| 1.4B | 13.31 | 13.35 | 13.20 | 13.22 |

### Kernel Benchmarks (H100 GPU, 65536 tokens)

| Comparison | Inference Speedup | Training Speedup |
|-----------|-------------------|------------------|
| TFLA mLSTMsig vs FLA limit_chunk | ~25% faster | ~20% faster |
| TFLA mLSTMsig vs Mamba 2 | ~1.5–2× faster | >2× faster |
| TFLA mLSTMsig vs FlashAttention 3 | Competitive | Faster on long sequences |

### Ablation Study: Chunk Size

| Chunk Size $L$ | Memory (GB) | Training Time (s) |
|---------------|-------------|-------------------|
| 64 | ~14 | ~1.8 |
| 128 | ~10 | ~1.7 |
| 256 | **~8** | **~1.65** |
| 512 | ~7 | ~1.75 |

### Key Findings
- **$L=256$ is the optimal chunk size**, balancing memory usage and speed.
- **TFLA kernels produce numerically equivalent results to FLA kernels** (loss difference <0.01).
- **mLSTMsig achieves comparable perplexity to mLSTMexp** while being ~30% faster in kernels.
- TFLA consistently outperforms Mamba 2 in training (>2×) and surpasses FlashAttention 3 on long sequences.

## Highlights & Insights
- **Two-level sequence parallelism as a general method**: The approach extends naturally beyond mLSTM to RetNet, DeltaNet, Mamba 2, and other linear RNNs (with derivations provided in the appendix), demonstrating strong generalizability.
- **Engineering elegance of sigmoid substitution**: By exploiting the mathematical property that sigmoid approximates exp in the negative regime, the method eliminates complex max-state tracking for a nearly free ~30% speedup.
- **Arithmetic-intensity-driven design philosophy**: Grounded in the Roofline model, increasing chunk size raises the FLOP/byte ratio to overcome the GPU memory wall, providing quantitative guidance for future kernel design.
- **The information-gating role of normalization layers** reveals that RMS norm in gated RNNs serves not merely as a stabilizer, offering broader insights into other gated architectures.

## Limitations & Future Work
- **Limited scale validation**: The largest model evaluated is 1.4B; performance on 7B+ LLMs remains unknown.
- **Restricted context length**: Experiments use sequences up to 8192 tokens; scenarios exceeding 100K tokens are not validated.
- **Empirical evaluation limited to mLSTM**: Despite theoretical extensibility to RetNet/GLA, empirical evidence is lacking.
- **Future directions**: (1) Further acceleration via Hopper GPU features (TMA/FP8); (2) Validation on 7B+ models; (3) Automated chunk size selection.

## Related Work & Insights
- **vs. FlashAttention 3**: FlashAttention remains $O(T^2)$; TFLA achieves $O(T)$ via linear RNNs and is faster on long sequences.
- **vs. Flash Linear Attention (FLA)**: FLA is limited to a fixed chunk size of $L=64$; TFLA extends this to $L=256+$, yielding a 25% speedup.
- **vs. Mamba 2**: TFLA trains >2× faster on the same GPU, and mLSTM's matrix memory state is more expressive than Mamba's diagonal state.

## Rating
- Novelty: ⭐⭐⭐⭐ — Two-level sequence parallelism is a direct and elegant idea, representing a natural generalization of FlashAttention tiling to linear RNNs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-scale evaluation, multiple baselines, and thorough ablations; large-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Intuitive figures, rigorous mathematics, and comprehensive appendix.
- Value: ⭐⭐⭐⭐ — High practical value; kernels outperforming FlashAttention 3 and Mamba 2 are immediately deployable in production.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](../../ICLR2026/llm_efficiency/xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[NeurIPS 2025\] Scale-invariant Attention](scale-invariant_attention.md)

<!-- RELATED:END -->
