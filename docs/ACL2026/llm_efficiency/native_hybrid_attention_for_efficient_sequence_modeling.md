---
title: >-
  [Paper Note] Native Hybrid Attention for Efficient Sequence Modeling
description: >-
  [ACL 2026][LLM Efficiency][Hybrid Attention] This paper proposes Native Hybrid Attention (NHA), which unifies the long-term memory slots of linear RNNs with short-term precise tokens from sliding windows through a single…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Hybrid Attention"
  - "Linear Attention"
  - "Sliding Window"
  - "Long-Short Term Memory Fusion"
  - "Efficient Sequence Modeling"
date: 2026-05-08
content_hash: db97bb7fa70dfb28
---

# Native Hybrid Attention for Efficient Sequence Modeling

**Conference**: ACL 2026  
**arXiv**: [2510.07019](https://arxiv.org/abs/2510.07019)  
**Code**: [GitHub](https://github.com/JusenD/NHA)  
**Area**: LLM Efficiency / Attention Mechanism  
**Keywords**: Hybrid Attention, Linear Attention, Sliding Window, Long-Short Term Memory Fusion, Efficient Sequence Modeling

## TL;DR

This paper proposes Native Hybrid Attention (NHA), which unifies the long-term memory slots of linear RNNs with short-term precise tokens from sliding windows through a single unified softmax attention. It achieves native unification both within and across layers—dynamically allocating long- and short-term attention weights without additional fusion parameters. NHA outperforms Transformer and other hybrid baselines on recall-intensive and common-sense reasoning tasks.

## Background & Motivation

**Background**: The $O(n^2)$ complexity of the Transformer self-attention mechanism limits long-sequence processing. The research community has followed two paths: (1) Sparse attention (e.g., Sliding Window Attention, SWA) calculates softmax within local windows; (2) Linear sequence models (e.g., Mamba, GLA, GSA) compress the full sequence into fixed-size states to achieve $O(n)$ efficiency.

**Limitations of Prior Work**: (1) SWA cannot capture tokens outside the window, while the extreme compression of linear models often loses precise token information—the strengths and weaknesses of both are complementary; (2) Existing intra-layer hybrid schemes (e.g., MesaNet, Titans) compute linear attention and local softmax separately and then merge them via weighted summation—requiring extra fusion parameters and fixed weights; (3) Existing inter-layer hybrid schemes (e.g., Jamba) stack different types of layers—requiring the management of heterogeneous modules and alignment, with layer type selection necessitating expensive searches.

**Key Challenge**: Pure linear models cannot perfectly preserve infinite information in a fixed-size memory (theoretically impossible), but maintaining a full KV cache at every layer for every token like a Transformer is too expensive and unnecessary. A better balance must be found between information retention and computational efficiency.

**Goal**: Design a natively unified hybrid attention mechanism that simultaneously achieves: (1) Intra-layer fusion—dynamically allocating long- and short-term attention without extra parameters; (2) Inter-layer hybridity—flexible configuration attained solely by adjusting the window size hyperparameter.

**Key Insight**: Representing the memory slots of a linear RNN in an $m \times d$ KV format (consistent with the SWA KV cache format) allows the two to be directly concatenated and processed by a unified softmax. Softmax itself can learn to dynamically allocate attention weights.

**Core Idea**: Long-term memory (RNN compression) and short-term memory (precise sliding window tokens) are naturally compatible in the KV dimension. By concatenating them and processing them with a single softmax, context-dependent fusion is achieved with zero additional parameters.

## Method

### Overall Architecture

NHA maintains two types of memory in each layer: (1) Long-term memory $K^{long}_t, V^{long}_t \in \mathbb{R}^{m \times d}$—updated recursively via a gated RNN, compressing all history outside the window; (2) Short-term memory $K^{short}_t, V^{short}_t \in \mathbb{R}^{w \times d}$—the precise token KV cache within the window. These are concatenated into $K^H_t \in \mathbb{R}^{(m+w) \times d}$, and the output is produced via a single softmax attention step. Inter-layer hybridity is achieved by adjusting the window size $w$: $w=0$ degenerates into a pure linear RNN, while $w=N$ reverts to full attention.

### Key Designs

1.  **Intra-layer Fusion — Unified Softmax Fusion**:
    - **Function**: Dynamically allocates attention weights between long-term and short-term memory without extra parameters.
    - **Mechanism**: Long-term memory is updated via a gated linear RNN $K^{long}_t = \text{Diag}(\alpha_t) K^{long}_{t-1} + (1-\alpha_t) \otimes k_t$, concatenated with the short-term window KV cache, and fed into the softmax: $o_t = \text{softmax}(\frac{q_t (K^H_t)^T}{\sqrt{d}}) V^H_t$. The softmax automatically calculates the attention ratio for long-term memory $\omega_L = \frac{\sum_{i \in long} \exp(q_t k_i^\intercal)}{\sum_{i \in long} \exp(q_t k_i^\intercal) + \sum_{j \in short} \exp(q_t k_j^\intercal)}$.
    - **Design Motivation**: Unlike weighted-sum fusion, the weight of the unified softmax depends on the similarity between the query and all keys—enabling per-token, per-head context-dependent weighting, where gradients naturally couple the learning of long- and short-term memory. Token shifting ensures only tokens outside the window update the long-term memory; RoPE positional encoding is used within the window, while long-term memory does not include positional encoding.

2.  **Inter-layer Mixing — Window Size Adjustment**:
    - **Function**: Enables flexible inter-layer hybrid configurations under a unified architecture.
    - **Mechanism**: All NHA layers share the same architectural design, with behavior controlled solely by adjusting the sliding window size $w$ of each layer—$w=0$ as a pure linear RNN layer, $w=N$ as a full attention layer, and $0 < w < N$ as a hybrid layer. This "duality" allows the same model to switch between different accuracy-speed configurations during inference without retraining.
    - **Design Motivation**: Previous inter-layer hybrids (e.g., Jamba) stacking heterogeneous layers required managing alignment across different modules, and searching for the optimal configuration was costly. NHA allows for zero-cost searches of optimal configurations at inference time by tuning window sizes.

3.  **Chunkwise Parallel Computing**:
    - **Function**: Efficient GPU implementation.
    - **Mechanism**: The sequence is divided into chunks of size $C$. Within each block, linear channel logits (via cumulative/reverse gated products $\mathcal{A}$) and sliding window logits (standard attention with offset windows) are computed in parallel, concatenated for a unified softmax, and finally aggregated from the linear memory and sliding window branches. Implemented using Triton kernels.
    - **Design Motivation**: Maintains near-linear computational complexity while fully utilizing GPU parallelism—NHA speed is comparable to GSA on long sequences, significantly outperforming the quadratic growth of FlashAttention.

### Loss & Training

Standard language modeling cross-entropy loss. The 340M model was trained on 15B tokens, and the 1.3B model on 100B tokens. Hybridization of pretrained LLMs utilized fine-tuning on 10B tokens from SlimPajama.

## Key Experimental Results

### Main Results

**1.3B Model Performance Comparison (100B tokens)**

| Model | Common Sense Reasoning Avg↑ | Recall Intensive Avg↑ | Wiki ppl↓ |
|------|-------------|-------------|----------|
| Trans++ | 50.71 | 37.31 | 17.61 |
| GSA | 51.79 | 32.05 | 16.69 |
| GSA-H (+Transformer layers) | 50.76 | 44.99 | 16.22 |
| GDN-H | 52.54 | 44.88 | 16.02 |
| **NHA** | **52.89** | **46.43** | **16.16** |

### Pretrained LLM Hybridization

| Model | Full Attention Layers | Common Sense Reasoning Avg↑ | Recall Intensive Avg↑ |
|------|-----------|-------------|-------------|
| Llama-3-8B | 32 | 71.30 | 60.08 |
| NHA-Llama-3-8B | 4 | 70.31 | 57.64 |
| Zamba2-7B | 9 | 71.50 | 54.56 |
| StripedHyena-7B | 16 | 68.10 | 57.59 |

### Key Findings

- NHA achieves optimal performance in both common-sense reasoning and recall-intensive tasks at the 1.3B scale, surpassing all pure linear and hybrid baselines.
- Pretrained LLM Hybridization: NHA-Llama-3-8B, with only 4 full attention layers and 10B tokens of fine-tuning, reached 57.64 on recall-intensive tasks, exceeding StripedHyena (57.59) which uses 16 full attention layers.
- In RULER long-context evaluation, NHA demonstrates the strongest extrapolation capability—when extrapolating from a 2K training length to 8K, it achieved 24.8 on the Hotpot task, far exceeding other hybrid models.
- Inference-time Architecture Search: By inserting a global window at Layer 11, an NHA with 4 full attention layers can match the performance of a 12-layer baseline—optimizing the position of layers is more important than the quantity.
- When NHA collapses into a pure Transformer, its performance surprisingly exceeds that of a Transformer trained from scratch—indicating that hybrid training has a regularization effect.

## Highlights & Insights

- Unified softmax fusion is the core innovation—downgrading fusion from explicit parameter learning to the implicit allocation of softmax simplifies the design while enhancing contextual adaptability. Gradient analysis proves that unified softmax naturally couples the gradient flow of long- and short-term memory.
- The "architectural duality" of NHA is highly practical—the same model can switch between different efficiency-accuracy configurations at inference time with zero cost, suitable for heterogeneous deployment scenarios.
- The discovery that "optimizing the position of full attention layers is more important than their quantity" provides direct guidance for hybrid architecture design.

## Limitations & Future Work

- During pretrained LLM hybridization, restricted by the 10B token fine-tuning budget and 2K training context, there was some performance degradation on knowledge-intensive benchmarks like MMLU.
- The choice of the number of long-term memory slots $m$ affects performance; currently fixed at 32/64, with adaptive slot counts yet to be explored.
- The Triton kernel implementation currently only supports training; the RNN mode kernel for inference requires further optimization.
- The effectiveness has not been verified in ultra-long context scenarios of 128K+.

## Related Work & Insights

- **vs Titans/MesaNet**: These intra-layer hybrid schemes compute two types of attention separately and then perform weighted fusion; NHA achieves zero-parameter fusion with unified softmax—which is simpler and context-adaptive.
- **vs Jamba/StripedHyena**: These inter-layer hybrid schemes stack heterogeneous layers; NHA uses a unified architecture + window size adjustment—supporting zero-cost search at inference time.
- **vs Atlas**: The window range of Atlas is equivalent to the sliding window of NHA, but Atlas's joint KV update cannot incorporate the softmax operation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Unified softmax fusion + architectural duality is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Training from scratch + LLM hybridization + RULER long-context + inference-time search + ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear explanation of the progressive three-layer architecture design with rigorous mathematical formalization.
- Value: ⭐⭐⭐⭐⭐ Provides a unified and practical hybrid solution for efficient LLM architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)
- [\[ICML 2026\] Proxy Compression for Language Modeling](../../ICML2026/llm_efficiency/proxy_compression_for_language_modeling.md)
- [\[ACL 2026\] Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention](threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->
