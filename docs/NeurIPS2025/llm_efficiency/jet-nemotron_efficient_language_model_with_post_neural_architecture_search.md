---
title: >-
  [Paper Note] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
description: >-
  [NeurIPS 2025][LLM Efficiency][hybrid attention] NVIDIA proposes the PostNAS pipeline — starting from a pretrained full-attention model, freezing MLP weights, and applying a four-step search (full-attention layer placement → linear attention block selection → novel JetBlock design → hardware-aware hyperparameter search) to yield the hybrid Jet-Nemotron architecture. The 2B model surpasses Qwen3-1.7B on MMLU-Pro while achieving 47× higher generation throughput.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - hybrid attention
  - linear attention
  - neural architecture search
  - efficient LLM
  - KV cache
date: 2026-05-08
content_hash: 3136ad1c4df4271c
---

# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

**Conference**: NeurIPS 2025
**arXiv**: [2508.15884](https://arxiv.org/abs/2508.15884)
**Code**: [https://github.com/NVlabs/Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron)
**Area**: LLM Efficiency
**Keywords**: hybrid attention, linear attention, neural architecture search, efficient LLM, KV cache

## TL;DR
NVIDIA proposes the PostNAS pipeline — starting from a pretrained full-attention model, freezing MLP weights, and applying a four-step search (full-attention layer placement → linear attention block selection → novel JetBlock design → hardware-aware hyperparameter search) to yield the hybrid Jet-Nemotron architecture. The 2B model surpasses Qwen3-1.7B on MMLU-Pro while achieving 47× higher generation throughput.

## Background & Motivation

**Background**: LLM inference efficiency is a critical deployment bottleneck. Full-attention's $O(n^2)$ complexity causes severe KV cache bloat in long-context generation. A large body of work has explored linear attention ($O(n)$) designs (Mamba2, RWKV7, GLA, etc.) as well as hybrid architectures that retain a small number of full-attention layers alongside linear attention layers.

**Limitations of Prior Work**: Existing efficient models trained from scratch incur high costs and exhibit noticeable accuracy gaps relative to full-attention SOTAs — particularly on MMLU-Pro, mathematical reasoning, and retrieval tasks. From-scratch training also carries high architectural design risk and long development cycles.

**Key Challenge**: Architecture exploration requires pretraining for validation, yet pretraining is prohibitively expensive, making LLM architectural innovation inaccessible to most academic groups and small organizations.

**Key Insight**: Rather than training from scratch, the paper starts from an existing full-attention model, freezes the MLP weights (which encode the bulk of learned knowledge), and explores only the attention block design. This substantially reduces cost while remaining competitive with SOTA.

**Core Idea**: Post Neural Architecture Search (PostNAS) — a coarse-to-fine four-step architecture search pipeline that inherits MLP knowledge from a pretrained full-attention model and systematically searches for the optimal attention block configuration.

## Method

### Overall Architecture
PostNAS begins from a pretrained full-attention model (e.g., Qwen2.5-1.5B), freezes all MLP weights, and progressively refines the attention block design through four steps: (1) learning optimal full-attention layer placement → (2) selecting the best linear attention block → (3) designing the novel JetBlock → (4) hardware-aware hyperparameter search. This produces the Jet-Nemotron model family.

### Key Designs

1. **Full-Attention Layer Placement Learning**:

    - **Function**: Determines which layers retain full attention and which are replaced by linear attention.
    - **Mechanism**: Constructs a Once-for-All supernet in which each layer has both a full-attention path and a linear attention path. During training, sub-networks are randomly sampled and optimized via a feature distillation loss. After training, beam search is used to find the optimal placement under a given constraint (e.g., retaining only 2 full-attention layers).
    - **Design Motivation**: Different tasks place full-attention requirements at different layers (retrieval tasks require full attention at layers 15/20; MMLU benefits from sliding-window attention). Learned placement significantly outperforms uniform spacing (multiple points of improvement on MMLU).

2. **Linear Attention Block Selection**:

    - **Function**: Identifies the best-performing block among 6 SOTA linear attention candidates.
    - **Mechanism**: Compares RetNet, Mamba2, GLA, Deltanet, and Gated DeltaNet in terms of accuracy and efficiency within the PostNAS framework, directly validated at the target scale without proxy small-model experiments.
    - **Conclusion**: Gated DeltaNet achieves the best overall performance, owing to its data-dependent gating combined with the Delta rule (incremental hidden-state updates that conserve limited state memory).

3. **JetBlock Design (Novel Linear Attention Block)**:

    - **Function**: Enhances Gated DeltaNet with dynamic convolution to increase representational capacity.
    - **Mechanism**: (a) Introduces a Kernel Generator — the input is linearly projected to a lower dimension (ratio = 8), passed through SiLU activation, and then linearly projected to produce dynamic convolution kernel weights; (b) dynamic convolution is applied only to Value tokens (not Q/K); (c) static convolution on Q/K is removed as it becomes redundant after introducing dynamic convolution.
    - **Design Motivation**: Prior linear attention mechanisms use static convolution kernels that cannot adaptively adjust feature extraction patterns based on input; dynamic convolution adapts to context.

4. **Hardware-Aware Hyperparameter Search**:

    - **Function**: Optimizes key/value dimensions and the number of attention heads.
    - **Key Finding**: **KV cache size has a greater impact on generation throughput than parameter count** (since the decode phase is typically memory-bandwidth-bound).
    - **Mechanism**: Fixing KV cache size to match the original design, a grid search is performed over key dimension, value dimension, and number of heads. The final design uses fewer key dimensions, more value dimensions, and more heads — resulting in a larger parameter count but identical KV cache size, thus maintaining throughput while improving accuracy.
    - **Optimal Configuration**: $d_K=96, d_V=256, n_{\text{head}}=12$ (vs. original $d_K=192, d_V=192, n_{\text{head}}=8$).

### Loss & Training
- **Stage 1**: Freeze MLPs; train attention blocks with distillation loss over 50B tokens (Nemotron-CC + Redstone-QA).
- **Stage 2**: Full-model training with math and code data added, over 350B tokens.

## Key Experimental Results

### Main Results — MMLU(-Pro) and BBH

| Model | Type | Params (B) | Cache (MB) | Throughput (tok/s) | MMLU | MMLU-Pro | BBH |
|-------|------|-----------|------------|-------------------|------|----------|-----|
| Qwen2.5-1.5B | $O(n^2)$ | 1.5 | 1,792 | 241 | 59.5 | 28.9 | 44.1 |
| Qwen3-1.7B-Base | $O(n^2)$ | 1.7 | 7,168 | 61 | 60.3 | 37.8 | 54.2 |
| Llama3.2-3B | $O(n^2)$ | 3.0 | 7,168 | 60 | 54.9 | 25.0 | 47.1 |
| Mamba2-2.7B | $O(n)$ | 2.7 | 80 | 2,507 | 25.1 | 8.6 | 25.7 |
| RWKV7-1.5B | $O(n)$ | 1.5 | 24 | 3,050 | 41.0 | 13.4 | 15.9 |
| **Jet-Nemotron-2B** | **Hybrid** | **2.0** | **154** | **2,885** | **60.8** | **39.0** | **58.3** |
| Jet-Nemotron-4B | Hybrid | 4.0 | 258 | 1,271 | 65.2 | 44.2 | 65.0 |

### Ablation Study — PostNAS Step-by-Step Gains

| Step | MMLU Gain | Math Gain | Retrieval Gain | Commonsense Gain |
|------|-----------|-----------|----------------|-----------------|
| Full-attention placement | +5.3 | — | +7.8 | — |
| Linear attention block selection | — | +6.3 | — | +0.6 |
| JetBlock dynamic convolution | +0.7 | +0.5 | +0.6 | -0.2 |
| Hardware-aware search | +1.8 | +2.1 | +0.5 | +1.0 |
| **Total Gain** | **+5.3** | **+8.4** | **+7.8** | **+3.2** |

### Math Tasks

| Model | Throughput | Avg | GSM8K | MATH | MathQA |
|-------|-----------|-----|-------|------|--------|
| Qwen3-1.7B-Base | 61 | 42.3 | 62.8 | 16.7 | 46.0 |
| Jet-Nemotron-2B | 2,885 | **49.6** | **76.2** | **23.3** | **53.8** |

### Key Findings
- Jet-Nemotron-2B outperforms even larger MoE models (DeepSeek-V3-Small 2.2B/15B achieves only 53.3 vs. 60.8 on MMLU).
- At a context length of 256K, Jet-Nemotron-2B achieves 6.14× prefilling speedup and 53.6× decoding speedup over Qwen3.
- KV cache size, rather than parameter count, is the dominant factor governing generation throughput — Jet-Nemotron-2B uses only 154 MB of cache vs. 7,168 MB for Qwen3.
- Multi-choice tasks such as MMLU primarily rely on the pattern-matching capability of softmax attention; sliding-window attention is sufficient to maintain accuracy on these benchmarks.

## Highlights & Insights
- **PostNAS Paradigm Innovation**: Rather than training from scratch, PostNAS performs architecture search starting from an existing model, substantially reducing exploration cost and risk. If a new design fails within this framework, it is unlikely to succeed from scratch either.
- **JetBlock's Dynamic Convolution**: Replacing fixed convolution kernels with a learnable kernel generator incurs minimal overhead (dimensionality reduction ratio of 1/8) yet yields meaningful accuracy gains on retrieval and math tasks.
- **Hardware-Aware Design**: The finding that "KV cache size impacts throughput more than parameter count" is highly actionable — it implies that one can allocate more parameters for higher accuracy without sacrificing throughput.
- **Task-Specific Importance of Full-Attention Placement**: Different tasks require full-attention layers at different positions; uniform placement is suboptimal.

## Limitations & Future Work
- The ceiling of PostNAS is constrained by the quality of the starting model — a weaker Qwen2.5-1.5B base limits subsequent search gains.
- Only attention block design is explored; the MLP-freezing strategy precludes optimization of the overall architecture.
- Stage 2 full-model training over 350B tokens remains non-trivial in cost.
- Hyperparameters of JetBlock's dynamic convolution (kernel size, reduction ratio, etc.) are not systematically searched.
- End-to-end evaluation on real long-context applications (e.g., RAG, long-document QA) is insufficient.

## Related Work & Insights
- **vs. Mamba2/RWKV7**: Pure linear attention models achieve high throughput but suffer significant accuracy degradation (20–35 points below on MMLU); Jet-Nemotron's hybrid design achieves the best of both worlds.
- **vs. Qwen3/Gemma3**: Full-attention SOTAs achieve high accuracy but at 40–50× lower throughput; Jet-Nemotron delivers superior accuracy with drastically improved throughput.
- **vs. Hymba/Zamba2**: Prior hybrid models still fall noticeably short of full-attention SOTAs in accuracy; PostNAS is the first to enable hybrid models to match or surpass full-attention SOTAs.
- **vs. From-Scratch NAS**: Conventional NAS requires search plus full pretraining; PostNAS requires only search plus lightweight retraining.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The PostNAS paradigm is highly original; JetBlock's dynamic convolution is creative; hardware-aware search offers unique practical insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers six categories — MMLU, math, commonsense, retrieval, code, and long-context — with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables; some details require consulting the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly practical; provides a fully reproducible end-to-end pipeline for efficient LLM architecture design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[AAAI 2026\] Scaling and Transferability of Annealing Strategies in Large Language Model Training](../../AAAI2026/llm_efficiency/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)

<!-- RELATED:END -->
