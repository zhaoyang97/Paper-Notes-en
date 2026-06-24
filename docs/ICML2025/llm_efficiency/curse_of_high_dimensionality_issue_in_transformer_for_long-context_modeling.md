---
title: >-
  [Paper Note] Curse of High Dimensionality Issue in Transformer for Long-context Modeling
description: >-
  [ICML 2025][LLM Efficiency][Attention Sparsity] This paper revisits the attention redundancy issue in sequence modeling from a supervised learning perspective and proposes the Dynamic Group Attention (DGA) mechanism. By dynamically grouping and aggregating unimportant tokens to reduce redundancy in attention computation, DGA maintains competitive performance while substantially reducing inference latency (achieving a 2.42x inference speedup for LLaMA2-7B under 16K context).
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Attention Sparsity"
  - "Dynamic Group Attention"
  - "Long-context Modeling"
  - "Redundancy Elimination"
  - "Group Encoding"
date: 2026-05-08
content_hash: 7aad07ab421d6017
---

# Curse of High Dimensionality Issue in Transformer for Long-context Modeling

**Conference**: ICML 2025  
**arXiv**: [2505.22107](https://arxiv.org/abs/2505.22107)  
**Code**: [https://github.com/bolixinyu/DynamicGroupAttention](https://github.com/bolixinyu/DynamicGroupAttention)  
**Area**: LLM Efficiency  
**Keywords**: Attention Sparsity, Dynamic Group Attention, Long-context Modeling, Redundancy Elimination, Group Encoding

## TL;DR
This paper revisits the attention redundancy issue in sequence modeling from a supervised learning perspective and proposes the Dynamic Group Attention (DGA) mechanism. By dynamically grouping and aggregating unimportant tokens to reduce redundancy in attention computation, DGA maintains competitive performance while substantially reducing inference latency (achieving a 2.42x inference speedup for LLaMA2-7B under 16K context).

## Background & Motivation
**Background**: Transformer-based LLMs capture long-range dependencies through self-attention, demonstrating outstanding performance in NLP tasks.

**Limitations of Prior Work**: Long-context modeling faces severe computational efficiency challenges—attention weights are typically sparse (with most tokens contributing minimally), yet all tokens still consume the same computational resources.

**Key Challenge**: Existing methods (e.g., StreamingLLM, LM-Infinite) simplify attention by discarding tokens, which may disrupt critical token interactions and negatively impact tasks that require comprehensive context understanding, such as QA and summarization.

**Goal**: To reduce redundant attention computations while preserving critical token interactions.

**Key Insight**: Reformulate traditional probabilistic sequence modeling as a supervised learning task, analyze attention optimization from the perspective of linear coding theory, and propose a group encoding strategy.

**Core Idea**: Replace token-by-token processing with group aggregation—important tokens retain full attention, while unimportant tokens are grouped and aggregated for computation, alongside the introduction of complementary KV pairs to address autoregressive constraints.

## Method

### Overall Architecture
The DGA method consists of three key steps: (1) partitioning tokens into focus tokens and non-focus tokens using importance scores; (2) grouping non-focus tokens by a group size $m$ and aggregating their KV pairs; (3) introducing complementary KV pairs to provide supplemental information for tokens that cannot access group information due to autoregressive properties. Finally, the grouped $K_{group}$ and $V_{group}$ are constructed for attention computation.

### Key Designs

1. **Reformulating Sequence Modeling from a Supervised Learning Perspective**:

    - **Function**: Reformulate next-token prediction as a supervised learning task.
    - **Mechanism**: Context is partitioned into relevant tokens and irrelevant tokens, making the redundancy issue more explicit.
    - **Design Motivation**: Traditional sequence modeling treats the entire context as an indivisible whole, making it difficult to analyze which tokens are redundant. The supervised learning perspective provides a structured analysis approach.

2. **Theoretical Analysis of Attention Sparsity and Group Encoding Strategy**:

    - **Function**: Theoretically prove that attention weights exhibit $\rho$-sparsity, meaning only a few tokens significantly contribute to target representations.
    - **Mechanism**: Formulate attention optimization as a linear coding problem and propose group encoding to partition $L$-dimensional weights into $k$ groups.
    - **Theorem 2** (Noise Tolerance): Group encoding reduces the variance of weight variation by $1/m^2$ (where $m$ is the group size), enhancing robustness.
    - **Theorem 3** (Optimization Efficiency): Group encoding reduces the condition number of the Hessian matrix, accelerating convergence.
    - Difference from prior methods: Unlike sparse methods that directly discard tokens, group encoding preserves information through group aggregation.

3. **Dynamic Group Attention (DGA) Mechanism**:

    - **Function**: Instantiate the theoretical group encoding strategy as a trainable attention mechanism.
    - **Mechanism**:
        - **Focus Token Identification**: Evaluate the importance of each token via accumulated attention weights; the top-$\gamma$ are designated as focus tokens.
        - **Fast Approximation**: Sample a small number of queries to approximate the full attention matrix, accelerating the computation of importance scores.
        - **Group Aggregation of Non-focus Tokens**: Group the KV pairs of non-focus tokens by group size $m$, and aggregate them via weighted summation within each group.
        - **Complementary Tokens**: To resolve information loss caused by grouping under autoregressive constraints, introduce complementary KV pairs to recover missing information.
    - **Design Motivation**: Direct utilization of the theoretical guarantees from Theorems 2 and 3 to reduce computational complexity through grouping.
    - Difference from CCA-Attention: CCA uses a fixed intra-group aggregation + sliding window approach, whereas DGA dynamically identifies key tokens and aggregates only unimportant tokens.

### Loss & Training
- Trained using standard language modeling loss.
- Grouped Inference: During the decoding phase, after generating every $m' = 1.1m$ tokens, the grouping weights are recomputed using the attention of the last query to dynamically update the KV groups.
- Training Configuration: 8x A800 GPUs, micro-batch=1, gradient accumulation=8, 1000 steps.

## Key Experimental Results

### Main Results (LongBench-E, 31K Context)

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-Shot Learning | Synthetic | Code | Average | ITL/ms |
|------|---------|---------|------|--------|------|------|--------|---------|
| LLaMA2-7B (Vanilla) | 6.43 | 2.37 | 13.65 | 56.65 | 3.04 | 48.0 | 21.69 | 69.70 |
| MInference | 5.86 | 2.65 | 14.33 | 55.99 | 2.63 | 48.41 | 21.64 | 94.34 |
| StreamingLLM | 4.99 | 4.13 | 11.51 | 45.43 | 2.16 | 30.38 | 16.43 | 78.28 |
| LM-Infinite | 3.54 | 2.61 | 3.31 | 48.97 | 1.33 | 35.26 | 15.84 | 102.22 |
| **DGA-LLM (Ours)** | 3.61 | 3.58 | 6.81 | **57.90** | 1.47 | **53.45** | 21.14 | **28.80** |

### EM Score (Different Context Lengths)

| Method | 4K | 8K | 16K | 32K | Average |
|------|-----|-----|------|------|------|
| LLaMA2-7B (Vanilla) | 37.2 | 36.4 | 33.8 | 26.8 | 33.6 |
| StreamingLLM | 30.2 | 25.8 | 22.2 | 20.8 | 24.8 |
| LM-Infinite | 29.4 | 28.6 | 23.8 | 22.4 | 26.1 |
| **DGA-LLM (Ours)** | **35.0** | 27.4 | **25.6** | **22.6** | **27.7** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| DGA Convergence Speed | Converges faster than Vanilla | Pre-training validation loss comparison on OPT-125M |
| Noise Robustness | Significantly lower KL divergence | Smaller output distribution deviation after adding Gaussian noise |
| ITL Growth with Length | Almost constant from 4K to 16K (26 to 29 ms) | Other methods' ITL increases sharply with length |

### Key Findings
- Attention sparsity strengthens as context length increases: when $\rho=0.02$, it approaches 1 at $L=400$.
- DGA inference speed is 2.42x faster than Vanilla Self-Attention and 3.28x faster than MInference.
- The group encoding strategy is theoretically guaranteed and experimentally validated for convergence speed and noise robustness.

## Highlights & Insights
- Reformulates sequence modeling as a supervised learning problem, providing a novel theoretical framework to understand attention redundancy.
- Derives two core advantages of group encoding (noise tolerance + accelerated optimization) from linear coding theory with rigorous theoretical support.
- The advantage of DGA in inference latency is prominent (ITL barely grows with context length), which is highly valuable for practical deployment.
- The design of complementary tokens elegantly addresses the information loss caused by grouping under autoregressive constraints.

## Limitations & Future Work
- On the LongBench-E average score, it is close to but does not surpass Vanilla Self-Attention; the primary advantage lies in efficiency rather than quality.
- The theoretical analysis is based on a simplified single-layer single-head model; the theoretical guarantees for multi-layer multi-head scenarios remain to be improved.
- The identification of focus tokens depends on approximate attention weights, and the approximation accuracy might affect grouping quality.
- Currently only validated on LLaMA2-7B; the efficacy on larger-scale models remains to be explored.

## Related Work & Insights
- Difference from CCA-Attention: CCA uses fixed grouping + sliding window, whereas DGA dynamically identifies key tokens.
- Difference from StreamingLLM/LM-Infinite: The latter directly discards tokens, while DGA retains information via aggregation.
- The group encoding theory can inspire redundancy elimination in other modules, such as the sparsification of FFN layers.
- The dynamic group update strategy in the inference phase is a practical engineering design.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel theoretical perspective, deriving attention optimization from coding theory)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple benchmarks and metrics, but model scale is relatively small)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations, complete structure)
- Value: ⭐⭐⭐⭐ (Significant inference acceleration with theoretical backing)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](../../ACL2026/llm_efficiency/comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)
- [\[ICLR 2026\] Revisiting Long-context Modeling from Context Denoising Perspective](../../ICLR2026/llm_efficiency/revisiting_long-context_modeling_from_context_denoising_perspective.md)
- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](nextlong_toward_effective_long-context_training_without_long_documents.md)

</div>

<!-- RELATED:END -->
