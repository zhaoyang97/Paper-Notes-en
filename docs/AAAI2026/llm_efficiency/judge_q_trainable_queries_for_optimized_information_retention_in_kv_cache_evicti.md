---
title: >-
  [Paper Note] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction
description: >-
  [AAAI 2026][LLM Efficiency][KV Cache Pruning] This paper proposes Judge Q, which introduces trainable soft tokens into the model vocabulary and trains their attention patterns to align with those of actual decoding token…
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "KV Cache Pruning"
  - "Trainable Soft Tokens"
  - "Attention Distillation"
  - "Long-context Inference"
  - "Global Information Retention"
date: 2026-05-08
content_hash: fae13b71972e4a50
---

# Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction

**Conference**: AAAI 2026
**arXiv**: [2509.10798](https://arxiv.org/abs/2509.10798)  
**Code**: [GitHub](https://github.com/Mambaaaaaaaaa/Judge-Q)  
**Area**: LLM Efficiency / KV Cache Compression
**Keywords**: KV Cache Pruning, Trainable Soft Tokens, Attention Distillation, Long-context Inference, Global Information Retention

## TL;DR
This paper proposes Judge Q, which introduces trainable soft tokens into the model vocabulary and trains their attention patterns to align with those of actual decoding tokens, enabling them to replace local-window queries for evaluating KV cache importance during the prefill stage. This approach better preserves global information, achieving ~1-point improvement on LongBench and 3+ points on RULER.

## Background & Motivation

**Background**: During LLM inference, the KV cache grows linearly with sequence length, becoming a memory bottleneck in long-context scenarios (>10K tokens). Existing KV cache pruning methods (H2O, SnapKV, PyramidKV, etc.) evaluate KV pair importance by computing attention scores during the prefill stage, retaining only the top-k most important KV pairs.

**Limitations of Prior Work**:
- **Over-reliance on local windows**: Existing methods use tokens from the last window as queries to compute KV importance scores, implicitly assuming that the query appears at the end of the input. Performance degrades significantly when this assumption is violated.
- **Neglect of global information**: Local windows can only observe the tail of the sequence, making it difficult to adequately assess the importance of KV pairs at distant positions for generation.
- **Theoretical upper bound not approached**: The authors find that using the attention of actual decoding tokens to select KV pairs yields the best results (theoretical upper bound), but decoding tokens are unknown during prefill.

**Key Challenge**: The ideal objective of KV cache pruning is to retain the KV pairs most important to future decoding, yet the decoding content is unknown at prefill time, and the local window serves as an inadequate proxy.

**Goal**: Design a method that approximates the theoretical upper bound of "selecting KV pairs using actual decoding tokens" during the prefill stage, while maintaining low training cost.

**Key Insight**: Since the theoretical upper bound relies on the attention maps of decoding tokens to select KV pairs, a set of soft tokens can be trained so that their attention maps approximate those of decoding tokens, serving as their proxy.

**Core Idea**: Train soft tokens to simulate the attention distribution of decoding tokens to guide KV cache pruning, with only the embedding layer parameters updated, incurring minimal overhead.

## Method

### Overall Architecture
Judge Q appends $n$ soft tokens (default $n=32$) to the end of the model vocabulary. During training, soft tokens are concatenated after the prompt, and their attention maps over the prompt are trained to align with those of the actual response. During inference, soft tokens are appended to the end of the input, and their attention scores replace local-window queries to compute KV importance. After pruning, the soft tokens are removed and decoding proceeds with the pruned KV cache.

### Key Designs

1. **Soft Token Attention Distillation**:

    - **Function**: Train soft tokens so that their attention distribution over the prompt matches that of actual decoding tokens attending to the same prompt.
    - **Mechanism**: Soft tokens and response tokens are each concatenated to the prompt separately; their respective attention maps over the prompt are computed and averaged along the token dimension to obtain $\mathbf{A}_{\text{soft}}$ and $\mathbf{A}_{\text{resp}}$. The training loss is the MSE between the two: $\mathcal{L} = \text{MSE}(\mathbf{A}_{\text{soft}}, \mathbf{A}_{\text{resp}})$.
    - **Design Motivation**: Decoding tokens naturally attend to KV pairs most critical for generation. If soft tokens learn the same attention patterns, they can substitute for decoding tokens during prefill to make equally informed KV importance decisions.
    - **Distinction from prompt tuning**: Conventional prompt tuning optimizes generation quality, whereas Judge Q optimizes attention pattern alignment — an entirely different objective.

2. **Extremely Low Training Cost**:

    - **Function**: Only the embedding parameters corresponding to the soft tokens are fine-tuned.
    - **Mechanism**: All model weights are frozen; only the embedding vectors of the 32 newly added tokens are trained. Training data consists of 50K samples from ShareGPT (45K general + 5K code), with responses generated by the model itself rather than using original annotations.
    - **Design Motivation**: Minimize training overhead to enable lightweight adaptation to any open-source model. Model-generated responses are used because the attention maps must be consistent with the model's own decoding behavior.

3. **KV Cache Pruning at Inference**:

    - **Function**: Guide KV pruning using soft token attention during the prefill stage.
    - **Mechanism**: Append 32 soft tokens to the end of the input → prefill → compute attention maps of soft tokens over the entire input → rank KV pairs by attention scores → retain top-k → remove soft tokens → proceed with normal decoding.
    - **Design Motivation**: Soft tokens act as "probes" that, through training, acquire global awareness and can more effectively identify globally critical information compared to local windows.

### Loss & Training
- Loss function: $\mathcal{L} = \frac{1}{d}\|\mathbf{A}_{\text{soft}} - \mathbf{A}_{\text{resp}}\|_2^2$
- Training data: ShareGPT 50K; responses generated by the model itself (not original annotations).
- Soft token count: $n=32$ achieves the best balance.
- Training involves only the embedding vectors of 32 new tokens; all other parameters remain frozen.

## Key Experimental Results

### Main Results

LongBench results (Llama-3.1-8B-Instruct):

| KV Budget | StreamingLLM | H2O | SnapKV | PyramidKV | **Judge Q** | Full KV |
|-----------|-------------|-----|--------|-----------|------------|---------|
| 128 | 30.50 | 33.67 | 34.31 | 34.08 | **35.90** | 41.23 |
| 256 | 31.79 | 34.37 | 36.56 | 36.00 | **37.69** | 41.23 |
| 512 | 32.64 | 35.30 | 38.31 | 37.58 | **39.17** | 41.23 |

RULER results (Llama-3.1-8B-Instruct, seq=8192):

| KV Budget | SnapKV | PyramidKV | **Judge Q** | Full KV |
|-----------|--------|-----------|------------|---------|
| 256 | 57.83 | 56.86 | **63.13** | 87.18 |
| 512 | 62.76 | 61.19 | **69.24** | 87.18 |
| 1024 | 68.21 | 66.30 | **74.12** | 87.18 |

### Ablation Study

Critical KV Hit Rate (overlap with theoretical upper bound):

| KV Budget | SnapKV | Judge Q | Gain |
|-----------|--------|---------|------|
| 128 | 53.44% | **61.37%** | +7.93% |
| 256 | 55.23% | **62.34%** | +7.11% |
| 512 | 58.46% | **65.06%** | +6.60% |

Text continuation tasks (DeepSeek-R1-Distill-Llama-8B):

| Dataset | SnapKV | Judge Q |
|--------|--------|---------|
| MATH-500 (budget=1024) | 52.4 | **55.0** |
| AIME24 (budget=3072) | 31.1 | **37.8** |

### Key Findings
- **Greater gains at low budgets**: +1.59 points at budget=128 vs. +0.86 points at budget=512; the method is more effective under tighter resource constraints.
- **Most significant improvements on RULER**: Exceeding 3 points, with a peak gain approaching 10 points, as RULER tasks rely more heavily on global retrieval capability.
- **Critical KV Hit Rate consistently ~8% higher**: Indicating that soft tokens learn attention patterns that genuinely approximate the theoretical upper bound.
- **Larger advantage when the query is not at the end of the prompt**: After prompt permutation, baseline performance drops ~10%, while Judge Q drops less than 7%.
- **Model-generated responses outperform original responses**: Because attention maps must align with the model's own decoding behavior.
- **Optimal soft token count is $n=32$**: Too few (16) provides insufficient information; too many (64+) increases training difficulty with diminishing returns.

## Highlights & Insights
- **Discovery and approximation of the theoretical upper bound**: The paper first establishes that "selecting KV pairs using actual decoding tokens" constitutes an upper bound, then designs soft tokens to approximate it. This "identify upper bound → approximate it" research paradigm is particularly clear and compelling.
- **Minimal training cost with significant gains**: Training only the embedding vectors of 32 tokens on 50K samples yields consistent improvements on LongBench and RULER, making the method easily adaptable to any open-source model.
- **Soft tokens as "probes"**: Combining attention distillation with prompt tuning, the soft tokens are not designed to alter model output but to "sense" which KV pairs are important. This idea is transferable to other scenarios requiring global awareness (e.g., token pruning, layer pruning).
- **Empirical demonstration of local-window limitations**: Prompt permutation experiments directly expose the fragility of local-window-based methods.

## Limitations & Future Work
- **One-shot pruning only at prefill**: The KV cache continues to grow during decoding as new tokens are generated; Judge Q does not address dynamic pruning during the decoding stage (acknowledged by the authors as future work).
- **Substantial gap from Full KV remains**: At budget=128, Judge Q (35.90) vs. Full KV (41.23) yields a gap of 5.33 points; information loss under extremely low budget settings remains significant.
- **Fixed number of soft tokens**: Different tasks may require different numbers of soft tokens to capture global information at varying granularities.
- **Training depends on a specific dataset**: The distribution of ShareGPT may not cover all downstream tasks, and each model requires its own training, despite evidence that self-generated responses are preferable.
- **Combination with KV cache quantization unexplored**: KV cache compression has two main directions (pruning + quantization); Judge Q addresses only pruning, and compatibility with quantization methods remains unverified.

## Related Work & Insights
- **vs. SnapKV**: SnapKV uses a local window with pooling; Judge Q replaces the local window with soft tokens. Judge Q consistently outperforms across all budget settings, with a critical KV hit rate ~8% higher.
- **vs. PyramidKV**: PyramidKV adds layer-wise budget allocation on top of SnapKV but still relies on local-window queries. Judge Q addresses the problem at its root (query selection).
- **vs. StreamingLLM**: StreamingLLM retains only the initial tokens and a local window, suffering the most severe information loss and lagging substantially across all tasks.
- **vs. Lookahead Q-Cache**: Also uses pseudo decoding tokens to guide pruning but requires additional decoding steps. Judge Q is more elegant, as trained soft tokens incur zero overhead during inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Soft token attention distillation for KV cache pruning is an original idea; the theoretical upper bound analysis is persuasive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, two models, multiple budgets, hit rate analysis, prompt perturbation, data ablation, and token count ablation — exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The progression from observation → motivation → method is logical and clear; Figure 1's method comparison is intuitive.
- **Value**: ⭐⭐⭐⭐ The method is concise and practical, with minimal training cost, and can be directly applied to any open-source LLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](../../ICML2026/llm_efficiency/a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ICML 2026\] Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](../../ICML2026/llm_efficiency/scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)
- [\[ICLR 2026\] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](../../ICLR2026/llm_efficiency/randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)
- [\[AAAI 2026\] Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)

</div>

<!-- RELATED:END -->
