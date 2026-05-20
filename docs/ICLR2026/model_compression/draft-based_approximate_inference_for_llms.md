---
title: >-
  [Paper Note] Draft-based Approximate Inference for LLMs
description: >-
  [ICLR 2026][Model Compression][approximate inference] This paper proposes the Draft-based Approximate Inference framework, which leverages lookahead predictions from a lightweight draft model to more accurately estimate…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "approximate inference"
  - "KV cache compression"
  - "prompt compression"
  - "draft model"
  - "sparse attention"
date: 2026-05-08
content_hash: 05d051df9a8d3b27
---

# Draft-based Approximate Inference for LLMs

**Conference**: ICLR 2026
**arXiv**: [2506.08373](https://arxiv.org/abs/2506.08373)  
**Code**: [GitHub](https://github.com/furiosa-ai/draft-based-approx-llm)  
**Area**: Model Compression
**Keywords**: approximate inference, KV cache compression, prompt compression, draft model, sparse attention

## TL;DR

This paper proposes the Draft-based Approximate Inference framework, which leverages lookahead predictions from a lightweight draft model to more accurately estimate token/KV pair importance. The framework comprises three methods — SpecKV (KV cache dropping), SpecPC (prompt compression), and SpecKV-PC (cascaded compression) — and consistently outperforms existing baselines on long-context benchmarks.

## Background & Motivation

Long-context LLM inference faces two major bottlenecks: attention computation scales quadratically with context length, and KV cache memory grows linearly (exceeding 16 GB for 128K tokens on Llama-3.1-8B). Existing approximate inference methods include KV cache dropping (H2O, SnapKV), sparse attention (MInference), and prompt compression (LLMLingua-2), but all rely on attention activations from the current input tokens to estimate importance — an inherently retrospective strategy that cannot accurately predict **which KV pairs future generated tokens will actually need**.

- **Key Challenge**: Importance estimation requires future information, yet future tokens have not been generated. LAQ++ attempts to generate draft queries via sparse approximations of the target model itself, but it requires storing the complete target KV cache and thus cannot reduce peak memory.
- **Key Insight**: A lightweight draft model (e.g., 0.5B–3B) is used to generate lookahead tokens, obtaining approximate future information at negligible cost. This enables more accurate importance estimation while avoiding the memory and computation burden of the target model.

## Method

### Overall Architecture

Draft-based Approximate Inference is a unified framework: a small draft model first generates lookahead tokens from the input, and the resulting information (draft outputs or draft attention activations) guides KV cache or prompt compression in the target model. Unlike speculative decoding, the goal of this framework is to **reduce the total computation and memory of the target model**, rather than to accelerate verification.

### Key Designs

1. **SpecKV (Speculative KV Dropping)**: The draft model generates $n_{\text{lookahead}}$ lookahead tokens; the input tokens and lookahead tokens are then jointly fed into the target model for prefill. For each attention head, KV pair importance is estimated via cross-attention between the queries of the last $n_{\text{window}}$ input tokens plus lookahead tokens and the keys of the remaining input tokens. The top-$C_{\max}$ KV pairs plus in-window KV pairs are retained. Compared to LAQ++, SpecKV does not require storing the full target KV cache, achieving genuine reduction in peak memory. **Theorem 1** provides a theoretical guarantee: the error in importance scores is proportional to the draft embedding error, i.e., $\|s - \hat{s}\|_2 \leq \epsilon \|W_q W_k^T\|_2$.

2. **SpecPC (Speculative Prompt Compression)**: The full prompt is fed into the draft model, and the attention activation matrix $A \in \mathbb{R}^{n_{\text{layer}} \times n_{\text{head}} \times (n_{\text{in}}+n_{\text{lookahead}}-1) \times n_{\text{in}}}$ is extracted directly to estimate token importance. A large window with non-uniform weights (lower weights for positions farther from the end) is adopted; the first $l_{\text{skip}}$ layers are skipped (shallow-layer attention is insufficiently focused); and aggregation is performed by averaging first and then taking the maximum. **Theorem 2** provides a theoretical guarantee: under the RIP condition on the input, the attention approximation error is proportional to the output approximation error.

3. **SpecKV-PC (Cascaded Compression)**: SpecPC first compresses the prompt (e.g., to 2048 tokens), then SpecKV further compresses the KV cache (e.g., to 256). Since the target model only processes the compressed short prompt, latency and memory are substantially reduced. Cascaded compression even outperforms SpecKV alone, as SpecPC acts as a pre-filter that removes clearly unimportant tokens.

### Loss & Training

- No training is required: all methods are training-free inference-time optimizations.
- SpecKV combines sparse prefill (Vertical-Slash pattern) with KV cache dropping.
- SpecPC applies local pooling to preserve token continuity and avoids static chunking.

## Key Experimental Results

### Main Results

**Table 1: LongBench Performance Comparison (Qwen2.5 32B, KV cache $C_{\max}$=256)**

| Category | Method | SingleQA | MultiQA | Summ. | Few-shot | Code | All |
|----------|--------|----------|---------|-------|----------|------|-----|
| Dense | Target | 56.01 | 43.99 | 25.90 | 64.06 | 44.74 | 47.78 |
| KV | SnapKV | 52.54 | 40.21 | 19.89 | 61.18 | 40.12 | 42.98 |
| KV | LAQ++ | 55.15 | 44.14 | 22.24 | 63.25 | 41.19 | 45.79 |
| KV | **SpecKV** | 53.48 | 43.77 | 24.02 | 63.79 | 44.80 | **46.06** |
| KV | **SpecKV-PC** | 52.60 | 44.52 | 24.11 | 63.38 | 48.45 | **46.48** |

**Table 2: Prompt Compression Comparison ($C_{\max}$=1024)**

| Method | SingleQA | MultiQA | Summ. | Few-shot | Code | All |
|--------|----------|---------|-------|----------|------|-----|
| LLMLingua-2 | 33.83 | 26.39 | 22.85 | 32.46 | 43.01 | 30.90 |
| SpecPrefill | 45.94 | 39.32 | 23.16 | 62.04 | 43.17 | 42.70 |
| **SpecPC** | 51.23 | 41.40 | 23.37 | 62.26 | 38.23 | **43.66** |

### Ablation Study

- **Lookahead length**: SpecKV achieves the best performance with the maximum token limit; SpecPC requires only 1 lookahead token.
- **Draft model size**: A larger draft model (1.5B vs. 0.5B) reduces $\epsilon$ and improves downstream scores.
- **Importance score correlation**: Token importance scores from the draft model (1.5B) are highly correlated with those from the target model (14B), with $R^2$ approaching 1.

### Key Findings

- SpecKV-PC cascaded compression outperforms SpecKV alone, confirming the effectiveness of SpecPC as a pre-filter.
- At 64K context length, SpecKV reduces latency by 75% and saves approximately 25 GB of peak memory compared to LAQ++.
- All methods substantially outperform the draft model itself, demonstrating the framework's robustness to weak draft models.
- On the RULER synthetic benchmark, the advantage of lookahead-based methods becomes increasingly pronounced as context length grows.

## Highlights & Insights

- The unified framework integrates KV cache dropping and prompt compression under a single draft-based paradigm.
- The theoretical analysis elegantly connects compressed sensing (RIP) with attention approximation error bounds.
- SpecKV is the first method to leverage draft model lookahead for KV cache optimization.
- The cascaded compression design — coarse-to-fine, using the most appropriate signal at each stage — offers a broadly applicable architectural insight.

## Limitations & Future Work

- The draft model introduces additional memory overhead, though it can be offloaded to CPU.
- For very short contexts (<4K), the draft overhead may not be worthwhile.
- Validation has been limited to same-family model pairs (e.g., Qwen2.5-0.5B→32B); cross-family effectiveness remains unknown.
- Prompt compression reassigns position IDs, which may affect models that rely on absolute positional information.

## Related Work & Insights

- **SnapKV** (Li et al., 2024): Performs KV compression based on attention over the last few tokens; SpecKV extends this with lookahead.
- **LAQ++** (Wang et al., 2025): Also employs lookahead but requires the full target KV cache, precluding peak memory reduction.
- **SpecPrefill** (Liu et al., 2025): A predecessor to SpecPC with a fixed window size of 1; SpecPC improves upon this with a large window and non-uniform weights.
- **Takeaway**: Beyond speculative decoding, draft models hold broad potential for inference optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework is well-unified and SpecKV pioneers draft lookahead for KV dropping, though each individual method offers incremental gains.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ RULER + LongBench + multiple models + efficiency measurements + multimodal extensions — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical analysis is clear, though the large number of method combinations makes the paper somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ A practical solution for long-context inference optimization; the cascaded compression design offers meaningful reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[NeurIPS 2025\] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](../../NeurIPS2025/model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)
- [\[ICLR 2026\] Highly Efficient and Effective LLMs with Multi-Boolean Architectures](highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [\[ICLR 2026\] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)

</div>

<!-- RELATED:END -->
