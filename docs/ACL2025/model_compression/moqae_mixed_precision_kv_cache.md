---
title: >-
  [Paper Note] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts
description: >-
  [ACL 2025][Model Compression][KV cache quantization] MoQAE creatively treats different quantization bit-width configurations as "experts" in MoE, employing a lightweight router to learn the optimal quantization strategy for each chunk. Combined with routing freezing and routing sharing mechanisms, it significantly reduces the KV cache memory of long-context inference with almost zero accuracy loss.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "KV cache quantization"
  - "mixed-precision"
  - "MoE"
  - "long-context inference"
  - "routing"
date: 2026-05-08
content_hash: da7801304d2d1b28
---

# MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts

**Conference**: ACL 2025  
**arXiv**: [2506.07533](https://arxiv.org/abs/2506.07533)  
**Code**: None  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: KV cache quantization, mixed-precision, MoE, long-context inference, routing

## TL;DR
MoQAE creatively treats different quantization bit-width configurations as "experts" in MoE, employing a lightweight router to learn the optimal quantization strategy for each chunk. Combined with routing freezing and routing sharing mechanisms, it significantly reduces the KV cache memory of long-context inference with almost zero accuracy loss.

## Background & Motivation

**Background**: In long-context LLM inference, the KV cache represents the memory bottleneck—with a 128k context, the KV cache of Llama2-13B can reach 100GB. Existing optimizations include: pruning (H2O/StreamingLLM), system-level management (vLLM/S3), and quantization (KIVI/Atom/KVQuant). Quantization is the most direct and effective method.

**Limitations of Prior Work**: (1) Uniform quantization (e.g., INT4/INT2) uses the same bit-width for all tokens, which leads to significant accuracy loss in crucial tokens; (2) Mixed-precision quantization (KVQuant/MiKV) can allocate higher bits to important tokens, but requires a complex and time-consuming search process to determine the optimal configuration; (3) Existing methods fail to balance effectiveness and efficiency.

**Key Challenge**: Determining the optimal mixed-precision configuration requires search (slow), while fast adaptation forces uniform quantization (poor accuracy).

**Goal**: Quickly and effectively determine the mixed-precision quantization configuration of the KV cache.

**Key Insight**: Formulate the mixed-precision configuration selection problem as an MoE routing task—since routers naturally possess fast training and efficient inference properties.

**Core Idea**: Employ an MoE router to automatically learn the quantization bit-width for each chunk, combining a comprehensive loss function to balance accuracy and memory.

## Method

### Overall Architecture
Input text → Split into equal-length chunks → Within each LLM block, chunks enter the router → The router outputs the selection probability of each token for different "quantization experts" → Majority voting within the chunk determines the quantization bit-width for that chunk → KV cache is quantized according to the selected bit-width → Standard attention computation is performed.

### Key Designs

1. **Quantization-Aware Experts**:

    - **Function**: Treat different quantization configurations (FP16, INT4, INT2) as experts in MoE.
    - **Mechanism**: The router MLP receives chunk $C \in \mathbb{R}^{N \times D}$ and outputs probability $\mathcal{P} = f(CW_1 \cdot CW_2)W_3 \in \mathbb{R}^{N \times M}$. Each token selects the expert with the highest probability, and a chunk-level majority vote determines the final quantization strategy for the chunk.
    - **Design Motivation**: Traditional token-by-token routing in MoE is too slow; chunk-by-chunk routing is much more efficient and aligns with the practical scenario of block-wise processing.
    - Difference from traditional MoE: Experts are quantization configurations rather than FFN layers, and the routing decision determines quantization precision instead of activating subnetworks.

2. **Comprehensive Loss Function**:

    - **Function**: Train the router to balance model accuracy and memory consumption.
    - $L = \lambda L_{model} + (1-\lambda) L_{mem}$
    - $L_{model}$: Multiplies the NLL loss by routing probabilities and penalizes with $1/B_j$ (lower-bit experts suffer larger penalties as lower precision impacts accuracy more severely).
    - $L_{mem}$: Penalizes with $16/B_j$ (higher-bit experts suffer larger penalties as higher precision consumes more memory).
    - $\lambda$ controls the trade-off between accuracy and memory, which can be adjusted according to user requirements.

3. **Routing Freezing (RF)**:

    - **Function**: Fix the quantization strategy of the first chunk to FP16.
    - **Design Motivation**: The initial tokens of LLMs exhibit high attention weights (known as the "attention sink" phenomenon), which are critical for model generation. Freezing the initial chunk protects the accuracy of these crucial tokens.

4. **Routing Sharing (RS)**:

    - **Function**: Share the quantization strategy of the router across different LLM blocks.
    - **Design Motivation**: Reduce the parameter size and inference overhead of the router. Experiments indicate standard layers share similar optimal quantization configurations.

### Loss & Training
- Freeze all parameters of the LLM and only fine-tune the router.
- Employ a small-scale calibration dataset.
- Training is highly efficient and lightweight.

## Key Experimental Results

### Main Results: Wikitext2 Perplexity (4~16 bit range)

| Method | AvgBit | LLaMA-7B | LLaMA-13B | LLaMA2-7B | LLaMA2-13B |
|------|--------|---------|----------|----------|-----------|
| FP16 | 16.00 | 5.68 | 5.09 | 5.11 | 4.57 |
| INT4 | 4.00 | 7.40 | 6.82 | 7.31 | 6.59 |
| KVQuant-4b | 4.00 | 7.13 | 6.65 | 6.70 | 6.11 |
| QoQ-4b-gs128 | 4.00 | 5.89 | 5.25 | 5.89 | 5.24 |
| MiKV | 5.50 | 6.25 | 5.58 | 5.89 | 5.33 |
| **MoQAE-λ0.5** | **4.13** | **5.76** | **5.15** | **5.22** | **4.65** |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| MoQAE Full (RF + RS) | Best | Routing freezing + Routing sharing |
| w/o RF (No freezing of initial chunk) | Performance Drop | Precision drops due to initial tokens being quantized with low bits |
| w/o RS (Independent routing per layer) | Slightly better but slower | Finer granularity but increases routing overhead |
| λ=0.8 (Precision-preferred) | High accuracy, large memory | More chunks are assigned higher bits |
| λ=0.2 (Memory-preferred) | Low accuracy, small memory | More chunks are assigned lower bits |

### Key Findings
- **Almost lossless quantization**: Under an average of 4.13 bits, MoQAE achieves a PPL only slightly higher than FP16 (5.76 vs 5.68), which is significantly superior to uniform INT4 (7.40).
- **Outperforming high average bit-width methods**: MoQAE (4.13 bit) outperforms MiKV (5.50 bit), proving that intelligent allocation is more effective than uniform high-bit quantization.
- **Routing freezing is critical for accuracy protection**: This validates the importance of the attention sink phenomenon in quantization contexts.
- **λ provides flexible control**: Users can flexibly trade off between accuracy and memory depending on deployment requirements.

## Highlights & Insights
- **Creative formulation of quantization search as MoE routing**: This analogy is highly ingenious—traditional mixed-precision methods require searches to find optimal configurations, while MoE routing fundamentally selects "the optimal processing approach for each input." This cross-domain conceptual shift is highly inspiring.
- **Chunk-by-chunk routing**: This is more efficient than token-by-token routing, and tokens within the same chunk typically share similar importance, conforming to the locality assumption.
- **Training via fine-tuning the router only**: Since all LLM parameters are frozen, the training overhead is extremely low, making it deployment-friendly.
- **Flexibility of adjustable λ**: Different deployment scenarios can employ different values of λ, enabling multi-scenario utility from a single training run.

## Limitations & Future Work
- Chunk size is a fixed hyperparameter; different tasks/context lengths may require different chunk sizes.
- Routing sharing might suffer precision loss when quantization requirements vary greatly across layers.
- Mixed-precision is only applied to the KV cache, without extension to weight quantization.
- Although the additional computational overhead introduced by the router is small, it might not be negligible in extremely latency-sensitive scenarios.
- Integration with the latest KV cache compression methods (such as MLA/GQA) remains unexplored.

## Related Work & Insights
- **vs KIVI (Liu et al., 2024)**: KIVI observes that the key cache has outlier channels requiring per-channel quantization. MoQAE focuses on mixed-precision at chunk granularity, meaning the two approaches can be complementary.
- **vs MiKV (Yang et al., 2024)**: MiKV utilizes H2O attention scores to determine important tokens, but the search is slow. MoQAE makes direct selections via a trained router, achieving zero search overhead at inference time.
- **vs MoICE (Lin et al., 2024)**: MoICE utilizes MoE for selection of RoPE bases, inspiring MoQAE to apply MoE for quantization configuration selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of treating quantization configurations as MoE experts is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models and datasets, with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Overall very clear, with detailed method descriptions.
- Value: ⭐⭐⭐⭐ Provides a practical solution for KV cache compression in long-context inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Channel-Aware Mixed-Precision Quantization for Efficient Long-Context Inference](../../ICLR2026/model_compression/channel-aware_mixed-precision_quantization_for_efficient_long-context_inference.md)
- [\[ACL 2025\] APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)
- [\[AAAI 2026\] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](../../AAAI2026/model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)
- [\[ACL 2025\] EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](eac_moe_expert_aware_compression.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

</div>

<!-- RELATED:END -->
