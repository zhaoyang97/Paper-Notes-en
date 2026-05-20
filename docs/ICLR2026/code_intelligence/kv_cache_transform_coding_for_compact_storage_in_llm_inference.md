---
title: >-
  [Paper Note] KV Cache Transform Coding for Compact Storage in LLM Inference
description: >-
  [ICLR 2026][Code Intelligence][KV cache compression] This paper proposes KVTC, a KV cache compression method inspired by classical media compression techniques (PCA-based feature decorrelation + adaptive quantization + e…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "KV cache compression"
  - "transform coding"
  - "PCA"
  - "adaptive quantization"
  - "entropy coding"
date: 2026-05-08
content_hash: 4c1c4694b2620040
---

# KV Cache Transform Coding for Compact Storage in LLM Inference

**Conference**: ICLR 2026
**arXiv**: [2511.01815](https://arxiv.org/abs/2511.01815)  
**Code**: None  
**Area**: Code Intelligence
**Keywords**: KV cache compression, transform coding, PCA, adaptive quantization, entropy coding

## TL;DR

This paper proposes KVTC, a KV cache compression method inspired by classical media compression techniques (PCA-based feature decorrelation + adaptive quantization + entropy coding). KVTC achieves up to 20× compression (40×+ in specific scenarios) on Llama 3, Mistral NeMo, and R1-Qwen 2.5, outperforming baselines including token eviction, quantization, and SVD-based methods.

## Background & Motivation

Large-scale LLM inference serving faces a critical bottleneck: **KV cache memory management**.

The KV cache (Key-Value Cache) is a fundamental component of Transformer inference, storing Key and Value vectors of previous tokens to avoid redundant computation. In practice, KV cache management presents multiple challenges:

**Large memory footprint**: In long-context scenarios (e.g., 128K tokens), the KV cache can consume tens of gigabytes of GPU memory, becoming the primary memory bottleneck during inference.

**Cache reuse requirements**: In conversational and iterative code editing scenarios, prompts with shared prefixes are common; caches can be reused across turns to avoid recomputation.

**Stale cache handling**: Inactive caches still occupy valuable GPU memory, forcing either eviction followed by recomputation, or offloading to CPU/disk.

**Offloading efficiency**: CPU–GPU data transfer bandwidth is limited, making uncompressed cache transfer costly.

Existing KV cache optimization methods each have limitations:
- **Token eviction** (e.g., H2O, StreamingLLM): discards less important tokens, but the information loss is irreversible.
- **Quantization methods** (e.g., KVQuant): reduces numerical precision, but compression ratios are limited (typically 2–4×).
- **SVD methods**: approximates KV matrices with low-rank decompositions, but performance degrades on long sequences.

The authors' core insight is that **KV caches contain substantial statistical redundancy** that can be efficiently exploited using mature transform coding techniques from classical signal and media compression—analogous to how JPEG compresses images via DCT, followed by quantization and entropy coding.

## Method

### Overall Architecture

KVTC adopts a classical transform coding pipeline adapted to the KV cache compression setting:

```
KV Cache → PCA Transform (Decorrelation) → Adaptive Quantization → Entropy Coding → Compressed Data
Compressed Data → Entropy Decoding → Dequantization → Inverse PCA → Approximate KV Cache
```

### Key Designs

1. **PCA-based Feature Decorrelation**:

    - Applies a PCA transform along the feature dimension of the KV cache.
    - Converts correlated feature dimensions into uncorrelated principal components.
    - PCA bases are computed offline via a brief calibration procedure using a small set of representative data, learned independently per attention head.
    - **Design Motivation**: Significant correlations exist across feature dimensions in the KV cache. After decorrelation, energy concentrates in a small number of principal components, facilitating subsequent quantization and coding—analogous to the role of DCT in image compression.

2. **Adaptive Quantization**:

    - Quantizes each principal component independently after PCA transformation.
    - Allocates bits adaptively according to the variance (importance) of each component—components with higher variance receive more bits.
    - Different attention layers and heads may have different quantization configurations.
    - **Design Motivation**: Energy distribution after PCA is highly non-uniform. Uniform quantization wastes bits on low-variance components or degrades high-variance ones. Adaptive allocation achieves optimal bit assignment in the rate-distortion sense.

3. **Entropy Coding**:

    - Applies lossless entropy coding (e.g., arithmetic coding or Huffman coding) to the quantized coefficients.
    - Further removes statistical redundancy among quantized coefficients.
    - **Design Motivation**: The distribution of quantized coefficients is typically non-uniform (e.g., concentrated near zero); entropy coding exploits this non-uniformity for additional compression.

4. **Lightweight Calibration**:

    - The entire method requires only a single brief calibration pass to compute per-layer, per-head PCA transform matrices and quantization parameters.
    - Calibration can be completed on a small dataset without any training or modification of model parameters.
    - Calibrated transform and quantization parameters are stored offline and used directly during inference.
    - **Design Motivation**: Preserves the practicality and generality of the method by avoiding any model modification.

5. **Model Parameter Invariance**:

    - KVTC operates entirely at inference time and does not modify any model parameters.
    - Can be seamlessly integrated into existing LLM inference pipelines.
    - **Design Motivation**: As an inference-time optimization, it does not interfere with the training workflow, maximizing practical applicability.

### Loss & Training

- **No training required**: KVTC is a purely inference-time method.
- **Calibration procedure**: PCA bases and quantization parameters are computed from a small set of representative prompts.
- **Rate-distortion optimization**: Bit allocation in adaptive quantization is optimized based on classical rate-distortion theory.

## Key Experimental Results

### Main Results

Evaluated on Llama 3, Mistral NeMo, and R1-Qwen 2.5.

| Benchmark | Task Type | KVTC Compression Ratio | Performance Retention |
|-----------|-----------|------------------------|----------------------|
| AIME25 | Math Reasoning | 20× | Accuracy maintained |
| GSM8K | Math Reasoning | 20× | Accuracy maintained |
| MATH-500 | Math Reasoning | 20× | Accuracy maintained |
| LiveCodeBench | Code Generation | 20× | Accuracy maintained |
| MMLU | Knowledge QA | 20× | Accuracy maintained |
| LongBench | Long-context Understanding | 20× | Accuracy maintained |
| Qasper | Document QA | 20× | Accuracy maintained |
| RULER | Long-context Evaluation | 20× | Accuracy maintained |
| Specific Scenarios | — | 40×+ | Task-dependent |

### Comparison with Baselines

| Method | Compression Ratio | Performance Retention | Notes |
|--------|------------------|-----------------------|-------|
| Token Eviction (H2O, etc.) | Moderate | Irreversible information loss | Discards tokens |
| Quantization (KVQuant, etc.) | 2–4× | Good | Reduces precision |
| SVD Methods | Moderate | Unstable | Low-rank approximation |
| **KVTC** | **20× (up to 40×+)** | **Accuracy maintained** | Transform coding |

### Key Findings

- **Clear compression ratio advantage**: 20× compression significantly exceeds quantization methods (2–4×) and SVD methods.
- **Quality preservation**: KVTC maintains original model performance on reasoning and long-context accuracy even at high compression ratios.
- **Generality**: Consistently outperforms baselines across three distinct model architectures and eight benchmarks.
- **Practical impact**: 20× compression means a KV cache that originally occupies 20 GB of GPU memory can be reduced to 1 GB, substantially lowering inference costs.

## Highlights & Insights

- **Cross-domain knowledge transfer**: Successfully migrates classical signal processing and media compression techniques (transform coding) to LLM inference, demonstrating the enduring relevance of classical theory in new application domains.
- **Breakthrough compression ratio**: Compared to prior quantization methods achieving 2–4×, KVTC improves compression by an order of magnitude.
- **Training-free**: A purely inference-time method that is plug-and-play, with no impact on model parameters or training workflow.
- **Methodological transparency**: Each component (PCA, quantization, entropy coding) is grounded in established signal processing theory, providing strong interpretability.
- **Practical applicability**: Particularly well-suited for cache reuse scenarios such as conversational AI and code editing, directly reducing LLM serving costs.

## Limitations & Future Work

- **Compression/decompression overhead**: The encoding and decoding process of transform coding introduces additional computation, requiring trade-offs between compression ratio and latency.
- **Calibration data sensitivity**: The quality of PCA bases depends on the representativeness of calibration data; different task domains may require separate calibration.
- **Lossy compression**: Although performance is well-preserved in experiments, quality degradation at extreme compression ratios (40×+) warrants further investigation.
- **Dynamic scenarios**: For settings with frequent KV cache updates (e.g., streaming inference), the frequency and overhead of compression/decompression require optimization.
- **Compatibility with other optimizations**: Compatibility with inference optimization techniques such as Flash Attention and PagedAttention is not discussed.
- **Hardware adaptation**: Operations such as entropy coding may be less efficient on GPUs than on CPUs, necessitating hardware-specific adaptation.

## Related Work & Insights

- **H2O** (Zhang et al.): Heavy Hitter Oracle; evicts less important tokens based on attention scores.
- **StreamingLLM** (Xiao et al.): Token eviction strategy that retains attention sink tokens and a recent window.
- **KVQuant** (Hooper et al.): Quantization method specifically designed for KV caches.
- **JPEG/MPEG**: Classical transform coding paradigm in media compression (DCT + quantization + entropy coding); the conceptual inspiration for KVTC.
- **PagedAttention** (Kwon et al., vLLM): Paged memory management for KV caches; complementary to KVTC's compression approach.

**Insights**: Classical signal processing theory retains enormous applicability in LLM inference optimization. The success of KVTC suggests that the redundancy in KV caches far exceeds prior expectations—20× compression with negligible performance loss implies that the Transformer attention mechanism may exhibit substantial inefficiency in information utilization, and opens the door to more aggressive inference compression approaches such as learned transforms.

## Rating

- Novelty: ⭐⭐⭐⭐ — Transform coding itself is not new, but its adaptation to KV caches and the 20× compression ratio represent a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 models × 8 benchmarks, with comprehensive comparison against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ — Method is clearly described; connections to classical compression theory are well articulated.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses a core bottleneck in LLM inference, with very high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](../../ACL2026/code_intelligence/river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](../../ACL2026/code_intelligence/storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](../../AAAI2026/code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)

</div>

<!-- RELATED:END -->
