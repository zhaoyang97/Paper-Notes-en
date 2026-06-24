---
title: >-
  [Paper Note] CommVQ: Commutative Vector Quantization for KV Cache Compression
description: >-
  [ICML 2025][Robotics][KV Cache Compression] This paper proposes CommVQ, which compresses the KV cache using Additive Vector Quantization (AVQ). By innovatively designing a codebook that commutes with RoPE and training it via the EM algorithm, CommVQ achieves near-lossless accuracy at 2-bit and retains usable accuracy at 1-bit, enabling LLaMA-3.1 8B to support a 128K context length on a single RTX 4090 GPU.
tags:
  - "ICML 2025"
  - "Robotics"
  - "KV Cache Compression"
  - "Vector Quantization"
  - "RoPE Commutativity"
  - "Long-Context Inference"
  - "1-bit Quantization"
date: 2026-05-08
content_hash: abf0f413a2486e84
---

# CommVQ: Commutative Vector Quantization for KV Cache Compression

**Conference**: ICML 2025  
**arXiv**: [2506.18879](https://arxiv.org/abs/2506.18879)  
**Code**: [https://github.com/UMass-Embodied-AGI/CommVQ](https://github.com/UMass-Embodied-AGI/CommVQ)  
**Area**: Robotics / Model Compression  
**Keywords**: KV Cache Compression, Vector Quantization, RoPE Commutativity, Long-Context Inference, 1-bit Quantization

## TL;DR
This paper proposes CommVQ, which compresses the KV cache using Additive Vector Quantization (AVQ). By innovatively designing a codebook that commutes with RoPE and training it via the EM algorithm, CommVQ achieves near-lossless accuracy at 2-bit and retains usable accuracy at 1-bit, enabling LLaMA-3.1 8B to support a 128K context length on a single RTX 4090 GPU.

## Background & Motivation

**Background**: LLM context lengths continue to grow (reaching 128K+), causing the KV cache to become the primary bottleneck for GPU memory. For instance, LLaMA-3.1 8B requires 88GB of KV cache for a 128K context with a batch size of 2.

**Limitations of Prior Work**: Existing KV cache quantization methods (such as KVQuant) perform scalar-wise independent quantization, leading to severe accuracy degradation below 2-bit, and fail to optimize the handling of Rotary Position Embedding (RoPE) in keys.

**Key Challenge**: Scalar-wise quantization incurs too much information loss at ultra-low bit-widths; vector-level quantization is required to preserve more information.

**Goal**: Efficient vector-level KV cache compression.

**Key Insight**: Treat the key/value vectors of each token as a whole for additive vector quantization, reducing quantization errors.

**Core Idea**: Design a codebook that commutes with the RoPE matrix, allowing the decoding process to be efficiently embedded into attention computation — where intermediate results can be precomputed on the codebook and reused.

## Method

### Overall Architecture
1. Use Additive Quantization (AQ) to encode key/value vectors as the sum of multiple codewords.
2. Design the codebook to commute with RoPE ($C \cdot R = R \cdot C$), allowing $Q \cdot C$ to be precomputed and reused across all tokens.
3. Train the codebook using the EM algorithm.

### Key Designs

1. **Additive Vector Quantization**:
    - **Function**: Quantize KV vectors into the weighted sum of codewords from multiple codebooks.
    - **Mechanism**: $v \approx c_{i_1} + c_{i_2} + \ldots + c_{i_M}$, where each codeword index requires only $\log_2 K$ bits.
    - **Design Motivation**: Vector-level quantization yields smaller errors than scalar-level quantization at the same bit-width.

2. **RoPE-Commutative Codebook**:
    - **Function**: Design the codebook such that $\text{Decode}(\text{RoPE}(\text{Encode}(k))) = \text{RoPE}(\text{Decode}(\text{Encode}(k)))$.
    - **Mechanism**: The codewords in the codebook do not change the quantization codebook structure under RoPE rotation, enabling $Q \cdot R \cdot C$ to be precomputed and reused.
    - **Design Motivation**: Avoid the $O(N \cdot d)$ overhead of token-by-token decoding and RoPE application, reducing it to $O(K \cdot d)$ (where $K$ is the codebook size).

3. **EM Algorithm Codebook Training**:
    - **Function**: Alternately perform the E-step (assigning codewords) and M-step (updating codebook centroids).
    - **Mechanism**: Minimize the quantization reconstruction error under the RoPE commutativity constraint.
    - **Design Motivation**: A classic approach to vector quantization training with convergence guarantees.

### Loss & Training
- Quantization reconstruction error + RoPE commutativity constraint.
- Triton kernel implementation to achieve actual memory savings.

## Key Experimental Results

### Main Results
LLaMA-3.1 8B long-context benchmarks:

| Method | Bit-width | LongBench | InfiniteBench | Memory Saving |
|------|------|-----------|--------------|---------|
| FP16 | 16-bit | 42.1 | 22.8 | 1× |
| KVQuant | 2-bit | 38.5 | 18.2 | 8× |
| **CommVQ** | **2-bit** | **41.8** | **22.1** | **8×** |
| KVQuant | 1-bit | 28.3 | 11.5 | 16× |
| **CommVQ** | **1-bit** | **36.2** | **17.8** | **16×** |

### Ablation Study

| Configuration | LongBench | Description |
|------|-----------|------|
| Scalar-wise 2-bit | 38.5 | Baseline |
| Vector quantization 2-bit (w/o RoPE commutation) | 40.9 | Advantages of vector quantization |
| Vector quantization 2-bit (+RoPE commutation) | **41.8** | Full method |

### Key Findings
- Near-lossless performance at 2-bit (42.1 $\rightarrow$ 41.8), outperforming all baselines.
- First to achieve usable accuracy at 1-bit (36.2 vs. 42.1 for FP16).
- Direct execution of a 128K context with LLaMA-3.1 8B on a single RTX 4090 (24GB).

## Highlights & Insights
- **RoPE commutativity design** is the core innovation — integrating the mathematical properties of positional embeddings into the quantization scheme.
- The paradigm shift from scalar-wise to vector-level quantization is particularly critical at ultra-low bit-widths.
- 1-bit KV cache makes long-context LLMs practical on consumer-grade GPUs, which holds immense practical significance.

## Limitations & Future Work
- Codebook training requires calibration data.
- RoPE commutativity requires a specific codebook structure, which may limit representation capacity.
- Evaluated only on the LLaMA model family.

## Related Work & Insights
- **vs KVQuant**: Scalar-wise quantization; CommVQ's vector-level quantization is superior.
- **vs any4 (weight quantization)**: any4 uses LUTs for weight quantization, while CommVQ uses vector quantization for KV cache compression, making them complementary.
- Inspires inference optimization for all Transformers using RoPE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The RoPE-commutative codebook design is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, 1-bit/2-bit, and memory analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivation.
- Value: ⭐⭐⭐⭐⭐ Enables long-context LLMs on consumer-grade GPUs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization](star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector.md)
- [\[CVPR 2026\] AstraNav-Memory: Contexts Compression for Long Memory](../../CVPR2026/robotics/astranav-memory_contexts_compression_for_long_memory.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](../../NeurIPS2025/robotics/vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/robotics/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ECCV 2024\] Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation](../../ECCV2024/robotics/decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)

</div>

<!-- RELATED:END -->
