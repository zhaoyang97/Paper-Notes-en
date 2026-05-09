---
title: >-
  [Paper Note] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Token Compression] This paper proposes PPE (Positional Preservation Embedding), which exploits the dimensional independence of rotations in RoPE to encode multiple original position IDs from merged tokens into distinct dimension segments, enabling a single compressed token to carry multiple spatial/temporal positional cues. PPE is a zero-parameter, plug-and-play operator that achieves an average performance drop of only 3.6% on image tasks at 55% compression, and maintains comparable performance at 90% compression via cascaded compression.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Token Compression
  - Positional Encoding
  - RoPE
  - MLLM Efficiency
  - Spatiotemporal Preservation
date: 2026-05-08
content_hash: bdd151f2dca5b390
---

# PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.22936](https://arxiv.org/abs/2510.22936)
**Code**: [GitHub](https://github.com/MouxiaoHuang/PPE)
**Area**: Multimodal VLM / Efficiency
**Keywords**: Token Compression, Positional Encoding, RoPE, MLLM Efficiency, Spatiotemporal Preservation

## TL;DR

This paper proposes PPE (Positional Preservation Embedding), which exploits the dimensional independence of rotations in RoPE to encode multiple original position IDs from merged tokens into distinct dimension segments, enabling a single compressed token to carry multiple spatial/temporal positional cues. PPE is a zero-parameter, plug-and-play operator that achieves an average performance drop of only 3.6% on image tasks at 55% compression, and maintains comparable performance at 90% compression via cascaded compression.

## Background & Motivation

**Background**: MLLMs (e.g., Qwen2.5-VL, LLaVA-OneVision) encode images/videos into dense visual tokens before feeding them into an LLM for joint understanding. However, dense representations are highly redundant—a high-resolution image can produce thousands of visual tokens, imposing substantial computational and memory costs. Token merging/pruning techniques reduce sequence length by clustering similar tokens.

**Limitations of Prior Work**:
1. **ChatUniVi**: After clustering, it assigns **randomized position IDs** to compressed tokens → completely discards the original spatial layout, causing significant performance degradation on layout-sensitive tasks (e.g., counting, OCR, temporal localization).
2. **PACT**: Retains only the position ID of the cluster center → each merged token has a single position → positional information is insufficient and imprecise.
3. **General Issue**: Higher compression ratios → each merged token represents more original tokens → a single position ID loses more layout information.

**Key Challenge**: An inherent conflict between the high compression ratio pursued by token merging and the preservation of positional information—merging reduces token count while simultaneously erasing spatial/temporal structure.

**Goal**: The paper observes that the rotary encoding in RoPE/M-RoPE operates independently per dimension ($\text{RoPE}(z_d, m) = e^{im\theta_d}z_d$), so different dimensions of the same token can encode different position IDs. The dimensions are divided into $K$ groups, each encoding the position of one merged token within the cluster → a single compressed token simultaneously carries $K$ positional cues.

## Method

### Overall Architecture

PPE workflow:
1. Apply clustering-based token compression (e.g., DPC-KNN from ChatUniVi) to visual tokens.
2. For each cluster, select the position IDs of the top-$K$ tokens nearest to the cluster center.
3. Divide the $D$-dimensional embedding into $K$ groups; group $k$ encodes the $k$-th position ID.
4. During attention computation, compressed tokens perceive multiple relative positional relationships via PPE.

### Key Design 1: PPE Positional Encoding

**1D RoPE scenario** (spatial only):

$$\hat{m}_d = m_{k,d}, \quad d = (k-1)\frac{D}{K}+1, \ldots, k\frac{D}{K}$$

**3D M-RoPE scenario** (spatiotemporal): Dimension $D$ is partitioned as $[D_1, D_2, D_3]$ across the temporal/height/width axes, with each axis further divided into $K$ groups:

$$\hat{m}_d^{3D} = m_{k,d}^{3D}$$

$K$ is set to the GCD of the M-RoPE section sizes (e.g., GCD of $[16,24,24]$ is 8).

**Core Idea**: Similar tokens can share feature embeddings (via average merging); by the same logic, they should also share positional information—PPE extends this idea from the feature dimension to the positional dimension.

### Key Design 2: Cascaded Compression

PPE naturally supports multi-level progressive compression:

1. Apply the first PPE compression between the visual encoder and the LLM.
2. Insert PPE compression modules at layers 11/23/35 of the LLM (36-layer Qwen2.5-VL-3B).
3. At each compression stage, PPE recomputes cluster center positions and selects top-$K$ IDs.

Benefits of cascaded compression:
- Early layers retain low-level semantics; aggressive compression is deferred to deeper layers → avoids premature collapse.
- A per-layer compression ratio of 0.45 achieves 90% total compression.
- PPE preserves positional information at each stage, maintaining layout fidelity even at extreme compression rates.

### Key Design 3: Zero-Parameter Plug-and-Play

Key properties of PPE:
- **Zero parameters**: No trainable parameters are introduced; only position IDs are manipulated.
- **Plug-and-play**: Seamlessly integrates into any token merging framework (ChatUniVi, PACT, ToMe, etc.).
- **Negligible overhead**: The cost of position ID selection and assignment is computationally insignificant.

## Key Experimental Results

### Main Results: Comprehensive Image and Video Benchmark Comparison

Based on Qwen2.5-VL-3B-Instruct, comparing Dense (no compression), Chat-UniVi, and PPE:

| Benchmark | Dense (0%) | Chat-UniVi (55%) | PPE (55%) | Δ (PPE vs ChatUniVi) |
|:---|:---:|:---:|:---:|:---:|
| MMBench (EN) | 85.89 | 84.92 | 84.73 | -0.19 |
| MMBench (CN) | 86.07 | 83.71 | **84.87** | **+1.16** |
| TextVQA | 79.50 | 57.66 | **77.14** | **+19.48** |
| DocVQA | 89.44 | 52.48 | **76.79** | **+24.31** |
| ChartQA | 79.96 | 49.60 | **74.52** | **+24.92** |
| VideoMME (w/o) | 57.81 | 57.22 | **58.70** | **+1.48** |
| MVBench | 67.90 | 66.90 | **67.38** | **+0.48** |

**Key Findings**:
- Remarkable gains on layout-sensitive tasks such as TextVQA/DocVQA/ChartQA (+19–25%), demonstrating that positional preservation is critical for OCR and document understanding.
- Differences on general visual understanding benchmarks (MMBench) are modest, yet PPE still outperforms Chat-UniVi.
- On video tasks, PPE at 55% compression even surpasses the Dense baseline.

### Ablation Study: Cascaded Compression and Cross-Framework Compatibility

| Method | MMBench (EN) | TextVQA | Compression Ratio |
|:---|:---:|:---:|:---:|
| PACT | 74.14 | 73.73 | 89% |
| PACT + PPE | **74.48** | **73.87** | 89% |
| ToMe | 74.31 | 74.94 | 57% |
| ToMe + PPE | **74.57** | **76.16** | 57% |

PPE yields consistent improvements on both PACT and ToMe frameworks, validating its plug-and-play generality.

### Comparison with State-of-the-Art MLLMs

| Model | VideoMME | MVBench | MMBench | TextVQA | Compression |
|:---|:---:|:---:|:---:|:---:|:---:|
| InternVL2.5-4B | 62.30 | 71.60 | 81.10 | 76.80 | 0% |
| Qwen2.5-VL-3B | 61.50 | 67.00 | 79.10 | 79.30 | 0% |
| PACT-7B | 57.60 | - | 80.30 | 75.00 | 67% |
| **PPE-3B** | **58.70** | **67.38** | **84.78** | **77.08** | **55%** |
| **PPE*-3B (cascaded)** | **58.48** | **67.35** | - | - | **90%** |

PPE-3B, using only 3B parameters at 55% compression, surpasses 7B PACT and 4B InternVL2.5 on MMBench.

## Highlights & Insights

### Strengths

1. **Novel and insightful core idea**: Exploiting RoPE dimensional independence to encode multiple positions is theoretically well-grounded and elegantly simple in implementation.
2. **Zero-parameter plug-and-play**: No additional training cost; directly integrable into existing frameworks with high practical utility.
3. **Significant gains on layout-sensitive tasks**: Improvements exceeding 20% on TextVQA/DocVQA confirm that positional preservation is indeed the key bottleneck in token compression.

### Limitations & Future Work

1. The value of $K$ is constrained by the GCD of the M-RoPE sections (e.g., $[16,24,24]$ → $K=8$), limiting flexibility.
2. When the number of tokens in a cluster is less than $K$, the IDs of high-weight tokens must be repeated as padding, reducing information content.
3. Thorough validation is limited to Qwen2.5-VL; compatibility with other architectures (LLaVA, InternVL) requires further investigation.

### Rating

⭐⭐⭐⭐

**Rationale**: The paper identifies a core bottleneck in token compression (loss of positional information) and proposes an elegant solution. The mapping from RoPE dimensional independence to multi-position encoding is natural, and the zero-parameter plug-and-play design confers high practical value. The substantial gains on layout-sensitive tasks further validate the method's effectiveness.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](../../CVPR2026/multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](directional_embedding_smoothing_for_robust_vision_language_models.md)

<!-- RELATED:END -->
