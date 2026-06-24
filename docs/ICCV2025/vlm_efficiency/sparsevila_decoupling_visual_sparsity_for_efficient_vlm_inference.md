---
title: >-
  [Paper Note] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference
description: >-
  [ICCV 2025][Multimodal Efficiency][VLM] This paper proposes SparseVILA—the first VLM inference acceleration framework that decouples visual sparsity between the prefill and decode stages: query-agnostic redundant token pruning during prefill, and query-aware relevant token retrieval during decode. The approach achieves up to 4.0× prefill speedup, 2.5× decode throughput improvement, and 2.6× end-to-end acceleration, while maintaining accuracy in multi-turn conversation setting…
tags:
  - "ICCV 2025"
  - "Multimodal Efficiency"
  - "VLM"
  - "Token Pruning"
  - "KV-Cache"
  - "Decoupled Sparsity"
  - "Multi-turn Conversation"
  - "Prefill-Decode"
date: 2026-05-08
content_hash: 152f2ab59bb7d059
---

# SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference

**Conference**: ICCV 2025
**arXiv**: [2510.17777](https://arxiv.org/abs/2510.17777)  
**Code**: Not released (built on TinyChat + AWQ inference framework)  
**Area**: Multimodal Large Models / Inference Acceleration / Token Pruning
**Keywords**: VLM, Token Pruning, KV-Cache, Decoupled Sparsity, Multi-turn Conversation, Prefill-Decode

## TL;DR
This paper proposes SparseVILA—the first VLM inference acceleration framework that decouples visual sparsity between the prefill and decode stages: query-agnostic redundant token pruning during prefill, and query-aware relevant token retrieval during decode. The approach achieves up to 4.0× prefill speedup, 2.5× decode throughput improvement, and 2.6× end-to-end acceleration, while maintaining accuracy in multi-turn conversation settings where existing methods suffer severe degradation due to permanent token deletion.

## Background & Motivation

### Problem Definition
In VLMs (e.g., LLaVA, Qwen2-VL), visual tokens account for 90%–99% of the input sequence when processing high-resolution images, long videos, and multi-turn dialogues, dominating inference latency and memory consumption. Accelerating inference without accuracy loss is therefore a critical challenge.

### Limitations of Prior Work

**Query-agnostic pruning (PruMerge/VisionZip)**:
- Prunes tokens based solely on visual saliency/redundancy, without considering the text query
- Loses fine-grained visual detail at high sparsity
- Cannot adapt to query-specific information demands

**Query-aware pruning (FastV/SparseVLM/PDrop)**:
- Selects tokens using query-vision attention scores
- **Fatal issue in multi-turn settings**: tokens pruned for the first query cannot be recovered; subsequent queries requiring those tokens suffer severe accuracy degradation
- Experiments show that even a "greedy oracle" (selecting optimal token subsets using ground-truth answers) leads to serious multi-turn performance degradation

**Latency distribution mismatch**:
- For image tasks, the decode stage accounts for 50–70% of total latency
- For video tasks, the decode stage accounts for 70–90% of total latency
- Existing methods primarily optimize prefill, overlooking the true latency bottleneck

### Core Insight
Visual sparsity should not be applied uniformly across the entire inference pipeline—prefill and decode have distinct computational characteristics and functional requirements. Decoupling sparsity achieves the best of both worlds: prefill retains sufficient coverage, while decode aggressively retrieves tokens relevant to the current query.

## Method

### Overall Architecture
SparseVILA applies different types of sparsity at each inference stage:

**Prefill stage** (executed once to build the multimodal context):
- Query-agnostic pruning: estimates token saliency using self-attention from the visual encoder
- Removes redundant tokens while retaining sufficient visual coverage to support subsequent multi-turn dialogues
- Typical sparsity: 45%–75%

**Decode stage** (token-by-token generation, dominates latency):
- Query-aware retrieval: selects visual tokens in the KV Cache most relevant to the current query
- Inactive tokens are retained in the cache for future turns
- Typical sparsity: 75%–95%

### Key Design 1: Query-Agnostic Pruning in Prefill

**Token saliency estimation**:
- Encoders with summary tokens (e.g., CLIP): each token's attention contribution to the summary token
- Encoders with multiple summary tokens (e.g., RADIO): average attention over all summary tokens
- Encoders without summary tokens (e.g., SigLIP/QwenVL): average self-attention across all token pairs

**Efficient implementation**:
A custom Triton kernel computes softmax and saliency accumulation in a streaming fashion, without explicitly constructing the full attention matrix:
- 3× speedup for SigLIP encoder
- 10× speedup for QwenVL encoder

### Key Design 2: Query-Aware Retrieval in Decode

Before decode begins, the relevance of each visual token to the current query is computed:
- Measures attention strength between query embeddings and visual KV Cache
- Retains the top-scoring token subset for decode attention computation
- Low-scoring tokens are **not deleted**—they remain in the cache for re-retrieval in subsequent turns

**Key distinction**: This is not permanent pruning but a "soft selection"—each conversational turn may select a different token subset.

**Triton kernel optimization**: Executes in parallel with FlashAttention2's prefill path, achieving 1.5× speedup.

### Key Design 3: Positional Encoding Handling

Different VLMs employ different positional encoding strategies:
- **Unified RoPE (e.g., LLaVA-NeXT)**: retains contiguous positional indices for pruned visual tokens
- **Multimodal RoPE (e.g., Qwen2.5-VL)**: reconstructs the minimal contiguous positional grid along temporal/height/width dimensions, then shifts subsequent text positions accordingly

### Key Design 4: Multi-turn Evaluation Protocol

Existing benchmarks are found to suffer from information leakage (Q1 reveals the answer to Q2). A KV Cache partial eviction strategy is designed: at the end of each turn, only the KV entries from the previous Q&A round are removed, while the visual KV cache is preserved.

## Key Experimental Results

### Inference Setup
- Quantization baseline: visual encoder W8A8 (SmoothQuant) + LLM W4A16 (AWQ), yielding a 2.4× baseline speedup
- All results are reported on top of this quantization baseline
- Hardware: single NVIDIA A6000 GPU

### Image Benchmark Results (LLaVA-NeXT-7B)

| Method | Prefill Sparsity | Decode Sparsity | E2E Speedup | AI2D | DocVQA | GQA | POPE | TextVQA |
|--------|:----------------:|:---------------:|:-----------:|------|--------|-----|------|---------|
| No compression | 0% | 0% | 1.0× | 63.9 | 63.6 | 63.5 | 84.5 | 58.2 |
| FastV | 80% | 0% | 1.2× | 61.8 | 33.5 | 55.3 | 76.7 | 52.7 |
| SparseVLM | 75% | 0% | 1.2× | 63.2 | 41.8 | 59.7 | 83.4 | 57.6 |
| VisionZip | 80% | 0% | 1.2× | 62.9 | 48.5 | 60.3 | 84.1 | 57.1 |
| **SparseVILA** | **60%** | **75%** | **1.2×** | **64.1** | **58.0** | **62.7** | **85.8** | **59.1** |

Key finding: At the same speedup ratio, SparseVILA outperforms FastV by 24.5 points and VisionZip by 9.5 points on DocVQA. On GQA, POPE, and TextVQA, it even surpasses the uncompressed baseline.

### Video Understanding Benchmark Results

| Model (frames) | Prefill | Decode | E2E Speedup | LVB | MLVU | NExT-QA | Video-MME |
|----------------|:-------:|:------:|:-----------:|-----|------|---------|-----------|
| LongVILA-7B (256f) No compression | — | — | 1.0× | 53.8 | 64.9 | 78.6 | 58.8 |
| + VisionZip 95% | 0.9× | 1.5× | 2.1× | 47.0 | 60.4 | 75.5 | 52.2 |
| + PruMerge 95% | 0.9× | 1.5× | 2.1× | 47.9 | 60.9 | 75.7 | 52.0 |
| **+ SparseVILA 75%/90%** | 1.0× | **1.6×** | **2.1×** | **54.1** | **65.3** | **79.0** | **58.7** |

Key finding: On video tasks, SparseVILA even **surpasses the uncompressed baseline** (e.g., MLVU 65.3 vs. 64.9), as more precise token retrieval directs the model's attention toward the most semantically important visual cues.

### Decoupled Sparsity Ablation

| Prefill Sparsity | Decode Sparsity | Prefill Speedup | Decode Speedup | E2E Speedup | RoboVQA |
|:----------------:|:---------------:|:---------------:|:--------------:|:-----------:|---------|
| 0% | 0% | 1.0× | 1.0× | 1.0× | 86.4 |
| 90% | 0% | 14.6× | 1.1× | 1.4× | 80.0 |
| **70%** | **85%** | **4.9×** | **1.2×** | **1.4×** | **89.1** |

Key finding: At the same 1.4× end-to-end speedup, redistributing sparsity from prefill-only (90%) to a decoupled setting (70%/85%) raises RoboVQA from 80.0 to 89.1—surpassing even the uncompressed baseline (86.4).

### Functional Analysis of Retrieved Tokens

Retrieved tokens in SparseVILA exhibit two distinct roles:
1. **Visual Attention Sinks**: anchor tokens that are consistently activated across queries, maintaining attention stability
2. **Visual Retrieval Tokens**: semantically relevant tokens that vary dynamically with the query, capturing task-specific information

## Highlights & Insights

1. **Paradigm innovation through Prefill-Decode decoupling**: This work is the first to explicitly argue that the two stages of VLM inference require fundamentally different sparsity strategies—a simple yet profound insight that challenges the prevailing "unified compression" paradigm.
2. **Principled solution for multi-turn dialogue**: Query-aware pruning is inherently irreversible (the oracle experiment demonstrates that even the upper bound is poor), whereas SparseVILA's "soft retrieval" design preserves all information—only the active subset changes per turn.
3. **"Less is more" effect**: On video tasks, sparse inference outperforms full inference—analogous to findings in StreamingLLM, where selectivity enhances performance.
4. **Engineering completeness**: Custom Triton kernels handle saliency computation and cache compaction, with end-to-end speedups measured empirically rather than estimated from theoretical FLOP reduction.
5. **RoPE compatibility**: Separate positional reconstruction strategies are designed for unified and multimodal RoPE, ensuring cross-modal positional consistency.

## Limitations & Future Work

1. **Fixed sparsity ratios for prefill and decode**: Adaptive per-layer or per-head sparsity strategies are not explored and may offer further optimization potential.
2. **Accuracy degradation in document understanding**: Although significantly better than prior methods, DocVQA still shows a ~5.6-point drop (58.0 vs. 63.6), as every detail may be critical in document-heavy scenarios.
3. **Dependence on visual encoder attention maps**: The method is inapplicable to black-box encoders that do not expose attention maps.
4. **Single-GPU evaluation only**: Results are reported under batch=1; effectiveness in distributed or multi-batch settings is unknown.
5. **Combined effect of quantization and sparsity**: All results build upon an AWQ quantization baseline; the isolated effect of sparsity alone requires additional validation.

## Related Work & Insights

- **Complementarity with SparseMM**: SparseMM allocates asymmetric budgets from a per-head perspective, while SparseVILA decouples sparsity from a per-stage perspective—the two approaches are combinable.
- The **Prefill-Decode decoupling** paradigm generalizes beyond multimodal models to general LLM inference acceleration.
- The multi-turn evaluation protocol (KV partial eviction to prevent information leakage) is itself a methodological contribution worthy of adoption in evaluation frameworks.
- The dual-role finding of Visual Attention Sinks and Visual Retrieval Tokens is consistent with observations in VisionZip and VAR, pointing to a universal structural property of VLM attention.

## Rating ⭐⭐⭐⭐⭐
- **Novelty**: ⭐⭐⭐⭐⭐ (Prefill-Decode decoupling is a paradigm-level contribution; the multi-turn dialogue insight is particularly deep)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (9 image benchmarks + 4 video benchmarks + multiple models + multi-turn evaluation + empirical end-to-end measurement—exceptionally comprehensive)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (The motivation–method–experiment logical chain is exceptionally clear; the oracle experiment is highly convincing)
- **Value**: ⭐⭐⭐⭐⭐ (Training-free, architecture-agnostic, 2.6× end-to-end speedup—directly deployable in production systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SegMo: Co-Designing Content-Aware Sparsity and Locally-Cohesive Segment Parallelism for Efficient VLM Inference](../../CVPR2026/vlm_efficiency/segmo_co-designing_content-aware_sparsity_and_locally-cohesive_segment_paralleli.md)
- [\[ICCV 2025\] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/vlm_efficiency/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[NeurIPS 2025\] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](../../NeurIPS2025/vlm_efficiency/hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)
- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/vlm_efficiency/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)

</div>

<!-- RELATED:END -->
