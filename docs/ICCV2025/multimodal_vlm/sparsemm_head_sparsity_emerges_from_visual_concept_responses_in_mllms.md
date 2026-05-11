---
title: >-
  [Paper Note] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs
description: >-
  [ICCV 2025][Multimodal VLM][Visual Head] This paper reveals a "visual head sparsity" phenomenon in Multimodal Large Language Models (MLLMs)…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Head"
  - "MLLM"
  - "KV-Cache Compression"
  - "Head Sparsity"
  - "Attention Analysis"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 8d7040187ebe115e
---

# SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

**Conference**: ICCV 2025
**arXiv**: [2506.05344](https://arxiv.org/abs/2506.05344)
**Code**: [https://github.com/CR400AF-A/SparseMM](https://github.com/CR400AF-A/SparseMM)
**Area**: Multimodal Large Language Models / Model Acceleration / KV Cache Optimization
**Keywords**: Visual Head, MLLM, KV-Cache Compression, Head Sparsity, Attention Analysis, Inference Acceleration

## TL;DR
This paper reveals a "visual head sparsity" phenomenon in Multimodal Large Language Models (MLLMs), where only approximately 5% of attention heads actively participate in visual understanding. It proposes a training-free visual head identification framework based on OCR tasks and introduces SparseMM — an acceleration strategy that asymmetrically allocates KV-Cache budgets across heads according to their visual scores — achieving 1.38× real-time speedup and 52% memory reduction with no performance degradation.

## Background & Motivation

### Problem Definition
MLLMs (e.g., LLaVA, Qwen2-VL) process multimodal inputs by connecting a visual encoder to a pretrained LLM. As the complexity of multimodal inputs grows (high-resolution images, long videos, multi-turn dialogues), the computational and memory overhead of maintaining a full KV-Cache becomes prohibitive.

### Limitations of Prior Work

**General KV-Cache compression (SnapKV/PyramidKV/AdaKV)**: Designed for text-only settings, these methods treat all attention heads uniformly and ignore the special role of visual tokens in MLLMs.

**Visual token pruning (FastV)**: Prunes redundant visual tokens layer-by-layer, but does not consider modality-specificity at the head level.

**Fundamental knowledge gap**: How LLMs acquire visual understanding capability through visual instruction tuning remains insufficiently studied.

### Core Findings
By analyzing the attention mechanisms of MLLMs, the paper identifies two key properties:

**Sparsity**: Even after extensive multimodal training, fewer than 5% of attention heads per layer are "visually active."

**Universality**: Visual heads consistently emerge across different LLM architectures (Vicuna/Qwen2) and attention mechanisms (MHA/GQA).

## Method

### Overall Architecture
SparseMM operates in two stages:
1. **Visual Head Identification**: Using OCR tasks as anchors, the precise text-image correspondences are exploited to quantify the visual relevance of each head in a training-free manner.
2. **Asymmetric KV-Cache Allocation**: Based on visual scores, visual heads receive larger cache budgets while non-visual heads are aggressively compressed.

### Key Design 1: Visual Head Identification Algorithm
OCR is adopted as the anchor task, since text-to-image-region mappings are exact:
- For each output token $y_i$, the corresponding image region is located via (text, bbox) pairs.
- The set of image tokens $I_{y_i}$ covering that region is determined.
- For each attention head $h$: a hit is recorded if the highest attention weight of head $h$ points to tokens within $I_{y_i}$.
- The Visual Score is defined as:

$$\text{Visual Score}(h) = \frac{1}{N}\sum_{i=1}^{N}\frac{\mathbb{I}_{hit(y_i, A_h)}}{\#\text{image\_tokens}}$$

Notably, smaller (more precise) regions yield higher scores. Cumulative scores are computed over 1,000 Synthdog OCR images and then normalized.

### Key Design 2: Three-Component KV-Cache Allocation

Given total budget $B$, $L$ layers, and $H$ heads per layer:

1. **Local Window Cache**: Each head is allocated a fixed local window of $w=32$ recent tokens.
2. **Uniform-Based Cache**: A fraction $\rho=0.1$ of the remaining budget is distributed uniformly across all heads:
   $$r = \frac{\rho \cdot (B - N \cdot w)}{N}$$
3. **Score-Preferred Cache**: The remaining budget is allocated proportionally to visual scores:
   $$b_{ij}^{score} = B_{remain2} \cdot \frac{s_{ij}}{\sum_{i,j} s_{ij}}$$

The final budget per head $(i,j)$ is: $b_{ij} = w + r + b_{ij}^{score}$

### KV-Cache Selection
Following SnapKV, attention scores computed over a trailing observation window of 32 tokens are used to select the top-K KV entries for retention, reducing complexity from $O(N^2)$ to $O(N \times L)$.

## Key Experimental Results

### Experimental Setup
- **Models**: LLaVA-NeXT-Vicuna-7B (MHA, 32 layers × 32 heads), LLaVA-NeXT-Mistral-7B (GQA, 32 layers × 8 KV heads), Qwen2-VL-7B (GQA, 28 layers × 4 KV heads)
- **Baselines**: SnapKV, PyramidKV, AdaKV + Random Head
- **Benchmarks**: DocVQA, OCRBench, TextVQA, ChartQA, TextCaps, MMBench, VQAv2

### Main Results (Accuracy–Speed Tradeoff)

| Method | DocVQA | OCRBench | TextVQA | ChartQA | TextCaps | Latency (ms) |
|------|--------|----------|---------|---------|----------|----------|
| FullKV | 0.68 | 0.52 | 0.65 | 0.55 | 0.73 | 52.9 |
| **SparseMM** | **0.68** | **0.52** | **0.65** | **0.54** | **0.73** | **37.1 (−30%)** |
| SnapKV | 0.64 | 0.46 | 0.62 | 0.50 | 0.65 | 35.3 |
| PyramidKV | 0.65 | 0.48 | 0.62 | 0.53 | 0.65 | 34.9 |
| AdaKV | 0.65 | 0.48 | 0.62 | 0.49 | 0.66 | 37.3 |

KV Cache budget = 256; input length = 16K tokens. SparseMM achieves a 30% latency reduction with negligible accuracy loss.

### Efficiency Evaluation (LLaVA-NeXT-Vicuna-7B, budget = 256)

| Input Length | Speedup | Memory Savings |
|----------|--------|----------|
| 8K tokens | 1.16× | ~2 GB |
| 16K tokens | ~1.5× | ~4 GB |
| 32K tokens | 1.87× | ~15.5 GB (32.87→17.38 GB, ~50%) |

### Ablation Study: Cache Allocation Strategy

| Local Window | Uniform | Score-Preferred | MMBench (512/256/128/96/64/48) |
|:---:|:---:|:---:|------|
| ✓ | ✗ | ✗ | 81.3/80.5/77.3/73.6/70.5/67.2 |
| ✓ | ✓ | ✗ | 81.5/81.4/79.3/77.6/74.6/73.9 |
| ✓ | ✓ | ✓ | 81.5/81.4/81.5/81.4/80.3/77.9 |

All three components are essential. In particular, when $\rho=0$ (relying solely on score-based allocation), performance on the Mistral model collapses from 0.519 to 0.145, demonstrating the importance of guaranteeing a minimum budget for every head.

### Visual Head Masking Experiment
- Masking the top 5% visual heads → ~20% drop on OCRBench, ~15% drop on TextVQA.
- Randomly masking an equivalent fraction of heads → negligible impact (<3%).
- This confirms that visual heads, though sparse, are indispensable.

### Cross-Dataset Robustness of Visual Heads
Visual head distributions identified from OCR datasets (MLT, CTW) are highly consistent with one another, and outperform those identified from detection tasks (COCO), as OCR provides more precise one-to-one mappings.

## Highlights & Insights

1. **Cognitive-level understanding of MLLMs**: This work is the first to systematically reveal the sparse distribution of visual heads in MLLMs — after visual instruction tuning, only a small fraction of heads learn to "see."
2. **Elegance of the OCR anchor task**: The precise text–bbox–image correspondence in OCR provides a lower-noise signal for quantifying visual relevance compared to detection tasks.
3. **Training-free and architecture-agnostic**: Visual scores can be computed offline once and incur zero additional overhead at inference time. Both MHA and GQA architectures are supported.
4. **Random Head ≈ SnapKV**: When head scores are randomly initialized, score-based allocation degenerates to uniform allocation, which is equivalent to SnapKV — this explains the fundamental limitation of SnapKV.
5. **Strong visualization evidence**: Visual heads are shown to genuinely attend to text and object regions in images, whereas non-visual heads scatter attention arbitrarily.

## Limitations & Future Work

1. KV-Cache compression is applied only during the decoding phase; prefill-stage token pruning is not considered.
2. Visual head identification relies on OCR data, and its applicability to non-text-intensive tasks (e.g., 3D understanding) requires further validation.
3. Experiments are conducted exclusively on 7B-scale models; the sparsity ratio and effectiveness on larger models (e.g., 70B+) remain unknown.
4. Evaluation is predominantly single-image; applicability to interleaved multi-image inputs and long-video scenarios requires further exploration.
5. In GQA models, aggregating scores of multiple query heads to represent a single KV head may lose fine-grained information.

## Related Work & Insights

- **Complementarity with FastV**: FastV performs layer-level visual token pruning, while SparseMM performs head-level KV allocation; the two approaches are orthogonal and potentially combinable.
- The visual head sparsity phenomenon may inform MLLM architecture design — specifically, whether training can be guided to encourage more heads to attend to visual inputs.
- The finding shares conceptual similarity with attention sink phenomena (e.g., StreamingLLM), but SparseMM identifies modality-specific "visual sink heads."

## Rating ⭐⭐⭐⭐
- **Novelty**: ⭐⭐⭐⭐ (The discovery of visual head sparsity is original; the OCR-based identification framework is concise and effective.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Covers 7 benchmarks × 3 model architectures, with efficiency evaluation, visualization, and comprehensive ablations.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, tight logical chain from finding to method.)
- **Value**: ⭐⭐⭐⭐⭐ (Training-free, plug-and-play; empirically achieves 50% memory reduction and 30% latency reduction, offering high practical deployment value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](../../ICLR2026/multimodal_vlm/sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)

</div>

<!-- RELATED:END -->
