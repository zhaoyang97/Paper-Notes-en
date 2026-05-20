---
title: >-
  [Paper Note] BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs
description: >-
  [ACL 2026][LLM Efficiency][Sliding Window Attention] BOSCH is a training-free head-level SWA mixing method that models SWA head selection as a large neighborhood search problem decomposed into three stages (layer importa…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Sliding Window Attention"
  - "Attention Head Selection"
  - "Black-Box Optimization"
  - "Large Neighborhood Search"
  - "KV-Cache"
content_hash: 3865638624679721
---

# BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs

**Conference**: ACL 2026
**arXiv**: [2604.05942](https://arxiv.org/abs/2604.05942)  
**Code**: N/A  
**Area**: LLM Efficiency / Attention Optimization
**Keywords**: Sliding Window Attention, Attention Head Selection, Black-Box Optimization, Large Neighborhood Search, KV-Cache

## TL;DR
BOSCH is a training-free head-level SWA mixing method that models SWA head selection as a large neighborhood search problem decomposed into three stages (layer importance probing → adaptive ratio allocation → grouped head selection), systematically outperforming layer-level heuristics and 6 static head-level methods across 4 models and 4 ratio settings.

## Method

### Key Designs

1. **Stage 1: Layer Importance Probing**: Evaluates each layer's sensitivity to head localization via cascaded small-budget black-box search.

2. **Stage 2: Adaptive Ratio Allocation**: Differentially allocates per-layer SWA ratios based on sensitivity, mapping layers into coarse-grained buckets.

3. **Stage 3: Multi-Layer Head Selection**: Groups layers sharing the same ratio and jointly optimizes head binary decisions within each group.

## Key Experimental Results

| Method | ρ=0.25 | ρ=0.5 | ρ=0.75 | ρ=0.875 |
|--------|--------|-------|--------|---------|
| BOSCH (8B) | 98.9 | 90.3 | 72.7 | 42.5 |
| Fisher (best baseline) | 94.2 | 89.3 | 63.4 | 29.0 |

## Highlights & Insights
- Large neighborhood search decomposition elegantly breaks N-dimensional binary optimization into three tractable subproblems
- "Entanglement problem" discovery: optimal head sets differ significantly across SWA ratios, proving static ranking methods insufficient

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[NeurIPS 2025\] From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](../../NeurIPS2025/llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ICLR 2026\] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](../../ICLR2026/llm_efficiency/tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)

</div>

<!-- RELATED:END -->
