---
title: >-
  [Paper Note] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling
description: >-
  [AAAI 2026][Image Generation][KV Cache Compression] This paper identifies that attention heads in VAR models naturally fall into two categories — Contextual Heads (semantic consistency…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "KV Cache Compression"
  - "Visual Autoregressive Models"
  - "VAR"
  - "Attention Head Classification"
  - "Next-Scale Prediction"
date: 2026-05-08
content_hash: f179f04cb2acdf28
---

# HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling

**Conference**: AAAI 2026  
**arXiv**: [2504.09261](https://arxiv.org/abs/2504.09261)  
**Code**: [https://github.com/Zr2223/HACK](https://github.com/Zr2223/HACK)  
**Area**: Image Generation / Model Compression  
**Keywords**: KV Cache Compression, Visual Autoregressive Models, VAR, Attention Head Classification, Next-Scale Prediction  

## TL;DR
This paper identifies that attention heads in VAR models naturally fall into two categories — Contextual Heads (semantic consistency, vertical attention patterns) and Structural Heads (spatial coherence, multi-diagonal patterns) — and proposes the HACK framework, which employs asymmetric budget allocation and pattern-specific compression strategies to achieve lossless generation quality at 70% compression, yielding 1.75× memory reduction and 1.57× speedup on Infinity-8B.

## Background & Motivation
VAR models adopt a next-scale prediction paradigm that generates high-quality images in far fewer steps than traditional next-token autoregressive models. However, the KV cache in VAR accumulates across scales, resulting in attention complexity of $O(n^4)$, requiring the processing of 10k+ tokens to generate a 1024×1024 image. Existing KV cache compression methods for LLMs (e.g., H2O, SnapKV, CAKE) perform poorly when directly applied to VAR, as they employ a one-size-fits-all strategy that ignores the functional heterogeneity of different attention heads in VAR.

## Core Problem
How can KV cache be efficiently compressed within VAR's next-scale generation paradigm to significantly reduce memory and computational overhead without degrading generation quality? The key challenge is that VAR attention heads exhibit two functionally distinct roles and attention patterns, and naïve uniform compression disrupts one of the two categories.

## Method

### Overall Architecture
The framework proceeds in three steps: (1) offline head classification (distinguishing Contextual/Structural heads via attention variance) → (2) asymmetric budget allocation (assigning larger budgets to compression-sensitive Structural heads) → (3) pattern-specific compression strategies (applying distinct eviction/merge policies for each head type). The method is training-free and requires only 50 samples for offline classification.

### Key Designs
1. **Head Classification (Contextual vs. Structural)**: Heads are distinguished by computing the column-wise variance of the attention matrix. Contextual heads attend to a small number of semantically critical tokens, yielding low variance; Structural heads attend dynamically based on spatial position, yielding high variance. The variance distribution exhibits a long-tail property with a natural decision boundary. Classification results are highly stable across samples and scales (even a single sample suffices), indicating that this is an intrinsic property of the model. Functional validation shows that masking Contextual heads causes semantic drift while preserving structure, whereas masking Structural heads preserves semantics but severely distorts spatial coherence.

2. **Asymmetric Budget Allocation**: $B = \alpha B_C + (1-\alpha) B_S$, where a smaller budget is assigned to Contextual heads ($B_C \ll B_S$), since they attend to only a few critical tokens and are robust to compression (maintaining quality at 90% compression). Structural heads degrade at compression ratios above 50% and thus require larger cache budgets. Since the proportion of each head type varies per layer, this naturally yields a layer-adaptive effect.

3. **Pattern-Specific Compression Strategies**: Contextual heads use cumulative attention top-K selection with a final-step merge-and-discard strategy to preserve semantic information. Structural heads use a scale-aware strategy — unconditionally retaining the first two scales (initial global context) and the most recent scale (current detail), while using attention-based selection for intermediate scales. This is inspired by the sink token phenomenon in LLMs, where initial and recent tokens tend to be most important.

4. **Efficient Subset Attention**: Rather than using all queries to estimate token importance, $N_{obs}=32$ queries are uniformly sampled as an approximation, with negligible overhead.

### Loss & Training
The method is entirely training-free. Offline classification requires only 50 samples and a few minutes. At deployment, head order is statically rearranged to group heads by type, enabling efficient inference.

## Key Experimental Results

| Model/Task | Method | Compression | GenEval↑ | HPSv2.1↑ | ImageReward↑ | FID↓ |
|--------|------|------|---------|---------|--------------|------|
| Infinity-2B T2I | Vanilla | 0% | 0.946 | 30.49 | 0.68 | 10.34 |
| Infinity-2B T2I | H2O | 70% | 0.910 | 29.60 | 0.68 | 10.68 |
| Infinity-2B T2I | SnapKV | 70% | 0.904 | 29.60 | 0.68 | 10.60 |
| Infinity-2B T2I | **HACK** | **70%** | **0.933** | **30.18** | **0.68** | **10.56** |
| Infinity-8B T2I | Vanilla | 0% | 1.049 | 30.99 | 0.81 | 8.75 |
| Infinity-8B T2I | **HACK** | **70%** | **1.043** | **30.69** | **0.82** | **8.62** |
| VAR-d30 Class | Vanilla | 0% | - | - | - | 1.92 (FID) |
| VAR-d30 Class | H2O | 50% | - | - | - | 3.04 |
| VAR-d30 Class | **HACK** | **50%** | - | - | - | **2.06** |
| VAR-d30 Class | **HACK** | **70%** | - | - | - | **2.78** |

Efficiency: Infinity-8B achieves 1.75× memory reduction (60.42→34.44 GB) and 1.57× speedup (8.14→5.17s). At 1024 resolution, HACK exhibits linear scaling while Vanilla grows exponentially, yielding up to 5.8× speedup in extreme cases.

### Ablation Study
- Both asymmetric allocation and pattern-specific compression contribute significantly (Table 4); neither alone is sufficient.
- Swapping strategies (applying Contextual strategy to Structural heads) causes substantial performance degradation (ImageReward 0.859 vs. 0.933), validating the necessity of pattern-specific design.
- Head classification method comparison: variance-based >> Order/Uniform/Random (FID 2.06 vs. 2.57/2.63/2.70).
- Classification is insensitive to sample size (consistent results from 1 to 100 samples).
- Query subset sampling at $N_{obs}=32$ approaches the accuracy of full-attention importance estimation.

## Highlights & Insights
- **The discovery of Contextual vs. Structural heads is a genuinely novel contribution** — it reveals the intrinsic functional division of labor in VAR attention, which is distinct from head analyses in LLMs.
- **Functional validation experiments are highly intuitive** — selective masking clearly demonstrates the complementary roles of the two head types.
- Near-lossless and sometimes superior performance at 70% compression indicates substantial redundancy in VAR models.
- Reducing complexity from $O(n^4)$ to $O(Bn^2)$ represents a theoretically significant improvement.
- The approach resonates conceptually with the "micro-expert" notion in CAMERA — both discover functional heterogeneity within transformer components and design differentiated strategies accordingly.

## Limitations & Future Work
- Only the attention module is optimized; FFN overhead is not addressed.
- The head ratio $\alpha$ requires manual tuning, though results are not highly sensitive to it.
- Integration with quantization methods (KV cache quantization + HACK) has not been explored, though combined gains may be possible.
- Validation is limited to VAR models and has not been extended to traditional next-token autoregressive generation models.

## Related Work & Insights
- **vs. H2O/SnapKV**: These general-purpose KV compression methods do not distinguish head types and perform poorly on VAR (FID 3.04/3.09 vs. HACK 2.06 @50%), as they disrupt Structural heads.
- **vs. LOOK-M/MEDA**: Merging-based methods exhibit the worst degradation on VAR (FID 6.89/18.88), as merge operations destroy spatial structural information.
- **vs. StreamingLLM**: Position-based methods fail to capture the semantic importance differences among tokens in VAR.
- **Strong resonance with CAMERA's "micro-expert" concept** — CAMERA analyzes the heterogeneity of micro-experts within MoE, while HACK analyzes functional heterogeneity of attention heads; both exploit this heterogeneity for differentiated compression.
- **Complementary to the "U-shaped pattern" in Distillation Dynamics** — the Contextual/Structural distinction in HACK essentially reflects a division between information compression (semantic aggregation) and information preservation (spatial structure).
- The framework may be extensible to KV cache compression in MLLMs, where a similar Contextual vs. Structural divergence among heads may exist.
- Related to cross-layer token budget allocation ideas; HACK provides a head-level budget allocation perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Contextual/Structural head discovery and VAR-specific KV compression constitute entirely original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 VAR models, T2I and class-conditional tasks, multiple compression ratios, detailed ablations, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation visualization is excellent; the analysis→design→validation logic is rigorous throughout.
- Value: ⭐⭐⭐⭐⭐ The first KV cache compression work for VAR models; demonstrates significant practical speedup and high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](../../ICML2026/image_generation/visual_implicit_autoregressive_modeling.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](../../CVPR2026/image_generation/depthvar_depth_adaptive_var.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](../../ICLR2026/image_generation/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICLR 2026\] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning](../../ICLR2026/image_generation/mvar_visual_autoregressive_modeling_with_scale_and_spatial_markovian_conditionin.md)
- [\[AAAI 2026\] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models](hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)

</div>

<!-- RELATED:END -->
