---
title: >-
  [Paper Note] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation
description: >-
  [ICLR 2026][Image Generation][parallel decoding] This paper proposes Locality-aware Parallel Decoding (LPD), which reduces the number of generation steps for 256×256 images from 256 to 20 by flexibly parallelizing autore…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "parallel decoding"
  - "autoregressive modeling"
  - "spatial locality"
  - "positional query"
  - "efficient inference"
date: 2026-05-08
content_hash: 4d34168c1b03309f
---

# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

**Conference**: ICLR 2026
**arXiv**: [2507.01957](https://arxiv.org/abs/2507.01957)  
**Code**: [GitHub](https://github.com/mit-han-lab/lpd)  
**Area**: Autoregressive Image Generation
**Keywords**: parallel decoding, autoregressive modeling, spatial locality, positional query, efficient inference

## TL;DR
This paper proposes Locality-aware Parallel Decoding (LPD), which reduces the number of generation steps for 256×256 images from 256 to 20 by flexibly parallelizing autoregressive modeling architectures and employing a locality-aware generation order schedule, achieving at least 3.4× latency reduction.

## Background & Motivation
- Next-patch prediction in autoregressive image generation is a memory-bound operation whose latency scales linearly with the number of steps.
- Next-scale prediction (e.g., VAR) requires fewer steps but relies on multi-scale token representations, making it incompatible with flat visual perception models (CLIP, DINO).
- Existing parallelization methods (PAR, RandAR) achieve only limited parallelism: PAR fixes the parallel order, while RandAR generates parallel tokens without mutual visibility.
- There is a clear need for efficient inference that preserves the generality and compatibility of flat token representations.

## Method

### Overall Architecture
LPD consists of two core components: a flexible parallel autoregressive modeling architecture (supporting arbitrary generation orders and parallelism degrees) and a locality-aware generation order schedule (maximizing contextual support while minimizing intra-group dependencies).

### Key Designs

1. **Flexible Parallel Autoregressive Modeling**: The approach decouples context representation from token generation — previously generated tokens supply context (KV cache), while learnable positional query tokens drive the parallel generation of target positions. A dedicated attention mask is employed:

    - Context Attention: subsequent tokens causally attend to context tokens.
    - Query Attention: positional query tokens within the same step attend to each other, but subsequent tokens are not permitted to attend to query tokens.

   During inference, encoding and decoding are fused into a single-step operation, and only the KV cache of generated tokens is stored.

2. **Locality Analysis**: Attention patterns are analyzed on LlamaGen-1.4B, revealing strong spatial locality — the attention of decoded tokens concentrates on spatially nearby tokens. Per-Token Attention (PTA) is defined as:
$$PTA_s = \frac{1}{N}\sum_{i=1}^N \frac{\sum_j \text{Attention}(T_i,T_j) \cdot \mathbb{I}[d(T_i,T_j)=s]}{\sum_j \mathbb{I}[d(T_i,T_j)=s]}$$
   PTA decays sharply with distance, validating two design principles: parallel tokens should be close to already-generated tokens (strong conditioning) and far from tokens within the same group (low dependency).

3. **Locality-aware Generation Order Schedule**: At each step $k$:

    - The Euclidean distance between unselected tokens and selected tokens is computed as a proximity measure.
    - Tokens are ranked by proximity, and a threshold $\tau$ is used to filter a high-proximity candidate set $c_1$.
    - Tokens are greedily selected from $c_1$; after each selection, nearby tokens within a repulsion threshold $\rho$ are filtered out.
    - If the group is not yet full, farthest-point sampling from the remaining set $c_2$ supplies additional tokens.

   Group sizes are typically increased via a cosine schedule, and the generation order can be precomputed.

### Loss & Training
A grouped autoregressive training objective is adopted: $p(x_1,...,x_N;c) = \prod_{g=1}^G p(X_g|X_{<g};c)$
Cross-entropy loss is used, with a dedicated attention mask enabling teacher-forcing and parallel prediction during training.

## Key Experimental Results

### Main Results (ImageNet 256×256)

| Type | Model | Params | FID↓ | IS↑ | #Steps | Latency(s) | Throughput |
|------|-------|--------|------|-----|--------|-----------|-----------|
| AR | LlamaGen-XXL | 1.4B | 2.34 | 253.9 | 576 | 24.40 | 0.72 |
| AR | RAR-XXL | 1.5B | 1.48 | 326.0 | 256 | 6.59 | 6.72 |
| Par.AR | PAR-XXL-4× | 1.4B | 2.35 | 263.2 | 147 | 6.26 | 2.33 |
| Par.AR | RandAR-L | 343M | 2.55 | 288.8 | 88 | 1.97 | 28.59 |
| **Par.AR** | **LPD-L** | **343M** | **2.31** | **284.9** | **20** | **0.40** | **92.42** |
| **Par.AR** | **LPD-XL** | **775M** | **1.97** | **304.0** | **20** | **0.57** | **60.27** |

### ImageNet 512×512

| Model | Params | FID↓ | #Steps | Latency(s) | Throughput |
|-------|--------|------|--------|-----------|-----------|
| LlamaGen-XXL | 1.4B | 2.59 | 1024 | - | - |
| **LPD-XXL** | **1.4B** | **2.25** | **48** | **2.78** | **6.56** |

### Key Findings
- LPD-L generates 256×256 images in only 20 steps, achieving FID=2.31, outperforming LlamaGen-XXL (2.34) which requires 576 steps.
- Throughput of 92.42 img/s substantially exceeds RandAR (28.59) and PAR (6.83).
- 512×512 generation requires only 48 steps (vs. 1024), with FID improving from 2.59 to 2.25.
- The locality-aware schedule significantly outperforms raster, random, and Halton orderings.
- Zero-shot image editing (class-conditional editing, inpainting, outpainting) is naturally supported.

## Highlights & Insights
- The decoupling design via positional query tokens elegantly resolves the flexibility limitations of standard decoder-only models.
- Query Attention ensures that tokens generated within the same step are mutually visible, preventing inconsistencies arising from independent sampling.
- The locality analysis provides an empirical foundation for designing parallelization strategies — PTA analysis is transferable to other visual autoregressive models.
- Flat token representations are preserved compared to VAR, maintaining compatibility with visual backbones such as CLIP and DINO.

## Limitations & Future Work
- Validation is currently limited to ImageNet class-conditional generation; extension to text-guided generation has not been explored.
- The positional query tokens introduce additional parameters and attention computation overhead.
- Hyperparameters of the generation order schedule ($\tau$, $\rho$, group size schedule) require tuning.
- A gap in FID remains relative to the best MAR/VAR methods, though throughput is substantially superior.

## Related Work & Insights
- The limitations of parallel autoregressive methods such as PAR, RandAR, and SAR motivate this work.
- MaskGIT's masked prediction inspires the design of progressively increasing group sizes.
- The spatial locality observation provides insights into understanding attention mechanisms in visual autoregressive models.
- This work offers an efficient solution for the image generation component in unified multimodal generation (text + image) systems.

## Technical Details
- Group sizes increase via a cosine schedule: fewer tokens are generated in early steps when context is scarce, with more generated in later steps.
- Positional query tokens = shared learnable embeddings + positional encodings of target positions.
- During inference, KV representations of query tokens are not stored; only those of generated tokens are cached.
- 256×256 generation completes in 20 steps; 512×512 in 48 steps.
- Zero-shot image editing is supported (class-conditional editing, inpainting, outpainting).
- LPD-L with 343M parameters achieves FID=2.31, surpassing LlamaGen-XXL with 1.4B parameters.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of positional query decoupling and locality-aware scheduling is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparisons are thorough, but text-to-image and multimodal experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, with thorough comparative analysis against alternative methods.
- Value: ⭐⭐⭐⭐⭐ Substantially reduces latency in autoregressive image generation, with significant implications for unified multimodal systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)
- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICLR 2026\] SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation](serum_simple_efficient_robust_and_unifying_marking_for_diffusion-based_image_gen.md)

</div>

<!-- RELATED:END -->
