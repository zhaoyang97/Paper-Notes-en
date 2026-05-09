---
title: >-
  [Paper Note] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models
description: >-
  [ICLR 2026][Image Generation][Compositional Alignment] This paper presents the first systematic comparison of Visual Autoregressive (VAR) models and diffusion models on compositional text-image alignment. Evaluating 6 T2I models across T2I-CompBench++ and GenEval benchmarks, it finds that Infinity-8B achieves state-of-the-art performance on nearly all compositional dimensions, demonstrating a clear architectural advantage of VAR models in compositional generation.
tags:
  - ICLR 2026
  - Image Generation
  - Compositional Alignment
  - VAR Autoregression
  - T2I-CompBench++
  - GenEval
  - Benchmark Evaluation
date: 2026-05-08
content_hash: 1ad5370c4207205b
---

# Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models

**Conference**: ICLR 2026
**arXiv**: [2512.11542](https://arxiv.org/abs/2512.11542)
**Code**: None
**Area**: Diffusion Models / Compositional Generation
**Keywords**: Compositional Alignment, VAR Autoregression, T2I-CompBench++, GenEval, Benchmark Evaluation

## TL;DR

This paper presents the first systematic comparison of Visual Autoregressive (VAR) models and diffusion models on compositional text-image alignment. Evaluating 6 T2I models across T2I-CompBench++ and GenEval benchmarks, it finds that Infinity-8B achieves state-of-the-art performance on nearly all compositional dimensions, demonstrating a clear architectural advantage of VAR models in compositional generation.

## Background & Motivation

**Background**: T2I models can generate high-quality, semantically rich images, yet compositional alignment — faithfully binding objects, attributes, and spatial relations described in text to visual outputs — remains a core challenge.

**Limitations of Prior Work**: While prior work has evaluated the compositional capabilities of diffusion models, the compositional alignment of emerging Visual Autoregressive (VAR) architectures (e.g., Infinity) has not been systematically assessed. VAR models generate hierarchical latent codes via next-scale autoregressive prediction, a paradigm fundamentally different from the denoising process of diffusion models.

**Key Challenge**: High visual quality does not imply reliable compositional correctness — models may produce visually appealing images with incorrect attribute binding or incoherent spatial relationships. Yet cross-architecture systematic comparisons are lacking.

**Goal**: To provide the first unified benchmark evaluation of VAR vs. diffusion models on compositional alignment, covering color/texture/shape binding, spatial relations, counting, and complex multi-attribute composition.

**Key Insight**: Two complementary benchmarks are employed — T2I-CompBench++ (detector-driven validation) and GenEval (rule-based constraints) — spanning 8 evaluation dimensions across 6 representative T2I models.

**Core Idea**: VAR models (especially Infinity-8B) systematically outperform diffusion models on compositional alignment, potentially because next-scale autoregressive generation naturally conditions each stage on the already-generated visual structure.

## Method

### Overall Architecture

This paper is an empirical benchmark study and proposes no new method. The core contribution is a comparison of 6 models under a unified evaluation protocol:

- **Diffusion (UNet)**: SDXL
- **Diffusion (Transformer)**: PixArt-$\alpha$
- **DiT**: Flux-Dev, Flux-Schnell
- **VAR**: Infinity-2B, Infinity-8B

### Key Designs

1. **Evaluation Dimensions**:

   - T2I-CompBench++ covers 8 dimensions: color binding, texture binding, shape binding, non-spatial relations, 2D spatial relations, 3D spatial relations, numeracy, and complex composition.
   - GenEval covers 7 dimensions: single object, two objects, counting, colors, position, color attribution, and overall.
   - The two benchmarks use different validation methods (detector-based vs. rule-based), serving as mutual sanity checks.

2. **Evaluation Protocol**:

   - T2I-CompBench++: 4 images with independent random seeds are generated per prompt; seed-averaged results are reported (standard deviations provided in the appendix).
   - GenEval: The official protocol is followed, generating 4 samples per prompt and reporting aggregated scores.

3. **Model Coverage**:

   - Model scale ranges from 0.6B (PixArt-$\alpha$) to 12B (Flux), spanning three architectural paradigms.
   - Infinity-2B and 8B are used to analyze scaling effects within the VAR architecture.
   - Flux-Dev and Schnell are used to analyze the quality-speed trade-off.

### Loss & Training

Not applicable — this is a purely evaluative study involving no training.

## Key Experimental Results

### Main Results on T2I-CompBench++

| Model | Color | Texture | Shape | Non-Spatial | 2D Spatial | 3D Spatial | Numeracy | Complex | Mean |
|------|-------|---------|-------|-------------|-----------|-----------|----------|---------|------|
| SDXL | 0.593 | 0.519 | 0.466 | 0.311 | 0.215 | 0.341 | 0.504 | 0.319 | 0.409 |
| PixArt-$\alpha$ | 0.407 | 0.444 | 0.367 | 0.308 | 0.202 | 0.350 | 0.506 | 0.324 | 0.364 |
| Flux-Dev | 0.746 | 0.644 | 0.482 | 0.309 | 0.273 | 0.393 | **0.613** | 0.363 | 0.478 |
| Flux-Schnell | 0.725 | 0.683 | 0.559 | 0.312 | 0.271 | 0.373 | 0.604 | 0.364 | 0.486 |
| Infinity-2B | 0.741 | 0.636 | 0.480 | 0.310 | 0.240 | 0.406 | 0.573 | 0.382 | 0.471 |
| **Infinity-8B** | **0.827** | **0.753** | **0.604** | **0.316** | **0.365** | **0.414** | 0.612 | **0.397** | **0.536** |

Infinity-8B ranks first on 7 of 8 dimensions, with a mean of 0.536 substantially exceeding the runner-up Flux-Schnell (0.486).

### GenEval Results

| Model | Colors | Color Attr. | Position | Single Obj. | Two Obj. | Counting | Overall |
|------|--------|------------|----------|------------|---------|---------|---------|
| SDXL | 0.862 | 0.210 | 0.105 | 0.984 | 0.664 | 0.409 | 0.539 |
| PixArt-$\alpha$ | 0.801 | 0.093 | 0.068 | 0.978 | 0.505 | 0.438 | 0.480 |
| Flux-Dev | 0.766 | 0.470 | 0.185 | 0.988 | 0.785 | 0.716 | 0.652 |
| Flux-Schnell | 0.785 | 0.505 | 0.263 | 1.000 | 0.894 | 0.597 | 0.674 |
| Infinity-2B | 0.830 | 0.590 | 0.270 | 0.997 | 0.798 | 0.597 | 0.680 |
| **Infinity-8B** | **0.886** | **0.765** | **0.578** | **1.000** | **0.937** | **0.778** | **0.824** |

GenEval results are highly consistent with T2I-CompBench++ trends. Infinity-8B leads across all dimensions.

### Key Findings

- **Clear VAR Architectural Advantage**: Infinity-8B achieves the strongest performance on nearly all dimensions across both benchmarks. Even the smaller Infinity-2B (2B parameters) matches or surpasses Flux models with 12B parameters.
- **Scaling Effects**: The gap between Infinity-8B and 2B is substantial (Overall: 0.824 vs. 0.680), though gains are limited on certain dimensions (e.g., Non-Spatial), suggesting diminishing returns from further scaling in those areas.
- **SDXL and PixArt-$\alpha$ Consistently Underperform**: High aesthetic quality does not correlate with compositional correctness.
- **Strong Cross-Benchmark Consistency**: T2I-CompBench++ (detector-driven) and GenEval (rule-driven) yield consistent rankings, reinforcing the reliability of the conclusions.
- **Spatial Reasoning Remains Universally Difficult**: All models score relatively low on 2D/3D spatial relations and position metrics.

## Highlights & Insights

- **Filling an Important Gap**: This is the first systematic evaluation of VAR models' compositional capabilities, establishing a unified cross-architecture baseline — a timely contribution given the rapid development of VAR architectures.
- **Possible Explanation for VAR Advantage**: Next-scale autoregressive generation explicitly conditions each stage on previously generated visual structure, whereas global consistency in the denoising process must emerge implicitly across multiple refinement steps. This insight merits further investigation.
- **Efficiency–Performance Trade-off**: Infinity-2B achieves comparable performance to much larger diffusion models with significantly fewer parameters and less memory (measured in the appendix), suggesting that VAR models exhibit better efficiency scaling for compositional generation.

## Limitations & Future Work

- Only two benchmarks are used; human evaluation is absent.
- All results rely on automatic evaluators, which may be sensitive to prompt ambiguity.
- Runtime and memory measurements are conducted on a single hardware configuration only.
- The paper does not analyze *why* VAR models perform better — only observing the outcome — and lacks in-depth analysis of architectural differences.
- Recent models such as SD3 and DALL-E 3/4 are not included for comparison.
- The primary contribution is a benchmark study; no new method is proposed.

## Related Work & Insights

- **vs. ReNO**: ReNO is a test-time optimization method; the appendix shows that Infinity-8B surpasses it without any additional optimization.
- **vs. Attend-Excite**: Compositional generation improvement methods are primarily designed for diffusion models; the VAR results suggest that the methodological assumptions underlying compositional generation may need to be reconsidered.
- **Inspiration**: The hierarchical generation process of VAR models may be naturally suited to handling structured compositional relationships — determining object layout coarse-to-fine before filling in detailed attributes. This principle is worth incorporating into the design of future compositional generation methods.

## Rating

- Novelty: ⭐⭐⭐ — The first cross-architecture evaluation is valuable, but as a benchmark study, the methodological contribution is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multi-seed averaging, and variance analysis; the evaluation protocol is rigorous.
- Writing Quality: ⭐⭐⭐⭐ — Concise and clear, with rich tables and well-defined conclusions.
- Value: ⭐⭐⭐ — Provides important empirical evidence and baseline references for the community, though the lack of deeper analysis limits its impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [\[ICLR 2026\] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)

</div>

<!-- RELATED:END -->
