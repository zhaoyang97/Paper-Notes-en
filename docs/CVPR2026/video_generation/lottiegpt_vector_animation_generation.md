---
title: >-
  [Paper Note] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation
description: >-
  [CVPR 2026][Video Generation][Vector Animation] LottieGPT is the first autoregressive vector animation generation framework, designing a Lottie tokenizer to encode hierarchical geometry, transforms…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Vector Animation"
  - "Lottie"
  - "Autoregressive Generation"
  - "Tokenizer"
  - "Multimodal"
date: 2026-05-08
content_hash: 2ff27874a88662ca
---

# LottieGPT: Tokenizing Vector Animation for Autoregressive Generation

**Conference**: CVPR 2026  
**arXiv**: [2604.11792](https://arxiv.org/abs/2604.11792)  
**Code**: [https://lottiegpt.github.io/](https://lottiegpt.github.io/)  
**Area**: Image/Animation Generation  
**Keywords**: Vector Animation, Lottie, Autoregressive Generation, Tokenizer, Multimodal

## TL;DR

LottieGPT is the first autoregressive vector animation generation framework, designing a Lottie tokenizer to encode hierarchical geometry, transforms, and keyframe motion into compact token sequences. It builds a 660K animation dataset and fine-tunes Qwen-VL to generate editable vector animations directly from text/image inputs.

## Background & Motivation

**Background**: Video generation (Sora, Kling, etc.) can now produce high-quality raster video, but all existing generative models operate in pixel space and cannot generate vector animations—a resolution-independent, editable, and compact mainstream multimedia format.

**Limitations of Prior Work**: Vector animations (e.g., UI motion effects, brand animations, After Effects motion graphics) offer key properties unavailable in pixel video: infinite resolution, semantic manipulability, parameterized motion, and small file sizes. Existing SVG generation methods are limited to static output and lack temporal modeling capability.

**Key Challenge**: Vector animations contain both hierarchical structure and time-dependent transformation logic; encoding them into token sequences suitable for autoregressive modeling is the core challenge. Additionally, the absence of large-scale vector animation datasets is a major bottleneck.

**Goal**: (1) Design a tokenizer that unifies hierarchical geometry and temporal motion encoding; (2) build a large-scale vector animation dataset; (3) train the first multimodal model for vector animation generation.

**Key Insight**: Adopt the Lottie format (a widely deployed JSON animation standard), leveraging its keyframe + easing function parameterized representation for compact encoding.

**Core Idea**: Tokenize vector animations using keyframes and interpolation functions instead of per-frame data, drastically reducing sequence length while preserving structural fidelity.

## Method

### Overall Architecture

LottieGPT is built on the Qwen2.5-VL architecture with three components: (1) a Lottie tokenizer that encodes JSON animations into compact token sequences; (2) a vision-language backbone for multimodal input processing; (3) a two-stage training strategy (static first, then dynamic). Input is text/image/keyframe video; output is a Lottie token sequence that can be decoded into a complete editable vector animation.

### Key Designs

1. **Lottie Tokenizer — Hierarchical Structure Encoding**:

    - Function: Encode the hierarchical structure of Lottie JSON (animation metadata → assets → layers → shapes → attributes) into discrete token sequences
    - Mechanism: Special tokens (e.g., `<|LAYER|>`, `<|ty|>`) mark hierarchical boundaries and relationships, directly corresponding to the Lottie schema. Supports complete shape primitive encoding (ellipses, fills, gradients, strokes, etc.), unlike OmniSVG which requires decomposition into atomic commands
    - Design Motivation: Preserve semantic information and hierarchical organization so the model can learn structural patterns rather than arbitrary text sequences

2. **Keyframe Motion Compression**:

    - Function: Achieve compact temporal encoding, distinct from per-frame encoding
    - Mechanism: Store only keyframe time points `<|t|>`, attribute values, and Bézier easing functions `<|ease|>`, rather than dense per-frame data. For a 100-frame animation, only 6 keyframes' worth of tokens are needed; at 300 frames, the compression ratio reaches 98%. Easing functions are encoded as first-class primitives, enabling the same keyframes to produce dramatically different motion feels
    - Design Motivation: The essence of vector animation is keyframes + interpolation; this encoding drastically reduces token count while preserving complete animation information

3. **Static-to-Dynamic Progressive Training**:

    - Function: Stabilize training through curriculum learning
    - Mechanism: Stage 1 trains on static vector graphics (50% text-to-Lottie + 50% image-to-Lottie); Stage 2 introduces temporal dynamics (34% text-only + 33% text + first frame + 33% text + video keyframes)
    - Design Motivation: Directly mixing static and dynamic training data causes convergence instability, as animation samples have far more tokens than static graphics

### Loss & Training

Standard causal language model cross-entropy loss: $\mathcal{L} = -\sum_{i=1}^{N} \log P(t_i | t_{<i}, \mathbf{c})$, where $\mathbf{c}$ is the multimodal condition. The tokenizer supports lossless round-tripping: decoded animations render identically to the originals.

## Key Experimental Results

### Main Results

| Method | Input | CLIP↑ | SSIM↑ | LPIPS↓ | DINOv2↑ | JSON↑ | Valid Rate |
|--------|-------|-------|-------|--------|---------|-------|-----------|
| OmniSVG-7B | Text | 0.832 | 0.563 | 0.512 | 0.727 | N/A | N/A |
| LottieGPT-7B | Text | **0.933** | **0.810** | **0.176** | **0.857** | 0.824 | 98.3% |
| StarVector-8B | Image | 0.766 | 0.385 | 0.465 | 0.529 | N/A | N/A |
| OmniSVG-7B | Image | 0.900 | 0.705 | 0.251 | 0.848 | N/A | N/A |
| LottieGPT-7B | Image | **0.945** | **0.835** | **0.154** | **0.876** | 0.843 | 98.8% |

### Ablation Study

| Config | CLIP↑ | SSIM↑ | Valid Rate |
|--------|-------|-------|-----------|
| full model (Stage1+2) | 0.933 | 0.810 | 98.3% |
| Stage 1 only (no animation) | 0.928 | 0.805 | 97.5% |
| No hierarchical encoding | 0.891 | 0.752 | 92.1% |
| Per-frame encoding instead of keyframes | 0.875 | 0.701 | 85.6% |

### Key Findings

- Keyframe encoding is critical for valid rate: per-frame encoding causes excessively long sequences, dropping valid rate from 98.3% to 85.6%
- Temporal modeling enhances static vector understanding: LottieGPT also achieves new SOTA on SVG generation
- JSON structural scores confirm high structural fidelity in generated Lottie files

## Highlights & Insights

- The keyframe + easing function encoding is an elegant design: it preserves complete animation semantics (motion curves as first-class primitives) while achieving extremely high compression ratios. This approach can generalize to other parameterized representation generation tasks
- The dataset contribution is significant: 660K vector animations + 15M static vector graphics, the first large-scale resource in this domain
- Framing 2D animation generation analogously to 3D animation production workflows (generate structure first, then add animation) is an inspiring perspective

## Limitations & Future Work

- Only supports the Lottie format; does not cover SVG SMIL or CSS animations
- Complex animation token sequences remain long, constrained by VLM context windows
- No human evaluation of temporal consistency and motion naturalness in generated animations
- Extensible to interactive animation editing and conditional generation

## Related Work & Insights

- **vs OmniSVG/StarVector**: These methods can only generate static SVGs; LottieGPT is the first to support temporal modeling and animation generation
- **vs Pixel video generation**: Pixel methods produce fixed-resolution, non-editable output; LottieGPT output is infinitely scalable and fully editable

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First autoregressive vector animation generation framework, opening a new direction
- Experimental Thoroughness: ⭐⭐⭐⭐ — Introduces LottieBench with multi-dimensional evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation and well-defined contributions
- Value: ⭐⭐⭐⭐⭐ — Complete contribution of dataset + benchmark + method

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens](omnilottie_generating_vector_animations_via_parameterized_lottie_tokens.md)
- [\[ICML 2026\] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](../../ICML2026/video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)
- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)

</div>

<!-- RELATED:END -->
