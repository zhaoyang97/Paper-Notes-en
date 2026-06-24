---
title: >-
  [Paper Note] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders
description: >-
  [CVPR 2025 (Oral)][Image Generation][Autoregressive generation] This paper proposes RandAR—the first decoder-only visual autoregressive model that supports arbitrary token generation orders. By inserting a "position instruction token" before each image token to specify the spatial position of the next token to be generated, it unlocks novel capabilities including parallel decoding (2.5x speedup), zero-shot inpainting/outpainting, and resolution extrapolation…
tags:
  - "CVPR 2025 (Oral)"
  - "Image Generation"
  - "Autoregressive generation"
  - "random order"
  - "decoder-only"
  - "position instruction token"
  - "parallel decoding"
date: 2026-05-08
content_hash: b98ababa2529a217
---

# RandAR: Decoder-only Autoregressive Visual Generation in Random Orders

**Conference**: CVPR 2025 (Oral)  
**arXiv**: [2412.01827](https://arxiv.org/abs/2412.01827)  
**Code**: [https://github.com/ziqipang/RandAR](https://github.com/ziqipang/RandAR)  
**Area**: Other  
**Keywords**: Autoregressive generation, random order, decoder-only, position instruction token, parallel decoding

## TL;DR

This paper proposes RandAR—the first decoder-only visual autoregressive model that supports arbitrary token generation orders. By inserting a "position instruction token" before each image token to specify the spatial position of the next token to be generated, it unlocks novel capabilities including parallel decoding (2.5x speedup), zero-shot inpainting/outpainting, and resolution extrapolation, without sacrificing performance.

## Background & Motivation

**Background**: Inspired by the success of "next-token prediction" in language models, the vision domain has explored using GPT-style decoder-only transformers for image generation. A typical practice is to tokenize images into discrete 2D tokens, arrange them into a 1D sequence in raster-scan order (from top-left to bottom-right), and predict them sequentially using a causal transformer.

**Limitations of Prior Work**: Enforcing a unidirectional raster-scan order restricts the ability of decoder-only transformers to model the bidirectional context of 2D images. This constraint is not faced by encoder-decoder counterparts (such as MaskGIT, MAR). More crucially, the fixed order prevents decoder-only models from naturally supporting parallel decoding (generating only one token at a time) and handling completion tasks for partially known images.

**Key Challenge**: Is the predefined raster-scan order really a necessary and useful inductive bias for decoder-only image generators? If not, how can these models acquire bidirectional modeling capabilities?

**Goal**: Design a decoder-only AR model that can operate under arbitrary token orders, breaking the limitation of fixed orders while maintaining or even improving generation quality.

**Key Insight**: Inspired by random permutation training in language models (e.g., XLNet), but direct application to visual tokens faces the challenge of position awareness—the model needs to know "where the next token to be predicted is."

**Core Idea**: Insert a "position instruction token" before each image token to be predicted, explicitly informing the model of the next token's 2D spatial position. This allows standard causal transformers to be trained and perform inference on token sequences of arbitrary permutations.

## Method

### Overall Architecture

RandAR is based on a standard GPT-style decoder-only transformer. The input sequence consists of class tokens, randomly permuted image tokens, and corresponding position instruction tokens interleaved together. During training, next-token prediction is performed on the randomly permuted token sequence; during inference, image tokens can be generated in any order (including in parallel).

### Key Designs

1. **Position Instruction Token**:

    - **Function**: Inserts a special token before each image token to be predicted, encoding the 2D spatial coordinates of that image token.
    - **Mechanism**: In standard AR models, the position of a token is implicitly determined by its order in the sequence. However, under a random order, the sequence position no longer corresponds to the spatial position. Therefore, a position instruction token $p_i$ is inserted before each image token $t_i$, where $p_i$ encodes the row and column coordinates of $t_i$ in the image grid. Upon seeing $p_i$, the model knows the spatial position of the next token to be predicted, thereby correctly utilizing the contextual information.
    - **Design Motivation**: This is the key to enabling random-order generation. Without the position instruction, the model cannot know which position of the image to predict next, causing random permutation training to fail completely.

2. **Random Permutation Training**:

    - **Function**: Enables the model to maintain consistent generation quality under any generation order.
    - **Mechanism**: During training, a permutation $\sigma$ is randomly sampled for the token sequence of each image. The tokens are reordered according to $\sigma$ for standard causal language model training. Since the model sees different permutations during each training step, it is forced to learn image representations independent of a specific order. Although this is more challenging than fixed-order training, the model still achieves an FID comparable to raster-order models.
    - **Design Motivation**: Random permutation training equips the model with "order invariance," which serves as the foundation for all subsequent zero-shot capabilities.

3. **Parallel Decoding with KV-Cache**:

    - **Function**: Simultaneously generates multiple tokens during inference, significantly accelerating the generation process.
    - **Mechanism**: Due to the model's order invariance, it can predict multiple tokens in a single step. Specifically, multiple position instruction tokens are inserted in one forward pass to concurrently predict the image tokens at corresponding positions. This is combined with KV-Cache to avoid redundant attention computation for already generated tokens. By generating $k$ tokens in parallel per step, the total steps are reduced from $N$ to $N/k$, achieving approximately 2.5x speedup with almost no drop in quality.
    - **Design Motivation**: The biggest bottleneck of AR models is the slow speed of token-by-token sequential generation. Parallel decoding is a capability naturally brought by random permutation training—since the model does not rely on a specific order, there are no strict causal dependencies among multiple positions.

### Loss & Training

Standard causal language model loss (cross-entropy) is used to compute the next-token prediction loss on randomly permuted token sequences. The model is optimized using AdamW and trained on ImageNet 256×256. The architecture is based on LLaMAGen.

## Key Experimental Results

### Main Results: ImageNet 256×256 Class-Conditional Image Generation

| Model | Type | Parameters | FID-50K ↓ | IS ↑ | Order |
|------|------|--------|-----------|------|------|
| LLaMAGen-L | AR | 0.3B | 2.18 | 256 | Raster |
| LLaMAGen-XL | AR | 0.7B | 2.62 | 244 | Raster |
| **RandAR-L** | **AR** | **0.3B** | **2.55** | **288** | **Random** |
| **RandAR-XL** | **AR** | **0.7B** | **2.25** | **318** | **Random** |
| MaskGIT | Masked | - | 6.18 | 182 | Bidirectional |
| MAR-L | Masked | 0.5B | 1.78 | 296 | Bidirectional |

### Parallel Decoding Acceleration Effect

| Parallelism $k$ | FID ↓ | Speedup | Description |
|-----------|-------|---------|------|
| 1 (Token-by-token) | 2.55 | 1.0x | Baseline |
| 2 | ~2.6 | ~1.8x | Almost no quality degradation |
| 4 | ~2.7 | ~2.5x | Slight quality degradation |
| 8 | ~3.0 | ~3.5x | Quality begins to degrade significantly |

### Key Findings

- RandAR achieves FID/IS comparable to or even better than its raster-order counterparts under random orders, proving that a fixed order is not a necessary inductive bias.
- Parallel decoding achieves a 2.5x speedup at $k=4$ with negligible FID increase, addressing the efficiency bottleneck of AR models.
- Zero-shot inpainting/outpainting qualitative results are promising, and the model can extrapolate a 256×256 image to 256×1024 zero-shot.
- Random permutation training equips the model with bidirectional context understanding capabilities, which can be utilized for feature extraction.

## Highlights & Insights

- **Design of Position Instruction Tokens**: An extremely elegant solution—breaking the limitations of fixed order simply by inserting position information into the input sequence, without modifying the transformer architecture. This design maintains the simplicity of the GPT architecture while granting brand-new capabilities.
- **Emergent Capabilities from Random Permutations**: The more difficult training task (random order) not only did not harm performance but also endowed the model with 5 zero-shot capabilities: parallel decoding, inpainting, outpainting, resolution extrapolation, and bidirectional feature encoding. This is analogous to the phenomenon in language models where "more challenging pre-training leads to stronger generalization."
- **Efficiency Solution for AR Models**: Parallel decoding requires no additional fine-tuning or architectural changes, representing an important step towards making AR models practical.

## Limitations & Future Work

- Currently only verified on ImageNet 256×256, and has not been scaled to higher resolutions (512, 1024) or text-to-image generation.
- Compared to methods like MAR that excel in bidirectional modeling, there is still a gap in FID (2.25 vs 1.78), indicating that random permutation training has not fully closed the gap.
- The optimal parallelism $k$ for parallel decoding needs to be manually selected based on the quality-speed trade-off.
- Resolution extrapolation capability is only demonstrated qualitatively, lacking quantitative evaluation.
- The application of random permutation training to other modalities, such as video generation and 3D generation, has not been explored.

## Related Work & Insights

- **vs LLaMAGen**: RandAR is based on the same architecture but breaks the raster-scan constraint, achieving comparable performance while gaining several new capabilities.
- **vs MaskGIT/MAR**: These encoder-decoder models naturally support bidirectional modeling but are not GPT-style; RandAR is the first to bring similar capabilities to decoder-only models.
- **vs XLNet**: Borrows the idea of permutation training but resolves the 2D spatial position awareness problem in the visual domain through position instruction tokens.
- **vs PAR (CVPR 2025)**: PAR, a subsequent work of RandAR, further optimizes the parallel decoding strategy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The design of position instruction tokens to break the fixed order of AR is highly elegant, representing a significant breakthrough for visual AR models.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ImageNet experiments are solid, and the demonstration of various zero-shot capabilities is comprehensive, but there is a lack of high-resolution and text-conditional experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation is clear, the logic is rigorous, and it flows seamlessly from problem to solution to validation.
- Value: ⭐⭐⭐⭐⭐ It opens up a new paradigm for decoder-only visual generative models, which may profoundly influence future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](random_conditioning_for_diffusion_model_compression_with_distillation.md)
- [\[ICML 2025\] Continuous Visual Autoregressive Generation via Score Maximization](../../ICML2025/image_generation/continuous_visual_autoregressive_generation_via_score_maximization.md)

</div>

<!-- RELATED:END -->
