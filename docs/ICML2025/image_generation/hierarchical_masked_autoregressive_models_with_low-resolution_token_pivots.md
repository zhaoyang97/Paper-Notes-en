---
title: >-
  [Paper Note] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots
description: >-
  [ICML 2025][Image Generation][Autoregressive Models] This work proposes Hi-MAR, which introduces low-resolution tokens as intermediate pivots in masked autoregressive image generation to establish a coarse-to-fine hierarchical generation process. It also enhances inter-token dependency modeling with a Diffusion Transformer Head, significantly outperforming MAR on ImageNet with less computational cost (FID improved by 0.38).
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Autoregressive Models"
  - "Hierarchical Generation"
  - "Masked Autoregressive"
  - "Global Context"
  - "Diffusion Transformer Head"
date: 2026-05-08
content_hash: a9da0b557c20a26d
---

# Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

**Conference**: ICML 2025  
**arXiv**: [2505.20288](https://arxiv.org/abs/2505.20288)  
**Code**: [https://github.com/HiDream-ai/himar](https://github.com/HiDream-ai/himar)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Autoregressive Models, Hierarchical Generation, Masked Autoregressive, Global Context, Diffusion Transformer Head

## TL;DR
This work proposes Hi-MAR, which introduces low-resolution tokens as intermediate pivots in masked autoregressive image generation to establish a coarse-to-fine hierarchical generation process. It also enhances inter-token dependency modeling with a Diffusion Transformer Head, significantly outperforming MAR on ImageNet with less computational cost (FID improved by 0.38).

## Background & Motivation
**Background**: Autoregressive (AR) models are gradually emerging in visual generation. Masked autoregressive models, represented by MAR, avoid the information loss caused by discretization through continuous-valued tokens and diffusion loss.

**Limitations of Prior Work**: Models like MAR only perform autoregressive modeling on a single-scale dense token sequence, lacking global contextual information, which is particularly detrimental to the prediction of early tokens. In addition, MAR uses an MLP-based diffusion head to process each token independently, ignoring spatial dependencies among tokens, which can result in artifacts such as abnormal bright spots.

**Key Challenge**: Single-scale autoregression conflates global structure construction with local detail refinement, which is counterintuitive to the human cognitive process of "global first, local second".

**Goal**: (a) How can global structural information be introduced in autoregressive modeling? (b) How can inter-token dependencies be modeled within the diffusion head?

**Key Insight**: First capture the global structure using a small number of low-resolution tokens, and then use this as a condition to guide the generation of high-resolution dense tokens.

**Core Idea**: Hierarchical masked autoregression using low-resolution tokens as "pivots" + a diffusion head that replaces MLP with a Transformer.

## Method

### Overall Architecture
Hi-MAR is a two-stage hierarchical masked autoregressive model. The input image is simultaneously encoded into low-resolution (128×128) and high-resolution (256×256) continuous token sequences. The first stage performs masked autoregressive modeling on the low-resolution tokens, outputting condition tokens (rather than direct visual tokens) that reflect the global structure. The second stage concatenates these condition tokens with high-resolution masked tokens, feeding them into the same Transformer for refined generation.

### Key Designs

1. **Hierarchical Masked Autoregressive Transformer (Hi-MAR Transformer)**:

    - **Function**: Two-stage modeling, from coarse to fine
    - **Mechanism**: The first stage performs MAR on low-resolution tokens with bidirectional attention to output condition tokens $Z^s$. The second stage concatenates $Z^s$ with high-resolution masked tokens and passes them through the Transformer again to generate dense condition tokens.
    - **Design Motivation**: Directly using low-resolution visual tokens (instead of condition tokens) for guidance causes training-inference mismatch—using ground-truth low-resolution tokens during training, but using predicted (noisy) low-resolution tokens during inference. Using condition tokens output by the Transformer instead of visual tokens mitigates this issue.

2. **Scale-aware Transformer Block**:

    - **Function**: Enables the shared Transformer to perceive which scale is currently being processed.
    - **Mechanism**: Scale information is encoded using sinusoidal embeddings to generate a scale vector $v$ via an MLP. Subsequently, the adaLN-Zero operation is employed to modulate the scale/shift parameters of LayerNorm and the scaling parameters of residual connections: $z_{a} = z^i + \gamma_1 \cdot \text{Attention}(\alpha_1 \cdot \text{LN}(z^i) + \beta_1)$.
    - **Design Motivation**: A shared Transformer processes tokens of two different scales simultaneously; without scale guidance, it would lead to blurriness.

3. **Diffusion Transformer Head**:

    - **Function**: Replaces the MLP-based diffusion head to model dependencies among all tokens during masked token prediction.
    - **Mechanism**: In the second stage, Transformer blocks with self-attention are used as the diffusion head. The input consists of representations of all (masked + unmasked) condition tokens modulated by adaLN, rather than processing only masked tokens independently with an MLP.
    - **Design Motivation**: The MLP head processes each token independently, losing global spatial structure information of the image. The Transformer head captures inter-token interactions through self-attention.

### Loss & Training
- In the first stage, the masking ratio is randomly sampled from $[0.7, 1.0]$ (same as MAR).
- The second stage utilizes the cosine masking strategy of MaskGIT.
- Both stages employ the standard diffusion denoising loss: $\mathcal{L}(z_i, x_i) = \mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(x_i^t|t, z_i)\|^2]$.
- During inference, the first stage takes 32 steps, while the second stage takes only 4 steps (due to the stronger modeling capability of the Transformer head and the global structure already provided by the first stage).

## Key Experimental Results

### Main Results

| Dataset | Model | FID (w/ CFG) ↓ | IS ↑ | Precision | Recall |
|--------|------|----------|------|-----------|--------|
| ImageNet 256 | MAR-B | 2.31 | 281.7 | 0.82 | 0.57 |
| ImageNet 256 | **Hi-MAR-B** | **1.93** | **293.0** | 0.81 | 0.59 |
| ImageNet 256 | MAR-H | 1.55 | 303.7 | 0.81 | 0.62 |
| ImageNet 256 | **Hi-MAR-H** | **1.52** | **322.78** | 0.80 | 0.63 |
| MS-COCO 256 | MAR | 6.36 | - | - | - |
| MS-COCO 256 | **Hi-MAR-S** | **4.77** | - | - | - |

### Ablation Study

| Configuration | FID ↓ | Description |
|------|-------|------|
| MAR-B baseline | 2.31 | Basic single-scale MAR |
| + Hierarchical (guided by visual tokens) | 2.28 | Training-inference mismatch, almost no improvement |
| + Hierarchical (guided by condition tokens) | 2.07 | Significant gain of 0.24 |
| + Diff Transformer Head (Stage 2) | 1.98 | Further decreased by 0.09 |
| + Scale vector (Full Hi-MAR) | **1.93** | Final optimal model |

### Key Findings
- Replacing visual tokens with condition tokens for guidance is the most crucial design, contributing to a 0.24 FID improvement.
- The Diffusion Transformer Head is only effective in the second stage; replacing the MLP head in the first stage yields no significant gain.
- Hi-MAR features faster inference speed: the second stage requires only 4 steps to reach near-saturated quality, with the total computational cost being only 54% of MAR.

## Highlights & Insights
- **Clever Decoupling of Hierarchical Generation**: Generating global structures first (low-resolution tokens) and refining details later (dense tokens) aligns with human perception while reducing computational cost.
- **Training-Inference Consistency Design**: Guiding the second stage with condition tokens output by the Transformer instead of direct visual tokens effectively bypasses the inconsistency between ground truth and predictions.
- **Transferable Concept**: The design of the Diffusion Transformer Head (using self-attention instead of MLP to model inter-token dependencies) can be transferred to other tasks requiring per-token prediction.

## Limitations & Future Work
- Only a two-level hierarchical structure is validated; the effects of more levels (e.g., 3-4 levels) remain unexplored.
- The impact of resolution choice for low-resolution tokens (128 vs 64 vs 32) is not analyzed in depth.
- Text-to-image generation is only validated on MS-COCO, lacking experiments on large-scale T2I datasets (such as LAION).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combined innovation of hierarchical MAR and Transformer diffusion head.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ ImageNet + MS-COCO + thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ An effective direction of improvement for autoregressive image generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](../../CVPR2025/image_generation/hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[ICLR 2026\] Generation then Reconstruction: Accelerating Masked Autoregressive Models via Two-Stage Sampling](../../ICLR2026/image_generation/generation_then_reconstruction_accelerating_masked_autoregressive_models_via_two.md)
- [\[NeurIPS 2025\] Conditional Panoramic Image Generation via Masked Autoregressive Modeling](../../NeurIPS2025/image_generation/conditional_panoramic_image_generation_via_masked_autoregres.md)

</div>

<!-- RELATED:END -->
