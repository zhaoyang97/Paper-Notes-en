---
title: >-
  [Paper Note] OminiControl: Minimal and Universal Control for Diffusion Transformer
description: >-
  [Image Generation] OminiControl is proposed to unify spatially aligned and non-aligned image control tasks on the DiT architecture with only 0.1% additional parameters. Core innovations include unified sequence processin…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 58744c7a821b2fd5
---

# OminiControl: Minimal and Universal Control for Diffusion Transformer

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2411.15098](https://arxiv.org/abs/2411.15098)
- **Code**: [GitHub](https://github.com/Yuanshi9815/OminiControl)
- **Area**: Diffusion Models · Controllable Generation
- **Keywords**: DiT, image-conditioned control, unified sequence processing, dynamic positional encoding, Subjects200K

## TL;DR
OminiControl is proposed to unify spatially aligned and non-aligned image control tasks on the DiT architecture with only 0.1% additional parameters. Core innovations include unified sequence processing, dynamic positional encoding, and an attention bias control mechanism.

## Background & Motivation

Existing image-conditioned control methods suffer from three major limitations:

**High parameter overhead**: ControlNet replicates the entire network; IP-Adapter requires an additional encoder.

**Task bias**: Spatially aligned control (edge, depth guidance) and non-aligned control (style transfer, subject-driven generation) typically demand different architectures.

**Architecture constraints**: Most methods are designed for UNet and transfer poorly to DiT (Diffusion Transformer).

**Core Problem**: Can a single unified, minimalist framework handle all image control tasks on the DiT architecture?

## Method

### Overall Architecture

OminiControl builds upon the DiT architecture of FLUX.1 and achieves minimal universal control through innovations at three levels.

### 1. Minimalist Architecture Design (Only 0.1% Extra Parameters)

**Parameter reuse strategy**: The DiT's own VAE encoder is reused to process condition images, projecting them into the same latent space as the noisy image. The DiT's transformer blocks are reused to process condition tokens, with only lightweight LoRA fine-tuning added.

Compared to ControlNet (which duplicates the entire network) and IP-Adapter (which introduces a CLIP encoder and cross-attention), the parameter overhead is minimal.

### 2. Unified Sequence Processing

Condition tokens are directly concatenated into the image token sequence:

$$\text{MMA}([X; C_T; C_I]) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

where $[X; C_T; C_I]$ denotes the concatenation of noisy image tokens, text tokens, and condition image tokens.

**Key advantage**: Compared to the conventional feature addition $h_X \leftarrow h_X + h_{C_I}$, unified sequence processing allows multi-modal attention to automatically discover appropriate inter-token relationships—whether spatial or semantic. Experiments confirm that training loss is consistently lower than with feature addition.

### 3. Dynamic Positional Encoding

An adaptive strategy based on RoPE (Rotary Position Embedding):

$$
(i,j)_{C_I} = \begin{cases}
(i,j)_X & \text{spatially aligned tasks} \\
(i,j) + \Delta & \text{non-aligned tasks}
\end{cases}
$$

- **Spatially aligned tasks**: Condition tokens share positional indices with image tokens, promoting direct spatial correspondence.
- **Non-aligned tasks**: Condition token positions are offset by $\Delta$ (e.g., $(0,32)$), avoiding spatial overlap, which accelerates convergence and improves performance.

### 4. Attention Bias for Flexible Control

A bias matrix $B(\gamma)$ is introduced to control conditioning strength:

$$\text{MMA}([X; C_T; C_I]) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B(\gamma)\right) V$$

The design of $B(\gamma)$ removes the conditioning effect when $\gamma=0$ and amplifies it when $\gamma>1$, allowing users to dynamically adjust the control intensity at inference time.

### Subjects200K Dataset

To address the data bottleneck in subject-driven generation, over 200K identity-consistent image pairs are synthesized using FLUX.1 itself. The pipeline proceeds as follows: GPT-4o generates diverse subject descriptions → descriptions are reorganized into structured prompts → the DiT generates paired images.

## Key Experimental Results

### Subject-Driven Generation Comparison

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| IP-Adapter | 0.631 | 0.772 | 0.289 |
| InstantID | 0.586 | 0.738 | 0.268 |
| **OminiControl** | **0.720** | **0.812** | **0.295** |

### Spatial Control Comparison (Canny-Guided)

| Method | SSIM↑ | RMSE↓ | Extra Params |
|------|-------|-------|---------|
| ControlNet | 0.491 | 0.357 | ~1.4B |
| T2I-Adapter | 0.425 | 0.412 | ~100M |
| **OminiControl** | **0.530** | **0.325** | **~14M** |

OminiControl achieves superior performance with only 1% of ControlNet's parameter count.

### Key Findings

1. Training loss under unified sequence processing is consistently lower than feature addition, validating the advantage of allowing attention to discover inter-token relationships autonomously.
2. In non-aligned tasks, positional offset yields a significant convergence speedup (approximately 2×).
3. The attention bias mechanism smoothly modulates conditioning strength over the range [0, 2] without requiring retraining.

## Highlights & Insights

1. **Extreme simplicity**: Comprehensive control is achieved with only 0.1% additional parameters, embodying a "less is more" design philosophy.
2. **Unified processing**: The artificial boundary between spatially aligned and non-aligned control is dissolved, validating the adaptive capacity of the attention mechanism.
3. **Practical contributions**: The Subjects200K dataset and open-source LoRA weights provide immediately usable resources for the community.
4. **Insight from dynamic positional encoding**: Spatially aligned tasks benefit from shared positions while non-aligned tasks require independent positions—a simple yet critical distinction.

## Limitations & Future Work

- The capability for complex compositional control (multiple simultaneous conditions) is not thoroughly investigated.
- The synthesis quality of Subjects200K is bounded by the capacity of the base model.
- The attention bias mechanism introduces additional hyperparameter tuning overhead at inference time.

## Related Work & Insights

- **Control methods**: ControlNet, T2I-Adapter, IP-Adapter, UniControl
- **DiT architectures**: FLUX.1, Stable Diffusion 3, PixArt
- **Data synthesis**: Bootstrapping strategies that leverage the generative model itself to create training data

## Rating
- Novelty: ★★★★☆ — The combination of unified sequence processing and dynamic positional encoding is concise and innovative.
- Technical Depth: ★★★★☆ — Experiments are comprehensive and insights are substantive.
- Practicality: ★★★★★ — Plug-and-play design with high community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)

</div>

<!-- RELATED:END -->
