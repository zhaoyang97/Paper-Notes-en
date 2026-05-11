---
title: >-
  [Paper Note] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention
description: >-
  [Image Generation] EDiT proposes a linear compressed attention mechanism that enhances query local information via ConvFusion and compresses key/value tokens via a Spatial Compressor…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 19ee8871f019b9d6
---

# EDiT: Efficient Diffusion Transformers with Linear Compressed Attention

## Basic Information

- **Conference**: ICCV 2025
- **arXiv**: 2503.16726
- **Code**: Not released
- **Area**: Image Generation
- **Keywords**: Diffusion Transformer, linear attention, efficient inference, multimodal DiT, knowledge distillation

## TL;DR

EDiT proposes a linear compressed attention mechanism that enhances query local information via ConvFusion and compresses key/value tokens via a Spatial Compressor, achieving up to 2.2× acceleration over DiT and MM-DiT architectures while maintaining comparable image quality.

## Background & Motivation

Diffusion Transformers (DiTs) have become the dominant architecture for text-to-image generation (FLUX, PixArt-Σ, SD3, etc.), but the quadratic complexity of standard scaled dot-product attention severely limits:

**High-resolution generation**: Token counts grow with resolution, making 4K/8K image generation computationally prohibitive.

**Edge device deployment**: Resource-constrained devices such as smartphones struggle to run these models.

Limitations of existing acceleration approaches:
- **SANA**: Linear attention + MixFFN convolution module, but MixFFN introduces additional computation.
- **LinFusion**: Multi-layer nonlinear transformations for token mapping, but performs poorly on DiTs (confirmed by both the authors and the original paper).
- **PixArt-Σ KV compression**: Applies depthwise convolution compression only to keys/values, while still maintaining quadratic complexity.

More critically, **multimodal DiTs (MM-DiTs)** process image and text tokens jointly in a shared attention mechanism, resulting in even larger token counts and more severe efficiency bottlenecks, yet **no effective linearization scheme** has previously been proposed for this setting.

## Method

### Overall Architecture

EDiT consists of two components:
- **EDiT**: Linear compressed attention for standard DiTs.
- **MM-EDiT**: Hybrid attention (linear + scaled dot-product) for MM-DiTs.

### ConvFusion: Enhancing Local Information in Queries

ConvFusion unifies the per-token processing of LinFusion and the MixFFN of SANA into a multi-layer convolutional mapping:

$$Q^{\text{EDiT}} = \phi_{\text{CF}}(X) = \text{ReLU}(\text{Linear}(X + \text{Conv}(\text{LeakyReLU}(\text{GN}(\text{Conv}(X))))))$$

Key designs:
- The 1D token sequence is reshaped into a 2D spatial layout before applying 2D convolutions, exploiting the spatial locality of images.
- The first convolution (3×3) compresses along the channel dimension; the second (1×1) restores the channel count.
- Group Normalization is used to improve training stability.

### Spatial Compressor: Compressing Keys and Values

A depthwise convolution is applied to spatially compress keys and values:

$$K^{\text{EDiT}} = \text{ReLU}(\phi_{\text{SC}}(X)), \quad V^{\text{EDiT}} = \phi_{\text{SC}}(X)$$
$$\phi_{\text{SC}}(X) = \text{Conv}(\text{Linear}(X))$$

A 3×3 depthwise convolution with stride=2 reduces the number of key/value tokens by a factor of 4. Unlike PixArt-Σ, this compression is integrated into linear attention (rather than quadratic attention), yielding further efficiency gains.

### Linear Attention Formulation

The ConvFusion queries and Spatial Compressor keys/values are substituted into the standard linear attention:

$$y_i = \frac{Q_i \sum_{j=1}^{N}(K_j^T V_j)}{Q_i \sum_{j=1}^{N} K_j}$$

### MM-EDiT: Hybrid Attention for Multimodal DiTs

Core observation: In joint attention, the $Q_I K_I^T$ (image-to-image) term dominates computation and scales quadratically with resolution.

Hybrid strategy:
- **Image-to-image interactions**: EDiT linear compressed attention.
- **Image-to-text / text-to-image / text-to-text interactions**: Standard scaled dot-product attention.

$$\mathbf{A}^{\text{Hybrid}} = \begin{pmatrix} \eta_I^{\text{Lin}} \cdot \mathbf{A}^{\text{Lin}}(Q_I, K_I, V_I) + (1-\eta_I^{\text{Lin}}) \cdot \mathbf{A}(Q_I, K_P, V_P) \\ \eta_P \cdot \mathbf{A}(Q_P, K_I, V_I) + (1-\eta_P) \cdot \mathbf{A}(Q_P, K_P, V_P) \end{pmatrix}$$

The normalization factor is approximated by the token count ratio $\hat{\eta}^{\text{Lin}} = \frac{N_I}{N_I + N_T}$, avoiding the implementation complexity of custom attention kernels. Experiments further show that this approximation marginally outperforms exact computation.

### Loss & Training

Knowledge distillation is employed:
- **Task loss**: Noise prediction / rectified flow loss.
- **Knowledge distillation**: Minimizes the discrepancy between student and teacher predictions.
- **Feature distillation**: Aligns self-attention outputs at each layer.

Multi-stage progressive resolution training: 512 → 1024 → 2048.

## Key Experimental Results

### Main Results: EDiT vs. PixArt-Σ and Other Linear Methods

| Method | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ |
|---|---|---|---|---|---|---|
| | **512×512** | | | **1024×1024** | | |
| PixArt-Σ (teacher) | 0.285 | 7.57 | 2.50 | 0.285 | 7.09 | 2.53 |
| **EDiT (ours)** | 0.283 | **7.06** | 2.57 | 0.290 | 7.82 | 2.64 |
| SANA-DiT | 0.283 | 8.43 | 3.31 | 0.286 | 9.31 | 3.16 |
| LinFusion-DiT | 0.289 | 15.87 | 5.98 | 0.283 | 44.66 | 11.01 |
| KV Comp. (k=2) | 0.275 | 10.69 | 3.77 | 0.283 | 10.32 | 3.50 |

EDiT achieves a lower FID than the teacher at 512 resolution (7.06 vs. 7.57). LinFusion-DiT degrades severely on DiT (44.66 FID at 1024).

### Ablation Study: Query/Key/Value Processing Strategies

| Q | K | V | FID (512) ↓ | FID (1024) ↓ |
|---|---|---|---|---|
| CF | SC | SC | **7.06** | **7.82** |
| CF | CF | - | 7.59 | 7.76 |
| - | SC | SC | 14.53 | 26.59 |
| CF | - | - | 7.06 | 7.70 |

Using only the Spatial Compressor without ConvFusion (row 3) leads to severe performance degradation, demonstrating that ConvFusion is critical for enhancing local information in queries.

### Latency Analysis

| Resolution | PixArt-Σ | SANA-DiT | EDiT | Speedup (vs. PixArt) |
|---|---|---|---|---|
| 1024×1024 | 0.047s | 0.043s | **0.034s** | 1.4× |
| 2048×2048 | 0.387s | 0.166s | **0.121s** | 3.2× |
| 4096×4096 | 4.770s | 0.687s | **0.461s** | 10.3× |
| 8192×8192 | 72.96s | 21.76s | **1.693s** | **43×** |

EDiT achieves a 43× speedup over PixArt-Σ at 8K resolution. On a Samsung S25 Ultra smartphone, EDiT also reduces latency by 38%.

### MM-EDiT vs. SD-v3.5M

| Method | Hybrid | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ |
|---|---|---|---|---|
| SD-v3.5M (teacher) | – | 0.283 | 10.49 | 3.86 |
| **MM-EDiT (Ours)** | ✓ | 0.285 | 11.60 | 3.91 |
| SANA-MM-DiT | × | 0.279 | 14.94 | 5.02 |
| Linear MM-DiT-α | × | 0.281 | 13.59 | 4.28 |

The hybrid attention in MM-EDiT substantially outperforms fully linearized baselines (FID 11.60 vs. 13.59–14.94).

## Highlights & Insights

1. **Complementary design of ConvFusion and Spatial Compressor**: ConvFusion enriches query local information while the Spatial Compressor reduces key/value token counts; their combination achieves the optimal quality–speed trade-off.
2. **Necessity of hybrid attention**: Retaining standard attention for image-to-text interactions in MM-DiT is critical; full linearization severely degrades cross-modal understanding.
3. **Normalization factor approximation**: Approximating the theoretical normalization factor with the token count ratio is not only faster but also marginally improves performance — a counterintuitive finding.
4. **Acceleration scales with resolution**: A 43× speedup at 8K resolution genuinely unlocks ultra-high-resolution generation.

## Limitations & Future Work

- Knowledge distillation training is required; the method cannot be plug-and-play applied to existing models.
- MM-EDiT is slower than SANA-MM-DiT on mobile devices, as image-to-text interactions still rely on standard attention.
- Validation is limited to PixArt-Σ and SD-v3.5M; applicability to more recent models such as FLUX remains unverified.
- Quantitative evaluation relies on FID and CLIP Score, without human preference studies.

## Related Work & Insights

- **SANA** was the first to introduce linear attention into DiTs, but placed convolutions in the FFN rather than within the attention mechanism.
- **LinFusion**'s multi-layer transformations are effective on UNets but fail on DiTs; EDiT's ConvFusion addresses this limitation.
- **CLEAR** is the only other linearization work targeting MM-DiTs, but relies on sparse neighborhood attention with poor hardware compatibility.
- The hybrid attention paradigm (partial linear + partial standard attention) is generalizable to other multimodal architectures.

## Rating

⭐⭐⭐⭐ — The method is well-motivated and experiments comprehensively cover both DiT and MM-DiT architectures. The high-resolution acceleration results are impressive (43×). The hybrid attention design provides a practical paradigm for accelerating MM-DiTs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
