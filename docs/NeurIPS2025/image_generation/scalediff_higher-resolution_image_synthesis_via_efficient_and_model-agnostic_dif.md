---
title: >-
  [Paper Note] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion
description: >-
  [NeurIPS 2025][Image Generation][High-resolution generation] ScaleDiff is a framework that eliminates redundant overlap computation in conventional patch-based methods via Neighborhood Patch Attention (NPA). Combined with Latent Frequency Mixing (LFM) and Structure Guidance (SG), it extends pretrained diffusion models to high resolutions (e.g., 4096²) without any additional training, achieving state-of-the-art quality among training-free methods and significant inference acceleration (8.9× faster than DemoFusion) on both U-Net and DiT architectures.
tags:
  - NeurIPS 2025
  - Image Generation
  - High-resolution generation
  - Training-free
  - Patch attention
  - Frequency mixing
  - Structure guidance
date: 2026-05-08
content_hash: 2c7e47fa171d131e
---

# ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2510.25818](https://arxiv.org/abs/2510.25818)
**Code**: None
**Area**: Diffusion Models / High-Resolution Image Generation
**Keywords**: High-resolution generation, Training-free, Patch attention, Frequency mixing, Structure guidance

## TL;DR

ScaleDiff is a framework that eliminates redundant overlap computation in conventional patch-based methods via Neighborhood Patch Attention (NPA). Combined with Latent Frequency Mixing (LFM) and Structure Guidance (SG), it extends pretrained diffusion models to high resolutions (e.g., 4096²) without any additional training, achieving state-of-the-art quality among training-free methods and significant inference acceleration (8.9× faster than DemoFusion) on both U-Net and DiT architectures.

## Background & Motivation

Text-to-image diffusion models perform well at standard resolutions (e.g., 1024²), but suffer severe degradation—manifesting as repetitive patterns and structural distortions—when generating images beyond their training resolution (e.g., 2048² or 4096²). Retraining on high-resolution data is prohibitively expensive, motivating research into training-free approaches for resolution extrapolation.

Existing training-free high-resolution methods share several fundamental limitations:

**Poor architectural compatibility**: Methods such as ScaleCrafter rely on dilated convolution modifications specific to U-Net and cannot be directly applied to DiT architectures.

**High computational overhead**: Patch-based methods such as MultiDiffusion require extensive overlap regions to ensure smooth transitions, inflating FLOPs in non-self-attention layers by approximately 4×.

**Loss of detail and over-smoothing**: Editing-based methods such as DiffuseHigh perform upsampling in RGB space; because such upsampling resembles the resize operations seen during training, the model tends to produce overly smooth textures.

The core mechanism of ScaleDiff is to **apply patch partitioning only within self-attention layers (with non-overlapping queries), while processing the full-resolution tensor directly in all other layers—thereby eliminating redundant computation while maintaining smooth patch boundaries**. A latent-space frequency mixing scheme is additionally employed to guide the denoising process toward fine-grained detail synthesis.

## Method

### Overall Architecture

ScaleDiff adopts an iterative "upsample → add noise → denoise" (SDEdit) pipeline. Starting from a low-resolution image latent, LFM upsamples it to a high-resolution reference latent $Z_\text{ref}$; noise is then injected up to an intermediate timestep $\tau$; and at each denoising step, NPA is applied for efficient denoising while SG enforces global structural consistency. The overall progression follows $1024^2 \to 2048^2 \to 4096^2$.

### Key Designs

1. **Neighborhood Patch Attention (NPA)**: The core innovation distinguishes between self-attention layers and non-self-attention layers. For linear layers, convolutional layers, cross-attention layers, and similar operations—which are token-wise or local and thus resolution-agnostic—the full high-resolution tensor $Z_t$ is processed directly, avoiding redundant computation from patch overlap. For self-attention layers, NPA partitions the queries into **non-overlapping** patches (of size $h/2 \times w/2$), while each query patch attends to a **larger, overlapping** key/value neighborhood window (of size $h \times w$). This keeps the total number of query tokens constant (eliminating duplication) while the overlapping key/value regions ensure smooth transitions at patch boundaries. Theoretical analysis shows that the self-attention FLOPs of NPA scale as $s^2 h^2 w^2 d$, substantially lower than MultiDiffusion's $(2s-1)^2 h^2 w^2 d$, and the non-self-attention FLOPs are likewise reduced from $(2s-1)^2$ back to $s^2$.

2. **Latent Frequency Mixing (LFM)**: LFM resolves the dilemma posed by different upsampling strategies. RGB-space upsampling ($Z_\text{RU}$) yields latents rich in frequency content but biases the model toward over-smooth outputs; direct latent-space upsampling ($Z_\text{LU}$) deviates from the training distribution (beneficial for avoiding smoothness) but lacks high-frequency components and introduces decoding artifacts. LFM combines the complementary strengths of both: the high-frequency components of $Z_\text{RU}$ (sharp details) are fused with the low-frequency components of $Z_\text{LU}$ (distribution shift away from smoothing), yielding the reference latent $Z_\text{ref} = Z_\text{RU}^h + Z_\text{LU}^l$.

3. **Structure Guidance (SG)**: At each denoising timestep $t$, a clean prediction $Z_{0|t}$ is estimated from the noisy latent; its low-frequency components are then blended with those of $Z_\text{ref}$ using a time-varying mixing coefficient $\gamma_t$, constraining the denoising trajectory to preserve global structure while leaving the model free to synthesize high-frequency details. Unlike prior work, ScaleDiff performs SG in latent space rather than RGB space, reducing unnecessary encoding/decoding overhead.

### Loss & Training

ScaleDiff is a fully training-free inference-time method and involves no training or fine-tuning. The key hyperparameter is the noise timestep $\tau$: it is set to 400 for SDXL and 600 for FLUX, achieving the best balance between structural fidelity and detail synthesis.

## Key Experimental Results

### Main Results

Evaluation is conducted on 1,000 text–image pairs sampled from LAION-5B, using FID, KID, IS, their patch-level variants, and CLIP Score.

| Model/Resolution | Method | FID↓ | KID↓ | FIDp↓ | ISp↑ | CLIP↑ | Time (s)↓ |
|---|---|---|---|---|---|---|---|
| SDXL/4096² | DemoFusion | 65.06 | 0.0041 | 41.29 | 19.59 | 32.61 | 1005 |
| SDXL/4096² | DiffuseHigh | 63.91 | 0.0034 | 42.30 | 19.54 | 32.68 | 325 |
| SDXL/4096² | AccDiffusion v2 | 64.64 | 0.0037 | 40.92 | 18.42 | 32.34 | 1599 |
| SDXL/4096² | **ScaleDiff** | **61.87** | **0.0025** | **38.89** | 20.41 | **33.04** | **113** |
| FLUX/4096² | FLUX+BSRGAN | 64.76 | 0.0051 | 49.30 | 16.92 | 31.19 | 34 |
| FLUX/4096² | **ScaleDiff** | **64.06** | **0.0044** | **44.29** | **17.41** | 31.14 | 407 |

ScaleDiff requires only 113 seconds on SDXL at 4096², making it the fastest among all training-free methods and **8.9× faster** than DemoFusion.

### Ablation Study

| Attention | LFM | SG | FID↓ | FIDp↓ | Time (s)↓ | Note |
|---|---|---|---|---|---|---|
| Base | ✓ | ✓ | 61.91 | 39.94 | 185 | Direct high-res inference; local artifacts |
| MultiDiffusion | ✓ | ✓ | 61.71 | 38.08 | 239 | Best quality but high computational cost |
| **NPA** | ✓ | ✓ | 61.87 | 38.89 | **113** | Near-MultiDiffusion quality at 2.1× speed |
| NPA | ✗ | ✗ | 64.17 | 41.55 | 113 | Without LFM+SG; severe repetition artifacts |
| NPA | ✓ | ✗ | 62.34 | 39.49 | 113 | Without SG; improved detail but repetition remains |
| NPA | ✗ | ✓ | 64.12 | 41.50 | 113 | Without LFM; consistent structure but over-smooth |

### Key Findings

- NPA achieves **2.8× speedup** over MultiDiffusion on FLUX (407 s vs. 1148 s) at comparable quality.
- LFM and SG address distinct failure modes (detail vs. structure) and are mutually complementary.
- The optimal noise timestep $\tau$ differs across architectures: SDXL favors $\tau=400$, while FLUX favors $\tau=600$.
- ScaleDiff is genuinely **architecture-agnostic**: it is effective on SDXL (U-Net), FLUX (DiT), and Lumina-T2X.

## Highlights & Insights

- **Precise diagnosis of computational redundancy in patch methods**: The observation that non-self-attention layers are inherently resolution-agnostic and require no patch partitioning is simple yet highly valuable.
- **Elegant complementary design in frequency mixing**: RGB-space and latent-space upsampling have complementary strengths and weaknesses; separating them in the frequency domain yields a superior combination of both.
- **Genuine model-agnosticism**: ScaleDiff is effective on both U-Net and DiT backbones, whereas most prior methods support only one architecture or perform poorly on the other.
- The design of non-overlapping queries paired with overlapping key/value regions in NPA resolves boundary artifacts while preserving computational efficiency.

## Limitations & Future Work

- As a training-free method, generation quality is fundamentally bounded by the capabilities of the underlying diffusion model.
- Patch-based approaches inherently rely on the model's prior knowledge of cropped image regions, which may cause local content inconsistencies when generating close-up images.
- Repetitive artifacts in background regions remain a persistent challenge common to patch-based methods.
- The paper does not explore more complex high-resolution generation settings such as video synthesis or 3D scene generation.

## Related Work & Insights

- MultiDiffusion pioneered the patch-based paradigm but suffers from severe computational redundancy; NPA in ScaleDiff constitutes a natural improvement over this approach.
- DiffuseHigh identified the over-smoothing problem caused by RGB-space upsampling; LFM provides an elegant solution to this issue.
- ScaleCrafter's dilated convolution approach is efficient but architecture-specific, highlighting the importance of model-agnostic design.
- Core insight: precisely identifying which operations within a patch-based method require localization and which do not can substantially reduce computational redundancy.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](../../ICCV2025/image_generation/infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)
- [\[NeurIPS 2025\] BlurDM: A Blur Diffusion Model for Image Deblurring](blurdm_a_blur_diffusion_model_for_image_deblurring.md)

</div>

<!-- RELATED:END -->
