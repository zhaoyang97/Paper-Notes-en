---
title: >-
  [Paper Note] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion
description: >-
  [CVPR 2025][Image Generation][Image Editing] This paper proposes SwiftEdit, the first text-guided image editing tool based on a one-step diffusion model. By leveraging a one-step inversion network trained in two stages and an attention-rescaling mask editing technique, it achieves image editing within 0.23 seconds, which is at least 50 times faster than multi-step methods.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "One-step Diffusion"
  - "Inversion Network"
  - "Attention Rescale"
  - "Real-time Editing"
date: 2026-05-08
content_hash: d4b6e1db12f9168f
---

# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2412.04301](https://arxiv.org/abs/2412.04301)  
**Code**: [https://swift-edit.github.io/](https://swift-edit.github.io/)  
**Area**: Image Generation  
**Keywords**: Image Editing, One-step Diffusion, Inversion Network, Attention Rescale, Real-time Editing

## TL;DR

This paper proposes SwiftEdit, the first text-guided image editing tool based on a one-step diffusion model. By leveraging a one-step inversion network trained in two stages and an attention-rescaling mask editing technique, it achieves image editing within 0.23 seconds, which is at least 50 times faster than multi-step methods.

## Background & Motivation

**Background**: Text-guided image editing typically relies on multi-step diffusion models—first inverting the source image to the noise space using methods like DDIM Inversion, and then performing editing through attention manipulation during the multi-step denoising process. Recent few-step methods (e.g., TurboEdit, ReNoise) reduce the step count to 3-4 steps.

**Limitations of Prior Work**: (1) Multi-step methods require complete inversion (25-50 steps) + sampling (25-50 steps), taking 12-134 seconds in total, which fails to meet real-time application demands. (2) Few-step methods, though faster (1-5 seconds), are still not instantaneous, and their editing quality may fall short of multi-step methods. (3) Existing one-step diffusion models lack compatible inversion methods—both DDIM Inversion and Null-text Inversion rely on multi-step iterations.

**Key Challenge**: One-step diffusion models generate quickly but do not inherently support inversion. Image editing requires finding the representation of the source image in the noise space, whereas current inversion methods all require multi-step iterations, negating the speed advantage of one-step generation.

**Goal**: To achieve genuine one-step inversion + one-step editing, reaching sub-second image editing speeds.

**Key Insight**: Drawing inspiration from encoder-based approaches in GAN Inversion—training a neural network to directly map an image back to the latent space to bypass iterative optimization. While GAN Inversion is restricted to specific domains, the latent space of diffusion models is much more generalized.

**Core Idea**: Train an inversion network that shares a symmetric architecture with the one-step generator. Through a two-stage training strategy (synthetic data first, then real data), the network is enabled to invert any arbitrary image to an editable noise space in a single step. During editing, an automatically generated mask and attention rescaling are utilized to achieve precise local editing.

## Method

### Overall Architecture

Based on SwiftBrushv2 (a one-step text-to-image model). The inversion network $\mathbf{F}_\theta$ maps the source image latent $\mathbf{z}$ and the text condition to the inverted noise $\hat{\boldsymbol{\epsilon}}$ in a single step. This is then combined with the editing text condition to generate the edited image in a single step using the generator $\mathbf{G}^{\text{IP}}$ equipped with an IP-Adapter. The entire process requires only a single forward pass.

### Key Designs

1. **One-Step Inversion Network + Two-Stage Training**:

    - **Function**: Maps any arbitrary image to the editable noise space of the one-step generator in a single step.
    - **Mechanism**: The inversion network shares the UNet architecture of SwiftBrushv2 and is initialized with its weights. An IP-Adapter branch is introduced to provide image conditioning $\mathbf{c_x}$, easing the burden on the inverted noise $\hat{\boldsymbol{\epsilon}}$ to encode too many image details. **Stage 1** (synthetic data): Inures the generator to sample $(\boldsymbol{\epsilon}, \mathbf{z})$ pairs, and trains the model using a reconstruction loss $\|\mathbf{z} - \hat{\mathbf{z}}\|_2^2$ plus a regression loss $\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|_2^2$ to bring the inverted noise closer to a standard normal distribution. **Stage 2** (real data): Replaces pixel reconstruction with the perceptual loss DISTS, and integrates an SDS-based regularization $\nabla_\theta \mathcal{L}_{\text{regu}}$ to prevent the inverted noise from deviating from the normal distribution.
    - **Design Motivation**: Directly training on real data with a reconstruction loss causes the inverted noise to encode patterns of the source image, leading to a deviation from the normal distribution and damaging editability. SDS regularization constrains the noise distribution while preserving reconstruction quality.

2. **Self-Guided Editing Mask**:

    - **Function**: Automatically localizes editing regions without requiring a user-provided mask.
    - **Mechanism**: The trained inversion network is utilized to predict two noises, $\hat{\boldsymbol{\epsilon}}_{\text{source}}$ and $\hat{\boldsymbol{\epsilon}}_{\text{edit}}$, using the source prompt and editing prompt respectively on the same image. The difference map between the two is thresholded to obtain the editing mask $M$. This can be performed within a single batchified forward pass.
    - **Design Motivation**: The inverted noise is sensitive to text conditions—different prompts yield distribution differences concentrated primarily on the regions that need editing, naturally providing a localization signal.

3. **Attention Rescale Mask Editing (ARaM)**:

    - **Function**: Achieves precise local editing while keeping the background unchanged.
    - **Mechanism**: The cross-attention formulation of the IP-Adapter is modified, replacing the global image condition scaling factor $s_\mathbf{x}$ with region-specific scaling: $\mathbf{h}_l = s_y \cdot M \cdot \text{Attn}(Q, K_y, V_y) + s_{\text{edit}} \cdot M \cdot \text{Attn}(Q, K_\mathbf{x}, V_\mathbf{x}) + s_{\text{non-edit}} \cdot (1-M) \cdot \text{Attn}(Q, K_\mathbf{x}, V_\mathbf{x})$. Decreasing $s_{\text{edit}}$ in the editing region allows more editing freedom, while increasing $s_{\text{non-edit}}$ in non-editing regions enhances fidelity. $s_y$ controls the editing strength.
    - **Design Motivation**: A single global scaling factor faces a trade-off—high values preserve fidelity but restrict editing, while low values provide flexibility at the cost of losing the background. Region-specific scaling decouples editing flexibility from background fidelity.

### Loss & Training

Stage 1: $\mathcal{L}^{\text{stage1}} = \mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{regr}}$ ($\lambda=1$), training the inversion network and the IP-Adapter. Stage 2: $\mathcal{L}^{\text{stage2}} = \mathcal{L}_{\text{perceptual}} + \lambda \mathcal{L}_{\text{regu}}$ ($\lambda=1$), training only the inversion network and freezing the IP-Adapter. During inference, $s_{\text{edit}}=0.3$, $s_{\text{non-edit}}=2.0$, and $s_y=3.0$.

## Key Experimental Results

### Main Results

Comparison of editing quality and speed on PIE-Bench:

| Method | Type | PSNR↑ | MSE×$10^4$↓ | CLIP Whole↑ | Time (s)↓ |
|---|---|---|---|---|---|
| NT-Inv + P2P | Multi-step (50) | 27.03 | 35.86 | 24.75 | 134.06 |
| DDIM + PnP | Multi-step (50) | 22.28 | 83.64 | 25.41 | 12.62 |
| TurboEdit | Few-step (4) | 22.43 | 9.48 | 25.49 | 1.32 |
| ICD | Few-step (4) | 26.93 | 3.32 | 22.42 | 1.62 |
| **SwiftEdit** | **One-step** | **23.33** | **6.60** | **25.16** | **0.23** |

### Ablation Study

| Configuration | PSNR↑ | MSE×$10^4$↓ | CLIP|
|---|---|---|---|
| w/o IP-Adapter | Lower | Higher | Lower |
| w/o Stage 2 Regularization | Lower | — | Lower (poor editability) |
| Global $s_\mathbf{x}$ (w/o ARaM) | Low | High | Unstable |
| **Full SwiftEdit** | **23.33** | **6.60** | **25.16** |

### Key Findings

- SwiftEdit takes only 0.23 seconds, accelerating editing by 55× compared to the fastest multi-step method (PnP, 12.62s) and by 5.7× compared to the few-step method TurboEdit (1.32s).
- It strikes a better balance between background preservation (PSNR 23.33) and editing semantics (CLIP 25.16) than most multi-step methods.
- The IP-Adapter branch is critical for mitigating overfitting in inverted noise; without it, the inverted noise would encode excessive source image details.
- SDS regularization effectively prevents the noise distribution from drifting during Stage 2 training, ensuring editing flexibility.
- The self-guided mask incurs no additional computational overhead and provides high enough quality for precise local editing.

## Highlights & Insights

1. **"One-Step Inversion + One-Step Editing" Paradigm**: It establishes the first complete one-step image editing pipeline, pushing the editing speed to a practical level (0.23s) and opening up possibilities for mobile deployment.
2. **Ingenious Application of SDS Regularization**: Shifting SDS from 3D generation (point optimization) to inversion (constraining the noise distribution). This borrowing demonstrates the generality and versatility of SDS.
3. **Regional Decoupling via Attention Rescaling**: Decoupling behaviors between the editing and non-editing regions within the same attention layer using a mask, which is simple yet highly effective.

## Limitations & Future Work

- The editing quality (PSNR/CLIP) does not yet completely outperform multi-step methods, reflecting a speed-quality trade-off.
- It is based on SwiftBrushv2 at a 512×512 resolution; high-resolution editing requires further exploration.
- The self-guided mask relies on the discrepancy between the source and target prompts, which might be less accurate for edits with minor semantic changes.
- Future work could extend this approach to video editing and higher resolutions.

## Related Work & Insights

- **vs TurboEdit**: TurboEdit achieves fast editing using 4-step SDXL Turbo and an offset noise schedule; SwiftEdit further compresses this to 1 step, resulting in a 5.7× speedup.
- **vs ICD**: ICD achieves precise inversion in 3-4 steps using consistency distillation (PSNR 26.93), exhibiting better background preservation; SwiftEdit sacrifices a minor amount of fidelity in exchange for extreme speed.
- **vs GAN Inversion**: Encoder-based approaches in GAN Inversion are confined to specific domains; SwiftEdit leverages the generality of diffusion models to transcend domain limitations.
- **Insight**: The inversion problem in one-step models can be approached by training a "backward network" rather than performing iterative optimization.

## Rating

- Novelty: 8/10 — The first one-step diffusion editing framework, featuring a novel two-stage training scheme and SDS regularization.
- Experimental Thoroughness: 7/10 — Detailed evaluation on PIE-Bench, though it lacks user studies and a wider variety of editing types.
- Writing Quality: 8/10 — Clear motivation, intuitive illustrations, and detailed methodological descriptions.
- Value: 8/10 — Elevates editing speeds to practical viability, holding significant importance for mobile deployment and real-time applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](../../CVPR2026/image_generation/chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)

</div>

<!-- RELATED:END -->
