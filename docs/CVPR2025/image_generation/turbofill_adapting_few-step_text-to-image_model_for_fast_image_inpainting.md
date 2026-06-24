---
title: >-
  [Paper Note] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting
description: >-
  [CVPR 2025][Image Generation][Image Inpainting] TurboFill proposes a three-step adversarial training scheme to train an inpainting adapter (ControlNet architecture) directly on the few-step distilled diffusion model DMD2. It achieves high-quality image inpainting that outperforms multi-step BrushNet in just 4 inference steps, reducing training costs by over 10 times.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Inpainting"
  - "Few-Step Diffusion Models"
  - "Adversarial Training"
  - "Inpainting Adapter"
  - "Fast Inference"
date: 2026-05-08
content_hash: a75196ceb56ab8f9
---

# TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting

**Conference**: CVPR 2025  
**arXiv**: [2504.00996](https://arxiv.org/abs/2504.00996)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Image Inpainting, Few-Step Diffusion Models, Adversarial Training, Inpainting Adapter, Fast Inference

## TL;DR

TurboFill proposes a three-step adversarial training scheme to train an inpainting adapter (ControlNet architecture) directly on the few-step distilled diffusion model DMD2. It achieves high-quality image inpainting that outperforms multi-step BrushNet in just 4 inference steps, reducing training costs by over 10 times.

## Background & Motivation

**Background**: Diffusion-model-based image inpainting has achieved significant progress. BrushNet uses an adapter with a ControlNet architecture to inject background conditions into a frozen diffusion U-Net, achieving high-quality inpainting. However, the computational overhead of multi-step diffusion inference (typically 20-50 steps) seriously limits practical deployment.

**Limitations of Prior Work**: (1) Multi-step BrushNet inference is slow (requiring 50+ steps), making it unsuitable for real-time applications; (2) directly transferring a pre-trained multi-step BrushNet adapter to the few-step distilled model DMD2 leads to severe artifacts, such as color oversaturated and semantic inconsistency; (3) training the adapter solely with diffusion loss on a few-step model produces blurry, low-quality outputs; (4) the two-stage scheme (training the multi-step adapter first and then distilling it) incurs enormous memory and computational overhead due to the large parameter size of BrushNet (requiring 64 A100 GPUs for 50 hours).

**Key Challenge**: Few-step distilled models can generate images quickly, but the diffusion loss fails to provide sufficient quality supervision signals in few-step settings, resulting in subsequent inpainting quality that is far inferior to multi-step models.

**Goal**: To design an efficient single-stage training scheme that directly trains an inpainting adapter on a few-step distilled model, achieving a quality that matches or even exceeds 50-step multi-step methods with only 4 inference steps.

**Key Insight**: The authors observe that diffusion loss excels at scene understanding but lacks texture details, while GAN loss is proficient in improving texture and details but lacks global scene understanding. Their complementary characteristics inspire a hybrid training scheme.

**Core Idea**: A three-step alternating training process: training the adapter with real diffusion loss on a slow generator (SDXL) to learn denoising directions, training the adapter with GAN loss on a fast generator (DMD2) to improve texture quality, while simultaneously training a diffusion discriminator to distinguish real and fake samples and using a fake diffusion loss to enhance the discriminator's scene understanding.

## Method

### Overall Architecture

TurboFill consists of three components: a fast generator (DMD2 + adapter), a slow generator (SDXL + adapter), and a diffusion discriminator. All three share the same inpainting adapter weights. Training alternates among three steps: Step 1 updates the adapter on the slow generator using a diffusion loss; Step 2 updates the adapter on the fast generator using a GAN loss and a background preservation loss; Step 3 updates the discriminator using real/fake diffusion losses and a discrimination loss. During inference, only the fast generator is used, requiring only 4 inference steps.

### Key Designs

1. **Slow Generator + Real Diffusion Loss (Step 1)**:

    - **Function**: Enables the adapter to learn the correct direction to guide the noisy latent toward the real image distribution.
    - **Mechanism**: Combines the inpainting adapter with the multi-step diffusion model SDXL as the "slow generator." The input is the concatenated noisy latent $x_t$, background image latent $x_{bg}$, and downsampled binary mask $m$. Features produced by the adapter are injected into the SDXL U-Net via residual connections. Training is performed using the standard DDPM diffusion loss $\mathcal{L}^R_{Diff}$ over full timesteps ($T=1000$), updating only the adapter parameters.
    - **Design Motivation**: SDXL acts as a teacher providing rich semantic understanding signals. Training on the full timesteps allows the adapter to learn robust denoising capabilities, laying the foundation for the quality of the fast generator.

2. **Fast Generator + GAN Loss (Step 2)**:

    - **Function**: Enables the adapter to generate high-quality textures and details under few-step inference.
    - **Mechanism**: Combines the adapter with DMD2 to form the "fast generator," using the LCM scheduler with 4 timesteps $\{999, 749, 499, 249\}$ for sampling. A clean latent $\hat{x}_0$ is generated in a single step and fed into the diffusion discriminator to compute the GAN loss $\mathcal{L}_\mathcal{G}$. Meanwhile, a reconstruction loss $\mathcal{L}_{BG} = \|x_0 \odot m - \hat{x}_0 \odot m\|^2_2$ is computed for the background area. Only the adapter is updated.
    - **Design Motivation**: GAN loss directly provides gradient signals based on the quality of generated images, which is effective for enhancing texture sharpness and detail realism—precisely what is most lacking when training few-step models solely with diffusion loss. The background preservation loss prevents obvious boundaries between the inpainted region and the background.

3. **Diffusion Discriminator + Fake Diffusion Loss (Step 3)**:

    - **Function**: Trains the discriminator to distinguish between real and synthesized latents while enabling it to understand scene structure.
    - **Mechanism**: The diffusion discriminator consists of an SDXL encoder, an auxiliary encoder, and a convolutional classifier. The auxiliary encoder receives the same three-channel input as the adapter. Standard GAN discriminator loss $\mathcal{L}_\mathcal{D}$ is used to distinguish between real $x_t$ and fake $\hat{x}_t$. A key innovation is the introduction of the "fake diffusion loss" $\mathcal{L}^F_{Diff}$—using an auxiliary decoder to predict noise on fake latents, allowing the discriminator to learn scene understanding simultaneously.
    - **Design Motivation**: Discriminators with pure GAN loss focus only on local texture differences and lack global scene understanding, which may lead to the generation of unrelated objects. The fake diffusion loss forces the discriminator to understand the overall structure of the image, making the gradient signals from the GAN loss more meaningful.

### Loss & Training

- Step 1: $\mathcal{L}^R_{Diff}$ updates the adapter (real diffusion loss on the slow generator).
- Step 2: $\lambda_1 \mathcal{L}_\mathcal{G} + \lambda_2 \mathcal{L}_{BG}$ updates the adapter (GAN + background loss on the fast generator).
- Step 3: $\mathcal{L}^F_{Diff} + \lambda_3 \mathcal{L}_\mathcal{D}$ updates the discriminator (fake diffusion + discriminator loss).
- Hyperparameters: $\lambda_1=10^{-3}$, $\lambda_2=10^{-1}$, $\lambda_3=10^{-2}$.
- 8 A100 GPUs, batch size 2, gradient accumulation 4, lr=1e-5, AdamW, 40K iterations.
- Training data: 1.2 million internet images + Florence-2 region captions + SAM2 segmentation masks (LocalCaptionData).

## Key Experimental Results

### Main Results

| Method | Steps | Q-Align (mask) | CLIPIQA+ (mask) | Q-Align (whole) | CLIP Sim |
|------|------|------------|-----------|------------|------|
| TurboFill | 4 | 4.570 | 0.733 | 4.719 | 25.35 |
| BrushNet* (50 steps) | 50 | 4.469 | 0.714 | 4.531 | 25.39 |
| BrushNet* (4 steps) | 4 | 4.184 | 0.657 | 4.449 | 25.34 |
| PowerPaint V2 (50 steps) | 50 | **4.777** | **0.777** | 4.723 | **26.26** |
| SDXL-Inpainting (50 steps) | 50 | 4.246 | 0.667 | 4.617 | 24.85 |

Consistent conclusions are drawn on HumanBench: TurboFill (4 steps) is close to the 50-step PowerPaint V2 in mask region quality and substantially outperforms the synchronized (4-step) BrushNet.

### Ablation Study

| Configuration | Q-Align (mask) | CLIPIQA+ (mask) | TOPIQ (mask) | Description |
|------|------------|-----------|----------|------|
| TurboFill (Full) | 4.570 | 0.733 | 5.275 | Full three-step training |
| - $\mathcal{L}_{BG}$ | 4.367 | 0.686 | 5.026 | Background color shift, obvious boundaries |
| - $\mathcal{L}^F_{Diff}$ | 4.188 | 0.655 | 4.870 | Conflicting elements appear |
| - $\mathcal{L}^R_{Diff}$ | 4.066 | 0.632 | 4.850 | Poor texture, inconsistent background |

### Key Findings

- **4-step TurboFill outperforms 50-step BrushNet on most metrics**, proving the effectiveness of the three-step adversarial training scheme.
- Ablation experiments clearly demonstrate the complementarity of the three losses: diffusion loss provides scene understanding, GAN loss provides texture details, and background loss provides region consistency.
- Using LocalCaptionData significantly improves text alignment, with CLIP Sim increasing from 21.6 to 25.4.
- Although PowerPaint V2 scores the highest on IQA metrics, qualitative analysis shows that its inpainting results are overly sharp and sometimes exhibit structural distortions (such as a two-headed tiger). Thus, its overall image quality is less coordinated than TurboFill.
- Training efficiency is improved by 10x: TurboFill requires only 8 A100 $\times$ 50h = 400 GPU-hours, whereas the two-stage scheme requires 8 V100 $\times$ 72h + 64 A100 $\times$ 50h $\approx$ 3776 GPU-hours.

## Highlights & Insights

- **The design of the three-step adversarial training is delicate**: the slow generator learns the direction, the fast generator learns the quality, and the discriminator learns the structure, forming a complete training loop. This paradigm can be extended to other task adapters requiring training on few-step models with conditional guidance.
- **Training the discriminator with a fake diffusion loss** is a profound insight. A pure GAN discriminator only focuses on low-level texture differences; adding diffusion loss allows it to understand high-level semantics simultaneously, thereby providing more meaningful gradient signals to the generator.
- **The construction of LocalCaptionData** is inspiring: using Florence-2 for dense region captioning and SAM2 for segmentation to automatically construct a large-scale image-mask-local description triplet database.

## Limitations & Future Work

- It still requires two models (SDXL and DMD2) for training. Although only DMD2 is needed for inference, the training memory footprint is still large.
- Lack of comparison with the latest FLUX or SD3 series models.
- Although validation benchmarks like DilationBench and HumanBench are introduced, there is still a lack of effective quantitative metrics for evaluating consistency between the inpainted region and the background.
- Future work: Applying the three-step training scheme to other conditional generation tasks (such as super-resolution and style transfer); exploring the limits of 1-step inference; and validating scalability on larger models (e.g., SDXL $\rightarrow$ FLUX).

## Related Work & Insights

- **vs BrushNet**: BrushNet achieves high-quality inpainting using the ControlNet architecture but requires 50 inference steps. TurboFill trains directly on a few-step model to surpass 50-step BrushNet in 4 steps. The key difference lies in the training strategy rather than the architecture.
- **vs Diffusion Distillation Methods (LCM, DMD2, etc.)**: The traditional distillation pipeline is to train a full model first and then compress the steps. TurboFill bypasses this distillation stage, directly training on the distilled few-step model, making it 10 times more efficient.
- **vs PowerPaint V2**: PowerPaint exhibits higher IQA metrics but is visually over-sharpened, and since it is based on SD1.5, it cannot be easily transferred to few-step models. TurboFill excels in coordination and naturalness.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-step adversarial training scheme is novel and effective, and the fake diffusion loss is a highly insightful design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two new benchmarks, comparisons with multiple methods, and complete step-by-step ablation analyses are provided.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described and the illustrations are intuitive.
- Value: ⭐⭐⭐⭐ It achieves high-quality inpainting on a few-step diffusion model for the first time, offering great value for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [\[CVPR 2025\] RAD: Region-Aware Diffusion Models for Image Inpainting](rad_region-aware_diffusion_models_for_image_inpainting.md)
- [\[CVPR 2026\] InverFill: One-Step Inversion for Enhanced Few-Step Diffusion Inpainting](../../CVPR2026/image_generation/inverfill_one-step_inversion_for_enhanced_few-step_diffusion_inpainting.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

</div>

<!-- RELATED:END -->
