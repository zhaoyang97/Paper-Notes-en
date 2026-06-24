---
title: >-
  [Paper Note] Color Alignment in Diffusion
description: >-
  [CVPR 2025][Image Generation][Color Alignment] This work proposes a color alignment diffusion method. By projecting intermediate samples or predictions into a conditional color space (using nearest-neighbor color mapping), the diffusion model strictly follows a given color distribution (color values and proportions) while retaining structural generation freedom. It supports three settings: retraining, fine-tuning, and zero-shot.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Color Alignment"
  - "Diffusion Models"
  - "Fine-grained Control"
  - "Color-conditioned Generation"
  - "Zero-shot"
date: 2026-05-08
content_hash: 092c8f2ba31ec68f
---

# Color Alignment in Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2503.06746](https://arxiv.org/abs/2503.06746)  
**Code**: None (not mentioned)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Color Alignment, Diffusion Models, Fine-grained Control, Color-conditioned Generation, Zero-shot

## TL;DR
This work proposes a color alignment diffusion method. By projecting intermediate samples or predictions into a conditional color space (using nearest-neighbor color mapping), the diffusion model strictly follows a given color distribution (color values and proportions) while retaining structural generation freedom. It supports three settings: retraining, fine-tuning, and zero-shot.

## Background & Motivation

**Background**: Diffusion models have made tremendous progress in image synthesis quality. Controllable generation methods like ControlNet support structural constraints (edge maps, semantic maps, etc.), and methods like IP-Adapter support style references. However, fine-grained control over pixel colors remains difficult—existing methods frequently generate pixels that fall outside the target color range.

**Limitations of Prior Work**: IP-Adapter and Style-Aligned transfer spatial and color information of reference images together, which completely fails when there is a significant domain gap between the reference image and the target prompt (e.g., using the colors of flowers to generate a car). ControlNet-Color treats color as a weak guidance, failing to accurately preserve color values and proportions. The core issue is that color conditioning does not participate in the sampling process (serving only as a parallel input), causing the denoising model to bias toward unconditional generation.

**Key Challenge**: There is a fundamental tension between three objectives: strictly controlling color values and proportions (accuracy), allowing colors to freely reorganize spatially to form new structures (disentanglement), and maintaining generation quality and diversity.

**Goal**: To achieve fine-grained color-conditioned generation in diffusion models—specifying color values and proportions without dictating the spatial arrangement of colors, thereby allowing the model to construct structures freely.

**Key Insight**: Directly intervene in the sampling process of diffusion. Before passing $\mathbf{x}_t$ into the denoising model, every pixel's color is replaced with its nearest counterpart in the conditional colors using nearest-neighbor mapping. Since the model always "sees" the target colors, it naturally generates images strictly confined within these color boundaries.

**Core Idea**: Apply pixel-level nearest-neighbor color mapping to intermediate samples during the diffusion process, constraining them to the conditional color manifold, while achieving color-structure disentanglement through random spatial permutation of the color condition.

## Method

### Overall Architecture
The framework accepts a color condition $\mathbf{c}$ (such as an image or a hand-drawn palette, defining color values and proportions) and an optional text prompt. During diffusion, every pixel of the noisy sample $\mathbf{x}_t$ is mapped at each step to the nearest color value in $\mathbf{c}$, denoted as $f(\mathbf{x}_t, \mathbf{c})$, before being input into the denoising model. During training, $\mathbf{c}$ is obtained directly from random pixel permutations $\psi(\mathbf{x}_0)$ of the training image $\mathbf{x}_0$ (requiring no extra training data). Three settings are supported: retraining from scratch, fine-tuning pre-trained Stable Diffusion, and zero-shot (training-free).

### Key Designs

1. **Pixel-Level Nearest-Neighbor Color Alignment Function $f(\mathbf{x}_t, \mathbf{c})$**

    - **Function**: Replaces each pixel color of intermediate diffusion samples with the nearest color from the condition, strictly limiting the generated color range.
    - **Mechanism**: $f(\mathbf{x}_t, \mathbf{c})[p] = \arg\min_{\mathbf{c}[q]} \|\mathbf{x}_t[p] - \mathbf{c}[q]\|_2^2$, meaning nearest-neighbor lookup is performed independently for each pixel. An equivalent adapted noise $\epsilon' = (f(\mathbf{x}_t, \mathbf{c}) - \sqrt{\bar{\alpha}_t}\mathbf{x}_0) / \sqrt{1-\bar{\alpha}_t}$ replaces the original noise as the training target. Crucially, only the denoising model's input is modified ($\mathbf{x}_t$ → $f(\mathbf{x}_t, \mathbf{c})$), while the forward process and sampling steps (Eq. 4) remain unchanged.
    - **Design Motivation**: To inject the color condition into the sampling process itself rather than just as an auxiliary input, forcing the model to operate within the color manifold and inherently eliminating out-of-range colors.

2. **Color-Space Disentanglement (Random Pixel Permutation $\psi$)**

    - **Function**: Disrupts the spatial structure in the color condition, forcing the model to learn only color values and proportions.
    - **Mechanism**: During training, the color condition $\mathbf{c} = \psi(\mathbf{x}_0)$ is constructed by randomly permuting the pixels of the training image. This preserves all color values and proportions while completely randomizing the spatial layout. The model must autonomously decide where to allocate colors spatially based on the text prompt, thereby disentangling color from structure. An additional benefit is that paired training data is not required (automatically constructed from training images).
    - **Design Motivation**: If the spatial structural information is not scrambled, the model will merely replicate the structure of the reference image, failing to reassemble the colors into new object shapes.

3. **Zero-Shot Approximation**

    - **Function**: Achieves color alignment without any training.
    - **Mechanism**: In each sampling step, a one-to-one nearest-neighbor color mapping $g(\hat{\mathbf{x}}_0, \mathbf{c})$ (optimal transport style, using each pixel in $\mathbf{c}$ exactly once) is applied to the unconditionally predicted $\hat{\mathbf{x}}_0$. The mapped result then replaces $\hat{\mathbf{x}}_0$ to guide the subsequent sampling steps. Although mapped results in the early steps ($t > 0.2T$) can be messy, the pre-trained diffusion model is capable of self-correction. In the late steps ($t < 0.2T$), the alignment is paused to allow detail refinement.
    - **Design Motivation**: To bypass training/fine-tuning overhead at the cost of some detail quality (resulting in flatter lighting or overly uniform textures).

### Loss & Training
Image-space retraining: Standard diffusion loss but replacing the original noise with the adapted noise $\epsilon'$. Latent-space fine-tuning: Fine-tune on Stable Diffusion for 160K steps, during which training images $\mathbf{x}_0$ are blurred beforehand to prevent high-frequency structures from leaking into the latents, and the alignment is paused in late steps. Inference uses 50 steps with a CFG scale of 5. Hardware used: 2 × RTX 3090.

## Key Experimental Results

### Main Results

| Method | FID↓ | CLIP Score↑ | CD-A↓ (Color Accuracy) | CD-C↓ (Completeness) |
|------|------|------------|-----------------|---------------|
| **Ours (Image; Re-train)** | 57.5/45.7 | — | **0.00/0.00** | 4.12/3.65 |
| **Ours (Latent; Fine-tune)** | 86.7/69.4 | **29.0/27.4** | 73.9/4.98 | 15.8/4.87 |
| Ours (Latent; Zero-shot) | 104/77.9 | 28.3/27.4 | 35.2/2.68 | 3.19/3.93 |
| IP-Adapter | 202/50.9 | 25.5/22.0 | 145/10.1 | 106/13.9 |
| ControlNet-Color | 105/81.0 | 22.9/23.4 | 129/13.1 | 49.8/6.54 |
| Style-Aligned | —/— | —/— | Poor | Poor |

(Format: Oxford-flower / Emoji datasets)

### Ablation Study

| Configuration | CD-A↓ | CLIP↑ | Description |
|------|-------|-------|------|
| No Color Alignment (DDPM) | 40.6/42.4 | — | Out-of-range colors |
| Color Alignment (with spatial structure) | — | Low | Replicates reference structure |
| **Color Alignment + Random Permutation** | **0.00/0.00** | High | Accurate color + structural freedom |

### Key Findings
- Image-space retraining can achieve CD-A = 0 (perfect color accuracy), indicating that the color alignment function fundamentally constrains the color range.
- The CLIP score (29.0 / 27.4) is significantly higher than all baselines (IP-Adapter 25.5 / 22.0), proving successful color disentanglement—where the model determines structures based on text instead of the reference image.
- Latent-space fine-tuning only increases inference overhead by 3-6%.
- The zero-shot version performs close to the fine-tuned version on in-the-wild image color conditions, but suffers a noticeable drop in quality under hand-drawn color conditions.
- Hand-drawn color conditions can be easily edited (scaling color proportions, adjusting color values, adding new colors), enabling intuitive creative control.

## Highlights & Insights
- **Injecting the color condition into the sampling process** rather than treating it merely as an input is the key to precise control—the model operates on the color manifold, making it fundamentally impossible to generate out-of-range colors.
- **Random pixel permutation** as a data-free way of constructing color conditions is extremely clever—seamlessly obtaining perfectly aligned training pairs while naturally achieving color-structure disentanglement.
- Methodologically, this follows the design of "constraining the output by constraining the input," which can be transferred to other generation tasks requiring fine-grained constraints (e.g., texture, frequency spectrum).

## Limitations & Future Work
- Inability to control the spatial location of colors (it only controls values and proportions, not which regions correspond to which colors).
- The zero-shot version has lower quality (flat lighting, uniform textures, lack of shadows).
- High-frequency structures in the latent space can leak non-target colors, necessitating a blurring pre-processing step as a workaround.
- The experimental datasets are relatively small (7K images at 64×64 resolution for image space, 300K images at 512×512 resolution for latent space).

## Related Work & Insights
- **vs IP-Adapter**: IP-Adapter injects reference image features via cross-attention, causing color and spatial information to be coupled. In contrast, this work achieves decoupling by applying color mapping in the sampling process and permuting spatial layouts.
- **vs ControlNet-Color**: ControlNet treats color as a weak spatial guidance, which cannot guarantee color precision. This work directly enforces color values during sampling.
- **vs Style-Aligned**: Style-Aligned transfers style via shared attention, but suffers from severe domain-crossing failures.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the color alignment function and random permutation is highly novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐ Qualitative results are rich, but the datasets are small and there is a lack of user studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is formulated clearly with mathematical descriptions, and the three settings are introduced in a well-paced manner.
- Value: ⭐⭐⭐⭐ Holds direct application value for creative design and palette-driven image generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GCC: Generative Color Constancy via Diffusing a Color Checker](gcc_generative_color_constancy_via_diffusing_a_color_checker.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[ECCV 2024\] ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement](../../ECCV2024/image_generation/colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)

</div>

<!-- RELATED:END -->
