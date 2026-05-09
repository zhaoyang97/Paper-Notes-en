---
title: >-
  [Paper Note] Precise Object and Effect Removal with Adaptive Target-Aware Attention
description: >-
  [CVPR 2026][Image Generation][Object Removal] This paper proposes ObjectClear, a framework that decouples foreground removal from background reconstruction via Adaptive Target-Aware Attention (ATA), combined with Attention-Guided Fusion (AGF) and Spatially Varying Denoising Strength (SVDS) strategies, enabling precise removal of target objects along with their associated visual effects such as shadows and reflections. The work also introduces OBER, the first large-scale dataset for Object-Effect Removal.
tags:
  - CVPR 2026
  - Image Generation
  - Object Removal
  - Shadow/Reflection Elimination
  - Diffusion Models
  - Attention-Guided Fusion
  - Dataset Construction
date: 2026-05-08
content_hash: afefa862cf2e9ebd
---

# Precise Object and Effect Removal with Adaptive Target-Aware Attention

**Conference**: CVPR 2026
**arXiv**: [2505.22636](https://arxiv.org/abs/2505.22636)
**Code**: [https://zjx0101.github.io/projects/ObjectClear](https://zjx0101.github.io/projects/ObjectClear)
**Area**: Image Generation
**Keywords**: Object Removal, Shadow/Reflection Elimination, Diffusion Models, Attention-Guided Fusion, Dataset Construction

## TL;DR
This paper proposes ObjectClear, a framework that decouples foreground removal from background reconstruction via Adaptive Target-Aware Attention (ATA), combined with Attention-Guided Fusion (AGF) and Spatially Varying Denoising Strength (SVDS) strategies, enabling precise removal of target objects along with their associated visual effects such as shadows and reflections. The work also introduces OBER, the first large-scale dataset for Object-Effect Removal.

## Background & Motivation

**State of the Field**: Diffusion model-based image inpainting and object removal has become the dominant paradigm, leveraging target segmentation masks with diffusion generators to erase unwanted objects. Representative methods include SDXL-Inpainting, PowerPaint, BrushNet, and RORem.

**Limitations of Prior Work**: Existing methods suffer from three core issues: (a) **Effect residuals**: only the object itself is removed, while associated visual effects such as shadows and reflections persist; (b) **Hallucination**: unwanted objects or textures are generated within the removal region; (c) **Background tampering**: colors and textures in non-target regions are inadvertently altered.

**Root Cause**: There is no explicit modeling of the relationship between target objects and their associated visual effects, nor effective constraints to guide generative model attention toward the removal region. Existing datasets are either purely synthetic (lacking real-world effect annotations), too small in scale, or not publicly available.

**Starting Point**: The paper decouples foreground removal from background reconstruction by learning target-aware attention maps to adaptively localize objects and their effect regions, while maintaining high-fidelity background preservation. A large-scale hybrid dataset with both object and effect mask annotations is also constructed.

**Core Idea**: ATA learns attention maps over object-effect regions; these maps are then used during inference for attention-guided fusion, simultaneously achieving precise removal and background preservation.

## Method

### Overall Architecture
ObjectClear is built upon SDXL-Inpainting. The input is $\langle z_t, I_{in}, M_o, c \rangle$ (noisy latent representation, original image, object mask, text prompt). Notably, the full original image $I_{in}$ is used as input rather than the masked image $I_m$ used in conventional methods, enabling the model to better capture the visual features of object effects and background information behind transparent objects.

### Key Designs

1. **Adaptive Target-Aware Attention (ATA)**

    - **Function**: Guides the model to simultaneously attend to the target object region and its associated effect regions (shadows, reflections).
    - **Mechanism**: The text embedding of the prompt "remove the instance of" is concatenated with the object visual embedding (encoded from $I_{in} \cdot M_o$ via a CLIP visual encoder and projected through an MLP) to serve as the cross-attention guidance signal. The cross-attention map $\mathbf{A}$ corresponding to visual embedding tokens is extracted and supervised using the object-effect mask $M_{fg}$.
    - **Key Loss**: $\mathcal{L}_{mask} = \text{mean}(\mathbf{A}[1-M_{fg}]) - \text{mean}(\mathbf{A}[M_{fg}])$, which minimizes attention over background regions and maximizes attention over foreground regions.
    - **Design Motivation**: Explicitly models the object-effect relationship rather than relying on implicit learning.

2. **Attention-Guided Fusion (AGF)**

    - **Function**: Leverages ATA-predicted attention maps during inference to perform adaptive input-output blending.
    - **Mechanism**: The cross-attention map from the first layer during inference (corresponding to the object embedding) is extracted, upsampled to the original image resolution, and smoothed via Gaussian blur to yield a soft-edge object-effect mask, which is used for alpha blending between the generated result and the original input.
    - **Design Motivation**: Reduces background changes introduced by diffusion denoising and VAE reconstruction, preserving fine-grained color and texture consistency. Unlike BrushNet, which relies on user-provided masks, AGF uses a model-generated object-effect mask automatically.

3. **Spatially Varying Denoising Strength (SVDS)**

    - **Function**: Applies different denoising strengths to the object mask region and the background region.
    - **Mechanism**: The object region uses $DS=1.0$ (fully generated from noise for complete removal), while the background region uses $DS=0.99$ (retaining original information to prevent color drift), achieved by re-injecting background information during inference.
    - **Design Motivation**: A uniform denoising strength presents a contradiction — $DS=1.0$ achieves complete removal but causes global color shift, while $DS=0.99$ preserves color consistency but results in incomplete removal.

### OBER Dataset Construction

The hybrid dataset consists of two parts:
- **Camera-captured data (2,878 pairs)**: Image pairs captured with a fixed camera with and without objects present; DINO+SAM is used to obtain object masks, and object-effect masks are computed via pixel-level difference between input and ground truth.
- **Synthetic data (10,000 images)**: Background images collected from the Internet (filtered for flat regions using Mask2Former and validated for depth consistency via Depth Anything V2), with foreground objects and effect layers composited via alpha blending, supporting multi-object occlusion scenarios. The alpha value is computed as: $\alpha(p) = (I_{gt} - I_{in})/(I_{gt} + \varepsilon)$ for effect regions.

### Loss & Training
- Built on SDXL-Inpainting; trained at 512×512 resolution, batch size 32, on 8× A100 GPUs for 100k steps, with learning rate 1e-5.
- Total loss = standard diffusion loss + $\mathcal{L}_{mask}$.
- Inference uses guidance scale = 1.0, 20 denoising steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | ObjectClear | OmniPaint (Prev. SOTA) | Gain |
|--------|------|-------------|---------------------|------|
| RORD-Val | PSNR↑ | **26.24** | 22.75 | +3.49 |
| RORD-Val | PSNR-BG↑ | **29.78** | 24.66 | +5.12 |
| RORD-Val | LPIPS↓ | **0.1157** | 0.1178 | -0.002 |
| OBER-Test | PSNR↑ | **33.04** | 29.06 | +3.98 |
| OBER-Test | PSNR-BG↑ | **35.62** | 30.04 | +5.58 |
| OBER-Test | LPIPS↓ | **0.0342** | 0.0521 | -0.018 |

Notably, ObjectClear using only object masks outperforms all methods that use object-effect masks. The substantial lead on PSNR-BG (+5 dB) highlights a clear advantage in background preservation.

### Ablation Study

| Configuration | PSNR↑ | PSNR-BG↑ | LPIPS↓ | Notes |
|------|-------|----------|--------|------|
| CC Data only | 27.29 | 27.96 | 0.0910 | Baseline |
| + ATA | 27.56 | 28.37 | 0.0845 | Effect of attention |
| + Sim. Data | 28.04 | 28.80 | 0.0805 | Contribution of synthetic data |
| + AGF | 32.77 | 35.50 | 0.0348 | **Largest contributor** |
| + SVDS | **33.04** | **35.62** | **0.0342** | Full model |

### Key Findings
- **AGF is the largest contributor**: Adding AGF raises PSNR from 28.04 to 32.77 (+4.73 dB) by directly exploiting the learned attention maps to protect the background.
- ATA and synthetic data each contribute approximately 0.5 dB; SVDS provides an additional ~0.3 dB.
- Multi-object synthetic data is critical for robustness in complex scenarios involving occlusion and object interactions.

## Highlights & Insights
- The **synergistic design of ATA and AGF** is particularly elegant: the attention maps learned during training not only improve removal precision but also serve as natural guidance signals for inference-time fusion, allowing a single module to serve dual purposes.
- The **spatially heterogeneous denoising** strategy of SVDS is broadly generalizable — applying different denoising strengths to different regions can benefit any diffusion-based editing task requiring region-differentiated processing.
- The **dataset construction pipeline** (pixel-difference extraction of effect masks + alpha blending for synthetic compositing) constitutes a reusable toolkit.

## Limitations & Future Work
- Training resolution is limited to 512×512, requiring additional adaptation for high-resolution real-world applications.
- Object masks depend on external segmentation models (DINO+SAM), and mask quality directly affects results.
- For highly complex multi-light-source scenes (e.g., intersecting shadows from multiple objects), automatic extraction of effect masks may be insufficiently accurate.
- The method handles only static images; video object removal would require additional temporal consistency design.

## Related Work & Insights
- **vs. OmniPaint**: Both methods require only object masks; OmniPaint implicitly learns effect removal, whereas ObjectClear explicitly models effect regions via ATA, leading to a 5+ dB advantage in PSNR-BG.
- **vs. RORem**: RORem depends on manually annotated quality assurance, while ObjectClear automatically acquires high-quality effect masks via camera-captured pairs and pixel-level difference.
- **vs. Attentive Eraser**: Both focus on attention mechanisms, but Attentive Eraser optimizes at test time, whereas ObjectClear learns through mask loss supervision during training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The joint design of ATA/AGF/SVDS is elegant, though the core idea (explicit effect modeling + attention-guided fusion) is relatively intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three test sets (including a self-collected in-the-wild set), comprehensive ablations, and fair comparisons under both mask settings.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed descriptions of the dataset construction pipeline.
- **Value**: ⭐⭐⭐⭐ The OBER dataset and precise effect removal capability offer significant practical value, though a high-resolution version is needed.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)

<!-- RELATED:END -->
