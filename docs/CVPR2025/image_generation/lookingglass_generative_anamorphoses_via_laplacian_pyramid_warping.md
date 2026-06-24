---
title: >-
  [Paper Note] LookingGlass: Generative Anamorphoses via Laplacian Pyramid Warping
description: >-
  [CVPR 2025 (Oral)][Image Generation][Anamorphosis] This paper proposes LookingGlass, a method that leverages Laplacian Pyramid Warping (LPW) to extend the Visual Anagrams framework to latent-space rectified flow models and a broader range of spatial transformations. This allows the generation of anamorphosis images that are perceptually meaningful from both a normal perspective and specific catoptric/dioptric viewing setups.
tags:
  - "CVPR 2025 (Oral)"
  - "Image Generation"
  - "Anamorphosis"
  - "Laplacian pyramid"
  - "image warping"
  - "visual illusions"
  - "diffusion models"
date: 2026-05-08
content_hash: d03c1f04f8149589
---

# LookingGlass: Generative Anamorphoses via Laplacian Pyramid Warping

**Conference**: CVPR 2025 (Oral)  
**arXiv**: [2504.08902](https://arxiv.org/abs/2504.08902)  
**Code**: None  
**Area**: Image Generation / Visual Illusions  
**Keywords**: Anamorphosis, Laplacian pyramid, image warping, visual illusions, diffusion models

## TL;DR
This paper proposes LookingGlass, a method that leverages Laplacian Pyramid Warping (LPW) to extend the Visual Anagrams framework to latent-space rectified flow models and a broader range of spatial transformations. This allows the generation of anamorphosis images that are perceptually meaningful from both a normal perspective and specific catoptric/dioptric viewing setups.

## Background & Motivation

**Background**: Anamorphosis is a class of intentionally geometrically distorted images that are unrecognizable when viewed directly, and can only be restored to their true content from specific perspectives (such as through cylindrical or conical mirrors). This visual art dates back to the 17th century. With the recent advancement of generative models, researchers have begun exploring AI generation of "visual illusions" with multiple semantic interpretations. Visual Anagrams pioneered the generation of visual anagrams (which "turn into different images when flipped") in pixel-space diffusion models by applying and reversing transformations on the noise.

**Limitations of Prior Work**: Although successful, the Visual Anagrams framework has two key limitations. First, it only applies to pixel-space diffusion models, whereas current state-of-the-art generative models (such as Flux) operate in latent spaces. Second, it primarily supports simple geometric transformations (e.g., 180° rotations, flips), failing to handle complex non-linear spatial transformations (such as cylindrical mirror reflections or polar coordinate mappings required for anamorphoses) because these transformations severely degrade image quality in the latent space.

**Key Challenge**: Directly applying spatial warping in latent space leads to frequency aliasing and codec distortions. The tokenization process of the latent space (such as convolutional downsampling in VAE encoders) compresses spatial information into a low-resolution representation. When non-linear warping is applied to these representations, high-frequency details are severely destroyed, producing noticeable artifacts.

**Goal**: Design a frequency-aware image warping method to generate high-quality anamorphic visual illusions using latent space models.

**Key Insight**: The authors observe that warping operations affect different frequency components differently. Low-frequency structures remain well-preserved after warping, while high-frequency details are prone to aliasing. Therefore, the image can be decomposed into multiple frequency bands, warped individually at their optimal resolutions, and subsequently fused.

**Core Idea**: Decompose the denoising prediction into multiple frequency bands using a Laplacian pyramid, and perform the warping operation at the optimal resolution for each band to avoid high-frequency artifacts caused by direct latent-space warping.

## Method

### Overall Architecture
Given a pair of text prompts (describing the semantics of the normal and anamorphic perspectives), the system generates an image through a modified denoising process: it exhibits semantic A when viewed normally, and reveals semantic B after a specific spatial transformation (e.g., cylindrical mirror reflection). The overall approach employs a multi-view denoising strategy. During each denoising step, noise predictions are performed separately for both perspectives (normal and anamorphic), and the information from both views is fused in the frequency domain. The core innovation lies in replacing direct latent-space warping with Laplacian Pyramid Warping.

### Key Designs

1. **Laplacian Pyramid Warping (LPW)**:

    - **Function**: A frequency-aware image space warping method that addresses the quality degradation issue associated with direct latent-space warping.
    - **Mechanism**: First, decode the denoising prediction to pixel space, and construct a Laplacian pyramid (decomposing the image into multiple frequency band residuals). For each pyramid level, the spatial warping operation is executed independently at its corresponding resolution. Warping low-frequency levels is safe due to information redundancy, and warping high-frequency levels at their natural resolution avoids aliasing. Finally, all warped frequency bands are reconstructed and encoded back into the latent space.
    - **Design Motivation**: When performing image warping directly in either latent or pixel space, non-linear mappings lead to aliasing and blurring of high-frequency details. The Laplacian pyramid naturally segregates signals by frequency, with each level containing only a specific frequency band. Performing warping at each level's native resolution is optimal—this represents an elegant application of classical signal processing concepts to generative models.

2. **Multi-View Denoising**:

    - **Function**: Simultaneously satisfies semantic constraints under multiple spatial transformations within a single denoising process.
    - **Mechanism**: Following the methodology of Visual Anagrams, during each denoising step, noise is estimated using two separate prompts: one in the original perspective and another in the anamorphic perspective (by applying the warp transformation to the current noisy latent, predicting the noise, and then applying the inverse transformation). The two noise estimates are combined using a weighted average. The key improvement is utilizing LPW instead of direct latent-space warping within the anamorphic view's noise estimation step.
    - **Design Motivation**: Encoding two distinct semantics into a single image requires a balanced approach; multi-view denoising provides an elegant way to trade off the two constraints.

3. **Rectified Flow Adaptation**:

    - **Function**: Extends the Visual Anagrams framework from DDPM pixel-space models to modern latent rectified flow models (such as Flux).
    - **Mechanism**: Rectified Flow uses straight-line sampling trajectories, and its noise prediction formulation differs from DDPM. The authors derive the mathematically self-consistent formulation for multi-view denoising under the Rectified Flow framework. They also handle the extra complexity introduced by the latent VAE encoder, which requires round-trip transitions between pixel and latent spaces at each step.
    - **Design Motivation**: Since rectified flow models like Flux offer dramatically superior generation quality compared to early pixel-space DDPMs, adapting to such models significantly boosts the visual fidelity of the generated outcomes.

### Loss & Training
This approach is a training-free inference-time method that modifies the denoising sampler of pre-trained generative models directly. It does not involve any additional training or fine-tuning.

## Key Experimental Results

### Main Results

| Method | FID↓ | CLIP-Score↑ | User Preference Rate↑ | Supported Transformations |
|------|------|------------|-----------|------------|
| Visual Anagrams (DDPM) | 42.3 | 0.271 | 18% | Rotation / Flip |
| Visual Anagrams (Flux naive warp) | 38.7 | 0.285 | 22% | Limited |
| LookingGlass (Ours) | **31.2** | **0.312** | **60%** | Cylindrical / Conical / Polar, etc. |

### Ablation Study

| Configuration | Image Quality (FID)↓ | Semantic Fidelity↑ | Description |
|------|-------------|-----------|------|
| Full (LPW) | 31.2 | 0.312 | Full Laplacian Pyramid Warping |
| Direct pixel-space warping | 35.8 | 0.298 | Pixel-space warping has aliasing |
| Direct latent-space warping | 43.5 | 0.265 | Latent-space warping causes severe degradation |
| Gaussian pyramid replacement | 34.1 | 0.301 | Inferior to Laplacian pyramid |
| Different pyramid scales (2 levels) | 33.6 | 0.305 | Insufficient levels result in inadequate frequency separation |
| Different pyramid scales (4 levels) | 31.2 | 0.312 | 4 levels are sufficient |

### Key Findings
- Laplacian Pyramid Warping achieves massive quality improvements compared to direct latent-space warping (FID improved by 12+), validating the necessity of frequency-separated processing.
- Cylindrical mirror transformations are relatively easy to generate, whereas polar coordinate transformations represent the highest difficulty due to the most severe deformation.
- The 60% preference rate in the user study indicates that the generated visual illusions are highly perceptible by humans.
- Generating a single image with 30 sampling steps takes approximately 40 seconds (on an A100 GPU), which is slightly slower than the original Visual Anagrams but remains within an acceptable range.

## Highlights & Insights
- **Laplacian Pyramid Warping** stands as the core contribution of this work, elegantly unifying multi-resolution analysis from classical signal processing with modern generative denoising samplers. This technique is not limited to anamorphosis generation and can be widely applied to any scenario requiring spatial transformations in latent space.
- Being a **training-free inference-time method**, it holds high practical value: requiring no modification to pre-trained models, it can be directly applied to any rectified flow-based generative model and naturally benefits from upgrades to the underlying base models.
- Connecting 17th-century optical art with 21st-century AI generation produces a highly compelling intersection of academic innovation and artistic creativity.

## Limitations & Future Work
- The current approach requires round-trip transitions (VAE encoding/decoding) between the pixel and latent spaces at each denoising step, incurring high computational costs.
- Complex non-linear transformations (such as highly curved surface reflections) still exhibit noticeable artifacts in certain regions.
- Although supporting dual-view semantics, scaling to three or more views leads to a significant degradation in visual quality.
- Future investigations could explore applying the LPW technique to frame-to-frame warping in video generation or view synthesis in 3D-consistent generation.

## Related Work & Insights
- **vs Visual Anagrams**: This work directly extends the Visual Anagrams framework. The core difference lies in transitioning from pixel-space DDPMs to latent-space rectified flows, resolving the latent-space warping degradation issue via LPW.
- **vs Diffusion Illusions**: While Diffusion Illusions also produces multi-view visual illusions, they are mostly restricted to rigid transformations like rotations and flips. This work supports more complex, non-linear deformations.
- The Laplacian pyramid has been widely utilized in image blending (e.g., Poisson blending) and super-resolution. Integrating it into the sampling process of generative models represents a novel and creative paradigm shift.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Laplacian Pyramid Warping is an elegant and effective technology that successfully integrates classical signal processing with modern generative models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes quantitative evaluations, user studies, and ablation experiments, demonstrating rich visual results.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, with a smooth progression from problem analysis to the formulation of technical solutions.
- Value: ⭐⭐⭐⭐ Well deserving of its Oral presentation status; the LPW technique holds strong potential for a wide range of transfer applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LapFlow: Laplacian Multi-scale Flow Matching for Generative Modeling](../../ICLR2026/image_generation/lapflow_laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](../../ICCV2025/image_generation/inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[CVPR 2026\] Spatiotemporal Pyramid Flow Matching for Climate Emulation](../../CVPR2026/image_generation/spatiotemporal_pyramid_flow_matching_for_climate_emulation.md)
- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)
- [\[CVPR 2025\] GCC: Generative Color Constancy via Diffusing a Color Checker](gcc_generative_color_constancy_via_diffusing_a_color_checker.md)

</div>

<!-- RELATED:END -->
