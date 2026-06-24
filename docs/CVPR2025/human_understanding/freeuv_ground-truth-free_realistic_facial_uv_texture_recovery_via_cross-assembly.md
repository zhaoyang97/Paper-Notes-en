---
title: >-
  [Paper Note] FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly
description: >-
  [CVPR 2025][Human Understanding][Facial UV Texture Recovery] FreeUV proposes a facial UV texture recovery framework that does not require ground-truth UV texture data. By separately training a UV-to-2D network focused on realistic appearance and a 2D-to-UV network focused on structural consistency, it cross-assembles their UV-related modules into a pre-trained Stable Diffusion model during inference to achieve high-fidelity UV-to-UV texture generation.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Facial UV Texture Recovery"
  - "GT-Free Training"
  - "Cross-Assembly Inference"
  - "Stable Diffusion"
  - "3DMM"
date: 2026-05-08
content_hash: 54267bb2f5c24722
---

# FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly

**Conference**: CVPR 2025  
**arXiv**: [2503.17197](https://arxiv.org/abs/2503.17197)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Facial UV Texture Recovery, GT-Free Training, Cross-Assembly Inference, Stable Diffusion, 3DMM

## TL;DR
FreeUV proposes a facial UV texture recovery framework that does not require ground-truth UV texture data. By separately training a UV-to-2D network focused on realistic appearance and a 2D-to-UV network focused on structural consistency, it cross-assembles their UV-related modules into a pre-trained Stable Diffusion model during inference to achieve high-fidelity UV-to-UV texture generation.

## Background & Motivation

**Background**: Recovering high-quality 3D facial UV textures from a single 2D face image is a long-standing challenge in computer vision and graphics. 3DMM-based reconstruction approaches can obtain coarse geometry and textures, but they struggle to capture fine texture details such as wrinkles, pores, facial hair, and cosmetics.

**Limitations of Prior Work**: Existing high-quality UV texture generation methods heavily rely on ground-truth UV data. One category depends on real UV datasets captured by expensive professional equipment, leading to poor generalization. Another category uses StyleGAN to synthesize training data, but it is limited by StyleGAN's capabilities (domain gap, multi-step pipeline causing identity/expression/lighting inconsistency) and struggles to handle diverse in-the-wild faces (such as occlusion by makeup, etc.).

**Key Challenge**: Obtaining complete, realistic ground-truth UV textures is inherently expensive and virtually impossible to achieve with high fidelity—this is the fundamental bottleneck for all supervised methods. Unwrapped UV textures obtained from single-view 3DMM fitting suffer from severe defects (distortion, missing regions, inaccurate alignment) and cannot be directly used as training targets.

**Goal**: How can high-fidelity, structurally consistent, complete UV textures be recovered from a single face image in the complete absence of complete UV texture ground truth?

**Key Insight**: The authors observe that the in-the-wild data domain provides **realistic appearance** but unreliable UV structures, whereas the 3DMM data domain provides **reliable structures** but unrealistic appearances. Each domain possesses distinct strengths and weaknesses—the UV-to-2D mapping is reliable in the in-the-wild domain (rendering process), and the 2D-to-UV mapping is consistent in the 3DMM domain (data alignment).

**Core Idea**: Train complementary networks in the two domains separately (learning appearance in the in-the-wild domain and learning structure in the 3DMM domain), and cross-assemble their UV-related modules during inference to achieve UV-to-UV texture generation.

## Method

### Overall Architecture
FreeUV is based on pre-trained Stable Diffusion v1.5 and contains two independently trained networks: (1) **Appearance network $\phi_a$**: trained in the in-the-wild domain, taking defective UV texture $\mathbf{T}_w$ as input and outputting masked 2D face images $\mathbf{I}_w$ (UV-to-2D mapping). The Flaw-Tolerant Detail Extractor $\psi_a$ learns to extract real facial details from defective UVs. (2) **Structure network $\phi_s$**: trained in the 3DMM domain, taking masked 3DMM face images $\mathbf{I}_m$ as input and outputting masked 3DMM UV textures $\mathbf{T}_m$ (2D-to-UV mapping). The UV Structure Aligner $\psi_s$ learns structure-consistent UV layout mappings. During inference, $\psi_a$ and $\psi_s$ are combined into the SD model to directly complete the UV-to-UV mapping.

### Key Designs

1. **Flaw-Tolerant Facial Detail Extractor ($\psi_a$, Flaw-Tolerant Facial Detail Extractor)**:

    - Feature: Extracts and preserves fine facial features (beards, wrinkles, makeup, etc.) from defective unwrapped UV textures while suppressing the impact of distortion and corrupted regions.
    - Mechanism: Employs a CLIP image encoder to extract features from multiple layers and concatenates them along the feature dimension (following the Stable-Makeup approach), followed by channel attention to selectively emphasize relevant information. Channel attention is adept at identifying key features during the "downsampling" process of UV textures. The network simultaneously receives a masked UV position map $\mathbf{I}_{uv}$ and 2D landmarks $\mathbf{I}_{lm}$ as structural guidance.
    - Design Motivation: 3DMM fitting cannot achieve pixel-level alignment, and directly training with defective UVs would propagate artifacts to the output. Channel attention can selectively focus on reliable feature channels while ignoring corrupted regions, and 2D landmarks compensate for the alignment deviation between the UV position map and the 2D image.

2. **UV Structure Aligner ($\psi_s$, UV Structure Aligner)**:

    - Feature: Guides the generated UV texture to precisely conform to the UV layout structure of 3DMM to ensure structural consistency.
    - Mechanism: Based on the ControlNet architecture, trained on the 3DMM domain with pixel-aligned masked 3DMM images $\mathbf{I}_m$ and masked UV position maps $\mathbf{T}_{uv}$ (both generated with the same 3DMM parameters) as input to perform 2D-to-UV mapping. The feature extractor utilizes CLIP's spatial-aware self-attention, because 2D-to-UV mapping is equivalent to an "upsampling" process, which requires interpolating between features.
    - Design Motivation: The 2D-to-UV mapping is pixel-level consistent in the 3DMM domain. Training ControlNet with this natural alignment allows learning accurate structural mapping. Using self-attention instead of channel attention is because "upsampling" requires capturing spatial relationships among features for accurate interpolation.

3. **Cross-Assembly Inference Strategy（Cross-Assembly Inference Strategy）**:

    - Feature: Combines the two independently trained network modules during inference to generate complete, high-fidelity UV textures directly from defective UV inputs.
    - Mechanism: Integrates both the appearance network's $\psi_a$ (providing realistic detail features) and the structure network's $\psi_s$ (providing UV layout guidance) into the SD model. $\psi_a$ extracts appearance features from $\mathbf{T}_w$, while $\psi_s$ provides structural guidance based on the complete UV position map $\mathbf{\Upsilon}_{uv}$ to generate the complete UV texture $\mathbf{\Upsilon}_w$. Finally, mean-variance matching in the Lab color space is utilized for color correction.
    - Design Motivation: The input-output mappings of the two networks during training are symmetric and complementary (UV→2D and 2D→UV), yet neither directly performs UV→UV mapping. Cross-assembly leverages their respective expertise learned in different domains, preventing structural failure or detail loss that would occur if either network generated the texture independently.

### Loss & Training
Both networks employ the standard diffusion model denoising loss: Appearance network $\mathcal{L}_a(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}_T^w, \mathbf{c}_I^{uv}, \mathbf{c}_I^{lm})\|_2^2]$; Structure network $\mathcal{L}_s(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}_I^m, \mathbf{c}_T^{uv})\|_2^2]$. Training is conducted on a single A100 GPU for 80,000 iterations, with a batch size of 4 and a learning rate of $3 \times 10^{-5}$. Inference uses DDIM with 30 steps and a guidance scale of 1.4.

## Key Experimental Results

### Main Results

| Dataset | Metric | FreeUV | HRN | UV-IDM | FLAME-based |
|--------|------|--------|-----|--------|-------------|
| FFHQ | CLIP-I↑ | **0.8490** | 0.8327 | 0.7986 | 0.8218 |
| FFHQ | DINO-I↑ | **0.7559** | 0.7389 | 0.5836 | 0.7269 |
| FFHQ | FID↓ | **142.39** | 166.19 | 228.74 | 158.06 |
| CelebAMask-HQ | CLIP-I↑ | **0.8272** | 0.8259 | 0.7458 | 0.8016 |
| CelebAMask-HQ | DINO-I↑ | **0.7948** | 0.7382 | 0.5690 | 0.7640 |
| LPFF (Large Angle) | CLIP-I↑ | **0.7997** | 0.7368 | 0.7440 | 0.7822 |
| LPFF (Large Angle) | DINO-I↑ | **0.6835** | 0.5951 | 0.5345 | 0.6724 |

### Ablation Study

| Configuration | RMSE↓ | SSIM↑ | LPIPS↓ | PSNR↑ |
|------|-------|-------|--------|-------|
| $\phi_a^{ch} + \phi_s^{self}$ (Ours) | **0.0276** | **0.8001** | **0.0463** | **30.848** |
| $\phi_a^{self} + \phi_s^{self}$ | 0.0302 | 0.7881 | 0.0474 | 30.397 |
| $\phi_a^{self} + \phi_s^{ch}$ | 0.0367 | 0.7876 | 0.0539 | 28.693 |
| $\phi_a^{ch} + \phi_s^{ch}$ | 0.0379 | 0.7648 | 0.0639 | 28.417 |
| w/o landmarks | 0.0292 | 0.7928 | 0.0481 | 30.624 |
| w/o color adjustment | 0.0282 | 0.7992 | 0.0531 | 30.828 |

### Key Findings
- **The choice of attention type is crucial**: The combination of channel attention for the appearance network and self-attention for the structure network performs best. Reversing them ($\phi_a^{self} + \phi_s^{ch}$) drops PSNR by over 2dB.
- UV-to-2D is a "downsampling" process, making it suitable for channel attention (selecting key features), whereas 2D-to-UV is an "upsampling" process, suited for self-attention (interpolating feature relationships).
- The inclusion of 2D landmarks effectively compensates for the alignment errors of 3DMM fitting; removing them increases the RMSE from 0.0276 to 0.0292.
- FreeUV performs particularly robustly in large-angle and occluded scenarios, benefiting from the Flaw-Tolerant design.

## Highlights & Insights
- **The Cross-Assembly inference strategy** is the core innovation—two networks learn complementary mappings in different domains separately and are assembled together during inference. This "divide-and-learn, combine-and-use" philosophy can be extended to any generative task where domain discrepancies exist but domains possess complementary advantages.
- **Eliminating the need for GT UV data** substantially lowers the data acquisition barrier, requiring only 33K images from FFHQ for training, which is more scalable than methods needing scanning devices or StyleGAN synthesis.
- The analysis of the roles of attention mechanisms in "upsampling vs. downsampling" mappings is highly insightful—channel attention is responsible for selection, while self-attention is responsible for interpolation.

## Limitations & Future Work
- Position shifts or quantity variations may occur for very fine facial elements (accessories, spots, blemishes).
- Recovery in occluded regions (such as under a hat) may unnaturally extend surrounding textures.
- It depends on the quality of 3DMM fitting; if the fitting fails (e.g., face segmentation errors), the output quality drops significantly.
- Inference speed is limited by SD's 30-step sampling (4.75 seconds per image), where distillation could be considered for acceleration.
- Only the skin region is handled, without extension to the full face including eyes, mouth, etc.

## Related Work & Insights
- **vs FFHQ-UV**: FFHQ-UV requires resource-intensive iterative optimization to create UV ground truth, whereas FreeUV completely eliminates this dependency.
- **vs UV-IDM**: UV-IDM first generates multi-view images with StyleGAN and then synthesizes UV pairs, which is limited by StyleGAN's capabilities and prone to inconsistencies due to the multi-step pipeline; FreeUV recovers end-to-end directly from a single image.
- **vs DSD-GAN**: Also ground-truth-free, but DSD-GAN uses GANs, introducing structural alignment artifacts, whereas FreeUV is more robust based on diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ The cross-domain complementary idea of Cross-Assembly is clever, and interpreting the attention mechanisms from the perspective of "downsampling/upsampling" provides novel insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies and comparison with multiple methods across three datasets.
- Writing Quality: ⭐⭐⭐⭐ Table 1 clearly summarizes the cross-domain selection logic, and the motivations are thoroughly explained.
- Value: ⭐⭐⭐⭐ Significantly lowers the data barrier for facial UV texture generation, demonstrating highly practical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing](../../ICCV2025/human_understanding/contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)
- [\[CVPR 2025\] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[CVPR 2025\] NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction](nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)

</div>

<!-- RELATED:END -->
