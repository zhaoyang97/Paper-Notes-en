---
title: >-
  [Paper Note] Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization
description: >-
  [CVPR 2025][Image Generation][Human Relighting] A unified framework for human relighting and background harmonization based on pre-trained diffusion models is proposed, which achieves illumination-consistent relighting for both static and video scenes using a coarse-to-fine strategy (spherical harmonics ControlNet providing coarse lighting + diffusion model learning fine residuals) and an unsupervised motion ControlNet.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Human Relighting"
  - "Background Harmonization"
  - "Coarse-to-Fine Diffusion"
  - "Temporal Consistency"
  - "Video Relighting"
date: 2026-05-08
content_hash: 1e8a46e72b0dfb9b
---

# Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization

**Conference**: CVPR 2025  
**arXiv**: [2504.03011](https://arxiv.org/abs/2504.03011)  
**Code**: [https://junyingw.github.io/paper/relighting](https://junyingw.github.io/paper/relighting) (Project Page)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Human Relighting, Background Harmonization, Coarse-to-Fine Diffusion, Temporal Consistency, Video Relighting

## TL;DR
A unified framework for human relighting and background harmonization based on pre-trained diffusion models is proposed, which achieves illumination-consistent relighting for both static and video scenes using a coarse-to-fine strategy (spherical harmonics ControlNet providing coarse lighting + diffusion model learning fine residuals) and an unsupervised motion ControlNet.

## Background & Motivation

**Background**: Monocular human relighting aims to alter the lighting conditions of human body images. Existing methods either use physical models (e.g., spherical harmonics), which are accurate but lack details like shadows, or use learning-based models that generate shadows but exhibit poor generalization. Background harmonization (aligning human lighting with the background) is typically treated as an independent task.

**Limitations of Prior Work**: (1) End-to-end diffusion methods directly learn the complete lighting transformation from noise, but this task is overly challenging, leading to unstable quality. (2) Static image methods fail to handle videos since frame-by-frame relighting causes temporal flickering. (3) There is no unified framework that simultaneously addresses both human relighting and background harmonization.

**Key Challenge**: Accurate physical lighting models (such as spherical harmonics) can handle diffuse reflection but fail to generate details like self-occlusion shadows and specular highlights, whereas diffusion models can generate these details but struggle to guarantee physical correctness and temporal consistency.

**Goal**: To unify human relighting and background harmonization, while supporting temporally consistent relighting for both static images and videos.

**Key Insight**: Coarse-to-fine decomposition: spherical harmonics provide the coarse illumination transition (as a physically correct foundation), and the diffusion model only needs to learn the fine residuals (such as shadows, environmental reflections, and specular highlights), thereby reducing the learning difficulty.

**Core Idea**: Utilizing a spherical harmonics ControlNet to provide coarse illumination conditions, a diffusion model to learn fine residuals for human relighting, and an unsupervised motion ControlNet to learn temporal consistency of lighting from real-world videos.

## Method

### Overall Architecture
Fine-tuned based on pre-trained Stable Diffusion. **Light ControlNet** encodes the coarse lighting shading (rendered by spherical harmonics) and the target background image to provide coarse lighting conditions. The diffusion model only learns the fine residuals that cannot be covered by the coarse lighting. **Motion ControlNet** learns lighting temporal loop-consistency from real-world videos in an unsupervised manner. During inference, spatiotemporal feature blending and guided refinement are used to preserve high-frequency details.

### Key Designs

1. **Coarse-to-Fine Illumination Decomposition**

    - **Function**: To reduce the learning difficulty of the diffusion model, allowing it to focus on fine-grained lighting effects.
    - **Mechanism**: First, spherical harmonics are used to render a coarse shading map from the normal map and target lighting parameters, which serves as the conditional input for the Light ControlNet. After receiving the coarse illumination, the diffusion model only needs to predict the residuals (such as self-occlusion shadows and environmental reflections). Ablation studies show that the coarse-to-fine scheme (PSNR 28.42) substantially outperforms the end-to-end scheme (26.42) and the non-diffusion alternative (17.10).
    - **Design Motivation**: Spherical harmonics process diffuse reflections quickly and accurately but lack details, whereas diffusion models are powerful at generating details but unstable for full-scale learning. The decomposition allows each to play to its strengths.

2. **Unsupervised Temporal Consistency Learning**

    - **Function**: To learn the temporal smoothness of light variations from unlabeled real-world videos.
    - **Mechanism**: The Motion ControlNet learns lighting loop-consistency from real-world video frame sequences—the same frame relit under different lighting conditions and then cycled back should remain consistent. No ground-truth relighting annotations are required. During inference, adjacent frame features are fused using a fixed spatiotemporal blending ratio (spatial 0.85:0.15, temporal 0.5:0.5).
    - **Design Motivation**: Since there are no ground-truth relighting video datasets for dynamic human bodies, the unsupervised approach bypasses data limitations.

3. **Guided Refinement**

    - **Function**: To prevent high-frequency details of the output from being blurred by the diffusion process.
    - **Mechanism**: In the late stages of denoising, high-frequency information from the original image is used to guide the output details.
    - **Design Motivation**: The diffusion process tends to smooth out high-frequency textures.

### Loss & Training
Standard diffusion denoising loss + temporal loop-consistency loss. Trained on synthetic data (OpenIllumination, LightStage, etc.) + real-world videos, with approximately 100K training samples.

## Key Experimental Results

### Main Results

| Scenario | Method | PSNR↑ | SSIM↑ |
|------|------|-------|-------|
| Portrait | DPR | 21.29 | 0.88 |
| Portrait | **Ours** | **23.04** | **0.90** |
| Full-body | GFR | 28.57 | 0.95 |
| Full-body | **Ours** | **30.81** | **0.97** |
| Video (dynamic lighting + moving human) | **Ours** | **26.61 PSNR / 38.32 tPSNR** | **0.94 / 0.98** |

### Ablation Study

| Configuration | PSNR↑ |
|------|-------|
| Non-diffusion (Spherical Harmonics only) | 17.10 |
| End-to-end Diffusion | 26.42 |
| **Coarse-to-Fine Diffusion** | **28.42** |
| + Background + Refinement | **28.78** |

### Key Findings
- The coarse-to-fine decomposition is the key contribution: it improves PSNR by 2 compared to end-to-end diffusion, demonstrating the effectiveness of decomposition learning.
- Unsupervised temporal consistency achieves optimal performance across all three video scenarios without requiring ground-truth relighting data.
- In the AMT user study, 32.2% of users preferred the relighting results of the proposed method, which is close to the ground truth at 34.8%.

## Highlights & Insights
- **Coarse-to-fine physical-learning hybrid** is a simple yet efficient paradigm—using physical models to handle modelable parts, while learning-based models only need to supplement the residuals.
- **Unsupervised temporal consistency learning from real-world videos** successfully bypasses the bottleneck of annotated data.

## Limitations & Future Work
- The training data is dominated by synthetic + LightStage datasets, which may lead to insufficient generalization in extreme real-world scenarios.
- The spatiotemporal blending ratio is a fixed hyperparameter, which might not be applicable to all scenarios.
- There is no publicly available ground-truth dataset for dynamic human relighting to conduct a comprehensive evaluation.

## Related Work & Insights
- **vs DPR**: DPR applies direct modifications using spherical harmonics, which lacks shadow details. This work adds diffusion residuals on top of spherical harmonics.
- **vs GFR**: GFR utilizes a conditional GAN, whereas this work uses a diffusion model to achieve superior quality and generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of coarse-to-fine decomposition and unsupervised video consistency is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quite comprehensive, including static + video + AMT user studies + ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework.
- Value: ⭐⭐⭐⭐ The unified framework has direct value for film/television and AR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)
- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](../../NeurIPS2025/image_generation/unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
