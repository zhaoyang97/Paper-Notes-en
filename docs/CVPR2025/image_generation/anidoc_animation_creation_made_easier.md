---
title: >-
  [Paper Note] "AniDoc: Animation Creation Made Easier"
description: >-
  [CVPR2025][Image Generation][Paper Note] Academic paper note for "AniDoc: Animation Creation Made Easier".
tags:
  - CVPR2025
  - Image Generation
date: 2024-12-18
content_hash: c0adfa8de2282d1b
---
# AniDoc: Animation Creation Made Easier

## Background & Motivation

Animation creation is a highly labor-intensive industry. In traditional animation workflows, **colorization** is one of the most time-consuming stages. A standard animated film consists of tens of thousands of frames, where the coloring of each frame must be done manually by professional artists. This is not only extremely costly but also highly dependent on human resources.

In recent years, deep learning-based video colorization methods have made progress, but existing approaches face the following key challenges:

**Temporal Consistency**: Frame-by-frame colorization is prone to color flickering, and color assignment between adjacent frames can be inconsistent.

**Sparse Reference**: In actual production, animators usually only provide color references for keyframes, requiring the intermediate frames to be automatically inferred.

**Sketch Quality Variance**: Hand-drawn sketches and digital line art differ significantly in clarity, line thickness, closure, and other aspects.

**Background Interference**: Complex backgrounds in training data interfere with the model learning to colorize the foreground characters.

This paper proposes AniDoc, a video sketch colorization system based on diffusion models, which addresses the aforementioned challenges through explicit correspondence guidance and data augmentation strategies.

## Method

### Overall Architecture

AniDoc is built upon a video diffusion model. The core workflow is: given a set of color reference frames and grayscale sketch videos, generate a complete colorized animation video. The system consists of three key modules:

1. **Correspondence Guidance Module** (Correspondence Guidance)
2. **Data Augmentation Strategy** (Data Augmentation)
3. **Sparse Sketch Training** (Sparse Sketch Training)

### Correspondence Guidance (Correspondence Guidance)

To resolve the temporal consistency issue, AniDoc builds explicit inter-frame correspondences:

| Stage | Technology | Effect |
|------|------|------|
| Feature Extraction | SIFT + LightGlue | Establish sparse feature matching between reference and target frames |
| Dense Tracking | Co-Tracker | Expand sparse matches into dense optical flow fields |
| Color Propagation | Warping + Attention | Propagate color from reference frames to target frames based on correspondence |

Specific workflow:
1. Extract SIFT feature points from reference frames and each target frame.
2. Use LightGlue for feature matching to obtain reliable sparse corresponding point pairs.
3. Use sparse corresponding points as initialization and input them into Co-Tracker to obtain dense pixel-level tracking results.
4. The correspondence information is injected into the cross-attention layer of the diffusion model in the form of feature maps.

### Data Augmentation Strategy

#### Binarization Augmentation

During training, random binarization is applied to the input sketch to simulate sketches of various styles and qualities:

$$I_{bin} = egin{cases} 1, & I_{gray} > 	au + \epsilon \ 0, & 	ext{otherwise} \end{cases}$$

Where $	au$ is an adaptive threshold, and $\epsilon \sim \mathcal{U}(-\delta, \delta)$ is a random perturbation.

#### Background Augmentation

To reduce background interference on foreground colorization, the background is randomly replaced during training:
- 50% probability of using a pure white background
- 30% probability of using a random solid color background
- 20% probability of retaining the original background

This forces the model to focus on the structure and color of the foreground characters instead of relying on background cues.

### Sparse Sketch Training

In practical applications, animators typically only paint color versions of the first and last keyframes. AniDoc proposes a sparse training strategy:

- During training, only the **first and last frames** of the video sequence are provided as color references.
- Intermediate frames are input in sketch form.
- The model must automatically infer the colorization scheme of the intermediate frames.

This training method enables the model to learn to perform reasonable color interpolation utilizing limited reference information.

### Sakuga-42M Dataset

This paper collects and organizes the Sakuga-42M dataset, which contains:
- Source: Public animation databases and video platforms
- Scale: Approximately 42 million frames of animation data
- Processing: Automatic sketch extraction, keyframe labeling, and filtering of low-quality samples

## Experimental Results

### Quantitative Comparison

| Method | FID↓ | FVD↓ | LPIPS↓ | Temporal Consistency↑ |
|------|------|------|--------|-----------|
| Reference-based (baseline) | 78.42 | 312.5 | 0.183 | 0.891 |
| w/o correspondence matching | 75.91 | 298.7 | 0.171 | 0.907 |
| AniDoc (full) | **54.33** | **215.8** | **0.124** | **0.952** |
| AniDoc (sparse, 2-ref) | 58.17 | 234.2 | 0.138 | 0.941 |

### Ablation Study

| Component | FID↓ | Description |
|------|------|------|
| Full Model | 54.33 | Full AniDoc |
| w/o Correspondence | 75.91 | Without correspondence guidance, chaotic color assignment |
| w/o Binarization Aug | 61.27 | Degraded generalization to hand-drawn sketches |
| w/o Background Aug | 59.84 | Degraded colorization quality in background regions |
| w/o Sparse Training | 63.15 | Only supports dense references, reducing practicality |

### Training Details

- Hardware: 16× NVIDIA A100 GPUs
- Training Time: 5 days
- Video Resolution: 512×512, 16 frames
- Optimizer: AdamW, lr=1e-5
- Batch Size: 2 video clips per GPU

## Summary & Outlook

By combining explicit correspondence guidance, targeted data augmentation, and sparse reference training, AniDoc significantly improves the quality and practicality of animation sketch colorization. The reduction of FID from 75.91 to 54.33 demonstrates the key role of correspondence guidance. The system can be directly integrated into existing animation production pipelines, substantially reducing manual labor costs in the colorization stage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[ICML 2025\] FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)

</div>

<!-- RELATED:END -->
