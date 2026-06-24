---
title: >-
  [Paper Note] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation
description: >-
  [CVPR 2025][Video Generation][Image-to-Video Generation] This paper proposes MotionStone, which decouples video motion into object motion and camera motion dimensions by training an independent motion intensity estimator. This decoupled motion is then injected into a Diffusion Transformer to achieve fine-grained, motion-intensity-controllable I2V generation.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Image-to-Video Generation"
  - "Motion Intensity Estimation"
  - "Object/Camera Motion Decoupling"
  - "Contrastive Learning"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 10693bca77c55c6f
---

# MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.05848](https://arxiv.org/abs/2412.05848)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Image-to-Video Generation, Motion Intensity Estimation, Object/Camera Motion Decoupling, Contrastive Learning, Diffusion Models

## TL;DR

This paper proposes MotionStone, which decouples video motion into object motion and camera motion dimensions by training an independent motion intensity estimator. This decoupled motion is then injected into a Diffusion Transformer to achieve fine-grained, motion-intensity-controllable I2V generation.

## Background & Motivation

In I2V generation, motion intensity control is a key but under-addressed issue. Existing methods have the following limitations:

1. **Unreliable traditional motion metrics**: Traditional metrics such as SSIM and optical flow fail to generalize to arbitrary videos and are inconsistent with human-perceived motion intensity.
2. **Lack of motion decoupling**: Motion in real-world videos is a superposition of object motion and camera motion. Existing methods model the motion intensity of the entire scene as a single value, failing to distinguish between the two motion types.
3. **Annotation difficulty**: It is impractical for human annotators to label absolute motion intensity scores for videos, as humans struggle to give consistent scores to abstract motion intensities.

*Key Insight*: Although humans find it difficult to give an absolute motion score to a single video, it is relatively easy to **compare which of two videos has stronger motion**. Based on this, the authors design a pairwise relative annotation + contrastive training framework to train the motion estimator.

## Method

### Overall Architecture

MotionStone consists of two main modules: (1) an independent motion intensity estimator that predicts the object motion score and camera motion score of a video (ranging from 1 to 10); (2) an I2V diffusion model based on CogVideoX, which generates videos conditioned on the decoupled motion embeddings.

### Key Designs

1. **Motion Estimator**: Uses TAdaConv as the backbone for video motion representation to extract motion features $M = \text{TAdaConv}(\mathbf{x}; \phi)$. After global average pooling, these are fed into two MLP heads: $s^{object} = \text{MLP}_{object}(\text{GAP}(M); \theta)$ and $s^{camera} = \text{MLP}_{camera}(\text{GAP}(M); \theta)$, which predict object and camera motion scores respectively. **Design Motivation**: Using a lightweight temporally-adaptive convolution as the backbone, the dual-head structure naturally achieves motion decoupling.

2. **Relative Annotation & Contrastive Training**: A dataset of 5,000 video pairs is constructed, where annotators only need to determine which video has stronger object/camera motion. Training utilizes a pairwise ranking loss: $L_o = \max(0, s_2^{object} - s_1^{object})$ (assuming video 1 has stronger motion), $L_c = \max(0, s_2^{camera} - s_1^{camera})$. To prevent the predicted scores from being highly concentrated, regression training is additionally applied using pseudo-labels generated from tracking trajectories: $\mathcal{L}_r = \|s^{object} - y^{object}\|_2^2 + \|s^{camera} - y^{camera}\|_2^2$. The total loss is $\mathcal{L}_{total} = \mathcal{L}_o + \mathcal{L}_c + \lambda \mathcal{L}_r$.

3. **Decoupled Motion Embedding**: Object and camera motion scores are mapped to high-dimensional vectors through independent MLPs, concatenated, and added to the timestep embedding $t$. They then modulate the visual and textual features in DiT via adaptive LayerNorm. **Design Motivation**: Since object motion and camera motion have different meanings in spatial dimensions, a mixed injection would blur their respective contributions. Decoupling preserves clear semantics.

### Loss & Training

- **Motion Estimator Training**: Ranking Loss + Regression Loss with pseudo-labels (balanced by $\lambda$).
- **Diffusion Model Training**: SFT fine-tuning based on the CogVideoX framework, utilizing 100K high-quality videos, 8 A100 GPUs, and a batch size of 16.
- Each video is sampled to 49 frames with a resolution of $480 \times 720$ and center-cropped.
- The motion estimator is frozen after pre-training, allowing the user to customize object/camera motion intensity scores during inference.

## Key Experimental Results

### Main Results (WebVID validation set, VBench metrics)

| Method | Background Consistency ↑ | Aesthetic Quality ↑ | Imaging Quality ↑ |
|------|-------------------------|--------------------|--------------------|
| I2VGen-XL | 90.93% | 40.14% | 58.35% |
| SVD | 93.17% | 42.38% | 59.61% |
| AnimateAnything | 93.89% | 46.04% | 61.69% |
| CogVideoX-5B | 94.91% | 45.88% | 61.99% |
| **MotionStone** | **95.76%** | **46.78%** | **62.29%** |

### Ablation Study

| Configuration | BG Consistency ↑ | Aesthetic ↑ | Imaging ↑ | Explanation |
|------|------------------|-------------|-----------|------|
| w/o Motion Estimator (Fixed to 5) | 95.13% | 45.61% | 60.15% | Motion diversity in training data causes confusion |
| w/ Feature Difference Estimation (S) | 94.97% | 46.13% | 60.73% | Inconsistent with human perception |
| w/ SSIM Estimation | 92.99% | 45.72% | 54.75% | SSIM cannot decouple, performs worst |
| w/o Decoupled Injection | 94.03% | 46.27% | 58.73% | Mixed injection blurs motion contributions |
| **MotionStone (Full)** | **95.76%** | **46.78%** | **62.29%** | Optimal |

### Motion Estimation Accuracy

| Method | Motion Estimation Accuracy |
|------|---------------|
| SSIM | 44.56% |
| **Ours (Motion Estimator)** | **72.80%** |

### Key Findings

- The motion estimator achieves a **28%** higher accuracy than SSIM in determining the relative motion relations of video pairs.
- Decoupled injection improves Imaging Quality by **3.56%** compared to mixed injection (58.73% → 62.29%).
- Object motion and camera motion can be adjusted independently. Users can set scores from 1 to 10 to achieve continuous control from static to intense motion.
- When keeping the camera motion fixed at 5 and varying the object motion intensity, the speed/amplitude of object motion in the generated video increases monotonically.
- When keeping the object motion fixed at 5 and varying the camera motion intensity, the zoom/pan amplitude increases monotonically.

## Highlights & Insights

- **Clever "comparison-based annotation" concept**: Bypasses the difficulty of absolute motion annotation, requiring only 5,000 video pairs to train an effective motion estimator.
- **Necessity of motion decoupling**: Experiments fully demonstrate that object and camera motion operate in different spatial dimensions; joint modeling significantly degrades performance.
- **Potential as a general-purpose plugin**: The trained motion estimator can serve as a data preprocessing tool or an enhancement module for other video generation models.
- The lightweight design of the TAdaConv backbone + dual MLP heads introduces almost no inference overhead.

## Limitations & Future Work

- The motion estimator only supports global-scene object/camera motion scoring and cannot achieve **per-object** motion control.
- The size of the annotated data (5,000 pairs) is relatively limited; increasing the volume and diversity may yield further improvements.
- Motion intensity scores are discrete integers ranging from 1 to 10, lacking finer-grained continuous control.
- Performance under complex scenes (multiple objects with different motion directions/speeds) has not been explored.
- Only validated on CogVideoX; the generalizability to other base models has not been tested.

## Related Work & Insights

- LivePhoto / Cinemo: Pioneers in text + SSIM coarse-grained motion control, but SSIM is unreliable.
- AnimateAnything: Supports coarse-grained motion intensity but the generated videos are often close to static.
- CogVideoX: The base model for this work, offering powerful spatiotemporal modeling capabilities via the DiT architecture.
- The contrastive learning + ranking loss paradigm is transferable to other visual tasks where absolute annotation is difficult (e.g., video aesthetics scoring).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel approach of decoupled motion estimation + relative annotation
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablation studies, validating the effectiveness of each component
- **Writing Quality**: ⭐⭐⭐⭐ Smooth overall flow with clear motivation
- **Value**: ⭐⭐⭐⭐ The motion estimator can serve as a plug-and-play module, offering practical value to the I2V community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)

</div>

<!-- RELATED:END -->
