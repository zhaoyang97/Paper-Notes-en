---
title: >-
  [Paper Note] Unified Dense Prediction of Video Diffusion
description: >-
  [CVPR 2025][Video Generation][Dense Prediction] This paper proposes UDPDiff, which, for the first time, achieves joint generation of RGB videos, entity segmentation, and depth estimation within video diffusion models. It enhances video quality and consistency through the Pixelplanes unified representation and learnable task embeddings.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Dense Prediction"
  - "Entity Segmentation"
  - "Depth Estimation"
  - "Unified Representation"
date: 2026-05-08
content_hash: f337255a1b856ef4
---

# Unified Dense Prediction of Video Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2503.09344](https://arxiv.org/abs/2503.09344)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Video Generation, Dense Prediction, Entity Segmentation, Depth Estimation, Unified Representation

## TL;DR

This paper proposes UDPDiff, which, for the first time, achieves joint generation of RGB videos, entity segmentation, and depth estimation within video diffusion models. It enhances video quality and consistency through the Pixelplanes unified representation and learnable task embeddings.

## Background & Motivation

Video generation has achieved remarkable progress, but existing models still face inter-frame consistency issues (such as changes in object appearance, unstable backgrounds, and unnatural motion). Most existing improvements focus on network architecture designs (such as 3D VAE, MM-DiT) but lack explicit semantic and geometric reasoning signals.

Works such as REPA demonstrate that aligning representations with self-supervised methods can accelerate diffusion training, but these representations remain implicit. Dense prediction signals (segmentation providing object shape and motion constraints, and depth providing spatial position awareness) can serve as explicit training signals.

**Key Challenge**: (1) There are no large-scale datasets that simultaneously contain videos, segmentation, and depth annotations; (2) How to design a unified representation and architecture to jointly generate videos and multiple dense prediction results without increasing computational costs.

Previously, the image-level UniGS utilized a position-aware colormap to represent segmentation, but this method fails to handle color ambiguity issues of moving entities in videos.

## Method

### Overall Architecture

UDPDiff is constructed based on CogVideoX 5B. The video latent code $z_t^v$ and the dense prediction latent code $z_t^c$ are concatenated along the channel dimension (totaling 32 channels) and input into the Transformer for denoising. A learnable task embedding $e_\theta^d(d)$ is added to the timestep embedding to distinguish different tasks. The input and output channels are doubled, and the dense prediction results are encoded and decoded using the same 3D VAE, introducing almost no inference time overhead. Meanwhile, a large-scale dataset, Panda-Dense (approximately 300K samples), is constructed.

### Key Designs 1: Pixelplanes Unified Representation

**Function**: Encoding entity segmentation and depth maps into RGB images, sharing the same VAE with the video.

**Mechanism**: For entity segmentation, a random RGB color $M_c = (r_n, g_n, b_n)$ is sampled for each entity to ensure that colors of different entities do not overlap. For depth maps, a spectral-style value projection $D_c = \Upsilon(D)$ is used to map single-channel depth into the RGB space. Once both tasks are unified into the RGB format, they can be directly encoded and decoded using the 3D VAE.

**Design Motivation**: The position-aware colormap in UniGS uses a fixed color grid and assigns colors based on entity centroid coordinates. Problems: (1) In dense scenes, different entities on a fixed grid might be assigned the same color; (2) Entity motion in videos causes centroid changes, leading to color ambiguity in subsequent frames. Random color assignment eliminates position dependency, completely avoiding motion ambiguity.

### Key Designs 2: Learnable Task Embeddings

**Function**: Explicitly distinguishing segmentation and depth estimation tasks within a single multi-task model.

**Mechanism**: A task embedding layer $e_\theta^d$ is defined, which takes the task ID $d$ as input, and its output is added to the timestep embedding $e_\theta^t(t)$: $t_d = e_\theta^d(d) + e_\theta^t(t)$. The training loss is the standard diffusion denoising loss $\mathcal{L}_{\text{train}} = \frac{1}{2}\|f_\theta(z_t, t_d, c_t) - \epsilon\|^2$. During inference, switching between segmentation and depth generation is achieved by inputting different task IDs.

**Design Motivation**: Distinguishing tasks solely with text prompts acts as an implicit condition, which is prone to semantic ambiguity. Learnable task embeddings provide explicit task signals, enabling the model to understand the current task type to be executed more accurately.

### Key Designs 3: Panda-Dense Dataset Construction

**Function**: Providing large-scale video + segmentation + depth annotation training data.

**Mechanism**: A subset of approximately 300K videos is sampled from Panda-70M. The segmentation annotation pipeline is: (1) Apply EntitySeg CropFormer to perform entity segmentation on the first frame; (2) Use SAM2 to propagate the segmentation results to the entire video. For depth annotation, DepthCrafter is utilized to generate consistent video depth maps. A 13B Video-LLaVA is used to regenerate detailed textual descriptions.

**Design Motivation**: Existing datasets do not simultaneously contain video, segmentation, and depth. EntitySeg ensures consistency in segmentation granularity (avoiding over-fine or over-coarse issues caused by SAM point grid initialization), while DepthCrafter guarantees inter-frame depth consistency (as frame-by-frame depth estimation would introduce jitter).

### Loss & Training

Standard diffusion denoising MSE loss: $\mathcal{L}_{\text{train}} = \frac{1}{2}\|f_\theta(z_t, t_d, c_t) - \epsilon\|^2$. During multi-task training, the model switches according to the task ID, jointly optimizing the task embedding and generation model parameters.

## Key Experimental Results

### Main Results (Multi-task Model vs CogVideoX 5B)

| Model | SC↑ | BC↑ | MS↑ | FVD↓ |
|------|-----|-----|-----|------|
| CogVideoX 5B | 94.57 | 95.80 | 97.67 | 343.92 |
| UDPDiff (seg) | 95.21 | 95.69 | 98.24 | 316.76 |
| UDPDiff (depth) | **97.07** | **96.89** | **99.23** | **302.55** |

*SC=Subject Consistency, BC=Background Consistency, MS=Motion Smoothness*

### Ablation Study

| Method | SC↑ | BC↑ | MS↑ |
|------|-----|-----|-----|
| Location-aware colormap (UniGS) | 81.26 | 79.33 | 88.79 |
| **Pixelplanes** | **94.98** | **95.92** | **98.62** |

| Task Distinction Method | SC↑ | BC↑ | MS↑ | FVD↓ |
|-------------|-----|-----|-----|------|
| Text prompt | 95.17 | 95.78 | 98.67 | 321.43 |
| **Task embedding** | **97.07** | **96.89** | **99.23** | **302.55** |

### Key Findings

1. **Dense prediction significantly enhances consistency**: Multi-task UDPDiff (depth) comprehensively outperforms CogVideoX across all metrics, with FVD reduced by 41.37 (a relative reduction of 12%).
2. **Pixelplanes far outperforms UniGS colormap**: SC increases from 81.26 to 94.98 (+13.72), demonstrating the effectiveness of the random color scheme in eliminating positional ambiguity.
3. **Task embedding outperforms text prompts**: FVD decreases from 321.43 to 302.55, indicating that explicit task conditioning is more effective.
4. **Almost zero inference overhead**: The single-task model takes 205.75s vs. the original CogVideoX taking 204.46s, an increase of less than 1%.
5. **Multi-task outperforms single-task**: The jointly trained multi-task model outperforms models trained individually on segmentation/depth, as segmentation and depth provide complementary signals.

## Highlights & Insights

- **Novel unified paradigm**: This work is the first to unify video-level generation and dense prediction into the same diffusion process, where dense predictions are output as "free" by-products.
- **Mutual benefit**: Dense prediction is not only an output but also a training signal—helping the video generation model learn better scene understanding.
- **Practical value**: Obtaining video, segmentation, and depth simultaneously in a single inference is highly valuable for downstream video editing tasks.

## Limitations & Future Work

- **Limited data scale**: Trained on only 300K samples, the depth estimation accuracy ($\delta_1=0.4176$) still lags behind specialized models like Depth Anything V2 ($\delta_1=0.5808$).
- **3D VAE limitations**: Segmentation and depth are encoded/decoded in the form of RGB colormaps; the compression loss of the VAE might affect precision.
- **Only two tasks**: Joint training on other dense prediction tasks (e.g., optical flow, normal estimation) remains unexplored.
- Future work can scale up the dataset, incorporate more dense prediction tasks, and explore utilizing dense prediction results as control conditions for video editing.

## Related Work & Insights

- **UniGS**: The pioneer of image-level colormap representation; this work extends it to the video level and resolves the motion ambiguity problem.
- **Marigold/SemFlow**: Representative works utilizing diffusion models for dense prediction, but they are limited to single-task single-image settings.
- **Insight**: The paradigm of "generation is understanding"—improving generation quality via joint training with dense prediction could be extended to areas like 3D generation.

## Rating

⭐⭐⭐⭐ — For the first time achieving joint training of generation and dense prediction in video diffusion, with a simple yet effective Pixelplanes design. The experimental conclusions that multi-task training improves generation quality are convincing. Dataset construction and experiments are comprehensive. The major limitation is that the depth estimation accuracy still lags behind specialized models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](../../ICCV2025/video_generation/ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](../../ICCV2025/video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[CVPR 2026\] FlashPortrait: 6× Faster Infinite Portrait Animation with Adaptive Latent Prediction](../../CVPR2026/video_generation/flashportrait_6x_faster_infinite_portrait_animation_with_adaptive_latent_predict.md)

</div>

<!-- RELATED:END -->
