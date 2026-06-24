---
title: >-
  [Paper Note] World-Consistent Video Diffusion with Explicit 3D Modeling
description: >-
  [CVPR 2025][Video Generation][3D-consistent generation] This paper proposes WVD (World-consistent Video Diffusion), which jointly models RGB and XYZ images (encoding global 3D coordinates) by training a diffusion model. This design achieves multi-view consistent video generation under explicit 3D constraints, and unifies various downstream tasks, such as single-image 3D reconstruction, multi-view stereo, and camera-controlled generation, through a flexible inpainting strategy…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "3D-consistent generation"
  - "diffusion models"
  - "XYZ images"
  - "multi-view synthesis"
  - "depth estimation"
date: 2026-05-08
content_hash: 19f3a104ecf22207
---

# World-Consistent Video Diffusion with Explicit 3D Modeling

**Conference**: CVPR 2025  
**arXiv**: [2412.01821](https://arxiv.org/abs/2412.01821)  
**Code**: [https://zqh0253.github.io/wvd](https://zqh0253.github.io/wvd)  
**Area**: Video Generation  
**Keywords**: 3D-consistent generation, diffusion models, XYZ images, multi-view synthesis, depth estimation

## TL;DR

This paper proposes WVD (World-consistent Video Diffusion), which jointly models RGB and XYZ images (encoding global 3D coordinates) by training a diffusion model. This design achieves multi-view consistent video generation under explicit 3D constraints, and unifies various downstream tasks, such as single-image 3D reconstruction, multi-view stereo, and camera-controlled generation, through a flexible inpainting strategy.

## Background & Motivation

**Background**: Diffusion models have achieved great success in image and video generation, and multi-view diffusion models implicitly learn 3D consistency through attention mechanisms. Camera control methods (e.g., CameraCtrl, MotionCtrl) control the perspective by injecting camera ray map conditions.

**Limitations of Prior Work**: (1) Implicit methods lack explicit 3D consistency guarantees, resulting in 3D-inconsistent artifacts even with large amounts of training data. (2) Reliance on camera ray inputs makes it difficult to scale to large-scale data, since fundamental ambiguities exist in camera representations across different datasets, requiring complex normalization. (3) Explicit 3D methods (e.g., volume rendering) are constrained by architectural limitations, making them hard to scale to complex data.

**Key Challenge**: To achieve 3D consistency in generative models, traditional implicit methods (learning across frames via attention) are unreliable, while explicit 3D methods (e.g., volume rendering) are incompatible with existing 2D Transformer architectures.

**Goal**: Design a method that provides explicit 3D supervision while remaining compatible with existing DiT architectures.

**Key Insight**: The authors propose representing 3D geometry using XYZ images, where each pixel records its global 3D coordinates. XYZ images share the same shape as RGB images and are naturally compatible with 2D Transformer architectures.

**Core Idea**: Represent the 3D scene as a "6D video" containing RGB + XYZ, and train a DiT to jointly diffuse these two modalities. This joint diffusion yields both appearance and geometry during the generation process.

## Method

### Overall Architecture

The input to WVD is a set of 6D videos, where each view contains an RGB image and an XYZ image (encoding global 3D coordinates). During training, the RGB and XYZ images are individually encoded into the latent space using a pre-trained VAE, concatenated along the channel dimension, and processed through a joint diffusion denoising process. During inference, a flexible inpainting strategy—replacing known modalities with ground truth to achieve conditional generation—is used to support various downstream tasks.

### Key Designs

1. **XYZ Image Representation**:

    - **Function**: Encodes 3D geometry into an image format compatible with RGB.
    - **Mechanism**: Translates point clouds into XYZ images $\boldsymbol{x}^{\text{XYZ}} = \mathcal{R}(\mathcal{N}(X), X, C)$ via normalization (centering and scaling to $[-1,1]$) and rasterization (projecting to the camera plane). The XYZ image shares the same shape as the RGB image, where each pixel value represents global 3D coordinates instead of color. Pixels with identical XYZ values in two different perspectives correspond to the exact same point in 3D space, directly providing explicit pixel correspondence.
    - **Design Motivation**: Unstructured point clouds ($\mathbb{R}^{N \times 3}$) are incompatible with DiT. XYZ images preserve a structured 2D format, enabling them to be encoded directly by a pre-trained VAE and eliminating the need for additional camera parameter inputs.

2. **Joint RGB-XYZ Diffusion**:

    - **Function**: Simultaneously generates appearance and 3D geometry.
    - **Mechanism**: Concatenates VAE latents of RGB and XYZ along the channel dimension to form $\boldsymbol{z}_n = [\mathcal{E}(\boldsymbol{x}_n^{\text{RGB}}); \mathcal{E}(\boldsymbol{x}_n^{\text{XYZ}})] \in \mathbb{R}^{L \times 2D}$, and performs standard diffusion training on this target. Since the XYZ images are normalized to $[-1,1]$, the pre-trained VAE can be directly applied without additional fine-tuning. For image-conditioned generation, the noise on the conditional image is removed at each training step.
    - **Design Motivation**: The channel concatenation design allows direct fine-tuning of pre-trained image/video diffusion models, significantly improving training efficiency. Joint modeling enables the explicit 3D correspondence of XYZ to impose constraints back on the multi-view consistency of RGB.

3. **Post Optimization and Flexible Inference**:

    - **Function**: Recovers precise camera parameters and depth maps from predicted XYZ images, while supporting multiple downstream tasks.
    - **Mechanism**: Performs gradient optimization with a reprojection loss $\min_{P,K,\boldsymbol{d}} \sum_{u,v} \|\tilde{\boldsymbol{x}}^{\text{XYZ}}_{u,v} - \hat{\boldsymbol{x}}^{\text{XYZ}}_{u,v}\|^2$ on the predicted XYZ images to recover camera pose, intrinsics, and depth maps. Inpainting strategies are used during inference to achieve task switching: (a) Estimate XYZ given RGB $\to$ monocular/multi-view depth estimation; (b) Generate RGB given XYZ $\to$ camera-controlled video generation; (c) Joint generation $\to$ single-image 3D reconstruction.
    - **Design Motivation**: The joint distribution $P(\text{RGB}, \text{XYZ})$ naturally supports conditional distribution estimation, allowing a single model to unify various 3D tasks without training separate models.

### Loss & Training

The standard diffusion denoising loss (predicting noise or clean data) is applied to the concatenated latents of RGB and XYZ. The training data combines RealEstate10K, ScanNet, MVImgNet, CO3D, and Habitat, covering both object-centric and scene-centric distributions. The model has 2 billion parameters and is trained using the AdamW optimizer with a learning rate of $3 \times 10^{-4}$ on 64 A100 GPUs for approximately two weeks.

## Key Experimental Results

### Main Results

| Method | FID↓ | KPM↑ | FC↑ |
|---|---|---|---|
| CameraCtrl | 12.1 | 88.6 | 94.0 |
| MotionCtrl | 12.9 | 68.6 | 94.6 |
| WVD | 15.8 | **95.8** | **95.4** |
| WVD w/o XYZ | 18.3 | 72.3 | 95.0 |

In the single-image 3D generation task, WVD's Key Points Matching (a multi-view consistency metric) significantly outperforms the baselines, reaching 95.8%.

### Ablation Study

| Configuration | FID↓ | KPM↑ | FC↑ |
|---|---|---|---|
| WVD (Full) | 15.8 | 95.8 | 95.4 |
| WVD w/o XYZ | 18.3 | 72.3 | 95.0 |

After removing the joint learning of XYZ, KPM drops sharply from 95.8% to 72.3%, and the image quality (FID) worsens from 15.8 to 18.3, fully verifying the critical role of explicit 3D supervision.

Depth estimation results:

| Method | NYU-v2 Rel↓ | BONN Rel↓ |
|---|---|---|
| DUSt3R-224 | 10.3 | 11.1 |
| DUSt3R-512* | 6.5 | 8.1 |
| WVD (256) | 9.7 | **7.0** |

On the BONN benchmark, WVD trained at 256 resolution outperforms all methods, including DUSt3R at 512 resolution.

### Key Findings

- Jointly learning XYZ is core—ablation shows that multi-view consistency significantly drops once XYZ is removed.
- WVD is highly competitive as a generative model for depth estimation, because jointly sampling consistent surrounding views makes depth prediction theoretically more 3D-grounded.
- Camera-controlled generation is achieved through an "estimate 3D $\to$ reproject $\to$ inpaint" pipeline, which eliminates the need to explicitly incorporate camera conditions during training.
- The synthesized point clouds can serve as a "spatial memory" that is progressively expanded to achieve consistent generation for long videos.

## Highlights & Insights

1. **XYZ images are an ingenious design**: Translating unstructured 3D geometry problems into structured image generation challenges cleverly repurposes powerful 2D generative infrastructures.
2. **Elimination of camera inputs**: Direct camera ray maps are no longer required as condition inputs, circumventing the complex engineering required for camera standardization across different datasets.
3. **A single model unifying multiple tasks**: Depth estimation, novel view synthesis, camera-controlled generation, and 3D reconstruction are all achieved via the inpainting strategy.
4. **A new paradigm for generative depth estimation**: Obtaining depths through the joint generation of multiple views is geometrically more consistent compared to single-image regression.

## Limitations & Future Work

- Currently trained only on static datasets, making it unable to handle dynamic scenes (4D).
- Lacks a confidence map, which hinders handling of unbounded or outdoor scenes.
- Resolution is limited to 256×256, which is still far from practical application.
- Future work could replace XYZ with other modalities (e.g., optical flow, splatter images) to extend to more tasks.

## Related Work & Insights

- Relationship with DUSt3R: DUSt3R directly regresses point clouds, whereas WVD learns distributions via generative modeling; their concepts are complementary.
- Relationship with CAT3D: CAT3D uses camera ray conditions, while WVD replaces them with XYZ images, providing a cleaner alternative.
- Insight 1: "Imaging" 3D representations serves as an effective bridge connecting 2D and 3D.
- Insight 2: Jointly modeling complementary modalities allows the model to learn superior representation.

## Rating

- **Novelty**: 8/10 — The idea of joint diffusion with XYZ images is simple, elegant, and pioneering.
- **Experimental Thoroughness**: 7/10 — Covers multiple tasks but lacks comparisons to more recent methods.
- **Writing Quality**: 8/10 — Clear methodological descriptions with intuitive framework diagrams.
- **Value**: 8/10 — Presents a viable path toward 3D foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling](../../ICLR2026/video_generation/geometry_forcing_marrying_video_diffusion_and_3d_representation_for_consistent_w.md)
- [\[CVPR 2025\] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control](gen3c_3d-informed_world-consistent_video_generation_with_precise_camera_control.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)

</div>

<!-- RELATED:END -->
