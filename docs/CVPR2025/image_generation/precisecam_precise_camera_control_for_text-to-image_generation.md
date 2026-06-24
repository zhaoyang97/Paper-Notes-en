---
title: >-
  [Paper Note] PreciseCam: Precise Camera Control for Text-to-Image Generation
description: >-
  [CVPR 2025][Image Generation][Camera Control] PreciseCam achieves precise camera perspective control in text-to-image generation through 4 camera parameters (roll, pitch, vFoV, distortion $\xi$) and Perspective Field-Unified Spherical representation, without requiring 3D geometry or multi-view data.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Camera Control"
  - "Text-to-Image Generation"
  - "Perspective Field"
  - "ControlNet"
  - "Cinematography"
date: 2026-05-08
content_hash: fc2449340ac9b045
---

# PreciseCam: Precise Camera Control for Text-to-Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.12910](https://arxiv.org/abs/2501.12910)  
**Code**: [Project Page](https://graphics.unizar.es/projects/PreciseCam2024)  
**Area**: 3D Vision / Camera Control  
**Keywords**: Camera Control, Text-to-Image Generation, Perspective Field, ControlNet, Cinematography

## TL;DR

PreciseCam achieves precise camera perspective control in text-to-image generation through 4 camera parameters (roll, pitch, vFoV, distortion $\xi$) and Perspective Field-Unified Spherical representation, without requiring 3D geometry or multi-view data.

## Background & Motivation

### Background

**Background**: The "cinematography" of an image is crucial for conveying emotions (e.g., low angle for dignity, Dutch angle for tension), but existing T2I models severely lack camera control capabilities:

### Limitations of Prior Work

**Limitations of Prior Work**: Generated images typically exhibit an **eye-level perspective**, where the camera is parallel to the ground plane and the horizon is centered.

### Key Challenge

**Key Challenge**: Prompt engineering is the only means of control, which is coarse and requires heavy trial-and-error.

### Core Idea

**Core Idea**: Models like Firefly offer tag-based controls such as "wide angle" or "high angle", but these lack precision.

### Additional Notes

**Additional Note**: Methods based on 3D representations (such as NeRF) require multi-view images and struggle with complex scenes and multiple objects.

### Additional Notes

**Additional Note**: Using ControlNet with depth or edge maps imposes strict constraints that go beyond camera control.

### Additional Notes

**Additional Note**: While relative camera motion control (video generation) has been explored, **absolute camera pose control** remains overlooked.

### Additional Notes

**Additional Note**: A generic framework is needed to achieve precise control using only simple and intuitive camera parameters.

## Method

### Overall Architecture

The user provides a text prompt $p$ and four camera parameters $\Omega = (\text{roll}, \text{pitch}, \text{vFoV}, \xi)$. These parameters are converted into a pixel-wise representation map called PF-US (Perspective Field - Unified Spherical). The PF-US map is injected into the generation process of SDXL via a ControlNet module. Only the ControlNet module is trained, while SDXL weights remain frozen.

### Key Design 1: PF-US Camera Perspective Representation

**Function**: Converts the 4 camera parameters into a pixel-wise geometric feature map, encoding the impact of camera parameters on the appearance of each pixel.

**Mechanism**: Based on the Perspective Field representation, each pixel is assigned an up-vector $\mathbf{u}_x$ (reflection of the gravity direction) and a latitude angle $\varphi_x$ (the angle between the ray and the horizontal plane). Utilizing the Unified Spherical camera model, the projection function is defined as $u = \frac{xf}{\xi\sqrt{x^2+y^2+z^2}+z} + u_0$, where the parameter $\xi \in (0,1)$ controls the degree of distortion ($\xi=0$ represents a pinhole camera). The PF-US map is computed solely from camera parameters, requiring no 3D scene geometry.

**Design Motivation**: PF-US provides **local pixel-level** information instead of global 3D representations, allowing the model to learn the relationship between camera parameters and pixel appearance without relying on heavy 3D representations. The yaw parameter is excluded, as there is no absolute left-right reference direction in a 2D image.

### Key Design 2: ControlNet Mid-Block Injection Strategy

**Function**: Achieves precise camera conditional control while minimizing interference with generation quality.

**Mechanism**: During training, the outputs of the ControlNet encoder and middle blocks are injected into the SDXL U-Net bottleneck and decoder skip connections via zero convolutions. During inference, it is observed that **injecting only the middle block** (bottleneck) output improves generation consistency without compromising condition adherence. This is because camera parameters are global properties (roll and pitch affect the entire image), and the global features of the middle block are sufficient for encoding.

**Design Motivation**: Full injection over-constrains the generation process, leading to degraded content quality; injecting only the middle block yields the best balance between control accuracy and generation quality.

### Key Design 3: 360° Image Dataset Construction

**Function**: Provides 57,380 training images with ground-truth (GT) camera parameters, covering a wide range of camera parameter values.

**Mechanism**: Camera parameters are sampled from 6 different 360° image datasets: roll ∈ (-90°, 90°), pitch ∈ (-90°, 90°), vFoV ∈ [15°, 140°], and $\xi \in (0,1)$. Corresponding perspective regions are cropped from the 360° images based on each parameter set, and the PF-US maps are calculated. BLIP-2 is used to generate text descriptions for each cropped image. Inaccuracies in BLIP-2 descriptions do not hinder training, as the ControlNet is tasked with learning camera perspectives independent of prompt semantics.

**Design Motivation**: Existing PF datasets mostly feature urban outdoor scenes with a narrow range of parameters (lacking large distortions and field of views); 360° images naturally contain all possible perspective directions.

### Loss & Training

Standard ControlNet training loss (diffusion model denoising loss) is employed. SDXL remains frozen, and only the ControlNet module is trained.

## Key Experimental Results

### Main Results: Camera Parameter Adherence Evaluation

| Method | Roll Error ↓ | Pitch Error ↓ | FoV Error ↓ |
|------|-----------|------------|----------|
| **PreciseCam** | **Best** | **Best** | **Best** |
| Prompt Engineering | Poor | Poor | Poor |
| Firefly Tags | Moderate | Moderate | Moderate |
| ControlNet (Depth) | N/A | N/A | N/A |

### Ablation Study: ControlNet Injection Strategy

| Injection Layer | Camera Adherence | Image Quality |
|--------|----------|---------|
| Mid-block only | High | **Best** |
| Encoder + Mid-block | Higher | Poor |
| Full injection | Highest | Worst |

### Key Findings

- PreciseCam provides precise control over roll, pitch, vFoV, and distortion, significantly outperforming prompt engineering approaches.
- Injecting the mid-block only is the optimal strategy—global camera attributes do not require fine-grained decoder-level control.
- The method supports both photographic and artistic image generation.
- It can be extended to video generation by providing precise initial camera poses.

## Highlights & Insights

- **Replacing 3D scenes with pixel-level representations** is an elegant solution to the camera control problem.
- The decision to exclude the yaw parameter reflects a deep understanding of the nature of 2D images.
- The dataset construction strategy (cropping from 360° images) is simple yet highly comprehensive.

## Limitations & Future Work

- Yaw (yaw rotation) is not controlled, which limits precise adjustment of left-right perspectives.
- The ControlNet-based approach may sometimes impact content diversity in complex scenes.
- The training data is derived from cropped 360° images, which may introduce resolution and stylistic biases.
- Future work can extend this to absolute camera trajectory control in video generation.

## Related Work & Insights

- The PF-US representation can be generalized to other generation tasks that require geometric condition control.
- The findings regarding ControlNet mid-block injection are also valuable for controlling other global attributes like lighting.
- It is complementary to relative camera control methods in video generation, such as CameraCtrl, by providing absolute starting positions.

## Rating

⭐⭐⭐ — Addresses a practical and important creative tool challenge. Although the PF-US representation is elegantly designed, the framework is primarily an application of ControlNet. The contributions of the dataset construction and the inference strategy are highly commendable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model](camfreediff_camera-free_image_to_panorama_generation_with_diffusion_model.md)
- [\[CVPR 2026\] Camera Control for Text-to-Image Generation via Learning Viewpoint Tokens](../../CVPR2026/image_generation/camera_control_for_text-to-image_generation_via_learning_viewpoint_tokens.md)
- [\[CVPR 2025\] GPS as a Control Signal for Image Generation](gps_as_a_control_signal_for_image_generation.md)
- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
