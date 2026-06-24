---
title: >-
  [Paper Note] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control
description: >-
  [CVPR 2025 (Highlight)][Video Generation][3D-consistent video generation] GEN3C proposes a video generation framework guided by a 3D cache (point cloud cache). By predicting depth for seed images and unprojecting them into 3D point clouds, it renders the 3D cache into 2D condition maps according to user-specified camera trajectories when generating subsequent frames, thereby achieving precise camera control and cross-frame 3D consistency.
tags:
  - "CVPR 2025 (Highlight)"
  - "Video Generation"
  - "3D-consistent video generation"
  - "camera control"
  - "point cloud cache"
  - "novel view synthesis"
  - "video diffusion models"
date: 2026-05-08
content_hash: 883cc64fab5a3829
---

# GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2503.03751](https://arxiv.org/abs/2503.03751)  
**Code**: [https://github.com/nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C)  
**Area**: Autonomous Driving / Video Generation / 3D Vision  
**Keywords**: 3D-consistent video generation, camera control, point cloud cache, novel view synthesis, video diffusion models

## TL;DR
GEN3C proposes a video generation framework guided by a 3D cache (point cloud cache). By predicting depth for seed images and unprojecting them into 3D point clouds, it renders the 3D cache into 2D condition maps according to user-specified camera trajectories when generating subsequent frames, thereby achieving precise camera control and cross-frame 3D consistency.

## Background & Motivation

**Background**: Video diffusion models (such as Sora and Cosmos) can generate highly realistic videos, but these models primarily operate in 2D space and have limited understanding of 3D geometry. Some works attempt to control the viewpoint of generated videos by using camera parameters as conditioning inputs.

**Limitations of Prior Work**: (1) Poor 3D consistency—purely 2D video models often suffer from objects suddenly appearing/disappearing or inconsistent deformations; (2) Imprecise camera control—directly feeding camera intrinsics and extrinsics into the network requires the model to implicitly learn the mapping from camera parameters to image structures, which is highly unreliable in complex scenes; (3) Consistency degradation in long videos—as the frame count increases, the model "forgets" previously generated content, leading to temporal inconsistency.

**Key Challenge**: 2D video diffusion models lack explicit 3D geometric representations, which fundamentally prevents them from guaranteeing multi-view consistency and precise camera motion in generated videos. Conditioning purely on 2D camera parameters is "asking the model to guess" rather than "allowing the model to see."

**Goal**: To design a video generation framework that possesses both precise camera control and 3D temporal consistency, applicable to various tasks such as single-image generation, multi-image scene reconstruction, and dynamic video re-rendering.

**Key Insight**: The authors' key observation is that explicitly lifting the content of existing frames into 3D space (point clouds) and rendering them back to 2D according to new camera poses provides a strongly constrained structural prior for the video generation model. The model no longer needs to "remember" what was previously generated, nor does it need to "infer" image structures from camera parameters.

**Core Idea**: Introducing the concept of "3D Cache"—reprojecting seed images or previously generated frames into a 3D point cloud via depth estimation. During the generation of new frames, this point cloud is rendered into 2D condition maps along user-specified camera trajectories, and these conditions are injected into the video diffusion model.

## Method

### Overall Architecture
The pipeline of GEN3C is divided into three core stages: (1) **3D Cache Construction**—predicting pixel-level depth for input images/frames and unprojecting them into a 3D point cloud; (2) **Cache Rendering**—rendering the 3D point cloud into 2D conditioning videos (including color and depth channels) based on user-provided new camera trajectories; (3) **Conditioned Video Generation**—feeding the rendered conditioning video into a video diffusion model to generate the final photorealistic video. This process can be repeated autoregressively, where newly generated frames are also appended to the 3D cache.

### Key Designs

1. **3D Cache Construction and Maintenance**:

    - Function: To build an explicit 3D geometric representation of the scene as a spatial prior for video generation.
    - Mechanism: For each seed image or generated frame, a pre-trained monocular depth estimation model (e.g., Metric3D, DPT) is used to predict pixel-wise depth maps. The depth map is then unprojected into a 3D point cloud $P = K^{-1} [u, v, 1]^T \cdot d$ using camera intrinsics, where $d$ denotes the depth value. Point clouds from multiple frames/views are unified into the world coordinate system via known camera extrinsics to form an accumulating "3D Cache." Each point in the cache carries color and confidence information.
    - Design Motivation: The explicit 3D representation allows the model to precisely know "what should be seen from a new camera angle," transforming the guarantee of 3D consistency from implicit network learning to explicit geometric constraints. This is more lightweight than directly employing 3D Gaussian Splatting, making it highly compatible with video diffusion models.

2. **Rendering 3D Cache to 2D Conditions (Cache Rendering)**:

    - Function: To render the 3D cache into 2D conditioning maps based on user-specified new camera trajectories, guiding video generation.
    - Mechanism: For each frame along the user-specified target camera trajectory, the 3D point cloud is projected to that viewpoint to obtain a rendered color map and a depth map. The rendering results are fed into the diffusion model as 2D conditioning maps alongside noisy latents. Regions with existing information in the rendered maps (parts visible from previous frames) provide strong structural constraints, whereas empty regions (newly disoccluded areas) are left for the diffusion model to generate. A visibility mask is also introduced to identify pixels with valid renderings.
    - Design Motivation: This design focuses the generation capabilities of the diffusion model on "previously unseen regions" and "scene evolution," rather than wasting model capacity on "remembering what was previously generated" and "inferring structures from camera parameters."

3. **Conditioned Video Diffusion Generation**:

    - Function: To generate high-quality, realistic video frames based on rendered conditioning maps.
    - Mechanism: Additional conditioning input channels are incorporated based on pre-trained video diffusion models (e.g., Cosmos or SVD architectures). The rendered color map, depth map, and visibility mask are concatenated along the channel dimension of the noisy latents. Through fine-tuning, the model learns to utilize this conditioning information: maintaining consistency with conditions in regions with rendered data, hallucinating content reasonably in empty areas based on context, and enhancing realism across all regions (e.g., repairing rendering artifacts such as point cloud holes and projection approximations).
    - Design Motivation: Directly rendering 3D point clouds often yields images with holes and artifacts, requiring the diffusion model to perform "inpainting" and "refinement." Treating the rendering results as conditions rather than final inputs provides the model with sufficient flexibility to generate realistic content.

### Loss & Training
- Standard video diffusion denoising loss: $L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$, where $c$ includes the rendered conditioning maps and text descriptions.
- Training data: Utilizing multi-view/video datasets containing camera poses and depth annotations (e.g., RealEstate10K, DL3DV, nuScenes).
- Autoregressive generation strategy: Generated frames are updated back into the 3D cache, supporting arbitrary-length video generation.

## Key Experimental Results

### Main Results
Comparison with SOTA methods across multiple tasks:

**Sparse-View Novel View Synthesis (RealEstate10K, 5 input images)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | 3D Consistency↑ |
|------|-------|-------|--------|----------|
| PixelNeRF | 20.4 | 0.72 | 0.31 | - |
| ZeroNVS | 21.8 | 0.76 | 0.26 | 0.82 |
| ViewCrafter | 23.1 | 0.79 | 0.22 | 0.86 |
| **GEN3C (SVD)** | **24.6** | **0.82** | **0.18** | **0.91** |
| **GEN3C (Cosmos)** | **25.3** | **0.84** | **0.16** | **0.93** |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | Description |
|------|-------|--------|------|
| Full model | 25.3 | 0.16 | Full GEN3C |
| w/o 3D cache (pure camera conditioning) | 21.5 | 0.28 | Degrates to conventional camera-conditioned generation |
| w/o depth map condition | 23.8 | 0.20 | Utilizing only rendered color maps |
| w/o visibility mask | 24.5 | 0.18 | Model cannot distinguish between valid and invalid regions |
| w/o autoregressive cache update | 23.2 | 0.22 | Newly generated frames are not added to the cache |

### Key Findings
- **3D Cache is the most critical design**: Removing the 3D cache (degrading to pure camera conditioning) results in a 3.8dB drop in PSNR, demonstrating the importance of explicit 3D geometric priors.
- **Autoregressive cache updates are crucial for long-sequence generation**: Failing to update the cache causes the rendered conditioning inputs for subsequent frames to become increasingly sparse, leading to a frame-by-frame quality degradation.
- GEN3C also performs excellently in challenging settings such as driving scenarios and monocular dynamic videos, exhibiting robust generalization.
- The Cosmos base model performs better than SVD, showing that more powerful base video models yield superior 3D consistency.

## Highlights & Insights
- **The design philosophy of the 3D cache is exceptionally elegant**—replacing "remembering" with "seeing" and "inferring" with "rendering," fundamentally addressing 3D consistency and camera control issues in video generation. This approach can be generalized to all video/image generation tasks requiring viewpoint consistency.
- **Plug-and-play framework**—GEN3C can be deployed across different base video models (SVD, Cosmos), proving the strong versatility of the 3D cache conditioning mechanism.
- **Driving simulation application is highly valuable**—allowing the generation of multi-angle simulation data from real driving videos, which can be used to train autonomous vehicles.

## Limitations & Future Work
- The accuracy of depth estimation directly affects the quality of the 3D cache; depth estimation remains unreliable for reflective surfaces, transparent objects, and distant regions.
- Point cloud representation underperforms meshes or 3DGS in preserving fine details, resulting in holes in rendered conditioning maps that require the model to hallucinate details.
- The handling of depth and positional changes for moving objects in dynamic scenes is not yet sufficiently refined.
- Significant computational overhead—requiring extra depth estimation and 3D rendering steps.

## Related Work & Insights
- **vs CamCo/MotionCtrl**: These methods inject camera parameters as MLP embeddings into the diffusion model, essentially requiring the network to implicitly learn the camera-to-image mapping. GEN3C makes this mapping deterministic via explicit 3D rendering, significantly improving camera control precision.
- **vs ViewCrafter**: ViewCrafter employs point clouds for conditioning but lacks an autoregressive cache update mechanism. GEN3C achieves superior long-sequence consistency by accumulating the cache.
- **vs ReconFusion/ZeroNVS**: These sparse-view reconstruction methods utilize NeRF/3DGS representations, which suffer from high computational overhead and fail to support dynamic scenes. GEN3C's point cloud cache solution is more lightweight and natively supports video.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The framework of 3D-cache-guided video generation is highly creative, making its CVPR Highlight recognition well-deserved.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage of four scenarios (single-image, multi-image, dynamic video, and driving simulation) and validated across two base models.
- Writing Quality: ⭐⭐⭐⭐⭐ From NVIDIA, clearly and fluidly written with outstanding visualizations.
- Value: ⭐⭐⭐⭐⭐ Outstanding framework versatility, highly promising driving simulation application prospects, and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](../../ICCV2025/video_generation/realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICLR 2026\] 3D Scene Prompting for Scene-Consistent Camera-Controllable Video Generation](../../ICLR2026/video_generation/3d_scene_prompting_for_scene-consistent_camera-controllable_video_generation.md)
- [\[ICML 2026\] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](../../ICML2026/video_generation/epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)
- [\[ICLR 2026\] MoCa: Modeling Object Consistency for 3D Camera Control in Video Generation](../../ICLR2026/video_generation/moca_modeling_object_consistency_for_3d_camera_control_in_video_generation.md)

</div>

<!-- RELATED:END -->
