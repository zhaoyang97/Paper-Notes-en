---
title: >-
  [Paper Note] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency
description: >-
  [CVPR 2025][Video Generation][Online Video Synthesis] This paper proposes a geometry-guided online video view synthesis method. It constructs view- and temporally-consistent depth representations through progressive depth map optimization and Truncated Signed Distance Field (TSDF) accumulation, and subsequently uses this depth to guide a pre-trained image blending network, achieving highly efficient and consistent novel-view video synthesis.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Online Video Synthesis"
  - "Multi-View Consistency"
  - "Temporal Consistency"
  - "Depth Guidance"
  - "TSDF"
date: 2026-05-08
content_hash: eaa66f7123ced983
---

# Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency

**Conference**: CVPR 2025  
**arXiv**: [2505.18932](https://arxiv.org/abs/2505.18932)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Online Video Synthesis, Multi-View Consistency, Temporal Consistency, Depth Guidance, TSDF

## TL;DR
This paper proposes a geometry-guided online video view synthesis method. It constructs view- and temporally-consistent depth representations through progressive depth map optimization and Truncated Signed Distance Field (TSDF) accumulation, and subsequently uses this depth to guide a pre-trained image blending network, achieving highly efficient and consistent novel-view video synthesis.

## Background & Motivation

**Background**: Novel View Synthesis (NVS) of multi-view videos is a core technology for immersive videos, VR/AR, and free-viewpoint videos. Traditional methods acquire scene videos using dense multi-view camera arrays (e.g., dozens to hundreds of cameras) and then estimate novel views through light field interpolation or neural rendering. Recently, some methods have also attempted novel view synthesis from sparse camera inputs to reduce cost.

**Limitations of Prior Work**: (1) Dense multi-view solutions consume massive computational resources, making them unsuitable for practical deployment; (2) Although sparse input schemes lower costs, they often suffer from multi-view inconsistency (flickering) and temporal inconsistency (temporal artifacts)—manifested as sudden jumps in object appearance across different views and inter-frame flickering within the same view; (3) Offline methods (such as per-frame optimized NeRF/3DGS) require heavy computation and cannot run in real time.

**Key Challenge**: High-quality novel view synthesis requires accurate 3D geometry to ensure consistency, but estimating accurate geometry from sparse inputs is inherently challenging. Image-Based Rendering (IBR) methods are fast but produce artifacts in areas with inaccurate geometry.

**Goal**: Design an online/streaming video view synthesis method that simultaneously ensures multi-view consistency and temporal consistency with high computational efficiency.

**Key Insight**: The authors observe that adjacent frames in a video contain highly redundant information, allowing geometric information to be accumulated across frames to progressively correct depth. A key insight is that maintaining a temporally consistent depth representation (TSDF) in the image space of the synthesized view is more beneficial for output consistency than doing so in the input view spaces.

**Core Idea**: Propose a geometry-guided pipeline consisting of progressive depth refinement and TSDF accumulation. It first progressively optimizes depth maps across frames using a color difference mask, then accumulates them through TSDF in the synthesized view space to obtain a spatio-temporally consistent depth representation, and finally uses this depth to guide a pre-trained blending network to fuse multiple forward-warped input view images into the final output.

## Method

### Overall Architecture
The input consists of video streams from multiple camera views (a sparse camera array), and the goal is to synthesize videos of arbitrary novel views online. The pipeline is divided into three stages: (1) depth estimation and progressive refinement; (2) TSDF accumulation to construct consistent depth; and (3) geometry-guided image blending. The entire system operates in an online streaming manner, processing each frame sequentially.

### Key Designs

1. **Progressive Depth Refinement**:

    - **Function**: Leverage temporal information redundancy to progressively correct errors in the initial depth estimation.
    - **Mechanism**: For each input view at each frame, an initial depth map is first obtained using monocular/multi-view depth estimation. The color difference between adjacent frames is then utilized to detect regions with inaccurate depth—if the color difference is large after warping pixels to the adjacent frame using the current frame's depth (i.e., high reprojection error), it indicates potential depth errors. A color difference mask $M_t = \|I_t(\text{warp}(p, D_t)) - I_{t-1}(p)\| > \tau$ is constructed, and depth in regions marked as inconsistent by the mask is either re-estimated or corrected using depth from adjacent frames. This process is accumulated across multiple frames, progressively optimizing the depth maps over time.
    - **Design Motivation**: Single-frame depth estimation is error-prone at occlusion boundaries, textureless regions, etc. However, these errors typically do not persist across all frames (as camera motion introduces new viewpoints). Progressive inter-frame refinement leverages temporal dimension information to "vote" for more reliable depth.

2. **TSDF Accumulation in Synthesized View Space**:

    - **Function**: Construct a view- and temporally-consistent depth representation in the image space of the target synthesized view.
    - **Mechanism**: Transform the refined depth maps of each input view into the coordinate system of the synthesized view and fuse them via a Truncated Signed Distance Field (TSDF). For each pixel of the synthesized view, TSDF values are accumulated at multiple sampling points along the ray: $\text{TSDF}(x) = \frac{\sum_t w_t \cdot \text{tsdf}_t(x)}{\sum_t w_t}$, where $w_t$ represents the temporal weight (where closer frames receive larger weights), and $\text{tsdf}_t(x)$ is the signed distance of the $t$-th frame projected to the synthesized view. The zero-crossing surface of the TSDF determines the estimated depth surface. Crucially, accumulation of TSDF occurs in the synthesized view space rather than the input view spaces, which guarantees that the output is view-consistent.
    - **Design Motivation**: Directly fusing multi-frame depth maps leads to inconsistencies due to changing viewpoints. Performing TSDF fusion in the synthesized view space unifies the depth information from different timesteps and input views into a single reference frame, naturally ensuring both view and temporal consistency. Additionally, the weighted averaging property of TSDF effectively filters out noise and outliers.

3. **Geometry-guided Blending Network**:

    - **Function**: Utilize the consistent depth provided by the TSDF to guide the blending of multi-view images, generating the final high-quality synthesized view image.
    - **Mechanism**: First, use the depth provided by the TSDF to perform forward warping/splatting on each input view image to the synthesized view, obtaining multiple forward-warped images. Since a unified depth source (the spatio-temporally consistent TSDF) is used, the forward-warping results from different input views are geometrically aligned. Then, a pre-trained U-Net blending network takes these warped images and the TSDF depth map as inputs to output the final synthesized image. The blending weights are learned by the network to perform weighted mixing in overlapping multi-view areas and inpainting in occluded/hole regions.
    - **Design Motivation**: Forward warping based on geometrically consistent depth ensures that the input images to the blending network are already roughly aligned. Consequently, the network only needs to learn local corrections and blending weights, easing the learning difficulty. Furthermore, leveraging the consistency propagation of the TSDF depth ensures that the output video is also spatio-temporally consistent.

### Loss & Training
- Blending network training loss: L1 reconstruction loss + perceptual loss (VGG) + temporal consistency loss (L1 difference between warped outputs of adjacent frames).
- The depth refinement component is unsupervised (based on reprojection color consistency).
- TSDF accumulation employs a sliding window strategy (retaining only the most recent $N$ frames) to balance accuracy and efficiency.

## Key Experimental Results

### Main Results
Comparison with SOTA methods on standard multi-view video datasets (such as Immersive/DNA-Rendering, etc.):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Temporal Consistency (TC)↑ | Online Running |
|------|-------|-------|--------|----------------|---------|
| NeRF-based (per-frame) | 28.5 | 0.90 | 0.12 | 0.85 | ✗ |
| IBRNet | 27.2 | 0.87 | 0.16 | 0.82 | ✗ |
| ENeRF | 27.8 | 0.88 | 0.14 | 0.84 | ✓ |
| MVSNeRF | 28.0 | 0.89 | 0.13 | 0.83 | ✗ |
| **GeoVS (Ours)** | **29.2** | **0.91** | **0.10** | **0.92** | **✓** |

### Ablation Study

| Configuration | PSNR↑ | TC↑ | Description |
|------|-------|-----|------|
| Full model | 29.2 | 0.92 | Full method |
| w/o progressive depth refinement | 28.1 | 0.87 | Initial depth errors affect rendering quality |
| w/o TSDF accumulation (single-frame depth) | 27.8 | 0.82 | Significant drop in temporal consistency |
| TSDF accumulated in input view space | 28.5 | 0.85 | Accumulation in synthesized view space yields better consistency |
| w/o geometry-guided blending | 28.3 | 0.86 | Blending quality degrades without TSDF guidance |
| Fixed window size = 1 | 27.6 | 0.80 | No temporal accumulation, severe flickering |

### Key Findings
- **TSDF accumulation is the core contribution**: Removing TSDF accumulation leads to a drop in temporal consistency from 0.92 to 0.82, and a 1.4 dB decrease in PSNR, demonstrating that cross-frame geometric accumulation is critical for consistency.
- **Accumulating in synthesized view space is superior to input view space**: TSDF accumulation in the synthesized view space (TC=0.92) significantly outperforms that in the input view space (TC=0.85), validating the design motivation.
- **Progressive depth refinement effectively improves initial depth quality**: Contributing approximately 1.1 dB gain in PSNR.
- The overall method can outperform multiple offline approaches even in online mode, balancing both efficiency and quality.

## Highlights & Insights
- **Performing TSDF accumulation in the synthesized view space** is the most ingenious design element—it directly guarantees output consistency, rather than relying on input consistency to indirectly ensure output quality. This concept is applicable to any video processing task requiring spatio-temporally consistent outputs.
- **Progressive depth refinement** exploits the temporal redundancy of videos—specifically, the same scene region is observed from different angles across frames, allowing cross-verification of depth estimations. This "inter-frame voting" strategy is highly practical.
- The design goal of **online/streaming processing** endows the method with significant deployment value, as it does not require access to the entire video sequence before starting.

## Limitations & Future Work
- For scenes with rapid motion or large displacements, the color difference mask may generate too many false positives, degrading depth refinement performance.
- The truncation distance of the TSDF is a global hyperparameter, making it difficult to adaptively handle regions with different depth ranges.
- The proposed method primarily targets static or slowly changing scenes, with limited capability to handle fast dynamic scenes (e.g., rapid human motion).
- Pre-training of the blending network might restrict generalization performance to entirely unseen scene types.

## Related Work & Insights
- **vs ENeRF**: ENeRF also supports online processing but utilizes a cost volume for multi-view fusion without explicit temporal accumulation. GeoVS achieves higher temporal consistency via TSDF accumulation.
- **vs IBRNet**: IBRNet is a representative image-based rendering method, but it processes each frame independently, lacking temporal modeling. The geometry-guided strategy of GeoVS enforces constraints along the temporal dimension as well.
- **vs TSDF Fusion (KinectFusion)**: While classical TSDF fusion is designed for dense 3D reconstruction, GeoVS ingeniously adapts it for novel view synthesis, introducing the innovation of accumulating in the synthesized view space.

## Rating
- Novelty: ⭐⭐⭐⭐ Accumulating TSDF in the synthesized view space is a novel concept, and the progressive depth refinement is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies cover each key module, and comparisons against other methods are thorough.
- Writing Quality: ⭐⭐⭐⭐ The methodology is logically articulated and the pipeline is easy to understand.
- Value: ⭐⭐⭐⭐ The combination of online execution and high consistency holds practical value for immersive video and VR/AR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](../../ECCV2024/video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)

</div>

<!-- RELATED:END -->
