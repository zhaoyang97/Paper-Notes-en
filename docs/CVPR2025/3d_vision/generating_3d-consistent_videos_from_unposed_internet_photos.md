---
title: >-
  [Paper Note] Generating 3D-Consistent Videos from Unposed Internet Photos
description: >-
  [CVPR 2025][3D Vision][Video Generation] This paper proposes KFC-W, a self-supervised method for generating 3D-consistent videos from unposed Internet photos. By jointly training multi-view inpainting and view interpolation objectives on a video diffusion model without any 3D annotations (such as camera parameters), the generated videos outperform the commercial model Luma Dream Machine in both geometric and appearance consistency.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Video Generation"
  - "3D Consistency"
  - "Internet Photos"
  - "Multi-view Learning"
  - "Self-supervised"
date: 2026-05-08
content_hash: 23654e49e4cb9fc5
---

# Generating 3D-Consistent Videos from Unposed Internet Photos

**Conference**: CVPR 2025  
**arXiv**: [2411.13549](https://arxiv.org/abs/2411.13549)  
**Code**: None (demo available on project page)  
**Area**: 3D Vision  
**Keywords**: Video Generation, 3D Consistency, Internet Photos, Multi-view Learning, Self-supervised

## TL;DR
This paper proposes KFC-W, a self-supervised method for generating 3D-consistent videos from unposed Internet photos. By jointly training multi-view inpainting and view interpolation objectives on a video diffusion model without any 3D annotations (such as camera parameters), the generated videos outperform the commercial model Luma Dream Machine in both geometric and appearance consistency.

## Background & Motivation
1. **Background**: Video foundation models have advanced rapidly (Sora/CogVideoX/Lumiere, etc.), making video generation from text/images highly mature. 3D reconstruction methods (NeRF/3DGS) achieve excellent results in controlled scenes but require dense views and camera parameters.
2. **Limitations of Prior Work**: Existing video models (e.g., Luma Dream Machine) fail to understand scene layouts when given sparse Internet photos, inventing new buildings/structures to "transition" between keyframes, which essentially performs morphing rather than true camera motion. Frame interpolation methods (e.g., FILM) only work under small baselines and cannot handle large viewpoint differences.
3. **Key Challenge**: General video models lack 3D-awareness, and introducing 3D supervision (camera poses) requires SfM pre-processing, which is computationally expensive and prone to failure (e.g., SfM succeeded in only 1/5 of the scenes in MegaScenes).
4. **Goal**: To train a model capable of generating 3D-consistent videos from sparse Internet photos without using any 3D annotations.
5. **Key Insight**: Videos and multi-view images are complementary—videos provide temporal continuity with small viewpoint changes, while Internet photos provide large viewpoint diversity but lack temporal relationships. Jointly utilizing both can unlock 3D-awareness.
6. **Core Idea**: Co-train a video diffusion model with two self-supervised objectives: multi-view inpainting (to learn 3D priors from Internet photos) and view interpolation (to learn temporal coherence from videos), achieving scene-level 3D understanding with zero 3D annotation.

## Method

### Overall Architecture
Based on fine-tuning a latent Diffusion Transformer (DiT). The input consists of 2-5 unposed Internet photos, and the output is an interpolated video of 15 frames between each adjacent pair of keyframes. During training, two objectives are alternated: (1) multi-view inpainting on MegaScenes multi-view photos, and (2) view interpolation on RealEstate10k/DL3DV videos. Both objectives are unified under the same denoising framework, where the model automatically identifies the task based on the input format.

### Key Designs

1. **Multiview Inpainting**:
    - **Function**: Learn 3D geometric priors from unordered Internet photos without camera poses.
    - **Mechanism**: Given $n$ condition images of the same scene and 1 target image, noise is added to 80% of the target image's patches, while keeping the condition images and the remaining 20% of the target image unchanged. The model extracts structural information and scene identity from the condition images via self-attention, and obtains illumination and layout from the visible parts of the target image to generate the complete target image. Training uses MegaScenes (8M Internet photos, 430K scenes), requiring only "same-scene" labels without camera parameters. A frozen semantic segmentation model is used to filter out transient objects like pedestrians and vehicles.
    - **Design Motivation**: Inspired by CroCo but with three major improvements: supporting an arbitrary number of inputs (CroCo only supports two), minimizing data annotation requirements, and using a probability generative model (vs. the deterministic reconstruction of MAE). The learned priors (such as symmetry and depth awareness) are crucial for large-baseline video generation.

2. **View Interpolation**:
    - **Function**: Learn smooth and consistent camera motion from the start frame to the end frame.
    - **Mechanism**: Randomly sample $16(n-1)+1$ frames from a video, where every 16th frame is a conditioning frame (kept clean), and the remaining 15 frames are target frames (noised). Frame index embeddings (0, 1, 2, ..., via a linear layer) are added to all patches. The model is trained to generate smooth transitions for intermediate frames while keeping the start and end frames unchanged.
    - **Design Motivation**: Using this objective alone can only handle small-baseline videos (e.g., Re10k) and cannot cope with the large viewpoint differences and illumination variations in Internet photos. However, each conditioning frame provides information to all frames via self-attention, enabling the model to learn to generate temporally coherent sequences.

3. **CLIP-embedded Illumination Control**:
    - **Function**: Maintain consistent lighting conditions across Internet photos.
    - **Mechanism**: During training, a frozen CLIP encoder extracts global features from each image, which are reshaped and added to the image patch tokens. Extreme ColorJitter enhancement is applied to conditioning frames to force the model to obtain lighting information from the CLIP embedding (rather than the conditioning frames themselves). At inference time, the CLIP embedding of the first input image is used to unify the lighting of all interpolated frames.
    - **Design Motivation**: Large lighting differences exist between Internet photos (e.g., sunny vs. cloudy). Uncontrolled models would generate frames with inconsistent lighting. CLIP embeddings capture coarse-grained lighting information ("cloudy"/"sunny"), which, although not precise, is sufficient for maintaining consistency.

### Loss & Training
- Standard diffusion denoising objective: loss is calculated only on noisy patches.
- Unified objectives: the model automatically infers the task based on the number and format of input frames, requiring no task flags.
- Training data: MegaScenes (multi-view inpainting) + RealEstate10k + DL3DV (view interpolation).
- Training configuration: trained on 32 A100 80G GPUs for 3 days.
- Inference: 50-step DDIM sampling, decoding only the intermediate frames.

## Key Experimental Results

### Main Results (User Study, Ours(Full) Win Rate)

| Compared Method | Consistency | Camera Motion | Aesthetics | Dataset |
|---------|--------|---------|--------|--------|
| FILM | 100.0% | 100.0% | 100.0% | Phototourism |
| FLAVR | 100.0% | 100.0% | 100.0% | Phototourism |
| Ours(Video-Only) | 100.0% | 96.7% | 96.7% | Phototourism |
| **Luma Dream Machine** | **60.2%** | **73.6%** | **60.2%** | **Phototourism** |
| Luma Dream Machine | 83.7% | 84.9% | 68.3% | Re10k |

### Ablation Study (COLMAP Reconstruction Success Rate)

| Configuration | SfM Success Rate | Registered Image Rate | Description |
|------|----------|----------|------|
| Original sparse photos only | 45% (115/252) | 43% (378/882) | Baseline |
| + Generated frames (Full) | **93% (235/252)** | **84% (741/882)** | Strong geometric consistency |
| + Generated frames (Video-Only) | 71% (179/252) | 67% (589/882) | Multi-view inpainting is key |

### Key Findings
- Frame interpolation methods (FILM/FLAVR/LDMVFI) completely fail on large-baseline Internet photos (0% win rate), as they are only suited for small motions.
- The model trained solely with the view interpolation objective performs well on Re10k but fails on Phototourism, proving that the multi-view inpainting objective is critical for wild scenes.
- Training with wider-baseline videos (Long-Video ablation) still cannot replace multi-view inpainting, because Internet photos contain extreme rotations/scaling rarely seen in videos.
- Generated frames boost the COLMAP success rate from 45% to 93%, indicating that the frames possess genuine geometric consistency.
- PSNR/SSIM/LPIPS for 3DGS trained on the generated frames systematically outperform those trained on the original sparse photos.

## Highlights & Insights
- The **emergent capability of self-supervised multi-task learning** is stunning: the model has never seen pairs of "Internet photos as input and consistent video as output." However, through joint training of multi-view inpainting and view interpolation, this capability naturally emerges.
- The ** complementarity between Internet photos and videos** is the core insight: photos provide viewpoint diversity (extreme rotation/scale), and videos provide temporal coherence. Combining both outperforms what either can achieve individually.
- **ColorJitter hard augmentation forcing the model to use CLIP conditioning** is a clever engineering trick: by destroying the color information of the conditioning frames, the model is forced to obtain lighting from the CLIP embedding, achieving controllable lighting during inference.
- This paradigm can be transferred to other tasks requiring 3D understanding but lacking 3D annotations, such as autonomous driving scene generation and indoor navigation.

## Limitations & Future Work
- Does not handle dynamic objects; only applicable to static scenes.
- CLIP embeddings only provide coarse-grained lighting control, failing to precisely control physical attributes like sun angle.
- The number of generated frames is fixed to 15 frames per keyframe pair; scaling up to longer or denser sequences requires more computation.
- Based on an internal text-to-video model (not open-sourced), though the method can adapt to any DiT architecture.
- The number of input keyframes is limited to 2-5; scalability to more inputs is unverified.
- Comparison with Luma is not entirely fair (Luma has higher resolution and a larger model).

## Related Work & Insights
- **vs Luma Dream Machine**: A commercial general video model, but it performs "creative warping" rather than true camera motion on this task. This work achieves scene layout understanding through a 3D-aware training objective.
- **vs CroCo**: CroCo performs masked modeling on dual views to learn 3D priors but only supports two images and is deterministic. This work extends to multiple images and probabilistic generation.
- **vs Zero-1-to-3/ZeroNVS**: Requires camera poses for conditioning; this work requires no poses at all.
- **vs VFusion3D**: Uses the consistency of video models to generate 3D assets; this work does the reverse, using 3D-aware training to make video models more consistent.
- This work shows that pure 2D data (videos + multi-view photos) can replace expensive 3D annotations for scene-level 3D learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The task definition is novel, the self-supervised approach is clever, and the joint multi-view inpainting and view interpolation design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated from multiple angles including user studies, COLMAP, and 3DGS, though it lacks quantitative frame-by-frame evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative, clear diagrams, and motivation is well-articulated.
- Value: ⭐⭐⭐⭐⭐ Proposes a scalable 3D learning paradigm that inspires both the scene understanding and video generation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](stereo4d_learning_how_things_move_in_3d_from_internet_stereo_videos.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)

</div>

<!-- RELATED:END -->
