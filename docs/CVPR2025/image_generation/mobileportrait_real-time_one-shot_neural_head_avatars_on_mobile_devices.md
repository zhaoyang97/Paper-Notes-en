---
title: >-
  [Paper Note] MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices
description: >-
  [CVPR 2025][Image Generation][neural head avatar] Proposes MobilePortrait, the first one-shot neural head avatar animation method capable of running in real-time on mobile devices. By combining mixed explicit-implicit keypoints with precomputable appearance knowledge, it achieves SOTA-comparable quality (100–600+ GFLOPs) using only 16 GFLOPs.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "neural head avatar"
  - "face reenactment"
  - "mobile deployment"
  - "mixed keypoints"
  - "lightweight U-Net"
date: 2026-05-08
content_hash: ca177a509fa21b3b
---

# MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices

**Conference**: CVPR 2025  
**arXiv**: [2407.05712](https://arxiv.org/abs/2407.05712)  
**Code**: To be confirmed  
**Area**: Image Generation  
**Keywords**: neural head avatar, face reenactment, mobile deployment, mixed keypoints, lightweight U-Net

## TL;DR

Proposes MobilePortrait, the first one-shot neural head avatar animation method capable of running in real-time on mobile devices. By combining mixed explicit-implicit keypoints with precomputable appearance knowledge, it achieves SOTA-comparable quality (100–600+ GFLOPs) using only 16 GFLOPs.

## Background & Motivation

**Background**: Existing Neural Head Avatar (NHA) methods (e.g., Real3D, MCNet, FaceV2V) have made significant progress in image quality and motion range. However, their computational costs generally exceed 100 GFLOPs, and they contain complex modules such as multi-scale feature warping, dynamic convolutions, and attention mechanisms, making them impossible to run on mobile devices.

**Core Problem**: With the popularization of LLMs and smartphones, mobile-side avatars will become key interfaces for AI interaction, yet there is almost no exploration of lightweight NHAs in the industry. How can the computational complexity be reduced by an order of magnitude while maintaining SOTA quality?

**Key Observations**:
1. Explicit modeling (e.g., Real3D using 3DMM) performs poorly in non-facial regions (inside the mouth, neck) because these regions are not defined.
2. Implicit modeling (e.g., MCNet using neural keypoints) suffers from blurriness at the boundary between the subject and the background due to the lack of face priors.
3. Combining both can be complementary—face knowledge strengthens the motion network, while appearance knowledge strengthens the synthesis network.

**Motivation**: Simplify complex tasks into an "open-book exam"—reduce the difficulty of network learning by introducing external prior knowledge, thereby achieving high-quality results even with a simple U-Net as the backbone.

## Method

### Overall Architecture

MobilePortrait consists of two major modules: **Motion Generation** and **Image Synthesis**, both of which use a simple U-Net (without multi-scale feature warping, dynamic convolutions, or attention) as the backbone. Given a source image and a driving frame, the pipeline proceeds through mixed keypoint detection $\rightarrow$ TPS transformation $\rightarrow$ Dense Motion Network (DMN) for optical flow generation $\rightarrow$ image warping $\rightarrow$ synthesis network to generate the final image. Since the face and appearance knowledge only need to be precomputed once given the source image, there is almost no additional overhead during inference.

### Key Design 1: Mixed Keypoint Representation

- **Problem**: After reducing computational complexity, pure neural keypoints (NK) cannot distinguish between facial and background motion, causing "liquefied" artifacts; pure facial keypoints (FK) fail to capture global motion.
- **Solution**: Introduce a pre-trained facial keypoint detector to extract 106 FKs, which are merged with 50 NKs into 50 mixed keypoints, fused via an MLP.
- **Auxiliary Design**: Add residual optical flow (2-channel output) in the last layer of the dense motion network to enhance the expressive power of optical flow.
- **Effect**: Mixed keypoints achieve a significant improvement compared to NK-Only (FID 48.3 $\rightarrow$ 29.2) and FK-Only (FID 33.2 $\rightarrow$ 29.2).

### Key Design 2: Face-Aware Motion Generation

- **Extra Input**: Add the foreground mask and facial keypoint mask of the source image to the inputs of the dense motion network (only needs to be computed once).
- **Auxiliary Training Losses**: Add two auxiliary predictors at the final feature layer of the DMN to predict the foreground mask ($\mathcal{L}_{mask}$) and keypoint mask ($\mathcal{L}_{landmark}$) of the driving image, respectively, trained with L1 loss (only present during training).
- **Effect**: Helps the network understand portrait integrity, achieving motion capture from the facial level to the video level.

### Key Design 3: Appearance Knowledge Enhancement

- **Foreground Enhancement**: Uniformly sample $T$ frames from the driving video to generate $T$ warped images with the source image, and extract features from the lowest resolution layer of the U-Net as pseudo-multi-view features. These are fused with current frame features through an additional convolutional layer. Experiments show that $T=4$ yields the best results.
- **Background Enhancement**: Use an offline inpainting model (LaMa) to complete the source image after removing the foreground, obtaining a complete background map as an additional input to the synthesis network. During training, inpainting is also performed on the driving image to ensure the model can utilize the background information.
- **Key Advantage**: All appearance knowledge can be precomputed, resulting in almost zero additional cost during inference.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{percep} + \mathcal{L}_{L1} + \mathcal{L}_{kp} + \mathcal{L}_{eq} + \mathcal{L}_{landmark} + \mathcal{L}_{mask}$$

- $\mathcal{L}_{percep}$: Perceptual loss, optimizing feature distance.
- $\mathcal{L}_{L1}$: Pixel-level L1 loss.
- $\mathcal{L}_{kp}$: Facial landmark distance loss.
- $\mathcal{L}_{eq}$: Equivariance loss, ensuring the stability of neural keypoints.
- $\mathcal{L}_{landmark}$, $\mathcal{L}_{mask}$: Facial knowledge auxiliary losses.

## Key Experimental Results

### Main Results

Comparison with SOTA methods on same- and cross-identity video-driven reenactment (Table 1):

| Method | FLOPs(G) | FID↓ | AKD↓ | HPD↓ | BCI↑ | CSIM↑(cross) |
|------|----------|------|------|------|------|---------------|
| PIRender | 131 | 39.1 | 2.14 | 0.99 | 96.9 | 45.7 |
| FaceV2V | 629 | 29.3 | 1.96 | 2.52 | 97.2 | 46.0 |
| TPS | 140 | 29.8 | 1.43 | 0.71 | 97.9 | 38.9 |
| MCNet | 200 | 27.2 | 1.33 | 0.81 | 97.8 | 27.6 |
| Real3D | 610 | 50.8 | 1.63 | 0.82 | 97.6 | 47.8 |
| **MobilePortrait** | **16** | **29.2** | **1.30** | **0.40** | **98.2** | **39.2** |

- With only 16 GFLOPs (less than 1/12 of MCNet), it achieves the **best** AKD and BCI, and its HPD score of 0.40 significantly outperforms all other methods.

### Ablation Study

**Motion Generation Ablation (Table 3)**:
- Mixed Keypoints vs NK-Only: FID 29.2 vs 48.3, AKD 1.30 vs 2.62.
- Removing facial knowledge loss: AKD degrades from 1.30 to 1.45.
- Removing residual optical flow: AKD degrades from 1.30 to 1.45.

**Image Synthesis Ablation (Table 4)**:
- Without any enhancement: FID 30.1, AKD 1.54.
- Inpainted BG only: FID 29.2, AKD 1.30 (best combination).
- Number of multi-views: 4 views is the optimal trade-off.

### Key Findings

1. **On-device Evaluation**: The latency of the 16G version is 15.8ms (63 FPS) on an iPhone 14 Pro, whereas the 4G lite version takes 5.9ms (169 FPS); on an iPhone 12, the 16G version achieves 25.5ms (39 FPS).
2. Even when compressed to 4 GFLOPs, it still maintains satisfactory FID and AKD with the help of external knowledge.
3. The mixed keypoint fusion method (MLP merger) outperforms alternatives such as transformation cascading or convolution-generated optical flow.
4. Training-time background inpainting is crucial for utilizing background information.

## Highlights & Insights

1. **Unique Compression Concept**: Instead of merely compressing the model, it reduces task difficulty by introducing external knowledge—turning a "closed-book exam" into an "open-book exam," enabling simple networks to perform competently.
2. **Explicit-Implicit Complementarity**: The mixed keypoint design is simple and effective, requiring only an MLP merger to avoid complex multi-scale or multi-stage integration.
3. **Ingenious Precomputation Strategy**: The facial mask, multi-view features, and background image only need to be computed once for the source image, incurring zero additional overhead during inference.
4. **Audio-Driven Extension**: Audio-driven animation is achieved via an audio-to-mesh + mesh-to-NK module, and expression editing is supported through 3DMM.

## Limitations & Future Work

1. The CSIM (identity preservation) metric in cross-identity scenarios is sub-optimal, although the visual results are acceptable.
2. It heavily relies on the quality of the pre-trained facial keypoint detector, which may limit performance in non-frontal or occluded scenarios.
3. The audio-driving module only provides a baseline solution, and more complex audio-to-motion designs could yield further improvements.
4. Background inpainting depends on the quality of LaMa, and complex backgrounds may introduce artifacts.

## Related Work & Insights

- **FOMM/TPS/MCNet**: Representative works of implicit motion modeling. This work introduces complementary facial priors on top of them.
- **Real3D/PIRenderer**: Representative works of explicit 3DMM-driven modeling. This work extracts the advantages of their facial priors while avoiding the loss of global motion.
- **SadTalker/VividTalk**: References for audio-driven schemes, with which MobilePortrait is compatible.
- **Insight**: Many "challenging" tasks can significantly reduce model complexity by introducing cheap external knowledge. This approach is worth generalizing to other real-time tasks (such as real-time face swapping or real-time segmentation).

## Rating

⭐⭐⭐⭐ — The first real-time NHA method on mobile devices, comparable to SOTA with a 10x reduction in computational cost, offering extremely high practical value. The core idea (reducing task difficulty with external knowledge) is simple and inspiring. 1 star is deducted because the cross-identity preservation and audio-driven components are relatively preliminary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices](../../CVPR2026/image_generation/efficient_real-time_raw-to-raw_denoising_for_extreme_low-light_ultra_hd_video_on.md)
- [\[CVPR 2025\] SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device](snapgen-v_generating_a_five-second_video_within_five_seconds_on_a_mobile_device.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)

</div>

<!-- RELATED:END -->
