---
title: >-
  [Paper Note] SketchVideo: Sketch-Based Video Generation and Editing
description: >-
  [CVPR 2025][Video Generation][Sketch control] Based on the DiT video generation architecture, this work proposes a memory-efficient sketch conditioning network and an inter-frame attention mechanism. It achieves precise spatial layout and geometric detail control via 1-2 sketch keyframes, while supporting sketch-based local video editing.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Sketch control"
  - "Video editing"
  - "DiT architecture"
  - "Inter-frame attention"
date: 2026-05-08
content_hash: 4e7305af76b144bf
---

# SketchVideo: Sketch-Based Video Generation and Editing

**Conference**: CVPR 2025  
**arXiv**: [2503.23284](https://arxiv.org/abs/2503.23284)  
**Code**: [http://geometrylearning.com/SketchVideo/](http://geometrylearning.com/SketchVideo/)  
**Area**: Video Generation  
**Keywords**: Sketch control, Video generation, Video editing, DiT architecture, Inter-frame attention

## TL;DR

Based on the DiT video generation architecture, this work proposes a memory-efficient sketch conditioning network and an inter-frame attention mechanism. It achieves precise spatial layout and geometric detail control via 1-2 sketch keyframes, while supporting sketch-based local video editing.

## Background & Motivation

**Background**: Text-to-video generation driven by diffusion models has made significant progress. DiT-based video generation models like CogVideoX are capable of generating temporally coherent, high-quality videos. However, text prompts can only describe high-level semantics and fail to achieve precise control over scene layout and geometric details.

**Limitations of Prior Work**: Existing video generation methods either rely on images as additional conditions (where generating the input image itself is a challenging problem), fill missing frames with white placeholder conditions as in SparseCtrl (which yields suboptimal results), or require conditional inputs for every single frame (which is overly tedious for sketch interaction). Furthermore, directly replicating half of the pre-trained blocks as a conditioning network on the DiT architecture (analogous to the PIXART-$\delta$ approach) leads to out-of-memory (OOM) issues.

**Key Challenge**: Users only need to draw sketches on 1 or 2 keyframes to convey spatial structure and motion details. However, effectively propagating control signals from such temporally sparse sketch conditions to all video frames remains a critical challenge. Simultaneously, a balance must be struck between control accuracy and memory efficiency.

**Goal**: (1) Design a memory-efficient sketch conditioning network tailored for DiT video architectures; (2) Propagate control signals from sparse keyframe sketches to all video frames; (3) Support precise sketch-based local video editing.

**Key Insight**: The authors observe that blocks at different depths of the DiT network process features at different levels. Instead of borrowing a continuous first half of the blocks like PIXART-$\delta$, a sparse, uniform skipping of a small number of blocks can cover multiple feature levels. Additionally, the inherent similarity between video frames can be leveraged to propagate sparse sketch conditions.

**Core Idea**: The proposed method replaces the approach of replicating half of the model by utilizing only 5 uniformly distributed sketch control blocks (skip-connected to the 0th, 6th, 12th, 18th, and 24th blocks out of 30 DiT blocks) to predict residual features. A novel inter-frame attention mechanism (where Query/Key come from the noisy latents and Value comes from the sketch features) is introduced to propagate the keyframe sketch conditions to all frames.

## Method

### Overall Architecture

The inputs consist of a text prompt and 1-2 keyframe sketches (which can be assigned to arbitrary timestamps), and the output is a video geometrically aligned with the sketches. The overall pipeline is divided into two branches: (1) the base video generation network CogVideoX-2b (comprising 30 DiT blocks) and (2) the sketch conditioning network (comprising 5 sketch control blocks). The sketches are first encoded into latent space representations via a VAE, and then processed through patchify and time-aware positional encoding to obtain sketch latents. The 5 sketch control blocks predict residual features and inject them into their corresponding DiT blocks. For the editing mode, a video insertion module and a latent blending strategy are additionally introduced.

### Key Designs

1. **Skip Residual Structure**:

    - **Function**: Achieves multi-level injection of sketch control signals with minimal parameter overhead.
    - **Mechanism**: Unlike PIXART-$\delta$ which uses a continuous chunk of blocks in the first half as an encoder, this work argues that DiT blocks at different depths process features of different granularities. Hence, only 5 sketch control blocks are employed, uniformly placed to predict residual features at the 0th, 6th, 12th, 18th, and 24th DiT block positions. This requires only $5/30 = 1/6$ of the extra parameters, significantly reducing GPU memory consumption.
    - **Design Motivation**: The traditional ControlNet implementation (replicating half of the pre-trained model) cannot be directly applied to the DiT architecture for video generation due to massive parameters causing OOM. Meanwhile, the skip structure successfully covers multiple feature hierarchies from shallow to deep layers.

2. **Inter-frame Attention Mechanism**:

    - **Function**: Propagates the sketch conditions from 1-2 keyframes to all of the frames in the video.
    - **Mechanism**: Unlike traditional cross-attention (where both Key and Value originate from the conditioning signals), the Query in this design originates from the noisy latents of all frames $h_i^{1:N}$, the Key comes from the noisy latents of the keyframes $h_i^{t_1,t_2}$, and the Value is derived from the keyframe sketch features $c_i^{t_1,t_2}$ processed by the DiT block. Consequently, the attention weights are computed based on inter-frame similarity, whereas the propagated content is the sketch feature information. Within each sketch control block, the sketch features are first updated via a FeedForward network, processed by a trainable DiT block copy (while ignoring white placeholders), and finally propagated to all frames via the inter-frame attention.
    - **Design Motivation**: Treating missing frames directly with white placeholders (such as in SparseCtrl) forces the network to process inputs of entirely different realities simultaneously, resulting in poor training performance. Leveraging the similarity of inter-frame latent features to determine propagation weights represents a more natural and elegant solution.

3. **Video Insertion Module**:

    - **Function**: Maintains spatio-temporal consistency between the newly edited content and the original video in editing scenarios.
    - **Mechanism**: For editing tasks, an additional trainable copy of the DiT block is introduced to process the original video latents masked outside the editing area. Then, the output of the sketch branch $\tilde{c}_i^{1:N}$ and the output of the video branch $\tilde{v}_i^{1:N}$ are concatenated according to the mask: $\text{Concat}(\tilde{c}_i^{1:N} * M^{1:N}, \tilde{v}_i^{1:N} * \bar{M}^{1:N})$, which is then fed into a FeedForward network to generate the final residual features.
    - **Design Motivation**: Pure sketch-based generation networks lack perception of the original video context, making it difficult to guarantee consistency between edited and unedited regions.

### Loss & Training

- **Two-stage Training**: The first stage mixes image and video data during training to accelerate convergence and compensate for the scarcity of video data; the second stage only leverages video data to enhance temporal consistency.
- **Editing Network Fine-tuning**: Initialized from the pre-trained weights of the generation network, the model incorporates the video insertion module. Since the pre-trained model already possesses strong sketch fidelity, it only needs to learn to parse video context during this phase. Training is conducted via self-supervised inpainting with random masks.
- **Inference Latent Blending**: For editing tasks, at steps 25 and 49 (out of 50 total steps), the unedited regions are replaced with the original video latents obtained via DDIM inversion, precisely preserving the details of the original video.
- The training loss inherits the diffusion objective of CogVideoX (v-prediction + zero SNR).

## Key Experimental Results

### Main Results

| Method | LPIPS ↓ | CLIP ↑ | Fidelity ↓ | Consistency ↓ | Realism ↓ |
|------|---------|--------|-----------|--------------|----------|
| AMT | 29.17 | 96.12 | 3.13 | 3.51 | 3.57 |
| SparseCtrl | 44.85 | 96.48 | 2.79 | 2.94 | 2.83 |
| Ctrl-CogVideo | 32.23 | 98.04 | 2.86 | 2.47 | 2.50 |
| **Ours** | **27.56** | **98.31** | **1.21** | **1.08** | **1.11** |

Editing comparisons (LPIPS/CLIP $\times 100$, PSNR evaluates preservation of unedited areas):

| Method | LPIPS ↓ | CLIP ↑ | PSNR ↑ | Fidelity ↓ | Preservation ↓ | Realism ↓ |
|------|---------|--------|--------|-----------|----------------|----------|
| InsV2V | 13.61 | 95.39 | 16.84 | 2.58 | 2.26 | 2.61 |
| AnyV2V | 11.92 | 93.47 | 13.68 | 2.35 | 2.69 | 2.34 |
| **Ours** | **9.74** | **98.34** | **36.48** | **1.07** | **1.05** | **1.04** |

### Ablation Study

| Variant | LPIPS ↓ | CLIP ↑ |
|------|---------|--------|
| w/o Inter-frame Attention | 36.33 | 98.10 |
| Using traditional Sketch K,V cross-attn | 32.59 | 98.19 |
| w/o Skip Structure (first 5 contiguous blocks) | 31.91 | 97.60 |
| w/o Image Data Training | 34.58 | 98.24 |
| **Full Model** | **30.79** | **98.48** |

### Key Findings

- Inter-frame attention has the most significant impact on sketch fidelity (removing it increases LPIPS from 30.79 to 36.33).
- The skip structure is crucial for video temporal consistency (removing it drops CLIP to 97.60).
- Mixing image training data substantially improves geometric matching accuracy.
- For editing tasks, pre-training the generator is key to maintaining sketch fidelity, while latent blending is crucial for preserving unedited areas (removing it drops PSNR from 36.48 to 31.69).
- The proposed method ranks first across all evaluation dimensions in the user study.

## Highlights & Insights

- **Minimalist and Efficient Design**: Leveraging only 5 control blocks (1/6 of the original model) achieves effective control, breaking the conventional assumption that "half of the model must be replicated".
- **Elegant Inter-frame Attention Design**: Query/Key obtained from original noisy features model the inter-frame relationships, while Value derived from sketch features handles condition injection, gracefully decoupling the two.
- **Unified Framework**: Both generation and editing share the same set of sketch control blocks; editing tasks only require the addition of the video insertion module.
- Sketches can be specified at arbitrary timestamps, not limited to the start or end frames, allowing for motion interpolation and extrapolation.

## Limitations & Future Work

- Generation capability is bounded by the quality ceiling of the pre-trained text-to-video model (CogVideoX-2b).
- Performance on complex scenes (e.g., human hands, multi-object interactions) remains suboptimal.
- The project currently focuses purely on geometric control; exploring appearance customization such as color brushstrokes is a future pathway.
- Currently supports only short video segments of around 6 seconds; long video generation remains to be explored.
- Incorporating 3D priors (e.g., SMPL-X) might improve results in human-centric scenes.

## Related Work & Insights

- **ControlNet Series**: This work represents an innovative adaptation of ControlNet principles to DiT video architectures, with the skip structure and inter-frame propagation serving as the core innovations.
- **SparseCtrl**: The most direct baseline; its white placeholder scheme yields underwhelming performance on DiT.
- **Conditional Control in DiT Architectures**: PIXART-$\delta$'s design of utilizing the first half of the blocks is unsuited for video models, whereas uniform skipping serves as a superior alternative.
- **Insight**: When designing conditional control for DiT-like architectures, researchers do not need to be constrained by the "first-half encoder" paradigm and can strategically design injection points based on target feature levels.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 8 |
| Overall Rating | 8.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)

</div>

<!-- RELATED:END -->
