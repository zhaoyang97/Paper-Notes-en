---
title: >-
  [Paper Note] Pathways on the Image Manifold: Image Editing via Video Generation
description: >-
  [CVPR 2025][Video Generation][Image Editing] Frame2Frame (F2F) reformulates image editing as a video generation task. It leverages an image-to-video model to generate a smooth temporal pathway on the image manifold from the source image to the target edit. By using a VLM to generate temporal editing captions and automatically select frames, F2F achieves a SOTA balance between editing precision and image fidelity.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Image Editing"
  - "Image Manifold"
  - "CogVideoX"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: ab6acb0872dbc8af
---

# Pathways on the Image Manifold: Image Editing via Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.16819](https://arxiv.org/abs/2411.16819)  
**Code**: None (available on the project page)  
**Area**: Video Generation  
**Keywords**: Image Editing, Video Generation, Image Manifold, CogVideoX, Temporal Consistency

## TL;DR

Frame2Frame (F2F) reformulates image editing as a video generation task. It leverages an image-to-video model to generate a smooth temporal pathway on the image manifold from the source image to the target edit. By using a VLM to generate temporal editing captions and automatically select frames, F2F achieves a SOTA balance between editing precision and image fidelity.

## Background & Motivation

**Background**: Text-guided image editing primarily relies on diffusion models, conditioning the generation process through latent space inversion or model fine-tuning of the source image. Representative methods include SDEdit (noise-adding and denoising), Imagic (fine-tuning + interpolation), InstructPix2Pix (supervised training), and LEDITS++ (training-free inversion-based editing). Concurrently, video generation models (such as Stable Video Diffusion, CogVideoX, and Sora) have evolved into outstanding "world simulators" with strong temporal consistency and physical understanding capabilities.

**Limitations of Prior Work**: Existing image editing methods face a trade-off between two core challenges: (1) Editing precision—complex editing instructions are difficult to execute accurately; (2) Fidelity—the editing process often accidentally modifies crucial, unchanging elements of the source image (such as text or background objects). A fundamental conflict exists between the two: the model must simultaneously achieve precise editing and content preservation within a single generation step.

**Key Challenge**: Traditional editing methods start from initial noise and project to a target point on the image manifold, requiring this single point to satisfy both source image fidelity and the editing request. This "jumping" transition easily deviates from the image manifold, landing in regions that satisfy the edit but destroy fidelity.

**Goal**: Leverage the temporal consistency of video generation models to transform image editing from "single-point jumping" to a "continuous path on the manifold," allowing the editing process to undergo physically plausible intermediate states, thereby achieving a better balance between fidelity and editing precision.

**Key Insight**: Observed from a geometric perspective—video generation models learn the continuous structure of the image manifold during training. The source image naturally anchors on the manifold as the first video frame, and subsequent frames evolve smoothly along the manifold, with each frame being a plausible image until the editing goal is reached.

**Core Idea**: Redefine image editing as "generating a video starting from the source image, where the final frame serves as the editing result." This leverages the video model's inherent temporal consistency to ensure that key attributes of the source image are preserved throughout the editing process.

## Method

### Overall Architecture

Frame2Frame (F2F) consists of three steps: (1) **Temporal editing caption generation**—using a VLM (GPT-4o) to convert the source image and editing instructions into video descriptions that depict temporal evolution (temporal editing caption); (2) **Video generation**—using CogVideoX (I2V-5B) to generate a video sequence with the source image as the first frame and the temporal description as the condition; (3) **Frame selection**—utilizing a VLM to automatically select the optimal frame that achieves the editing goal from the generated video.

### Key Designs

1. **Temporal Editing Captions**:

    - **Function**: Convert static editing instructions into video scene descriptions that depict the process of temporal evolution.
    - **Mechanism**: Leverage GPT-4o as a VLM, take the source image $I_s$ and target editing description $c$ as input, and guide the model through in-context learning (providing 9 exemplar prompt-caption pairs) to generate descriptive, concise video scenes $\tilde{c}$. These emphasize how elements change or move over time while maintaining a static camera viewpoint. For example, converting "a person making a heart gesture" into "a person very slowly raising both hands to form a heart shape."
    - **Design Motivation**: Directly using the editing description as the text condition for the I2V model leads to uncontrollable generation—video models require "process descriptions" rather than "state descriptions." Temporal editing captions bridge the semantic gap between image editing and video generation.

2. **Video-based Editing**:

    - **Function**: Generate a continuous pathway on the image manifold from the source image to the target edit.
    - **Mechanism**: Utilize CogVideoX (I2V-5B), a Transformer-based video latent diffusion model, whose 3D VAE compresses the video in both spatial and temporal dimensions. The source image $I_s$ is encoded and concatenated with noise in the latent space, and the denoising process is guided by the temporal description $\tilde{c}$. AdaptiveLayerNorm achieves a deep fusion of visual and textual modalities. The model generates $T=49$ frames, approximately 6 seconds in duration (8fps). During image preprocessing, 1:1 images are resized to 480×480 and then padded with 120-pixel black borders on both the left and right sides to reach 720×480.
    - **Design Motivation**: Video models trained on large-scale internet data acquire an understanding of the physical world's dynamics. This "world knowledge" ensures that the generated intermediate frames are physically realistic—e.g., a person transition from standing to raising hands will not suddenly change clothes or background.

3. **Frame Selection**:

    - **Function**: Automatically identify the frame that represents the best editing result from the video sequence.
    - **Mechanism**: Sample every 4 frames, arrange each labeled frame into an image collage, and feed it to GPT-4o along with the source image $I_s$ and editing instruction $c$, instructing it to select the earliest frame $f_{t^*}$ that completes the edit. The "earliest" frame is chosen because subsequent frames tend to drift further away from the source image.
    - **Design Motivation**: Different edits require varying numbers of frames—simple modifications may take only a few frames to complete, while complex transformations require more. Fixedly selecting the last frame would lead to excessive deviation from the source image. Automatic frame selection also serves as a flexible interface for user interaction.

### Loss & Training

- **Training-free**: F2F involves no model training or fine-tuning, fully utilizing pretrained CogVideoX and GPT-4o.
- Video generation uses default hyperparameters: guidance scale = 6, 49 frames, 50 denoising steps.
- During evaluation, each method generates results with 15 random seeds per source image, and the best is manually selected.

## Key Experimental Results

### Main Results (TEdBench)

| Method | LPIPS↓ (Fidelity) | CLIP-I↑ (Fidelity) | CLIP↑ (Edit Accuracy) |
|------|:---:|:---:|:---:|
| SDEdit | 0.30 | 0.85 | 0.60 |
| Pix2Pix-ZERO | 0.29 | 0.84 | 0.62 |
| Imagic | 0.52 | 0.86 | 0.63 |
| LEDITS++ | 0.23 | 0.87 | 0.63 |
| FlowEdit | 0.22 | 0.89 | 0.61 |
| **F2F** | **0.22** | **0.89** | **0.63** |

### PosEdit (Human Pose Editing)

| Method | Source LPIPS↓ | Source CLIP-I↑ | Target LPIPS↓ | Target CLIP-I↑ | CLIP↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| SDEdit | 0.39 | 0.61 | 0.39 | 0.64 | 0.57 |
| LEDITS++ | 0.26 | 0.65 | 0.28 | 0.69 | 0.64 |
| **F2F** | **0.14** | **0.82** | **0.15** | **0.84** | **0.64** |
| GT (Ref) | 0.08 | 0.91 | 0 | 1.0 | 0.61 |

### Human Evaluation (TEdBench, F2F vs LEDITS++)

| Metric | F2F | LEDITS++ |
|------|:---:|:---:|
| Edit Accuracy (Overall) | **54.1%** | 45.9% |
| Edit Quality (Overall) | **65.6%** | 34.4% |

### Key Findings

- F2F achieves SOTA or comparable performance across both fidelity and editing accuracy metrics on TEdBench.
- On PosEdit, F2F's Source LPIPS (0.14) is significantly lower than that of LEDITS++ (0.26), demonstrating the immense advantage of the video path approach in preserving identity features.
- Human evaluation shows that F2F's lead in edit quality (fidelity) is much more pronounced than what automatic metrics reflect (65.6% vs 34.4%).
- F2F can be directly applied to traditional vision tasks such as denoising, deblurring, outpainting, and relighting, as these operations naturally correspond to common video scenarios (e.g., focusing, camera movement, time-lapse photography).
- Image manifold visualization experiments intuitively demonstrate the advantages of the video path approach—moving smoothly along the manifold preserves details like "AI" t-shirt text, whereas jump-based editing loses them.

## Highlights & Insights

- Redefining image editing as video generation is a true paradigm shift—leveraging the video model's world knowledge to guarantee the physical plausibility of editing.
- The image manifold visualization analysis (PCA projection + cluster visualization) excellently explains "why editing along a path is superior to jump-based editing."
- The method is extremely simple—requiring no training, no inversion, utilizing a three-step pipeline, which endows this method with high practicality and scalability.
- The PosEdit dataset (58 editing tasks, containing ground truth) serves as a valuable supplementary benchmark.

## Limitations & Future Work

- Unintended camera motion may occur during video generation, leading to pose/perspective shift in the edited results.
- Computational overhead is high—generating a 49-frame video is much slower than direct image editing (though video generation speeds are accelerating rapidly, such as LTX-Video completing in seconds).
- Heavily relies on two closed-source/large models, CogVideoX and GPT-4o, resulting in high end-to-end costs.
- Editing style is limited by the training data distribution of the video model—training data dominated by real-world transitions might struggle to handle "magical" editing.
- Future directions: fine-tuning video generators specifically for image editing, reducing the frame count to speed up generation, and customizing frame selection strategies.

## Related Work & Insights

- The fundamental difference compared to traditional editing methods (SDEdit, Imagic, InstructPix2Pix) lies in changing "single-step projection" into a "multi-step pathway."
- Difference from AnyDoor and MagicFixup: The latter construct training data pairs from video samples, whereas F2F directly performs editing via video generation.
- Similar strategies of "leveraging video models to understand the 3D/physical world" also appear in works like Make-A-Video3D and PhysDreamer.

## Rating

- **Novelty**: 9/10 — Redefining image editing as video generation is a genuine paradigm innovation; the manifold path perspective is highly insightful.
- **Experimental Thoroughness**: 7/10 — The evaluation on TEdBench + PosEdit + human study is relatively comprehensive, but lacks comparisons with more recent methods, and traditional vision tasks are only shown qualitatively.
- **Writing Quality**: 9/10 — The writing is smooth, and the manifold visualization analysis is highly convincing.
- **Value**: 8/10 — The paradigm is highly inspiring, and the method is simple yet practical; its value will continue to increase with the progress of video generation technology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)

</div>

<!-- RELATED:END -->
