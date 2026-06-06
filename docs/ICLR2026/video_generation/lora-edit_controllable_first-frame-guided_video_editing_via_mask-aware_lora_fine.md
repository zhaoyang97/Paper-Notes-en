---
title: >-
  [Paper Note] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning
description: >-
  [ICLR 2026][Video Generation][video editing] This paper proposes LoRA-Edit, which leverages spatiotemporal masks to guide LoRA fine-tuning of a pretrained I2V model…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "video editing"
  - "LoRA fine-tuning"
  - "first-frame guidance"
  - "spatiotemporal mask"
  - "appearance control"
date: 2026-05-08
content_hash: 53dc5324c8f8dd01
---

# LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning

**Conference**: ICLR 2026
**arXiv**: [2506.10082](https://arxiv.org/abs/2506.10082)  
**Code**: [Project Page](https://cjeen.github.io/LoRAEdit)  
**Area**: Video Editing
**Keywords**: video editing, LoRA fine-tuning, first-frame guidance, spatiotemporal mask, appearance control

## TL;DR
This paper proposes LoRA-Edit, which leverages spatiotemporal masks to guide LoRA fine-tuning of a pretrained I2V model, enabling controllable first-frame-guided video editing. The mask simultaneously serves as an instruction for the editing region and a guidance signal for LoRA learning, supporting motion inheritance and appearance control.

## Background & Motivation
- Large-scale pretrained methods for video editing are costly and inflexible; first-frame-guided editing offers a more adaptable alternative.
- Existing first-frame-guided methods (AnyV2V, I2VEdit) control only the first frame and cannot govern the temporal evolution of subsequent frames.
- Naive LoRA fine-tuning can capture motion but lacks fine-grained control—it cannot distinguish between regions to preserve and regions to modify.
- The mask conditioning mechanism built into I2V models harbors underexplored potential for spatial control.

## Method

### Overall Architecture
LoRA-Edit trains LoRA under two complementary mask configurations: motion learning (learning motion patterns from the foreground mask of the source video) and appearance learning (learning target appearance from reference frames), without modifying the model architecture.

### Key Designs

1. **Dual Role of the Mask**:

    - **As an instruction**: informs the model which regions to preserve (mask=1) and which to generate (mask=0), enhancing the model's responsiveness to mask signals.
    - **As a learning guide**: by masking different content, it directs LoRA to focus on either motion patterns or target appearance.
   Exploratory analysis shows that the original I2V model can handle simple full-frame instructions but fails at selective spatial editing (foreground masks)—LoRA fine-tuning is required to augment this capability.

2. **Decoupling Editing from Background (Motion Learning)**:

    - During training: the first frame is preserved with mask=1; subsequent frames use foreground/background masks—unedited regions=1, edited regions=0.
    - $\mathbf{V}_{\text{cond}}$ is constructed by applying the mask to the input video; $\mathbf{V}_{\text{target}}$ is the original video.
    - Under mask guidance, LoRA learns to preserve the background and generate content in the foreground region consistent with the source video's motion.

3. **Appearance Control (Appearance Learning)**:

    - When the edited region rotates, deforms, or follows its own motion trajectory, the first frame alone is insufficient to infer subsequent appearance.
    - Users are allowed to edit arbitrary subsequent frames as additional references.
    - During training, edited frames are used as $\mathbf{V}_{\text{target}}$; multiple edited frames are treated as independent static images to avoid erroneous temporal dynamics inference.

### Loss & Training
Modified flow matching objective:
$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\left[\|v_\theta(\mathbf{x}_t, t; \mathbf{V}_{\text{cond}}, \mathbf{M}_{\text{cond}}, [p^*]+c) - (\mathbf{x}_0 - \mathbf{x}_1)\|_2^2\right]$$
Based on the Wan2.1-I2V 480P model:
- Motion learning: 100-step LoRA training (LR=1e-4)
- Appearance learning: additional 100 steps
- 49 frames, 832×480 resolution, 20 GB GPU memory

## Key Experimental Results

### Main Results (Quantitative Comparison for First-Frame-Guided Editing)

| Method | CLIP Score↑ | DEQA Score↑ | Input Similarity↑ |
|--------|------------|------------|-------------------|
| AnyV2V | 0.8995 | 3.7348 | 0.7569 |
| Go-with-the-Flow | 0.9047 | 3.5622 | 0.7504 |
| I2VEdit | 0.9128 | 3.4480 | 0.7536 |
| **LoRA-Edit** | **0.9172** | **3.8013** | **0.7608** |

### User Study (Reference-Guided Editing Rankings, Lower is Better)

| Method | Motion Consistency↓ | Background Preservation↓ |
|--------|-------------------|------------------------|
| Kling1.6 | 1.869 | 1.806 |
| VACE (14B) | 2.511 | 2.460 |
| **LoRA-Edit** | **1.620** | **1.734** |

### Key Findings
- Outperforms existing first-frame-guided methods across all three quantitative metrics.
- Ranks first in both motion consistency and background preservation in the user study.
- Mask precision analysis: loose masks (bounding boxes) outperform tight masks (precise segmentation), as generated entities require spatial buffer for contour variation.
- High-quality editing is achievable by training a per-video LoRA for only 100–200 steps.
- Motion learning and appearance learning LoRA weights can be freely combined at inference time.

## Highlights & Insights
- Reveals that the mask conditioning mechanism of I2V models holds broader spatial control potential beyond first-frame preservation.
- The "dual role of the mask" is the central insight: it functions simultaneously as a model instruction and as a directional signal for LoRA learning.
- The finding that loose masks outperform tight masks is both interesting and practically useful—pixel-perfect precision is unnecessary.
- Reference frames are used only during training (not at inference), providing greater flexibility in appearance guidance.

## Limitations & Future Work
- Each video requires independent LoRA training (100–200 steps), precluding instant generation.
- Users must manually or semi-automatically provide masks and interact with the pipeline.
- Obtaining edited frames relies on external image editing tools.
- The method inherits biases from the pretrained I2V model.
- No comparison with large-scale trained video editing models on more complex scenarios.

## Related Work & Insights
- The first-frame-guided paradigms of AnyV2V and I2VEdit motivated this work.
- The motion–appearance decoupling idea from AnimateDiff finds a new realization within the mask-guided framework.
- VACE's globally trained approach may generalize less effectively out-of-domain compared to per-video LoRA.
- This work provides a lightweight and flexible solution for general video manipulation based on I2V models.

## Technical Details
- Based on the Wan2.1-I2V 480P model; HunyuanVideo-I2V is also validated.
- LoRA is inserted into self-attention and cross-attention layers.
- Florence-2 is used to automatically generate captions, augmented with a special token $p^*$.
- Training on 49-frame videos requires only 20 GB GPU memory.
- Reference frames are used exclusively during training and are not required at inference, offering greater flexibility.
- The automatic mask acquisition workflow is based on SAM2 and segmentation bounding boxes.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-role design of mask-guided LoRA is elegant, though individual components are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons, user study, and ablations, though the test scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; exploratory experiments on mask configurations have pedagogical value.
- Value: ⭐⭐⭐⭐ Provides a flexible, lightweight, and architecture-agnostic practical solution for video editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] First Frame Is the Place to Go for Video Content Customization](../../CVPR2026/video_generation/first_frame_is_the_place_to_go_for_video_content_customization.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[ICML 2026\] Exploring Data-Free LoRA Transferability for Video Diffusion Models](../../ICML2026/video_generation/exploring_data-free_lora_transferability_for_video_diffusion_models.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICML 2026\] MiVE: Multiscale Vision-language features for reference-guided video Editing](../../ICML2026/video_generation/mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)

</div>

<!-- RELATED:END -->
