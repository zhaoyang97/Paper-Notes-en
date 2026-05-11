---
title: >-
  [Paper Note] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions
description: >-
  [NeurIPS 2025][Image Generation][Subject-driven video customization] OmniVCus proposes a feedforward DiT framework that achieves multi-subject, multimodal-controlled video customization through a data construction pipeli…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Subject-driven video customization"
  - "multimodal control"
  - "DiT"
  - "multi-subject generation"
  - "feedforward generation"
date: 2026-05-08
content_hash: 6fccfddc2b2f1046
---

# OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions

**Conference**: NeurIPS 2025
**arXiv**: [2506.23361](https://arxiv.org/abs/2506.23361)
**Code**: [https://caiyuanhao1998.github.io/project/OmniVCus/](https://caiyuanhao1998.github.io/project/OmniVCus/) (project page)
**Area**: Diffusion Models / Video Customization Generation
**Keywords**: Subject-driven video customization, multimodal control, DiT, multi-subject generation, feedforward generation

## TL;DR
OmniVCus proposes a feedforward DiT framework that achieves multi-subject, multimodal-controlled video customization through a data construction pipeline called VideoCus-Factory and two novel embedding mechanisms (Lottery Embedding and Temporally Aligned Embedding), significantly surpassing prior SOTA in identity preservation and controllability.

## Background & Motivation
Subject-driven video customization aims to generate videos containing specific identity subjects based on user-provided reference images. Existing feedforward methods face three major challenges:

**Difficulty in multi-subject data construction**: ConceptMaster supports only limited categories; Video Alchemist relies on scarce high-quality text–video pairs, resulting in limited data scale.

**Restricted number of subjects at inference time**: The number of subjects in training videos is limited (typically 1–2), and composing more subjects at inference time remains largely unexplored.

**Absence of multimodal control conditions**: Integrating control signals such as depth maps, segmentation masks, camera trajectories, and text editing instructions into subject-driven customization remains an open problem.

Core Idea: Construct large-scale multi-subject training data and design two specialized positional embedding mechanisms, enabling a single DiT model to flexibly compose control signals of different modalities.

## Method

### Overall Architecture
OmniVCus is built upon the DiT architecture. Text, images, video, and control signals are encoded via patchification and concatenated into a one-dimensional long token sequence as model input. The model supports mixed training across multiple tasks: single/dual-subject customization, depth/mask-to-video, text-to-multi-view, text-to-image/video, and image editing.

### Key Designs

1. **VideoCus-Factory Data Construction Pipeline**

    - **Video captioning**: A frame is randomly sampled from each video; Kosmos-2 generates descriptions and detects subjects, inserting image tags such as IMG1/IMG2.
    - **Subject filtering**: SAM-2 is used to track and segment subjects; samples with failed segmentation (e.g., severely occluded objects) are discarded.
    - **Data augmentation**: Segmented subjects undergo random rotation, scaling, centering, and color augmentation, and are placed on randomly selected backgrounds. This prevents leakage of subject position, size, and background during training, mitigating the copy-paste effect.
    - **Control signal generation**: Training data pairs for mask-to-video (subject mask sequences) and depth-to-video (depth sequences) are constructed simultaneously. These control data are not paired with subject customization data but can be flexibly combined at inference time.

2. **Lottery Embedding (LE)**

    - Core motivation: The number of subjects $K$ in training videos is limited, yet it is desirable to compose $M > K$ subjects at inference time.
    - Design: $K$ integers are uniformly sampled at random from $[1, M]$ to form a set $S$; after sorting, these are assigned as frame positional embeddings to the $K$ training subjects.
    - Effect: More frame positional embeddings are activated during training, enabling zero-shot composition of additional subjects at inference time (e.g., training with 2 subjects but composing 4 at inference).

3. **Temporally Aligned Embedding (TAE)**

    - Core motivation: Control signals such as depth and mask are temporally aligned with the generated video and should share the same temporal positional information.
    - Design: For dense semantic control signals (depth/mask), the 3D-VAE-encoded features share the same frame positional embeddings $\{M+1, \ldots, M+N\}$ as the noise tokens; timestep embeddings are added only to noise tokens to distinguish them. For sparse camera signals (Plücker coordinates), an MLP maps the signals and adds them directly to the noise tokens to reduce token length.

4. **Image-Video Transfer Mixed (IVTM) Training**

    - Motivation: Paired training data combining subject customization with editing instructions are unavailable.
    - Method: Image editing data are mixed with single-subject image/video customization data during training; by aligning frame positional embeddings, editing effects are transferred from images to video. At inference time, combining editing instructions with subject customization prompts activates the editing behavior.

### Loss & Training
- Flow-matching loss is used for joint training.
- Noisy input is constructed via linear interpolation: $X^t = tX^1 + (1-t)X^0$
- The model predicts the velocity field $V^t = X^1 - X^0$
- Training data across different tasks are unpaired; certain input conditions are naturally absent for different samples.
- Fine-tuned for 100K steps on a 5B-parameter T2V DiT with batch size 356 on 64 A100 GPUs.

## Key Experimental Results

### Main Results

| Method | # Subjects | CLIP-T | CLIP-I | DINO-I | Consistency | Dynamics |
|--------|------------|--------|--------|--------|-------------|----------|
| VideoBooth | Single | 0.2541 | 0.5891 | 0.3033 | 0.9593 | 0.4287 |
| DreamVideo | Single | 0.2799 | 0.6214 | 0.3792 | 0.9609 | 0.4696 |
| Wan2.1-I2V | Single | 0.2785 | 0.6319 | 0.4203 | 0.9754 | 0.5310 |
| SkyReels | Single | 0.2820 | 0.6609 | 0.4612 | 0.9797 | 0.5238 |
| **OmniVCus** | **Single** | **0.3293** | **0.7154** | **0.5215** | **0.9928** | **0.5541** |
| SkyReels | Multi | 0.2785 | 0.6429 | 0.4107 | 0.9710 | 0.5892 |
| **OmniVCus** | **Multi** | **0.3264** | **0.6672** | **0.4965** | **0.9908** | **0.6878** |

### Ablation Study

| Configuration | CLIP-T | DINO-I | Consistency | Dynamics |
|---------------|--------|--------|-------------|----------|
| Baseline (no filtering/augmentation) | 0.2175 | 0.2405 | 0.9588 | 0.3759 |
| + Subject filtering | 0.2431 | 0.5053 | 0.9617 | 0.3826 |
| + Data augmentation | 0.3293 | 0.5215 | 0.9928 | 0.5541 |

| TAE Ablation | CLIP-T | DINO-I | Consistency | Dynamics |
|--------------|--------|--------|-------------|----------|
| Naive embedding | 0.2618 | 0.2947 | 0.9751 | 0.4948 |
| Added to noise | 0.1722 | 0.1680 | 0.9319 | 0.5437 |
| **TAE** | **0.3054** | **0.3794** | **0.9909** | **0.4965** |

| LE Ablation (Multi-subject) | CLIP-T | DINO-I | Consistency | Dynamics |
|-----------------------------|--------|--------|-------------|----------|
| Without LE | 0.2105 | 0.3364 | 0.9702 | 0.6943 |
| **With LE** | **0.2728** | **0.4163** | **0.9810** | 0.6806 |

### Key Findings
- Random background placement in data augmentation substantially improves dynamic diversity in generated videos (dynamics score increases from 0.38 to 0.55).
- In TAE, directly adding depth signals to noise tokens causes model collapse, as fine-grained spatial information is corrupted by noise.
- LE enables the composition of 3–4 subjects at inference time, yielding a DINO-I gain of 0.08.
- The IVTM training strategy significantly outperforms naive mixed training (CLIP-T: 0.3126 vs. 0.2585).

## Highlights & Insights
- The VideoCus-Factory pipeline is elegantly designed, automatically constructing multi-subject training pairs and control signal data from unlabeled raw video.
- Lottery Embedding achieves generalization from few subjects at training time to multiple subjects at inference time in a simple manner, with the key insight of randomly activating more frame positional embeddings.
- TAE is motivated by the temporal alignment between control signals and video frames, and adopts different handling strategies for dense and sparse signals.
- A user study (37 participants) shows that OmniVCus substantially outperforms baselines in identity preservation, alignment, and quality.

## Limitations & Future Work
- The method relies on a large-scale internal video data pool, making reproduction costly (64 A100 GPUs, 300M text-to-image data).
- Detection and segmentation quality from Kosmos-2 and SAM-2 directly affects data quality.
- Handling interactions and occlusions between multiple subjects still has room for improvement.
- Instruction-based editing on hard cases (e.g., style transfer) remains limited.

## Related Work & Insights
- Compared to ConceptMaster and Video Alchemist, VideoCus-Factory is more general and operates at a larger data scale.
- The idea behind LE can be extended to other scenarios requiring capability generalization at inference time.
- The IVTM image-to-video transfer training strategy offers a useful reference for settings where paired data are scarce.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](../../ICCV2025/image_generation/freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)
- [\[NeurIPS 2025\] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](../../CVPR2026/image_generation/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)

</div>

<!-- RELATED:END -->
