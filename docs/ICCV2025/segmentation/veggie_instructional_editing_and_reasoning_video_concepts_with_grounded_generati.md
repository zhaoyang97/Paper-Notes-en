---
title: >-
  [Paper Note] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation
description: >-
  [ICCV 2025][Segmentation][Video Editing] VEGGIE proposes an end-to-end unified framework that bridges an MLLM with a video diffusion model, enabling a single model to simultaneously accomplish 8 tasks—including instructional video editing, concept grounding, and reasoning segmentation—using only the diffusion loss.
tags:
  - ICCV 2025
  - Segmentation
  - Video Editing
  - Instructional Editing
  - Video Concept Grounding
  - Reasoning Segmentation
  - Unified Multi-Task Model
date: 2026-05-08
content_hash: 2a2bab26b0c82e5c
---

# VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation

**Conference**: ICCV 2025
**arXiv**: [2503.14350](https://arxiv.org/abs/2503.14350)
**Code**: [https://veggie-gen.github.io/](https://veggie-gen.github.io/) (project page)
**Area**: Semantic Segmentation
**Keywords**: Video Editing, Instructional Editing, Video Concept Grounding, Reasoning Segmentation, Unified Multi-Task Model

## TL;DR

VEGGIE proposes an end-to-end unified framework that bridges an MLLM with a video diffusion model, enabling a single model to simultaneously accomplish 8 tasks—including instructional video editing, concept grounding, and reasoning segmentation—using only the diffusion loss.

## Background & Motivation

While video editing methods have advanced considerably in recent years, three core challenges remain before reaching the goal of a simple, general-purpose video concept editor:

**Non-end-to-end**: Most methods rely on intermediate steps—layouts, masks, manually annotated or model-generated descriptions as guidance (e.g., VPLM requires mask inputs; tool-use methods require multi-step pipelines)—imposing additional user burden and disrupting seamless editing experiences.

**Complex training objectives**: Existing pipelines connecting MLLMs and video diffusion models (VidDMs) require multiple training objectives (language loss, mask loss, etc.), increasing optimization difficulty and hyperparameter tuning costs.

**Incomplete task coverage**: Existing models excel at only a subset of editing tasks. For instance, LGVI excels at removal but does not support stylization; VidToMe handles global editing well but performs poorly on local edits; TokenFlow does not support object addition or deletion.

The root causes of these challenges are twofold: (1) a lack of high-quality multi-task video editing training data covering a broad range of skills; and (2) models lacking two critical capabilities—**multimodal reasoning** (inferring editing intent from instructions) and **language grounding** (precisely localizing regions to be edited).

The core idea of VEGGIE is to unify video grounding and editing into a **pixel-space end-to-end generative task**. No additional detection or segmentation modules are required, no intermediate text tokens serve as conditions, and both editing, grounding, and reasoning are learned jointly using only the diffusion loss. The key innovation is replacing discrete text tokens with **continuous learnable grounded task query embeddings** as the bridge from the MLLM to the VidDM, enabling end-to-end gradient propagation.

## Method

### Overall Architecture

VEGGIE comprises four components:

1. **MLLM**: Receives a sequence of video frames $V = [f_1, ..., f_n]$ and a user instruction $I$, and generates per-frame grounded task tokens $C = [c_1, ..., c_n]$.
2. **Learnable Grounded Task Queries**: Continuous embedding vectors, one set per frame, processed in parallel as both input and output of the MLLM.
3. **Alignment Network**: A single-layer MLP that maps MLLM outputs to the diffusion model's conditioning space.
4. **Video Diffusion Model**: Initialized from an instructional image editing model; receives the original video (concatenated to the noise) and task tokens (fed via cross-attention) to generate the edited video.

The key distinction from prior methods is that task queries are continuous and differentiable, allowing gradients to backpropagate from the VidDM into the MLLM for truly end-to-end training. Methods such as VPLM use discrete text tokens, which cut off the gradient flow.

### Key Designs

1. **Curriculum Learning: Two-Stage Training**

    - **Function**: Aligns the MLLM and VidDM at the image level before end-to-end fine-tuning at the video level.
    - **Stage 1 (Image–Language Space Alignment)**: The MLLM is frozen; the alignment network, task queries, and diffusion UNet are updated. Trained on 3.4 million image editing samples, enabling the diffusion model to understand MLLM-generated task guidance.
    - **Stage 2 (Video Temporal Enhancement)**: All components are unfrozen (including the MLLM); the 2D UNet is inflated into a 3D model via temporal attention layers. End-to-end fine-tuning is performed on 136,000 video editing samples.
    - **Design Motivation**: Directly training end-to-end on video data causes model collapse due to misalignment between the MLLM and VidDM representation spaces. Pre-alignment with large-scale image data is essential.

2. **Unified Task Formulation**

    - **Function**: Unifies editing, grounding, and segmentation as video-to-video generation tasks.
    - **Mechanism**:
        - Video editing: output is the edited video frames.
        - Video concept grounding: output is video frames with target objects highlighted (color-filled).
        - Reasoning segmentation: output is video frames with segmentation masks visualized.
        - All tasks share a single pixel-space output format, requiring only the diffusion loss.
    - **Design Motivation**: Eliminates the need to design separate heads and loss functions for different tasks, greatly simplifying the training pipeline.

3. **Data Synthesis Pipeline (VEG-Edit)**

    - **Function**: Lifts high-quality image editing data to video editing data.
    - **Core Pipeline**:
        - Input: original image $I$, edited image $\bar{I}$, editing instruction.
        - Step 1: MLLM generates image descriptions and animation prompts.
        - Step 2: An Image-to-Video model animates $I$ into video $V$.
        - Step 3: A first-frame-conditioned video editing model uses $\bar{I}$ to generate the edited video $\bar{V}$.
        - Step 4: Automatic video quality assessment filters low-quality samples.
    - **Design Motivation**: High-quality annotated video editing data is extremely scarce, while image editing data is abundant (e.g., MagicBrush and Seed-Data-Edit together provide 3.4 million samples). I2V generation combined with quality filtering offers an efficient path to scaling video training data.

### Loss & Training

- Only the **diffusion loss** is used throughout training; no language loss or mask loss is introduced.
- **Classifier-Free Guidance**: At inference, CFG is applied separately to the task token condition and the input video condition:
    $$\tilde{e_\theta}(z_t, c_T, c_V) = e_\theta(z_t, \varnothing, \varnothing) + g_T \cdot (e_\theta(z_t, c_V, c_T) - e_\theta(z_t, c_V, \varnothing)) + g_V \cdot (e_\theta(z_t, c_V, \varnothing) - e_\theta(z_t, \varnothing, \varnothing))$$
- **Data**: Stage 1 uses 3.4 million image pairs (Seed-Data-Edit: 3 million + segmentation/reasoning data); Stage 2 uses 136,000 video pairs.

## Key Experimental Results

### Main Results

Performance of VEGGIE vs. baselines on VEG-Bench across 8 editing skills (selected key skills):

| Capability | Metric | VidToMe | TokenFlow | InsV2V | LGVI | **VEGGIE** |
|---|---|---|---|---|---|---|
| Concept Addition | MLLM-Judge↑ | 5.00 | 5.80 | 5.69 | 2.73 | **7.44** |
| Concept Addition | Detection↑ | 47.98 | 49.53 | 48.01 | 14.42 | **57.96** |
| Concept Removal | MLLM-Judge↑ | 2.60 | 3.73 | 2.78 | 6.59 | 5.07 |
| Concept Removal | Detection↑ | 34.31 | 55.16 | 25.64 | **78.40** | 70.22 |
| Object Replacement | MLLM-Judge↑ | 5.00 | 6.53 | 6.60 | 2.06 | **6.63** |
| Video Grounding | mIoU↑ | 0.00 | 0.00 | 0.00 | 0.00 | **47.30** |
| Reasoning Segmentation | mIoU↑ | 0.00 | 0.00 | 0.00 | 0.00 | **32.80** |

VEGGIE is the only method capable of accomplishing all 8 tasks simultaneously. All other baselines completely fail on grounding and reasoning segmentation tasks (mIoU = 0).

### Ablation Study

| Configuration | Add Judge↑ | Remove Detect↑ | Grounding mIoU↑ | Note |
|---|---|---|---|---|
| w/o video data (Stage 2) | 6.80 | 65.10 | 40.20 | Image pre-training only |
| w/o image data (Stage 1) | 5.20 | 55.30 | 35.10 | Direct training on video |
| w/o grounding data | 7.10 | 68.50 | 0.00 | Editing capability intact but no grounding |
| w/o editing data | 3.20 | 30.20 | 45.80 | Grounding intact but editing collapses |
| **VEGGIE (full)** | **7.44** | **70.22** | **47.30** | Multi-task mutual promotion |

### Key Findings

- **Multi-task mutual promotion**: Grounding data helps the editing model more precisely identify regions to edit; editing data helps the grounding model understand object semantics. Removing either data category degrades performance on both related and unrelated tasks.
- VEGGIE significantly outperforms the instructional baseline InsV2V on creative editing tasks such as addition, replacement, and stylization, while approaching the removal-specialized LGVI on the removal task.
- **Zero-shot emergent capabilities**: VEGGIE exhibits multimodal instruction-following abilities (e.g., adding objects or transferring styles from reference images) and few-shot in-context editing abilities (learning editing patterns from example pairs), neither of which was explicitly trained.

## Highlights & Insights

1. **Unifying multiple tasks with "diffusion loss only"** is an elegant design choice—it eliminates the need for multi-loss weight tuning and demonstrates that pixel-space generation can naturally subsume understanding tasks such as segmentation and grounding.
2. **Replacing discrete text tokens with continuous task queries** is the key design decision enabling end-to-end training, resolving the long-standing gradient disconnection problem in MLLM→VidDM pipelines.
3. The necessity of curriculum learning is clearly validated by ablation experiments: skipping Stage 1 and training directly on video data causes model collapse.
4. The data synthesis pipeline (image-to-video lifting) provides a scalable solution to the data scarcity problem in video editing.
5. VEG-Bench is the first unified evaluation benchmark covering 8 distinct editing skills.

## Limitations & Future Work

- The SD1.5-based UNet imposes an upper bound on video quality and resolution; future work may migrate to stronger base models such as SD3 or FLUX.
- Performance on the removal task still lags behind the removal-specialized LGVI (70.22 vs. 78.40 Detection), suggesting that unified models may face a performance ceiling on specialized subtasks.
- Temporal smoothness scores are uniformly high (>94) across all baselines, indicating limited discriminative power of this metric.
- The reasoning segmentation mIoU (32.8) remains considerably below image-domain methods, leaving video reasoning segmentation as an open problem.

## Related Work & Insights

- The instructional editing paradigm of InstructPix2Pix is extended to the video domain.
- The reasoning segmentation approach of LISA is integrated into the generative framework without requiring a dedicated segmentation head.
- The data synthesis strategy (high-quality image data → I2V → video data) offers broadly applicable reference for addressing data bottlenecks in the video domain.
- For other tasks combining MLLMs with generative models (e.g., 3D generation, audio editing), the "continuous query + curriculum learning" framework may be equally applicable.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)
- [\[ACL 2026\] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation](../../ACL2026/segmentation/anchorseg_language_grounded_query_banks_for_reasoning_segmentation.md)
- [\[ICLR 2026\] VINCIE: Unlocking In-context Image Editing from Video](../../ICLR2026/segmentation/vincie_unlocking_in-context_image_editing_from_video.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)

<!-- RELATED:END -->
