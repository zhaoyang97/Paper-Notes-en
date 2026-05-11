---
title: >-
  [Paper Note] YOEO: You Only Erase Once - Erasing Anything without Bringing Unexpected Content
description: >-
  [CVPR 2026][Image Generation][Object Erasure] YOEO proposes a single-pass erasure framework that distills a multi-step diffusion model into a few-step model for efficient inference. It introduces a Sundries Suppression L…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Object Erasure"
  - "Diffusion Distillation"
  - "Hallucination Suppression"
  - "Entity Consistency"
  - "Unpaired Training"
date: 2026-05-08
content_hash: d173ccbbe03a2c89
---

# YOEO: You Only Erase Once - Erasing Anything without Bringing Unexpected Content

**Conference**: CVPR 2026
**arXiv**: [2603.27599](https://arxiv.org/abs/2603.27599)
**Code**: [https://zyxunh.github.io/YOEO-ProjectPage/](https://zyxunh.github.io/YOEO-ProjectPage/)
**Area**: Image Generation / Image Editing
**Keywords**: Object Erasure, Diffusion Distillation, Hallucination Suppression, Entity Consistency, Unpaired Training

## TL;DR

YOEO proposes a single-pass erasure framework that distills a multi-step diffusion model into a few-step model for efficient inference. It introduces a Sundries Suppression Loss (which detects newly generated spurious objects via entity segmentation) and an Entity Feature Coherence Loss (which ensures semantic consistency between the erased region and its surroundings), addressing the hallucination problem of diffusion models in object erasure.

## Background & Motivation

Diffusion models excel at image inpainting, yet when applied to object erasure they tend to "add what should not be there"—generating new objects in the masked region after removing the target. Existing closed-source solutions (ChatGPT, Nano Banana) achieve strong results but incur heavy computational costs, making them unsuitable for edge deployment.

**Two root causes**: (1) Lack of real erasure data—synthetic paired data (random masking + original image as GT) fails to represent genuine erasure scenarios; (2) SFT only teaches the model to "denoise" rather than to "erase"—pixel-level reconstruction losses impose no constraint against generating new objects.

## Method

### Overall Architecture

A pretrained erasure diffusion model serves as the teacher and is distilled into a few-step student model. Training uses two data streams: paired data $\mathcal{D}_1$ (random background masking with the original image as GT) and unpaired data $\mathcal{D}_2$ (target object masking without GT). The Sundries Suppression Loss and Entity Feature Coherence Loss are applied on top of the distillation objective.

### Key Designs

1. **Erasure Diffusion Distillation**:

    - Function: Compresses a multi-step diffusion model into a few-step model while preserving erasure quality.
    - Mechanism: Employs a DMD2 + GAN distillation framework. The few-step model produces sharp outputs at early denoising steps (rather than blurry ones), which makes subsequent sundries detection and coherence evaluation feasible—early outputs of multi-step models are too blurry for meaningful assessment.
    - Design Motivation: Distillation not only improves inference efficiency but, more critically, enables end-to-end supervision on unpaired data.

2. **Sundries Suppression Loss**:

    - Function: Detects and suppresses spurious objects generated in the masked region after erasure.
    - Mechanism: A pretrained entity segmentation model segments the erased output; the Intersection over Segment (IoS) between each detected entity and the inpainting mask is computed. Entities whose IoS exceeds threshold $\lambda$ are classified as newly generated "sundries," and a loss is constructed to penalize the model for generating them.
    - Design Motivation: Conventional pixel-level losses carry no knowledge of "what should not appear." Using an entity segmentation model as an automatic detector is equivalent to injecting the prior that "no independent entity should exist within the erased region."

3. **Entity Feature Coherence Loss**:

    - Function: Ensures semantic consistency between the erased region and its surrounding context.
    - Mechanism: Features are extracted from a pretrained segmentation network; cosine similarity is computed between the generated content inside the mask and the original content outside the mask. If the generated region is semantically consistent with its surroundings, its features should cluster around the same representational centroid.
    - Design Motivation: Even without generating sundries, a fill that is stylistically or semantically inconsistent with the surrounding context constitutes a failed erasure.

### Loss & Training

Total loss = LPIPS distillation loss + DMD loss + GAN loss + Sundries Suppression Loss + Entity Feature Coherence Loss. Paired and unpaired data are used in alternating training.

## Key Experimental Results

### Main Results

| Method | Erasure Cleanliness | Semantic Consistency | Inference Speed | Notes |
|--------|--------------------|--------------------|----------------|-------|
| SmartEraser | Low | Low | Slow | Prone to generating sundries |
| ASUKA | Medium | Medium | Slow | MAE + diffusion |
| **YOEO** | **High** | **High** | **Fast (few-step)** | Single-pass clean erasure |

YOEO outperforms existing methods comprehensively on both quantitative and qualitative metrics.

### Ablation Study

| Configuration | Sundries Rate↓ | Consistency↑ | Notes |
|--------------|---------------|-------------|-------|
| Distillation only | High | Low | Same hallucinations as teacher |
| + Sundries Suppression Loss | Significantly reduced | Low | Effectively reduces sundries |
| + Entity Coherence Loss | Significantly reduced | High | Semantic consistency improved |
| Full YOEO | Lowest | Highest | Two losses are complementary |

### Key Findings

- Distilling to a few-step model is a prerequisite for enabling unpaired supervision—intermediate states of multi-step diffusion are too blurry for meaningful evaluation.
- The Sundries Suppression Loss contributes most, indicating that "not generating new objects" is the most critical constraint in erasure tasks.
- Entity Feature Coherence provides a sense of "harmony," preventing the filled region from appearing inconsistent with its surroundings.

## Highlights & Insights

- **Paradigm shift from "denoising" to "erasing"**: Conventional pixel-level reconstruction losses only teach the model to "restore images." YOEO explicitly teaches the model "what not to do" through sundries detection and coherence constraints.
- **Unexpected value of distillation**: Distillation not only accelerates inference but also enables end-to-end unpaired training that was previously impossible—an insight transferable to other generative tasks requiring end-to-end evaluation.
- **Entity segmentation as a universal evaluator**: Using a pretrained segmentation model to automatically detect "things that should not appear" is more robust and general than manually designed rules.

## Limitations & Future Work

- Dependent on the quality of the entity segmentation model—missed or false detections degrade loss accuracy.
- Single-pass erasure may be insufficient for very large masked regions.
- Few-step distillation may sacrifice some generation detail.
- Future work could explore object erasure in video (temporal consistency).

## Related Work & Insights

- **vs. SmartEraser**: SmartEraser relies on synthetic paired data and explicit target prompts; YOEO requires neither paired data nor explicit prompts.
- **vs. ASUKA**: ASUKA reduces hallucinations via MAE + diffusion; YOEO addresses this more directly through the Sundries Suppression Loss.
- **vs. TurboFill**: TurboFill focuses on efficient diffusion inpainting but lacks erasure-specific constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ The Sundries Suppression Loss and the idea of enabling unpaired training via distillation are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons are comprehensive and qualitative results are convincing.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough.
- Value: ⭐⭐⭐⭐ High practical value for real-world image editing applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)
- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)

</div>

<!-- RELATED:END -->
