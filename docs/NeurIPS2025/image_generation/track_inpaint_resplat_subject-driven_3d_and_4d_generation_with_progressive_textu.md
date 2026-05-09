---
title: >-
  [Paper Note] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling
description: >-
  [NeurIPS 2025][Image Generation][Subject-driven generation] This paper proposes TIRE (Track, Inpaint, REsplat), a three-stage pipeline that locates unobserved regions via video tracking, progressively infills textures using a subject-driven inpainting model, and back-projects multi-view consistent results into 3D, enabling identity-preserving 3D/4D generation.
tags:
  - NeurIPS 2025
  - Image Generation
  - Subject-driven generation
  - 3D/4D generation
  - Identity preservation
  - Texture infilling
  - Video tracking
date: 2026-05-08
content_hash: 8a3af282a4db64a0
---

# Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling

**Conference**: NeurIPS 2025
**arXiv**: [2510.23605](https://arxiv.org/abs/2510.23605)
**Code**: [Project Page](https://zsh2000.github.io/track-inpaint-resplat.github.io/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Subject-driven generation, 3D/4D generation, Identity preservation, Texture infilling, Video tracking

## TL;DR

This paper proposes TIRE (Track, Inpaint, REsplat), a three-stage pipeline that locates unobserved regions via video tracking, progressively infills textures using a subject-driven inpainting model, and back-projects multi-view consistent results into 3D, enabling identity-preserving 3D/4D generation.

## Background & Motivation

Existing 3D/4D generation methods (e.g., LGM, L4GM, TRELLIS, Hunyuan3D) primarily focus on realism, efficiency, and visual quality, but perform poorly in maintaining **semantic identity consistency** across multiple viewpoints. Given a single reference image, the generated 3D/4D assets often suffer from color shifts and texture inconsistencies on side and back views.

**Limitations of Prior Work**:

**SDS-based optimization**: Extremely time-consuming; appearance and motion tend to be averaged during optimization.

**Multi-view diffusion models**: Exhibit systematic color and appearance bias on novel views due to training data bias.

**Native 3D generation** (TRELLIS, Hunyuan3D-v2.5, etc.): Despite high efficiency, still fail to satisfactorily preserve the identity of the reference image.

**Key Challenge**: A fundamental tension exists between efficient 3D generation and identity preservation. Existing methods must hallucinate unseen regions in a single forward pass from limited input views, offering little precise control over subject identity.

**Key Insight**: Rather than improving 3D generation models directly, this work treats them as initialization and leverages powerful 2D video tracking and inpainting tools to progressively repair incorrect regions in 3D assets — an approach **orthogonal and complementary** to existing feed-forward 3D generation methods.

## Method

### Overall Architecture

TIRE takes the output of existing 3D/4D generation models (e.g., LGM, L4GM) as initialization, renders multi-view observations, and proceeds through three stages: Track (locating regions to repair) → Inpaint (progressive identity-preserving inpainting) → Resplat (back-projection into 3D).

### Key Designs

1. **Track — Backward Tracking for Inpainting Region Localization**:

    - Multi-view rendered frames are assembled into a video sequence ordered by camera motion.
    - The CoTracker video tracking model is used to establish correspondences between source and target views.
    - **Key innovation**: **Backward tracking** is adopted instead of forward tracking. Tracking from target views back to the source view maximizes the use of identity information, as the source view contains the richest subject appearance.
    - Forward tracking produces fragmented small inpainting regions with granular artifacts; backward tracking yields more accurate and inpainting-friendly masks.
    - This approach is agnostic to the underlying 3D representation, ensuring broad applicability.

2. **Inpaint — Progressive Identity-Preserving Texture Infilling**:

    - LoRA weights are injected into a pretrained Stable Diffusion inpainting model.
    - The loss is computed only within the valid foreground region: $\mathcal{L} = m_v \odot [\epsilon_\theta(x_t, t, p, m_i, (1-m_i) \odot x) - \epsilon]$
    - **Key design of the progressive strategy**:
        - Step 1: Train using only the original source-view image with data augmentation (flipping + small rotations up to 15°).
        - Step 2: Inpaint the "sweet-spot" views at $\pm 20°$ (balancing exploration and exploitation).
        - Step 3: Use $\pm 20°$ views as anchors and extend to $\pm 90°$ via backward tracking.
        - Step 4: Use $\pm 90°$ views as anchors and continue inpainting to $\pm 180°$.
    - Denoising is restricted to the first 30% of the schedule to avoid excessive structural changes.
    - **Design Motivation**: Views far from the source view differ substantially and are difficult to inpaint directly; progressive expansion ensures reliable context at each step.

3. **Resplat — Multi-View Consistent 3D Reconstruction**:

    - Independently inpainting each frame may introduce cross-view inconsistencies.
    - A multi-view diffusion model is introduced for consistency refinement via mask-aware latent updates: $z_{t-1} = \tilde{z}_{t-1} \odot M + \hat{z}_{t-1} \odot (1-M)$
    - Source-view latents are kept unchanged ($M=0$); only other views are updated ($M=1$), enforcing source-view identity.
    - The first 30% of the denoising schedule is likewise applied here.
    - Finally, LGM/L4GM back-projects the multi-view observations into 3D Gaussians.

### Loss & Training

- The training loss follows the standard inpainting objective, restricted to the valid foreground mask.
- LoRA is injected into the pretrained inpainting model for parameter-efficient adaptation.
- A fixed text prompt "A photo of sks" is used.
- No additional 3D data or large-scale fine-tuning is required.

## Key Experimental Results

### Main Results — DINO Identity Similarity (Video-to-4D)

| Method | DINO (ViT-S/16) ↑ | DINO (ViT-B/16) ↑ |
|--------|-------|-------|
| Customize-It-3D | 0.5773 | 0.6087 |
| SV4D | 0.5213 | 0.5426 |
| STAG4D | 0.5287 | 0.5592 |
| L4GM | 0.5506 | 0.5694 |
| **TIRE (Ours)** | **0.5665** | **0.5815** |

### VLM Multi-Dimensional Identity Preservation Evaluation (Image-to-3D)

| Method | GPT-4o | o4-mini | Gemma 3 27B | Gemini 2.0 | Qwen2.5-VL | Mistral | **Avg.** |
|--------|--------|---------|-------------|------------|------------|---------|----------|
| TRELLIS | 1.332 | 1.426 | 1.870 | 1.402 | 1.596 | 1.228 | 1.476 |
| Hunyuan3D-v2.5 | 1.614 | 1.690 | 2.098 | 1.533 | 1.780 | 1.501 | 1.703 |
| **TIRE (Ours)** | **1.777** | **1.834** | **2.103** | **1.793** | **1.880** | **1.739** | **1.854** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o progressive inpainting | The model projects source-view frontal textures onto all views (e.g., cat whiskers appear on side faces) |
| Denoising schedule 15% | Some regions remain unmodified |
| Denoising schedule 30% (default) | Best overall balance |
| Denoising schedule 50% | Textures change too drastically, reducing realism |

### Key Findings

- Even state-of-the-art methods such as TRELLIS and Hunyuan3D-v2.5 still perform poorly on identity preservation, indicating the problem remains unsolved.
- A user study (18 participants × 10 samples) shows TIRE achieves the highest overall quality score, even without informing participants of the evaluation focus in advance.
- TIRE not only improves appearance identity preservation but also incidentally improves geometry quality by reducing artifacts caused by cross-view inconsistencies.
- VLM scores for all methods remain well below the maximum of 4, confirming that subject-driven 3D/4D generation is far from solved.

## Highlights & Insights

- **Strong generalizability**: Operating solely on 2D rendered frames, the method is independent of the underlying 3D representation and can be integrated with any 3D/4D generation pipeline.
- **Elegant progressive strategy**: The "sweet-spot" concept effectively balances exploration (covering more unseen regions) and exploitation (maintaining reliable inpainting quality).
- **Insight behind backward tracking**: The source view contains the richest identity information; tracking backward from target to source maximizes utilization of known appearance.
- **Revealing limitations of DINO metrics**: Customize-It-3D achieves the highest DINO score yet produces qualitatively inferior results, exposing the inadequacy of conventional metrics for evaluating 3D identity preservation.

## Limitations & Future Work

- The approach depends on the initial quality of existing 3D generation models; severely incorrect initial geometry is difficult to remedy.
- The multi-stage pipeline introduces complexity, and end-to-end inference speed is constrained by the inpainting model fine-tuning step.
- The fixed denoising schedule ratio (30%) may not be optimal across all scenarios.
- Back-facing regions must be entirely hallucinated, limiting effectiveness in heavily occluded settings.
- The "sweet-spot" angle selection ($\pm 20°$) for progressive inpainting is empirically determined and lacks an adaptive mechanism.

## Related Work & Insights

- Compared to DreamBooth3D, TIRE does not rely on image-to-image translation but instead leverages more precise video tracking combined with inpainting.
- The subject-driven inpainting idea from RealFill is extended to 3D scenes.
- This work establishes a research direction orthogonal to feed-forward 3D/4D generation, with both directions capable of advancing synergistically.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Creatively combining 2D video tracking and inpainting for 3D identity preservation is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers both image-to-3D and video-to-4D settings, incorporating VLM-based evaluation and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, method descriptions are thorough, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Practically meaningful as a post-processing enhancement for existing 3D generation pipelines, with strong generalizability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[NeurIPS 2025\] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions](omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)
- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](../../ICCV2025/image_generation/freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[CVPR 2026\] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](../../CVPR2026/image_generation/taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)

<!-- RELATED:END -->
