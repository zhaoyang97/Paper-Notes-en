---
title: >-
  [Paper Note] Vinedresser3D: Agentic Text-guided 3D Editing
description: >-
  [CVPR 2026][Image Generation][3D editing] This paper presents Vinedresser3D, a 3D editing agent centered on a multimodal large language model (MLLM) that requires no user-provided 3D masks. The system automatically inter…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "3D editing"
  - "text-guided"
  - "agent"
  - "Trellis"
  - "flow model inversion"
date: 2026-05-08
content_hash: 05fc707d570fa139
---

# Vinedresser3D: Agentic Text-guided 3D Editing

**Conference**: CVPR 2026
**arXiv**: [2602.19542](https://arxiv.org/abs/2602.19542)  
**Area**: Image Generation
**Keywords**: 3D editing, text-guided, agent, Trellis, flow model inversion

## TL;DR

This paper presents Vinedresser3D, a 3D editing agent centered on a multimodal large language model (MLLM) that requires no user-provided 3D masks. The system automatically interprets editing intent, localizes editing regions, generates multimodal guidance, and performs inversion-based inpainting in the latent space of a native 3D generative model (Trellis), enabling high-quality text-guided 3D asset editing.

## Background & Motivation

Text-guided 3D editing is a fundamental problem in 3D computer vision, with broad applications in digital content creation, VR/AR, and robotics. Despite substantial progress in 3D generation, high-quality 3D editing remains heavily dependent on professional artists and manual tools, resulting in low efficiency and a high skill barrier.

Existing 3D editing methods face three major challenges:

**Insufficient semantic understanding**: Difficulty in accurately interpreting complex editing requests.

**Automatic localization**: Inability to automatically detect precise 3D editing regions from text alone.

**Poor editing fidelity**: Difficulty in following editing instructions while preserving unedited regions.

Existing approaches fall into three categories, each with notable drawbacks:

- **SDS-based methods** (Score Distillation Sampling): Optimize 3D representations via gradients from 2D diffusion models. Computationally expensive, require per-scene optimization, and prone to unintended global changes.
- **"2D editing + 3D reconstruction" pipelines**: Edit multi-view images first and then reconstruct. Limited by multi-view inconsistency and information loss due to occlusion.
- **Native 3D editing** (e.g., VoxHammer): Directly edit in 3D latent space, but still require manually provided 3D masks and cannot handle complex editing requests.

The authors argue that the natural next step is to build a 3D editing agent capable of **understanding high-level text instructions, automatically localizing editing regions, and coordinating multiple tools**.

## Method

### Overall Architecture

Vinedresser3D uses an MLLM (Gemini-2.5-flash) as its core, with a four-stage pipeline:

1. **Multimodal guidance generation**: The MLLM parses editing intent and generates text and image guidance.
2. **Editing region detection**: Automatically localizes regions to be edited in the 3D asset.
3. **Inversion-based 3D editing**: Performs inpainting-style editing in the Trellis latent space.
4. **Output decoding**: Decodes the edited SLAT into 3D Gaussians or meshes.

### Key Design 1: MLLM-based Multimodal Guidance Generation

**Text guidance** employs a multi-step prompting strategy:
- Step 1: The MLLM analyzes multi-view renderings and the editing instruction → generates a description of the original asset, identifies the target part to edit, and classifies the edit type (add/modify/delete).
- Step 2: Predicts a complete post-edit description (constraining the MLLM to maximally preserve descriptions of unedited regions).
- Step 3: Extracts an independent description of the newly added or modified target part.
- Step 4: Decomposes the description into structure-related (Stage 1 geometry) and appearance-related (Stage 2 features) components.

**Image guidance**: The MLLM selects the best viewpoint from 24 multi-view candidates (maximizing visibility of the editing target), which is then fed into an image editing model (Nano Banana) to generate a reference image.

### Key Design 2: Automatic Editing Region Detection

Requiring no user-provided 3D masks is a core advantage over prior methods.

1. PartField (a 3D segmentation model) decomposes the asset into $S$ semantic parts ($S \in [3,8]$).
2. The original asset renderings, segmentation color maps, and target text are fed into the MLLM → the editing region $P_{\text{edit}}$ is selected.
3. Editing regions are defined per edit type:

$$R_{\text{edit}} = \begin{cases} C \backslash A & \text{add (all non-asset voxels)} \\ P_{\text{edit}} & \text{delete (directly remove target part)} \\ P_{\text{edit}} \cup (C \backslash bbox_{\text{pres}}) \cup V & \text{modify (with KNN boundary criterion)} \end{cases}$$

where $V = \{v \mid v \in bbox_{\text{pres}} \backslash A, \text{PropKNN}(v) > \tau\}$

For modification, a KNN proportion threshold determines the assignment of empty voxels within the preservation bounding box, preventing Trellis from inadvertently modifying voxel layers above the preserved region.

### Key Design 3: Interleaved Trellis Inversion–Inpainting Editing

**Inversion stage**: RF-Solver (second-order Taylor expansion for improved inversion accuracy) inverts the original 3D asset back to structured noise:

$$X_{i-1} = X_i + (t_{i-1} - t_i) v_\theta(X_i, t_i) + \frac{1}{2}(t_{i-1} - t_i)^2 v_\theta^{(1)}(X_i, t_i)$$

CFG strength is set to 0 during inversion to stabilize the inversion trajectory and minimize reconstruction error.

**Editing stage**: An **Interleaved Trellis editing module** is proposed, alternating between Trellis-text and Trellis-image for denoising:
- Trellis-text: Provides broad semantic alignment and instruction-following capability.
- Trellis-image: Provides high-fidelity detail (but is limited by single-view occlusion).
- Stepwise alternation leverages the complementary strengths of both.

At each denoising step, latent features outside the editing mask (unedited regions) are injected from the original inversion trajectory, enabling mask-guided inpainting.

**Mask processing details**:
- Stage 1 masks are downsampled from $64^3$ to $16^3$ latent space.
- Stage 2 uses **soft masks**: boundary voxels of preserved regions are blended between denoised and inverted features via distance weighting, eliminating boundary floating artifacts.
- Delete operations skip Stage 1 and directly remove the target part, with Stage 2 smoothing the boundary.

### Loss & Training

This is an **inference-time method** with no training involved. The agent autonomously explores different positive/negative prompt combinations and selects the best result. Iterative multi-round editing is supported.

## Key Experimental Results

### Main Results: Quantitative Comparison (57 3D assets covering add/modify/delete)

| Method | Manual Mask | CLIP-T↑ | CD↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|:------:|---------|-----|-------|-------|--------|------|
| Instant3dit | ✓ | 0.227 | 0.027 | 20.86 | 0.851 | 0.153 | 80.35 |
| VoxHammer | ✓ | 0.235 | 0.027 | 24.36 | 0.890 | 0.087 | 34.95 |
| Trellis | ✓ | 0.247 | 0.010 | 37.35 | 0.984 | 0.017 | 31.10 |
| **Ours (auto mask)** | **✗** | **0.252** | 0.016 | 29.45 | 0.953 | 0.045 | **29.49** |
| **Ours + manual mask** | ✓ | **0.252** | **0.008** | **37.69** | **0.984** | **0.015** | **27.38** |

### User Study (Human Preference)

| Comparison | Text Alignment Win Rate | Unedited Preservation Win Rate | 3D Quality Win Rate |
|----------|:-----------:|:-------------:|:---------:|
| vs. Trellis | 92.5% | 82.0% | 90.8% |
| vs. VoxHammer | 89.8% | 79.3% | 90.2% |

### Ablation Study

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|-------|-------|--------|------|
| **Full method** | **29.45** | **0.953** | **0.045** | **29.49** |
| w/o Trellis-text (image only) | 28.06 | 0.943 | 0.054 | 30.59 |
| w/o editing region mask | 25.65 | 0.921 | 0.068 | 33.95 |

### Key Findings

- Even without manual 3D masks, Vinedresser3D achieves the best CLIP-T (text alignment 0.252) and FID (29.49) across all methods.
- With manual masks, all metrics reach their optimum, with PSNR improving notably from 29.45 to 37.69.
- The method achieves ~90% win rates over all baselines in the user study.
- Both the interleaved Trellis design and the editing region detection module contribute significantly to final quality, as confirmed by ablation.
- Using Trellis-image alone produces distorted or implausible outputs in occluded regions.

## Highlights & Insights

1. **Methodological innovation via the agent paradigm**: This is the first work to use an MLLM as the "brain" for 3D editing, coordinating an image editing model, a 3D segmentation model, and a 3D generative model to achieve end-to-end text-guided 3D editing. This paradigm-level contribution is more broadly instructive than improvements to individual model components.
2. **2D MLLMs can perform 3D reasoning**: Although the MLLM is trained solely on 2D image-text data, multi-view rendering inputs enable it to implicitly understand 3D spatial semantics, such as accurately localizing editing regions and understanding spatial relationships.
3. **Manageable gap between automatic and manual masks**: Automatic region detection already surpasses baseline methods that rely on manual masks, achieving the best performance on text alignment and overall quality.
4. **Interleaved denoising is simple yet effective**: Trellis-text and Trellis-image each have individual weaknesses; alternating between them yields complementary benefits.
5. **Unified framework for three edit types**: Addition, modification, and deletion are all handled within a single framework through different definitions of $R_{\text{edit}}$.

## Limitations & Future Work

1. **MLLM does not accept native 3D input**: Reliance on multi-view renderings to convey 3D information introduces information loss.
2. **Imperfect external tools**: PartField occasionally produces unreasonable segmentation results, affecting the accuracy of editing region detection.
3. **High inference cost**: Multiple MLLM calls, multi-view rendering, 3D segmentation, and image editing are all required, resulting in substantial latency and overhead.
4. **Limited dataset scale**: Evaluation is conducted on only 57 3D assets (24 generated + 33 manually created), which is relatively small.
5. **Deep coupling with Trellis**: Migrating to other 3D generative models would require redesigning the inversion and editing modules.
6. **Single best view may be insufficient**: For structurally complex objects, image guidance from a single viewpoint may not cover all editing details.

## Rating

⭐⭐⭐⭐ (4/5)

Employing an MLLM agent for 3D editing is a compelling direction. The method is well-designed, and experimental results demonstrate clear advantages in text alignment and user preference. Automatic mask detection eliminates the need for manual annotation, significantly improving usability. Weaknesses include the limited evaluation scale, tight coupling with Trellis, and insufficient discussion of inference cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](morphany3d_unleashing_the_power_of_structured_latent_in_3d_morphing.md)

</div>

<!-- RELATED:END -->
