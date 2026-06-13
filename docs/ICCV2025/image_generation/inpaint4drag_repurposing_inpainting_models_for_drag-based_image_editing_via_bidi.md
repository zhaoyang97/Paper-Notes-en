---
title: >-
  [Paper Note] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping
description: >-
  [ICCV 2025][Image Generation][Drag-based editing] This paper proposes Inpaint4Drag, which decomposes drag-based image editing into two stages: pixel-space bidirectional warping and image inpainting. Inspired by elastic o…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Drag-based editing"
  - "image inpainting"
  - "bidirectional warping"
  - "real-time preview"
  - "pixel-space deformation"
date: 2026-05-08
content_hash: 031a2ee4ccdcae5a
---

# Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping

**Conference**: ICCV 2025
**arXiv**: [2509.04582](https://arxiv.org/abs/2509.04582)  
**Code**: [Project Page](https://visual-ai.github.io/inpaint4drag)  
**Area**: Image Generation / Image Editing
**Keywords**: Drag-based editing, image inpainting, bidirectional warping, real-time preview, pixel-space deformation

## TL;DR

This paper proposes Inpaint4Drag, which decomposes drag-based image editing into two stages: pixel-space bidirectional warping and image inpainting. Inspired by elastic object deformation, the proposed bidirectional warping algorithm enables real-time preview (0.01s) and efficient generation (0.3s), achieving a 600× speedup over existing methods while serving as a universal adapter for arbitrary inpainting models.

## Background & Motivation

- **Background**: Methods such as DragGAN and DragDiffusion enable intuitive image manipulation via mouse drag interactions.
- **Limitations of Prior Work**: Three fundamental limitations exist in current approaches:
  1. **Insufficient precision**: Latent-space manipulation downsamples control points from 512×512 to 32×32, resulting in significant loss of spatial accuracy.
  2. **Poor interactivity**: The generation process provides no immediate visual feedback, forcing users into repeated trial-and-error cycles.
  3. **Limited capability**: General text-to-image models handle occluded regions poorly (e.g., head rotation, mouth opening with large exposed areas).
- **Core Idea**: Drag-based editing can be fundamentally decomposed into two steps: geometric transformation (warping) + content generation (inpainting). Pixel-space deformation estimation provides precise control and real-time preview, while specialized inpainting models outperform general generative models in filling newly exposed regions.

## Method

### Overall Architecture

Inpaint4Drag consists of three modules:
1. **Region specification and boundary refinement** (optional, SAM-based)
2. **Bidirectional warping algorithm** (real-time pixel-space deformation)
3. **Inpainting model integration** (standard inpainting format input)

### Module 1: Region Specification and SAM Boundary Refinement

The user draws a mask to specify the deformable region; an optional SAM refinement module improves boundary quality.

**Problem**: Direct SAM application may produce disconnected regions or capture unintended objects.

**Solution**: Dilated/eroded masks are used to constrain SAM predictions:
$$M = (M_{\text{pred}} \cap M_{\text{dilated}}) \cup M_{\text{eroded}}$$

where $M_{\text{dilated}}$ and $M_{\text{eroded}}$ are obtained by dilating and eroding the user mask with a kernel of radius $r_1=10$, respectively.

### Module 2: Bidirectional Warping Algorithm

Inspired by elastic object deformation, the image region is treated as a deformable material. The algorithm consists of four steps:

**Step 1 – Contour extraction and control point association**:
$$\mathcal{C} = \text{findContours}(M)$$
Each contour responds only to control points located within it.

**Step 2 – Forward warping** (defines target region boundary and establishes initial mapping):
$$p_t = p + \sum_{i=1}^{N_\mathcal{C}} w_i(t_i - h_i)$$

Weights are computed via inverse-distance weighting:
$$w_i = \frac{1/(\|p - h_i\| + \epsilon)}{\sum_{j=1}^{N_C} 1/(\|p - h_j\| + \epsilon)}$$

**Step 3 – Inverse mapping** (fills sampling gaps from forward warping):
$$p_s = p_t + \sum_{i=1}^{N_n} w_i(p_i^{\text{src}} - p_i^{\text{tgt}})$$

Using $N_n=4$ nearest-neighbor reference points, local-neighborhood-based inverse mapping ensures complete pixel coverage.

**Step 4 – Generating warped result and inpainting mask**:
$$I_{\text{warped}}(p_t) = I(p_s) \quad \text{for all valid pairs}$$
$$M_{\text{inpaint}} = \text{dilate}(M_{\text{temp}} \cup \partial M_{\text{warped}}, K_2)$$

The inpainting mask comprises: (1) unmapped regions exposed by deformation; (2) a narrow buffer band along the warp boundary. Dilation kernel radius $r_2=5$.

### Module 3: Inpainting Model Integration

$I_{\text{warped}}$ and $M_{\text{inpaint}}$ are fed as standard inpainting inputs:
$$I_{\text{edit}} = \text{Inpaint}(I_{\text{warped}}, M_{\text{inpaint}})$$

The SD 1.5 Inpainting Checkpoint is used, together with TinyAutoencoder SD (TAESD), LCM LoRA (reducing sampling steps to 8), empty-text prompts (eliminating CFG computation), and cached null prompt embeddings.

**Key design**: A real-time preview is provided prior to inpainting — users may adjust masks and control points until satisfied before executing inpainting.

## Key Experimental Results

### Main Results: Comparison with Drag-Based Editing Methods

| Method | DragBench-S MD↓ | DragBench-S LPIPS↓ | DragBench-D MD↓ | DragBench-D LPIPS↓ | VRAM (GB)↓ | Time (s)↓ |
|---|---|---|---|---|---|---|
| DragDiffusion | 7.0 | 18.0 | 6.7 | 10.2 | 11.6 | 177.7 |
| DiffEditor | 23.6 | 17.6 | 22.1 | 10.9 | 6.6 | 43.1 |
| SDE-Drag | 7.5 | 11.4 | 8.1 | 14.9 | 6.9 | 126.1 |
| FastDrag | 4.1 | 24.1 | 5.1 | 13.5 | 5.0 | 4.2 |
| **Inpaint4Drag** | **3.6** | **11.4** | **3.9** | **9.1** | **2.7** | **0.3** |

(MD and LPIPS values ×100)

Key findings:
- **Highest precision**: Lowest MD scores (3.6/3.9), best drag accuracy.
- **Fastest**: 14× faster than FastDrag, **600×** faster than DragDiffusion.
- **Lowest memory**: Requires only 2.7 GB VRAM.
- **Best image consistency**: LPIPS is optimal or competitive on both benchmarks.

### Per-Stage Latency Breakdown

| Stage | Time |
|---|---|
| SAM boundary refinement | 0.02s |
| Bidirectional warping preview | 0.01s |
| SD inpainting generation | 0.29s |

### Ablation Study: Unidirectional vs. Bidirectional Warping

Qualitative comparisons show:
- **Unidirectional (forward-only) warping**: Produces visible sampling artifacts in stretched regions with unmapped gaps at target positions.
- **Bidirectional warping**: By first determining the target contour via forward warping and then filling gaps through per-pixel inverse mapping, smooth and artifact-free transformations are achieved.

### Mask Refinement Module Effectiveness

Qualitative results demonstrate improvements across three scenarios:
- Raw user mask → raw SAM prediction (may include disconnected regions or unintended objects) → refined result (preserving user intent with improved boundary accuracy).

## Highlights & Insights

1. **Paradigm innovation**: Reframes drag-based editing from "guiding generation in latent space" to "pixel-space warping + inpainting," achieving a clean separation between geometric transformation and content generation.
2. **Physics-inspired design**: Treating image regions as elastic materials is intuitive and aligns naturally with users' physical-world intuition.
3. **Universal adapter**: By outputting standard inpainting format, the method integrates seamlessly with any inpainting model and automatically inherits future advances in inpainting technology.
4. **Real-time interaction**: The 0.01s warp preview allows users to visualize results before executing the costly inpainting step, substantially improving the editing experience.
5. **Extreme efficiency**: At 2.7 GB VRAM and 0.3s inference, drag-based editing becomes practically deployable on consumer-grade GPUs for the first time.

## Limitations & Future Work

- The bidirectional warping relies on inverse-distance interpolation, which may produce unnatural results for complex non-rigid deformations such as facial expression changes.
- Generation quality depends on the downstream inpainting model; when exposed regions are large, quality is bounded by the inpainting model's capability.
- DragBench's deformable regions were re-annotated (differing from the original editable region concept), which may introduce evaluation inconsistencies.
- The method is not applicable to drag editing scenarios requiring texture or style changes rather than geometric deformation.

## Related Work & Insights

- **Fundamental distinction from DragGAN/DragDiffusion**: Prior methods perform iterative optimization in latent space, whereas this work applies analytic transformation in pixel space.
- **Comparison with FastDrag**: FastDrag also adopts a "stretching" intuition but operates sequentially in latent space, lacking pixel-level precision and inpainting capability.
- **Broader insight**: Many generative tasks can achieve substantial gains in efficiency and precision by decomposing into "deterministic geometric transformation + localized generation."

## Rating ⭐⭐⭐⭐

The method is elegant and concise, directly addressing the core pain points of precision and efficiency in drag-based image editing. The 600× speedup and real-time preview confer strong practical value. Reframing drag editing as an inpainting problem is both novel and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[ICCV 2025\] Dense Policy: Bidirectional Autoregressive Learning of Actions](dense_policy_bidirectional_autoregressive_learning_of_actions.md)
- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)

</div>

<!-- RELATED:END -->
