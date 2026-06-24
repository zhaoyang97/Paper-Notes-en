---
title: >-
  [Paper Note] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects
description: >-
  [CVPR 2025][3D Vision][3D editing] This paper reformulates the 3D editing problem into a multi-view consistent 2D inpainting task. By fine-tuning the SDXL-inpainting model to simultaneously generate consistent filled content on a $2 \times 2$ view grid and then reconstructing the 3D asset using a Large Reconstruction Model (LRM), high-quality 3D editing is achieved in approximately 3 seconds—hundreds of times faster than Score Distillation Sampling (SDS) methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D editing"
  - "multiview inpainting"
  - "3D masking"
  - "diffusion model"
  - "SDXL"
  - "mesh editing"
date: 2026-05-08
content_hash: c9cce9cde133367c
---

# Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects

**Conference**: CVPR 2025  
**arXiv**: [2412.00518](https://arxiv.org/abs/2412.00518)  
**Code**: [Project Page](https://amirbarda.github.io/Instant3dit.github.io/)  
**Area**: 3D Vision  
**Keywords**: 3D editing, multiview inpainting, 3D masking, diffusion model, SDXL, mesh editing

## TL;DR

This paper reformulates the 3D editing problem into a multi-view consistent 2D inpainting task. By fine-tuning the SDXL-inpainting model to simultaneously generate consistent filled content on a $2 \times 2$ view grid and then reconstructing the 3D asset using a Large Reconstruction Model (LRM), high-quality 3D editing is achieved in approximately 3 seconds—hundreds of times faster than Score Distillation Sampling (SDS) methods.

## Background & Motivation

**Background**: 3D generative editing (e.g., adding or modifying components on an existing model) typically relies on Score Distillation Sampling (SDS) or Iterative Dataset Update (IDU) to progressively optimize the 3D representation, which takes tens of minutes to hours.

**Limitations of Prior Work**:
1. **Extremely slow execution time**: SDS requires repeated rendering, diffusion model inference, and gradient backpropagation. Even with Gaussian Splatting acceleration, it still takes ~15 minutes.
2. **Low-quality outputs**: SDS tends to seek mode-seeking behaviors, leading to a "lack of diversity" in generation results as well as oversaturated or blurry textures.
3. **Multi-view inconsistency**: Existing 2D inpainting models lack 3D consistency, making 3D reconstruction difficult after direct per-view inpainting.

**Key Challenge**: Achieving fast 3D editing requires avoiding iterative optimization like SDS, but directly applying 2D generative models loses multi-view consistency.

**Goal**: To design a fast, high-quality, and representation-agnostic 3D inpainting method that allows users to perform localized 3D editing via simple 3D masks and text prompts.

**Key Insight**: Instead of optimizing in the 3D space, this work trains a diffusion model capable of simultaneously inpainting a $2 \times 2$ multi-view image grid, and then reconstructs the 3D asset using off-the-shelf LRMs.

## Method

### Overall Architecture

1. The user draws a mask $M$ (a coarse 3D geometry) on the 3D object and provides a text prompt $y$.
2. The masked object $I_c(S,M)$ and mask $I_b(M,S)$ are rendered from 4 canonical viewpoints and assembled into a $2 \times 2$ grid.
3. A multi-view inpainting diffusion model $\epsilon_\theta$ generates four consistent inpainted views simultaneously on the grid.
4. Off-the-shelf LRMs (such as NeRF-LRM, MeshLRM, or GS-LRM) are utilized to reconstruct the new 3D representation from the multi-view images.

### Key Designs

**1. Multi-view Consistent Inpainting Fine-tuning Strategy**
- **Function**: Starting from the SDXL-inpainting model, fine-tune it on 3D multi-view inpainting data to simultaneously acquire both inpainting capability and multi-view consistency.
- **Mechanism**: 
    - The training data consists of 5K high-quality 3D objects filtered from Objaverse, with each object rendered into a 4-view $2 \times 2$ grid.
    - Input consists of 9 channels: noise latent (4ch) + mask image latent (4ch) + downsampled mask (1ch).
    - Mask conditioning is randomly dropped $10\%$ of the time to degenerate into pure multi-view generation training.
    - Text conditioning is provided by high-quality captions generated for 3D objects using LLaVA.
- **Design Motivation**: Compared to the reverse path (fine-tuning inpainting capabilities starting from a multi-view generation model like Instant3D), fine-tuning from an inpainting model is easier. This is because multi-view consistency can be learned from a relatively small amount of 3D data, whereas inpainting requires training on large-scale datasets.

**2. 3D Mask Design (Three Editing Modes)**
- **Function**: Design three types of 3D mask datasets to simulate practical user editing behaviors.
- **Type I (Coarse Edit)**: Cut the object with a random plane, and use the convex hull of the subset scaled up by $20\%$ as the mask—suitable for coarse replacement.
- **Type II (Mesh Sculpting)**: Cut with a random plane, and directly select the object surface above the cutting plane—suitable for precise sculpting.
- **Type III (Surface Editing)**: Sample a point on the object surface, and use multiple random elliptical cylinders to select local surface patches—suitable for texture modification.
- **Design Motivation**: Mask design is key to inpainting training. The closer the training mask distribution is to user behavior at test time, the better the performance. 3D-consistent masks (taking occlusion into account) perform significantly better than random 2D masks.
- Each object has 30 masks (10 of each type), totaling approximately 150K training samples.

**3. Representation-Agnostic 3D Reconstruction**
- **Function**: The inpainted multi-view images can be fed into different LRMs to obtain various 3D representations.
- **Mechanism**: 
    - NeRF: Reconstructed in milliseconds using NeRF-LRM.
    - Mesh: Uses MeshLRM + ROAR adaptive remeshing (~20 seconds) to preserve properties of the original mesh, such as UVs and topology.
    - Gaussian Splatting: Reconstructed using GS-LRM.
    - A normal estimator directly predicts surface normals from the diffusion output to preserve details during mesh optimization.
- **Design Motivation**: Editing at the 2D image level naturally decouples the choice of 3D representation, allowing the system to flexibly adapt to different downstream requirements.

### Loss & Training

- Standard latent diffusion v-prediction loss.
- Base model: SDXL-inpainting.
- Training data: 5K high-quality objects from Objaverse, with 150K multi-view mask sample pairs.
- Inference: Euler scheduler, 29 steps, ~3 seconds per edit on an A100 GPU.
- Reconstruction overhead: NeRF-LRM ~0.7s, Mesh ~3s, Mesh+ROAR ~20s.

## Key Experimental Results

### Main Results (500-sample benchmark)

| Method | ClipL↑ | ClipG↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | FID↓ |
|---|---|---|---|---|---|---|
| SDXL | Low | Low | Low | High | High | High |
| SDXL-inpainting | Low | Low | Low | High | High | High |
| Instant3D | Medium | Medium | **Highest** | **Lowest** | **Lowest** | High |
| **Ours (SDXL-inp.)** | **Highest** | **Highest** | Second Highest | Second Lowest | Second Lowest | **Lowest** |

Instant3D achieves the best performance in multi-view consistency (as it is inherently a multi-view model) but falls short of the proposed method in prompt adherence and visual quality.

### Ablation Study on Masks

| Training Mask | Type I FID↓ | Type II FID↓ | Type III FID↓ | All FID↓ | User Mask ClipG↑ |
|---|---|---|---|---|---|
| Random 2D | Poor | Poor | Poor | Poor | Worst |
| Type I | Best | Medium | Medium | Medium | Medium |
| Type II | Medium | Best | Medium | Medium | Medium |
| Type III | Medium | Medium | Best | Medium | Medium |
| **I+II+III** | Second Best | Second Best | Second Best | **Best** | **Best** |

### User Preference Study

In 208 pairwise comparisons (judged by 15 users), Instant3dit vs. NeRFiller:
- **Instant3dit: 86% preference** vs. NeRFiller: 14%

### Speed Comparison

| Method | Editing Time |
|---|---|
| Vox-E | ~1h |
| Progressive3D | ~1h |
| NeRFiller | ~30K steps |
| MVEdit | ~30min |
| **Instant3dit** | **~3s + reconstruction** |

### Key Findings

1. **Fine-tuning from inpainting is more effective than from multi-view generation**: Starting from SDXL-inpainting significantly outperforms starting from Instant3D in prompt adherence and visual quality, which demonstrates that the pre-existing large-scale inpainting priors serve as a crucial advantage.
2. **3D-aware masks are critical**: Models trained with random 2D masks perform significantly worse across all metrics compared to those trained with 3D-aware masks.
3. **Hybrid mask training offers the best generalization**: Models trained on a single mask type perform optimally on that type but generalize poorly; hybrid training (I+II+III) exhibits the most balanced performance across arbitrary masks.
4. **Generalization to novel viewpoints**: Although training is conducted only on fixed viewpoints, the self-attention mechanism of the inpainting model and the priors from the original foundation model enable generalization to arbitrary azimuthal offsets.
5. **Fundamental limitations of SDS**: In addition to being slow, SDS produces significantly worse quality in inpainting scenarios (tending toward mode collapse), validating the observations in the original DreamFusion paper.

## Highlights & Insights

- Formulating 3D editing as multi-view 2D inpainting represents an elegant paradigm shift, resolving both speed and quality challenges simultaneously.
- Ingenious 3D mask dataset design: The three mask types correspond to three practical editing actions (coarse replacement, precise sculpting, and local texturing), ensuring that the training distribution aligns closely with real-world scenarios.
- Representation agnosticism is a key engineering advantage: A single multi-view inpainting pipeline can be directly integrated with various LRMs supporting NeRF, Mesh, or GS representations.
- Experimental ablations on fine-tuning strategies (from inpainting vs. from multi-view) provide valuable design guidelines.

## Limitations & Future Work

- Extremely thin masks may be ignored (a common issue in 2D inpainting, where models tend to redraw larger areas).
- Since the training data contains only white backgrounds, large masks and the lack of strong inductive biases can lead to white background generation instead of adhering to the text prompt.
- It supports only a 4-view $2 \times 2$ grid, which might be insufficient for complex occlusions or large-scale scenes.
- The quality relies heavily on LRM reconstruction; currently, limited tri-plane resolutions in LRMs lead to overly smoothed geometries.
- The training data consists of only 5K objects from Objaverse, limiting the diversity of object categories and editing types covered.

## Related Work & Insights

- Instant3D demonstrates that $2 \times 2$ grid generation combined with LRM reconstruction is a viable pathway for fast 3D generation; this work extends it to conditional inpainting.
- NeRFiller performs 3D inpainting using single-view inpainting with IDU optimization, but it is slow and suffers from multi-view inconsistency; this work achieves multi-view consistent inpainting for the first time.
- MagicClay presents a local constraint scheme (ROAR) for mesh editing, a technique that is reused in this work after mesh reconstruction.
- Insight: Fine-tuning a 2D foundation model for 3D consistency can be more efficient than training directly on 3D data, as 2D pre-training transfers rich visual priors.

## Rating

⭐⭐⭐⭐ — The paradigm shift (3D editing $\rightarrow$ multi-view inpainting) brings a dual leap in both speed and quality. The mask design and fine-tuning strategies are well-supported by extensive ablation studies. The main limitations reside in the scale of the training dataset and the bottleneck regarding reconstruction quality of LRMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)
- [\[CVPR 2025\] Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories](perturb-and-revise_flexible_3d_editing_with_generative_trajectories.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)
- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)

</div>

<!-- RELATED:END -->
