---
title: >-
  [Paper Note] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting
description: >-
  [ICCV 2025][3D Vision][Amodal completion] This paper proposes the first amodal completion framework tailored for human-object interaction (HOI) scenes. It leverages human body topology and contact information to identify occluded regions via convex hull operations, and employs a multi-regional inpainting strategy on a pretrained diffusion model to achieve high-quality occluded object completion without any additional training.
tags:
  - ICCV 2025
  - 3D Vision
  - Amodal completion
  - human-object interaction
  - multi-regional inpainting
  - diffusion models
  - contact estimation
date: 2026-05-08
content_hash: 33e90e614089f07b
---

# Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting

**Conference**: ICCV 2025
**arXiv**: [2508.00427](https://arxiv.org/abs/2508.00427)
**Code**: None
**Area**: 3D Vision / Human-Object Interaction Understanding
**Keywords**: Amodal completion, human-object interaction, multi-regional inpainting, diffusion models, contact estimation

## TL;DR

This paper proposes the first amodal completion framework tailored for human-object interaction (HOI) scenes. It leverages human body topology and contact information to identify occluded regions via convex hull operations, and employs a multi-regional inpainting strategy on a pretrained diffusion model to achieve high-quality occluded object completion without any additional training.

## Background & Motivation

Amodal completion — inferring the complete appearance of partially occluded objects — is essential for understanding complex HOI scenes. Key limitations of existing methods:

**Inaccurate inpainting regions**: Directly using the occluder's mask (e.g., the human body) as the inpainting region typically yields a region far larger than the actual occlusion, causing the diffusion model to produce over-extended or inaccurate completions.

**Lack of HOI priors**: Existing methods fail to exploit the unique characteristics of HOI scenes — the visible region is often concave, human body topology is accessible, and contact points provide critical spatial relationships.

**Single-region inpainting limitations**: Conventional methods handle only a single mask region and cannot apply differentiated strategies to regions with varying occlusion probabilities.

## Method

### Overall Architecture

The pipeline consists of two core stages:
1. **Occluded Region Identification**: Contact information and convex hull operations are used to partition the occluded region into a primary region $M_p$ (high occlusion probability) and a secondary region $M_s$ (low occlusion probability).
2. **Multi-Regional Inpainting**: A pretrained Stable Diffusion v2 inpainting model is used with differentiated denoising strategies applied to the two regions.

### Key Designs

1. **Contact-aware Convex Hull**:

    - Dilation is applied to the human mask $M_{human}$ and object mask $M_{object}$ to obtain the occlusion boundary mask $M_{boundary}$.
    - $M_{boundary}$ is merged with the contact map $M_{contact}$: $C = M_{boundary} \cup M_{contact}$.
    - The convex hull of point set $C$ is computed: $H = \text{ConvexHull}(C)$.
    - Primary region mask: $M_p = M_{in} \cap M_{hull}$, i.e., the area within the convex hull that overlaps with the occluder.
    - Secondary region mask: $M_s = M_{in} \setminus M_p$, i.e., the area within the occluder mask but outside the convex hull.
    - **Design Motivation**: In HOI scenarios, occluded object parts tend to concentrate near contact points; the convex hull operation precisely localizes the critical inpainting region.

2. **Multi-Regional Inpainting Strategy**:

    - The standard SD-inpaint pipeline is extended to handle multi-region masks:
    $I_{out} = F_{T \to T'}(I_{in}, M_p, \mathcal{P}) \,|\, F_{T' \to 0}(I_{in}, M_p \cup M_s, \mathcal{P})$
    - where $T' = \lfloor T \cdot r \rfloor$ and $r$ is a strength parameter (default 0.5).
    - **Stage 1** ($T \to T'$): Denoising is performed on $M_p$ only to establish a coarse structure.
    - **Stage 2** ($T' \to 0$): Denoising is performed on $M_p \cup M_s$, progressively refining details based on the coarse structure from Stage 1.
    - No additional training is required; the method relies entirely on the pretrained SD-inpaint model.
    - **Design Motivation**: This exploits the diffusion model's inherent "structure-first, details-later" property, ensuring the primary region receives a coherent completion first, with the secondary region naturally following.

3. **In-the-Wild Pipeline**:

    - SAM is used to generate human and object masks (replacing ground-truth segmentation).
    - An HMR model estimates SMPL parameters to obtain 3D joint coordinates.
    - A VLM generates interaction descriptions (e.g., "a man is holding an object with both hands") from which relevant SMPL joint IDs are extracted.
    - 3D joint coordinates are projected to 2D space to generate the contact mask.
    - **Design Motivation**: This eliminates dependence on ground-truth annotations, enabling the method to operate on real-world scenes.

### Loss & Training

No training is required. The method is entirely inference-based on the pretrained Stable Diffusion v2 Inpainting model. The key hyperparameters are the strength parameter $r$ (controlling the timing of secondary region inpainting) and the DDIM scheduler step count $T=50$.

## Key Experimental Results

### Main Results — Amodal Completion Performance Comparison

| Method | BEHAVE CLIP↑ | BEHAVE mIoU↑ | InterCap CLIP↑ | InterCap mIoU↑ | Win-rate |
|--------|-------------|-------------|---------------|---------------|---------|
| Naive outpainting | 27.34 | 50.92% | 27.55 | 52.07% | 94.0% |
| LaMa | 25.97 | 60.47% | 26.43 | 51.38% | 92.4% |
| Inst-Inpaint | 26.08 | 63.71% | 26.12 | 57.54% | 88.0% |
| pix2gestalt | 23.45 | 69.58% | 26.14 | 68.32% | 68.0% |
| Xu et al. | 26.34 | 71.03% | 26.21 | 69.23% | 65.8% |
| **Ours** | **26.91** | **77.64%** | **26.97** | **72.34%** | - |

### Ablation Study — Region Strategy and Strength Parameter

| Method | Region | r | CLIP↑ | mIoU↑ |
|--------|--------|---|-------|-------|
| Naive outpainting | Full image | - | 27.34 | 50.92% |
| Human mask (single-region) | $M_p \cup M_s$ | 1.0 | 26.27 | 69.98% |
| Convex hull w/o contact | $M_p$ | 0.0 | 26.43 | 75.24% |
| Convex hull w/ contact | $M_p$ | 0.0 | 26.63 | 76.11% |
| **Ours (multi-region)** | **{$M_p, M_s$}** | **0.5** | **26.91** | **77.64%** |
| Ours + GT contact | {$M_p, M_s$} | 0.5 | 27.07 | 80.15% |

| Occlusion Level | r=0.0 mIoU | r=0.1 mIoU | r=0.5 mIoU | r=0.9 mIoU | r=1.0 mIoU |
|----------------|-----------|-----------|-----------|-----------|-----------|
| Light (10–40%) | 84.97% | **85.44%** | 84.70% | 80.33% | 72.45% |
| Heavy (40–70%) | 70.20% | 71.54% | **72.93%** | **73.94%** | 68.33% |
| Overall | 76.11% | 77.10% | **77.64%** | 76.50% | 69.98% |

### Key Findings

- **Substantial mIoU improvement**: The proposed method achieves 77.64% on BEHAVE, surpassing the strongest baseline (Xu et al., 71.03%) by 6.6 percentage points.
- **Dominant user preference**: In pairwise user studies, the method achieves win rates exceeding 65% against all baselines.
- **Contact information is critical**: Incorporating contact information improves convex hull mIoU from 75.24% to 76.11%.
- **Multi-regional strategy is effective**: Transitioning from single-region to multi-region inpainting yields more than 1.5 percentage points improvement in mIoU.
- **In-the-wild pipeline is reliable**: Without ground-truth contact, performance drops by only 2.5 percentage points in mIoU compared to using GT contact.
- **$r=0.5$ is the optimal trade-off**: Lighter occlusion favors smaller $r$; heavier occlusion favors larger $r$; 0.5 achieves the best overall performance.
- **3D reconstruction application**: Downstream 3D Gaussian Splatting reconstruction quality is significantly improved following amodal completion.

## Highlights & Insights

1. **First amodal completion framework specifically designed for HOI**: This work fills an important research gap by elegantly exploiting the unique geometric constraints of HOI scenes.
2. **Training-free multi-regional inpainting**: The standard inpainting pipeline is extended to apply different noise levels to regions of different priority, yielding a simple yet effective design.
3. **Contact + convex hull as physical priors**: Geometric priors are naturally integrated into the inpainting pipeline, substantially improving occlusion region localization accuracy.
4. **High practical applicability**: The in-the-wild pipeline combines SAM, HMR, and a VLM to operate without any manual annotation.

## Limitations & Future Work

- Validation is primarily conducted on indoor single-person single-object scenarios; generalization to complex multi-person multi-object scenes remains unexplored.
- The method operates on single images and lacks temporal consistency, making it unsuitable for video tasks.
- Performance depends on the inpainting capability of the diffusion model and may degrade for unseen object categories.
- The convex hull assumption may not be appropriate for certain non-convex occlusion patterns.
- Future work could extend this framework to temporally consistent video amodal completion and more complex multi-person interaction scenarios.

## Related Work & Insights

- The proposed method is complementary to general amodal completion approaches such as pix2gestalt, which do not account for the physical constraints specific to HOI.
- The idea of differentiated multi-regional inpainting can be generalized to other image editing tasks requiring zone-wise processing.
- The convex hull plus contact point region identification strategy could be applied to occluded object understanding in robotic grasping scenarios.
- The 3D Gaussian Splatting reconstruction application demonstrates the downstream value of amodal completion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of amodal completion specifically to HOI; the multi-regional inpainting strategy is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, user studies, extensive ablations, and a 3D application are included, though the datasets are limited to indoor settings.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, pipeline diagrams are intuitive, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ A practical tool for HOI understanding and 3D reconstruction that opens a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](../../CVPR2026/3d_vision/cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)

</div>

<!-- RELATED:END -->
