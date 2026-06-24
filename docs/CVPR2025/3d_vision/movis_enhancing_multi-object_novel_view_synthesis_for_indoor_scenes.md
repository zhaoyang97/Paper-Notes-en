---
title: >-
  [Paper Note] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes
description: >-
  [CVPR 2025][3D Vision][novel view synthesis] Addressing novel view synthesis (NVS) for multi-object indoor scenes, this paper significantly improves cross-view object placement and geometric consistency through three key designs: injecting structure-aware features (depth + object masks), introducing an auxiliary mask prediction task, and designing a structure-guided timestep sampling scheduler.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "novel view synthesis"
  - "multi-object"
  - "structure-aware diffusion"
  - "timestep scheduling"
  - "cross-view consistency"
date: 2026-05-08
content_hash: 2de4c29e3e0dc395
---

# MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.11457](https://arxiv.org/abs/2412.11457)  
**Code**: [Project Page](https://jason-aplp.github.io/MOVIS/)  
**Area**: 3D Vision  
**Keywords**: novel view synthesis, multi-object, structure-aware diffusion, timestep scheduling, cross-view consistency

## TL;DR

Addressing novel view synthesis (NVS) for multi-object indoor scenes, this paper significantly improves cross-view object placement and geometric consistency through three key designs: injecting structure-aware features (depth + object masks), introducing an auxiliary mask prediction task, and designing a structure-guided timestep sampling scheduler.

## Background & Motivation

**Background**: Single-object NVS based on pretrained diffusion models (such as Zero-1-to-3) has achieved impressive results. However, almost all methods are trained on single-object datasets like Objaverse and cannot be directly scaled to multi-object compositional scenes.

**Limitations of Prior Work**: Directly applying single-object NVS methods to multi-object scenes leads to severe issues: incorrect object placement, deformed shapes, inconsistent appearances, and even missing objects. The fundamental reason is the lack of **structure-awareness**.

**Key Challenge**: Structural information in multi-object scenes is hierarchical—high-level object placement (position/orientation) and low-level per-object geometric appearance. The one-to-one mapping paradigm of single-object methods cannot handle this compositional complexity.

**Goal**: To enhance the perception of multi-object compositional structures in view-conditioned diffusion models, achieving cross-view consistent multi-object NVS.

**Key Insight**: Comprehensively inject structure-aware signals from three dimensions: model inputs, auxiliary tasks, and training strategies.

**Core Idea**: By leveraging depth/mask inputs, target-view mask prediction, and timestep resampling scheduling, the diffusion model is forced to learn global layout before local details.

## Method

### Overall Architecture

Based on the pretrained Stable Diffusion, the model takes an input view and a target view image pair for fine-tuning. Structure-aware features are concatenated at the input stage of the denoising U-Net, semantic information is injected via cross-attention, and the object mask of the target view is simultaneously predicted.

### Key Designs

#### 1. Structure-Aware Feature Amalgamation

The depth map and object mask of the input view are incorporated as additional inputs:
- Object Mask: Normalizes the instance ID rendering map into a continuous image, providing a coarse perception of object placement and shape.
- Depth Map: Encodes the relative positions and shapes of visible objects.
- Both are replicated to 3 channels to simulate RGB, encoded by the VAE, and concatenated with the noisy target view image.
- During inference, these can be obtained using off-the-shelf detectors like SAM + Marigold.

The modified learning objective is: $\mathbb{E}[\|\epsilon_\theta(\alpha_t x_0 + \sigma_t \epsilon, t, C_{SA}(\hat{x}_0, R, T, \hat{D}, \hat{M})) - \epsilon\|^2]$

#### 2. Auxiliary Target-View Mask Prediction

Analogous to the concept of classifier guidance, target-view object mask prediction is introduced as an auxiliary training task:
- The mask predictor extracts features from the last layer of the denoising U-Net, conditioned on the noisy image $x_t$, timestep $t$, and structure-aware features.
- Joint training loss: diffusion reconstruction + $\gamma \| M_{tgt} - M_t \|^2$ ($\gamma = 0.1$).
- This forces the model to explicitly learn "where objects should be placed in the target view."

#### 3. Structure-Guided Timestep Sampling Scheduler

Key Observation: Global object placement is restored in the early denoising stage (large $t$), while fine geometric details are restored in the late stage (small $t$).

The uniform sampling $t \sim \mathcal{U}(1, 1000)$ is modified to Gaussian sampling $t \sim \mathcal{N}(\mu(s), \sigma)$, where $\mu(s)$ linearly decays from $\mu_{global}=1000$ to $\mu_{local}=500$ ($\sigma=200$):
- First 4000 steps of warmup: $\mu = 1000$, emphasizing global layout learning.
- Linear decay from 4000 to 6000 steps.
- After 6000 steps: $\mu = 500$, shifting focus to fine-details learning.

### Loss & Training

$\mathcal{L} = \|\epsilon_\theta - \epsilon\|^2 + \gamma \|M_{tgt} - M_t\|^2$, where $\gamma = 0.1$, combining the diffusion reconstruction loss and the mask prediction MSE loss.

## Key Experimental Results

### Main Results: C3DFS Test Set

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | IoU↑ | Hit Rate↑ | Dist↓ |
|------|-------|-------|--------|------|-----------|-------|
| ZeroNVS | 10.7 | 0.533 | 0.481 | 21.6 | 1.4 | 135.2 |
| Zero-1-to-3 | 14.3 | 0.771 | 0.302 | 33.7 | 4.4 | 86.7 |
| Free3D | 14.4 | 0.774 | 0.297 | 34.2 | 4.8 | 83.6 |
| **MOVIS** | **17.4** | **0.825** | **0.171** | **58.1** | **19.3** | **44.9** |

MOVIS improves IoU (object placement accuracy) by **72%** (vs Zero-1-to-3) and Hit Rate (cross-view matching) by **339%**.

### Generalization: Objaverse + Room-Texture

- Objaverse: PSNR 17.7 / IoU 51.3 / Hit Rate 17.0 (outperforming others by a large margin)
- Room-Texture: PSNR 10.0 / IoU 24.2 / Hit Rate 4.4 (maintaining advantages across domains)

### Ablation Study

| Variant | PSNR↑ | LPIPS↓ | IoU↑ |
|------|-------|--------|------|
| w/o depth | 17.1 | 0.178 | 57.2 |
| w/o mask (auxiliary task) | 16.9 | 0.187 | 54.7 |
| w/o scheduler | 16.2 | 0.212 | 49.1 |
| **Full MOVIS** | **17.4** | **0.171** | **58.1** |

### Key Findings

1. **Timestep scheduler is the most critical component**: Removing it degrades IoU by 9 percentage points, indicating that the "global first, local second" learning sequence is crucial for multi-object scenes.
2. The auxiliary mask prediction task is the second most major contributor (IoU -3.4); direct supervision helps the model distinguish object instances.
3. Cross-view consistency metrics (Hit Rate/Dist) complement traditional NVS metrics, revealing structural problems that traditional metrics fail to reflect.

## Highlights & Insights

- **New Evaluation Dimension**: Proposes cross-view consistency metrics (Hit Rate and Dist based on MASt3R image matching) to fill the blind spot in NVS evaluation.
- **Hierarchical Analysis of Denoising**: By visualizing intermediate predictions at different timesteps, it uncovers the pattern of global layout restoration in the early stage and fine-mask prediction in the late stage.
- **Timestep Scheduler Design Philosophy**: Standardizes curriculum learning (easy-to-hard) in diffusion training; the "coarse-to-fine" process in multi-object scenes naturally aligns with the denoising progression.
- **Structure-Aware Input Design**: Though simple and intuitive, combining depth and masks with ready-to-use monocular predictors (SAM + Marigold) makes them applicable during inference, ensuring strong practicality.

## Limitations & Future Work

1. Only focuses on foreground objects without modeling the background (left for future work).
2. Trained on the synthetic dataset C3DFS, limiting generalization to real-world indoor scenes (such as SUNRGB-D).
3. Still requires input-view depth and mask as extra conditions, increasing inference costs.
4. Performance degrades under large view changes, where occluded regions demand stronger generative capabilities.

## Related Work & Insights

- **Zero-1-to-3**: Pioneered using diffusion models as NVS synthesizers, but limited to single objects; MOVIS demonstrates that multi-object extension requires explicit structure awareness.
- **Compositional 3D Reconstruction series (ComboVerse, etc.)**: A pipeline paradigm of segmentation -> completion -> single-object 3D -> composition suffers from accumulated cascade errors; the end-to-end scheme of MOVIS is more concise.
- **Related Insight**: The concept of a timestep resampling scheduler can be extended to other diffusion tasks requiring "hierarchical generation" (e.g., scene generation from layout to texture).

## Rating

⭐⭐⭐⭐ — Clear problem definition. The three design choices are orthogonal to each other and their contributions are well validated through ablation. The insight behind the timestep scheduler is particularly interesting. Generalization on real-world data still needs further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] SoundVista: Novel-View Ambient Sound Synthesis via Visual-Acoustic Binding](soundvista_novel-view_ambient_sound_synthesis_via_visual-acoustic_binding.md)

</div>

<!-- RELATED:END -->
