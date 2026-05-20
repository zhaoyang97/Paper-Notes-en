---
title: >-
  [Paper Note] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields
description: >-
  [ICLR 2026][3D Vision][HDR reconstruction] This paper proposes HDR-NSFF, which shifts HDR video reconstruction from the conventional 2D pixel-level fusion paradigm to 4D spatiotemporal modeling. From alternating-exposure…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "HDR reconstruction"
  - "neural scene flow fields"
  - "dynamic scene"
  - "tone-mapping"
  - "4D radiance field"
date: 2026-05-08
content_hash: 01710eb729ffff9e
---

# HDR-NSFF: High Dynamic Range Neural Scene Flow Fields

**Conference**: ICLR 2026
**arXiv**: [2603.08313](https://arxiv.org/abs/2603.08313)  
**Code**: [Project Page](https://shin-dong-yeon.github.io/HDR-NSFF/)  
**Area**: 3D Vision
**Keywords**: HDR reconstruction, neural scene flow fields, dynamic scene, tone-mapping, 4D radiance field

## TL;DR
This paper proposes HDR-NSFF, which shifts HDR video reconstruction from the conventional 2D pixel-level fusion paradigm to 4D spatiotemporal modeling. From alternating-exposure monocular videos, it jointly reconstructs HDR radiance fields, 3D scene flow, geometry, and tone-mapping, enabling temporally and spatially consistent dynamic HDR novel view synthesis.

## Background & Motivation
The dynamic range of real-world scenes far exceeds the capture capability of ordinary cameras. Traditional HDR methods recover lost information by fusing frames with different exposures, but suffer from fundamental limitations:

1. **Inherent Defects of 2D Alignment**: Existing HDR video methods (e.g., LAN-HDR, HDRFlow, NECHDR) rely on 2D pixel-level alignment, operating within a narrow temporal window of only 3–7 frames, and lack physical understanding of the 3D scene.
2. **Color Drift and Geometric Flickering**: Without radiometric and spatiotemporal consistency constraints across distant frames, 2D methods frequently produce noticeable artifacts.
3. **Information Scarcity in Monocular Input**: Alternating-exposure monocular video provides only a single viewpoint at any given moment and is frequently affected by overexposure, making the reconstruction problem highly underdetermined.

These issues motivate the authors to propose a paradigm shift from 2D pixel fusion to 4D spatiotemporal modeling.

## Core Problem
How to reconstruct a spatiotemporally consistent dynamic HDR radiance field from alternating-exposure monocular video? Specific challenges include:

- Exposure variation causes severe inter-frame color inconsistency, rendering conventional optical flow methods ineffective.
- Monocular video provides only a single viewpoint, and information in overexposed regions is completely lost.
- The coupled relationships among HDR radiance, 3D motion, geometry, and tone-mapping must be modeled simultaneously.

## Method

### Overall Architecture
HDR-NSFF builds upon the 4D representation of Neural Scene Flow Fields (NSFF), mapping the entire video into a unified 4D scene representation. The framework comprises three core components:

### 1. Learnable Tone-Mapping Module
A learnable tone-mapping module $\mathcal{T}$ is introduced to map rendered HDR radiance $E$ into the LDR domain:

$$C = \mathcal{T}(E; \theta) = g_\theta(w(E))$$

where $w$ denotes per-channel white balance correction and $g_\theta$ is the camera response function (CRF). A piecewise parametric CRF design is adopted to balance flexibility and regularization. Compared to a fixed CRF (overly constrained) or an MLP-based CRF (overly flexible and unstable), the piecewise CRF achieves the best performance on the HDR-GoPro dataset.

### 2. Semantics-Based Exposure-Robust Optical Flow Estimation
The core insight is that while pixel-level appearance fluctuates drastically with exposure changes, the semantic features of objects remain invariant. Accordingly, the robust embedding space of DINOv2 is leveraged in place of traditional color matching for motion estimation:

- DINO-Tracker is adopted as the motion estimation backbone with task-specific modifications.
- Tracking points are re-initialized at each time step to prevent error accumulation over long sequences.
- Motion masks from SAM2 confine tracking to dynamic regions, filtering background noise.

### 3. Generative Prior as Regularizer
To compensate for the information deficiency of monocular video, a generative prior $\mathcal{G}$ is introduced to distill missing information into the radiance field:

- During optimization, candidate novel views $\hat{C}$ are periodically rendered and enhanced by the generative prior to produce $C^{\text{gen}}$.
- A patch-wise perceptual loss enforces alignment: $\hat{\mathcal{L}}_{\text{gen}} = \sum_{p} \|\phi(\hat{C}_p) - \phi(C_p^{\text{gen}})\|_1$
- The prior is activated with probability $p_{\text{gen}} = 0.1$ only after a warm-up period of $T_{\text{warm}} = 200K$ iterations, preventing generative hallucinations from overriding physically grounded observations.

### 4. Overall Objective
The model is jointly optimized with a tone-mapped photometric loss, optical flow constraints, depth priors, CRF smoothness regularization, and the generative prior loss.

## Key Experimental Results

### Datasets
- **HDR-GoPro** (newly proposed): The first real-world dynamic HDR dataset, captured with 9 synchronized GoPro Hero 13 Black cameras at 3 exposure levels across 12 indoor and outdoor scenes.
- **Synthetic data**: Used for comparative evaluation.

### Novel View Synthesis (GoPro Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| NSFF | 18.02 | 0.6792 | 0.2061 |
| 4DGS | 20.94 | 0.7905 | 0.1541 |
| NeRF-WT | 29.70 | 0.9333 | 0.0598 |
| HDR-HexPlane | 20.70 | 0.6694 | 0.1917 |
| **HDR-NSFF (Full)** | **32.63** | **0.9444** | **0.0554** |

### Novel View + Temporal Synthesis (Synthetic Data)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| NSFF | 15.98 | 0.6457 | 0.1388 |
| HDR-HexPlane | 29.95 | 0.9055 | 0.0527 |
| **HDR-NSFF** | **35.07** | **0.9465** | **0.0483** |

### Ablation Study
- Removing DINO-Tracker (DT): PSNR drops from 32.66 to 31.04 (dynamic region PSNR from 25.65 to 24.93), demonstrating that semantic optical flow is critical for motion consistency.
- Removing generative prior (GP): LPIPS increases from 0.0554 to 0.0557, indicating that GP primarily improves perceptual quality.
- Tone-mapping design comparison: piecewise CRF (PSNR 31.01) >> MLP CRF (28.76) >> Fixed CRF (25.55) >> No tone-mapping (17.79).

## Highlights & Insights
1. **Paradigm Innovation**: The first work to elevate HDR video reconstruction from 2D pixel fusion to 4D spatiotemporal modeling, establishing a global temporal receptive field.
2. **Semantic Optical Flow**: Elegantly exploits DINOv2's semantic invariance to resolve optical flow failures caused by exposure variation.
3. **Representation Agnosticism**: The framework is compatible with multiple dynamic representations, including NeRF and 4D Gaussian Splatting.
4. **First Real-World Dynamic HDR Dataset**: The HDR-GoPro dataset, comprising 9 synchronized cameras and 12 scenes, fills a critical evaluation gap in the community.
5. **End-to-End Joint Optimization**: Simultaneously learns radiance, motion, geometry, and tone-mapping, ensuring physical consistency.

## Limitations & Future Work
1. **Reliance on COLMAP Pose Estimation**: Pose estimation may fail under extreme exposure variation, limiting practical applicability.
2. **Motion Blur Not Addressed**: Motion blur induced by long exposures is not explicitly modeled.
3. **Training Efficiency**: The NeRF-based framework incurs high training costs; while 4DGS compatibility is claimed, it is not thoroughly validated in the main experiments.
4. **Limited Contribution of Generative Prior**: Ablation results show that GP yields a marginal PSNR decrease (32.66→32.63), with improvements mainly reflected in LPIPS.
5. **Limited Scene Scale**: Validation is conducted on a restricted set of scenes; generalization to large-scale outdoor environments remains unknown.

## Related Work & Insights
- **vs. HDR Video Methods** (LAN-HDR, HDRFlow, NECHDR): This work approaches the problem from 4D modeling rather than 2D alignment, naturally resolving temporal consistency issues.
- **vs. Dynamic Scene Reconstruction** (NSFF, 4DGS, MotionGS): These methods assume photometrically consistent LDR inputs and cannot handle HDR scenes.
- **vs. HDR-HexPlane**: The most closely related work, but its decomposed grid representation lacks explicit motion modeling, limiting temporal synthesis capability. HDR-NSFF achieves superior spatiotemporal interpolation through explicit 3D scene flow modeling.

**Broader Insights:**
1. **Semantic Features over Pixel Matching**: When pixel-level signals are unreliable (due to exposure changes, weather variation, etc.), exploiting invariances at the semantic level is a generalizable strategy applicable to other low-level vision tasks.
2. **Generative Prior as Regularizer**: Using generative models to regularize and supplement missing information is a promising strategy, but requires careful design of activation schedules to prevent hallucination.
3. **Lifting 2D Problems to 4D**: Elevating problems originally solved in 2D to higher-dimensional continuous representations fundamentally provides global consistency guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ (The 2D→4D paradigm shift is innovative; module combinations are well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐ (New dataset + synthetic data with complete ablations, though real-scene scale is limited)
- Writing Quality: ⭐⭐⭐⭐ (Clear framework, well-articulated motivation, high-quality figures)
- Value: ⭐⭐⭐⭐ (Opens a new direction for dynamic HDR 4D reconstruction; dataset contribution is valuable to the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[ICLR 2026\] Einstein Fields: A Neural Perspective To Computational General Relativity](einstein_fields_a_neural_perspective_to_computational_general_relativity.md)
- [\[ICLR 2026\] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)

</div>

<!-- RELATED:END -->
