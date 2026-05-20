---
title: >-
  [Paper Note] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation
description: >-
  [NeurIPS 2025][Autonomous Driving][3D semantic scene generation] This work introduces the first sketch-to-3D outdoor semantic scene generation task along with a benchmark dataset, SketchSem3D…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "3D semantic scene generation"
  - "sketch-guided"
  - "diffusion model"
  - "state space model"
  - "cylindrical coordinate Mamba"
date: 2026-05-08
content_hash: 97f73aaf7aa2f1d3
---

# CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.13245](https://arxiv.org/abs/2510.13245)  
**Code**: [https://github.com/Lillian-research-hub/CymbaDiff](https://github.com/Lillian-research-hub/CymbaDiff)  
**Area**: Autonomous Driving
**Keywords**: 3D semantic scene generation, sketch-guided, diffusion model, state space model, cylindrical coordinate Mamba

## TL;DR
This work introduces the first sketch-to-3D outdoor semantic scene generation task along with a benchmark dataset, SketchSem3D, and proposes CymbaDiff (Cylinder Mamba Diffusion), a denoising network that achieves structured spatial modeling via dual-path Mamba blocks combining cylindrical and Cartesian scanning. CymbaDiff reduces FID by 75% over 3D Latent Diffusion and 71% over 3D DiT.

## Background & Motivation

**Background**: 3D outdoor semantic scene generation has attracted increasing attention in recent years. Methods such as UrbanDiff rely on BEV maps as conditional inputs; however, BEV representations lack fine-grained 3D structural information, limiting semantic richness and geometric fidelity. Multi-scale approaches further require repeated synthesis across multiple resolutions, resulting in high computational complexity.

**Limitations of Prior Work**: (a) No publicly available large-scale standardized benchmark exists — UrbanDiff uses a privately preprocessed dataset that precludes fair comparison; (b) sketch-guided generation methods are restricted to single objects or simple indoor scenes; (c) in Cartesian coordinates, adjacent voxel sequences may incorrectly represent spatial proximity, degrading sequential modeling quality.

**Key Challenge**: Outdoor large-scale scenes exhibit complex spatial structures and diverse semantics, yet existing generation methods lack both appropriate datasets and spatial encoding strategies that account for cylindrical continuity and vertical hierarchy.

**Key Insight**: (a) Construct the first large-scale benchmark pairing sketches and pseudo-annotated satellite imagery with 3D ground truth; (b) design an SSM module combining dual cylindrical and Cartesian scanning to preserve spatial consistency.

**Core Idea**: The Cylinder Mamba Block performs Mamba scanning in cylindrical coordinates ordered by $(θ, r, z)$ to preserve angular-radial continuity, then fuses Cartesian tri-directional Mamba to retain accurate geometric distance relationships.

## Method

### Overall Architecture
The inputs consist of a sketch (Canny edge map) and a pseudo-annotated satellite image (PSA). A Scene Structure Estimation Network (SSEN) extracts coarse structural priors, which are then encoded into a VAE latent space via a Latent Mapping Network. The CymbaDiff denoising network performs diffusion denoising in the latent space and decodes the result into a $256 \times 256 \times 32$ voxel semantic grid.

### Key Designs

1. **SketchSem3D Dataset Construction**:

    - Function: The first large-scale benchmark for sketch-guided 3D outdoor scene generation.
    - Mechanism: GPS information from KITTI/KITTI-360 is used to retrieve satellite imagery. CLIP encodes category description texts, and SAM extracts mask-level embeddings; cosine similarity matching generates PSA pseudo-annotations. Sketches are obtained via Canny edge detection from BEV projections.
    - Scale: Sketch-based SemanticKITTI (58,987 frames) + Sketch-based KITTI-360 (36,057 frames), totaling 95,044 frames — far exceeding UrbanDiff's 34,149 frames (NuScenes). Voxel resolution is $256^2 \times 32$ vs. $192^2 \times 16$.

2. **Scene Structure Estimation Network (SSEN)**:

    - Function: Produces a coarse structural prior of the target 3D scene to accelerate diffusion convergence.
    - Mechanism: A multi-scale feature extraction module (cascaded $3 \times 3 \times 3$ convolutions replacing large kernels) combined with Dimensional Decomposition Residual (DDR) blocks — decomposing a $k^3$ 3D convolution into three sequential steps of $1 \times 1 \times k$, $1 \times k \times 1$, and $k \times 1 \times 1$, reducing parameters from $C_{in} \times C_{out} \times k^3$ to $C_{in} \times C_{out} \times 3k$.
    - Design Motivation: Coarse structural guidance directs the diffusion model toward geometrically plausible outputs from early generation steps.

3. **VAE / Latent Mapping Network**:

    - The encoder downsamples the input voxel grid by a factor of $f=4$ in spatial resolution and is trained jointly with cross-entropy and Lovász-Softmax losses to avoid the blurriness induced by L2.

4. **CymbaDiff Denoising Network — Cylinder Mamba Block**:

    - Function: The core denoising module, which integrates Cartesian and cylindrical coordinate representations to enhance spatial consistency.
    - Mechanism:
        - **Triple Mamba Layer** (Cartesian space): Applies forward $\psi_i^f$, backward $\psi_i^b$, and random inter-slice $\psi_i^u$ SSM scanning along three directions on features $z_{TMB}(t)$ after residual LayerNorm, yielding $\psi_i(z_{TMB}) = \psi_i^f + \psi_i^b + \psi_i^u$.
        - **C-Mamba Layer** (cylindrical space): Reorders voxels by $(θ, r, z)$ (angular-radial-vertical) and applies the same tri-directional scanning $\omega_i(z_{CMB})$, with outputs mapped back to Cartesian space.
        - **Fusion**: $\psi_i^{all} = \text{MLP}(\text{LN}(\psi_i)) + \psi_i + \text{MLP}(\text{LN}(\omega_i)) + \omega_i$.
    - SSM discretization: $h(t) = \bar{A} h(t-1) + \bar{B} z(t),\; y(t) = \bar{C} h(t)$, where $\bar{A} = \exp(\Delta A)$.
    - Design Motivation: Cartesian coordinates preserve precise geometric distance relationships, while cylindrical coordinates provide vehicle-centric angular-radial semantic coherence; the two representations are complementary.

5. **Cross-Scale Contextual Block (CSCB) & Dilated Decomposed Conv Block (DDCB)**:

    - CSCB: Cascaded $3 \times 3 \times 3$ convolutions for multi-scale extraction with skip connections and residuals.
    - DDCB: DDR blocks with dilation rates 1/2/3 to capture multi-scale contextual information.

### Loss & Training
- VAE stage: trained with cross-entropy and Lovász-Softmax losses.
- CymbaDiff stage: diffusion denoising training performed in the VAE latent space.
- Evaluation metrics: 3D FID and MMD, which better measure geometric fidelity in voxel space compared to the 2D FID used in UrbanDiff.

## Key Experimental Results

### Main Results — 3D Semantic Scene Generation

| Dataset | Method | Condition | FID ↓ | MMD ↓ |
|--------|------|------|-------|-------|
| SemanticKITTI | SSD | - | 112.82 | - |
| SemanticKITTI | Semcity | - | 56.55 | - |
| SemanticKITTI | 3D Latent Diffusion | SK+PSA | 165.65 | 0.09 |
| SemanticKITTI | 3D DiT | SK+PSA | 138.86 | 0.08 |
| **SemanticKITTI** | **CymbaDiff** | **SK+PSA** | **40.67** | **0.04** |
| KITTI-360 | 3D Latent Diffusion | SK+PSA | 330.86 | 0.12 |
| KITTI-360 | 3D DiT | SK+PSA | 272.83 | 0.11 |
| **KITTI-360** | **CymbaDiff** | **SK+PSA** | **107.53** | **0.08** |

Cross-dataset generalization: training on SemanticKITTI and directly transferring to KITTI-360 (16 overlapping classes) still yields state-of-the-art performance.

### Ablation Study

| Configuration | FID ↓ | MMD ↓ | Note |
|------|-------|-------|------|
| w/o CSCB | 90.53 | 0.06 | Removing the cross-scale contextual block doubles FID |
| w/o DDCB | 76.57 | 0.06 | Removing the dilated decomposed conv block |
| w/o C-Mamba | 74.09 | 0.05 | Retaining only Cartesian Triple Mamba |
| **CymbaDiff (full)** | **40.67** | **0.04** | Full model |

### 3D Semantic Scene Completion

On the SemanticKITTI validation set, CymbaDiff achieves IoU 43.2% and mIoU 14.6% using only sketch+PSA conditions, surpassing methods such as MonoScene, TPVFormer, NDC-Scene, and OccFormer that utilize real monocular/stereo RGB inputs.

### Key Findings
- CSCB has the largest individual impact on performance (removing it raises FID from 40.67 to 90.53, a 122% increase).
- The cylindrical C-Mamba also contributes significantly in isolation (removing it raises FID by 82%), demonstrating the unique value of cylindrical scanning for outdoor scene modeling.
- Strong zero-shot cross-dataset generalization indicates that CymbaDiff learns general spatial structures.

## Highlights & Insights
- **Dual-path cylindrical+Cartesian Mamba** is the central innovation — it elegantly exploits the vehicle-centric radial characteristics of outdoor driving scenes by performing sequence scanning in cylindrical coordinates to preserve angular continuity, avoiding spatial relationship distortion introduced by Cartesian voxel serialization.
- **DDR decomposition** reduces 3D convolution parameter count to approximately $3k/k^3 \approx 1\%$–$3\%$ of the original, making it well-suited for large-scale 3D scenes.
- **The sketch+PSA conditioning combination** is more accessible than BEV (requiring only a hand-drawn sketch and satellite image at test time), improving practical usability.

## Limitations & Future Work
- Reconstruction quality for small objects (pedestrians, traffic signs, etc.) is relatively weak due to sparse training samples for minor categories.
- PSA pseudo-annotations derived from satellite imagery (~2025) exhibit a temporal gap with ground truth (~2013), potentially introducing semantic drift.
- Validation is limited to the KITTI family of datasets; generalization to different sensor configurations such as NuScenes or Waymo has not been explored.

## Related Work & Insights
- **vs. UrbanDiff**: UrbanDiff employs BEV conditioning with purely Cartesian diffusion at resolution $192^2 \times 16$, achieving FID 291.4; CymbaDiff uses sketch+PSA conditioning with cylindrical Mamba at $256^2 \times 32$, achieving FID 40.67.
- **vs. Semcity/SSD**: Conventional scene generation methods do not support conditional control; CymbaDiff allows users to flexibly specify scene layouts via sketches.
- **vs. SegMamba**: SegMamba targets discriminative SSC tasks; CymbaDiff is the first work to apply SSMs to 3D generative tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Defines a new task, introduces a new benchmark, and presents an original cylindrical Mamba denoising design
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both generation and completion tasks with comprehensive ablations, though validation is limited to the KITTI family
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables
- Value: ⭐⭐⭐⭐ — Open-source dataset and code offer substantial community value, though the contribution leans toward benchmark establishment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](../../ICCV2025/autonomous_driving/decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)

</div>

<!-- RELATED:END -->
