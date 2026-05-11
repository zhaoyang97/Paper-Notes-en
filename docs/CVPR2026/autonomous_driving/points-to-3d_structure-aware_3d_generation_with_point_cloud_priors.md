---
title: >-
  [Paper Note] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors
description: >-
  [CVPR 2026][Autonomous Driving][Point Cloud Priors] This paper proposes Points-to-3D, which encodes visible-region point clouds into TRELLIS's sparse structure latent (SS latent) and completes unobserved regions via a mask-aware inpainting network. Combined with a two-stage sampling strategy of structure completion followed by boundary refinement, the method achieves geometry-controllable, high-fidelity 3D asset/scene generation, attaining an F-Score of 0.964 on Toys4K (0.998 for visible regions).
tags:
  - CVPR 2026
  - Autonomous Driving
  - Point Cloud Priors
  - 3D Generation
  - Diffusion Models
  - Structure Completion
  - Geometry-Controllable
date: 2026-05-08
content_hash: 0d5965db7dfc71a6
---

# Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors

**Conference**: CVPR 2026
**arXiv**: [2603.18782](https://arxiv.org/abs/2603.18782)
**Code**: [Project Page](https://jiatongxia.github.io/points2-3D/)
**Area**: Autonomous Driving / 3D Generation
**Keywords**: Point Cloud Priors, 3D Generation, Diffusion Models, Structure Completion, Geometry-Controllable

## TL;DR

This paper proposes Points-to-3D, which encodes visible-region point clouds into TRELLIS's sparse structure latent (SS latent) and completes unobserved regions via a mask-aware inpainting network. Combined with a two-stage sampling strategy of structure completion followed by boundary refinement, the method achieves geometry-controllable, high-fidelity 3D asset/scene generation, attaining an F-Score of 0.964 on Toys4K (0.998 for visible regions).

## Background & Motivation

**Background**: 3D generative models (e.g., TRELLIS, GaussianAnything) can synthesize realistic 3D assets from images or text, but they are conditioned on 2D images or text and lack direct constraints from real 3D geometry, making geometric accuracy uncontrollable.

**An Overlooked Information Source**: In autonomous driving, robotics, and similar settings, visible-region point clouds are readily available—from LiDAR, structured light, or feedforward predictors such as VGGT. These point clouds provide explicit geometric constraints that current generative frameworks cannot exploit.

**Technical Limitation**: TRELLIS initializes SS latents from pure Gaussian noise, guided only by image/text embeddings, and cannot be anchored to real 3D observations. Naively injecting point clouds as an additional conditioning signal is insufficient—structural priors must be embedded into the latent space itself.

**Mechanism**: Point-cloud-guided 3D generation is reformulated as a **latent inpainting** problem: the visible region is encoded as a fixed constraint, while the unobserved region is synthesized by a completion network.

## Method

### Overall Architecture

Input visible-region point cloud $\mathbf{P}$ → voxelized to $\mathbf{M}' \in \{0,1\}^{N \times N \times N}$ ($N=64$) → encoded by TRELLIS's SS VAE into a partial SS latent $\mathbf{q}_{\text{vis}} = \mathcal{E}_s(\mathbf{M}')$ → visible/invisible regions marked by occupancy mask $\mathbf{m}_s$ → unobserved voxels filled with noise to produce the combined input $\mathbf{q}_{\text{comb}}$ → Inpainting Flow Transformer completes the latent → two-stage sampling yields the complete SS latent → TRELLIS downstream SLAT generation and rendering.

### Key Designs

1. **Point-Cloud-Prior-Driven Latent Initialization**:

    - **Function**: Encodes the visible point cloud into TRELLIS's SS latent space as the generation starting point, replacing pure-noise initialization.
    - **Core Formula**:
    $$\mathbf{q}_{\text{comb}} = \mathbf{m}_s \odot \mathbf{q}_{\text{vis}} + (1 - \mathbf{m}_s) \odot \boldsymbol{\epsilon}_s$$
      where $\mathbf{q}_{\text{vis}} = \mathcal{E}_s(\mathbf{M}')$ is the encoded visible-region latent and $\mathbf{m}_s$ is the occupancy mask downsampled to latent resolution ($r=16$).
    - **Design Motivation**: Directly anchors visible geometry in the latent space, constraining the diffusion process with real 3D observations rather than relying solely on implicit guidance from images or text.

2. **Mask-Aware Structure Completion Network $\mathcal{G}_{inp}$**:

    - **Function**: Fine-tuned from TRELLIS's Structure Flow Transformer to learn geometry inference from visible to unobserved regions.
    - **Input Design**: The mask $\mathbf{m}_s$ is concatenated with $\mathbf{q}_{\text{comb}}$ along the channel dimension, and the original input layer is replaced to accommodate the new channel count $(c_s + c_m)$.
    - **Training Data Construction**: Depth maps are rendered from $T=24$ viewpoints for each complete 3D asset; visible point clouds per viewpoint are extracted via depth consistency checking (threshold $\tau$), constructing training pairs $(\mathbf{q}_{\text{comb}}^t, \mathbf{m}_s^t, \mathbf{I}_t, \mathbf{q}_{\text{gt}})$.
    - **Training Objective**: Conditional Flow Matching loss:
    $$\mathcal{L}_{CFM} = \mathbb{E}_{t, \mathbf{q}_{\text{gt}}, \epsilon} \|\mathcal{G}_{inp}(\mathbf{x}_{\text{inp}}, t) - (\epsilon - \mathbf{q}_{\text{gt}})\|_2^2$$

3. **Two-Stage Sampling Strategy (Staged Sampling)**:

    - **Function**: Divides $t$ sampling steps into a structure completion phase ($s$ steps) and a boundary refinement phase ($t-s$ steps).
    - **Structure Completion Phase**: At each step, $\mathbf{q}_{\text{pred}}$ is reconstructed and re-concatenated with mask $\mathbf{m}_s$, iterating for $s$ steps while keeping the visible region anchored.
    - **Boundary Refinement Phase**: The mask is replaced with all-ones ($\mathbf{m}_1$), switching to standard denoising to repair geometric hole artifacts at inpainting boundaries.
    - **Design Motivation**: Pure inpainting produces geometric holes at visible/invisible region boundaries due to downsampling information loss; the refinement stage repairs boundaries without disrupting global structure.
    - **Optimal Configuration**: $s=25$, $t-s=25$ (50 steps total).

### Loss & Training

- **Loss**: Conditional Flow Matching Loss, computed solely in the SS latent space.
- **Training**: Conducted on the combined 3D-FUTURE + HSSD + ABO datasets for 20k iterations, batch size 8, on 4× A100 GPUs.
- **Per-object sampling**: $S=50{,}000$ points, $T=24$ viewpoints, constructing visible/complete latent pairs.
- **Inference**: Supports two input modalities: (a) real sensor point clouds; (b) point clouds estimated from a single image via VGGT.

## Key Experimental Results

### Main Results

**Single-Object Generation (Toys4K)**:

| Method | PSNR↑ | SSIM(%)↑ | LPIPS↓ | CD↓ | F-Score↑ |
|---|---|---|---|---|---|
| TRELLIS | 21.94 | 91.46 | 0.105 | 0.034 | 0.832 |
| SAM3D | 22.42 | 91.45 | 0.111 | 0.033 | 0.835 |
| Points-to-3D (VGGT) | 22.55 | 92.09 | 0.088 | 0.024 | 0.881 |
| **Points-to-3D (P.C.)** | **22.91** | **92.83** | **0.070** | **0.013** | **0.964** |

**Scene-Level Generation (3D-FRONT)**:

| Method | PSNR↑ | LPIPS↓ | CD↓ | F-Score↑ |
|---|---|---|---|---|
| TRELLIS | 18.21 | 0.239 | 0.094 | 0.478 |
| MIDI | 19.23 | 0.166 | 0.075 | 0.513 |
| **Points-to-3D (P.C.)** | **21.63** | **0.124** | **0.025** | **0.886** |

### Ablation Study

| Configuration (Inp./Ref. Steps) | CD↓ | F-Score↑ | PSNR-N↑ | Note |
|---|---|---|---|---|
| 50/0 (pure inpainting) | 0.014 | 0.960 | 25.88 | Boundary holes present |
| 25/25 (optimal) | **0.013** | **0.963** | **27.10** | Holes eliminated |
| 10/40 | 0.014 | 0.961 | 26.72 | Insufficient inpainting |

### Key Findings

- Visible-region F-Score reaches **0.998** with CD of only **0.007**, nearly perfectly preserving the input prior.
- Even with noisy VGGT-estimated point clouds, the method significantly outperforms all baselines, demonstrating framework robustness.
- SAM3D also uses point clouds but integrates them indirectly via attention, failing to achieve explicit geometric control.
- Improvements are especially pronounced at the scene level (F-Score: 0.513 → 0.886), highlighting the value of point cloud priors for complex multi-object scenes.

## Highlights & Insights

1. **Core Innovation**: Reformulating point-cloud-conditioned generation as a latent-space inpainting problem is a concise, elegant, and effective paradigm.
2. **Two-Stage Sampling**: The decoupled structure completion and boundary refinement strategy elegantly resolves inpainting boundary artifacts.
3. **Flexible Input**: Supporting both real sensor point clouds and VGGT-predicted point clouds covers scenarios with and without real 3D priors.
4. **Plug-and-Play**: Built on the TRELLIS framework with minimal architectural changes—only the input layer and training data are modified.

## Limitations & Future Work

1. A performance gap remains between VGGT-predicted and real point clouds (F-Score: 0.881 vs. 0.964), limited by feedforward prediction accuracy.
2. The voxelization resolution $N=64$ may restrict fine-grained geometric representation.
3. Evaluation is limited to objects and indoor scenes; large-scale outdoor settings (e.g., autonomous driving point clouds) remain unvalidated.

## Related Work & Insights

- **TRELLIS**: Provides the SS latent framework that enables point cloud injection.
- **VGGT**: A feedforward 3D predictor that supplies point cloud inputs for scenes without active sensors.
- **VoxHammer**: Also leverages 3D priors but adopts a 3D inversion strategy, performing poorly when completing unobserved regions.
- **Insight**: The paradigm of "encoding priors as latent initialization + inpainting completion" is potentially generalizable to other conditional generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Latent-space inpainting perspective is novel with a clear paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers objects, scenes, real images, and ablations comprehensively)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured; problem–solution–result logic flows smoothly)
- Value: ⭐⭐⭐⭐⭐ (Establishes a paradigm for introducing explicit 3D priors into 3D generation; broad application prospects for LiDAR/RGBD-conditioned generation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](igasa_integrated_geometry-aware_and_skip-attention_modules_for_enhanced_point_cl.md)
- [\[CVPR 2026\] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)
- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](../../ICCV2025/autonomous_driving/trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[AAAI 2026\] Unleashing Semantic and Geometric Priors for 3D Scene Completion](../../AAAI2026/autonomous_driving/unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)

</div>

<!-- RELATED:END -->
