---
title: >-
  [Paper Note] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation
description: >-
  [ICLR 2026][3D Vision][Object manipulation] This paper proposes Ctrl&Shift, an end-to-end diffusion framework that decomposes object manipulation into object removal and reference-guided inpainting…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Object manipulation"
  - "diffusion models"
  - "geometric consistency"
  - "camera pose control"
  - "image editing"
date: 2026-05-08
content_hash: a617a1dd6e528467
---

# Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation

**Conference**: ICLR 2026
**arXiv**: [2602.11440](https://arxiv.org/abs/2602.11440)  
**Code**: To be confirmed  
**Area**: 3D Vision / Visual Generation
**Keywords**: Object manipulation, diffusion models, geometric consistency, camera pose control, image editing

## TL;DR
This paper proposes Ctrl&Shift, an end-to-end diffusion framework that decomposes object manipulation into object removal and reference-guided inpainting, and injects relative camera pose control, achieving geometry-consistent fine-grained object manipulation for the first time without relying on explicit 3D reconstruction.

## Background & Motivation

**Background**: Object-level manipulation (repositioning and rotating objects while preserving scene realism) is a fundamental operation in film post-production, AR, and creative editing. Mainstream approaches fall into two camps: geometry-based methods (manipulation after NeRF/3DGS reconstruction) and diffusion-based methods (text/trajectory-conditioned editing).

**Limitations of Prior Work**:
   - Geometry-based methods (NeRF/3DGS) provide precise control but require explicit 3D reconstruction, incurring high per-scene optimization costs and poor generalization
   - Diffusion-based methods (DragAnything/VACE, etc.) generalize well but lack fine-grained geometric control and cannot precisely specify object pose transformations
   - No existing method simultaneously achieves: background preservation, geometry-consistent viewpoint transformation, and user-controllable transformation

**Key Challenge**: A fundamental trade-off exists between geometric precision and generalization capability

**Goal**: Achieve geometry-consistent, fine-grained, controllable object manipulation without explicit 3D reconstruction

**Key Insight**: Rather than lifting content into 3D for editing, inject precise viewpoint control directly into the 2D diffusion process

**Core Idea**: Decompose object manipulation into three sub-tasks—removal, reference-guided inpainting, and camera pose control—and learn them jointly within a unified diffusion framework through multi-task, multi-stage training

## Method

### Overall Architecture
**Inputs**: source image/video frame + reference object image + source mask + target mask + relative camera pose descriptor. **Output**: target frame with the object moved/rotated to the target position/viewpoint. The architecture is based on a ControlNet-style DiT, injecting conditioning signals through a control branch, with camera pose injected via cross-attention.

### Key Designs

1. **Task Decomposition and Multi-Task Training**

    - **Function**: Decompose object manipulation into three separable tasks for joint training
    - **Mechanism**:
        - **Primary task**: Full object manipulation—remove the object from its source location and re-synthesize it at the target location with the target viewpoint
        - **Auxiliary task 1 (object removal)**: Set the reference image to white, target mask to all-zero, and pose outside the frame, learning to cleanly remove objects
        - **Auxiliary task 2 (reference inpainting + camera control)**: Set source mask to all-zero and input to background frame, learning to synthesize the reference object at a specified pose
    - Task weight ratio is 8:1:1; each conditioning signal has an explicit functional role
    - **Design Motivation**: The five conditioning signals (source frame, reference image, source mask, target mask, camera pose) are highly entangled; the multi-task strategy explicitly disentangles the contribution of each signal

2. **Relative Camera Pose Encoding**

    - **Function**: Encode the geometric transformation from source viewpoint to target viewpoint
    - **Mechanism**: A look-at camera model is adopted, parameterizing each viewpoint as $(yaw, pitch, d, r_x, r_y)$. The axis-angle representation of the relative rotation matrix $\text{aa}(\mathbf{R}_{rel})$, relative translation $\mathbf{t}_{rel}$, and NDC offset $(\Delta r_x, \Delta r_y)$ are concatenated into an 8-dimensional descriptor $\mathbf{f} \in \mathbb{R}^8$
    - After Fourier positional encoding and MLP projection, this is mapped to 8 tokens ($d=4096$) and injected into the DiT via cross-attention
    - **Design Motivation**: Relative pose is more intuitive than absolute pose (adjustments are made relative to the input frame, analogous to dragging), avoiding the difficulty of defining a canonical absolute pose

3. **Mask Encoding Strategy**

    - **Function**: Align binary masks to the VAE latent space
    - **Mechanism**: Instead of encoding masks through the VAE (which risks treating binary semantics as appearance), space-to-depth (pixel unshuffle) is applied to directly downsample the mask to match the VAE stride
    - At inference time, the target mask is approximated by scaling and translating the bounding box of the source mask

4. **Two-Stage Training**

    - **Stage I (Synthetic Data)**: Pre-training on ~2M synthetic image pairs with white backgrounds and random camera poses, learning object priors and pose representations; both the backbone and control branch are jointly updated
    - **Stage II (Real Data)**: Fine-tuning on 100K high-quality real image/video pairs with the backbone frozen and only the control branch updated, focusing on background preservation and photorealism

5. **Data Construction Pipeline**

    - **Function**: Automatically construct training pairs with pose annotations from real images
    - **Mechanism**: Hunyuan3D-2 reconstructs the object mesh → differentiable rendering estimates the source camera pose (filtered by IoU ≥ 0.90) → target poses are sampled and rendered → MiniMax-Remover obtains clean backgrounds → an object pasting network performs harmonized compositing

### Loss & Training
Flow-matching training is adopted with a linear path $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\boldsymbol{\varepsilon}$ and a velocity matching loss $\|\mathbf{v}_\theta(\mathbf{z}_t, \mathbf{c}, t) - \mathbf{v}^*(\mathbf{z}_t, t)\|_2^2$.

## Key Experimental Results

### Main Results

Zero-shot evaluation on ObjectMover-A:

| Method | PSNR↑ | DINO↑ | CLIP↑ | DreamSim↓ |
|--------|-------|-------|-------|-----------|
| ObjectMover | 25.27 | 85.07 | 93.16 | 0.142 |
| **Ctrl&Shift** | **28.69** | **88.07** | **93.58** | **0.075** |

GeoEditBench (proposed benchmark for geometry-aware editing evaluation):

| Method | PSNR↑ | DINO↑ | Pose MAPE↓ | Obj IoU↑ |
|--------|-------|-------|-----------|----------|
| VACE | 24.32 | 75.38 | 30.56% | 0.72 |
| Nano-Banana | 26.38 | 78.05 | 24.36% | 0.78 |
| **Ctrl&Shift** | **28.71** | **85.23** | **17.70%** | **0.83** |

### Ablation Study
- Removing Stage 1: Pose MAPE increases from 17.70% to 32.50%, severely degrading geometric understanding
- Removing Stage 2: PSNR drops from 28.71 to 24.83, with degraded background preservation and visual quality
- Removing Auxiliary Task 1: CLIP-Score drops to 86.32, impairing semantic consistency
- Removing Auxiliary Task 2: Obj IoU drops to 0.65 and Pose MAPE rises to 28.60%, with object-level precision most severely affected

## Highlights & Insights
- A key conceptual breakthrough: geometry-consistent object manipulation without 3D reconstruction
- The multi-task decomposition strategy is elegant, enabling the model to learn disentangled signals from each task
- The data construction pipeline is scalable and supports real-world images and videos
- GeoEditBench provides a systematic evaluation protocol for geometry-aware editing

## Limitations & Future Work
- The approximate target mask estimation at inference (bounding box scaling and translation) may be inaccurate under extreme transformations
- Based on the Wan-1.3B backbone, the model scale is modest, which may limit performance on complex scenes
- Currently supports only single-object manipulation; multi-object collaborative editing remains unexplored
- Data construction relies on Hunyuan3D-2 and an object pasting model, inheriting errors from these components
- Video manipulation capability is demonstrated but lacks sufficient quantitative evaluation

## Related Work & Insights
- vs. DragAnything: A trajectory-conditioned diffusion method with poor generalization and no pose control
- vs. VACE: Preserves background well but effectively translates the entire frame rather than genuinely manipulating objects
- vs. Nano-Banana/Qwen-Image-Edit: High generation quality but imprecise camera pose control driven by text instructions
- vs. 3DiT/GeoDiffuser: Relies on 3D reconstruction or geometric conditions, limiting generalization
- vs. ObjectMover: A video-prior method; Ctrl&Shift achieves +3.42 PSNR and halves DreamSim

The approach of injecting 3D geometric control without performing 3D reconstruction is generalizable to other editing tasks. The multi-task disentanglement training strategy is worth adopting in other multi-condition generation settings. Relative pose encoding is better suited to interactive editing scenarios than absolute pose.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (conceptual innovation in task decomposition + pose injection)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (multiple benchmarks + ablations + proposed benchmark)
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐⭐ (first to unify geometric precision and diffusion generalization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](../../CVPR2026/3d_vision/qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)
- [\[AAAI 2026\] Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine](../../AAAI2026/3d_vision/free-form_scene_editor_enabling_multi-round_object_manipulation_like_in_a_3d_eng.md)
- [\[AAAI 2026\] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](../../AAAI2026/3d_vision/4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)

</div>

<!-- RELATED:END -->
