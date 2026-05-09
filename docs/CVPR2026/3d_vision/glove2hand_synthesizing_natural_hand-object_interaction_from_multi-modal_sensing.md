---
title: >-
  [Paper Note] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves
description: >-
  [CVPR 2026][3D Vision][hand-object interaction] This paper proposes the Glove2Hand framework, which translates egocentric videos of instrumented sensing gloves into photorealistic bare-hand videos while preserving tactile and IMU signals. It also introduces HandSense, the first multi-modal hand-object interaction dataset, and demonstrates significant improvements on downstream bare-hand contact estimation and occluded hand tracking.
tags:
  - CVPR 2026
  - 3D Vision
  - hand-object interaction
  - sensing gloves
  - video translation
  - 3D Gaussian hand model
  - diffusion model
date: 2026-05-08
content_hash: acacf4193b26ed70
---

# Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves

**Conference**: CVPR 2026
**arXiv**: [2603.20850](https://arxiv.org/abs/2603.20850)
**Code**: [https://mlzxy.github.io/glove2hand](https://mlzxy.github.io/glove2hand)
**Area**: 3D Vision
**Keywords**: hand-object interaction, sensing gloves, video translation, 3D Gaussian hand model, diffusion model

## TL;DR
This paper proposes the Glove2Hand framework, which translates egocentric videos of instrumented sensing gloves into photorealistic bare-hand videos while preserving tactile and IMU signals. It also introduces HandSense, the first multi-modal hand-object interaction dataset, and demonstrates significant improvements on downstream bare-hand contact estimation and occluded hand tracking.

## Background & Motivation

**State of the Field**: Hand-object interaction (HOI) understanding is a foundational problem in computer vision, robotics, and AR/VR. Mainstream approaches rely on egocentric video to develop data-driven algorithms, but these systems are predominantly unimodal, relying solely on visual input.

**Limitations of Prior Work**: Vision-only HOI data suffers from two fundamental deficiencies: (1) physical quantities such as force and contact are unavailable—existing methods like ContactPose can only estimate binary fingertip contact and are limited to pre-scanned rigid objects; (2) restricted viewpoints cause severe hand occlusion, while multi-camera studio setups are infeasible in the wild. Although sensing gloves can provide IMU and tactile signals, the large appearance gap between gloves and bare hands prevents visual models trained on glove data from generalizing to bare-hand tasks.

**Root Cause**: Sensing gloves offer rich physical signals but introduce a domain gap, whereas bare-hand video provides favorable visual appearance but lacks physical information—these two desiderata are fundamentally at odds.

**Paper Goals**: How can sensing glove videos be translated into photorealistic bare-hand videos while retaining tactile/IMU signals, so that physical information can be leveraged for bare-hand learning tasks? Specific sub-problems include: (1) achieving cross-frame spatiotemporal consistency rather than processing only static images; and (2) handling complex interactions with unknown and non-rigid objects.

**Starting Point**: The key observation is that despite their large appearance difference, gloves and bare hands share the same skeletal structure (hand pose). The problem can therefore be decomposed into two steps: first convert the glove video into a temporally consistent in-air bare-hand sequence (using 3D reconstruction to enforce consistency), then embed the bare hand into the scene and restore interaction details (using a diffusion model to ensure flexibility).

**Core Idea**: Combine the spatiotemporal consistency of a 3D Gaussian hand model with the generative flexibility of a diffusion hand restorer to achieve sensing-glove-to-bare-hand video translation, while preserving multi-modal sensing signals.

## Method

### Overall Architecture
The input is an egocentric video of a hand wearing a sensing glove, together with the corresponding hand pose and glove/object masks. The output is a high-quality video with appearance consistent with a bare hand. The pipeline consists of two stages: (1) a 3D Gaussian hand model renders a temporally consistent bare-hand-only sequence from the hand pose; (2) a diffusion hand restorer seamlessly composites the rendered hand into the scene, repairing the hand-object interaction boundary and the wrist connection. During training, bare-hand videos are used for supervision; during inference, an optical-flow-based background inpainter first erases the glove region, after which the rendered hand is composited.

### Key Designs

1. **Surface-Grounded 3D Gaussian Hand**:

    - **Function**: Render temporally consistent, relightable bare-hand images from a given hand pose.
    - **Mechanism**: 3D Gaussian distributions are defined directly on the triangular faces of a canonical hand mesh. Each Gaussian is parameterized by barycentric coordinate weights $\mathbf{w}$, 2D scale $\mathbf{s}$, and rotation $\phi$. When the hand deforms, only the mesh triangles are transformed and the Gaussians are recomputed accordingly, avoiding per-Gaussian linear blend skinning. Gaussian ellipses are mapped via the 2D affine deformation gradient $\mathbf{A}=\mathbf{M}_{\text{deform}}\mathbf{M}_{\text{canon}}^{-1}$. Compared with 2DGS, which defines Gaussians in 3D space and regularizes them into surfaces, the proposed method directly uses mesh faces as a stronger geometric prior.
    - **Design Motivation**: The canonical mesh provides a strong geometric prior but lacks learning flexibility, while Gaussian splatting is flexible but unstructured. Face-anchoring unifies the advantages of both; moreover, mesh surface normals naturally support illumination estimation.

2. **Relightable Hand Gaussians**:

    - **Function**: Handle dynamically varying illumination conditions in egocentric scenes.
    - **Mechanism**: A small MLP predicts spherical harmonics coefficients $\mathbf{l}$ conditioned on hand pose $\mathbf{P}$, and color is computed as the product of albedo and lighting $\mathbf{c}\odot\text{SH}(\mathbf{l},\mathbf{n})$. Two independent environment maps are predicted separately for the palm and the back of the hand. Since normals are derived from mesh geometry rather than the Gaussians themselves, the albedo–illumination ambiguity is substantially alleviated.
    - **Design Motivation**: LumiGauss assumes a single static environment map and is unsuitable for the dynamic illumination of egocentric scenes; face-anchored Gaussians provide a consistent source of surface normals.

3. **Diffusion Hand Restorer**:

    - **Function**: Seamlessly composite the rendered bare-hand sequence into the scene, repairing hand-object interactions and the wrist connection.
    - **Mechanism**: Built on ControlNet + AnimateDiff and trained on bare-hand videos. The rendered hand is overlaid onto the original frame (with dilated mask and occluded wrist region) as a conditioning input, and the network learns to recover the original video from this corrupted input. During inference, SAM-2 and Grounding DINO are used to detect glove/object masks; Propainter inpaints the glove region via optical flow while preserving object pixels; the rendered hand is then overlaid and fed into the diffusion restorer.
    - **Design Motivation**: Directly compositing the rendered hand produces physically implausible interactions (e.g., penetration or floating), unnatural wrist transitions, and glove residual artifacts. Using a diffusion model to handle objects and background in pixel space is more flexible than explicitly modeling object geometry.

### Loss & Training
The 3D Gaussian hand model is trained with image reconstruction loss via differentiable rendering, with a separate model optimized per subject. After freezing the subject-specific Gaussian hand, a unified diffusion hand restorer is trained. Training data are sourced from HOT3D and HandSense.

## Key Experimental Results

### Main Results

| Method | FID ↓ | FVD ↓ | FVD-long ↓ |
|--------|-------|-------|------------|
| HandRefiner | 35.5 | 24.2 | 29.7 |
| BrushNet | 37.9 | 34.5 | 40.4 |
| Pix2Pix | 38.6 | 24.7 | 31.4 |
| **Glove2Hand (Ours)** | **30.1** | **19.5** | **24.5** |

### Ablation Study

| Configuration | FID ↓ | FVD ↓ | FVD-long ↓ |
|---------------|-------|-------|------------|
| 2DGS | 91.1 | 50.0 | 62.9 |
| +Surface Grounding | 60.3 | 35.1 | 46.6 |
| +Relightable | 56.7 | 30.7 | 40.2 |
| +Diffusion | 32.3 | 19.8 | 22.7 |
| +Glove Removal | 31.2 | 20.9 | 25.0 |
| +Object Mask (Full) | 30.1 | 19.5 | 24.5 |

### Downstream Task: Contact Estimation

| Training Data | Contact IoU (%) | Precision (%) | Recall (%) |
|---------------|----------------|---------------|------------|
| Glove only | 71.5 | 82.8 | 83.9 |
| G2H only | 75.6 | 90.6 | 82.0 |
| Hand only | 85.3 | 90.0 | 94.2 |
| **Hand + G2H** | **88.2** | **92.6** | **94.9** |

### Downstream Task: Occluded Hand Tracking

| Method | MKPE (Occ) ↓ | MKPE (All) ↓ |
|--------|-------------|-------------|
| UmeTrack | 19.2 | 19.5 |
| UmeTrack + Glove | 27.2 | 26.5 |
| **UmeTrack + G2H** | **16.6** | **17.8** |

### Key Findings
- Surface Grounding and the Diffusion Restorer contribute most significantly, yielding large FID reductions from 91.1→60.3 and 56.7→32.3, respectively.
- Training the hand tracker directly on glove data actually degrades performance (19.5→26.5), confirming the severity of the domain gap.
- Combining synthesized bare-hand videos with real bare-hand data for contact estimator training achieves the best results, validating the framework's value as a data generation engine.
- A human perceptual study shows that the generated hands are nearly indistinguishable from real hands in static images.

## Highlights & Insights
- By aligning hardware sensing (tactile + IMU) with visual generation, this work opens a new data generation paradigm of "sensors as annotation-free ground truth," which is transferable to other domains requiring expensive annotation.
- The surface-grounded Gaussian design is elegant: a single simple geometric prior simultaneously addresses spatiotemporal consistency, relighting, and deformation, with minimal implementation complexity.
- The framework learns a glove-to-hand mapping from unpaired data, avoiding the collection cost of paired annotations.

## Limitations & Future Work
- A separate 3D Gaussian hand model must be optimized per subject, limiting direct generalization to new users.
- The automated pipeline depends on SAM-2 and Grounding DINO, which may fail under extreme occlusion or complex backgrounds.
- The dataset scale is limited (5 subjects), and generalization requires validation at larger scale.
- Although interactions with non-rigid objects are handled flexibly in pixel space, physical plausibility (e.g., force consistency) is not guaranteed.

## Related Work & Insights
- **vs. HandRefiner**: HandRefiner is a diffusion-only method focused on single-frame hand restoration; the proposed method incorporates 3D reconstruction to enforce spatiotemporal consistency, achieving an FID improvement of 5.4 points.
- **vs. hand avatar methods (HandSplat, etc.)**: Conventional avatars require dense multi-view capture and controlled illumination; the surface-grounded design proposed here is suited to sparse egocentric cameras and dynamic lighting.
- **vs. video translation methods (MeDM, etc.)**: General-purpose video translation cannot handle the large embodiment gap between gloves and bare hands; the proposed method decouples the problem by exploiting the shared skeletal structure.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined architecture of surface-anchored Gaussians and a diffusion restorer is novel, though each individual component is a well-motivated integration of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation covers video quality metrics, human perceptual study, two downstream tasks, and a comprehensive ablation study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The problem is clearly defined, the motivation chain is complete, and the figures are intuitive.
- **Value**: ⭐⭐⭐⭐ The framework introduces a new data generation paradigm for the HOI community, and the HandSense dataset holds long-term value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[CVPR 2026\] PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](pad-hand_physics-aware_diffusion_for_hand_motion_recovery.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)

<!-- RELATED:END -->
