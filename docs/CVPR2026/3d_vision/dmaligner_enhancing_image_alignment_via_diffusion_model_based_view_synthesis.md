---
title: >-
  [Paper Note] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis
description: >-
  [CVPR2026][3D Vision][image alignment] This paper proposes DMAligner, which reformulates image alignment from the traditional optical flow warping paradigm into an "alignment-oriented view synthesis" task. By leveraging a conditional diffusion model to directly generate complete aligned images, and combining a purpose-built DSIA synthetic dataset with a Dynamics-aware Mask Producing (DMP) module, DMAligner effectively eliminates the ghosting and occlusion artifacts inherent to warp-based methods, achieving state-of-the-art performance across multiple benchmarks.
tags:
  - CVPR2026
  - 3D Vision
  - image alignment
  - diffusion model
  - view synthesis
  - dynamic scenes
  - occlusion handling
date: 2026-05-08
content_hash: 51da9c2c0ae0734c
---

# DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis

**Conference**: CVPR2026
**arXiv**: [2602.23022](https://arxiv.org/abs/2602.23022)
**Code**: [boomluo02/DMAligner](https://github.com/boomluo02/DMAligner)
**Area**: 3D Vision
**Keywords**: image alignment, diffusion model, view synthesis, dynamic scenes, occlusion handling

## TL;DR

This paper proposes DMAligner, which reformulates image alignment from the traditional optical flow warping paradigm into an "alignment-oriented view synthesis" task. By leveraging a conditional diffusion model to directly generate complete aligned images, and combining a purpose-built DSIA synthetic dataset with a Dynamics-aware Mask Producing (DMP) module, DMAligner effectively eliminates the ghosting and occlusion artifacts inherent to warp-based methods, achieving state-of-the-art performance across multiple benchmarks.

## Background & Motivation

Image alignment is a fundamental task in computer vision, aiming to register two images captured from different viewpoints or at different times into a unified coordinate system. It is essential for applications such as video stabilization, panoramic stitching, super-resolution, and multi-frame denoising.

The conventional image alignment pipeline typically involves:

1. **Optical flow estimation**: Computing pixel-level motion fields using methods such as RAFT and FlowFormer.
2. **Image warping**: Applying the estimated flow to backward-warp the source image to produce the aligned result.
3. **Post-processing**: Fusing or inpainting artifacts introduced by warping.

This paradigm suffers from two fundamental limitations:

- **Failure in occluded regions**: When certain regions in the target viewpoint are occluded in the source image, optical flow cannot establish valid correspondences, resulting in holes or ghosting artifacts after warping.
- **Dynamic object interference**: Moving objects alter geometric relationships in the scene, causing inaccurate flow estimation and severe ghosting in dynamic regions.

The core motivation is: **Can the indirect "estimate flow → warp" paradigm be bypassed to directly generate a complete aligned image?** Inspired by the powerful generative capability of diffusion models, the authors propose reformulating alignment as a conditional image generation problem.

## Core Problem

- Warp-based alignment methods inevitably produce ghosting artifacts in occluded and dynamic regions.
- Generative approaches require large volumes of high-quality training data, yet existing datasets lack paired annotations for alignment tasks (i.e., ground-truth images at time $t_2$ under camera pose $P_1$).
- The diffusion model must learn to distinguish dynamic foreground from static background to correctly handle moving objects in the scene.

## Method

### Task Reformulation

**Traditional alignment**: Given a reference image $I_{ref}$ (time $t_1$, camera $P_1$) and a source image $I_{src}$ (time $t_2$, camera $P_2$), warp $I_{src}$ into the coordinate system of $P_1$.

**DMAligner's perspective**: Directly generate the target image $I_{gt}$ (time $t_2$, camera $P_1$), i.e., preserving the scene content at $t_2$ while observing from the viewpoint of $P_1$. This is essentially a conditional view synthesis problem.

### DSIA Dataset Construction

To obtain $I_{gt}$ (time $t_2$ + camera $P_1$) — a ground-truth configuration that cannot be captured in real-world settings — the authors construct the **DSIA** (Dynamic Scene Image Alignment) synthetic dataset using Blender:

- **Scene diversity**: 25 character models + 100+ object models + multiple camera motion trajectories.
- **Dynamic scene simulation**: Characters perform actions such as walking, running, and waving; objects undergo translation and rotation.
- **Data rendering**: For each scene, $I_{ref}(t_1, P_1)$, $I_{src}(t_2, P_2)$, and $I_{gt}(t_2, P_1)$ are rendered separately.
- **Dataset scale**: 1,033 scenes in total, producing 30K+ high-quality image triplets.
- **Camera motion types**: Forward, backward, left, right, rotation, and other motions covering diverse alignment scenarios.

### Dynamics-aware Diffusion Training

Training is conducted within the Latent Diffusion Model (LDM) framework:

**Encoding**: The reference image $I_{ref}$ and source image $I_{src}$ are each encoded by a VAE encoder into the latent space, yielding $z_{ref}$ and $z_{src}$.

**Forward diffusion**: Gaussian noise is progressively added to the latent representation $z_{gt}$ of the ground truth:

$$z_t = \sqrt{\bar{\alpha}_t} z_{gt} + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Conditional denoising**: A U-Net takes the concatenated $[z_t, z_{ref}, z_{src}]$ as input and is trained to predict the noise $\epsilon_\theta$:

$$\mathcal{L} = \mathbb{E}_{z_{gt}, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, z_{ref}, z_{src}) \|_2^2 \right]$$

### DMP (Dynamics-aware Mask Producing) Module

The DMP module is a key component that extracts dynamics-aware masks from the intermediate hidden features of the U-Net:

1. **Feature extraction**: Intermediate representations $F_{mid}$ are extracted from multi-scale features in the U-Net decoder.
2. **Mask prediction**: A lightweight convolutional head maps $F_{mid}$ to a binary mask $M_{dyn}$ that distinguishes dynamic foreground from static background.
3. **Dynamic region enhancement**: $M_{dyn}$ provides spatial attention guidance during the denoising process — dynamic regions require stronger generative capacity, while static regions primarily rely on geometric transformation.
4. **Auxiliary loss**: Optical flow inconsistency is used as pseudo-labels to supervise mask prediction:

$$\mathcal{L}_{mask} = \text{BCE}(M_{dyn}, M_{pseudo})$$

The DMP module enhances the network's explicit modeling of dynamic information, enabling the model to adaptively allocate generative resources based on regional characteristics.

### Inference Pipeline

1. Input reference image $I_{ref}$ and source image $I_{src}$.
2. Encode both to the latent space via VAE.
3. Starting from pure Gaussian noise, perform DDIM denoising conditioned on $z_{ref}$ and $z_{src}$.
4. The DMP module provides dynamics-aware guidance throughout the denoising process.
5. Decode via VAE to obtain the final aligned image $I_{align}$.

## Key Experimental Results

### DSIA Test Set

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| RAFT + Warp | 22.31 | 0.782 | 0.189 |
| FlowFormer + Warp | 23.15 | 0.801 | 0.172 |
| LoFTR + Warp | 21.87 | 0.764 | 0.203 |
| **DMAligner** | **27.43** | **0.893** | **0.078** |

DMAligner achieves a PSNR gain of over 4 dB and reduces LPIPS by approximately 50% on the synthetic dataset.

### MPI Sintel Evaluation

| Method | Occluded Region PSNR↑ | Dynamic Region PSNR↑ |
|--------|----------------------|---------------------|
| RAFT + Warp | 18.7 | 17.2 |
| **DMAligner** | **23.1** | **22.6** |

The advantage is particularly pronounced in occluded and dynamic regions, validating the fundamental superiority of the generative approach in these challenging areas.

### DAVIS Video Sequences

Qualitative evaluation on real-world dynamic videos demonstrates that DMAligner produces aligned results free of ghosting artifacts, yielding visually natural outputs even under large motions and severe occlusions.

### Ablation Study

- **Removing the DMP module**: PSNR drops from 27.43 to 26.15 (−1.28), confirming the effectiveness of dynamics-aware guidance.
- **Removing DSIA pretraining**: PSNR drops to 24.87, indicating that synthetic data is critical for learning alignment priors.
- **Training on static scenes only**: Dynamic region PSNR drops by 3.2 dB, validating the necessity of dynamic scene data.

## Highlights & Insights

- **Paradigm shift**: The transition from the indirect "optical flow estimation → warping" paradigm to direct "conditional generation" fundamentally eliminates occlusion and ghosting issues.
- **Elegant DSIA dataset design**: Blender rendering is used to produce $I_{gt}(t_2, P_1)$ — a ground-truth configuration unobtainable in the real world — cleverly resolving the training data challenge.
- **Lightweight yet effective DMP module**: Dynamic masks are extracted from latent features without requiring a dedicated dynamic detection module, providing a plug-and-play enhancement to the diffusion model's handling of dynamic scenes.
- **No optical flow required**: End-to-end image alignment is achieved, avoiding the cascading propagation of optical flow estimation errors.

## Limitations & Future Work

- Diffusion-based inference is relatively slow; DDIM sampling still requires multiple iterations, making real-time deployment impractical.
- The domain gap between the DSIA synthetic dataset and real-world scenes may limit generalization.
- The training set of 30K+ samples is comparatively limited in scale, and scene diversity (25 characters + 100 objects) could be further expanded.
- Alignment performance and efficiency at high resolutions (e.g., 4K) remain unexplored.
- Integration with neural scene representations such as NeRF and 3DGS has not been investigated.

## Related Work & Insights

| Dimension | Traditional Warp Methods | Deep Homography | DMAligner |
|-----------|--------------------------|-----------------|-----------|
| Core operation | Optical flow estimation → pixel warping | Learning global transformation matrix | Conditional diffusion generation |
| Occlusion handling | Cannot handle; produces holes | Relies on inpainting post-processing | Generative filling by design |
| Dynamic objects | Severe ghosting | Assumes static scenes | Explicit modeling via DMP |
| Training data | None required / flow GT | Image pairs + transformation matrices | DSIA synthetic data |
| Inference speed | Fast (single forward pass) | Fast (single forward pass) | Slower (multi-step denoising) |

The fundamental distinction is that DMAligner reformulates alignment as a generation problem, leveraging the generative capacity of diffusion models to address the inherent limitations of warp-based methods in occluded and dynamic regions.

The approach of "reformulating problem X as conditional generation" is broadly applicable and can be extended to other vision tasks that suffer from occlusion challenges, such as occluded regions in stereo matching and video inpainting. The DSIA dataset construction strategy — using a rendering engine to produce ground-truth annotations unobtainable in the real world — is transferable to other tasks requiring specialized annotations. The DMP module's design philosophy of "mining auxiliary information from latent features" is analogous to FGSIM in DiffRefiner, both of which incorporate semantic understanding into the generation process. Future work could explore distillation or consistency model acceleration to bring generative alignment methods toward real-time performance.

## Rating

- **Novelty**: 8/10 — Reformulating alignment as view synthesis represents a clear and well-motivated paradigm shift.
- **Experimental Thoroughness**: 7/10 — Evaluation covers both synthetic and real data, though comparisons with more recent generative baselines are lacking.
- **Writing Quality**: 8/10 — Problem definition is clear, method description is fluent, and DSIA dataset construction is elaborated in detail.
- **Value**: 7/10 — Offers a fresh perspective on alignment, though inference efficiency constraints limit immediate practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](../../ICCV2025/3d_vision/polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](../../ICCV2025/3d_vision/zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](../../NeurIPS2025/3d_vision/va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)

<!-- RELATED:END -->
