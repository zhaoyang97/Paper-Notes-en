---
title: >-
  [Paper Note] NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction
description: >-
  [ICCV 2025][Human Understanding][garment reconstruction] This paper proposes NGD, a neural gradient-based deformation method that decomposes the Jacobian field into a frame-invariant static component and a frame-dependent dynamic component. Combined with an adaptive remeshing strategy, NGD reconstructs high-fidelity dynamic garment geometry and appearance from monocular video, significantly outperforming existing SOTA methods on challenging scenarios such as loose-fitting gar…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "garment reconstruction"
  - "neural Jacobian field"
  - "adaptive remeshing"
  - "monocular video"
  - "differentiable rendering"
date: 2026-05-08
content_hash: 83956842cb764616
---

# NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2508.17712](https://arxiv.org/abs/2508.17712)  
**Code**: [https://github.com/astonishingwolf/NGD/](https://github.com/astonishingwolf/NGD/)  
**Area**: Human Understanding
**Keywords**: garment reconstruction, neural Jacobian field, adaptive remeshing, monocular video, differentiable rendering

## TL;DR

This paper proposes NGD, a neural gradient-based deformation method that decomposes the Jacobian field into a frame-invariant static component and a frame-dependent dynamic component. Combined with an adaptive remeshing strategy, NGD reconstructs high-fidelity dynamic garment geometry and appearance from monocular video, significantly outperforming existing SOTA methods on challenging scenarios such as loose-fitting garments.

## Background & Motivation

Reconstructing dynamic garments from monocular video is an important yet challenging task. Existing methods fall into two main categories, each with notable limitations:

**Implicit surface methods** (e.g., SCARF based on NeRF, REC-MV): Volume rendering yields limited geometric quality, producing overly smooth surfaces that lose high-frequency details.

**Explicit template-based methods** (e.g., Pergamo, DGarments): Deformation is performed via vertex displacements, but direct displacement tends to introduce jagged surface artifacts and requires additional regularization that over-smooths the surface; fixed templates also cannot model dynamic topology changes (e.g., skirt folds).

Core motivation: A method is needed that preserves high-frequency details (e.g., wrinkles, folds) while handling large deformations of loose-fitting garments. NGD adopts Jacobian field parameterization instead of vertex displacements to avoid local discontinuities, and employs adaptive remeshing to increase resolution in detail-rich regions.

## Method

### Overall Architecture

NGD consists of a geometry reconstruction module and an appearance reconstruction module. The geometry component introduces a neural Jacobian field-based deformation parameterization, decomposing deformation into static (global shape) and dynamic (per-frame local deformation) components. The appearance component learns a static base texture and a dynamic texture to capture per-frame lighting and shading effects.

### Key Designs

1. **Intrinsic Deformation Fields**: The per-frame Jacobian field $J_t^F$ is decomposed into two sub-fields:

    - **Static Jacobian field $J^S \in \mathbb{R}^{M \times 3 \times 3}$**: Frame-invariant, defined at each face center of the base mesh, capturing global garment shape features (e.g., neckline, skirt hem), optimized directly across all frames.
    - **Dynamic Jacobian field $J_t^D \in \mathbb{R}^{M \times 3 \times 3}$**: Frame-dependent, predicted by a neural network $f_G = f_\Theta \circ f_\varphi$, taking face centers, face normals, and PCA-encoded pose parameters $\gamma(\theta_t)$ as input.
    - The final $J_t^F = J^S + J_t^D$; a Poisson solve yields the canonical-space garment mesh $M_t^C$, which is then transformed via skinning to produce the reposed mesh $M_t^P$.
    - Core advantage: The Jacobian field combined with Poisson solving guarantees global smoothness, eliminating the jagged artifacts associated with vertex displacements.

2. **Gradient-Based Adaptive Remeshing**:

    - **Edge selection**: The gradient $\mathcal{G}(p)$ of the diffuse rendering loss with respect to each pixel is computed and aggregated to each face to obtain face-level gradient values; faces in the top quantile by gradient magnitude are selected.
    - **Pruning**: Faces with edge lengths below threshold $\delta_{\text{length}}$ are excluded; this threshold decays linearly during training.
    - **Remeshing operations**: Edge splitting and edge flipping are applied to selected edges, producing a new topology $M_r^B$.
    - Attribute recomputation: The static Jacobian field, optimizer moments, and skinning weights are recomputed via k-NN interpolation.
    - Core significance: Allows high-frequency detail regions (wrinkles, pockets) to obtain higher resolution, and enables free deformation of the template to model extremely loose garments.

3. **Appearance Reconstruction Module**:

    - Static texture $T^S \in \mathbb{R}^{q \times q \times 3}$: A directly optimized, frame-invariant base texture.
    - Dynamic texture $T_t^D$: Predicted by an MLP $f_T$ conditioned on hash-encoded UV coordinates and pose parameters.
    - $T_t^F = T^S + T_t^D$, optimized via color loss and SSIM loss through differentiable rendering.
    - Innovation: Linearly decaying Gaussian noise is introduced to pose parameters to prevent overfitting.

### Loss & Training

Geometry loss:
$$\mathcal{L}_{geo} = \lambda_1 \mathcal{L}_{render} + \lambda_2 \mathcal{L}_{mask} + \lambda_3 \mathcal{L}_{reg} + \lambda_4 \mathcal{L}_{depth}$$

- $\mathcal{L}_{render}$: Huber + SSIM loss on diffuse images (diffuse images rather than normal images are used as supervision to avoid normal ambiguity at grazing view angles).
- $\mathcal{L}_{reg}$: Regularization loss constraining the Jacobian to remain close to the identity matrix.
- $\mathcal{L}_{mask}$: Segmentation mask loss.
- $\mathcal{L}_{depth}$: Depth ordering loss.

Training strategy:
- NVDiffrast is used as the differentiable rasterizer; training runs on a single RTX 4090 GPU.
- A 100-frame sequence requires approximately 2.5 hours of training.
- Two-stage training: a warmup stage optimizes only the static $J^S$ and $T^S$, after which dynamic components are introduced for joint optimization.
- Adaptive remeshing is performed at fixed intervals.
- To address local minima, exponentially decaying noise is added to final skinned mesh vertices, prioritizing global geometry in early training.

## Key Experimental Results

### Main Results (Tables)

Quantitative geometry reconstruction evaluation on the 4D-Dress dataset:

| Method | Chamfer Distance ($\times 10^3$) ↓ | Normal Consistency ↑ |
|--------|-----------------------------------|--------------------|
| | Seq 123 / 148 / 169 / 185 / 187 / **Avg** | Seq 123 / 148 / 169 / 185 / 187 / **Avg** |
| SCARF | 8.622 / - / 6.507 / 2.423 / 3.261 / 5.203 | 0.915 / - / 0.872 / 0.837 / 0.753 / 0.844 |
| DGarment | 0.076 / 0.863 / 0.154 / 0.431 / 1.722 / 0.649 | 0.904 / 0.755 / 0.872 / 0.856 / 0.777 / 0.833 |
| **NGD (Ours)** | **0.050** / **0.660** / **0.127** / 0.393 / **0.923** / **0.431** | **0.934** / **0.766** / **0.891** / **0.879** / **0.794** / **0.853** |

Novel view synthesis quantitative evaluation:

| Method | Seq 123 PSNR/SSIM/LPIPS | Seq 169 | Seq 185 | Seq 187 |
|--------|------------------------|---------|---------|---------|
| SCARF | 43.02 / 0.992 / 0.018 | 45.01 / 0.992 / 0.026 | 33.82 / 0.986 / 0.025 | 25.32 / 0.918 / 0.083 |
| **Ours** | **46.78** / **0.998** / **0.008** | **47.91** / **0.996** / **0.014** | **35.21** / **0.990** / **0.017** | **25.85** / **0.948** / **0.040** |

### Ablation Study (Tables)

Ablation of design choices (4D-Dress dataset, averaged over 5 sequences):

| Configuration | CD ↓ (Avg) | NC ↑ (Avg) |
|---------------|------------|------------|
| **NGD (Full)** | **0.431** | **0.853** |
| w/o remeshing | 0.441 | 0.850 |
| w normals (replacing diffuse) | 0.554 | 0.832 |

### Key Findings

- NGD improves CD over DGarment by an average of 33.6% (0.649→0.431) and NC by 2.4%.
- The advantage is most pronounced on loose-fitting garments (Seq 187, long skirt): CD drops from 1.722 to 0.923 (46.4% improvement).
- The quantitative gain from adaptive remeshing is marginal (CD: 0.441→0.431), but qualitative differences are substantial — preservation of complex wrinkles and curved surfaces improves significantly.
- Diffuse image supervision substantially outperforms normal image supervision (CD: 0.554→0.431), as normals exhibit ambiguity along directions perpendicular to the viewing angle.

## Highlights & Insights

- **Jacobian field decomposition** is the core innovation: the static + dynamic decomposition ensures global geometric consistency while expressing per-frame local deformations, representing an elegant extension of NJF to temporal settings.
- **Adaptive remeshing** driven by rendering gradients automatically identifies regions requiring higher resolution in a simple, effective, and computationally tractable manner.
- The supervision strategy using diffuse images rather than normal maps is worth noting — normal direction ambiguity is a common issue in differentiable rendering.
- Decoupled learning of appearance and geometry avoids the mutual compensation problem.

## Limitations & Future Work

- Mesh representations are more prone to self-intersections than implicit functions; more robust self-intersection prevention methods are needed.
- The absence of physics-based simulation constraints may result in physically implausible deformations.
- The method relies on pretrained models (4DHumans, Sapiens) to extract SMPL parameters and normal/depth pseudo-ground-truth.
- Handling of face flipping and degenerate triangles during remeshing remains imperfect.

## Related Work & Insights

- NJF (Neural Jacobian Fields)-based deformation parameterization is a key building block; this paper extends it to temporal dynamic settings.
- TextDeformer also employs NJF with differentiable rendering, but operates only on single static meshes.
- The adaptive mesh refinement idea draws from the real-time mesh deformation method of Dunyach et al.
- Gaussian Garments achieves multi-view garment reconstruction using Gaussian splatting combined with physics simulation, representing a complementary direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The temporal decomposition of the Jacobian field and gradient-driven remeshing are both meaningful contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, comparisons with diverse baselines, and thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Provides a high-quality explicit method for monocular video garment reconstruction, particularly for loose-fitting garment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](monocular_facial_appearance_capture_in_the_wild.md)
- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](../../CVPR2025/human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)

</div>

<!-- RELATED:END -->
