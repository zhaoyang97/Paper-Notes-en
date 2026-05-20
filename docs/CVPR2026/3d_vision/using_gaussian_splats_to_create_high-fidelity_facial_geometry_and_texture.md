---
title: >-
  [Paper Note] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture
description: >-
  [CVPR2026][3D Vision][Gaussian Splatting] This paper proposes a face reconstruction pipeline based on improved Gaussian Splatting. It tightly couples Gaussians with triangle meshes via soft constraints and semantic segme…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "Gaussian Splatting"
  - "facial geometry reconstruction"
  - "de-lit texture"
  - "semantic segmentation constraint"
  - "neural texture"
  - "MetaHuman"
date: 2026-05-08
content_hash: 2b51c18cc60cebd9
---

# Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture

**Conference**: CVPR2026
**arXiv**: [2512.16397](https://arxiv.org/abs/2512.16397)  
**Code**: Not open-sourced (Epic Games / Stanford)  
**Area**: 3D Vision / Face Reconstruction
**Keywords**: Gaussian Splatting, facial geometry reconstruction, de-lit texture, semantic segmentation constraint, neural texture, MetaHuman

## TL;DR

This paper proposes a face reconstruction pipeline based on improved Gaussian Splatting. It tightly couples Gaussians with triangle meshes via soft constraints and semantic segmentation supervision, reconstructing high-fidelity triangular mesh geometry from only 11 uncalibrated images. A PCA prior combined with a relightable Gaussian model is used to disentangle illumination and recover de-lit albedo textures, with outputs fully compatible with standard graphics pipelines (MetaHuman).

## Background & Motivation

1. **Demand-driven**: The demand for high-fidelity, controllable, and relightable facial digitization in VR/gaming/film continues to grow, yet existing methods typically rely on multi-camera calibration or light stages, impeding large-scale democratization.
2. **Limitations of NeRF**: NeRF's implicit representation struggles to precisely disentangle geometry from appearance, and does not directly produce triangle meshes, preventing seamless integration into standard graphics pipelines.
3. **Limitations of vanilla 3DGS**: Although explicit, standard Gaussian Splatting decouples Gaussians from the underlying geometry—Gaussians can freely deform to fit images, resulting in poor mesh quality.
4. **Texture–illumination entanglement**: Separating albedo from illumination using only a small number of images without a light stage is a severely under-constrained problem; existing methods often produce baked-in shadows.
5. **Standard pipeline compatibility**: Industrial graphics pipelines have been refined over decades of hardware and software optimization; neural rendering methods must be converted to mesh-plus-texture representations to be useful in real-time applications.
6. **Challenges of sparse input**: Compared to long video sequences or dense multi-view setups, reconstructing high-quality facial geometry and texture from only 11 images places much higher demands on regularization and constraint design.

## Method

### Overall Architecture

Input → monocular video captured with an iPhone rear camera → select 11 frames at predefined poses → coarse geometry initialization (MetaHuman Animator) → improved Gaussian Splatting training → geometry refinement → texture reconstruction and de-lighting → output triangle mesh + de-lit texture → MetaHuman conversion.

### Improved Gaussian Splatting Model

**Core Idea**: Each triangle face is bound to exactly one Gaussian; densification and pruning are disabled to maintain a one-to-one correspondence between Gaussians and faces. Mesh vertex positions are **not** jointly optimized during training, decoupling Gaussian optimization from mesh deformation.

**Soft constraint regularization** ($\mathcal{L}_{\text{reg}}$): Inspired by Laplacian smoothing, each Gaussian's geometric feature vector $\mathbf{z}_i$ is encouraged to be consistent with the mean of its edge-adjacent neighbors:

$$\mathcal{L}_{\text{reg}} = \sum_i \left\| \mathbf{z}_i - \frac{1}{|\mathcal{E}(i)|} \sum_{j \in \mathcal{E}(i)} \mathbf{z}_j \right\|^2$$

Three feature types are regularized independently:

- **Center displacement** $\mathcal{L}_{\text{reg}}^{\text{center}}$: The offset between each Gaussian center and its face centroid is kept smooth across the neighborhood.
- **Local normal** $\mathcal{L}_{\text{reg}}^{\text{normal}}$: The local normal of each Gaussian varies smoothly across the mesh (UV coordinates are used to construct a consistent local frame, resolving ambiguities).
- **Boundary displacement** $\mathcal{L}_{\text{reg}}^{\text{boundary}}$: The distance from the Gaussian's outer boundary point to the face centroid is kept smooth across the neighborhood, constraining Gaussian shape and extent.

**Semantic segmentation supervision** ($\mathcal{L}_{\text{seg}}$): A Mask2Former segmentation network is trained on 1,600 MetaHuman synthetic samples to partition the face into semantic regions (face, nose, lips, eyes, ears, etc.). Each Gaussian inherits the label of its associated face; a predicted segmentation map is constructed via alpha blending and compared against network predictions to compute the loss. This prevents Gaussians from "sliding" into incorrect semantic regions.

**Eye regularization** ($\mathcal{L}_{\text{eyes}}$): Penalizes intersection between eyeball Gaussians and eye-socket Gaussians, preventing eyeball Gaussians from occluding the socket and causing geometric inaccuracies.

### Triangle Mesh Geometry Refinement

After training, camera extrinsics are fixed and the mesh is refined iteratively:

1. Re-optimize Gaussian parameters to extract supervision signals (Gaussian boundary points $\mathbf{x}_i^*$).
2. Deform mesh vertices by minimizing $\mathcal{L}_{\text{centroid}} = \sum_i \| \mathbf{v}_i^{\text{centroid}} - \mathbf{x}_i^* \|^2$.
3. Two refinement rounds: the first optimizes MetaHuman PCA coefficients; the second optimizes individual vertex positions.

### Neural Texture Scheme

Gaussians are transformed from world space into UV texture space and splatted orthographically along the surface normal direction, while colors still depend on world-space view directions. This allows Gaussian Splatting to be used as view-dependent neural textures within a standard graphics pipeline without modifying any other pipeline components.

### Loss Function

- **Image reconstruction**: $\mathcal{L}_{\text{img}} = 0.8 \cdot \mathcal{L}_1 + 0.2 \cdot \mathcal{L}_{\text{D-SSIM}}$
- **Geometry constraints**: $\mathcal{L}_{\text{reg}}^{\text{center/normal/boundary}}$, $\mathcal{L}_{\text{scale}}$
- **Semantics**: $\mathcal{L}_{\text{seg}}$ ($\lambda=50$)
- **Eyes**: $\mathcal{L}_{\text{eyes}}$ ($\lambda=20$)
- **Lighting/texture**: $\mathcal{L}_{\text{lighting}}$, $\mathcal{L}_{\text{rotation}}$, $\mathcal{L}_{\text{blending}}$, $\mathcal{L}_{\text{view}}$

### De-lit Texture Generation

- Ambient illumination is modeled with spherical harmonics, corrected by occlusion maps and normal maps.
- A PCA prior (top-20 MetaHuman basis functions) regularizes the albedo texture.
- A learnable blending weight $\beta_p$ controls the contribution ratio of Gaussian versus mesh texture, regularized toward zero to favor mesh texture.
- After training, view-dependent color and lighting are disabled, and high-frequency details are recovered by high-pass filtering the target images.

## Key Experimental Results

### Geometry Reconstruction Comparison

| Method | Semantic Alignment | Side-view Silhouette | Neutral Expression | Data Requirement |
|--------|-------------------|---------------------|-------------------|-----------------|
| **Ours** | ✅ Precise | ✅ Accurate | ✅ Direct | 11 images |
| NextFace | ❌ Semantic drift | ❌ Fails on side views | ✅ | Multiple images |
| NHA | ❌ Texture sliding | ⚠️ Moderate | ❌ Expression overfitting | Multiple images |
| CoRA | ⚠️ Nose/jaw artifacts | ⚠️ Blurry boundaries | ✅ | Flash capture |

### Ablation Study

| Ablation | Effect |
|----------|--------|
| Remove semantic segmentation | Gaussians slide into incorrect regions; geometric artifacts appear |
| Remove soft constraints | Gaussians decouple from faces; irregular sizes and shapes; poor mesh quality |
| Remove eye loss | Eyeball Gaussians occlude the socket; socket geometry becomes too small |
| Remove occlusion map | Baked shadows remain in de-lit texture (under nose, lip seam) |

### De-lit Texture Quality

- De-lit results remain highly consistent across different lighting conditions (Fig. 16, two de-lit texture columns are visually similar).
- Relighting under novel illumination outperforms CoRA (CoRA textures retain more baked lighting).
- Supports joint training on heterogeneous data (outdoor + flash), further improving rigid alignment and geometric accuracy.

## Highlights & Insights

1. **Minimal data requirement**: High-quality face reconstruction from only 11 iPhone selfie images, achieving genuine democratization of facial digitization.
2. **Elegant soft constraint design**: Three sets of Laplacian constraints on center/normal/boundary tightly couple Gaussians to the mesh, preserving 3DGS fitting capacity while ensuring geometric quality.
3. **Semantic segmentation supervision**: A segmentation network trained on MetaHuman synthetic data provides zero-cost semantic annotations and prevents texture sliding.
4. **Neural texture innovation**: Transforming Gaussians into texture space as view-dependent neural textures introduces zero intrusion into industrial graphics pipelines.
5. **Complete de-lighting pipeline**: PCA prior + spherical harmonics lighting + occlusion maps + high-frequency recovery yields high-quality albedo without a light stage.
6. **End-to-end MetaHuman compatibility**: Outputs are directly usable in UE5 standard pipelines, supporting animation and relighting.
7. **Text-driven extension**: A text-driven asset creation workflow is demonstrated via ChatGPT-generated images → Veo 3-generated video → pipeline reconstruction.

## Limitations & Future Work

1. **Limited de-lighting accuracy**: Shadows cannot be fully removed without a light stage; fine-grained geometric details (e.g., wrinkles) are sacrificed during de-lighting.
2. **Difficult eye region reconstruction**: Significant Gaussian overlap around the eyes and eyelids; segmentation granularity is insufficient; better landmark prediction is needed.
3. **Hair and neck not handled**: The framework focuses on the face; Gaussians in hair and neck regions have no structured constraints and do not participate in geometry optimization.
4. **Dependence on MetaHuman topology**: The entire pipeline is strongly coupled to the MetaHuman template; generalizing to other topologies requires additional effort.
5. **Synthetic-to-real domain gap**: The segmentation network is trained on MetaHuman synthetic data; robustness to extreme real-world lighting and occlusion has not been fully validated.

## Related Work & Insights

- **vs. NeRF-based methods** (HeadNeRF, HQ3DAvatar, etc.): NeRF's implicit representation cannot directly output meshes; this paper explicitly constrains Gaussians to triangle faces, directly yielding standard-pipeline-compatible outputs.
- **vs. Gaussian Avatars** (Qian et al.): Gaussian Avatars jointly optimize meshes and Gaussians; this paper decouples the two, constraining them independently and then using Gaussians to drive mesh deformation, achieving greater flexibility and geometric accuracy.
- **vs. 2DGS / SuGaR**: 2DGS uses flat Gaussians with depth distillation; SuGaR uses SDF regularization; this paper uses semantic segmentation and soft constraints to more directly establish semantic correspondences.
- **vs. CoRA** (Han et al.): CoRA requires flash capture and produces nose/jaw artifacts with residual illumination; this paper requires only ordinary capture and achieves more thorough de-lighting.
- **vs. NextFace / NHA**: NextFace fails on side views; NHA overfits expressions making neutral poses unusable; this paper outperforms both across all viewpoints and neutral expressions.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of soft constraints, semantic segmentation, and neural textures is novel; the idea of transforming Gaussians into texture space is particularly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations and multiple comparison methods, but quantitative metrics (PSNR/SSIM, etc.) and larger-scale user studies are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear exposition, rigorous mathematical derivations, and rich informative figures.
- Value: ⭐⭐⭐⭐ — Direct industrial applicability (especially within the Epic/MetaHuman ecosystem); academic contribution lies in the systematic integration of multiple techniques.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[AAAI 2026\] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting](../../AAAI2026/3d_vision/splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)

</div>

<!-- RELATED:END -->
