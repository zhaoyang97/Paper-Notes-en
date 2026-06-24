---
title: >-
  [Paper Note] SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] SPAR3D proposes a two-stage method for reconstructing 3D objects from a single image. The first stage utilizes a lightweight point cloud diffusion model to generate a sparse point cloud to handle occlusion uncertainty, while the second stage employs a triplane transformer to convert the point cloud into a high-quality mesh with PBR materials, achieving 0.7-second inference and supporting interactive editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "Point Cloud Diffusion"
  - "Two-stage Reconstruction"
  - "Interactive Editing"
  - "PBR Material"
date: 2026-05-08
content_hash: be969421573ac51c
---

# SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images

**Conference**: CVPR 2025  
**arXiv**: [2501.04689](https://arxiv.org/abs/2501.04689)  
**Code**: [https://spar3d.github.io](https://spar3d.github.io)  
**Area**: 3D Vision  
**Keywords**: Single-image 3D Reconstruction, Point Cloud Diffusion, Two-stage Reconstruction, Interactive Editing, PBR Material

## TL;DR
SPAR3D proposes a two-stage method for reconstructing 3D objects from a single image. The first stage utilizes a lightweight point cloud diffusion model to generate a sparse point cloud to handle occlusion uncertainty, while the second stage employs a triplane transformer to convert the point cloud into a high-quality mesh with PBR materials, achieving 0.7-second inference and supporting interactive editing.

## Background & Motivation

**Background**: Single-image 3D object reconstruction is a fundamental problem in computer vision. Presently, the field is split into two paradigms: regression-based methods (e.g., TripoSR, SF3D), which efficiently infer visible surfaces but produce blurry results in occluded regions; and generative methods (diffusion models), which better model uncertain regions but suffer from high computational costs and poor alignment with visible surfaces.

**Limitations of Prior Work**: Regression-based methods assume a bijective mapping from images to 3D, leading to over-smoothing in occluded areas. Diffusion-based methods suffer from slow iterative sampling in high-resolution 3D spaces, and their generated outputs often exhibit misalignment with the visible surfaces of the input image. Multi-view diffusion methods (generating multi-view images before reconstruction) introduce cross-view inconsistency artifacts and are even slower.

**Key Challenge**: High-quality 3D reconstruction requires satisfying three key demands simultaneously: (1) probabilistic modeling capability for uncertain regions (to avoid over-smoothing), (2) high-fidelity alignment with visible surfaces, and (3) high computational efficiency. Existing methods can only fulfill a subset of these requirements.

**Goal**: To combine the strengths of both regression and generative methods—leveraging a diffusion model to handle uncertainty while utilizing a regression model to guarantee efficiency and fidelity.

**Key Insight**: Point clouds serve as the most lightweight 3D representation (where all information bits are allocated to represent the surface), acting as a bridge between the two stages: performing diffusion sampling on a low-resolution point cloud (fast), and then utilizing a regression model to convert the point cloud into a high-precision mesh.

**Core Idea**: Deallocate the uncertainty modeling to the low-resolution point cloud diffusion phase (512 points, enabling fast sampling). In the mesh generation phase, high-quality results are regressed by utilizing both point cloud and image features, naturally supporting interactive editing by users.

## Method

### Overall Architecture
Given an input image $I \in \mathbb{R}^{3 \times h \times w}$, SPAR3D operates in two stages. Point sampling stage: A DDPM-based point cloud diffusion model, conditioned on the input image, generates 512 6-channel points (XYZ + RGB albedo) in approximately 0.4 seconds. Mesh generation stage: A triplane transformer generates a high-resolution $384 \times 384$ triplane conditioned on both point cloud and image features. A shallow MLP then queries density, albedo, and normal values, and a mesh is generated via differentiable Marching Tetrahedra, while simultaneously estimating PBR materials and lighting, taking about 0.3 seconds.

### Key Designs

1. **Point Diffusion Model**:

    - **Function**: Performs probabilistic modeling in a low-dimensional space to generate a sparse point cloud representing the 3D structure and albedo color of the object.
    - **Mechanism**: A 16-layer transformer is used as a denoiser. The noisy point cloud $\boldsymbol{p}_t \in \mathbb{R}^{512 \times 6}$ is linearly projected into tokens, concatenated with image tokens encoded by DINOv2, and then fed into the transformer to predict noise. DDIM sampling and Classifier-Free Guidance (CFG) are employed to enhance fidelity. Key innovation: Directly generating the albedo point cloud (rather than RGB), which relocates the uncertainty of material decomposition into the diffusion stage.
    - **Design Motivation**: Performing diffusion sampling directly in high-resolution 3D spaces is too slow, but a low-resolution space of 512 points allows for fast iterative sampling. Crucially, the topological independence of point clouds becomes an advantage here, enabling effortless user-interactive editing followed by instant mesh regeneration.

2. **Two-stream Triplane Transformer**:

    - **Function**: Fuses sparse point clouds and image features into high-resolution triplane features to generate high-quality meshes.
    - **Mechanism**: It consists of three sub-modules: a point cloud encoder (12-layer vanilla transformer), an image encoder (DINOv2-large), and 4 two-stream blocks (containing 3 self-attentions + 2 cross-attentions), utilizing 3072 latent tokens (feature dimension of 1024). The triplane resolution is $384 \times 384$, from which density, albedo, and normal values are queried through a shallow MLP. DMTet is used at a resolution of 160 to convert this into a mesh, with additional predictions of vertex offsets and normals to minimize artifacts.
    - **Design Motivation**: Adopts a decoupled two-stream design similar to PointInfinity and SF3D to ensure efficient generation of high-resolution triplanes. The point cloud provides coarse 3D structural guidance, while the image provides fine-grained texture and visible surface details.

3. **Differentiable Rendering & Inverse Rendering**:

    - **Function**: Jointly estimates PBR materials (albedo, metallic, roughness) and environment lighting to mitigate "baked-in lighting" artifacts.
    - **Mechanism**: Implements a differentiable renderer based on the Disney BRDF model, employing Monte Carlo integration and Multiple Importance Sampling (MIS) to estimate incoming irradiance. Lighting estimation is based on a learned prior from RENI++. Innovatively, screen-space shadow ray marching is implemented to model self-occlusion shadows. AlphaCLIP is used instead of CLIP to estimate metallic/roughness, improving stability under variations in object size.
    - **Design Motivation**: Generating the albedo point cloud significantly reduces the uncertainty of inverse rendering (mitigating the ambiguity of illumination-albedo decomposition), making material decomposition learning feasible in the mesh stage. Additionally, the screen-space shadow test further improves the modeling of specular surfaces.

### Loss & Training
Point sampling stage: Standard DDPM noise prediction loss $L_{simple}$ is optimized using a sigmoid noise schedule. Mesh stage: Rendering loss consists of L2 image distance, LPIPS perceptual distance, and foreground mask L2 distance, augmented with mesh and shading regularization. Training data curation follows TripoSR. The two stages are trained independently, with the mesh stage utilizing GT point clouds during training.

## Key Experimental Results

### Main Results
Quantitative comparison on the GSO dataset:

| Method | CD↓ | FS@0.1↑ | FS@0.2↑ | PSNR↑ | LPIPS↓ | Time (s)↓ |
|------|------|---------|---------|-------|--------|---------|
| Shap-E | 0.204 | 0.359 | 0.638 | 15.3 | 0.205 | 3.1 |
| TripoSR | 0.145 | 0.501 | 0.784 | 18.5 | 0.151 | 0.2 |
| InstantMesh | 0.135 | 0.545 | 0.812 | 18.1 | 0.146 | 36.1 |
| SF3D | 0.137 | 0.540 | 0.806 | 18.0 | 0.145 | 0.3 |
| **SPAR3D** | **0.120** | **0.584** | **0.850** | **18.6** | **0.139** | **0.7** |

### Ablation Study

| Configuration | CD↓ | FS@0.1↑ | Description |
|------|------|---------|------|
| W/o point cloud conditioning (pure regression) | Higher | Lower | Occluded areas are over-smoothed |
| Random point cloud | Medium | Medium | Provides minimal structural guidance |
| GT point cloud | Lowest | Highest | Ideal upper bound |
| Diffusion sampled point cloud | **0.120** | **0.584** | Close to the GT upper bound |
| w/o albedo point cloud | Higher | - | Inverse rendering is harder to converge |
| w/ albedo point cloud | Lower | - | Significantly improves material decomposition |

### Key Findings
- SPAR3D outperforms SOTA methods across all geometric and texture metrics, with an inference time of only 0.7 seconds, which is 50x faster than InstantMesh (36.1s).
- The point cloud diffusion model effectively alleviates the over-smoothing issue in occluded regions—reducing CD from 0.137 (SF3D) to 0.120.
- The albedo point cloud is key to stable inverse rendering—addressing the uncertainty of material decomposition during the diffusion stage significantly simplifies learning in the mesh stage.
- The interactive editing capability is a unique advantage: users can edit the low-resolution point cloud and generate an updated mesh within 0.3 seconds.

## Highlights & Insights
- **Smart division of labor in the two-stage design**: Decouples uncertainty modeling from deterministic reconstruction. Diffusion rapidly resolves ambiguity in low-dimensional space, while regression secures high quality and efficiency in high-dimensional space, complementing rather than conflicting with each other.
- **Multiple advantages of point clouds as intermediate representations**: Lightweight (only 512 points), free of topological constraints (convenient for editing), and high information density (every bit is dedicated to representing the surface). This achieves an optimal trade-off in representation selection for this specific scenario.
- **Clever design of the albedo point cloud**: By directly predicting albedo colors during the diffusion stage, the challenging albedo-illumination ambiguity inherent in inverse rendering is shifted forward, representing a highly practical engineering innovation.

## Limitations & Future Work
- The point cloud resolution is restricted to 512 points, which limits the capacity to represent details in geometrically complex objects.
- Currently, only single-object reconstruction is supported; multi-object scenes require additional segmentation preprocessing.
- Inverse rendering still exhibits errors on certain specular or semi-transparent materials.
- The number of DDIM sampling steps can be further optimized to accelerate inference.

## Related Work & Insights
- **vs TripoSR/SF3D**: While both are fast reconstruction methods, they lack probabilistic modeling, resulting in over-smoothed occluded regions. SPAR3D addresses this via point cloud diffusion, reducing the CD from 0.145/0.137 to 0.120.
- **vs InstantMesh**: Employs multi-view diffusion followed by regression-based reconstruction, but multi-view inconsistency causes artifacts and runs slowly (36s). SPAR3D avoids this issue through 3D-consistent sampling.
- **vs Point-E**: While both utilize point cloud diffusion, Point-E generates outputs directly from points, lacking mesh refinement. SPAR3D's two-stage design achieves significantly higher output quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The two-stage design cleverly integrates diffusion and regression, and the albedo point cloud is a meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive quantitative and qualitative comparisons across multiple datasets, with ablation studies validating the contributions of each component.
- Writing Quality: ⭐⭐⭐⭐ Clear exposition of motivation and methodology, supported by aesthetic and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ 0.7-second high-quality 3D reconstruction combined with interactive editing offers immense practical value, pushing the Pareto frontier of this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)

</div>

<!-- RELATED:END -->
