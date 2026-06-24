---
title: >-
  [Paper Note] GenVDM: Generating Vector Displacement Maps From a Single Image
description: >-
  [CVPR 2025][3D Vision][Vector Displacement Maps] The first method to generate Vector Displacement Maps (VDMs) from a single image is proposed. By fine-tuning Zero123++ to generate multi-view normal maps, using neural SDF to reconstruct meshes, and then parameterizing them into VDM images using neural deformation fields, the authors construct the first academic VDM dataset. This provides 3D artists with the ability to generate customized geometric detail stamps on-demand.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Vector Displacement Maps"
  - "Single-image 3D Reconstruction"
  - "Multi-view Normal Generation"
  - "Neural Deformation Field"
  - "3D Modeling"
date: 2026-05-08
content_hash: fc8a5080ddfc708f
---

# GenVDM: Generating Vector Displacement Maps From a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2503.00605](https://arxiv.org/abs/2503.00605)  
**Code**: [https://yyuezhi.github.io/GenVDM/](https://yyuezhi.github.io/GenVDM/)  
**Area**: 3D Vision  
**Keywords**: Vector Displacement Maps, Single-image 3D Reconstruction, Multi-view Normal Generation, Neural Deformation Field, 3D Modeling

## TL;DR

The first method to generate Vector Displacement Maps (VDMs) from a single image is proposed. By fine-tuning Zero123++ to generate multi-view normal maps, using neural SDF to reconstruct meshes, and then parameterizing them into VDM images using neural deformation fields, the authors construct the first academic VDM dataset. This provides 3D artists with the ability to generate customized geometric detail stamps on-demand.

## Background & Motivation

**Background**: Although 3D generative models are developing rapidly, they are still not widely adopted in artistic workflows for two reasons: (1) generating fine geometric details is difficult; (2) they lack the precise spatial and compositional control that artists require. Existing Image-to-3D methods (such as LRM, Wonder3D, Magic123, etc.) focus on generating complete objects rather than localized geometric details.

**Limitations of Prior Work**: (1) VDM is a widely supported detail stamp representation in 3D modeling (supported by Blender, Maya, ZBrush, etc.), but creating VDMs is extremely difficult. Artists often rely on expensive third-party stamp packs, which limits customization and versatility; (2) Existing Image-to-3D methods do not generate physical parameterized 2D domains, making them unusable as direct stamps; (3) Single-view depth estimation fails to capture complex geometries such as occluded regions, overhanging structures, and cavities.

**Key Challenge**: VDMs need to represent arbitrary 3D displacements (including undercut and overhangs), whereas existing depth maps/scalar displacement maps can only represent height fields, failing to handle occlusion and self-occlusion. Additionally, there is no public VDM dataset available for training.

**Goal**: How to generate high-quality VDMs from a single RGB image? Specifically: (1) how to generate multi-view geometric representations that resolve occlusion; (2) how to parameterize the reconstructed mesh into a VDM image format; (3) how to construct a training dataset.

**Key Insight**: The authors observe that a VDM represents a smaller, simpler geometric region compared to a complete object, meaning diffusion models fine-tuned on small datasets can be utilized to generate its multi-view normal maps. The key insight is that only normal maps need to be generated (without RGB), as the focus is purely on geometric detail. Parameterizing VDMs using a neural deformation field (MLP) achieves natural smoothness while handling complex topologies.

**Core Idea**: Relying on multi-view normal generation to solve the occlusion problem, neural SDF to reconstruct meshes, and an MLP deformation field to parameterize the mesh into a VDM image, achieving high-quality zero-shot VDM generation with only 1200 training samples.

## Method

### Overall Architecture

Inputting a single RGB image (which can be sourced from a text-to-image model), a three-step pipeline is used to output a VDM image: (1) Multi-view normal generation—fine-tuning Zero123++ to generate normal maps of 6 predefined views from the input image; (2) Mesh reconstruction—utilizing the neural SDF optimization of Wonder3D to reconstruct the mesh from multi-view normals; (3) VDM parameterization—employing an MLP deformation field to deform and fit a 2D square to the reconstructed mesh to obtain the VDM image. The entire reconstruction pipeline takes approximately 6 minutes.

### Key Designs

1. **Multi-view Normal Map Generation**:

    - **Function**: Generates normal maps of 6 views from a single image to solve the occlusion problem in a single view.
    - **Mechanism**: Fine-tuning Zero123++ (an Image-to-Multiview model based on Stable Diffusion) to generate only normal maps instead of RGB. The 6 camera poses are redesigned: four horizontal directions $(0°, ±30°)$ and $(0°, ±60°)$, and two vertical directions $(±45°, 0°)$, with no back views since they are unnecessary for VDMs. Orthographic projection is adopted to reduce distortion. The input image is padded with a gray square background to simulate how a VDM looks when applied to a flat plane. Fine-tuning takes 3 days on 8 A100 GPUs.
    - **Design Motivation**: The geometry of a VDM may contain overhangs and cavities, which single-view depth estimation fails to capture due to occlusion. Generating only normals without RGB is because VDMs are solely concerned with geometry. The redesigned camera layout (excluding the back view) aligns with the characteristic that VDMs only require anterior hemisphere information.

2. **Neural SDF Reconstruction + VDM Parameterization (Two-step Reconstruction)**:

    - **Function**: Reconstructs the 3D mesh from multi-view normal maps and parameterizes it into a VDM image.
    - **Mechanism**: **Step 1** optimizes a neural SDF using Wonder3D's method, aligning the predicted normals with the generated multi-view normals via differentiable rendering (with $L_{rgb}$ removed since RGB is not predicted). Due to the gray square background design, the reconstructed mesh contains a flat base, allowing easy separation of the attached VDM component. **Step 2** uses an MLP $\phi_\theta$ to define a deformation field from a 2D square $[0,1]^2$ to 3D space. For a 2D point $p$, its 3D position is $p' = \phi_\theta(p)$. The optimization objective minimizes the symmetric Chamfer Distance between the deformed points and the target mesh, plus a boundary constraint loss.
    - **Design Motivation**: Directly using LRM for feed-forward reconstruction fails to generalize due to the limited training data (only 1200 samples). Directly optimizing mesh vertices requires carefully designed regularization and easily falls into local optima. The smooth inductive bias of the MLP naturally acts as a regularizer, encouraging deformation smoothness and avoiding the noise and distortion issues prevalent in traditional topology repair + parameterization pipelines.

3. **VDM Dataset Construction Pipeline**:

    - **Function**: Efficiently extracts and processes VDM training data from Objaverse 3D objects.
    - **Mechanism**: (a) Filtering Objaverse objects using keywords (organic shapes such as animals and characters); (b) developing a 3D lasso tool to allow annotators to select boundaries for cutting parts of interest; (c) densely sampling points on the extracted parts, removing internal points (using winding numbers), and performing Screened Poisson reconstruction to form a single manifold mesh; (d) performing least-squares plane fitting, projecting boundaries onto the plane, and deforming the parts to align boundaries coplanarly using a method similar to Poisson Image Editing; (e) stitching the parts onto a square mesh, followed by data augmentation with random coloring, scaling, and rotation. Ultimately, 1200 VDM patches are obtained, requiring only 24 person-hours of annotation.
    - **Design Motivation**: The lack of public VDM datasets represents a gap in the field. Extracting components directly from 3D objects is significantly more efficient than manual modeling. The Poisson-style boundary deformation ensures that the components seam-free fit the flat base.

### Loss & Training

Multi-view normal generation: Standard diffusion denoising loss, rendering random views as input so the model can handle various input view angles. VDM reconstruction: Symmetric Chamfer Distance + boundary constraint loss, sampling mesh points at each step for optimization, taking about 3 minutes per step.

## Key Experimental Results

### Main Results

| Method | CLIPImg↑ | CLIPText↑ | 3D-FID↓ |
|------|----------|-----------|---------|
| **GenVDM (Ours)** | **0.8520** | **0.2701** | **192.7** |
| Wonder3D | 0.8246 | 0.2542 | 199.5 |
| Magic123 | 0.8293 | 0.2510 | 213.2 |
| LRM | 0.8144 | 0.2510 | 239.9 |
| Scalar DM (DepthAnything) | 0.8223 | 0.2564 | 213.0 |

### Ablation Study (VDM Parameterization Methods)

| Configuration | CLIPImg↑ | CLIPText↑ | 3D-FID↓ | Description |
|------|----------|-----------|---------|------|
| Reconstructed Mesh (Reference Upper Bound) | 0.8440 | 0.2636 | 198.0 | Mesh before parameterization |
| (a) Topology Repair + Tutte Embedding | 0.8401 | 0.2617 | 209.9 | Topology repair without considering distortion |
| (b) Mesh Optimization | 0.8245 | 0.2525 | 217.2 | Prone to falling into local optima |
| **(c) Ours (MLP Deformation Field)** | **0.8521** | **0.2701** | **192.7** | Even better than the mesh before parameterization |

### Key Findings

- **GenVDM significantly outperforms all baselines on all metrics**: It scores $2.7\%$ higher on CLIPImg and shows a $3.4\%$ reduction in 3D-FID compared to the second-best model Wonder3D. This indicates that designs tailored for VDMs (localized geometric stamps) are more effective than general-purpose 3D generation methods.
- **Scalar displacement maps cannot replace VDMs**: While the frontal view of DepthAnything's scalar DM looks reasonable, the side view fails due to its inability to represent occluded areas.
- **MLP deformation field is the optimal parameterization scheme**: It performs better than both topology repair and mesh optimization, and even surpasses the raw mesh before parameterization, indicating that the smooth inductive bias of the MLP acts as a denoiser.
- **Only 1200 training samples are sufficient**: Fine-tuning on top of the pre-trained Zero123++ requires only a small amount of data to adapt to the VDM task, demonstrating the transfer capability of pre-trained models.

## Highlights & Insights

- **Targeting VDM, an industry-standard but academically rare representation**: VDMs are widely used in 3D modeling tools but rarely studied in academia. The topic precisely captures a vacancy with high practical value.
- **Using MLP for parameterization is a stroke of genius**: Traditional topology repair + parameterization pipelines are extremely fragile. The implicit bias of the MLP naturally provides smooth regularization while avoiding local optima issues in mesh optimization. This approach can be generalized to other scenarios requiring mesh parameterization.
- **The data construction pipeline is highly instructive**: The 3D lasso tool combined with an automated processing pipeline annotates 1200 samples in only 24 person-hours, demonstrating high efficiency. The Poisson-style boundary processing is also very practical, ensuring components blend seamlessly with the base.
- **Elegant design of only generating normal maps**: The decision to discard RGB generation reduces task complexity, allowing the model to focus exclusively on geometric quality.

## Limitations & Future Work

- VDM reconstruction relies on per-instance optimization (taking about 6 minutes each), which is significantly slower than feed-forward LRM methods, presenting the biggest bottleneck for practical application.
- The training dataset contains only 1200 samples, which limits category diversity (predominantly organic shapes).
- Failures occur in generating thin structures: the multi-view normal maps might look reasonable individually but suffer from cross-view inconsistency, leading to reconstruction failure.
- Only the ancestral hemisphere view is captured, making it unable to handle cases that require back-facing geometry.
- Compositional generation of VDMs has not been explored—such as generating multiple complementary VDM patches simultaneously and organizing them.

## Related Work & Insights

- **vs Wonder3D**: Wonder3D is designed for complete objects and performs poorly when generating localized VDM-like shapes (such as isolated noses or ears); GenVDM is better suited for VDM generation thanks to its specialized camera layout and gray background design.
- **vs DepthAnything (Scalar DM)**: Scalar DMs can only represent height fields and cannot handle overhangs and cavities; VDM's three-channel displacement vectors can represent arbitrarily complex geometry.
- **vs LRM / Magic123**: These methods rely heavily on texture to hallucinate geometric details, yielding low geometric quality after stripping away textures; GenVDM focuses solely on normals/geometry, resulting in more realistic details.
- **Relationship with Geometry Images**: VDMs are conceptually similar to Geometry Images. While recent work has leveraged diffusion models to generate Geometry Images to synthesize 3D shapes, GenVDM concentrates on the specific requirements of localized geometric stamps.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ This work introduces generative AI to VDMs for the first time, a representation of great industrial importance yet academic absence. The methodology incorporates multiple innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ It compares against multiple baselines and provides detailed parameterization ablations, though the test set comprises only 50 images.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the workflow chart is complete, and major design choices are accompanied by comparative validation.
- Value: ⭐⭐⭐⭐⭐ The method can be directly applied to 3D modeling workflows, and the proposed first public VDM dataset holds significant value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting](../../ICCV2025/3d_vision/text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)
- [\[CVPR 2025\] PhysGen3D: Crafting a Miniature Interactive World from a Single Image](physgen3d_crafting_a_miniature_interactive_world_from_a_single_image.md)
- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](../../ICCV2025/3d_vision/a_recipe_for_generating_3d_worlds_from_a_single_image.md)

</div>

<!-- RELATED:END -->
