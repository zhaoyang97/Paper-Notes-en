---
title: >-
  [Paper Note] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image
description: >-
  [CVPR 2025][3D Vision][4D Human Generation] Disco4D proposes a 4D human generation framework that disentangles clothing (represented by a Gaussian model) from the human body (represented by the SMPL-X model), generating animatable, editable, and layered 3D clothed human models from a single image, and supporting realistic 4D clothing dynamics.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D Human Generation"
  - "Clothing Disentanglement"
  - "Gaussian Splatting"
  - "Human Animation"
  - "Single-image Reconstruction"
date: 2026-05-08
content_hash: 8158cf0fed662441
---

# Disco4D: Disentangled 4D Human Generation and Animation from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2409.17280](https://arxiv.org/abs/2409.17280)  
**Code**: [https://disco-4d.github.io/](https://disco-4d.github.io/)  
**Area**: 3D Vision  
**Keywords**: 4D Human Generation, Clothing Disentanglement, Gaussian Splatting, Human Animation, Single-image Reconstruction

## TL;DR

Disco4D proposes a 4D human generation framework that disentangles clothing (represented by a Gaussian model) from the human body (represented by the SMPL-X model), generating animatable, editable, and layered 3D clothed human models from a single image, and supporting realistic 4D clothing dynamics.

## Background & Motivation

- Existing single-image 3D human reconstruction methods (such as PIFu, ECON) reconstruct the body and clothing as a single-layer holistic mesh, which cannot support applications like virtual try-on or clothing editing.
- The single-layer non-animatable mesh produced by holistic reconstruction makes re-animation and dynamic customization extremely difficult.
- Prior work has not achieved 4D layered human generation and animation from a single image or a few images.
- A representation that can separate the body and clothing is needed, while simultaneously supporting high-fidelity generation, fine-grained editing, and realistic animation.

## Method

### Overall Architecture

Disco4D adopts a bottom-up construction approach: (1) represent the human body using the SMPL-X parametric model; (2) represent clothing using 3D Gaussians; (3) bind the clothing Gaussians to the SMPL-X mesh and iteratively optimize them; (4) achieve separation and extraction of clothing categories via identity encoding. The complete human representation is denoted as $S_{human} = S_{body} \cup S_{cloth}$.

### Key Designs

1. **SMPL-X Gaussian Human Representation**:
    - **Function**: Provides a stable human anchor structure.
    - **Mechanism**: Directly binds flat 3D Gaussians to each triangular face of the SMPL-X mesh (similar to SuGaR). The Gaussian positions $\mu_{body}$ are determined by predefined barycentric coordinates, rotations are derived from face normals, and opacity is fixed to 1.0.
    - **Design Motivation**: SMPL-X excels at capturing human structure and kinematics. Fixing the body representation maintains integrity, allowing the learning process to focus on clothing.

2. **Mesh Embedding and Separable Optimization of Clothing Gaussians**:
    - **Function**: Models clothing as a layer independent of the human body.
    - **Mechanism**: Each clothing Gaussian is embedded onto a triangular face of the canonical SMPL-X mesh, located using a local coordinate system $\mu = O + \sigma i + \beta j + \gamma k$. SDF loss and pruning are used to ensure clothing remains outside the body. An identity encoding $e \in \mathbb{R}^{15}$ is introduced to associate each Gaussian with its clothing category.
    - **Design Motivation**: Embedding Gaussians into triangular faces allows them to animate naturally with SMPL-X deformations; identity encoding enables different garments to be independently extracted and edited.

3. **Diffusion Model-Enhanced Texture Completion and 4D Animation**:
    - **Function**: Completes textures in occluded regions and learns clothing dynamics.
    - **Mechanism**: Extracts texture details of occluded areas from a diffusion model using SDS loss. 4D animation predicts changes in position, rotation, and scale of clothing Gaussians via a deformation network $S'' = \phi(S', t)$, while the body is driven by SMPL-X. Thus, the clothing follows the body while exhibiting its own dynamics.
    - **Design Motivation**: Single images inevitably contain occluded regions, for which diffusion models provide reasonable priors; decoupled animation of clothing and body yields more realistic results.

### Loss & Training

Total loss function: $\mathcal{L} = \mathcal{L}_{ori} + \mathcal{L}_{id} + \mathcal{L}_{ani} + \mathcal{L}_{sdf} + \mathcal{L}_{SDS}$

- $\mathcal{L}_{ori}$: Standard 3D Gaussian rendering loss.
- $\mathcal{L}_{id} = \mathcal{L}_{2d} + \mathcal{L}_{3d}$: Identity encoding loss, incorporating cross-entropy classification and 3D nearest-neighbor consistency regularization.
- $\mathcal{L}_{ani}$: Anisotropy constraint, preventing overly thin Gaussian kernels during deformation.
- $\mathcal{L}_{sdf}$: SDF loss, ensuring clothing Gaussians remain outside the SMPL-X body.
- $\mathcal{L}_{SDS}$: Score Distillation Sampling loss for texture completion in occluded regions.

## Key Experimental Results

### Main Results

| Method | CLIP(All)↑ | CLIP(Pants)↑ | PSNR(NV)↑ | SSIM(NV)↑ | LPIPS(NV)↓ |
|------|-----------|-------------|----------|----------|-----------|
| Disco4D(CloSe) | **0.856** | **0.858** | 20.10 | 0.918 | 0.081 |
| LGM | 0.829 | 0.727 | **20.50** | **0.939** | **0.077** |
| DreamGaussian | 0.734 | 0.693 | 20.08 | 0.939 | 0.089 |
| SHERF | 0.777 | 0.785 | 18.96 | 0.912 | 0.083 |

### Ablation Study

| Configuration | CLIP(All)↑ | Assets↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-----------|---------|-------|-------|--------|
| Disco4D (reposed + deform) | **0.900** | **0.865** | **25.46** | **0.96** | **0.035** |
| Disco4D (reposed) | 0.853 | 0.774 | 23.94 | 0.95 | 0.049 |
| DG4D (Disco4D init) | 0.870 | 0.849 | 21.02 | 0.93 | 0.065 |
| GaussianAvatar | 0.822 | 0.768 | 20.01 | 0.93 | 0.069 |

### Key Findings

- Disco4D significantly outperforms all baselines in clothing disentanglement (indicated by CLIP Assets scores).
- Incorporating learned clothing deformations (+learned deformations) improves PSNR from 23.94 to 25.46 and reduces LPIPS from 0.049 to 0.035.
- In user studies, Disco4D achieves an image consistency score of 3.142 (on a 5-point scale), far exceeding LGM's 2.338 and DreamGaussian's 2.017.
- Using SMPL-X as a body anchor significantly improves the geometric accuracy of the face and limbs.

## Highlights & Insights

- **First to achieve single-image 4D layered human generation**: Supports simultaneous body-clothing separation, animation, and editing.
- **Ingenious identity encoding mechanism**: Employs a 15-dimensional learnable vector with 2D segmentation supervision and 3D nearest-neighbor regularization to achieve fine-grained clothing category separation.
- **No training on human-specific datasets**: Leverages general-purpose diffusion models, removing the need for human-specific dataset training.
- **Rich editing capabilities**: Supports clothing removal, color changing, material modification, and cross-character clothing transfer.

## Limitations & Future Work

- The accuracy of the initial SMPL-X estimation affects the quality of all downstream steps.
- Modeling loose clothing (like long dresses) and complex accessories remains challenging.
- The texture quality from diffusion model distillation might be suboptimal in heavily occluded areas such as the back.
- Clothing dynamics learning relies on video inputs; under the single-image modality, animation is limited to simple SMPL-X driving.

## Related Work & Insights

- Compared to NeRF-based methods like SHERF/ELICIT, the Gaussian representation offers significant advantages in efficiency and editing flexibility.
- The layered representation idea of SMPL-X + Gaussian can be extended to other layered reconstruction tasks.
- The design of identity encoding can inspire finer-grained scene decomposition approaches.
- It provides a practical framework for virtual try-on, digital fashion, and content creation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce body-clothing disentanglement to single-image 4D generation with an elegant framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative evaluation on multiple datasets + user studies, but lacks more real-world demos.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and detailed method description.
- **Value**: ⭐⭐⭐⭐⭐ High practical value, providing a new paradigm for fields like virtual try-on.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[CVPR 2025\] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)
- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)

</div>

<!-- RELATED:END -->
