---
title: >-
  [Paper Note] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh
description: >-
  [CVPR 2025][3D Vision][Avatar Generation] DAGSM is proposed, a text-driven disentangled digital human generation method that represents the body and individual clothing items separately using GS-enhanced Mesh (GSM), supporting outfit customization, realistic animation, and texture editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Avatar Generation"
  - "Disentangled Clothing"
  - "3D Gaussian Mesh"
  - "Text-driven"
  - "Physical Simulation"
date: 2026-05-08
content_hash: b1fffc84d6a9d2ba
---

# DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh

**Conference**: CVPR 2025  
**arXiv**: [2411.15205](https://arxiv.org/abs/2411.15205)  
**Code**: [Project Page](https://dagsm.github.io/)  
**Area**: 3D Vision / Digital Avatar Generation  
**Keywords**: Avatar Generation, Disentangled Clothing, 3D Gaussian Mesh, Text-driven, Physical Simulation

## TL;DR

DAGSM is proposed, a text-driven disentangled digital human generation method that represents the body and individual clothing items separately using GS-enhanced Mesh (GSM), supporting outfit customization, realistic animation, and texture editing.

## Background & Motivation

### Background

**Background**: Existing text-driven 3D human generation methods represent the body and clothing as a single integrated model. This prevents outfit swapping, results in unrealistic animations (with clothes adhering to the body), and offers limited user control over clothing combinations.

### Limitations of Prior Work

**Limitations of Prior Work**: Textures generated directly by SDS suffer from low quality (oversmoothing, color oversaturation) and lack visual appeal.

### Key Challenge

**Key Challenge**: The requirement of a disentangled digital avatar that is animatable, supports customizable outfits, possesses high-quality textures, and accommodates various clothing topologies (e.g., skirts vs. pants).

## Method

### Overall Architecture

A three-stage pipeline: (1) generation of the undergarment-clad body (based on SMPL-X + 2DGS); (2) sequential generation of clothing items (creating a mesh proxy followed by binding 2DGS to generate textures); (3) visually consistent texture refinement.

### Key Designs

1. **GS-enhanced Mesh (GSM) Hybrid Representation**:
    - **Function**: Binds 2D Gaussian Splatting onto mesh triangular faces.
    - **Mechanism**: Each 2DGS defines its position in the local coordinate system of its bound triangular face (barycentric coordinates $\lambda_1, \lambda_2$ + normal offset $z$). During rendering, the world coordinates are $\hat{\mu} = \lambda_1 x_A + \lambda_2 x_B + (1-\lambda_1-\lambda_2)x_C + z\vec{n}$. UV feature maps $(\mathcal{U}_c, \mathcal{U}_\alpha)$ are utilized to store color and opacity, facilitating ease of editing.
    - **Design Motivation**: Combines the physical simulation capabilities of meshes with the high-quality rendering of Gaussians, resulting in more realistic clothing motion and simplified texture editing.

2. **SAM-based Clothing Disentanglement and Filtering**:
    - **Function**: Removes non-clothing Gaussians during the clothing generation process to achieve body-clothing disentanglement.
    - **Mechanism**: Allocates a category attribute $o$ (0 = body, 1 = clothing) to each Gaussian. Precise semantic masks of dressed human images are obtained via SAM as labels to optimize $o$ using MSE loss. Gaussians with $o < 0.5$ are pruned every 500 iterations.
    - **Design Motivation**: SDS optimization inevitably generates parts of the body within clothing regions; semantic information provided by SAM assists in precise separation.

3. **Visually Consistent Texture Refinement**:
    - **Function**: Improves the quality and multi-view consistency of SDS-generated textures.
    - **Mechanism**: Proposes a cross-view attention mechanism to maintain texture style consistency. An Incident Angle-Weighted Denoising Estimation (IAW-DE) strategy is designed to adjust the per-pixel denoising intensity based on the incident angle.
    - **Design Motivation**: Direct SDS optimization yields oversmoothed and multi-view inconsistent textures. The refinement stage utilizes the RFDS loss of Stable Diffusion 3 to improve quality.

### Loss & Training

- Body color branch: $\mathcal{L}_{\mathcal{G}_b} = \mathcal{L}_{\text{rfds}}^{I_b} + \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s + \lambda_r \mathcal{L}_r$ (including position/scale/rotation regularization).
- Clothing generation: $\mathcal{L}_{\mathcal{G}_m} = \mathcal{L}_{\text{rfds}}^{I_a} + \mathcal{L}_{\text{sam}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$
- Distance regularization $\mathcal{L}_{\text{dis}}$ constrains the distance from Gaussians to the mesh surface.
- Smoothness regularization $\mathcal{L}_{\text{smooth}}$ ensures a smooth clothing surface.
- RFDS loss (adapted for rectified-flow models such as SD3) is used to replace traditional SDS.

## Key Experimental Results

### Main Results

| Method | Texture Quality | Disentanglement Capability | Outfit Customization | Realistic Animation |
|------|---------|---------|---------|---------|
| TADA | Moderate | ✗ | ✗ | Limited |
| HumanGaussian | Good | ✗ | ✗ | Limited |
| TELA | Moderate | ✓(NeRF) | ✓ | Unrealistic |
| SO-SMPL | Poor | ✓(Limited) | ✓(Limited) | Limited |
| DAGSM | **Best** | **✓** | **✓** | **Realistic** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| w/o SAM filtering | Mixed body-clothing | Boundaries cannot be clearly separated |
| w/ SAM filtering | Clear separation | Semantic guidance is effective |
| w/o Cross-view attention | Multi-view texture inconsistency | Caused by independent denoising in each view |
| w/ Cross-view attention | Good consistency | Cross-view feature sharing is effective |
| w/o IAW-DE | Poor side texture quality | Weak signal in regions with large incident angles |
| w/ IAW-DE | Uniform high quality | Weighting strategy compensates for side signals |

### Key Findings

- GSM representation supports physical simulation-driven realistic clothing movement (e.g., natural hem flapping), which is far superior to traditional skeleton-driven methods.
- Excellent texture diversity is demonstrated, supporting text-based descriptions of various clothing fabrics (lace, denim, wool, shear fabric).
- Precise appearance control can be achieved by providing reference images.
- Manual editing is supported by directly modifying UV texture maps.

## Highlights & Insights

- For the first time, text-driven fully disentangled digital human generation (body + multiple independent clothing items) is realized, where each garment can be replaced independently.
- The GSM representation ingeniously integrates the advantages of meshes (structural representation, physical simulation) and Gaussians (rendering quality, complex textures).
- The sequential generation design (body first, then clothing) is simple yet effective; generating clothing conditioned on the body naturally avoids interpenetration.
- RFDS loss is utilized to replace SDS, leveraging the stronger priors of rectified-flow models such as SD3.

## Limitations & Future Work

- Clothing mesh extraction relies on the TSDF algorithm, which may lack precision for complex topologies.
- Generation speed is limited by the multi-stage optimization process.
- Physics-based simulation of clothing requires support from external simulators.
- Handling occlusion relationships for multi-layer clothing (e.g., jacket + shirt) needs further validation.

## Related Work & Insights

- SMPL-X → Human prior; 2DGS → High-quality surface reconstruction.
- RFDS loss → Stronger distillation signal; SAM → Semantic segmentation assistance.
- TELA → Pioneer of the disentanglement concept (though using NeRF restricted animation realism).

## Rating

- Novelty: ⭐⭐⭐⭐ The GSM hybrid representation is elegantly designed, and the disentangled generation pipeline is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ A variety of generation results are presented, demonstrating support for outfit changes and animations.
- Writing Quality: ⭐⭐⭐⭐ Detailed methodology description with clear overall illustrations.
- Value: ⭐⭐⭐⭐⭐ Solves realistic pain points in virtual human generation; support for clothing customization and physical animation substantially boosts utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)

</div>

<!-- RELATED:END -->
