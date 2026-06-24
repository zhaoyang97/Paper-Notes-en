---
title: >-
  [Paper Note] TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features
description: >-
  [CVPR 2025][3D Vision][Texture Transfer] This paper proposes TriTex, a method for learning a volumetric texture field from a single textured mesh. By projecting Diff3F semantic features into a triplane representation, TriTex utilizes a convolutional network and an MLP to achieve semantic-aware, feed-forward texture transfer, outperforming existing methods in both inference speed and texture fidelity.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Texture Transfer"
  - "Triplane Representation"
  - "Semantic Features"
  - "One-Shot Learning"
  - "3D Mesh"
date: 2026-05-08
content_hash: e36e5a13cf95b733
---

# TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features

**Conference**: CVPR 2025  
**arXiv**: [2503.16630](https://arxiv.org/abs/2503.16630)  
**Code**: [Project Page](https://danacohen95.github.io/TriTex/)  
**Area**: 3D Vision  
**Keywords**: Texture Transfer, Triplane Representation, Semantic Features, One-Shot Learning, 3D Mesh

## TL;DR

This paper proposes TriTex, a method for learning a volumetric texture field from a single textured mesh. By projecting Diff3F semantic features into a triplane representation, TriTex utilizes a convolutional network and an MLP to achieve semantic-aware, feed-forward texture transfer, outperforming existing methods in both inference speed and texture fidelity.

## Background & Motivation

- 3D texture transfer (applying the semantic texture of a source mesh to a target mesh) is a fundamental requirement in game development, simulation, and video production.
- Existing diffusion model-based methods (such as TEXTure, EASI-TEX) are proficient at texture generation but struggle to faithfully preserve the original appearance of the source texture.
- SDS optimization solutions (such as Latent-NeRF, Paint-it) require a long time to process a single object, making them unsuitable for large-scale scenes.
- Iterative depth-conditioned inpainting methods are prone to viewpoint inconsistency artifacts.
- IP-Adapter-based methods (such as MVEdit, EASI-TEX) only obtain vague inspiration from reference images, frequently deviating from the source texture.
- Methods requiring training on large-scale 3D datasets (such as Texturify, AUV-net) are restricted by category and data availability.
- There is a lack of efficient methods that can learn solely from a single textured mesh and generalize to new target meshes of the same category.
- Texture transfer requires implicit or explicit semantic correspondence rather than simple pixel-level copying.

## Method

### Overall Architecture

The architecture of TriTex takes a 3D mesh with pre-extracted Diff3F semantic features and a 3D query point as input, and outputs the color of that point. First, it projects semantic features from 6 orthogonal views into a triplane representation $\mathcal{T} \in \mathbb{R}^{3 \times W \times H \times 2D}$, which is then processed by a triplane-aware convolutional block to generate $\mathcal{T}'$. During inference, the query point of the target mesh samples features from the three planes, which are concatenated and passed through a coloring MLP $c: \mathbb{R}^{3D'} \to [0,1]^3$ to output the RGB color. Training is completed solely on a single textured mesh using a rendering reconstruction loss.

### Key Designs

**Design 1: Diff3F Semantic Features + Triplane Projection**
- **Function**: Establishes semantic correspondences between the source and target meshes.
- **Mechanism**: Utilizes Diff3F (frozen diffusion model + DINO features) to extract zero-shot 3D semantic descriptors for the mesh, which are then projected onto the triplane representation from 6 orthogonal directions (3 positive and 3 negative directions along the axes). The triplane-aware convolutional block aggregates features across the three planes (by averaging each plane along its axis and replicating it to the others) to achieve cross-plane information interaction.
- **Design Motivation**: Diff3F features exhibit cross-shape semantic consistency, allowing the semantic-to-color mapping learned on a single mesh to generalize to other objects in the same category with significant geometric variation. The triplane representation enables efficient processing of 3D information using 2D convolutions.

**Design 2: One-shot Training Strategy and Data Augmentation**
- **Function**: Learns from a single textured mesh while preventing overfitting to the specific triplane projection.
- **Mechanism**: Training utilizes a rendering reconstruction loss $\mathcal{L} = \mathbb{E}_\theta[\mathcal{L}_{MSE}(\theta) + \delta_{app}\mathcal{L}_{app}(\theta)]$ to compare the predicted texture against the ground truth texture under random viewpoints. It applies two levels of data augmentation: (1) pre-processing level 3D transformations on the mesh with re-extracted features, and (2) training-level perturbations including translation, scaling, and small rotations.
- **Design Motivation**: The core challenge of one-shot learning is generalization. Data augmentation expands the distribution of semantic features, preventing the network from overfitting to the source mesh's specific triplane projection.

**Design 3: Coloring Neural Field**
- **Function**: Maps triplane semantic features to RGB colors.
- **Mechanism**: For any 3D query point, features are sampled via bilinear interpolation from the three processed planes, concatenated into a single vector, and mapped to the $[0,1]^3$ color space through a lightweight MLP. Positional encoding is used to enhance detail generation capability (since the triplane resolution is only $32 \times 32$).
- **Design Motivation**: The implicit representation of MLP naturally possesses spatial continuity and generalization capability, making it more robust than operating directly in the UV space.

### Loss & Training

The total loss is $\mathcal{L} = \mathbb{E}_\theta[\mathcal{L}_{MSE}(\theta) + \delta_{app}\mathcal{L}_{app}(\theta)]$, where the MSE loss ensures pixel-level accuracy and the perceptual loss $\mathcal{L}_{app}$ emphasizes high-level semantic feature alignment, improving texture realism.

## Key Experimental Results

### Main Results: Texture Transfer Quality Comparison

| Method | SIFID↓ | CLIP sim.↑ | Inference Time |
|------|--------|-----------|---------|
| TEXTure | 0.34 | 0.84 | 5 min |
| MVEdit | 0.38 | 0.84 | 1 min |
| EASI-TEX | 0.29 | 0.85 | 15 min |
| **TriTex** | **0.22** | **0.87** | **1 min** |

### Ablation Study: Component Contributions

| Setting | SIFID↓ | CLIP sim.↑ |
|------|--------|-----------|
| w/o network (Nearest Neighbor) | 0.23 | 0.86 |
| w/o $\mathcal{L}_{MSE}$ | 0.23 | 0.86 |
| w/o $\mathcal{L}_{app}$ | 0.28 | 0.85 |
| w/o augmentations | 0.21 | 0.85 |
| **TriTex (full)** | **0.22** | **0.87** |

### Key Findings
- TriTex outperforms the best baseline EASI-TEX in both SIFID (0.22 vs 0.29) and CLIP similarity (0.87 vs 0.85).
- In an Amazon Mechanical Turk user study, TriTex was strongly preferred in all three comparisons (>65% of votes).
- Removing the perceptual loss leads to blurry outputs (SIFID rises to 0.28), demonstrating that high-level feature alignment is crucial.
- Removing augmentations results in a drop in the CLIP score (0.85), validating that generalization ability relies heavily on augmentation.
- Inference takes only 1 minute, which is comparable to MVEdit but delivers superior quality.

## Highlights & Insights

1. **Elegant Combination of One-shot Learning and Semantic Correspondence**: Leveraging pre-trained semantic features (Diff3F) to transform one-shot learning into semantic mapping learning.
2. **Practicality of the Triplane Representation**: Strikes an excellent balance between 3D and 2D operations, supporting efficient 2D convolutional processing.
3. **Strong Cross-Shape Generalization**: Although trained on only a single mesh, it retains high-quality transfer performance even under significant shape variations in objects of the same category.
4. **Fast Feed-forward Inference**: No per-object optimization is required during inference, making it ideal for large-scale scene applications.

## Limitations & Future Work

- Lack of generative capabilities: Cannot generate new details that do not exist in the source texture (e.g., if the source dog lacks a tongue, the target dog's tongue cannot be textured).
- Does not leverage text-to-image priors: Cannot compensate when semantic feature matching is ambiguous.
- Requires target shapes to be aligned with the source shape orientation; allowing arbitrary rotation degrades detail preservation.
- Cross-category transfer performance depends on the degree of semantic overlap.
- Future work could incorporate cross-attention to handle reference images, enabling image-based feed-forward texturing.

## Related Work & Insights

- Unlike Texturify/AUV-net, which require large-scale datasets, TriTex only requires a single textured mesh.
- Similar to Splice, which utilizes DINO features for 2D color transfer, TriTex extends this to 3D semantic texture transfer.
- The cross-shape semantic consistency of Diff3F features is a critical foundation for the success of this method.

## Rating

⭐⭐⭐⭐ — The method is simple yet highly efficient. The design that maps single-instance learning with triplane semantic mapping is ingenious. The experiments are thorough (both quantitative and user studies), demonstrating clear advantages in texture fidelity and speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](blade_single-view_body_mesh_estimation_through_accurate_depth_estimation.md)
- [\[CVPR 2025\] Coherent 3D Portrait Video Reconstruction via Triplane Fusion](coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)

</div>

<!-- RELATED:END -->
