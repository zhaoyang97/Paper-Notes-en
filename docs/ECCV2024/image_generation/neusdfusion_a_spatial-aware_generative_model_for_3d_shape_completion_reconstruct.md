---
title: >-
  [Paper Note] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation
description: >-
  [ECCV 2024][Image Generation] Proposes NeuSDFusion, a 3D shape generation framework based on a hybrid tri-plane SDF representation (NeuSDF) and a spatial-aware Transformer autoencoder. By preserving the spatial correspondences among tri-planes, it achieves state-of-the-art (SOTA) performance in tasks such as unconditional generation, multimodal shape completion, single-view reconstruction, and text-to-3D generation.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: ce69a858458509fb
---

# NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.18241](https://arxiv.org/abs/2403.18241)  
**Area**: Image Generation

## TL;DR

Proposes NeuSDFusion, a 3D shape generation framework based on a hybrid tri-plane SDF representation (NeuSDF) and a spatial-aware Transformer autoencoder. By preserving the spatial correspondences among tri-planes, it achieves state-of-the-art (SOTA) performance in tasks such as unconditional generation, multimodal shape completion, single-view reconstruction, and text-to-3D generation.

## Background & Motivation

- **Rise of Tri-plane Representations**: Tri-planes compactly represent 3D information using three orthogonal 2D planes, which is more efficient than voxels or point clouds.
- **Neglect of Spatial Consistency in Existing Methods**:
    - Methods like NFD and DiffusionSDF concatenate the three planes along the channel dimension and treat them as an RGB image, ignoring that there is no explicit relationship between the same coordinates of different planes.
    - Rodin's 3D-aware convolution aggregates features using pooling, which loses contextual information and fails to generate smooth surfaces.
- **Memory Constraints of 3D Representations**: TSDF stores distance fields using 3D voxels, which consumes substantial memory and makes it difficult to capture fine-grained shape features.
- **Representation Limitations of NFD**: Tri-planes are learned from discrete occupancy grids, limiting their representational capacity by the grid resolution.
- **Goal**: Create an efficient 3D representation and generation framework that can generate high-fidelity and diverse 3D shapes while maintaining spatial consistency.

## Method

### Overall Architecture

Three-stage pipeline:
1. **NeuSDF Fitting**: Encodes each 3D object into a hybrid SDF representation combining tri-planes and an MLP.
2. **Spatial-Aware Autoencoder**: Compresses the original tri-planes into a compact latent tri-plane representation while preserving spatial correspondences.
3. **Latent Diffusion Model**: Performs conditional/unconditional generation within the compressed latent space.

### Key Designs

**NeuSDF Representation**:
- Represents a 3D shape using three axis-aligned planes (XY, YZ, XZ).
- To query any 3D point $p$, it is projected onto the three planes to obtain features $F_{xy}, F_{xz}, F_{yz}$ via bilinear interpolation. These features are element-wise summed and decoded into an SDF value by an MLP.
- Optimization Method: Jointly optimizes the tri-plane and MLP parameters for each object.
- Sampling Strategy: Surface points ($\Omega_0$) + space-filling points ($\Omega$), with normal vectors sampled for additional supervision.

**Spatial-Aware Transformer Autoencoder**:
- **Limitations of Prior Work**: Existing methods (such as roll-out or channel concatenation) employ CNNs to process tri-planes, leading to the loss of cross-plane spatial relationships.
    - Roll-out convolves across two planes at the boundaries of the tri-planes, resulting in boundary artifacts.
    - Channel concatenation ignores the fact that there is no spatial connection between the same spatial positions across different planes.
- **Solution**: A U-shaped Transformer encoder-decoder architecture.
    - Each stage first downsamples each plane independently using grouped convolutions (more parameter-efficient).
    - The tri-planes are then flattened into a 1D token sequence $x \in \mathbb{R}^{C \times 3HW}$ and fed into the Transformer.
    - The Transformer attention learns the global relationships among the three planes.
- **Spatial-Aware Position Embedding (SAPE)**: Generates independent, learnable positional embeddings for each of the three planes. This introduces an inductive bias that allows each token to distinguish whether other tokens belong to the same plane or different planes.
- **Linear Attention**: Utilizes a linear attention mechanism to reduce complexity from $O(n^2)$ to $O(n)$, enabling direct processing of high-resolution tri-planes ($3 \times 64 \times 64 = 12288$ tokens at the first stage).

**Conditional Diffusion Model**:
- Trains a U-Net diffusion model within the latent tri-plane space.
- Condition Injection: Injects condition encoder outputs (image, text, or point cloud features) into each block of the U-Net via cross-attention layers.
- Classifier-free Guidance: Replaces the conditioning signal with a zero mask with a probability of 10%.

### Loss & Training

**NeuSDF Fitting**: $\mathcal{L}_{geo} = \mathcal{L}_{sdf} + \mathcal{L}_{normal} + \mathcal{L}_{eikonal}$
- SDF Loss: Enforces zero distance for surface points and consistency with ground-truth SDF values for space points.
- Normal Loss: Aligns the gradient direction with the ground-truth normals.
- Eikonal Loss: Constrains the norm of the SDF gradient to 1 (maintaining the physical properties of SDF).

**Autoencoder**: $\mathcal{L}_{ae} = \mathcal{L}_{rec} + \mathcal{L}_{KL} + \mathcal{L}_{geo}$

**Latent Diffusion**: $\mathcal{L}_{ldm} = \|\Psi(z_t, \gamma(t)) - z_0\|^2$

## Key Experimental Results

### Main Results

**Unconditional Generation (ShapeNet, 1-NNA↓)**:

| Method | Representation | Airplane CD/EMD | Chair CD/EMD | Car CD/EMD |
|------|------|----------------|-------------|-----------|
| IM-GAN | Occupancy | 79.48/82.94 | 58.59/69.05 | 95.69/94.79 |
| LION | Point Cloud | 67.41/61.23 | 53.70/52.34 | 53.41/51.14 |
| 3DQD | TSDF | 56.29/54.78 | 55.61/52.94 | 55.75/52.80 |
| **NeuSDFusion** | **NeuSDF** | **52.33/52.47** | **51.95/52.60** | **53.06/51.11** |

**Multimodal Shape Completion ($\times 10^2$)**:

| Method | Bottom Half MMD↓ | Bottom Half AMD↓ | Octant MMD↓ | Octant AMD↓ |
|------|-----------------|-----------------|------------|------------|
| AutoSDF | 3.51 | 8.20 | 5.72 | 12.79 |
| 3DQD | 2.93 | 6.30 | 4.69 | 10.93 |
| **NeuSDFusion** | **2.29** | **5.90** | **3.03** | **9.59** |

### Ablation Study

**Single-View Reconstruction (Pix3D)**:

| Method | CD↓ | F-Score↑ |
|------|-----|---------|
| Pix2Vox | 3.00 | 0.39 |
| AutoSDF | 2.28 | 0.42 |
| SDFusion | 1.85 | 0.43 |
| **NeuSDFusion** | **0.92** | **0.61** |

**Text-to-3D Generation**:

| Method | PMMD↓ | CLIP-S↑ | FPD↓ | TMD↑ |
|------|-------|---------|------|------|
| 3DQD | 1.49 | 32.11 | 59.00 | 2.80 |
| **NeuSDFusion** | **1.49** | **32.52** | **55.01** | **3.20** |

### Key Findings

1. **Representational Capacity**: NeuSDF outperforms representations like TSDF (3DQD) and point clouds (LION) in unconditional generation across all categories, proving the superiority of the hybrid tri-plane SDF representation.
2. **Spatial Consistency**: Compared to roll-out and channel-concatenation methods, the spatial-aware Transformer autoencoder generates smoother and more complete object surfaces without boundary artifacts.
3. **Single-View Reconstruction**: The Chamfer Distance (CD) drops from 1.85 (SDFusion) to 0.92 (**50% Gain**), and the F-Score increases from 0.43 to 0.61 (**42% Gain**), benefiting from the ability of the NeuSDF representation to capture finer shape details.
4. **Multimodal Completion**: Leads in quality metrics (MMD/AMD) while maintaining competitive diversity (TMD), indicating that a structured latent space benefits conditional generation.

## Highlights & Insights

- **Elegant Design of Hybrid Representation**: The NeuSDF representation, combining tri-planes and an MLP, enjoys the high efficiency of tri-planes while avoiding the resolution limits of discrete grids via continuous SDF learning.
- **Spatial-Aware Position Embedding (SAPE)**: A simple yet effective inductive bias that enables the Transformer to distinguish spatial relationships among the three planes.
- **Crucial Role of Linear Attention**: Makes it feasible to perform global attention directly on token sequences of length 12,288, breaking through the resolution-32 limitations of methods like LRM.
- **Versatile Conditional Generation Framework**: The same pipeline supports unconditional generation, shape completion, single-view reconstruction, and text-to-3D, demonstrating the versatility of the representation and architecture.

## Limitations & Future Work

- NeuSDF fitting in the first stage requires independent optimization for each object, which is time-consuming when scaling up the dataset.
- Shape extraction via Marching Cubes is still limited in resolution.
- Although linear attention reduces complexity, it may underperform compared to standard self-attention on highly complex geometric details.

## Rating

⭐⭐⭐⭐ (4/5) — The three-stage pipeline is clearly and completely designed. Both the NeuSDF representation and the spatial-aware Transformer exhibit substantial innovation. Comprehensive evaluation across four tasks is highly convincing, representing a major advancement in the field of 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)

</div>

<!-- RELATED:END -->
