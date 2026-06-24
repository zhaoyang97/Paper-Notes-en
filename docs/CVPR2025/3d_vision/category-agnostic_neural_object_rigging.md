---
title: >-
  [Paper Note] Category-Agnostic Neural Object Rigging
description: >-
  [CVPR 2025][3D Vision][Object Rigging] Proposes CANOR (Category-Agnostic Neural Object Rigging), which automatically discovers low-dimensional pose spaces of deformable objects in a completely category-agnostic, data-driven manner by encoding deformable 4D objects into a sparse set of spatially localized blobs and instance-aware feature volumes, enabling intuitive pose manipulation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Object Rigging"
  - "Pose Manipulation"
  - "Category-Agnostic"
  - "Deformable Objects"
  - "Neural Representation"
date: 2026-05-08
content_hash: d8293705104f5a3d
---

# Category-Agnostic Neural Object Rigging

**Conference**: CVPR 2025  
**arXiv**: [2505.20283](https://arxiv.org/abs/2505.20283)  
**Code**: [https://guangzhaohe.com/canor](https://guangzhaohe.com/canor)  
**Area**: 3D Vision  
**Keywords**: Object Rigging, Pose Manipulation, Category-Agnostic, Deformable Objects, Neural Representation

## TL;DR

Proposes CANOR (Category-Agnostic Neural Object Rigging), which automatically discovers low-dimensional pose spaces of deformable objects in a completely category-agnostic, data-driven manner by encoding deformable 4D objects into a sparse set of spatially localized blobs and instance-aware feature volumes, enabling intuitive pose manipulation.

## Background & Motivation

In the dynamic 4D world, the motions of various deformable objects (humans, animals, articulated objects, etc.) reside in low-dimensional manifolds. Traditional methods designed heuristic representations like skeletal rigging for this, but these methods rely on domain-expert knowledge and require designing structures for each category separately, which severely hinders scalability.

Although some methods attempt to automatically discover similar structures (e.g., skeleton extraction methods), most still rely on some degree of category priors (e.g., human model SMPL), limiting their applicability to general categories. The **Key Challenge** is: **how to automatically discover low-dimensional interpretable rigging representations from limited animated 3D sequences without relying on any category priors?**

**Key Insight** of this work: Inspired by skeletal rigging, sparse spatially localized blobs (anisotropic spheres) are introduced as an intermediate representation to decouple object pose and identity information. Users can directly drag the blobs to modify poses while identity features remain unchanged.

## Method

### Overall Architecture

Given the point cloud input $\mathbf{P} \in \mathbb{R}^{n_p \times 3}$, the encoder $\mathcal{E}$ maps it to a set of blobs $\mathcal{B}$. Each blob encodes the spatial position and local features of a semantic part. After the blobs are voxelized into feature volumes, they are reconstructed into occupancy fields to output 3D meshes using a Transformer decoder conditioned on identity features. Users can pose the object by editing blob positions and rotations.

### Key Designs

1. **Blob Pose Representation**:
    - **Function**: Represents the dynamic pose of the object using a sparse set of parameterized spheres.
    - **Mechanism**: Each blob $\mathbf{b} = (\mathbf{x}, \mathbf{r}, \mathbf{s}, \mathbf{o}, \mathbf{f})$ is composed of a position, rotation (quaternion), scale, opacity, and feature vector; parameters are divided into pose-related parameters $\mathcal{B}_P = \{(\mathbf{x}_i, \mathbf{r}_i)\}$ and identity-related parameters $\mathcal{B}_I = \{(\mathbf{s}_i, \mathbf{o}_i, \mathbf{f}_i)\}$, achieving explicit decoupling between pose and identity.
    - **Design Motivation**: Compared to skeletons which require strict hierarchical structures and manual design for each category, blobs model the object as a set of semi-rigid parts, being more flexible, easier to learn, and naturally supporting cross-category generalization.

2. **Learnable Codebook Encoder**:
    - **Function**: Extracts point-level features from point clouds and aggregates them into sparse blob parameters.
    - **Mechanism**: PointTransformer is used to extract point-level features $\mathbf{F}$, followed by cross-attention between a set of learnable codebook tokens $\mathcal{Q}$ and the point-level features to obtain attention weights $\mathbf{W}$. The pose features $\mathcal{F}_P[i] = \sum_j \mathbf{W}[i,j] \cdot \gamma(\mathbf{P}[j])$ aggregate positional encodings, and the identity features $\mathcal{F}_I[i] = \sum_j \mathbf{W}[i,j] \cdot \phi(\mathbf{F}[j] \oplus \gamma(\mathbf{P})[j])$ aggregate semantic + geometric information; each parameter is regressed by independent MLPs.
    - **Design Motivation**: The cross-category shared codebook ensures consistent semantic correspondence of blobs across different instances, with each token corresponding to a specific semantic part.

3. **Blob Feature Voxelization + Transformer Decoding**:
    - **Function**: Decodes edited blobs back into 3D meshes.
    - **Mechanism**: Through differentiable voxelization, the feature of each grid point is a weighted sum of blob features $\mathbf{F}_G[i] = \frac{\sum_j w_{ij} \mathbf{f}_j}{\sum_j w_{ij} + \epsilon}$ where weights are $w_{ij} = \mathbf{o}_j \cdot \exp(-c \cdot \|\frac{\mathbf{g}_i - \mathbf{x}_j}{\mathbf{s}_j}\|^2)$; the blob features are enhanced with positional encoding (addressing the loss of detail in compact blob representations); they are iteratively refined through self-attention layers and conditioned on identity features from the encoder via cross-attention; finally, occupancy values are predicted through MLPs, and meshes are extracted via Marching Cubes.
    - **Design Motivation**: Differentiable voxelization ensures that blob parameters (position, rotation, size) have explicit and interpretable effects; identity conditioning retains fine-grained geometric details.

### Loss & Training

- Reconstruction Loss: Binary cross-entropy $\mathcal{L}_\text{recon}$ supervises occupancy field prediction.
- Voxelization Regularization: $\mathcal{L}_\text{vox}$ constrains the voxel weight distribution to be consistent with ground-truth occupancy (using cosine similarity).
- Total Loss: $\mathcal{L} = \mathcal{L}_\text{recon} + \lambda_\text{vox} \mathcal{L}_\text{vox}$.
- Training Strategy: Two frames of the same identity but with different poses are sampled; one frame provides the identity parameters $\mathcal{B}_I$, and the other provides the pose parameters $\mathcal{B}_P$ to simulate a user editing scenario.
- The proportion of near-surface sampling starts at 0 (for the first 200k iterations) and gradually increases to 0.8, ensuring stable blob initialization before capturing high-frequency details.

## Key Experimental Results

### Main Results

| Method | DeformingThings4D IoU↑ | CD₁↓ | FaMoS IoU↑ | CD₁↓ | Fish IoU↑ | CD₁↓ |
|------|----------------------|------|-----------|------|----------|------|
| KeypointDeformer | 0.536 | 0.060 | 0.923 | 0.029 | 0.499 | 0.062 |
| NeuralDeformGraph | 0.875 | 0.020 | 0.800 | 0.019 | 0.686 | 0.040 |
| SkeRig | 0.802 | 0.057 | 0.790 | 0.045 | 0.782 | 0.049 |
| **CANOR (Ours)** | **0.937** | **0.017** | **0.960** | **0.018** | **0.860** | **0.024** |

### Ablation Study

| Configuration | IoU ↑ | CD₁ ↓ | CD₂ ↓ |
|------|-------|-------|-------|
| Without Identity Conditioning | 0.853 | 0.025 | 0.018 |
| Isotropic blobs | 0.934 | 0.017 | 0.012 |
| K=8 blobs | 0.845 | 0.028 | 0.020 |
| K=16 blobs | 0.927 | 0.020 | 0.013 |
| **K=24 blobs (Full)** | **0.937** | **0.017** | **0.011** |

### Key Findings

- CANOR significantly outperforms all baselines across all five datasets (quadrupeds, human faces, fish, refrigerators, eyeglasses).
- Identity conditioning is crucial for reconstruction quality (dropping it decreases IoU from 0.937 to 0.853).
- Increasing the number of blobs yields substantial benefits (as K grows from 8 to 24, IoU increases from 0.845 to 0.937).
- The impact of anisotropic vs. isotropic blobs is relatively small, but anisotropic blobs produce superior results.
- On a hand-sculpted "clay-monster", a controllable rigging representation can be trained using only an iPhone scan.

## Highlights & Insights

- The **completely category-agnostic** design is the biggest highlight: the same framework works on quadrupeds, facial expressions, fish, refrigerators, and eyeglasses without any category prior.
- **Blob as an intermediate representation** strikes an elegant balance between skeletal rigging and purely implicit representations: offering spatial interpretability without requiring topology design.
- The design of a shared codebook + cross-attention cleverly establishes semantic correspondence among different instances within the same category.
- The "clay-monster" experiment demonstrates the user-friendliness and practical application potential of this method for non-professional users.

## Limitations & Future Work

- Different categories must be trained separately (no cross-category generalization).
- Multiple animation sequences of the same category are required as training data.
- The number of blobs requires a manually specified upper bound.
- Topology changes (e.g., object splitting/merging) are not supported.
- The training time is relatively long (approximately 7 days on 2×A6000 GPUs).

## Related Work & Insights

- BlobGAN uses blobs to model indoor scene layouts in 2D images, whereas this work (Ours) extends it to 3D deformable objects.
- KeypointDeformer uses keypoints for deformation but is limited to static shape deformation.
- NeuralDeformationGraph optimizes node representations but lacks category priors, making it prone to overfitting.
- The decoupling concept (pose vs. identity) in this work can be extended to 4D reconstruction and animation generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Category-agnostic automatic rigging discovery is a brand-new task, and the blob representation is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thoroughly validated on five diverse categories with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and highly detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ Highly inspiring for animation production and 4D understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](../../CVPR2026/3d_vision/cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](mv_3dcd_multiview_change_detection.md)
- [\[NeurIPS 2025\] RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data](../../NeurIPS2025/3d_vision/riganyface_scaling_neural_facial_mesh_auto-rigging_with_unlabeled_data.md)
- [\[CVPR 2025\] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)

</div>

<!-- RELATED:END -->
