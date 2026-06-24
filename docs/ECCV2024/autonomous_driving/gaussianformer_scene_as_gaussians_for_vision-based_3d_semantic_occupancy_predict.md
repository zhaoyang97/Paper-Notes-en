---
title: >-
  [Paper Note] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction
description: >-
  [ECCV 2024][Autonomous Driving][3D Occupancy] This paper proposes an object-centric 3D semantic Gaussian representation to replace traditional dense voxels. It describes driving scenes using a set of sparse 3D semantic Gaussians and generates occupancy predictions via Gaussian-to-voxel splatting, reducing memory consumption by 75%–82% while yielding comparable performance.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Occupancy"
  - "3D Gaussian Splatting"
  - "Sparse Representation"
  - "Object-Centric"
  - "Semantic Occupancy"
date: 2026-05-08
content_hash: eba91bf4d467e415
---

# GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction

**Conference**: ECCV 2024  
**arXiv**: [2405.17429](https://arxiv.org/abs/2405.17429)  
**Code**: [GitHub](https://github.com/huang-yh/GaussianFormer)  
**Area**: Autonomous Driving  
**Keywords**: 3D Occupancy, 3D Gaussian Splatting, Sparse Representation, Object-Centric, Semantic Occupancy

## TL;DR

This paper proposes an object-centric 3D semantic Gaussian representation to replace traditional dense voxels. It describes driving scenes using a set of sparse 3D semantic Gaussians and generates occupancy predictions via Gaussian-to-voxel splatting, reducing memory consumption by 75%–82% while yielding comparable performance.

## Background & Motivation

3D semantic occupancy prediction is a critical task for the robustness of vision-based autonomous driving systems. However, existing scene representation methods exhibit clear limitations:

**Redundancy in Voxel Representation**: Dense voxel methods allocate equal computational resources to every 3D location. However, over 90% of driving scenes consist of empty space, leading to significant computation and storage waste.

**Information Loss in Plane Representations**: BEV and TPV improve efficiency by compressing the height dimension, but they inevitably lose fine-grained 3D structural information.

**Lack of Flexibility in Grid representation**: Grid-based representations cannot adapt to varying object scales and regional complexities (e.g., bus vs. pedestrian, road vs. vegetation), leading to suboptimal resource allocation.

**Challenges in Dynamic Scene Modeling**: Grid-based representations struggle to capture the dynamics of the scene, since it is the objects that move, not the grids.

Inspired by 3D Gaussian Splatting, the core idea of this work is to describe the scene using the universal approximation capability of Gaussian Mixture Models, where each Gaussian adaptively covers a region of interest.

## Method

### Overall Architecture

GaussianFormer consists of three stages:
1. **Image Encoding**: Backbone + FPN extracts multi-scale 2D features.
2. **Gaussian Generation (Image to Gaussians)**: 3D Gaussians are iteratively updated through alternating self-attention (self-encoding), image cross-attention, and attribute refinement.
3. **Gaussian-to-Voxel Splatting**: Sparse Gaussians are converted into dense 3D occupancy predictions via local aggregation.

### Key Designs

#### Object-Centric 3D Scene Representation

Each 3D semantic Gaussian is represented by a $d$-dimensional vector: $G = (m, s, r, c)$
- **$m$** (3D): Mean/Position
- **$s$** (3D): Scale
- **$r$** (4D): Rotation (Quaternion)
- **$c$** ($|C|$-D): Semantic logits

The occupancy prediction at position $p$ is the sum of contributions from all Gaussians at that point: $\hat{o}(p) = \sum_i \exp\left(-0.5 (p-m_i)^T \Sigma_i^{-1} (p-m_i)\right) \cdot c_i$, where the covariance matrix is defined as $\Sigma = R S S^T R^T$. Compared to voxel representations, the mean and covariance properties of Gaussians allow for adaptive resource allocation—large objects are covered by large Gaussians, while small objects are precisely depicted by small Gaussians.

#### GaussianFormer Model (Image to Gaussians)

**Gaussian Attributes and Queries**: Processed as two sets of features:
- Gaussian attributes $G$ (physical properties, which are the learning targets)
- Gaussian queries $Q$ (high-dimensional feature vectors that implicitly encode 3D information)

Both are initialized as learnable vectors and iteratively refined through $B$ blocks.

**Self-Encoding Module**: Treats Gaussians as a point cloud located at their means. It uses 3D sparse convolutions for processing (instead of deformable self-attention), achieving linear computational complexity. Since the number of Gaussians $P$ is much smaller than the total number of voxels, sparse convolutions can fully exploit the sparsity of the Gaussians.

**Image Cross-Attention Module (ICA)**: For each Gaussian, 3D reference points are generated based on its mean and covariance, which are projected onto multi-view image features for sampling. Deformable attention is utilized to aggregate image features. The offsets of reference points are computed based on the covariance to reflect the shape of the Gaussian distribution.

**Attribute Refinement Module**: Decodes intermediate attributes from the Gaussian queries $Q$ via an MLP. The mean is updated residually (to maintain consistency during iterations), while other attributes (scale, rotation, semantics) are directly overwritten (to avoid vanishing gradient issues associated with sigmoid/softmax functions).

#### Gaussian-to-Voxel Splatting

Directly iterating through all Gaussians is highly expensive (complexity of $O(XYZ \cdot P)$). Leveraging the locality of the Gaussian distribution, only the neighboring Gaussians for each voxel are considered:

1. Embed Gaussians into the target voxel grid (localized by their means).
2. Calculate the neighborhood radius of each Gaussian based on its scale attributes.
3. Construct a list of (Gaussian index, voxel index) pairs.
4. Sort the pairs by voxel index to retrieve the set of Gaussians to be aggregated for each voxel.
5. Perform local aggregation to generate occupancy predictions.

The entire process is implemented in CUDA to fully utilize GPU parallel computing capabilities.

### Loss & Training

- **Loss Functions**: Cross-Entropy Loss + Lovasz-Softmax Loss, applying supervision on the output of each refinement module.
- **Total Loss**: $L = \sum_{i=1}^{B} (\mathcal{L}_{ce}^i + \mathcal{L}_{lov}^i)$
- **Optimizer**: AdamW with a weight decay of 0.01. The learning rate warms up for 500 iterations to a peak of 2e-4, then decays using a cosine schedule.
- **Training**: 20 epochs with a batch size of 8. ResNet101-DCN (FCOS3D pre-trained) is used for nuScenes, and ResNet50 is used for KITTI-360.
- **Number of Gaussians**: 144,000 for nuScenes, 38,400 for KITTI-360.
- **GaussianFormer Blocks**: 4
- **Data Augmentations**: Random flip + photometric distortion.

## Key Experimental Results

### Main Results (nuScenes val, SurroundOcc Annotations)

| Method | SC IoU | SSC mIoU |
|------|--------|----------|
| MonoScene | 23.96 | 7.31 |
| BEVFormer | 30.50 | 16.75 |
| TPVFormer* | 30.86 | 17.10 |
| OccFormer | 31.39 | 19.03 |
| SurroundOcc | 31.49 | 20.30 |
| **GaussianFormer** | 29.83 | 19.10 |

With comparable performance (mIoU 19.10 vs. 20.30 of SurroundOcc), GaussianFormer achieves this using only **17.8%** of the memory consumption of SurroundOcc, and **24.8%** of OccFormer.

### Ablation Study

**Effect of the Number of Gaussians on Performance and Memory** (nuScenes):

| Number of Gaussians | mIoU | Memory (MB) |
|---------|------|-----------|
| 36000 | 17.06 | 5755 |
| 72000 | 18.40 | 8115 |
| 144000 | 19.10 | 12741 |
| 288000 | 19.23 | 21993 |

144,000 Gaussians offer the best trade-off between performance and efficiency.

**Module Ablations**:
- Sparse convolution self-encoding vs. No self-encoding: +0.75 mIoU
- Covariance-aware offset vs. Fixed offset in ICA: +0.39 mIoU
- Iterative refinement (4 blocks) vs. Single prediction: Significant improvement

### Key Findings

1. Only 144,000 Gaussians are needed to achieve comparable performance with the dense method (640,000 voxels).
2. A memory saving of 75%–82% is a substantial engineering contribution, which is crucial for deployment on edge devices.
3. Key representations can adaptively fit object scales—large objects are covered by large Gaussians while small objects are wrapped in compact Gaussians.
4. Applying 3D sparse convolutions to unstructured Gaussians serves as an effective self-encoding paradigm.

## Highlights & Insights

- **First Object-Centric Occupancy Representation**: Breaks free from the conventions of voxel/plane grid representations, utilizing the flexibility of Gaussian distributions to describe the scene.
- **Novel Application of 3D-GS**: Successfully migrates 3D Gaussian Splatting from offline rendering to online perception. Crucially, this is a 3D-to-3D formulation (rather than rendering to 2D), demonstrating strong originality.
- **CUDA-Implemented Splatting**: An efficient implementation of local aggregation and sorting makes the sparse-to-dense conversion highly feasible.
- **Natural Integration with Downstream Tasks**: Being object-centric, Gaussians are inherently suitable for downstream tasks like motion prediction and end-to-end driving.

## Limitations & Future Work

1. The number of Gaussians remains a manually defined hyperparameter, and adaptive count adjustment has not yet been implemented.
2. Underrepresentation may occur in scenes with extreme object scales (e.g., highly distant small pedestrians).
3. The SC IoU (29.83) lags behind dense methods (31.49), indicating that geometric completeness requires further improvement.
4. There is a lack of temporal modeling, leaving historical frame information unexploited.
5. The Gaussian-to-voxel splatting stage still involves dense voxel grids; thus, the final output is not fully sparse.

## Related Work & Insights

- **3D Gaussian Splatting**: Provides the physical formulation of Gaussian representations, but GaussianFormer performs online inference rather than offline optimization.
- **TPVFormer**: The efficiency of tri-perspective view representations contrasts with GaussianFormer’s Gaussian representation.
- **SurroundOcc**: A representative dense voxel-based framework; GaussianFormer achieves comparable performance with a fraction of the memory footprint.
- **SparseOcc**: Another sparse occupancy approach (mask-based) that is complementary to the Gaussian representation.
- Gaussian representation can be further extended to 4D scene prediction and open-vocabulary occupancy.

## Rating

- **Novelty**: 5/5 - Introduces the first object-centric occupancy representation, offering a brand-new application of 3D-GS.
- **Experimental Thoroughness**: 4/5 - Validated on two datasets with thorough ablations, although performance does not surpass SOTA.
- **Writing Quality**: 4/5 - Clearly explained motivation and methodology with excellent diagrams.
- **Value**: 4/5 - Offers significant memory savings though relies on a CUDA-supported splatting module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[AAAI 2026\] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](../../AAAI2026/autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](../../ICCV2025/autonomous_driving/semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)

</div>

<!-- RELATED:END -->
