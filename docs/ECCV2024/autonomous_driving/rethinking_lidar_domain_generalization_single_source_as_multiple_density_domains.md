---
title: >-
  [Paper Note] Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR semantic segmentation] This work proposes a Density-Discriminative Feature Embedding (DDFE) module that leverages the inherent density diversity in a single LiDAR source domain (dense nearby and sparse far away) to learn density-aware feature representations, achieving generalization to unseen domains under different sensor configurations without requiring target domain data.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR semantic segmentation"
  - "domain generalization"
  - "point cloud density"
  - "feature embedding"
  - "data augmentation"
date: 2026-05-08
content_hash: bfd99898f5d43957
---

# Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains

**Conference**: ECCV 2024  
**arXiv**: [2312.12098](https://arxiv.org/abs/2312.12098)  
**Code**: Yes ([https://github.com/dgist-cvlab/MultiDensityDG](https://github.com/dgist-cvlab/MultiDensityDG))  
**Area**: Autonomous Driving  
**Keywords**: LiDAR semantic segmentation, domain generalization, point cloud density, feature embedding, data augmentation

## TL;DR

This work proposes a Density-Discriminative Feature Embedding (DDFE) module that leverages the inherent density diversity in a single LiDAR source domain (dense nearby and sparse far away) to learn density-aware feature representations, achieving generalization to unseen domains under different sensor configurations without requiring target domain data.

## Background & Motivation

LiDAR semantic segmentation is crucial for autonomous driving, but model performance degrades significantly under domain shift. The primary cause of degradation is the difference in point cloud density distributions caused by varying LiDAR sensors (such as different channels, FOVs) and deployment environments. For instance, Waymo utilizes a 64-channel LiDAR ($2560\times64$ beams), while nuScenes employs a 32-channel LiDAR ($1080\times32$ beams), resulting in a stark difference in point cloud density.

**Limitations of Prior Work**:
- **UDA methods** (e.g., CoSMIX, LiDAR-UDA) require unlabeled target domain data, demanding retraining/fine-tuning for every domain shift.
- **Existing DG methods** (e.g., DGLSS, LiDomAug) treat point cloud density simply as a global attribute (64 channels = dense, 32 channels = sparse), neglecting the complexity of density variation along distance within a single LiDAR scan.

**Core Idea**: The point cloud density captured by a 64-channel LiDAR at a far distance (35m) can match the density of a 32-channel LiDAR at a medium distance (12m). In other words, LiDAR data from a single source domain inherently contains a density spectrum that spans various potential target domain densities. Leveraging this intrinsic density diversity for domain generalization represents a novel and reasonable perspective.

## Method

### Overall Architecture

The DDFE module contains four core components processed in sequence to output density-aware features for the 3D backbone to perform semantic segmentation:

1. **Point-voxel feature encoding**
2. **Beam density estimation**
3. **Density soft clipping**
4. **Density-aware embedding**

Additionally, a **Density augmentation** strategy is incorporated to expand the density spectrum of the training data.

### Key Designs

**Point-voxel feature encoding**:
- The input only uses 3D coordinates (excluding LiDAR intensity values since intensity distributions vary across different sensors).
- The point cloud coordinates are transformed into spherical coordinates $(\cos\theta, \sin\theta, \phi, r)$.
- Voxel-level features $F^v$ and point-level features $F^p$ are generated simultaneously. Voxel-level encoding bypasses intra-voxel local information via direct encoding, eliminating variances brought by grid-size variations.

**Beam density estimation**:
- Construct 1-D binary vectors $\mathbf{B}_h$ and $\mathbf{B}_v$ using LiDAR sensor configurations (horizontal/vertical beam numbers and FOVs), indicating whether each projection pixel has a beam passing through.
- Apply 1D Gaussian kernel convolutions with four standard deviations $\sigma_k = \{10, 30, 50, 70\}$ to obtain multi-scale smoothed beam densities.
- The final density is computed as $\mathcal{D}_i = [\sqrt{\hat{B}_h^{(k)} \cdot \hat{B}_v^{(k)} / r_i^2}]_{k=1}^4$, where the $r^2$ factor reflects the physical law of beam density decreasing quadratically with distance.

**Density soft clipping**: A $\tanh$ function is utilized to constrain the density values to a reasonable range of the source domain (10th to 90th percentiles), preventing the model from collapsing during inference when encountering extreme density values unseen in the source domain:

$$\mathcal{D}_i^c = \tanh\left(\frac{\mathcal{D}_i - m}{l}\right) \cdot l + m$$

**Density-aware embedding**:
- Point-level attention: $\hat{F}_i^p = f_p(\mathcal{D}_i^c) \odot F_i^p$
- Voxel-level attention: $\hat{F}_j^v = \text{Concat}(f_v(\mathcal{D}_j^c) \odot F_j^v, g(F_j^p))$
- Here, $f_p$ and $f_v$ are both two-layer 1D convolutions + sigmoid, enabling density-conditioned feature modulation.

**Density augmentation**:
- **Enhanced-Mix3D**: Based on Mix3D, random translations along the driving direction and rotations are added to simulate more diverse density variations.
- **Beam sampling**: Selectively removes specific LiDAR beams to enhance adaptation to lower densities.
- The two augmentations are applied each with a probability of 0.5.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{point}^{lovasz} + \mathcal{L}_{point}^{wce} + \mathcal{L}_{voxel}^{lovasz} + \mathcal{L}_{voxel}^{wce}$$

Lovász-Softmax loss and weighted cross-entropy loss are calculated at both point and voxel levels. The Adam optimizer is used with an initial learning rate of 1e-3, decaying by 0.99 every epoch, over 30 training epochs.

## Key Experimental Results

### Main Results

Comparison with domain generalization methods under the DGLSS setting (MinkowskiNet backbone, mIoU):

| Method | W→K | W→N | K→W | K→N | N→W | N→K |
|------|-----|-----|-----|-----|-----|-----|
| Base | 49.40 | 47.83 | 35.24 | 37.42 | 38.65 | 36.24 |
| IBN-Net | 51.13 | 44.72 | 36.99 | 38.74 | 36.53 | 36.93 |
| DGLSS | 51.23 | 49.61 | 40.67 | 44.83 | 40.93 | 38.98 |
| **Ours** | **57.07**| **56.75**| **42.73**| **49.43**| **45.98**| **46.52**|

Comparison under the LiDomAug setting (MinkNet42 backbone, voxel=5cm):

| Method | K→N | N→K |
|------|-----|-----|
| Base | 37.8 | 36.1 |
| Mix3D | 43.1 | 44.7 |
| PolarMix | 45.8 | 39.1 |
| LiDomAug | 39.2 | 37.9 |
| **Ours (v=5cm)** | **48.6** | **51.3** |
| **Ours (v=20cm)** | **50.1** | **46.3** |

### Ablation Study

Contribution of each component (MinkNet42, voxel=20cm):

| Point-voxel encoding | Density embedding | Density clipping | Density augmentation | K→N | N→K |
|------------|---------|---------|---------|-----|-----|
| | | | | 40.7 | 31.4 |
| ✓ | | | | 43.0 (+5.7%) | 35.0 (+11.5%) |
| ✓ | ✓ | | | 45.7 (+12.3%)| 40.5 (+29.0%)|
| ✓ | ✓ | ✓ | | 46.2 (+13.5%)| 41.8 (+33.1%)|
| ✓ | ✓ | ✓ | ✓ | **50.1 (+23.1%)**| **46.3 (+47.5%)**|

### Key Findings

1. **All components contribute to the performance**: The density-aware embedding yields the most significant improvement (+12.3%/+29.0%), followed by the point-voxel encoding, while density clipping and augmentation further boost the performance.
2. **Significantly outperforms DGLSS**: Yields an average improvement of +12.9% on unseen domains when trained on Waymo, and +15.8% when trained on nuScenes.
3. **Outperforms LiDomAug which requires multi-frame data**: While LiDomAug relies on ego-motion and sequence label data, this method achieves superior results using only a single frame.
4. **Voxel size of 20cm is a solid default choice**: Increasing voxel size from 5cm to 20cm reduces training time by 30.3% and inference time by 62.5% with comparable performance (slight decrease of 3.5% on MinkNet42 but a 4.7% increase on C&L).
5. **DDFE is highly lightweight**: Adds only about 23.8K parameters (+0.06%), with an additional inference overhead of only 8ms.
6. **Feature similarity analysis** confirms that DDFE effectively reduces the feature distance between the source domain (nuScenes) and the unseen domain (Waymo) within the same density intervals.

## Highlights & Insights

- **Novel and intuitive core insight**: Instead of treating different LiDAR configurations as distinct domains, treating the density variation along distance as natural multi-density domains is a very elegant problem reformulation.
- **Plug-and-play**: DDFE is a general module that can be seamlessly integrated into any voxel-based 3D backbone (e.g., MinkowskiNet, Cylinder3D).
- **No target domain data required**: Does not rely on ego-motion, sequence labels, or target domain data, achieving true single-source single-frame domain generalization.
- **Physical modeling of density**: Calculates expected density based on LiDAR beam configurations, and applies Gaussian smoothing to obtain multi-scale density representations, which has a stronger physical foundation than heuristic alternatives.

## Limitations & Future Work

- When the density spectra of the source and target domains do not overlap at all (extreme sensor differences), the effectiveness may be limited—density augmentation alleviates this to some extent but cannot fully solve it.
- Only validated on three datasets: Waymo, SemanticKITTI, and nuScenes; its generalization ability to a wider variety of sensors (e.g., Pandaset-64ch, SemanticPOSS-40ch) remains to be examined.
- Currently excludes LiDAR intensity information to improve domain generalization. However, intensity carries discriminative value for certain categories (e.g., road vs. vehicle), pointing to a potential direction for exploring domain-invariant intensity normalization.
- Variations in driving environments (e.g., different city layouts between the US and Singapore) are not considered; such semantic-level domain gaps can be as crucial as density gaps.

## Related Work & Insights

- **Relationship with DGLSS**: While DGLSS focuses on sparsity-invariant feature consistency, this work approaches the problem from a density-discriminative perspective, making them complementary. DGLSS requires tuning augmentation strategies for each dataset, whereas this method employs uniform hyperparameters.
- **Comparison with LiDomAug**: LiDomAug generates dense world models via multi-frame aggregation and random LiDAR configuration sampling, relying heavily on ego-motion information. This method is simpler and more efficient.
- The insight of density as a core factor for domain gap can be extended to other 3D perception tasks (e.g., object detection, panoramic segmentation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "single source as multiple density domains" perspective is highly novel.
- **Technical Quality**: ⭐⭐⭐⭐ — The module designs are physically grounded with comprehensive ablation studies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across two experimental settings and multiple backbones.
- **Practicality**: ⭐⭐⭐⭐⭐ — Lightweight, plug-and-play, single-frame with no extra data required.
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)
- [\[ECCV 2024\] Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation](train_till_you_drop_towards_stable_and_robust_source-free_unsupervised_3d_domain.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)

</div>

<!-- RELATED:END -->
