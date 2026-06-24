---
title: >-
  [Paper Note] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation
description: >-
  [ECCV 2024][Autonomous Driving][Semi-supervised learning] The proposed IT2 framework significantly improves semi-supervised LiDAR semantic segmentation by leveraging consistency learning between peer representations (range image + voxel grid) of LiDAR data as a novel form of perturbation, and introducing cross-distribution contrastive learning based on Gaussian Mixture Models (GMMs).
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Semi-supervised learning"
  - "LiDAR semantic segmentation"
  - "peer-representation consistency"
  - "contrastive learning"
  - "Gaussian Mixture Model"
date: 2026-05-08
content_hash: a0921905eee94d41
---

# ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.07171](https://arxiv.org/abs/2407.07171)  
**Code**: [https://github.com/yyliu01/IT2](https://github.com/yyliu01/IT2)  
**Area**: Autonomous Driving / 3D Semantic Segmentation  
**Keywords**: Semi-supervised learning, LiDAR semantic segmentation, peer-representation consistency, contrastive learning, Gaussian Mixture Model

## TL;DR

The proposed IT2 framework significantly improves semi-supervised LiDAR semantic segmentation by leveraging consistency learning between peer representations (range image + voxel grid) of LiDAR data as a novel form of perturbation, and introducing cross-distribution contrastive learning based on Gaussian Mixture Models (GMMs).

## Background & Motivation

**Background**: Outdoor LiDAR semantic segmentation is a foundational task in autonomous driving. Current methods extract features by converting point clouds into different representations (range image, voxel grid, BEV, etc.), but heavily rely on large-scale annotated data. Labeling LiDAR point clouds is extremely time-consuming and expensive, driving the development of semi-supervised learning (SSL) methods.

**Limitations of Prior Work**: 

**Limitations of Consistency Learning**: Existing SSL methods (such as LaserMix, Mean Teacher) only perform consistency learning on a **single LiDAR representation**, with limited forms of perturbation (such as data augmentation, network perturbation), which constrains their generalization capability. Different representations have their own distinct weaknesses—the range view loses spatial detail in projection (e.g., Pole category mIoU is only 52.02), whereas the voxel grid suffers from poor accuracy in distant sparse regions (e.g., Sidewalk category mIoU is only 69.50).

**Bottlenecks of Contrastive Learning**: Existing contrastive learning approaches randomly sample positive and negative embedding pairs from limited mini-batches, failing to comprehensively capture the full distribution of the embedding space. Under multi-representation scenarios, noisy predictions from one representation can also propagate to another through contrastive learning, leading to confirmation bias.

**Key Challenge**: The same 3D point cloud should share the same semantics across different representations—the same point in the range image and voxel grid must have the identical "terrain" label—yet their prediction confidence and error patterns are drastically different. This complementarity is entirely ignored by existing single-representation SSL methods.

**Key Insight**: The differences between different LiDAR representations can be treated as a natural form of "perturbation." Unlike traditional data augmentation perturbations, peer-representation perturbations intrinsically represent different ways of observing the same semantic information, providing much more effective consistency learning signals.

**Core Idea**: Achieve consistency learning through cross-peer pseudo-label supervision (range predictions supervise voxels, and vice versa), while modeling the cross-representation embedding distribution using a GMM to sample highly informative samples for contrastive learning.

## Method

### Overall Architecture

IT2 simultaneously trains two networks: a range network (FIDNet/ResNet34) processing range images, and a voxel network (Cylinder3D) processing voxel grids. The same point cloud is converted into both representations and fed into their respective networks. On unlabeled data, predictions from one representation are mapped into the other's representation space via representation conversion, serving as pseudo-labels for cross-supervision. Meanwhile, the embedding spaces of both networks are modeled using GMMs for cross-distribution contrastive learning.

### Key Designs

1. **Peer-Representation Consistency**: The core mechanism is to project predictions from the voxel network onto the range image space (or vice versa), which then act as pseudo-labels to supervise the peer network. Locally, this is implemented using bidirectional projection functions between representations:

    $\tilde{\mathbf{y}}^r(\omega^r) = \text{argmax}\ \Psi_{v \to r}^{(\omega^r)}(\hat{\mathbf{y}}^v), \quad \tilde{\mathbf{y}}^v(\omega^v) = \text{argmax}\ \Psi_{r \to v}^{(\omega^v)}(\hat{\mathbf{y}}^r)$

   where $\Psi_{v \to r} = \psi_{p \to r} \circ \psi_{v \to p}$ completes the projection between representations via 3D point space intermediate mapping. The total loss is the sum of labeled and unlabeled losses across both representations:

    $\ell_{\text{IT2}} = \ell_{\text{range}} + \ell_{\text{voxel}}$

   Design Motivation: Different representations have complementary error patterns for the same scene—the range view is weak on thin, tall objects (Pole), whereas voxels are weak on distant dense surfaces (Sidewalk). Cross-supervision leverages this complementarity to mitigate the confirmation bias inherent in single-representation pseudo-labeling.

2. **Cross-Distribution Contrastive Learning**: Instead of the conventional approach of sampling embeddings from a mini-batch, IT2 utilizes a **class-level Gaussian Mixture Model (GMM)** to model the joint embedding space of both representations. The parameters of the GMM are optimized via the EM algorithm, with the **pseudo-label confidence** integrated as a weight factor to suppress the influence of lower-quality labels:

    $\mathbf{P}_{\Gamma^y}(\mathbf{z}|y) = \sum_{m=1}^{M} \boldsymbol{\pi}_m^y \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_m^y, \boldsymbol{\Sigma}_m^y)$

   Each category uses $M=5$ Gaussian components. During training, virtual positive and negative prototype samples are sampled directly from the GMM distribution rather than being randomly selected from actual training samples:

    $\ell_{\text{cross}} = \sum_{(\mathbf{z},y) \in \mathcal{A}} \sum_{\mathbf{s} \in \mathcal{S}^y} -\log \frac{\exp(\mathbf{z} \cdot \mathbf{s} / \tau)}{\exp(\mathbf{z} \cdot \mathbf{s} / \tau) + \sum_{\mathbf{s}^- \in \bar{\mathcal{S}}^y} \exp(\mathbf{z} \cdot \mathbf{s}^- / \tau)}$

   Design Motivation: GMMs can capture the complete distribution characteristics (multi-modality, covariance structure) of the embedding space. Virtual prototypes sampled this way are more representative and informative than random samples, and they do not participate in backpropagation, thereby avoiding the propagation of cross-representation noise.

3. **Representation-Specific Augmentation**: Different representations are suited to different augmentation strategies—range images use multi-boxes CutMix, while voxels employ single-inclination LaserMix. This differs from past approaches which applied identical augmentations across all representations.

   Design Motivation: The range image is a 2D projection, suited for 2D spatial cutting augmentations; the voxel representation is a 3D structure, suited for 3D augmentations based on laser beam inclinations. Tailored augmentation strategies yield better generalization.

### Loss & Training

- Total training loss: $\mathcal{L} = \ell_{\text{IT2}} + \ell_{\text{cross}}$, optimized end-to-end.
- Segmentation loss $\ell$ consists of Cross-Entropy + Lovász-Softmax.
- Contrastive learning temperature coefficient $\tau = 0.1$.
- The embedding projector is a 3-layer MLP yielding a 64-dimensional output.
- The number of GMM components is $M=5$, with uniform weights $\pi_m^y = 1/M$.

## Key Experimental Results

### Main Results (Uniform Sampling)

**nuScenes Dataset (mIoU%)**:

| Method | Representation | 1% | 10% | 20% | 50% |
|------|------|-----|------|------|------|
| LaserMix | Range | 49.5 | 68.2 | 70.6 | 73.0 |
| **IT2 (Ours)** | **Range** | **56.5** | **71.3** | **73.4** | **74.0** |
| LaserMix | Voxel | 55.3 | 69.9 | 71.8 | 73.2 |
| **IT2 (Ours)** | **Voxel** | **57.5** | **72.1** | **73.6** | **74.1** |

**SemanticKITTI + ScribbleKITTI (mIoU%)**:

| Method | Representation | KITTI 1% | KITTI 10% | Scribble 1% | Scribble 10% |
|------|------|---------|----------|------------|-------------|
| LaserMix | Range | 43.4 | 58.8 | 38.3 | 54.4 |
| **IT2** | **Range** | **51.9** | **60.3** | **46.6** | **57.1** |
| LaserMix | Voxel | 50.6 | 60.0 | 44.2 | 53.7 |
| **IT2** | **Voxel** | **52.0** | **61.4** | **47.9** | **56.7** |

Under the 1% label setting on nuScenes, Range improves by +7.0%, and Voxel improves by +2.2%. Under 1% labeling on ScribbleKITTI, Range improves by +8.3%.

### Ablation Study

**Contribution of Each Component (nuScenes + ScribbleKITTI, Range Representation)**:

| IT2 Architecture | Contrastive Learning | Augmentation | nuScenes 10% | ScribbleKITTI 10% | Notes |
|---------|---------|------|-------------|-------------------|------|
| ✗ | ✗ | ✗ | 60.8 | 50.0 | CPS Baseline |
| ✓ | ✗ | ✗ | 67.3 (+6.5) | 53.2 (+3.2) | Peer-representation consistency |
| ✓ | ✓ | ✗ | 70.3 (+9.5) | 54.9 (+4.9) | + Cross-distribution contrastive learning |
| ✓ | ✓ | ✓ | **71.3 (+10.5)** | **57.1 (+7.1)** | + Representation-specific augmentation |

All three components make obvious contributions, with the IT2 architecture itself contributing the most (+6.5%), contrastive learning further lifting performance by roughly 3%, and the augmentation strategy contributing another 1-2%.

### Key Findings

- **Comparison of Contrastive Learning**: Compared to ContrasSeg (state-of-the-art pixel-level contrastive learning), the GMM-based method improves mIoU by about 1% on nuScenes 10% (range: 71.3 vs 70.3, voxel: 72.1 vs 71.2), demonstrating that distribution sampling outperforms random sampling.
- **Temperature Coefficient**: $\tau=0.10$ is the optimal value (on nuScenes 10%); values too large or too small degrade performance.
- **Scalability to Multiple Representations**: Integrating BEV (PolarNet) as a third representation yields further improvements (+6.3% when paired with range, +7.8% when paired with voxel), proving that the method is extensible to arbitrary combinations of representations.
- **Partial/Significant Sampling**: On SemanticKITTI partial 5%, it outperforms GPC by +4.5% (43.8 vs 40.2), and on significant 10% it beats lim3D by +2.2%.
- **Augmentation Tactics**: The combination of multi CutMix for range and 1-inc LaserMix for voxel is optimal, showing a +1.3% boost for the range representation compared to using LaserMix uniformly.

## Highlights & Insights

- Excellent core insight: Different LiDAR representations are distinct "observations" of the same 3D scene, which naturally satisfies the clustering hypothesis—this presents a more fundamental form of perturbation than manual data augmentation.
- Modeling the embedding distribution with GMM before sampling is an elegant design: it not only covers the full distribution but also mitigates the negative impact of noisy pseudo-labels via confidence weighting.
- Cross-representation pseudo-label supervision cleverly capitalizes on the complementary strengths of different representations, diminishing the confirmation bias inherent to single-representation SSL.
- High degree of framework versatility: It extends seamlessly to any combination of LiDAR representations (range+voxel+BEV) and consistently yields substantial improvements.

## Limitations & Future Work

- Employing two representations requires training two independent networks, which doubles the training computational overhead.
- Projection mappings between representations heavily rely on precise point cloud coordinates; sensor noise might degrade the quality of cross-representation labels.
- GMM assumes a Gaussian mixture distribution for the embedding space, which might not be flexible enough for highly complex categories.
- Worth exploring: Incorporating the camera modality as a third representation to achieve unified LiDAR-Camera SSL fusion.
- Worth exploring: Extending the concept of peer-representation consistency to other SSL fields (e.g., multimodal SSL in medical imaging).

## Related Work & Insights

- LaserMix introduced laser beam-mixing augmentations for single representations. IT2 advances beyond this by treating the "asymmetry between representations" itself as a higher-level perturbation.
- Contrastive learning in GPC and lim3D is restricted by actual physical samples; IT2 overcomes this bottleneck using virtual GMM prototypes.
- CPS (Cross Pseudo Supervision) in 2D image SSL uses two identical networks for cross-supervision. IT2 extends this to cross-supervising between two **different representations**, naturally introducing superior perturbation diversity.
- Inspiration: The intrinsic complementarity of multi-view/multimodal data is a valuable asset for semi-supervised learning, proving more effective than synthetic augmentations.

## Rating

- Novelty: ⭐⭐⭐⭐ Peer-representation consistency learning and GMM-based contrastive learning are both significant and meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 3 sampling strategies × extensive ablation experiments + representation extension + comparison of contrastive learning. Extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Thorough motivation analysis (illustrated well by the complementarity diagram in Fig.1) and complete, clear mathematical formalization.
- Value: ⭐⭐⭐⭐ Significantly drives the LiDAR SSL field forward, with a +7% mIoU boost under the nuScenes 1% setting carrying strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CSOT: Cross-Scan Object Transfer for Semi-Supervised LiDAR Object Detection](csot_cross-scan_object_transfer_for_semi-supervised_lidar_object_detection.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](../../CVPR2025/autonomous_driving/exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)
- [\[ECCV 2024\] RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation](rapid-seg_range-aware_pointwise_distance_distribution_networks_for_3d_lidar_segm.md)

</div>

<!-- RELATED:END -->
