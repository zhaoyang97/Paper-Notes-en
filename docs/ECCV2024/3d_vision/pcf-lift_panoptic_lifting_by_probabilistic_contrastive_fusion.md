---
title: >-
  [Paper Note] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion
description: >-
  [ECCV 2024][3D Vision][Panoptic Lifting] This paper proposes PCF-Lift, which replaces deterministic features with probabilistic feature embeddings (multivariate Gaussian distributions) and combines contrastive loss based on the Probabilistic Product Kernel (PP Kernel) with cross-view constraints. This effectively addresses the issues of inconsistent segmentation and inconsistent IDs in 2D segmentation, significantly outperforming state-of-the-art methods on the ScanNet and Me…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Panoptic Lifting"
  - "Probabilistic Feature Embedding"
  - "Contrastive Learning"
  - "Multi-view Fusion"
  - "NeRF"
date: 2026-05-08
content_hash: dec9f20ceae53d63
---

# PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion

**Conference**: ECCV 2024  
**arXiv**: [2410.10659](https://arxiv.org/abs/2410.10659)  
**Code**: [GitHub](https://github.com/Runsong123/PCF-Lift)  
**Area**: 3D Vision  
**Keywords**: Panoptic Lifting, Probabilistic Feature Embedding, Contrastive Learning, Multi-view Fusion, NeRF  

## TL;DR

This paper proposes PCF-Lift, which replaces deterministic features with probabilistic feature embeddings (multivariate Gaussian distributions) and combines contrastive loss based on the Probabilistic Product Kernel (PP Kernel) with cross-view constraints. This effectively addresses the issues of inconsistent segmentation and inconsistent IDs in 2D segmentation, significantly outperforming state-of-the-art methods on the ScanNet and Messy Room datasets.

## Background & Motivation

1. 3D panoptic segmentation requires simultaneous prediction of semantic and instance labels, representing a crucial task for achieving holistic scene understanding.
2. Due to the scarcity of 3D annotated data, recent methods have turned to leverage the panoptic segmentation results of 2D foundation models for "panoptic lifting."

**Inconsistent ID Problem**: The same 3D object is assigned different instance IDs by the 2D segmenter in different views.

**Inconsistent Segmentation Problem**: The same object is segmented into different parts in different views (e.g., a chair is cut in half in View 1 but remains complete in View 2).
5. Existing methods (Panoptic Lifting, Contrastive Lift) use deterministic feature embeddings, which are not robust to noise.
6. Deterministic features cannot model uncertainty, leading to unstable training when facing inconsistent segmentation, and significantly reduced performance in complex scenarios.

## Method

### Overall Architecture

PCF-Lift builds a 3D panoptic field based on the TensoRF architecture, including semantic, instance, density, and color fields. The core innovations focus on the design of the instance field:

1. **Training Phase**: Rays are sampled from two views to obtain probabilistic feature maps via volume rendering. The instance field is optimized using probabilistic contrastive loss and cross-view constraints.
2. **Inference Phase**: A prototype feature set is extracted through the Multi-view Object Association (MVOA) algorithm to generate matching and consistent panoptic segmentation results.

### Key Designs

#### Module 1: Probabilistic Feature Embedding

Each 3D point in the instance field is mapped to a multivariate Gaussian distribution random variable:

$$\mathcal{F} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \text{diag}(\sigma^{(1)^2}, \sigma^{(2)^2}, \cdots, \sigma^{(N)^2})$$

where $\boldsymbol{\mu} \in \mathbb{R}^N$ is the mean vector (central feature values), and $\boldsymbol{\Sigma}$ is the diagonal covariance matrix (uncertainty). The instance field predicts $(\boldsymbol{\mu}, \sigma^2) \in \mathbb{R}^{2N}$ for each query point $\mathbf{x} \in \mathbb{R}^3$ ($N=3$ in experiments).

The similarity between two Gaussian distributions is measured by the **Probabilistic Product Kernel (PP Kernel)**:

$$K_\rho(\mathcal{F}_i, \mathcal{F}_j) = \left(\prod_{d=1}^{N} \frac{\sigma_i^{(d)^2}/\sigma_j^{(d)^2} + \sigma_j^{(d)^2}/\sigma_i^{(d)^2}}{2}\right)^{-\frac{1}{2}} \exp\left(-\sum_{d=1}^{N} \frac{(\mu_i^{(d)} - \mu_j^{(d)})^2}{4(\sigma_i^{(d)^2} + \sigma_j^{(d)^2})}\right)$$

The output range of the PP Kernel is $[0, 1]$, which is the same as the RBF kernel used in deterministic methods but offers stronger expressive power.

**Theoretical Property (Corollary 1)**: When the covariances of all Gaussian distributions are isotropic and fixed ($\Sigma_i = \Sigma_j = \sigma \mathbf{I}$), the PP Kernel degenerates into the RBF kernel. Thus, the deterministic method is a special case of the probabilistic method.

#### Module 2: Probabilistic Contrastive Loss and Cross-View Constraints

**Pixel-level contrastive loss** replaces the RBF kernel with the PP Kernel:

$$\mathcal{L}_{\text{pixel-contra}} = -\frac{1}{|\Omega|} \sum_{u \in \Omega} \log \frac{\sum_{u' \in \Omega} \mathbf{1}_{(u,u')} \exp(K_\rho(\mathcal{F}_u, \mathcal{F}_{u'}))}{\sum_{u' \in \Omega} \exp(K_\rho(\mathcal{F}_u, \mathcal{F}_{u'}))}$$

**Concentration loss** encourages features of the same instance to cluster together:

$$\mathcal{L}_{\text{concen}} = -\frac{1}{|\Omega|} \sum_{u \in \Omega} \log K_\rho\left(\mathcal{F}_u, \frac{\sum_{u'} \mathbf{1}_{(u,u')} \mathcal{F}_{u'}}{\sum_{u'} \mathbf{1}_{(u,u')}}\right)$$

**Cross-view constraint** enhances the consistency of features of the same object across different views:

$$\mathcal{L}_{\text{cross}} = -\frac{1}{|\mathcal{P}|} \sum_{(\mathcal{F}_r, \mathcal{F}_s) \in \mathcal{P}} \log K_\rho(\mathcal{F}_r, \mathcal{F}_s)$$

where positive sample pairs are defined as $\mathcal{P} = \{(\mathcal{F}_r, \mathcal{F}_s) \mid K_\rho(\mathcal{F}_r, \mathcal{F}_s) > \tau\}$ with threshold $\tau = 0.9$.

#### Module 3: Multi-view Object Association (MVOA) Algorithm

During inference, a prototype feature set $\mathcal{D}$ is extracted using a greedy algorithm resembling NMS:

1. **Instance Grouping**: For each view, pixel features with the same instance ID are averaged to form a group feature $\mathcal{C}_l^p$, and the feature concentration score $\mathcal{S}_l^p = \Phi(\mathcal{C}_l^p)$ is calculated.
2. **Multi-view Matching**: An undirected similarity graph $G = (\mathcal{C}, E)$ is constructed. Nodes with the highest scores are greedily selected to join the prototype set $\mathcal{D}$, while nodes with similarity exceeding a threshold $\mathcal{T}$ to the selected nodes are suppressed.
3. **Mask Generation**: For any view, foreground pixels are assigned instance labels based on matching results with the most similar prototype in $\mathcal{D}$.

### Loss & Training

Total loss function:

$$\mathcal{L} = \mathcal{L}_{\text{contra}} + w_{\text{cross}} \mathcal{L}_{\text{cross}} + w_{\text{reg}} \mathcal{L}_{\text{reg}}$$

- $w_{\text{cross}} = 0.05$ (active only during the last few epochs), set to 0 in earlier stages
- $w_{\text{reg}} = 0.001$, covariance regularization $\mathcal{L}_{\text{reg}} = \log(\prod_{d=1}^{N} \sigma^{(d)^2})$
- The instance field uses a 5-layer shallow MLP with a slow-fast architecture
- The probabilistic feature dimension is $N = 3$

## Key Experimental Results

### Main Results

**ScanNet Dataset (12 scenes)**:

| Method | Conference | Type | $\text{SQ}^{\text{scene}}$ | $\text{RQ}^{\text{scene}}$ | $\text{PQ}^{\text{scene}}$ |
|------|------|------|:---:|:---:|:---:|
| DM-NeRF | ICLR'23 | 3D Panoptic Seg. | 53.3% | 46.1% | 41.7% |
| PNF | CVPR'22 | 3D Panoptic Seg. | 63.0% | 50.7% | 48.3% |
| Panoptic Lifting | CVPR'23 | 2D Panoptic Lifting | 73.5% | 65.0% | 58.9% |
| Contrastive Lift | NeurIPS'23 | 2D Panoptic Lifting | 75.7% | 63.6% | 62.0% |
| **PCF-Lift (Ours)** | - | 2D Panoptic Lifting | **78.5%** | **65.4%** | **63.5%** |

**Messy Room Dataset (Average PQ^scene)**:

| Method | 25 Objects | 50 Objects | 100 Objects | 500 Objects | Mean |
|------|:---:|:---:|:---:|:---:|:---:|
| Panoptic Lifting | 69.4% | 70.5% | 63.1% | 50.0% | 63.2% |
| Contrastive Lift | 77.7% | 75.7% | 68.9% | 53.8% | 69.0% |
| **PCF-Lift** | **81.0%** | **78.9%** | **74.4%** | **59.6%** | **73.4%** |

### Ablation Study

| Model | Feature Space | Clustering Method | $\text{PQ}^{\text{scene}}$ |
|------|---------|---------|:---:|
| (a) Contrastive Lift | Deterministic | HDBSCAN | 69.0% |
| (b) | Deterministic | MVOA | 70.4% |
| (d) | Probabilistic Gaussian | MVOA | 72.3% |
| (f) **PCF-Lift** | Probabilistic Gaussian + Cross-view Constraint | MVOA | **73.4%** |

### Key Findings

1. **Probabilistic vs. Deterministic**: Probabilistic feature embedding increases PQ from 70.4% to 72.3% (+1.9%), demonstrating the effectiveness of modeling uncertainty with Gaussian distributions.
2. **Universality of MVOA Algorithm**: Even when applied to deterministic methods, it brings a +1.4% improvement (69.0% → 70.4%).
3. **Cross-View Constraints**: Further improves results by +1.1% (72.3% → 73.4%), enhancing feature consistency across multiple views.
4. **Uncertainty Analysis**: The learned high-covariance regions are primarily distributed near instance boundaries, conforming to intuition.
5. **Robustness**: Consistently outperforms deterministic methods under different 2D segmentation models and noise levels.

## Highlights & Insights

- Introducing probabilistic modeling to panoptic lifting is a highly natural and effective design, as 2D segmentation inherently contains substantial uncertainty.
- The theoretical analysis of the PP Kernel elegantly proves that the probabilistic method generalizes the deterministic one (RBF Kernel $\subset$ PP Kernel).
- The strategy of enabling the cross-view constraint only in the late training stages is well-designed, avoiding noise introduced by unreliable early features.
- As a general clustering method, the MVOA algorithm can be plugged and played to boost the performance of other approaches.

## Limitations & Future Work

- The method relies on the reconstruction quality of TensoRF; panoptic segmentation fails in regions where geometric reconstruction fails.
- The probabilistic feature dimension is only 3D, which might be insufficient in more complex scenes.
- The cross-view constraint requires additional dual-view sampling, increasing the training cost.
- Evaluation was performed only on indoor scenes, leaving the applicability to large-scale outdoor scenes unknown.

## Related Work & Insights

- **Contrastive Lift**: A contrastive learning baseline using deterministic feature embeddings, which PCF-Lift directly improves upon.
- **Panoptic Lifting**: Learns instance representations via ID permutation fitting, which has limited scalability.
- **PP Kernel**: Though the Probabilistic Product Kernel has been studied in machine learning, this paper applies it to the panoptic lifting scenario for the first time.
- **Insights**: The core idea of probabilistic feature embedding can be extended to other tasks requiring multi-view fusion (e.g., 3D semantic segmentation, scene editing).

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](../../AAAI2026/3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[ECCV 2024\] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](../../CVPR2025/3d_vision/layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[CVPR 2026\] Fusion of Depth and Semantics for Probabilistic Floorplan Localization](../../CVPR2026/3d_vision/fusion_of_depth_and_semantics_for_probabilistic_floorplan_localization.md)
- [\[ECCV 2024\] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)

</div>

<!-- RELATED:END -->
