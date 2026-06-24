---
title: >-
  [Paper Note] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification
description: >-
  [ECCV 2024][Human Understanding][Unsupervised Person Re-Identification] A Multi-Memory Matching (MMM) framework is proposed for unsupervised visible-infrared person re-identification. It establishes reliable cross-modal correspondences through three modules: Cross-Modal Clustering (CMC), Multi-Memory Learning and Matching (MMLM), and Soft Cluster-level Alignment loss (SCA), achieving a Rank-1 accuracy of 61.6% on SYSU-MM01 and 89.7% on RegDB.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Unsupervised Person Re-Identification"
  - "Visible-Infrared Cross-Modal"
  - "Multi-Memory Matching"
  - "Pseudo Label"
  - "Clustering"
date: 2026-05-08
content_hash: e4e747e39aaa7e66
---

# Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification

**Conference**: ECCV 2024  
**arXiv**: [2401.06825](https://arxiv.org/abs/2401.06825)  
**Code**: [GitHub](https://github.com/shijiangming1/MMM)  
**Area**: Human Understanding  
**Keywords**: Unsupervised Person Re-Identification, Visible-Infrared Cross-Modal, Multi-Memory Matching, Pseudo Label, Clustering

## TL;DR

A Multi-Memory Matching (MMM) framework is proposed for unsupervised visible-infrared person re-identification. It establishes reliable cross-modal correspondences through three modules: Cross-Modal Clustering (CMC), Multi-Memory Learning and Matching (MMLM), and Soft Cluster-level Alignment loss (SCA), achieving a Rank-1 accuracy of 61.6% on SYSU-MM01 and 89.7% on RegDB.

## Background & Motivation

Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to retrieve pedestrians across visible and infrared modalities without relying on annotations, which is a key technology for achieving 24-hour intelligent surveillance. Existing methods generate pseudo-labels and establish cross-modal correspondences based on clustering, but suffer from the following core problems:

**Unreliable cross-modal correspondences**: The authors introduced the ARI (Adjusted Rand Index) metric for evaluation and found that although existing methods perform well in terms of retrieval metrics, the quality of their cross-modal correspondences is poor (low ARI values).

**Insufficient single-memory representation**: Existing methods represent an identity using a single memory (i.e., a single cluster center), which fails to capture fine-grained variations such as multi-view and multi-pose representations of individuals, leading to highly noisy cross-modal matching.

**Paradox phenomenon**: Different pedestrians with overlapping attributes are further confused due to noisy correspondences. Although this may increase intra-class feature similarity and bring improvement in metrics, the actual precise retrieval capability is limited.

The core insight of this paper: **Compared to a single memory, multiple memories can more completely represent the diverse features of an identity (e.g., front, back views), thereby establishing more reliable cross-modal correspondences.**

## Method

### Overall Architecture

MMM employs ResNet50 (pre-trained on ImageNet) as the shared backbone network to extract 2048-dimensional features. The overall pipeline is: (1) The CMC module generates pseudo-labels; (2) The MMLM module establishes cross-modal correspondences through multi-memory matching; (3) The SCA loss reduces the modality gap and mitigates the impact of noisy pseudo-labels.

### Key Designs

1. **Cross-Modal Clustering (CMC)**: The foundational module for generating pseudo-labels.

    - The DBSCAN algorithm is used to cluster visible samples, infrared samples, and their mixed samples separately: $Y^t = DBSCAN(F^t)$.
    - Unlike existing methods, it performs not only intra-modality clustering ($t=v$ or $t=r$) but also joint inter-modality clustering ($t=\{v,r\}$), indirectly establishing cross-modal correspondences.
    - Three memories are computed for each cluster: visible memory $C_{V^p}$, infrared memory $C_{R^p}$, and mixed memory $C_{VR^p}$.
    - Optimization is based on the ClusterNCE contrastive loss: $L_{CMC} = L_V + L_R + L_{VR}$.

2. **Multi-Memory Learning and Matching (MMLM)**: The core innovation to establish reliable cross-modal correspondences.

    - **Multi-memory learning**: A single cluster is further subdivided into $n$ sub-clusters, and the center of each sub-cluster acts as a memory. The intra-sub-cluster distance is minimized via K-Means:
    $\min_{F_{C_{V_i^p}}} \sum_{i=1}^{n} \|f^v - K_{C_{V_i^p}}\|_2^2$
    - For example, Memory 1 captures frontal features, while Memory 2 captures back-view features, representing the individual more comprehensively.
    - **Multi-memory matching**: The cross-modal matching problem is formulated as weighted bipartite graph matching. The cost matrix is designed as the sum of nearest-neighbor distances between multiple memories:
    $M(K_{C_{V^p}}, K_{C_{R^{p'}}}) = \sum_{i=1}^{n} \min_{j \in \{1,...,n\}} \|K_{V_i^p} - K_{R_j^{p'}}\|_2$
    - The Hungarian algorithm is used to solve for the optimal matching $Q$ to transfer infrared pseudo-labels to the visible domain: $Y^v := QY^r$.

3. **Soft Cluster-level Alignment Loss (SCA)**: Mitigates the impact of noisy pseudo-labels and narrows the modality gap.

    - **Confidence estimation**: A two-component Gaussian Mixture Model (GMM) is used to model the loss distribution, calculating the label confidence $W^v$ for each sample via posterior probability.
    - **Confidence-weighted memory update**: Memories are updated with confidence weights to reduce the influence of noisy samples: $C_{V^p} := \frac{1}{N_p} \sum_i f(V_i^p) W_{V_i^p}$.
    - **Intra-modality alignment (Intra)**: Aligns samples of the same ID to their confidence-weighted cluster centers: $L_{Intra} = \sum_p \sum_{f^v} \|f^v - C_{V^p}\|_2^2 + \sum_p \sum_{f^r} \|f^r - C_{R^p}\|_2^2$.
    - **Inter-modality alignment (Inter)**: MMD² (Maximum Mean Discrepancy) is used to measure the difference in feature distributions of the same ID across the two modalities. Minimizing this difference achieves a soft many-to-many alignment:
    $L_{Inter} = \frac{1}{P} \sum_p \frac{1}{2}[D(F_p^v, sg(F_p^r)) + D(F_p^r, sg(F_p^v))]$
    - A stop-gradient instruction is used to prevent mutual collapse between the two modalities.
    - Total SCA loss: $L_{SCA} = \lambda_{Intra} L_{Intra} + \lambda_{Inter} L_{Inter}$.

### Loss & Training

Total loss: $L_{overall} = L_{CMC} + L_{SCA}$

- Backbone: ResNet50, pre-trained on ImageNet.
- Trained for 80 epochs. At each step, 8 IDs are sampled, with 4 visible + 4 infrared images selected per ID.
- Image size is 288×144, with random flipping and cropping augmentation.
- SGD optimizer with momentum=0.9, weight decay=5e-4.
- Intra loss is added from the 1st epoch, and Inter loss is added from the 15th epoch.
- Temperature coefficient $\tau=0.05$, DBSCAN parameters eps=0.6, min_samples=4.
- Optimal hyperparameters: $n=4$ (number of memories), $\lambda_{Intra}=0.5$, $\lambda_{Inter}=0.05$.

## Key Experimental Results

### Main Results

**SYSU-MM01 All Search & RegDB Visible2Thermal**

| Method | Type | SYSU R-1 | SYSU mAP | RegDB R-1 | RegDB mAP |
|------|------|----------|----------|-----------|-----------|
| ADCA | USL | 45.5 | 42.7 | 67.2 | 64.1 |
| ADCA+MMM | USL | 49.7 | 44.7 | 77.8 | 70.9 |
| GUR* | USL | 61.0 | 57.0 | 73.9 | 70.2 |
| PCLHD | USL | 64.4 | 58.7 | 84.3 | 80.7 |
| **MMM** | **USL** | **61.6** | **57.9** | **89.7** | **80.5** |
| **MMM+PCLHD** | **USL** | **65.9** | **61.8** | **89.6** | **83.7** |
| DPIS | Semi | 58.4 | 55.6 | 62.3 | 53.2 |
| AGW | Sup. | 47.5 | 47.7 | 70.1 | 66.4 |

MMM surpasses several semi-supervised and supervised methods under the unsupervised setting. On RegDB, compared to GUR, it improves Rank-1 by +15.8% and mAP by +10.3%.

### Ablation Study

| Configuration | SYSU R-1 | SYSU mAP | Indoor R-1 | Indoor mAP |
|------|----------|----------|------------|------------|
| Baseline (CMC only) | 51.74 | 49.81 | 56.34 | 64.46 |
| + MMLM | 55.15 | 52.21 | 58.76 | 65.47 |
| + MMLM + Intra | 58.48 | 55.05 | 62.19 | 68.09 |
| + MMLM + Inter | 57.26 | 53.81 | 60.26 | 66.66 |
| + MMLM + Intra + Inter | **61.56** | **57.92** | **64.37** | **70.40** |

### Key Findings

- The MMLM module yields an improvement of +3.41% in Rank-1, validating that multiple memories establish cross-modal correspondences more effectively than a single memory.
- Intra and Inter losses complement each other. The complete SCA loss improves Rank-1 by +9.82% and mAP by +8.11% compared to the baseline.
- The number of memories $n=4$ is optimal, indicating that too few memories cannot express diversity, while too many introduce noise.
- Visualization analysis shows that with MMM, the mean intra-modality distance decreases and the mean inter-modality distance increases, making the feature distribution more discriminative.
- The ARI metric indicates that the reliability of cross-modal correspondences of MMM is significantly superior to methods like GUR.

## Highlights & Insights

1. **Discovery of an important paradox**: Revealing the contradictory phenomenon in existing USL-VI-ReID methods where cross-modal correspondences achieve good retrieval results but are actually highly unreliable.
2. **Concept of multi-memory**: Splitting a single cluster center into multiple sub-cluster centers to describe identity diversity in a more fine-grained manner, which is an effective improvement over the Cluster-Contrast paradigm.
3. **GMM confidence estimation**: Utilizing loss distribution modeling to soften the impact of noisy pseudo-labels, which is more elegant than hard threshold filtering.
4. **Method versatility**: MMM can serve as a plug-and-play module to enhance other methods (e.g., ADCA+MMM, MMM+PCLHD), validating the broad applicability of the framework.

## Limitations & Future Work

1. The authors admit there is still a gap compared to supervised methods (e.g., DEEN Rank-1 74.7% vs MMM 61.6%), mainly limited by the lack of cross-modal data annotation.
2. The number of sub-clusters for multi-memory, $n$, needs to be set manually, and different datasets/identities may require different values.
3. DBSCAN clustering parameters are sensitive to the results and require careful tuning.
4. Leveraging vision-language pre-trained models such as CLIP to introduce semantic priors could be explored to enhance the quality of cross-modal matching.
5. Computing Hungarian matching for multi-memory may become a bottleneck in large-scale scenarios.

## Related Work & Insights

- **Cluster-Contrast**: A contrastive learning method using a single cluster center; MMM is a generalization of its memory representation.
- **PGM**: Models cross-modal matching as bipartite graph matching, which inspired the matching strategy of MMLM.
- **DivideMix**: Uses GMM to model loss distributions to handle noisy labels, which inspired the confidence estimation of SCA.
- **Insights**: Multi-memory representation and soft alignment strategies can be generalized to other unsupervised cross-domain/cross-modal matching tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-memory matching idea is novel, and the ARI metric reveals the blind spots of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on two standard datasets with three settings (supervised/semi-supervised/unsupervised) for comparison, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Deep problem analysis, and the observation of the paradox phenomenon is highly valuable.
- **Value**: ⭐⭐⭐⭐ Achieves new SOTA in unsupervised VI-ReID, and the framework can serve as a plug-and-play module compatible with other methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BIT: Matching-based Bi-directional Interaction Transformation Network for Visible-Infrared Person Re-Identification](../../CVPR2026/human_understanding/bit_matching-based_bi-directional_interaction_transformation_network_for_visible.md)
- [\[AAAI 2026\] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](../../AAAI2026/human_understanding/modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)
- [\[CVPR 2026\] Spatial-Frequency Collaborative Learning for Occluded Visible-Infrared Person Re-Identification](../../CVPR2026/human_understanding/spatial-frequency_collaborative_learning_for_occluded_visible-infrared_person_re.md)
- [\[CVPR 2026\] MFEN: Multi-Frequency Expert Network for Visible-Infrared Person Re-ID](../../CVPR2026/human_understanding/mfen_multi-frequency_expert_network_for_visible-infrared_person_re-id.md)
- [\[CVPR 2026\] Towards Cross-Modal Preservation, Consistency and Alignment for Privacy-Preserving Visible-Infrared Person Re-Identification](../../CVPR2026/human_understanding/towards_cross-modal_preservation_consistency_and_alignment_for_privacy-preservin.md)

</div>

<!-- RELATED:END -->
