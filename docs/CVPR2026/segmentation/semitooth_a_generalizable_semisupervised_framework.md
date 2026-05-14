---
title: >-
  [Paper Note] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR 2026][Segmentation][Semi-supervised learning] This paper proposes SemiTooth, a framework that addresses distribution discrepancies across multi-source CBCT data in semi-supervised tooth segmentation via a multi-tea…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Semi-supervised learning"
  - "multi-source data"
  - "tooth segmentation"
  - "CBCT"
  - "pseudo-labels"
date: 2026-05-08
content_hash: 55f9ee2211b63dd8
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)
**Code**: None
**Area**: Segmentation / Medical Imaging
**Keywords**: Semi-supervised learning, multi-source data, tooth segmentation, CBCT, pseudo-labels

## TL;DR
This paper proposes SemiTooth, a framework that addresses distribution discrepancies across multi-source CBCT data in semi-supervised tooth segmentation via a multi-teacher–multi-student architecture and a Stricter Weighted Confidence (SWC) constraint, achieving state-of-the-art performance on the newly constructed MS3Toothset dataset.

## Background & Motivation

1. **Background**: CBCT (Cone Beam CT) tooth structure segmentation is a fundamental task in clinical dentistry. Fully supervised methods have achieved notable progress, but voxel-level annotation is time-consuming and costly, leaving large amounts of de-identified, unlabeled CBCT data underutilized.
2. **Limitations of Prior Work**: Semi-supervised medical image segmentation (SSMIS) methods such as Mean Teacher can exploit unlabeled data, but are primarily designed for single-source settings. In practice, CBCT scans acquired across different hospitals and devices exhibit significant distribution shifts in density, intensity, and feature distributions, and naïvely mixing them leads to poor generalization. Moreover, publicly available multi-source CBCT tooth segmentation datasets are scarce.
3. **Key Challenge**: The domain gap among multiple sources makes unified training difficult—a simple single-teacher–single-student framework cannot simultaneously accommodate the feature heterogeneity of multiple data sources, and pseudo-label quality degrades severely in cross-source scenarios.
4. **Goal**: ① Construct a multi-source semi-supervised CBCT tooth segmentation dataset; ② design a semi-supervised framework capable of handling multi-source distribution discrepancies.
5. **Key Insight**: Assign dedicated student networks to different data sources for source-specific learning, provide supervision via corresponding teacher networks, and design region-level confidence constraints to filter noisy pseudo-labels.
6. **Core Idea**: A multi-teacher–multi-student branched framework combined with region-level strict confidence constraints, enabling each data source to be effectively utilized in semi-supervised learning.

## Method

### Overall Architecture
SemiTooth reorganizes all data sources into three subsets: **main** (labeled data from the primary source), **other** (unlabeled data from other sources), and **mixed** (unlabeled samples whose distribution is similar to the primary source, identified via Wasserstein distance). Three student networks are assigned to the three subsets respectively, while two teacher networks supervise the students handling the *mixed* and *other* subsets. Teachers are updated via EMA: $\theta_t^{(k)} \leftarrow \gamma \theta_t^{(k-1)} + (1-\gamma) \theta_s^{(k)}$.

### Key Designs

1. **Multi-Teacher–Multi-Student Architecture (SemiTooth)**

    - **Function**: Enables source-specific learning within dedicated student networks for each data source, avoiding interference from multi-source mixed training.
    - **Mechanism**: Unlike Mean Teacher (single teacher–single student), SemiTooth equips each data subset with an independent student network. Two teachers update their parameters from the corresponding students via EMA, providing stable pseudo-label supervision. Student networks share similar architectures to facilitate knowledge transfer while maintaining sufficient diversity. The *mixed* subset serves as a distributional bridge connecting the primary source and other sources.
    - **Design Motivation**: A single teacher–student framework produces unstable pseudo-labels in multi-source scenarios and cannot distinguish the feature styles of different sources. The "dedicated branch per source" design allows each student to learn source-aware knowledge representations.

2. **Stricter Weighted Confidence Constraint (SWC)**

    - **Function**: Provides reliable consistency regularization signals on multi-source heterogeneous data by filtering noisy pseudo-labels.
    - **Mechanism**: The predicted probability map is uniformly partitioned into non-overlapping cubic regions $\{r\}$. For each region, the teacher's regional confidence is computed as $c(r) = \mathbb{E}_{i \in r}[\max_c P_{i,c}^T]$; regions below threshold $\tau$ are discarded. Within the retained reliable regions, voxel-level confidence is used to weight the teacher–student alignment loss: $\mathcal{SWC}(P^S, P^T) = \mathbb{E}_{r \in \mathcal{R}_\tau}[\mathbb{E}_{i \in r}[c_i \cdot \mathcal{A}(P_i^S, P_i^T)]]$.
    - **Design Motivation**: The heterogeneity of CBCT data causes standard consistency regularization to introduce substantial noise, particularly at difficult regions such as tooth boundaries. SWC first applies region-level gating to exclude unreliable regions, then refines the alignment signal at the voxel level via weighting, balancing structural reliability and voxel-level accuracy—particularly well-suited for 3D CBCT segmentation.

3. **MS3Toothset Dataset**

    - **Function**: Provides a standardized multi-source semi-supervised CBCT tooth segmentation benchmark.
    - **Mechanism**: Integrates semi-annotated data from ShanghaiTech with unlabeled private data from PKU-SS and AFMC; after filtering and preprocessing, the dataset comprises 98 labeled samples (20 for testing) and 438 unlabeled samples.
    - **Design Motivation**: Existing public CBCT tooth datasets are scarce and typically single-source, insufficient to support the development and evaluation of multi-source semi-supervised methods.

### Loss & Training
The total loss consists of three terms: the supervised loss on labeled data $\mathcal{L}_{sup} = CE(P^S(x^l), y)$, and two SWC consistency losses corresponding to the *other* and *mixed* subsets:

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^u + \beta \mathcal{L}_{cons}^h$$

where $\alpha = \beta = 0.5$, SWC threshold $\tau = 0.9$, and EMA decay rate $\gamma = 0.99$. The backbone is V-Net, trained with the Adam optimizer at a learning rate of 0.0001 for 300 epochs.

## Key Experimental Results

### Main Results

| Method | Venue | Year | mIoU | Dice | Recall | Acc |
|--------|-------|------|------|------|--------|-----|
| V-Net | IEEE 3DV | 2016 | 61.36 | 73.65 | 70.77 | 66.75 |
| MT | NeurIPS | 2017 | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | MICCAI | 2019 | 68.37 | 79.18 | 80.42 | 76.17 |
| ASDA | IEEE TIP | 2022 | 73.75 | 83.63 | 80.93 | 78.79 |
| CMT | ACM MM | 2024 | 76.14 | 85.07 | 87.14 | 84.32 |
| Uni-HSSL | CVPR | 2025 | 75.76 | 85.42 | 84.26 | 81.88 |
| **SemiTooth** | - | 2025 | **76.67** | **85.69** | **88.66** | **86.44** |

### Ablation Study

| Exp | V-Net | MT | ST | SWC | mIoU | Dice | Recall | Acc |
|-----|-------|----|----|-----|------|------|--------|-----|
| 1 | ✓ | | | | 61.36 | 73.65 | 70.77 | 66.75 |
| 2 | ✓ | ✓ | | | 67.69 | 78.72 | 78.06 | 73.68 |
| 3 | ✓ | ✓ | | ✓ | 69.94 | 80.29 | 79.67 | 75.34 |
| 4 | ✓ | ✓ | ✓ | | 75.37 | 84.56 | 83.07 | 80.48 |
| 5 | ✓ | ✓ | ✓ | ✓ | **76.67** | **85.69** | **88.66** | **86.44** |

### Key Findings
- Comparing Exp2 to Exp4, the multi-teacher–multi-student architecture contributes the largest gain (mIoU +7.68%), confirming that the multi-branch design is the core component.
- The SWC constraint yields a +2.25% mIoU improvement over MT (Exp2 vs. Exp3) and a +1.30% improvement over SemiTooth without SWC (Exp4 vs. Exp5), demonstrating consistent gains even on top of a strong base framework.
- t-SNE visualizations confirm that SemiTooth effectively compresses the feature distribution gap across sources, achieving cross-source domain generalization.
- Qualitative results show that SemiTooth performs best at tooth root regions and at adhesion issues along adjacent tooth boundaries.

## Highlights & Insights
- **Hierarchical confidence filtering combining region-level and voxel-level granularity** is the most elegant design: coarse region-level gating first excludes noise-dense areas, followed by fine-grained voxel-level weighting within reliable regions. This hierarchical filtering strategy is transferable to other 3D semi-supervised segmentation tasks.
- **The *mixed* subset as a distributional bridge** is a noteworthy design choice: Wasserstein distance is used to identify unlabeled samples whose distribution closely resembles the labeled source, serving as a connector between heterogeneous sources.
- The overall framework is relatively compact, introducing few complex components, yet achieves competitive performance through well-motivated branching and constraint design.

## Limitations & Future Work
- The dataset scale is limited (98 labeled + 438 unlabeled samples), which constrains the persuasiveness of experimental validation.
- Only V-Net is evaluated as the backbone; stronger 3D segmentation backbones (e.g., nnU-Net, Swin UNETR) remain untested.
- The multi-teacher–multi-student architecture incurs higher computational and memory costs than single-teacher designs (3 students + 2 teachers), which must be considered in deployment under resource constraints.
- The partitioning of the *mixed* subset depends on the Wasserstein distance threshold, whose sensitivity is not thoroughly discussed.
- Incorporating contrastive learning to further align cross-source feature representations is a promising future direction.

## Related Work & Insights
- **vs. Mean Teacher [15]**: MT employs only a single teacher–student pair and cannot perceive multi-source distribution differences. SemiTooth's multi-branch design with dedicated teachers significantly improves cross-source performance (+9% mIoU).
- **vs. CMT [20]**: CMT uses multi-student co-training with shared weights but without teacher supervision, lacking stable pseudo-label guidance. SemiTooth surpasses CMT by +0.53% mIoU.
- **vs. ASDA [12]**: ASDA is a domain adaptation method designed for multi-source semi-supervised settings. SemiTooth substantially outperforms it on Recall (+7.73%), indicating greater suitability for clinically sensitive metrics.

## Rating
- **Novelty**: ⭐⭐⭐ The multi-teacher–multi-student idea represents a reasonable incremental advance; the SWC constraint is novel but the improvement margin is modest.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablation and comparison studies are relatively complete, but the dataset is small and only one backbone is evaluated.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich and intuitive illustrations.
- **Value**: ⭐⭐⭐ Offers meaningful reference for multi-source semi-supervised medical segmentation, but broader generalizability requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](ctfs_collaborative_teacher_framework_for_forward-looking_sonar_image_semantic_se.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](../../NeurIPS2025/segmentation/omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)

</div>

<!-- RELATED:END -->
