---
title: >-
  [Paper Note] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR 2026][Medical Imaging][tooth segmentation] This paper proposes SemiTooth, a framework that addresses annotation scarcity and cross-source domain discrepancy in multi-source CBCT tooth segmentation via a multi-teacher multi-student architecture and Strict Weighted Confidence (SWC) constraints. It also introduces MS3Toothset, the first multi-source semi-supervised tooth segmentation dataset.
tags:
  - CVPR 2026
  - Medical Imaging
  - tooth segmentation
  - CBCT
  - semi-supervised learning
  - multi-source data
  - multi-teacher multi-student
date: 2026-05-08
content_hash: a9f1fcc6407df9dc
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: tooth segmentation, CBCT, semi-supervised learning, multi-source data, multi-teacher multi-student

## TL;DR

This paper proposes SemiTooth, a framework that addresses annotation scarcity and cross-source domain discrepancy in multi-source CBCT tooth segmentation via a multi-teacher multi-student architecture and Strict Weighted Confidence (SWC) constraints. It also introduces MS3Toothset, the first multi-source semi-supervised tooth segmentation dataset.

## Background & Motivation

CBCT tooth structure segmentation is central to intelligent oral diagnosis, yet faces two major challenges:

**Annotation scarcity**: Voxel-level annotation is costly and time-consuming, leaving large quantities of de-identified CBCT data unutilized.

**Multi-source domain discrepancy**: CBCT data from different institutions and devices exhibit significant differences in density distribution, intensity distribution, and feature space (verified via Kernel Density Estimation and t-SNE visualization), making cross-source generalization difficult.

Existing semi-supervised medical segmentation methods (Mean Teacher, UA-MT, MCF, etc.) are primarily designed for single-source data and lack cross-source knowledge transfer capability. Multi-source methods such as ASDA and Dual-Teacher either require complex networks or strong supervision, or have not been validated on multi-source CBCT tooth data.

## Method

### Overall Architecture

SemiTooth adopts a **three-student + two-teacher** multi-branch architecture. Data are organized into three subsets:
- **Main** (labeled data from the primary source)
- **Other** (unlabeled data from other sources)
- **Mixed** (unlabeled data with distribution similar to the primary source, selected via Wasserstein distance)

Each subset is processed by a corresponding student network. Two teachers supervise the Mixed and Other students respectively, updated via EMA:

$$\theta_t^{(k)} \leftarrow \gamma \theta_t^{(k-1)} + (1-\gamma) \theta_s^{(k)}, \quad \gamma = 0.99$$

### Key Designs

1. **Multi-Teacher Multi-Student Architecture (SemiTooth)**:

    - Compared to Mean Teacher (single teacher–single student): SemiTooth assigns dedicated students to data from different sources, with teachers providing cross-source knowledge guidance.
    - Compared to Co-training (multi-student with shared weights, no teacher): SemiTooth provides stable pseudo-labels via EMA teachers.
    - The **Mixed subset** serves as an inter-source bridge, containing unlabeled samples with distributions similar to the primary source, alleviating training instability from direct cross-source learning.
    - Student networks share similar architectures to facilitate knowledge transfer while maintaining sufficient diversity.

2. **Strict Weighted Confidence Constraint (SWC)**:

   Addresses the degradation of consistency regularization caused by noise introduced by CBCT heterogeneity. SWC combines region-level gating with voxel-level weighting:

    - **Region-level gating**: Samples are uniformly divided into non-overlapping cubic regions $\{r\}$. Region confidence is computed as $c(r) = \mathbb{E}_{i \in r}[\max_c P^T_{i,c}]$; regions below threshold $\tau=0.9$ are deemed unreliable and discarded.
    - **Voxel-level weighting**: Within retained regions, voxel-level confidence $c_i = \max_c P^T_{i,c}$ is used to weight the teacher–student alignment:
    $\mathcal{SWC}(P^S, P^T) = \mathbb{E}_{r \in \mathcal{R}_\tau}\left[\mathbb{E}_{i \in r}\left[c_i \cdot \mathcal{A}(P^S_i, P^T_i)\right]\right]$
    - This design is particularly suited to 3D CBCT data, as region-level filtering exploits the spatial continuity inherent in volumetric data.

3. **Multi-Source Dataset MS3Toothset**:

    - Collected from three sources: ShanghaiTech (public, labeled), PKU-SS, and AFMC (private, unlabeled).
    - After preprocessing and filtering: 98 labeled samples (20 for testing) and 438 unlabeled samples.
    - The first comprehensive dataset for multi-source semi-supervised tooth segmentation.

### Loss & Training

The total loss combines a supervised loss and two SWC consistency losses:

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^u + \beta \mathcal{L}_{cons}^h, \quad \alpha = \beta = 0.5$$

- $\mathcal{L}_{sup} = \text{CE}(P^S(x^l), y)$: supervised loss on labeled data from the primary source.
- $\mathcal{L}_{cons}^u$, $\mathcal{L}_{cons}^h$: SWC losses for the Other and Mixed sources, respectively.
- Backbone: V-Net; optimizer: Adam; learning rate: 0.0001; training: 300 epochs.

## Key Experimental Results

### Main Results

| Method | mIoU | Dice | Recall | Acc |
|--------|------|------|--------|-----|
| V-Net (fully supervised baseline) | 61.36 | 73.65 | 70.77 | 66.75 |
| Mean Teacher | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | 68.37 | 79.18 | 80.42 | 76.17 |
| ASDA | 73.75 | 83.63 | 80.93 | 78.79 |
| CMT | 76.14 | 85.07 | 87.14 | 84.32 |
| Uni-HSSL | 75.76 | 85.42 | 84.26 | 81.88 |
| **SemiTooth** | **76.67** | **85.69** | **88.66** | **86.44** |

### Ablation Study

| Exp | Configuration | mIoU | Dice | Recall | Acc |
|-----|--------------|------|------|--------|-----|
| 1 | V-Net | 61.36 | 73.65 | 70.77 | 66.75 |
| 2 | + Mean Teacher | 67.69 | 78.72 | 78.06 | 73.68 |
| 3 | + SWC (w/o SemiTooth) | 69.94 | 80.29 | 79.67 | 75.34 |
| 4 | + SemiTooth (w/o SWC) | 75.37 | 84.56 | 83.07 | 80.48 |
| 5 | + SemiTooth + SWC | **76.67** | **85.69** | **88.66** | **86.44** |

### Key Findings

- The multi-branch SemiTooth architecture contributes the most (Exp 2→4: mIoU +7.68), demonstrating the importance of source-aware learning for multi-source data.
- SWC yields consistent improvements under both single-teacher (Exp 2→3: +2.25 mIoU) and multi-teacher (Exp 4→5: +1.30 mIoU) settings.
- SemiTooth's advantage is most pronounced on the Recall metric (88.66 vs. second-best 87.14), which is clinically meaningful for reducing missed detection rates.
- t-SNE visualizations confirm that SemiTooth effectively reduces feature distribution gaps across sources.

## Highlights & Insights

- **Dataset contribution**: MS3Toothset fills a gap in multi-source semi-supervised tooth segmentation benchmarks.
- The two-stage region-level + voxel-level filtering in SWC is intuitively grounded and well-suited to the spatial continuity of 3D medical data.
- Using Wasserstein distance to select distributionally similar samples for the Mixed subset as an inter-source bridge is a simple yet effective design choice.
- The ablation study is well-structured, clearly illustrating both the individual and joint contributions of each component.

## Limitations & Future Work

- MS3Toothset is relatively small in scale (98 labeled + 438 unlabeled samples).
- Validation is limited to the authors' own dataset; generalizability to public standard benchmarks remains unknown.
- The teacher count and subset partitioning strategy depend on the choice of Wasserstein distance threshold, with no sensitivity analysis provided.
- Only V-Net is used as the backbone; effectiveness with stronger backbones (e.g., nnUNet, Swin UNETR) is not evaluated.
- Ablation of the region size and threshold $\tau$ in SWC is insufficient.

## Related Work & Insights

- SemiTooth outperforms recent multi-source methods such as CMT (ACM MM 2024) and Uni-HSSL (CVPR 2025).
- The multi-teacher multi-student paradigm is generalizable to other multi-domain semi-supervised scenarios (e.g., multi-center CT/MRI analysis).
- The SWC constraint can be readily integrated into any Mean Teacher-based framework.

## Rating

- Novelty: ⭐⭐⭐ The multi-teacher multi-student and confidence-weighting ideas are not entirely new, but their systematic integration for tooth segmentation is valuable.
- Experimental Thoroughness: ⭐⭐⭐ Ablations are complete, but validation is limited to a single in-house dataset without external benchmarking.
- Writing Quality: ⭐⭐⭐ The framework is described clearly, though the paper is short (ICASSP length) and lacks detail in places.
- Value: ⭐⭐⭐ The dataset contribution has clinical value; generalizability of the method remains to be verified.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_th.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semisupervised_framework_for_breast_ultrasound_s.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
