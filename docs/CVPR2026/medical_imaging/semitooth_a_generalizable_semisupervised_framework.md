---
title: >-
  [Paper Note] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Semi-supervised Learning] This paper proposes SemiTooth, a multi-teacher multi-student semi-supervised framework coupled with a Stricter Weighted-Confidence (SWC) constraint…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Semi-supervised Learning"
  - "Multi-Source Data"
  - "Tooth Segmentation"
  - "CBCT"
  - "Pseudo Labels"
date: 2026-05-08
content_hash: f9d947ee06b497c6
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)  
**Code**: N/A  
**Area**: Medical Image Segmentation
**Keywords**: Semi-supervised Learning, Multi-Source Data, Tooth Segmentation, CBCT, Pseudo Labels

## TL;DR

This paper proposes SemiTooth, a multi-teacher multi-student semi-supervised framework coupled with a Stricter Weighted-Confidence (SWC) constraint, which effectively leverages multi-source unlabeled data for multi-source CBCT tooth segmentation and achieves cross-source generalization.

## Background & Motivation

CBCT (Cone Beam Computed Tomography) tooth segmentation is a core task in clinical dental diagnosis, yet it faces two major challenges:

1. **Scarcity of annotated data**: Voxel-level annotation is extremely costly, leaving large volumes of de-identified CBCT data unutilized.
2. **Multi-source data heterogeneity**: CBCT data acquired from different institutions and devices exhibit significant distribution discrepancies (in density distribution, grayscale intensity, and feature clustering), making unified model training difficult.

Existing semi-supervised medical image segmentation methods (e.g., Mean Teacher, Co-training) are predominantly designed for single-source data and cannot effectively handle distribution shifts in multi-source settings. Furthermore, publicly available multi-source CBCT tooth segmentation datasets are extremely scarce, limiting method validation.

## Method

### Overall Architecture

SemiTooth is a multi-branch semi-supervised framework adopting a **three-student two-teacher** architecture:

- Multi-source data are reorganized into three subsets: **main** (labeled primary source), **other** (unlabeled data from other sources), and **mixed** (an unlabeled subset with distribution similar to the primary source, selected via Wasserstein distance measurement).
- Three student networks are each responsible for learning from one subset.
- Two teacher networks supervise the students on the mixed and other subsets respectively, providing stable pseudo labels through EMA updates.

### Key Designs

1. **Multi-source subset partitioning strategy**:
    - Inter-source Wasserstein distances are used to divide unlabeled data into a *mixed* group (distribution close to the labeled source) and an *other* group (larger distribution discrepancy).
    - The mixed subset serves as a bridge across different sources, improving cross-source training robustness.

2. **Multi-Teacher Multi-Student architecture**:
    - Three student networks share similar architectures but learn independently, promoting effective knowledge transfer while maintaining diversity.
    - Teachers are updated via EMA: $\theta_t^{(k)} \leftarrow \gamma \theta_t^{(k-1)} + (1-\gamma) \theta_s^{(k)}$, with decay rate $\gamma = 0.99$.
    - Compared to Mean Teacher (single teacher, single student) and Co-training (no teacher, multiple students), SemiTooth combines the stability of teacher supervision with the cross-source capability of multi-student collaboration.

3. **Stricter Weighted-Confidence (SWC) Constraint**:
    - Each sample is uniformly partitioned into non-overlapping cubic regions $\{r\}$.
    - Region-level confidence is defined as the mean of the maximum class probability across all voxels within a region: $c(r) = \mathbb{E}_{i \in r}[\max_c P_{i,c}^T]$.
    - Low-confidence regions ($c(r) < \tau$) are flagged as unreliable and ignored.
    - Within reliable regions, voxel-level confidence $c_i = \max_c P_{i,c}^T$ further weights teacher–student alignment.
    - This dual-layer design balances structural reliability at the region level with fine-grained accuracy at the voxel level.

### Loss & Training

The total loss consists of three components:

- **Supervised loss** (labeled data from the primary source): $\mathcal{L}_{sup} = CE(P^S(x^l), y)$
- **SWC consistency loss** (unlabeled data from other and mixed sources):

$$\mathcal{L}_{SWC} = \mathbb{E}_{r \in \mathcal{R}_\tau} \left[ \mathbb{E}_{i \in r} \left[ c_i \cdot CE(P_i^S, P_i^T) \right] \right]$$

- **Total loss**: $\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^u + \beta \mathcal{L}_{cons}^h$

where $\alpha = \beta = 0.5$ balance contributions from different sources, and the confidence threshold is set to $\tau = 0.9$.

Training details: V-Net is used as the backbone, with the Adam optimizer, learning rate $10^{-4}$, batch size 4, trained for 300 epochs on 4 NVIDIA A4500 GPUs.

## Key Experimental Results

### Dataset: MS3Toothset

A self-constructed multi-source semi-supervised tooth segmentation dataset comprising 98 annotated samples (20 for testing) and 438 unlabeled samples from three sources (ShanghaiTech, PKU-SS, and AFMC). The sources exhibit significant differences in density, grayscale intensity, and feature distributions.

### Main Results

| Method | Venue | Year | mIoU | Dice | Recall | Acc |
|--------|-------|------|------|------|--------|-----|
| V-Net | IEEE 3DV | 2016 | 61.36 | 73.65 | 70.77 | 66.75 |
| Mean Teacher | NeurIPS | 2017 | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | MICCAI | 2019 | 68.37 | 79.18 | 80.42 | 76.17 |
| ASDA | IEEE TIP | 2022 | 73.75 | 83.63 | 80.93 | 78.79 |
| MLRPL | MIA | 2024 | 72.86 | 83.29 | 79.75 | 77.39 |
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

1. The SemiTooth multi-branch architecture (Exp4 vs. Exp2) contributes the largest performance gain (+7.68 mIoU), demonstrating that the multi-teacher multi-student structure is critical for cross-source learning.
2. The SWC constraint yields +2.25 mIoU over the Mean Teacher baseline (Exp3 vs. Exp2) and +1.30 mIoU on top of SemiTooth (Exp5 vs. Exp4).
3. t-SNE visualizations show that after training with SemiTooth, feature distributions across different sources become more clustered, validating the improvement in cross-source generalization.

## Highlights & Insights

- **Clear problem formulation**: This work is the first to systematically address multi-source semi-supervised CBCT tooth segmentation and constructs a dedicated benchmark dataset.
- **Well-motivated architecture**: The asymmetric three-student two-teacher design is more effective than a naive multi-student framework, as the teachers provide stable pseudo-label training signals.
- **Elegant SWC constraint**: The dual-layer filtering strategy—region-level gating combined with voxel-level weighting—is better suited to the spatial structure of 3D CBCT than simple confidence thresholding.
- **Wasserstein-based subset partitioning**: Using distributional distance to bridge labeled and unlabeled sources is a practical and effective strategy.

## Limitations & Future Work

1. **Limited dataset scale**: Only 98 annotated and 438 unlabeled samples; validation on larger-scale data is needed.
2. **Restricted number of sources**: Only three sources are considered; scalability to a larger number of sources has not been verified.
3. **Single backbone**: Only V-Net is evaluated; more advanced 3D segmentation networks (e.g., nnU-Net, SwinUNETR) may yield further improvements.
4. **Out-of-distribution generalization unverified**: Model performance on completely unseen new-source data remains unknown.
5. **Fixed cubic region size in SWC**: An adaptive region partitioning scheme may be preferable.

## Related Work & Insights

- The **Mean Teacher** paradigm is the foundational framework for semi-supervised medical image segmentation (SSMIS); SemiTooth extends it via a multi-branch design.
- **CMT (ACM MM 2024)** and **Uni-HSSL (CVPR 2025)** are recent multi-source semi-supervised methods, both of which SemiTooth outperforms.
- **Cross-domain medical segmentation** provides inspiration: Wasserstein distance for source distribution measurement and multi-teacher structures for cross-domain knowledge transfer.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[CVPR 2026\] Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)

</div>

<!-- RELATED:END -->
