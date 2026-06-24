---
title: >-
  [Paper Note] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR2025][Medical Imaging][semi-supervised learning] This paper proposes SemiTooth, a multi-teacher-student semi-supervised framework, which achieves cross-domain generalization for multi-source CBCT tooth segmentation via a Stricter Weighted-Confidence Constraint.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "semi-supervised learning"
  - "multi-source"
  - "tooth segmentation"
  - "CBCT"
  - "multi-teacher-student"
date: 2026-05-08
content_hash: 5d4e607be8f9bc7f
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**Conference**: CVPR2025  
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)  
**Code**: To be confirmed  
**Area**: Medical Image  
**Keywords**: semi-supervised learning, multi-source, tooth segmentation, CBCT, multi-teacher-student

## TL;DR

This paper proposes SemiTooth, a multi-teacher-student semi-supervised framework, which achieves cross-domain generalization for multi-source CBCT tooth segmentation via a Stricter Weighted-Confidence Constraint.

## Background & Motivation

Cone-beam computed tomography (CBCT) is a widely used imaging modality in clinical dentistry, where tooth structure segmentation serves as a fundamental task for orthodontics, implant planning, and lesion analysis. However, voxel-level annotation is extremely costly, leaving a large amount of de-identified CBCT data underutilized. Existing semi-supervised medical image segmentation (SSMIS) methods are mostly designed for single-source data. However, CBCT data collected from different institutions or devices exhibit significant distribution discrepancies (with entirely different density distributions and feature space clustering), making cross-source generalization difficult. In addition, publicly available multi-source CBCT tooth segmentations datasets are extremely scarce.

The core motivations of this paper are: (1) to construct a multi-source semi-supervised CBCT tooth dataset, and (2) to design a semi-supervised method that can effectively utilize multi-source unlabeled data.

## Method

### MS3Toothset Dataset
CBCT data from three institutions are integrated:
- **ShanghaiTech**: Provides semi-labeled data (primary source, with labels)
- **PKU-SS**: Private unlabeled data
- **AFMC**: Private unlabeled data

The dataset contains a total of 98 labeled samples (20 for testing) and 438 unlabeled samples. Significant distribution discrepancies among the three sources are confirmed through kernel density estimation curves, mid-slice intensity distributions, and t-SNE visualizations. By leveraging the Wasserstein distance to measure inter-source similarity, the unlabeled data are partitioned into two subsets: *mixed* (distribution close to the labeled source) and *other* (large distribution discrepancy).

### SemiTooth Framework
A multi-branch architecture is adopted, consisting of three student networks and two teacher networks:
- **Main Student**: Undergoes supervised training on the labeled primary source data, utilizing the standard cross-entropy loss.
- **Mixed Student**: Learns from unlabeled data with distributions close to the primary source, supervised by the corresponding teacher via pseudo-labels.
- **Other Student**: Learns from unlabeled data with large distribution discrepancies, supervised by the other teacher.
- The teacher networks update their parameters via Exponential Moving Average (EMA, decay rate $\gamma=0.99$) to provide stable pseudo-labels.

Unlike the single teacher-student paradigm of Mean Teacher and the weight-sharing scheme of Co-training, SemiTooth achieves better cross-source knowledge transfer through a source-aware multi-branch design. The student networks share similar architectures to facilitate knowledge transfer while maintaining sufficient diversity.

### Stricter Weighted-Confidence (SWC) Constraint
To address the noise issue introduced by the heterogeneity of multi-source CBCT data, a dual-level confidence constraint (region-level + voxel-level) is proposed:

1. Uniformly partition the samples into non-overlapping cubic regions $\{r\}$.
2. **Region-level Gating**: Calculate the average confidence $c(r)$ for each region; regions with confidence below the threshold $\tau=0.9$ are considered unreliable and discarded (denoted as $R_u$).
3. **Voxel-level Weighting**: Within the reliable region $R_\tau$, use the maximum class probability $c_i$ of each voxel to weight the alignment loss between the teacher and student outputs.
4. The final SWC loss is calculated as the average of the weighted cross-entropy across all reliable regions.

This dual-level design offers both structural reliability and voxel-level precision, which is particularly suited to the spatial structural characteristics of 3D CBCT. Compared to simple global confidence thresholds, region-level gating can handle local noise more effectively.

### Total Loss Function
$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^{u} + \beta \mathcal{L}_{cons}^{h}$$

where $\alpha=\beta=0.5$ balances the consistency loss contribution from different sources. The model is trained for 300 epochs using the Adam optimizer with a learning rate of $1e-4$.

## Key Experimental Results

| Method | mIoU | Dice | Recall | Acc |
|------|------|------|--------|-----|
| V-Net (Baseline) | 61.36 | 73.65 | 70.77 | 66.75 |
| Mean Teacher | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | 68.37 | 79.18 | 80.42 | 76.17 |
| CMT | 76.14 | 85.07 | 87.14 | 84.32 |
| Uni-HSSL (CVPR2025) | 75.76 | 85.42 | 84.26 | 81.88 |
| **SemiTooth (Ours)** | **76.67** | **85.69** | **88.66** | **86.44** |

Contribution breakdown of each component in the **Ablation Study**:

| Exp | Module | mIoU | Dice |
|------|------|------|------|
| Exp1 | V-Net only | 61.36 | 73.65 |
| Exp2 | +MT | 67.69 | 78.72 |
| Exp3 | +MT+SWC | 69.94 | 80.29 |
| Exp4 | +V-Net+MT+ST | 75.37 | 84.56 |
| Exp5 | Full Model | 76.67 | 85.69 |

As observed from Exp4 $\rightarrow$ 5, the SWC constraint brings an additional improvement of 1.3 mIoU under the SemiTooth framework. Qualitative comparisons demonstrate that the full model generates more natural tooth shapes, particularly in reducing adhesion at root regions and adjacent tooth boundaries.

t-SNE visualization confirms that the multi-source features output by the SemiTooth student are more clustered than those of the raw data, validating the enhancement of cross-source generalization capabilities. Training is conducted on $4 \times \text{NVIDIA A4500}$ GPUs with a batch size of 4 for 300 epochs, using the Adam optimizer with a learning rate of $1e-4$.

## Highlights

1. **Precise problem definition**: Accurately applies multi-source semi-supervised learning to CBCT tooth segmentation for the first time, establishing the MS3Toothset to fill the dataset gap.
2. **Inherent SWC constraint design**: The dual-level mechanism consisting of region-level gating and voxel-level weighting filters out unreliable regions while performing fine-grained weighting within reliable regions, which is highly suited to the structural properties of 3D CBCT.
3. **Rational framework design**: The multi-teacher-student architecture naturally aligns with multi-source scenarios, implementing source-awareness by partitioning into mixed/other subsets using the Wasserstein distance.
4. **Outperforming state-of-the-art**: Under identical experimental settings, it outperforms recent methods such as Uni-HSSL (CVPR2025) and CMT (ACM MM 2024).
5. **t-SNE feature visualization**: Visually demonstrates the clustering effect of multi-source features after training, providing intuitive evidence of cross-domain generalization.

## Limitations & Future Work

1. The dataset scale is relatively small (98 labeled + 438 unlabeled) with only three sources (ShanghaiTech, PKU-SS, AFMC); hence, the generalization capability to more sources or larger scales remains unverified.
2. Only V-Net is used as the backbone; more powerful 3D segmentation architectures (such as nnU-Net, Swin UNETR, MedNeXt) have not been explored.
3. Inter-source similarity is measured using the Wasserstein distance, yet a rigorous analysis of the division criteria for mixed/other subsets is lacking, and the threshold selection may impact performance.
4. Evaluation is performed solely on the self-built dataset, lacking validation on standard public benchmarks (such as the STS challenge).
5. The multi-teacher-student framework introduces a high training overhead, requiring $4 \times \text{A4500}$ GPUs. Although only a single student model is needed for actual deployment, inference efficiency is not discussed.
6. The impact of the labeling ratio on performance was not analyzed; only results based on a fixed annotation budget are reported.
7. No sensitivity analysis was conducted for the choice of the SWC threshold $\tau=0.9$.
8. Only four metrics (mIoU, Dice, Recall, and Acc) are reported, without including distance-based boundary metrics like HD95.

## Rating
- Novelty: 3/5 — The combination of multi-teacher-student and confidence constraints is interesting, but the individual components are not entirely novel.
- Experimental Thoroughness: 3/5 — The ablation and visualization analyses are thorough, but the evaluation is limited to the self-built dataset.
- Writing Quality: 3/5 — The structure is clear, but certain details (such as the specific application of the Wasserstein distance and the threshold for partitioning mixed subsets) are insufficiently explained.
- Value: 3/5 — It holds practical value for clinical dentistry scenarios, though the generalizability of the method requires further experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](../../NeurIPS2025/medical_imaging/match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)

</div>

<!-- RELATED:END -->
