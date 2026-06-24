---
title: >-
  [Paper Note] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth
description: >-
  [ECCV 2024][Remote Sensing][cross-view localization] To address the performance degradation of fine-grained cross-view localization models when deployed in new areas, a weakly-supervised learning method based on knowledge self-distillation is proposed. Employing three strategies—mode-based pseudo GT generation, coarse-level supervision, and outlier filtering—it reduces localization errors by 12% to 20% on VIGOR and KITTI using only ground-to-aerial image pairs from the target…
tags:
  - "ECCV 2024"
  - "Remote Sensing"
  - "cross-view localization"
  - "knowledge self-distillation"
  - "pseudo ground truth"
  - "domain adaptation"
  - "weakly-supervised learning"
date: 2026-05-08
content_hash: be4dc22e0a2094cf
---

# Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth

**Conference**: ECCV 2024  
**arXiv**: [2406.00474](https://arxiv.org/abs/2406.00474)  
**Code**: No public code  
**Area**: Visual Localization / Cross-View Geo-localization / Unsupervised Domain Adaptation  
**Keywords**: cross-view localization, knowledge self-distillation, pseudo ground truth, domain adaptation, weakly-supervised learning

## TL;DR
To address the performance degradation of fine-grained cross-view localization models when deployed in new areas, a weakly-supervised learning method based on knowledge self-distillation is proposed. Employing three strategies—mode-based pseudo GT generation, coarse-level supervision, and outlier filtering—it reduces localization errors by 12% to 20% on VIGOR and KITTI using only ground-to-aerial image pairs from the target area (without requiring precise GT).

## Background & Motivation
Fine-grained cross-view localization aims to accurately estimate the ground camera's position in an aerial image based on ground and corresponding aerial images. Existing SOTA methods (such as CCVPE, GGCVT) achieve sub-meter accuracy through coarse-to-fine heatmap predictions, but these models rely on precise ground truth (GT) position annotations in the training area. When deployed to a new area, the localization accuracy drops significantly due to the domain gap (differences in regional appearance, architectural styles, etc.).

In practice, obtaining precise GT annotations ($<2.5\text{m}$ error) for new areas is highly expensive—standard GNSS errors can reach tens of meters in urban canyons, while collecting image pairs with coarse positions is easy. Therefore, how to leverage images with only coarse location information from the target area to improve pre-trained models' performance is a practical and crucial problem.

## Core Problem
How to improve the cross-area performance of fine-grained cross-view localization models pre-trained in a source area under the condition of **no precise position GT** in the target area? This is essentially a source-free unsupervised domain adaptation (UDA) problem. However, unlike traditional classification UDA, the output of localization tasks is spatially ordered heatmaps, making conventional entropy minimization or entropy-based uncertainty filtering methods inapplicable.

## Method

### Overall Architecture
Input: Ground-aerial image pairs from the target area (without precise GT positions) and a teacher model pre-trained on the source area. Output: A student model adapted to the target area.

The workflow consists of three steps:
1. **Pseudo GT Generation**: Infer with the teacher model on target images to generate mode-based pseudo GT.
2. **Auxiliary Student Training & Outlier Filtering**: First train an auxiliary student, and then compare its prediction discrepancy with the teacher's to identify unreliable samples.
3. **Final Student Training**: Train the final student model using the filtered, reliable pseudo GT.

### Key Designs

1. **Mode-based Pseudo GT Generation**: The output heatmaps of the teacher model on the target domain are often multi-modal (i.e., multiple peaks exist). Directly using the multi-modal heatmap as pseudo GT would propagate uncertainty to the student. This paper proposes taking only the peak location of the heatmap (argmax) and generating a Gaussian-smoothed unimodal pseudo GT centered at this point: $X(u,v) = \mathcal{N}((u,v) | y^\alpha, I_2 \sigma^2)$, where $\sigma=4$ pixels. This converts the uncertain multi-modal supervision signal into a deterministic unimodal one, reducing noise propagation.

2. **Coarse-only Supervision**: Both CCVPE and GGCVT feature coarse-to-fine multi-level heatmap outputs. Since the pseudo GT itself contains positional errors, using it to supervise high-resolution layers would amplify noise. This paper leverages the natural noise-suppression property of downsampling, computing losses only for the first $K'$ low-resolution layers of the student (CCVPE: $K'=2$, GGCVT: $K'=3$), thereby allowing the high-resolution layers to naturally refine through the model's own capabilities.

3. **Outlier Filtering**: An auxiliary student model $\mathcal{M}_o$ is first trained using all pseudo GTs. It is observed that when there is a significant discrepancy between the predicted positions of the teacher and the auxiliary student, the teacher's prediction is likely an outlier. Thus, the L2 distance $d^{\alpha,o}$ between their predictions is calculated, keeping only the top-$T\%$ samples with the smallest distances (CCVPE: $T=80\%$, GGCVT: $T=70\%$) to train the final student.

### Loss & Training
- Loss function: Weighted InfoNCE loss, weighted by the pseudo GT heatmap $P_k$: $\mathcal{L}_k(H_k^\beta, P_k) = \frac{1}{\sum P_k} \sum_{m,n} P_k^{m,n} \cdot \mathcal{L}_{\text{infoNCE}}(H_k^\beta | (m,n))$
- Loss is averaged only across the first $K'$ levels.
- The student is initialized with the teacher's weights, using the Adam optimizer with a learning rate of $1 \times 10^{-4}$.
- CCVPE batch=8, GGCVT batch=4.
- Low training overhead: On VIGOR, CCVPE requires only about 6 hours of additional training (on a single V100 32GB).

## Key Experimental Results

| Dataset | Method | Metric | Teacher (baseline) | Student (Ours) | Gain |
|--------|------|------|----------|------|------|
| VIGOR (known orientation) | CCVPE | Mean (m) | 4.38 | 3.85 | -12% |
| VIGOR (known orientation) | CCVPE | Median (m) | 1.76 | 1.57 | -11% |
| VIGOR (unknown orientation) | CCVPE | Mean (m) | 5.35 | 4.27 | -20% |
| VIGOR (unknown orientation) | CCVPE | Median (m) | 1.97 | 1.67 | -15% |
| VIGOR (known orientation) | GGCVT | Mean (m) | 5.19 | 4.34 | -16% |
| VIGOR (known orientation) | GGCVT | Median (m) | 1.39 | 1.32 | -5% |
| KITTI | CCVPE | Long. Mean (m) | 6.55 | 6.18 | -6% |
| KITTI | GGCVT | Long. Mean (m) | 9.27 | 8.56 | -8% |

Oracle (fine-tuning with precise GT): CCVPE 2.31m / GGCVT 2.91m, indicating that although the method does not reach full supervision performance, it is highly effective given no precise GT.

### Ablation Study
- **Component contribution**: On GGCVT, St-M-OF (pure distillation) yields 5.34m > Teacher 5.16m (which is worse), adding M reduces it to 4.67m, and adding M+OF (the complete method) further reduces it to 4.28m. This indicates that self-distillation without specialized designs can be counterproductive for localization.
- **Outlier filtering ratio $T$**: The optimal setting is $T=80\%$ for CCVPE and $T=70\%$ for GGCVT; i.e., filtering out 20% to 30% of the most unreliable samples gives the best results. Too much filtering leads to overfitting.
- **Supervision level $K'$**: CCVPE $K'=2$ is optimal, while GGCVT $K'=3$ (all levels) is optimal.
- **Pseudo GT type**: Unimodal pseudo GT outperforms directly using the teacher's heatmap as pseudo GT.
- **Entropy minimization fails completely**: Increasing the weight of entropy minimization only increases localization error, as it merely makes the heatmap sharper but cannot correct false modes.
- **Comparison with other methods**: Bidirectional fusion pseudo-labeling (4.49m) and entropy-based outlier filtering (CCVPE 4.17m / GGCVT 4.52m) are both inferior to the proposed method.
- **Supervised fine-tuning with noisy GT**: When GT error > 2.5m/orientation, supervised fine-tuning is actually worse than the proposed weakly-supervised method.

## Highlights & Insights
- **Insightful domain adaptation design for localization**: It is discovered that traditional classification UDA (entropy minimization, entropy-based uncertainty filtering) is not suitable for spatially ordered localization heatmaps. Consequently, mode-based pseudo GT and position-consistency-based outlier filtering are specifically proposed.
- **Simple yet clever outlier filtering strategy**: It identifies unreliable samples by comparing the prediction consistency between the teacher and an auxiliary student, bypassing the need for additional uncertainty estimation.
- **Insight on coarse-only supervision**: Downsampling naturally suppresses spatial noise, preventing noise from being amplified in high-resolution layers.
- The method possesses excellent generalizability, consistently showing performance gains across two different architectures (CCVPE, GGCVT) on two datasets.
- t-SNE visualization shows that the student model learns a better cross-view feature alignment.

## Limitations & Future Work
- **Teacher needs to be "good enough"**: When the teacher's predictions on the target domain are close to random (such as across highly divergent datasets, sensors, or resolutions), the method is inapplicable.
- **Only addresses regional domain gaps**: It cannot handle larger domain gaps like sensor changes or drastic resolution variations.
- **Single-round iteration**: The pipeline is auxiliary student $\rightarrow$ filtering $\rightarrow$ final student; the potential of multi-round iterations was not explored.
- **Single-frame localization**: It does not utilize sequential information, finding it hard in repetitive texture areas. Utilizing video sequences for temporal voting might bring further improvements.
- **Privacy risks**: Precise localization technology can be abused to track individual locations.

## Related Work & Insights
- **vs. Traditional UDA methods (entropy minimization, adversarial training)**: This paper experimentally proves that entropy minimization completely fails on spatially continuous localization outputs—it only sharpens the heatmap but cannot correct the incorrect modes in multi-modal predictions. Adversarial training was not evaluated, but it is inapplicable anyway due to the source-free setting.
- **vs. Born-Again Networks / Best Teacher Distillation**: Directly applying this to localization tasks makes the student worse than the teacher (GGCVT: 5.34m vs 5.16m) because the spatial noise in localization pseudo GTs requires specialized treatment. The mode-based pseudo GT and outlier filtering are key here.
- **vs. Uncertainty-based pseudo-label filtering**: Traditional methods use entropy to measure uncertainty, which disregards spatial distance—two heatmaps with mode distances of 1m or 10m can have the identical entropy, but the latter incurs a much larger localization error. The proposed position-consistency-based filtering is much better suited for localization tasks.

## Insights & Connections
- The core insight of this paper—**domain adaptation from classification to regression/localization needs to be redesigned**—serves as an inspiration for other spatial prediction tasks (such as 3D detection, boundary regression in semantic segmentation, and depth estimation).
- The concept of mode-based pseudo GT can be generalized to any self-distillation scenarios involving multi-modal outputs.
- The use of "teacher-student prediction consistency" as a reliability metric in outlier filtering shares concepts with consistency regularization in semi-supervised learning, representing an interesting direction to explore in other domain adaptation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ This is the first systematic exploration of GT-free domain adaptation for cross-view localization, with each design tailored to the characteristics of the localization task.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two models across two datasets with exceptionally detailed ablation studies (including entropy minimization, alternative pseudo-labeling schemes, and comparisons with noisy GT supervision).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, natural problem motivation, and each design is fully backed by experimental results.
- Value: ⭐⭐⭐⭐ Resolves a critical pain point in real-world deployment with a generalizable method and low training overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](../../CVPR2026/remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization.md)
- [\[ECCV 2024\] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[AAAI 2026\] Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](../../AAAI2026/remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)
- [\[CVPR 2026\] MOGeo: Beyond One-to-One Cross-View Object Geo-localization](../../CVPR2026/remote_sensing/mogeo_beyond_one-to-one_cross-view_object_geo-localization.md)

</div>

<!-- RELATED:END -->
