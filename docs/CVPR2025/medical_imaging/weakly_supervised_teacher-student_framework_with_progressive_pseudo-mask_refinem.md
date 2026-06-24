---
title: >-
  [Paper Note] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation
description: >-
  [CVPR 2025][Medical Imaging][Gland segmentation] This paper proposes a weakly supervised teacher-student framework that leverages sparse pathologist annotations and an EMA-stabilized teacher network to generate progressively refined pseudo-masks, achieving excellent performance with mIoU of 80.10 and mDice of 89.10 on the gland segmentation task using far fewer annotations than full supervision.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Gland segmentation"
  - "weakly supervised semantic segmentation"
  - "teacher-student framework"
  - "pseudo-mask refinement"
  - "colorectal cancer"
date: 2026-05-08
content_hash: 05c88f3a5ecc868e
---

# Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.08605](https://arxiv.org/abs/2603.08605)  
**Code**: None  
**Area**: Medical Image / Weakly Supervised Segmentation  
**Keywords**: Gland segmentation, weakly supervised semantic segmentation, teacher-student framework, pseudo-mask refinement, colorectal cancer

## TL;DR
This paper proposes a weakly supervised teacher-student framework that leverages sparse pathologist annotations and an EMA-stabilized teacher network to generate progressively refined pseudo-masks, achieving excellent performance with mIoU of 80.10 and mDice of 89.10 on the gland segmentation task using far fewer annotations than full supervision.

## Background & Motivation

**Background**: Histographical grading of colorectal cancer relies on precise segmentation of gland structures. Deep learning methods have achieved good results under fully supervised conditions, but they require a large amount of pixel-level annotations—producing a fully annotated whole-slide image (WSI) can take hours or even days of a pathologist's work.

**Limitations of Prior Work**: Weakly supervised semantic segmentation (WSSS) is an alternative with low annotation costs, but existing class activation map (CAM)-based methods suffer from severe limitations: CAMs tend to focus only on the most discriminative regions (often local parts of the glands), resulting in incomplete pseudo-masks with blurry boundaries. More critically, CAM-based methods cannot provide any supervisory signals for unannotated gland regions, leading to these regions being ignored.

**Key Challenge**: Glands in histopathological images exhibit diverse morphologies and dense distributions, making it difficult to generalize to all gland structures using only a few annotated points. Simply training with incomplete pseudo-masks causes the student network to inherit and amplify the teacher's errors, creating a vicious cycle.

**Goal**: To design an annotation-efficient framework capable of progressively discovering and segmenting all gland regions starting from sparse annotations, while ensuring the quality of pseudo-labels.

**Key Insight**: To leverage the complementary nature of the Teacher-Student architecture—where the teacher network maintains stable predictions through EMA smoothing while the student network quickly learns the latest features. The fusion of both networks can progressively expand to unannotated regions.

**Core Idea**: To generate stable predictions using an EMA teacher network, fuse teacher predictions with limited ground-truth annotations through confidence filtering and adaptive fusion to construct pseudo-masks, and then progressively improve the coverage and quality of pseudo-masks using a curriculum learning strategy.

## Method

### Overall Architecture
The input consists of H&E stained histopathological slice images and their sparse annotations (pixel-level annotations for a subset of glands). The framework comprises a student network and an EMA teacher network. The student network updates its gradients normally at each training step, while the parameters of the teacher network are updated via exponential moving average: $\theta_T \leftarrow \alpha \theta_T + (1-\alpha) \theta_S$. The training loop proceeds as follows: teacher network prediction $\rightarrow$ confidence filtering $\rightarrow$ adaptive fusion with ground truth $\rightarrow$ refined pseudo-mask generation $\rightarrow$ student network training.

### Key Designs

1. **EMA-stabilized Teacher Network**:

    - **Function**: Generate temporally smoothed and stable predictions to prevent noisy prediction feedback during the early stages of student network training.
    - **Mechanism**: The teacher parameters are the exponential moving average of the student parameters, with the momentum coefficient increasing progressively during training (from 0.99 to 0.999). Since the EMA averages student parameters across multiple steps, the teacher's predictions are more stable than the student network at any single time step, providing more reliable pseudo-labels, especially in the early training stages.
    - **Design Motivation**: The Mean Teacher paradigm has proven effective in semi-supervised learning; this work extends it to weakly supervised scenarios to address the challenge of unstable pseudo-label quality.

2. **Confidence Filtering and Adaptive Fusion**:

    - **Function**: Filter out high-quality regions from the teacher's predictions and fuse them with the limited ground-truth annotations.
    - **Mechanism**: A dynamic threshold (e.g., >0.7 for high confidence) is applied to the prediction probability maps of the teacher network, retaining only high-confidence regions as pseudo-labels. In areas with ground-truth annotations, ground truth is always used; in unannotated areas with high teacher confidence, teacher predictions are used; in low-confidence areas, no supervisory signal is provided (loss is ignored). The fusion formula is $M_{fused}(x) = \mathbb{1}_{GT}(x) \cdot M_{GT}(x) + \mathbb{1}_{conf}(x) \cdot M_{teacher}(x)$.
    - **Design Motivation**: Directly using all predictions from the teacher as pseudo-labels would introduce significant noise. Confidence filtering ensures that only reliable information is propagated, and adaptive fusion preserves accuracy where ground-truth annotations are available.

3. **Curriculum-guided Progressive Refinement**:

    - **Function**: Gradually expand the coverage of pseudo-masks as training progresses.
    - **Mechanism**: At the beginning of training, a relatively high confidence threshold is used (accepting only the most reliable pseudo-labels), which is dynamically decreased as the model's capacity improves to incorporate more teacher-predicted regions. This forms a curriculum learning strategy—first learning the easy, high-confidence regions, and then gradually challenging low-confidence regions. Meanwhile, the update frequency of the pseudo-masks is also adjusted during training, updating more frequently in the early stages for rapid coverage expansion and less frequently in later stages to maintain stability.
    - **Design Motivation**: Training with all pseudo-labels at once easily leads to overfitting on noisy labels. A progressive curriculum strategy allows the model to gradually establish a comprehensive understanding of gland morphology.

### Loss & Training
A combination of standard cross-entropy loss and Dice loss is utilized: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Dice}$, where the loss is computed only on pixels that are annotated or have high-confidence pseudo-labels. Low-confidence regions are marked as ignore regions and do not participate in gradient backpropagation.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Fully Supervised | Weakly Supervised SOTA |
|--------|------|------|---------|-----------|
| GlaS | mIoU | 80.10 | 85.20 | 74.30 |
| GlaS | mDice | 89.10 | 92.10 | 83.50 |
| TCGA-COAD | mDice | 84.70 | 88.50 | 78.20 |
| TCGA-READ | mDice | 82.30 | 87.10 | 76.80 |
| SPIDER | mDice | 71.50 | 82.40 | 65.10 |

### Ablation Study

| Configuration | GlaS mIoU | GlaS mDice | Description |
|------|----------|-----------|------|
| Full model | 80.10 | 89.10 | Full framework |
| w/o EMA Teacher | 73.40 | 82.60 | No stable teacher, -6.7 mIoU |
| w/o Confidence Filtering | 75.80 | 85.20 | Noisy pseudo-labels lead to -4.3 |
| w/o Progressive Refinement | 77.20 | 86.50 | Static threshold, -2.9 |
| w/o Adaptive Fusion | 76.50 | 85.90 | Using only teacher predictions, -3.6 |

### Key Findings
- The EMA teacher network is the most critical component; removing it causes a severe performance drop of 6.7 mIoU, demonstrating that stable pseudo-labels are the foundation of success.
- Performance drops significantly on the SPIDER dataset (due to domain shift), reflecting the framework's sensitivity to the domain distribution of training data.
- The performance gap compared to fully supervised methods is only around 5 mIoU, while annotation costs can be reduced by over 80%.
- Cross-dataset evaluation (TCGA-COAD/READ) shows decent performance without additional annotations, indicating strong generalization capability.

## Highlights & Insights
- **Elegant Design of Progressive Pseudo-Label Strategy**: The dynamic adjustment of confidence thresholds mimics an "easy-to-hard" learning process. This curriculum learning paradigm can be generalized to other weakly supervised tasks (e.g., weakly supervised object detection, point-supervised segmentation, etc.).
- **Efficient Utilization of Sparse Annotations**: The framework fully utilizes a small number of precise annotations as anchors, "diffusing" knowledge to unannotated regions through the teacher network, achieving an order-of-magnitude increase in annotation efficiency.
- **Plug-and-Play Framework Design**: The Teacher-Student + EMA architecture is decoupled from the specific segmentation network structure and can be replaced with any advanced segmentation backbone.

## Limitations & Future Work
- The performance drop on the SPIDER dataset indicates that the framework is sensitive to domain shift, which requires integration with domain adaptation techniques in the future.
- Currently validated only on gland segmentation; its effectiveness on more complex multi-class segmentation (e.g., simultaneous segmentation of glands and stroma) remains unknown.
- The scheduling strategy for confidence thresholds is currently a preset fixed schedule; adaptive adjustments could be considered in the future.
- For extreme scenarios with extremely sparse annotations (e.g., only one gland labeled per image), the robustness of the framework remains to be verified.

## Related Work & Insights
- **vs CAM-based WSSS**: Traditional CAM-based methods can only locate the most salient regions and easily generate false positives in background regions. The proposed teacher-student framework and confidence filtering effectively circumvent both issues.
- **vs Mean Teacher (Semi-supervised)**: The original Mean Teacher assumes a large amount of unlabeled data and a small amount of precisely annotated data. The innovation of this work lies in combining the pseudo-mask refinement mechanism with Mean Teacher to adapt to weakly supervised scenarios (incomplete annotations instead of no annotations).
- **vs SAM (Segment Anything)**: SAM requires prompts (points/boxes), whereas the proposed method can autonomously discover unannotated gland regions.

## Rating
- Novelty: ⭐⭐⭐ Although individual components (EMA, pseudo-labeling, curriculum learning) are not entirely new, their combination and application are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets, including cross-domain evaluations, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-introduced medical background.
- Value: ⭐⭐⭐ High practical value, significantly reducing pathological annotation costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[CVPR 2025\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)

</div>

<!-- RELATED:END -->
