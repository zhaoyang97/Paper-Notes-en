---
title: >-
  [Paper Note] Knowledge Distillation with Refined Logits
description: >-
  [ICCV 2025][Model Compression][Knowledge Distillation] RLD refines teacher knowledge into two complementary forms — Sample Confidence and Masked Correlation — to mitigate the negative effects of teacher mispredictions wi…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Logit Distillation"
  - "Class Correlation"
  - "Teacher Error Correction"
date: 2026-05-08
content_hash: 8fcbc2397d376398
---

# Knowledge Distillation with Refined Logits

**Conference**: ICCV 2025
**arXiv**: [2408.07703](https://arxiv.org/abs/2408.07703)  
**Code**: [https://github.com/zju-SWJ/RLD](https://github.com/zju-SWJ/RLD)  
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Logit Distillation, Class Correlation, Teacher Error Correction, Model Compression

## TL;DR

RLD refines teacher knowledge into two complementary forms — Sample Confidence and Masked Correlation — to mitigate the negative effects of teacher mispredictions without disrupting inter-class correlations. It consistently outperforms existing logit distillation methods on both CIFAR-100 and ImageNet.

## Background & Motivation

Knowledge distillation leverages high-performance teacher models to guide the training of lightweight student models. **Logit distillation** has regained attention due to its simplicity, effectiveness, and generality. Zhao et al. (DKD) demonstrated, through decoupling the classical KL divergence loss, that logit distillation can match or even surpass feature distillation. However, **existing logit distillation methods largely overlook a critical issue: teacher models also make mistakes**.

When a teacher produces an incorrect prediction, a severe conflict arises between the standard distillation loss (KL divergence, which forces the student to align with the teacher's output) and the cross-entropy loss (which encourages the student to predict the correct class). This conflict impedes student performance, as the student is forced to trade off between "imitating the teacher" and "predicting correctly."

Existing correction methods (LA, RC, LR) attempt to rectify teacher logits using label information:
- **LA (Label Augment)**: Swaps the values of the predicted top-1 class and the ground-truth class.
- **RC (Review & Correction)**: Amplifies the probability of the ground-truth class.
- **LR (Label Revision)**: Interpolates one-hot labels with teacher soft labels.

However, through a toy example (Figure 1), this paper clearly demonstrates the fatal flaw of these approaches: **swap/augmentation operations destroy inter-class correlations**. For instance, if an image depicts a "lion" but the teacher predicts "forest," the swap operation severely disrupts the ranking of "lion" and "tiger" (highly correlated classes), impeding the transfer of "dark knowledge" (semantic relationships between classes).

- **Key Challenge**: How to correct teacher mispredictions while preserving valuable inter-class correlation information.

- **Core Idea**: Refine teacher knowledge into two complementary forms — (1) **Sample Confidence**: transmit only the teacher's confidence level for a given sample, without forcing the student to replicate incorrect class rankings; (2) **Masked Correlation**: dynamically mask potentially erroneous portions of the teacher's predictions (classes ranked higher than the ground-truth class), preserving correlations among the remaining classes. The more errors the teacher makes, the more classes are masked, and the less knowledge is transferred — but no incorrect information is propagated.

## Method

### Overall Architecture

The total loss of RLD consists of three components:

$$L_{RLD} = L_{CE} + \alpha \cdot L_{SCD} + \beta \cdot L_{MCD}$$

- $L_{CE}$: Standard cross-entropy loss (student predictions vs. ground-truth labels).
- $L_{SCD}$: Sample Confidence Distillation loss.
- $L_{MCD}$: Masked Correlation Distillation loss.

The three components are complementary: $L_{CE}$ ensures correctness, $L_{SCD}$ transfers the teacher's confidence level, and $L_{MCD}$ transfers semantic inter-class relationships.

### Key Designs

1. **Sample Confidence Distillation (SCD)**:

    - **Function**: Compresses the teacher's and student's logit distributions into binary distributions, transmitting the information of "how confident the teacher is about the current sample."
    - **Mechanism**:
        - Teacher binary distribution: $b^T = \{\hat{p}_{max}^T, 1 - \hat{p}_{max}^T\}$ (maximum predicted probability vs. the rest).
        - Student binary distribution: $b^S = \{\hat{p}_{true}^S, 1 - \hat{p}_{true}^S\}$ (ground-truth class probability vs. the rest).
        - Alignment: $L_{SCD} = \tau^2 \text{KL}(b^T, b^S)$.
    - **Key Insight**: The teacher side uses the probability of the predicted top-1 class, while the student side uses the probability of the ground-truth class. When the teacher predicts correctly, both sides align on the same class. When the teacher errs, the student is guided to "predict the ground-truth class with the teacher's level of confidence," rather than replicating the teacher's incorrect prediction.
    - **Gradient Analysis**: $\frac{\partial L_{SCD}}{\partial z_i} = p_i^S - p_{max}^T$ (for $i$ = true), which differs from $\frac{\partial L_{CE}}{\partial z_i} = p_i^S - y_i$ — SCD provides a softer supervision signal, mitigating overfitting.
    - **Design Motivation**: Pure cross-entropy hard-codes the target as 1 (overfitting risk), whereas SCD sets the target as the teacher's maximum predicted probability (typically $<1$), providing a regularization effect.

2. **Masked Correlation Distillation (MCD)**:

    - **Function**: Dynamically masks all classes ranked above the ground-truth class in the teacher's logits, performing KL alignment only over the remaining classes.
    - **Mechanism**:
        - Mask set: $M_{ge} = \{i \mid z_i^T \geq z_{true}^T, 1 \leq i \leq C\}$.
        - Masked distribution: $\tilde{p}_i = \frac{\exp(z_i/\tau)}{\sum_{c \notin M_{ge}} \exp(z_c/\tau)}$, computed only for $i \notin M_{ge}$.
        - Alignment: $L_{MCD} = \tau^2 \text{KL}(\tilde{p}^T, \tilde{p}^S)$.
    - **Adaptive Property**: The more accurate the teacher's prediction (i.e., the higher the ground-truth class is ranked), the fewer classes are masked and the more inter-class correlations are transferred. The worse the teacher's prediction, the more classes are masked (in the extreme case, only the ground-truth class itself is masked), reducing the propagation of erroneous information.
    - **Why $M_{ge}$ (≥) rather than $M_g$ (>)**: Ablation experiments and the toy example (Figure 6) show that $M_g$ places ground-truth class knowledge simultaneously into both SCD and MCD, causing conflicts between the two losses. $M_{ge}$ ensures the ground-truth class is always masked, eliminating this conflict.
    - **Design Motivation (Figure 3b)**: Masked classes grant the student sufficient freedom — its predictions for those classes can differ substantially from the teacher's without incurring a large loss. This allows the student to "correct" the teacher's ranking errors.

3. **Theoretical Connection to DKD**:

    - When the teacher always predicts correctly, RLD reduces to DKD.
    - RLD provides a new interpretation of why DKD's NCKD component (distribution alignment after masking the ground-truth class) is effective: it grants the student freedom to adjust the ground-truth class ranking, indirectly mitigating the impact of teacher errors.
    - RLD tolerates larger $\alpha$ values than DKD, since the refined knowledge eliminates erroneous information.

### Loss & Training

- Default hyperparameters: $\alpha = 1, \beta = 4$.
- Temperature $\tau$ follows standard settings.
- The training framework is consistent with LSKD, DKD, and CRD to ensure fair comparison.
- All results are averaged over three runs.

## Key Experimental Results

### Main Results

**CIFAR-100 Homogeneous Distillation (same architecture family for teacher and student)**:

| Teacher→Student | KD | DKD | LA | RC | LR | **RLD** |
|---|---|---|---|---|---|---|
| ResNet32×4→ResNet8×4 | 73.33 | 76.32 | 73.46 | 74.68 | 76.06 | **76.64** |
| VGG13→VGG8 | 72.98 | 74.68 | 73.51 | 73.37 | 74.66 | **74.93** |
| WRN-40-2→WRN-40-1 | 73.54 | 74.81 | 73.75 | 74.07 | 74.42 | **74.88** |
| ResNet56→ResNet20 | 70.66 | 71.97 | 71.24 | 71.63 | 70.74 | **72.00** |
| ResNet110→ResNet32 | 73.08 | 74.11 | 73.39 | 73.44 | 73.52 | 74.02 |
| ResNet110→ResNet20 | 70.67 | 71.06 | 70.86 | 71.41 | 70.61 | **71.67** |

**ImageNet (Large-Scale Validation)**:

| Teacher→Student | KD | DKD | RC | LR | **RLD** |
|---|---|---|---|---|---|
| Res34→Res18 Top-1 | 71.03 | 71.70 | 71.59 | 70.29 | **71.91** |
| Res50→MN-V1 Top-1 | 70.50 | 72.05 | 71.86 | 71.76 | **72.75** |

RLD surpasses all feature and logit distillation methods on ImageNet, with larger gains than on CIFAR-100.

### Ablation Study

| $L_{CE}$ | $L_{SCD}$ | $L_{MCD}$ (Mask) | Accuracy | Notes |
|---|---|---|---|---|
| ✓ | | | 72.50 | CE only |
| ✓ | ✓ | | 73.55 | +SCD provides regularization |
| ✓ | | ✓ ($M_g$) | 75.50 | MCD with $M_g$ mask |
| ✓ | | ✓ ($M_{ge}$) | 75.64 | MCD with $M_{ge}$ mask (better) |
| ✓ | ✓ | ✓ ($M_g$) | 75.53 | SCD + MCD($M_g$) conflict |
| ✓ | ✓ | ✓ ($M_{ge}$) | **76.64** | Full RLD (optimal) |

### Key Findings

- **The lower the teacher's training accuracy, the greater the advantage of RLD over DKD** (Figure 4). Since teacher training accuracy is lower on ImageNet than on CIFAR-100, RLD achieves more substantial improvements on ImageNet.
- **Reverse distillation** (weak teacher guiding a stronger student): RLD still substantially outperforms DKD (maximum $\Delta$ of +0.69%), demonstrating that refined knowledge is especially valuable when teacher capacity is limited.
- The subtle distinction between $M_{ge}$ and $M_g$ is clearly explained by the ablation study: $M_g$ induces conflicting losses between SCD and MCD on the ground-truth class.
- Students trained with RLD exhibit larger logit discrepancies from the teacher compared to DKD (Figure 5), yet achieve better performance — demonstrating that "blindly aligning with the teacher" is suboptimal.

## Highlights & Insights

1. The problem of "correcting teacher errors while preserving inter-class correlations" is clearly formulated and addressed for the first time. Prior correction methods (LA, RC, LR) focused on correcting prediction values while overlooking this critical trade-off.
2. The asymmetric design in SCD — using the teacher's maximum predicted probability on the teacher side and the ground-truth class probability on the student side — simultaneously achieves error correction and regularization.
3. The adaptive masking mechanism in MCD elegantly realizes the principle of "learn less when the teacher errs more," without requiring additional hyperparameters to control the masking ratio.
4. The theoretical connection to DKD is clearly established, offering a new explanation for why the NCKD component in DKD is effective.
5. RLD is orthogonally composable with logit normalization techniques such as LSKD (Table 5).

## Limitations & Future Work

- The optimal values of $\alpha$ and $\beta$ vary substantially across different distillation pairs (Figure 7); automated hyperparameter tuning is an important direction for future improvement.
- Validation is limited to classification tasks; extension to downstream tasks such as detection and segmentation has not been explored.
- The masking strategy is based on logit rankings, which may be unstable when logit value differences are extremely small.
- Combination with feature distillation has not been explored (the paper mentions in Future Work that feature alignment via CAM could be incorporated).

## Related Work & Insights

- DKD (Decoupled Knowledge Distillation) is the most direct baseline; RLD builds upon it to address teacher error through knowledge refinement.
- The dynamic temperature idea from CTKD can be combined with RLD.
- The approach of "adaptively masking unreliable signals" may serve as a useful reference for other scenarios involving learning from noisy signals, such as noisy label learning.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
