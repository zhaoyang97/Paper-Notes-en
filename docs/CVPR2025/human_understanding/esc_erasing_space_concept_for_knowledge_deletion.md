---
title: >-
  [Paper Note] ESC: Erasing Space Concept for Knowledge Deletion
description: >-
  [CVPR 2025][Human Understanding][Knowledge Deletion] This paper proposes ESC (Erasing Space Concept), which performs SVD on the feature space of the data to be forgotten and removes the principal component directions, achieving training-free, feature-level knowledge deletion. It defines the "Knowledge Deletion" task for the first time and proposes the Knowledge Retention Score to evaluate the effectiveness of feature-level unlearning.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Knowledge Deletion"
  - "Machine Unlearning"
  - "SVD Subspace"
  - "Feature-level Privacy"
  - "Training-free Method"
date: 2026-05-08
content_hash: 940e60d256990192
---

# ESC: Erasing Space Concept for Knowledge Deletion

**Conference**: CVPR 2025  
**arXiv**: [2504.02199](https://arxiv.org/abs/2504.02199)  
**Code**: [https://github.com/KU-VGI/ESC](https://github.com/KU-VGI/ESC)  
**Area**: Human Understanding  
**Keywords**: Knowledge Deletion, Machine Unlearning, SVD Subspace, Feature-level Privacy, Training-free Method

## TL;DR

This paper proposes ESC (Erasing Space Concept), which performs SVD on the feature space of the data to be forgotten and removes the principal component directions, achieving training-free, feature-level knowledge deletion. It defines the "Knowledge Deletion" task for the first time and proposes the Knowledge Retention Score to evaluate the effectiveness of feature-level unlearning.

## Background & Motivation

**Background**: Machine Unlearning (MU) aims to remove the influence of specific data from a trained model. Existing methods (such as Negative Gradient, Random Label, SalUn, etc.) achieve unlearning by modifying model weights through end-to-end training.

**Limitations of Prior Work**: Existing MU methods suffer from severe feature-level knowledge retention. Although the classification head is effectively modified to alter predictions, the knowledge in the feature extractor remains largely untouched. Experiments show that simply training a new linear probe on the frozen features of the "unlearned" model can recover a significant amount of "deleted" knowledge (e.g., the linear probe recovery rate in Figure 1 is close to that of the original model).

**Key Challenge**: Existing methods employ logit-based loss functions for end-to-end training. The model tends to find a "shortcut"—modifying only the classification head suffices to minimize the unlearning loss on logits, leaving the knowledge in the feature extractor intact.

**Goal**: To achieve feature-level knowledge deletion, ensuring that deleted knowledge cannot be recovered from features even when using methods like linear probing.

**Key Insight**: Operate directly in the feature space—use SVD to identify the principal directions of the data to be forgotten, and then project features onto the remaining subspace to eliminate activations along these directions. This process can be accomplished without training.

**Core Idea**: Decomposing the feature matrix of the data to be forgotten using SVD and removing the top p% principal component directions equates to training-free, feature-level knowledge deletion.

## Method

### Overall Architecture

The classification model is divided into a feature extractor $h_\psi$ and a classification head $g_\phi$. The data to be forgotten $\mathcal{D}_f$ is passed through $h_\psi$ to obtain the feature matrix $Z_f$. SVD is performed on $Z_f$ to obtain the principal directions $U$, and the top $k$ principal directions are removed to yield $U_P$. During inference, all features are projected using $U_P U_P^\top$.

### Key Designs

1. **ESC (Training-Free Version)**:

    - **Function**: Deleting unlearning knowledge in the feature space without requiring training.
    - **Mechanism**: Perform SVD on the feature matrix of the data to be forgotten $Z_f = U \Sigma V^\top$, and remove the top $k = \frac{d}{100} \cdot p$ principal component directions to obtain $U_P = U[k:]$. During inference, features are projected as $h_{\psi_P}(x) = U_P U_P^\top h_\psi(x)$. The removed principal directions correspond to those with the largest variance in the data to be forgotten, which contain the most discriminative information.
    - **Design Motivation**: Toy experiments in Figure 3 show that after removing the principal components, the cosine similarity between the features of the forgotten class and the original features drops from >0.5 to <0.35, while other classes remain largely unaffected.

2. **ESC-T (Training-Based Version)**:

    - **Function**: Achieve more fine-grained knowledge deletion through a learnable mask, balancing unlearning and retention.
    - **Mechanism**: Instead of directly removing the entire principal direction, a learnable mask $M_0$ (initialized to all ones) is introduced for each principal direction. The mask is trained using a Penalized Cross-Entropy Loss. A penalty is applied when the model correctly predicts the forgotten data, driving the mask to turn off corresponding elements, ultimately yielding the refined principal directions $U_R$.
    - **Design Motivation**: The hard thresholding in ESC might lead to over-forgetting (removing the entire direction instead of key elements within that direction). ESC-T achieves "precision-surgery" style deletion through element-wise masking.

3. **Knowledge Retention Score (KR)**:

    - **Function**: Evaluate the degree of knowledge retention at the feature level.
    - **Mechanism**: Freeze the feature extractor of the model after unlearning, and train only a new linear probe, measuring the classification accuracy on both forgotten and retained data. If the linear probe can recover high accuracy on forgotten data, it indicates that feature-level knowledge still exists.
    - **Design Motivation**: Existing evaluations (accuracy, MIA) only focus on the output layer, failing to detect residual knowledge within the features.

### Loss & Training

ESC is entirely training-free. ESC-T utilizes Penalized Cross-Entropy Loss: $\mathcal{L}_{PCE} = -\sum_c \hat{y}_c \log(1 - p_c)$, which produces a high loss when the model correctly predicts the forgotten classes, driving the mask to turn off relevant features. Only the mask parameters are trained, while the backbone is frozen.

## Key Experimental Results

### Main Results

CIFAR-10 knowledge deletion comparison (All-CNN):

| Method | $D_f$↓ | $D_r$↑ | $D_{ft}$↓ | HM↑ | MIA | KR-$D_f$↓ |
|------|--------|--------|-----------|-----|-----|-----------|
| Original | 98.42 | 98.29 | 85.90 | 3.11 | 57.68 | 98.40 |
| Retrain | 0.00 | 96.93 | 0.00 | 98.44 | 50.06 | 41.28 |
| SalUn | 0.00 | 98.86 | 0.00 | 99.43 | 56.42 | **62.03** |
| **ESC** | 9.46 | 96.52 | 10.73 | 93.43 | 53.02 | **10.21** |
| **ESC-T** | **0.00** | **97.23** | **0.00** | **98.60** | 56.72 | **14.62** |

Key Findings: SalUn achieves perfect unlearning at the output layer ($D_f$=0) but suffers from a KR as high as 62% (feature-level knowledge remains!). ESC/ESC-T reduces the KR to 10-15%.

### Ablation Study

| Configuration | Unlearning Effect | Retention Effect | Description |
|------|---------|---------|------|
| p=10% | Partial Unlearning | Excellent Retention | Incomplete deletion |
| p=30% | Decent Unlearning | Good Retention | Optimal trade-off |
| p=50% | Complete Unlearning | Degraded Retention | Over-forgetting |
| ESC-T | Complete Unlearning | Optimal Retention | Precise control via learnable mask |

### Key Findings
- **Existing MU methods fail at the feature level**: The linear probe recovery rate is as high as 80-96%, indicating that while classification head knowledge is deleted, knowledge in the feature extractor remains intact.
- **ESC achieves training-free feature-level deletion**: Simply using SVD + projection can reduce KR from 98% to 10% without any gradient computation.
- **ESC-T achieves better refinement**: The learnable mask offers a superior trade-off between unlearning and retention, yielding an HM score close to the ideal value of Retrain.
- **Applicable to face scenarios**: It is also effective on face datasets such as CelebA-HQ, satisfying real-world privacy deletion requirements.

## Highlights & Insights

- **Exposed the "false security" of existing machine unlearning**: Using the KR metric, this work demonstrates that many unlearning methods merely modify the classification head instead of actually deleting knowledge—a critical wake-up call for the entire machine unlearning community.
- **Elegant application of SVD**: Using principal component directions to represent the "concept space", where removing principal directions equates to removing concepts—an abstraction that is both intuitively sound and mathematically rigorous.
- **Training-free speed advantage**: ESC only requires a single SVD decomposition (on the scale of seconds), which is several orders of magnitude faster than any gradient-based method.

## Limitations & Future Work

- **Principal directions may be shared across classes**: If the classes to be forgotten and retained share certain principal directions (e.g., background features), removal may degrade the performance of the retained classes.
- **The fixed truncation ratio p requires tuning**: Different datasets and models require different values of p.
- **Validated only on classification tasks**: Feature-level unlearning in generative models (such as diffusion models) remains a greater challenge.
- **The KR metric relies on linear probing**: Non-linear probing might recover more information.

## Related Work & Insights

- **vs SalUn**: SalUn guides gradients using saliency maps to achieve perfect unlearning at the output layer, but its KR is as high as 62%. ESC operates directly in the feature space, reducing KR to 10%.
- **vs ℓ1-sparse**: The ℓ1-sparse method can also reduce $D_f$ to 0, but it suffers from a larger drop in retention accuracy (89.95% vs 97.23% for ESC-T).
- **vs Retrain**: Although ESC-T's KR is lower than Retrain's (14.6% vs 41.3%), their HM scores are close, indicating that it can delete knowledge even more thoroughly at the feature level than training from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define the feature-level knowledge deletion problem, with the KR metric highlighting a blind spot in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets and models with in-depth KR analysis, but lacks large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, but mathematical symbols are somewhat heavy.
- Value: ⭐⭐⭐⭐⭐ Holds directional influence for the machine unlearning community; the KR metric could become a standard evaluation tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Homogeneous Dynamics Space for Heterogeneous Humans](homogeneous_dynamics_space_for_heterogeneous_humans.md)
- [\[NeurIPS 2025\] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](../../NeurIPS2025/human_understanding/k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](../../ICCV2025/human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[NeurIPS 2025\] Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge](../../NeurIPS2025/human_understanding/foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)

</div>

<!-- RELATED:END -->
