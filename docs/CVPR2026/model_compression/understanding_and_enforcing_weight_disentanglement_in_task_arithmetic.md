---
title: >-
  [Paper Note] Understanding and Enforcing Weight Disentanglement in Task Arithmetic
description: >-
  [CVPR 2026][Model Compression][Paper Note] This work proposes Task Feature Specialization (TFS) as a sufficient condition for weight disentanglement, revealing that its geometric consequence is the orthogonality of weight vectors. Based on this, the OrthoReg regularization method is introduced. By enforcing orthogonality among the column vectors of the weight u
tags:
  - CVPR 2026
  - Model Compression
date: 2026-05-08
content_hash: b6bf9c679bec1128
---
# Understanding and Enforcing Weight Disentanglement in Task Arithmetic

**Conference**: CVPR 2026  
**arXiv**: [2604.17078](https://arxiv.org/abs/2604.17078)  
**Code**: [GitHub](https://github.com/RL-MIND/OrthoReg)  
**Area**: Model Compression  
**Keywords**: Task Arithmetic, Model Merging, Weight Disentanglement, Orthogonal Regularization, Task Vectors

## TL;DR
This work proposes Task Feature Specialization (TFS) as a sufficient condition for weight disentanglement, revealing that its geometric consequence is the orthogonality of weight vectors. Based on this, the OrthoReg regularization method is introduced. By enforcing orthogonality among the column vectors of the weight update matrix during fine-tuning, OrthoReg promotes task vector disentanglement and significantly enhances the performance of various task arithmetic methods.

## Background & Motivation

1.  **Background**: Task Arithmetic is an efficient, training-free model editing paradigm. It combines, removes, or analogizes different skills by computing task vectors $\tau_t = \theta_t^* - \theta_0$ (the difference between fine-tuned and pre-trained weights) and performing algebraic operations (addition, subtraction).
2.  **Limitations of Prior Work**: While task arithmetic is effective in practice, it lacks a fundamental theoretical explanation. The existing concept of "weight disentanglement" (proposed by TTA) describes the ideal outcome—where the effects of different task vectors do not interfere—but fails to reveal its root cause. Specifically, the intrinsic properties required by the pre-trained model $\theta_0$ or task vectors $\tau_t$ to achieve disentanglement remain under-explored.
3.  **Key Challenge**: Weight disentanglement is a phenomenological description rather than a causal explanation. Existing methods are either computationally expensive (e.g., TTA requires Jacobian computations) or lack theoretical guarantees, failing to reliably generate high-quality task vectors.
4.  **Goal**: To answer two core questions: (1) What properties of a pre-trained model make it suitable for task arithmetic? (2) How can task vectors be constructed to actively promote weight disentanglement?
5.  **Key Insight**: Starting from the model's internal feature allocation mechanism, it is discovered that "Task Feature Specialization" (TFS) is a sufficient condition for disentanglement, with weight vector orthogonality being its observable geometric consequence.
6.  **Core Idea**: While TFS is abstract and cannot be directly enforced, its geometric result—orthogonality—is concrete and actionable. By enforcing the internal orthogonal structure of the weight update matrix during fine-tuning, weight disentanglement can be indirectly promoted.

## Method

### Overall Architecture
This work aims to explain why task arithmetic works and design a fine-tuning method to actively improve disentanglement quality. The logic chain follows: identifying the root cause of weight disentanglement theoretically (Task Feature Specialization, TFS), translating this abstract cause into an observable and optimizable geometric quantity (weight vector orthogonality), and finally implementing it as a plug-and-play regularization term, OrthoReg. In practice, each downstream task is fine-tuned individually. In addition to the standard task loss, a regularization term is added to constrain the column vectors of the weight update matrix $\Delta W$ to be mutually orthogonal. Once fine-tuning for each task is complete, task vectors are summed using standard task arithmetic $\theta_{MT} = \theta_0 + \sum_t \alpha_t \tau_t$ to obtain a multi-task model. In other words, OrthoReg only modifies the fine-tuning phase; the merging phase and inference process remain unchanged.

### Key Designs

**1. Task Feature Specialization (TFS): Attributing "Disentanglement" to an Internal Mechanism**

Previously, "weight disentanglement" was merely a description of an ideal phenomenon where the effects of task vectors do not interfere. This paper identifies TFS as the underlying property: the model assigns distinct internal features (i.e., different column vectors in the weight matrix) to different tasks. Formally, let $I_t$ be the set of specialized feature indices for task $t$ that make the model output sensitive to the corresponding activations $z_k$. TFS requires these feature sets to be disjoint ($I_t \cap I_j = \emptyset$). Under the NTK linearization assumption, this disjointness guarantees that the interference term $\tau_j^\top J(x) = 0$ for all $x \in \mathcal{D}_t$, meaning task vector $j$ has no impact on samples from task $t$, which is the definition of weight disentanglement. Furthermore, the authors prove that TFS naturally leads to a block-orthogonal structure in the weight matrix, linking a functional property to a measurable geometric property (orthogonality).

**2. OrthoReg: Enforcing the Geometric Consequences of TFS**

Since TFS cannot be directly used as a training objective because feature sets in real networks almost inevitably overlap, this work targets its geometric consequence: orthogonality. OrthoReg adds an orthogonality constraint as a loss term during fine-tuning. Specifically, the loss is defined as $\mathcal{L} = \mathcal{L}_{\text{task}}(\theta_0 + \Delta \theta) + \lambda \cdot \mathcal{L}_{\text{ortho}}(\Delta \theta)$, where

$$\mathcal{L}_{\text{ortho}} = \sum_l \big\|(\Delta W^{(l)})^\top \Delta W^{(l)} - I\big\|_F^2$$

This penalizes the deviation of the Gram matrix $(\Delta W)^\top \Delta W$ of each layer's update matrix from the identity matrix. This effectively enforces the columns of $\Delta W$ to be mutually orthogonal with unit norms. Theoretically, this promotes disentanglement via two mechanisms: norm control (preventing one task vector from dominating) and angle control (pushing the angles between task vectors toward 90°). This approach is efficient, requiring no architectural changes and being compatible with any fine-tuning method, including Parameter-Efficient Fine-Tuning (PEFT) like LoRA.

**3. Unification with TTA: Two Different Approaches to the Same Goal**

The authors revisit Tangent Task Arithmetic (TTA) through the lens of orthogonality, noting that both TTA and OrthoReg aim to push the inner product between task vectors toward zero ($\langle \tau_t, \tau_j \rangle \approx 0$). The difference lies in the implementation: TTA achieves this implicitly via linearization in the tangent space and NTK geometry at the cost of doubled memory and 2-3x training time, while OrthoReg explicitly enforces it via regularization more efficiently. This unification reinforces the claim that orthogonality is the underlying mechanism for disentanglement.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{ortho}}$, where $\lambda$ is selected from $[0.1, 100]$ via a validation set. During fine-tuning, the text encoder is frozen while the image encoder is updated. In the merging phase, a uniform scaling factor $\alpha$ is grid-searched across $\{0.0, 0.05, \dots, 1.0\}$.

## Key Experimental Results

### Main Results

**Task Addition (8 tasks, ViT-L-14)**:

| Method | Absolute Acc | Normalized Acc | Gain |
| :--- | :--- | :--- | :--- |
| Non-lin. FT | 84.07% | 89.19% | — |
| Non-lin. FT + OrthoReg | 88.23% | **100.08%** | +4.16 |
| TTA | 86.19% | 93.14% | — |
| TTA + OrthoReg | 87.52% | 96.44% | +1.33 |
| ATT-FT | 87.81% | 93.59% | — |
| ATT-FT + OrthoReg | **90.41%** | **100.05%** | +2.60 |

**Task Negation (Forgetting target task, ViT-L-14)**:

| Method | Target Acc↓ | Control Acc↑ | Forgetting Gain |
| :--- | :--- | :--- | :--- |
| ATT-FT | 24.85% | 76.42% | — |
| ATT-FT + OrthoReg | **14.67%** | 75.40% | -10.18 |

### Ablation Study

| Configuration | Absolute Acc | Description |
| :--- | :--- | :--- |
| ATT-FT + OrthoReg (ViT-L-14) | 90.41% | Full method |
| ATT-FT (No Reg) | 87.81% | -2.6% without OrthoReg |
| LoRA-ATT + OrthoReg | 89.16% | Effective for PEFT |
| LoRA-ATT (No Reg) | 87.02% | -2.14% without Reg |

### Key Findings
- **Normalized accuracy exceeding 100%**: Non-lin. FT + OrthoReg on ViT-L-14 reached 100.08%, meaning the merged multi-task model performs as well as or better than 8 independently fine-tuned models, achieving near-ideal weight disentanglement.
- **Significant reduction in task vector cosine similarity**: OrthoReg brings the cosine similarity of different task vectors close to 0, validating the predicted "angle control" mechanism.
- **Robustness to hyperparameters**: Performance improves steadily with increasing $\lambda$ and consistently outperforms baselines across a wide range of $\alpha$ values.

## Highlights & Insights
- The **causal chain of TFS → WVO → WD** is elegant: identifying "Task Feature Specialization" as the common cause connecting functional and geometric properties provides a paradigm for bridging abstract properties to actionable constraints.
- The **100%+ normalized accuracy** is the most impressive result: it proves that orthogonal constraints not only reduce inter-task interference but can also lead the merged model to outperform individual models, suggesting a beneficial regularization effect.
- The **simplicity of OrthoReg** is commendable: requiring only one regularization term $\|(\Delta W)^\top \Delta W - I\|_F^2$, it can be integrated into any fine-tuning pipeline without modifying architecture or inference.

## Limitations & Future Work
- The theory relies on the NTK linearization assumption; its applicability to deep, highly non-linear networks requires further verification.
- Validation is currently limited to CLIP-based ViT; experiments on other pre-training paradigms (e.g., MAE, DINOv2) are missing.
- Only 8 classification tasks were considered; performance on more tasks (e.g., 20+) or heterogeneous tasks (detection, segmentation) remains to be seen.
- Orthogonal constraints might be too restrictive when the number of columns $d$ is much larger than the number of rows $m$, potentially limiting expressivity.
- Future work could explore adaptive orthogonal constraints that adjust strength based on task similarity.

## Related Work & Insights
- **vs TTA (Tangent Task Arithmetic)**: TTA achieves task vector orthogonality implicitly through tangent space linearization but is computationally heavy (2-3x training time). OrthoReg explicitly enforces orthogonality and is more efficient; both share the same underlying mechanism.
- **vs TIES-Merging / DARE**: These are "during-merging" methods that reduce interference via pruning or sign voting. OrthoReg is a "pre-merging" method that generates high-quality task vectors at the source, making it complementary to merging-stage methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical chain of TFS → Orthogonality → Disentanglement is novel and complete; OrthoReg is simple yet powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across three model scales and multiple baselines, though task types are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation of theory and methodology; the logic from principle to experiment is seamless.
- Value: ⭐⭐⭐⭐⭐ Provides a profound theoretical foundation for task arithmetic, and OrthoReg offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Discovering Adaptive Task Dependencies for Efficient Multi-Task Representation Compression](discovering_adaptive_task_dependencies_for_efficient_multi-task_representation_c.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](../../CVPR2025/model_compression/task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[ICML 2026\] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](../../ICML2026/model_compression/the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)

</div>

<!-- RELATED:END -->
