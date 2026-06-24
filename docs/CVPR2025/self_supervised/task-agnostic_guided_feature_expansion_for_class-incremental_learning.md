---
title: >-
  [Paper Note] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning
description: >-
  [CVPR 2025][Self-Supervised Learning][Class-Incremental Learning] The TagFex framework is proposed to continuously capture task-agnostic features via continual self-supervised learning. These features are adaptively integrated with task-specific features using merge attention and then distilled back into the inference model, mitigating the feature collision problem in expansion-based class-incremental learning.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Class-Incremental Learning"
  - "Feature Expansion"
  - "Task-Agnostic Features"
  - "Feature Collisions"
date: 2026-05-08
content_hash: 553e3c18a950107f
---

# Task-Agnostic Guided Feature Expansion for Class-Incremental Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.00823](https://arxiv.org/abs/2503.00823)  
**Code**: [GitHub](https://github.com/bwnzheng/TagFex_CVPR2025)  
**Area**: Others  
**Keywords**: Class-Incremental Learning, Feature Expansion, Task-Agnostic Features, Self-Supervised Learning, Feature Collisions

## TL;DR

The TagFex framework is proposed to continuously capture task-agnostic features via continual self-supervised learning. These features are adaptively integrated with task-specific features using merge attention and then distilled back into the inference model, mitigating the feature collision problem in expansion-based class-incremental learning.

## Background & Motivation

Expansion-based class-incremental learning methods (such as DER) expand new feature extractors for each new task while keeping old models frozen. Although effective in mitigating forgetting, they suffer from the **feature collision** problem: features learned by the new task may overlap with old task features (e.g., both tasks relying on color features to distinguish different categories), leading to cross-task misclassifications.

Existing solutions (such as DER's auxiliary classifier) rely on a small number of rehearsal samples to encourage diverse feature learning, which leads to suboptimal results and unbalanced training due to the limited sample size. Centered Kernel Alignment (CKA) similarity analysis shows high feature similarity among various models in DER (~0.35), and GradCAM visualizations also indicate that they focus on similar regions.

Key Insight: **Classification tasks only require models to capture minimal necessary features** (task-specific), whereas a vast amount of useful yet classification-irrelevant features (task-agnostic) are ignored. If these task-agnostic features can be captured and transferred to subsequent tasks, the new models can learn more diverse features.

## Method

### Overall Architecture

TagFex consists of three stages: (1) training an independent model via Continual Self-Supervised Learning (CSSL) to continuously capture task-agnostic features; (2) adaptively fusing task-agnostic and task-specific features via merge attention; and (3) transferring the enriched fused features back to the task-specific model via KL-divergence distillation (only the task-specific model is used for inference).

### Key Design 1: Continual Self-Supervised Learning for Task-Agnostic Feature Capturing

- **Function**: Continuously learn classification-irrelevant visual representations across tasks.
- **Mechanism**: CaSSLe (a SimCLR-based continual self-supervised method) is adopted to train an independent task-agnostic model $f_{\text{ta}}$. A standard InfoNCE loss is used in the initial task, while incremental tasks incorporate a predictive loss. This loss trains a predictor $g(\cdot)$ so that the current features can predict the previous features $f'_{\text{ta}}$, ensuring that the representation capacity increases incrementally across tasks.
- **Design Motivation**: Self-supervised learning is unconstrained by classification targets, enabling it to discover features ignored by classification tasks (e.g., texture, shape), and continual learning ensures the accumulation of increasingly rich representations as incremental tasks progress.

### Key Design 2: Merge Attention for Adaptive Feature Fusion

- **Function**: Extract useful information for the current classification task from task-agnostic features.
- **Mechanism**: The task-specific feature map is treated as the Query, which is then concatenated with both task-specific and task-agnostic Key/Value representations to perform multi-head attention: $O^{(h)} = \text{Softmax}(\frac{Q^{(h)}[K_{\text{ts}}^{(h)}, K_{\text{ta}}^{(h)}]^T}{\sqrt{d/h}})[V_{\text{ts}}^{(h)}, V_{\text{ta}}^{(h)}]$. The gradient of the task-agnostic model is stopped, rendering it unaffected by classification.
- **Design Motivation**: Since the two features reside in different spaces, direct concatenation is unsuitable. The attention mechanism allows the task-specific features to selectively "retrieve" valuable information from task-agnostic features. During training, attention gradually shifts from the ta-side to the ts-side, indicating that information is progressively assimilated.

### Key Design 3: Knowledge Transfer (Distillation back to the Inference Model)

- **Function**: Transfer the enriched fused feature information to the task-specific model, eliminating the need for the task-agnostic model during inference.
- **Mechanism**: KL-divergence $\mathcal{L}_{\text{trans}} = D_{\text{KL}}(\text{StopGrad}(p_m) \| p_{\text{ts}})$ is utilized to distill the logits of the merge classifier into the task-specific classifier. During inference, only the task-specific model is used, maintaining the same parameter count as DER.
- **Design Motivation**: Performing inference directly with fused features would be susceptible to distribution shifts caused by continuous updates to the task-agnostic model. Distilling back to the task-specific model ensures both stability and the transfer of diverse feature information.

### Loss & Training

$\mathcal{L} = \lambda_{\text{ta}}\mathcal{L}_{\text{ta}} + \lambda_{\text{mcls}}\mathcal{L}_{\text{mcls}} + \mathcal{L}_{\text{ts}}$, where $\mathcal{L}_{\text{ts}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{aux}} + \mathcal{L}_{\text{trans}}$.

## Key Experimental Results

### Main Results: Class-Incremental Learning Results on Various Datasets

| Method | CIFAR100 10-10 (Last/Avg) | ImageNet100 10-10 (Last/Avg) | ImageNet1000 100-100 (Last/Avg) |
|------|---------------------------|-----------------------------|---------------------------------|
| iCaRL | 49.52/64.64 | 50.98/67.11 | 40.47/57.55 |
| DER | 64.35/75.36 | 66.71/77.18 | 58.83/66.87 |
| BEEF | 60.98/71.94 | 68.78/77.62 | 58.67/67.09 |
| **TagFex** | **68.23/78.45** | **70.84/79.27** | **61.45/68.32** |
| TagFex-P | 67.34/78.02 | 69.21/78.56 | 60.14/67.65 |

### Ablation Study (CIFAR100)

| Task-agnostic | Merge Attn | Knowledge Transfer | 10-10 Last/Avg |
|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✓ | 64.45/75.34 |
| ✓ | ✓ | ✗ | 65.86/76.32 |
| ✓ | ✓ | ✓ | **68.23/78.45** |

### Key Findings

- TagFex consistently improves accuracy by 3-4% compared to DER, while maintaining the same number of parameters during inference.
- The CKA similarity decreases from ~0.35 in DER to ~0.2, demonstrating a significant improvement in feature diversity.
- Attention visualizations show that the model focuses on the task-agnostic side in the early stages of training, which later shifts to the task-specific side.
- The pruned version, TagFex-P, reduces parameters from 61.6M to 11.6-14.4M, with only a minor drop in accuracy.
- Replacing the SSL method (SimCLR $\to$ BYOL) yields further improvements, indicating that the framework is insensitive to the specific SSL method used.

## Highlights & Insights

1. **Clear formulation of the feature collision problem**: CKA and GradCAM visualizations are utilized to demonstrate the lack of feature diversity in existing methods.
2. **Decoupled training-inference design**: The task-agnostic model assists during training but is completely bypassed during inference, incurring zero additional inference overhead.
3. **Attention evolution of Merge Attention**: The attention migration from ta $\to$ ts intuitively illustrates the process of knowledge assimilation.

## Limitations & Future Work

- An additional task-agnostic model needs to be maintained during training (storage equivalent to approximately 300 samples), though its superiority has been verified under memory-aligned experimental setups.
- Currently, the method is only validated on CNN backbones (ResNet18). Its applicability to ViT and other architectures remains to be explored.
- The additional training overhead introduced by self-supervised learning might be non-negligible in edge deployment scenarios.

## Related Work & Insights

- The concept of using self-supervised learning to discover features overlooked by classification tasks is novel and could be extended to other scenarios requiring feature diversity.
- The design paradigm of "assisting training without participating in inference" is worth exploring in broader tasks.

## Rating

⭐⭐⭐⭐ — The problem is thoroughly analyzed, and the proposed solution is elegant and principled (feature collision $\to$ diversity $\to$ task-agnostic features). Zero additional inference overhead is a significant practical advantage. The experiments are comprehensive, featuring a fair comparison under memory-aligned configurations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)
- [\[CVPR 2025\] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [\[CVPR 2026\] Semantic-Guided Global-Local Collaborative Prompt Learning for Few-Shot Class Incremental Learning](../../CVPR2026/self_supervised/semantic-guided_global-local_collaborative_prompt_learning_for_few-shot_class_in.md)
- [\[CVPR 2026\] DGS: Dual Gradient and Semantic-Shift Guided Low-Rank Adaptation for Class Incremental Learning](../../CVPR2026/self_supervised/dgs_dual_gradient_and_semantic-shift_guided_low-rank_adaptation_for_class_increm.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](../../CVPR2026/self_supervised/representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
