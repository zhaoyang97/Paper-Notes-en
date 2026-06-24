---
title: >-
  [Paper Note] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation
description: >-
  [ICCV 2025][Model Compression][Knowledge Distillation] This paper proposes PAT (Perspective-Aware Teaching), a framework that addresses the view mismatch problem across heterogeneous architectures via Region-Aware Attention (RAA) and the teacher unawareness problem via Adaptive Feedback Prompting (AFP), enabling feature-level distillation to comprehensively surpass logit-level methods in heterogeneous knowledge distillation for the first time.
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Heterogeneous Distillation"
  - "Perspective Alignment"
  - "Adaptive Teacher"
  - "Feature Distillation"
date: 2026-05-08
content_hash: c5bdee0fa588db57
---

# Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation

**Conference**: ICCV 2025
**arXiv**: [2501.08885](https://arxiv.org/abs/2501.08885)  
**Code**: [https://github.com/jimmylin0979/PAT.git](https://github.com/jimmylin0979/PAT.git)  
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Heterogeneous Distillation, Perspective Alignment, Adaptive Teacher, Feature Distillation

## TL;DR

This paper proposes PAT (Perspective-Aware Teaching), a framework that addresses the view mismatch problem across heterogeneous architectures via Region-Aware Attention (RAA) and the teacher unawareness problem via Adaptive Feedback Prompting (AFP), enabling feature-level distillation to comprehensively surpass logit-level methods in heterogeneous knowledge distillation for the first time.

## Background & Motivation

Knowledge distillation (KD) aims to transfer knowledge from a large pre-trained teacher model to a lightweight student model. Existing KD methods predominantly assume homogeneous teacher–student architectures (e.g., CNN→CNN), yet the growing diversity of architectures in practice (CNN, ViT, MLP-Mixer, etc.) has driven increasing demand for cross-architecture distillation.

Cross-architecture distillation faces two core challenges:

**View Mismatch**: Different architectures exhibit distinct receptive fields and inductive biases (ViT: global→local; CNN: local→global), causing features at the same stage to capture fundamentally different information.

**Teacher Unawareness**: The teacher model is frozen after independent pre-training and remains unaware of the student's learning progress; its intermediate features are not necessarily well-suited for distillation.

The prior SOTA method OFA-KD supports cross-architecture distillation but projects student features into logit space, discarding spatial information and limiting performance on downstream tasks such as object detection.

## Method

### Overall Architecture

The PAT framework consists of two core modules: RAA addresses view mismatch, and AFP enables the teacher to adapt based on student feedback. Both teacher and student are divided into 4 stages, with stage-wise feature alignment performed throughout.

### Key Designs

1. **Region-Aware Attention (RAA)**: Resolves view mismatch. Student features from each stage are patchified into $\frac{N_q}{4}$ patches, projected to a unified dimension $\mathbb{R}^{\frac{N_q}{4}\times d}$ via convolutional projection, and concatenated before being fed into a self-attention module:
    $F^{S'} = \text{Softmax}\left(\frac{(W_qF^S)(W_kF^S)^T}{\sqrt{d}}\right)W_vF^S$
   The attention mechanism enables the student to integrate features across different spatial regions and stages, producing representations more aligned with the teacher's perspective. The design motivation lies in the flexibility of attention to seamlessly accommodate the distinct spatial observation patterns of arbitrary architectures.

2. **Adaptive Feedback Prompting (AFP)**: Resolves teacher unawareness. An AFP module is inserted before each teacher stage, comprising a Fusion Block and a Prompt Block. The difference between student and teacher features from the previous iteration serves as feedback:
    $\text{Feedback}_i = M_i^{AFP}(F_{prev,i}^S) - F_{prev,i}^T$
   The feedback is concatenated with teacher features and processed through the Fusion and Prompt Blocks to produce distillation-friendly teacher features. Features from the previous iteration (rather than the current one) are used to prevent the model from trivially copying spatially aligned feedback.

3. **Regularization Loss**: A KL divergence constraint $L_{Reg} = L_{KL}(p^T, p^{T'})$ is introduced to prevent AFP from over-modifying teacher features and degenerating into an identity mapping of the student features.

### Loss & Training

The overall loss function is:
$$L_{PAT} = L_{CE} + \alpha L_{KL} + \beta L_{FD} + \gamma L_{Reg}$$
- $L_{CE}$: Standard cross-entropy loss
- $L_{KL}$: Logit-level KL divergence distillation loss
- $L_{FD}$: Feature matching loss using ReviewKD's Hierarchical Context Loss (HCL)
- $L_{Reg}$: AFP regularization loss

## Key Experimental Results

### Main Results (CIFAR-100 Heterogeneous Distillation)

| Teacher → Student | From Scratch | OFA (SOTA) | FitNet | PAT |
|-------------------|-------------|------------|--------|-----|
| ConvNeXt-T → DeiT-T | 68.00 | 75.76 | 60.78 | **79.59** |
| ConvNeXt-T → ResMLP-S12 | 66.56 | 81.22 | 45.47 | **83.50** |
| Swin-T → ResNet18 | 74.01 | 80.54 | 78.87 | **81.22** |
| ViT-S → MobileNetV2 | 73.68 | 78.45 | 73.54 | **78.87** |
| **Average Gain** | - | +7.47 | -5.20 | **+8.17** |

**ImageNet-1K**:

| Teacher → Student | KD | OFA | PAT |
|-------------------|-----|-----|-----|
| Swin-T → ResNet18 | 71.14 | 71.85 | **71.54** |
| ConvNeXt-T → DeiT-T | 74.00 | 74.41 | **74.44** |
| Swin-T → ResMLP-S12 | 76.67 | 77.31 | **77.59** |

**COCO Detection (Faster RCNN)**:

| Teacher → Student | KD | OFA | FitNet | PAT |
|-------------------|-----|-----|--------|-----|
| Swin-T → ResNet18 (mAP) | 34.07 | 33.37 | 35.23 | **35.62** |
| Swin-T → MobileNetV2 (mAP) | 31.46 | 31.69 | 32.48 | **32.97** |

### Ablation Study (Module Effectiveness, ConvNeXt-T → DeiT-T)

| Configuration | CIFAR-100 Acc |
|------|-------------|
| Baseline (FitNet) | 60.71 |
| + RAA | 70.12 |
| + RAA + AFP (w/o feedback) | 79.13 |
| + RAA + AFP (w/ feedback) | **79.59** |

The RAA module contributes the largest gain (+9.41%), with AFP providing further improvement and student feedback yielding the best overall performance.

### Key Findings

- **Feature distillation surpasses logit-level methods in heterogeneous KD for the first time**: Prior feature-based methods such as FitNet performed poorly in heterogeneous settings (e.g., only 24.06% on ConvNeXt-T→Swin-P); PAT resolves this longstanding limitation.
- PAT achieves a maximum improvement of 16.94% on CIFAR-100 (ConvNeXt-T→DeiT-T), outperforming FitNet by 18.81%.
- On object detection, PAT's feature-level distillation substantially outperforms logit-based methods such as OFA, owing to the importance of spatial information in detection tasks.
- RAA attention maps reveal markedly different patterns across architecture pairs: MLP students learning from CNN teachers exhibit diagonal patterns (local aggregation), while CNN students learning from ViT teachers exhibit grid patterns (global context aggregation).
- Removing the KL loss still yields near-SOTA performance, confirming that the primary gain stems from feature-level alignment rather than logit matching.
- Applying AFP across all 4 stages achieves the best results, with earlier stages contributing more substantially (stage 1: 75.43%).
- Additional parameters are required only during training and do not affect student inference efficiency.

## Highlights & Insights

- **Precise problem formulation**: The challenges of heterogeneous KD are decomposed into two orthogonal problems—view mismatch and teacher unawareness—yielding a principled and coherent solution design.
- **Strong generality**: RAA is architecture-agnostic and requires no customization for specific teacher–student pairs.
- **Convincing attention visualizations**: These clearly demonstrate how different architecture pairs learn distinct feature mixing strategies within RAA.
- **Validation on detection tasks**: Results confirm that spatial-information-preserving feature distillation provides greater value than logit-level methods for downstream tasks.

## Limitations & Future Work

- The additional parameter count is substantial (14.48M), and training time is approximately three times that of standard KD (208s vs. 66s per epoch).
- Performance gains for CNN students are less pronounced than for ViT/MLP students, potentially requiring longer training schedules.
- The choice of $N_q$ involves a trade-off between accuracy and memory consumption (performance improves from 36→144 but memory usage also increases).
- Validation on larger-scale models (e.g., ViT-L, Swin-L) has not been conducted.
- AFP may cause teacher features to degenerate into an identity mapping of student features for certain model pairs, necessitating regularization constraints.

## Related Work & Insights

- OFA-KD pioneered universal heterogeneous KD but at the cost of spatial information.
- ReviewKD's Hierarchical Context Loss is adopted by PAT as the feature distance function.
- Prompt tuning techniques are transferred from NLP to teacher adaptation in KD, representing a novel cross-domain perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The RAA and AFP module designs are insightful and address the core bottlenecks of heterogeneous KD.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 12 heterogeneous pairs, 3 datasets, detection tasks, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and figures are informative.
- **Value**: ⭐⭐⭐⭐ Makes feature-level distillation genuinely viable in heterogeneous settings, with significant methodological contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
