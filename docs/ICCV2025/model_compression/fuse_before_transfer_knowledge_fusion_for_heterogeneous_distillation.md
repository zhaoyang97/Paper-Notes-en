---
title: >-
  [Paper Note] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation
description: >-
  [ICCV 2025][Model Compression][Cross-architecture knowledge distillation] This paper proposes FBT (Fuse Before Transfer), which mitigates the feature gap in cross-architecture knowledge distillation (CAKD) by first fusing modules (CNN/MSA/MLP) from heterogeneous teachers and students to construct an adaptive intermediate fusion model before knowledge transfer, and replaces the conventional MSE loss with a spatial-agnostic InfoNCE loss. FBT achieves an average improvement of 8…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Cross-architecture knowledge distillation"
  - "heterogeneous model fusion"
  - "CNN-ViT-MLP"
  - "InfoNCE loss"
  - "feature alignment"
date: 2026-05-08
content_hash: 49cb50af703d8e29
---

# Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation

**Conference**: ICCV 2025
**arXiv**: [2410.12342](https://arxiv.org/abs/2410.12342)  
**Code**: [https://github.com/liguopeng0923/FBT](https://github.com/liguopeng0923/FBT)  
**Area**: Model Compression / Knowledge Distillation
**Keywords**: Cross-architecture knowledge distillation, heterogeneous model fusion, CNN-ViT-MLP, InfoNCE loss, feature alignment

## TL;DR

This paper proposes FBT (Fuse Before Transfer), which mitigates the feature gap in cross-architecture knowledge distillation (CAKD) by first fusing modules (CNN/MSA/MLP) from heterogeneous teachers and students to construct an adaptive intermediate fusion model before knowledge transfer, and replaces the conventional MSE loss with a spatial-agnostic InfoNCE loss. FBT achieves an average improvement of 8.38% on CIFAR-100 and 2.31% on ImageNet-1K.

## Background & Motivation

Most knowledge distillation (KD) methods focus on homogeneous teacher–student pairs (e.g., CNN→CNN), which constrains the **potential** and **flexibility** of distillation:
- **Limited potential**: The pool of homogeneous teachers is narrow and may not provide optimal knowledge. OFA has demonstrated that heterogeneous ViT-Base distillation to ResNet50 outperforms homogeneous ResNet152 distillation.
- **Limited flexibility**: As new architectures continuously emerge, domain-specific tasks may lack suitable homogeneous teachers.

The central challenge of CAKD is the **large representational gap between heterogeneous models**, arising from:

**Inductive bias discrepancy**: CNNs exhibit locality and translation equivariance, whereas ViTs/MLPs rely on global dependencies.

**Module functional discrepancy**: Different modules read, encode, and process inputs differently, leading to significant distribution shifts in intermediate features at each stage.

Limitations of existing methods:
- Feature-based methods employ simple projectors that cannot bridge the heterogeneous feature gap.
- Pixel-wise MSE loss is ill-suited for heterogeneous features with large spatial distribution differences (e.g., FitNet achieves only 24.06% on ConvNeXt-T→Swin-P).
- OFA projects features into logit space, sacrificing structural feature information.

## Method

### Overall Architecture

FBT adopts a three-level distillation scheme (Teacher–Fusion–Student), with the core principle of **fuse before transfer**:
1. An adaptive fusion model is constructed by concatenating modules from the teacher and student.
2. Three sets of losses are applied simultaneously: $\mathcal{L}_{\text{FBT}} = \mathcal{L}(K_t, K_s) + \mathcal{L}(K_t, K_f) + \mathcal{L}(K_f, K_s)$
3. The fusion model acts as a bridge between the teacher and the student.

### Key Designs

1. **Adaptive Knowledge Fusion**:

    - The fusion model comprises the first three CNN stages of the student, an L2G projector, and the final MSA/MLP stage of the teacher.
    - Formulation: $p_f(x) = fc_m \circ S_m^4 \circ (MSA \circ PE) \circ S_c^3 \circ S_c^2 \circ S_c^1(x)$
    - The **L2G module** consists of a Patch Embedding layer (for dimension transformation) and a Swin Block (for local-to-global feature transition).
    - **Design Motivation**: CNNs and MSA are complementary (the former excels at local features, the latter at global dependencies); weight sharing minimizes additional parameters.
    - The fusion model is adaptive: different teacher–student pairs yield different fusion architectures.

2. **Spatial-Agnostic Knowledge Supervision**:

    - Only the final features after Average Pooling and the logits are transferred, since weight sharing truly integrates different inductive biases only at the final feature level.
    - Average Pooling smooths spatial discrepancies, and InfoNCE loss then aligns structural feature information.
    - Knowledge is defined as $K_i = \{f_i, p_i\}$, where $f_i$ is the pooled feature embedding and $p_i$ is the output logits.

3. **L2G (Local-to-Global) Projector**:

    - Serves as the bridge connecting CNN and MSA/MLP modules.
    - Includes a Patch Embedding layer that transforms CNN features into the dimensions required by MSA/MLP.
    - Appends a Swin Block to achieve local-to-global receptive field transition.
    - Introduces minimal additional learnable parameters.

### Loss & Training

The overall loss consists of two components applied to each knowledge pair $(K_i, K_j)$:
- **OFA loss** (logits): A modulated KL divergence variant that upweights target-class information via a modulation parameter $\gamma$ when the teacher is uncertain.
- **InfoNCE loss** (features): A spatial-agnostic contrastive loss in which teacher–student features from the same image form positive pairs, capturing complex inter-feature dependencies without relying on spatial positions.
- The temperature parameter $\tau_2$ is learnable.

Total loss: $\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(f_i, f_j) + \mathcal{L}_{\text{OFA}}(p_i, p_j)$, applied to all three pairs: T-S, T-F, and F-S.

## Key Experimental Results

### Main Results (Tables)

**CIFAR-100 Cross-Architecture Distillation Results (Top-1 Accuracy %)**

| Teacher | Student | KD | FitNet | CRD | OFA | **FBT** |
|---------|---------|-----|--------|-----|-----|---------|
| Swin-T | ResNet18 | 78.74 | 78.87 | 77.63 | 80.54 | **81.61** |
| ViT-S | ResNet18 | 77.26 | 77.71 | 76.60 | 80.15 | **81.93** |
| ViT-S | MobileNetV2 | 72.77 | 73.54 | 78.14 | 78.45 | **82.10** |
| ConvNeXt-T | DeiT-T | 72.99 | 60.78 | 65.94 | 75.76 | **79.57** |
| ConvNeXt-T | ResMLP-S12 | 72.25 | 45.47 | 63.35 | 75.21 | **78.03** |
| Avg. Gain | | +3.12 | -5.21 | -0.02 | +6.19 | **+8.38** |

**ImageNet-1K Cross-Architecture Distillation Results (Top-1 Accuracy %)**

| Teacher | Student | OFA | **FBT** |
|---------|---------|-----|---------|
| Swin-T | ResNet18 | 71.76 | **72.21** |
| Swin-T | MobileNetV2 | 72.32 | **72.54** |
| ConvNeXt-T | DeiT-T | 74.41 | **75.26** |
| ResNet50 | Swin-N | 77.76 | **77.79** |
| Avg. Gain | | +2.05 | **+2.31** |

### Ablation Study (Table)

| Ablation Setting | Swin-T→ResNet18 | ConvNeXt-T→Swin-P | Swin-T→ResNet18 (IN-1K) |
|-----------------|------------------|---------------------|--------------------------|
| KD Baseline | 78.74 (-2.87) | 76.44 (-4.29) | 71.14 (-1.04) |
| (A) w/o MSA and $S_m^4$ | 75.95 (-5.66) | 77.65 (-3.18) | 70.86 (-1.35) |
| (B) w/o $S_m^4$ | 77.21 (-4.40) | 77.84 (-2.89) | 71.78 (-0.43) |
| (C) w/o $\mathcal{L}(K_t,K_f)$ | 25.57 (-56.04) | 50.46 (-30.27) | 71.34 (-0.87) |
| (F) w/o InfoNCE | 80.95 (-0.66) | 78.89 (-1.84) | 71.47 (-0.74) |
| (G) w/o OFA | 77.91 (-3.70) | 80.32 (-0.41) | 70.37 (-1.84) |
| **Full FBT** | **81.61** | **80.73** | **72.21** |

### Key Findings

- The fusion model's performance consistently lies between that of the teacher and the student, validating its role as a knowledge bridge.
- Removing the teacher-to-fusion loss $\mathcal{L}(K_t,K_f)$ causes catastrophic degradation (from 81.61% to 25.57% on CIFAR-100), confirming that the fusion model must learn from the teacher.
- InfoNCE and OFA losses exhibit varying importance across different teacher–student pairs; their complementary use yields the best results.
- FBT also achieves competitive performance in same-architecture knowledge distillation (SAKD), marginally surpassing FCFD and OFA on ImageNet-1K.
- The fusion strategy $S_c^{1 \to 3} \to S_m^{4 \to fc}$ (3 CNN stages + 1 MSA/MLP stage) achieves the best balance between simplicity and adaptability.

## Highlights & Insights

- **Fusion over alignment**: Rather than attempting to bridge the heterogeneous feature gap with projectors, FBT directly constructs a fusion model incorporating modules from both architectures, fundamentally reducing the feature gap.
- **Adaptive design**: The fusion model automatically adapts to each teacher–student pair without manual architecture engineering.
- **Elegant weight sharing**: By reusing module weights from the student and teacher, the fusion model introduces almost no extra parameters while implicitly aligning module functionality.
- **Importance of spatial-agnostic loss**: MSE fails in heterogeneous settings (FitNet achieves 24.06%), whereas InfoNCE circumvents the spatial alignment problem through contrastive learning.

## Limitations & Future Work

- For certain well-established student models (e.g., ResNet18), heterogeneous teacher distillation may underperform homogeneous counterparts.
- Fusion may disrupt spatial alignment between heterogeneous features; distribution-level (rather than pixel-level) spatial alignment could be explored.
- Generalization to broader downstream tasks such as object detection and NLP remains unvalidated.
- The current fixed 3+1 fusion ratio could potentially be improved through automatic architecture search.

## Related Work & Insights

- **OFA** (NeurIPS 2023): The first general-purpose heterogeneous distillation method, but it sacrifices feature information by projecting features into logit space.
- **FCFD** (ICLR 2023): Aligns functional similarity via module connectivity, directly inspiring FBT's module fusion design.
- **Hybrid model designs** (CoAtNet, ConvMLP, etc.): The complementarity of CNN and MSA directly motivates the fusion strategy.
- **CRD**: A pioneer in applying InfoNCE loss to knowledge distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "fuse before transfer" paradigm is elegant and effective; drawing inspiration from hybrid model design to address distillation is a clever insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 12 heterogeneous combinations on CIFAR-100 and 14 on ImageNet-1K, with comprehensive ablations over fusion strategies, loss functions, and module configurations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear motivation analysis; the taxonomy diagram (Fig. 2) is particularly informative.
- **Value**: ⭐⭐⭐⭐ — Provides a general and effective framework for cross-architecture distillation with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICML 2026\] EVL-ECG: Efficient ECG Interpretation With Multi-Aspect Heterogeneous Knowledge Distillation](../../ICML2026/model_compression/evl-ecg_efficient_ecg_interpretation_with_multi-aspect_heterogeneous_knowledge_d.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
