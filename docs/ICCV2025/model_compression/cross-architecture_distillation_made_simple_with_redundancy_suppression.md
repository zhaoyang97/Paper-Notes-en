---
title: >-
  [Paper Note] Cross-Architecture Distillation Made Simple with Redundancy Suppression
description: >-
  [ICCV 2025][Model Compression][knowledge distillation] This paper proposes RSD (Redundancy Suppression Distillation), which extracts architecture-agnostic knowledge via cross-architecture invariance maximization and feature decorrelation. Using a single simple RSD loss and a lightweight MLP decoupling module, RSD substantially outperforms OFA—the pioneering cross-architecture distillation method—on both CIFAR-100 and ImageNet-1k, while incurring only a fraction of OFA's param…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "knowledge distillation"
  - "cross-architecture"
  - "redundancy suppression"
  - "feature decorrelation"
  - "CNN-ViT-MLP"
date: 2026-05-08
content_hash: 198d39e0f96d6d54
---

# Cross-Architecture Distillation Made Simple with Redundancy Suppression

**Conference**: ICCV 2025
**arXiv**: [2507.21844](https://arxiv.org/abs/2507.21844)  
**Code**: N/A  
**Area**: Model Compression
**Keywords**: knowledge distillation, cross-architecture, redundancy suppression, feature decorrelation, CNN-ViT-MLP

## TL;DR
This paper proposes RSD (Redundancy Suppression Distillation), which extracts architecture-agnostic knowledge via cross-architecture invariance maximization and feature decorrelation. Using a single simple RSD loss and a lightweight MLP decoupling module, RSD substantially outperforms OFA—the pioneering cross-architecture distillation method—on both CIFAR-100 and ImageNet-1k, while incurring only a fraction of OFA's parameter overhead.

## Background & Motivation
Knowledge distillation (KD) aims to transfer the capabilities of a pretrained teacher model to a lightweight student model. Conventional KD predominantly operates within homogeneous architectures (e.g., CNN→CNN); however, with the emergence of ViTs, MLP-Mixers, and other novel architectures, cross-architecture knowledge distillation (CAKD) has become increasingly important, as the best-performing models are often unsuitable for direct deployment. → **Core challenge**: Heterogeneous features differ in dimensionality and exhibit distinct or even conflicting representational patterns; forcing a student to blindly absorb teacher features leads to performance degradation. → The pioneering method OFA requires architecture-specific projection modules (e.g., depthwise separable convolutions for CNNs, attention blocks for ViTs) to project features into an "architecture-agnostic" logit space, resulting in complex designs and substantial parameter overhead (the projector parameters are 3× the student's when distilling ConvNeXt-T→Swin-N). → **Core insight of this paper**: Complex projections are unnecessary; extracting common knowledge across heterogeneous representations can be achieved simply through redundancy suppression.

## Method

### Overall Architecture
RSD operates on the penultimate-layer embeddings of both teacher and student. After aligning dimensions via a lightweight AAD decoupling module, it computes a cross-architecture Pearson correlation matrix and applies the RSD loss. The total loss is $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{RSD}$. The AAD module is discarded after training, introducing no additional inference overhead.

### Key Designs
1. **Redundancy Suppression Distillation (RSD) Loss**:

    - **Function**: Extracts architecture-agnostic common knowledge from teacher and student representations.
    - **Mechanism**: Constructs a Pearson correlation matrix $\mathbf{P} \in \mathbb{R}^{D \times D}$ between teacher features $\mathbf{z}^t$ and student features $\mathbf{z}^s$, with the optimization target being the identity matrix $\mathbf{T} = I$. (1) Diagonal elements → 1: maximizes cross-architecture invariance along matched dimensions (extracting common knowledge); (2) Off-diagonal elements → 0: decorrelates mutual information across different feature dimensions (suppressing redundant architecture-specific information). The loss is $\mathcal{L}_{RSD} = d(\mathbf{P}(h(\mathbf{z}^s), \mathbf{z}^t), \mathbf{T})$, using MSE distance. The off-diagonal loss can be weighted by a coefficient κ.
    - **Design Motivation**: Inspired by classical unsupervised feature learning theory (Barlow Twins information maximization and feature decorrelation), minimizing mutual information is equivalent to extracting statistically independent, architecture-agnostic features.

2. **Architecture-Agnostic Decoupling (AAD) Module**:

    - **Function**: Acts as a buffer to prevent student internal representations from being completely overridden by the RSD objective, preserving architecture-specific beneficial capabilities of the student.
    - **Mechanism**: A two-layer FC network (expander + adaptor) with BatchNorm and GeLU activation in between. The expander maps student embeddings to a higher-dimensional space; the adaptor aligns them to the teacher embedding dimension.
    - **Design Motivation**: Different architectures possess unique strengths (e.g., CNNs' sensitivity to local textures, which ViTs lack); fully overwriting these with architecture-agnostic knowledge would discard such capabilities. AAD serves as a buffer layer, directing RSD optimization onto the projected representations rather than directly modifying student internal representations.

3. **Design Rationale for Using Penultimate-Layer Embeddings**:

    - **Function**: Avoids complex dimension alignment issues associated with intermediate features.
    - **Mechanism**: Penultimate-layer embeddings are always 1D vectors (not feature maps or token sequences), requiring no architecture-specific operations (depthwise separable convolutions, token operations, etc.). Being closer to the network output than intermediate features, they exhibit weaker architecture-specificity and are more amenable to extracting architecture-agnostic information.
    - **Design Motivation**: This is precisely the root of OFA's complexity—it necessitates designing distinct projection modules for intermediate features of different architectures.

### Loss & Training
The RSD loss can be implemented in approximately 8 lines of PyTorch code: normalize features → compute cross-correlation matrix → diagonal MSE + weighted off-diagonal MSE. The training configuration follows OFA. RSD can also be applied in the logit space as a logit distiller, where it likewise performs strongly.

## Key Experimental Results

### Main Results
CIFAR-100 (12 heterogeneous teacher–student pairs; partial results shown):

| Teacher→Student | From Scratch | KD | OFA | RSD | RSD vs OFA |
|---|---|---|---|---|---|
| Swin-T→ResNet18 | 74.01 | 78.74 | 80.54 | **83.92** | +3.38 |
| ViT-S→MobileNetV2 | 73.68 | 72.77 | 78.45 | **81.68** | +3.23 |
| ConvNeXt-T→DeiT-T | 68.00 | 72.99 | 75.76 | **82.46** | +6.70 |
| ConvNeXt-T→ResMLP-S12 | 66.56 | 72.25 | 81.22 | **84.21** | +2.99 |
| **Average Gain** | - | +3.17 | +7.47 | **+10.69** | **+3.22** |

ImageNet-1k (15 heterogeneous teacher–student pairs; partial results shown):

| Teacher→Student | From Scratch | OFA | RSD | RSD vs OFA |
|---|---|---|---|---|
| Swin-T→ResNet18 | 69.75 | 71.85 | **72.13** | +0.28 |
| ConvNeXt-T→Swin-N | 75.53 | 77.50 | **77.70** | +0.20 |
| ConvNeXt-T→ResMLP-S12 | 76.65 | 77.53 | **78.41** | +0.88 |
| **Average Gain** | - | +2.20 | **+2.34** | **+0.14** |

### Ablation Study

| Configuration | Swin-T→ResNet18 | ConvNeXt-T→ResMLP-S12 | Note |
|---|---|---|---|
| Baseline (scratch) | 74.01 | 76.65 | No distillation |
| + RSD-corr only | 80.65 | 83.40 | Invariance maximization only |
| + RSD-decorr | **83.92** | **84.21** | Full RSD with decorrelation |

Effect of AAD (CIFAR-100 / ImageNet):

| Configuration | ViT-S→ResMLP | ConvNeXt-T→Mixer |
|---|---|---|
| Full RSD | 82.94 | 80.73 |
| w/o AAD | 82.26 (−0.68) | 79.93 (−0.80) |

Parameter overhead comparison (ConvNeXt-T→Swin-N @ ImageNet):

| Method | Student Params | Extra Params | Extra/Student Ratio |
|---|---|---|---|
| OFA | 9.6M | 28.2M | 2.94× |
| **RSD** | 9.6M | ~2.8M | **0.29×** |

RSD as a logit distiller:

| Logit Loss | Swin-T→ResNet18 | ConvNeXt-T→ResMLP |
|---|---|---|
| KD | 78.74 | 72.25 |
| DKD | 80.26 | 73.22 |
| OFA (logit component only) | 80.60 | 78.87 |
| **RSD on logits** | **83.23** | **81.15** |

### Key Findings
- RSD achieves an average gain of +10.69% on CIFAR-100, substantially surpassing OFA's +7.47%.
- On ConvNeXt-T→DeiT-T, RSD outperforms OFA by 6.70%—nearly equal to the gap between OFA and no distillation.
- RSD alone as a logit distiller surpasses OFA's full framework (including all complex projectors).
- The decorrelation objective further improves performance in most settings.
- RSD is complementary to OFA: replacing all losses in OFA with RSD losses yields additional gains.
- CKA visualization confirms that RSD substantially increases feature similarity between heterogeneous architectures at intermediate and deep layers.

## Highlights & Insights
- A paradigmatic example of "simple yet effective": an 8-line RSD loss outperforms the considerably more complex OFA, with only 1/10 of its parameter overhead.
- The redundancy suppression perspective offers a precise reformulation of the CAKD problem: rather than learning how to align heterogeneous features, the goal is to eliminate architecture-specific redundant information.
- Choosing penultimate-layer embeddings over intermediate features is a clever design decision that entirely circumvents the heterogeneous feature alignment challenge.
- The AAD module's design philosophy of "preserving student-specific capabilities" reflects a deep understanding of the essence of knowledge distillation.

## Limitations & Future Work
- The advantage on ImageNet is less pronounced than on CIFAR-100 (+2.34% vs. +10.69%); further gains on large-scale datasets remain to be explored.
- Relying solely on 1D embeddings precludes the exploitation of rich spatial information in 2D feature maps, limiting applicability to spatially sensitive tasks such as object detection.
- The hyperparameters λ and κ exhibit some sensitivity and require careful tuning.
- Cross-architecture distillation for additional downstream tasks (e.g., object detection, semantic segmentation) has not been explored.

## Related Work & Insights
- The information maximization and decorrelation principles from Barlow Twins are cleverly repurposed as cross-architecture distillation objectives.
- A conceptual connection exists with "domain-invariant representation learning" in domain generalization, though the context and methodological essence differ substantially.
- The simplicity and generality of RSD position it as a potential baseline method for the CAKD community.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The redundancy suppression perspective is novel, though the core technique (correlation matrix + decorrelation) is borrowed from the SSL literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12+15 heterogeneous model pairs, validation on both CIFAR-100 and ImageNet, with comprehensive ablation, compatibility, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, naturally motivated derivations, and thorough comparative analysis against OFA.
- **Value**: ⭐⭐⭐⭐⭐ A simple and effective method of great community value; strong potential to become the standard strong baseline for CAKD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [\[ICML 2025\] From Logits to Hierarchies: Hierarchical Clustering made Simple](../../ICML2025/model_compression/from_logits_to_hierarchies_hierarchical_clustering_made_simple.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](../../NeurIPS2025/model_compression/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
