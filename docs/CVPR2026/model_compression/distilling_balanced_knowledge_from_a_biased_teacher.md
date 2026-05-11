---
title: >-
  [Paper Note] Distilling Balanced Knowledge from a Biased Teacher
description: >-
  [CVPR 2026][Model Compression][Knowledge Distillation] To address the head-class bias of teacher models in knowledge distillation under long-tailed distributions…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Long-Tail Distribution"
  - "KL Divergence Decomposition"
  - "Class Imbalance"
date: 2026-05-08
content_hash: 58d6f4958889e2fc
---

# Distilling Balanced Knowledge from a Biased Teacher

**Conference**: CVPR 2026
**arXiv**: [2506.18496](https://arxiv.org/abs/2506.18496)
**Code**: N/A
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Long-Tail Distribution, Model Compression, KL Divergence Decomposition, Class Imbalance

## TL;DR

To address the head-class bias of teacher models in knowledge distillation under long-tailed distributions, this paper decomposes the conventional KL divergence loss into a cross-group component and a within-group component. By rebalancing the cross-group loss to calibrate the teacher's group-level predictions and reweighting the within-group loss to ensure equal contribution across groups, the proposed method consistently outperforms existing approaches on CIFAR-100-LT, TinyImageNet-LT, and ImageNet-LT — and even surpasses the teacher model itself.

## Background & Motivation

Knowledge distillation (KD) is a standard technique for transferring knowledge from a large teacher model to a lightweight student model. Conventional KD methods **implicitly assume that training data are class-balanced**.

In practice, however, real-world data typically follow a **long-tailed distribution**: head classes are abundant while tail classes are scarce. Teacher models trained under such distributions exhibit severe head-class bias. Directly applying standard KD to have the student mimic a biased teacher is not only ineffective but potentially harmful — the student inherits the bias and performs even worse on tail classes.

The key question is: **Can balanced knowledge be distilled from a biased teacher?**

- **Key Insight**: The KL divergence loss is mathematically decomposed into cross-group and within-group components. Each component is shown to be affected differently by teacher bias — the cross-group term inflates head-class probabilities, while the weighting mechanism of the within-group term allows head groups to dominate the gradients.
- **Core Idea**: **Rather than modifying the teacher model, the bias introduced by the teacher is corrected directly within the distillation objective.**

## Method

### Overall Architecture

LTKD partitions classes into three groups: Head (33%), Medium (34%), and Tail (33%). The standard KL divergence KD loss is decomposed into a cross-group loss and a within-group loss, each subject to rebalancing and reweighting corrections, respectively. The final loss is the sum of the rebalanced cross-group KL and equally weighted within-group KL terms.

### Key Designs

1. **Cross-Group and Within-Group Decomposition of KL Divergence**

   - **Function**: Reveals the failure mechanism of standard KD under long-tailed settings.
   - **Mechanism**: Group-level probability $p_\mathcal{G} = \sum_{i \in \mathcal{G}} p_i$ and within-group probability $\tilde{p}_{\mathcal{G}_i} = p_i / p_\mathcal{G}$ are defined. Using the identity $p_i = p_\mathcal{G} \cdot \tilde{p}_{\mathcal{G}_i}$, the KL divergence is exactly decomposed into a cross-group KL term plus a sum of within-group KL terms weighted by the teacher's cross-group probabilities.
   - **Design Motivation**: This is a mathematical identity that introduces no approximation error, yet separates two distinct pathways through which bias manifests.

2. **Rebalanced Cross-Group Loss**

   - **Function**: Calibrates the teacher's skewed group-level probability distribution.
   - **Mechanism**: Within each batch, the teacher's group-level probability sums are aggregated, and scaling factors are computed to align all three groups toward a uniform distribution. Per-sample probabilities are scaled and renormalized to maintain valid probability distributions.
   - **Design Motivation**: Empirical observation shows that a biased teacher outputs approximately uniform predictions $[22.54, 20.76, 20.70]$ on balanced data, but skewed predictions $[27.88, 19.28, 16.83]$ on long-tailed data.

3. **Reweighted Within-Group Loss**

   - **Function**: Eliminates imbalanced weighting of within-group KL divergence terms.
   - **Mechanism**: Unequal weights (i.e., the teacher's cross-group probabilities) are replaced by a uniform constant, ensuring each group contributes equally to the total loss.
   - **Design Motivation**: Prevents head groups from dominating gradient flow, enabling tail groups to receive sufficient supervision signals.

### Loss & Training

- **Total Loss**: Cross-entropy + temperature-scaled LTKD loss (with hyperparameters $\alpha$ and $\beta$ balancing the cross-group and within-group terms).
- **Class Partitioning**: Classes are sorted by sample count; the top 33% form the Head group, the next 34% the Medium group, and the bottom 33% the Tail group.
- **Imbalance Factors**: $\{10, 20, 100\}$ for CIFAR-100-LT and TinyImageNet-LT; $\{5, 10, 20\}$ for ImageNet-LT.
- **Test sets remain class-balanced.**

## Key Experimental Results

### Main Results: CIFAR-100-LT ($\gamma=100$, Most Extreme Imbalance)

| Teacher → Student | Method | Tail Accuracy (%) | Overall Accuracy (%) |
|---|---|---|---|
| ResNet32x4 → ResNet8x4 | DKD | 13.25 | 46.11 |
| | ReviewKD | 15.09 | 45.91 |
| | **LTKD** | **27.21** | **51.08** |
| | Gain | **+12.12** | **+4.97** |
| ResNet50 → MobileNetV2 | DKD | 12.45 | 39.21 |
| | **LTKD** | **21.04** | **42.45** |
| | Gain | **+8.59** | **+3.24** |

### Ablation Study

| Configuration | Tail (%) | All (%) | Note |
|---|---|---|---|
| Standard KD | 13.38 | 42.48 | Inherits teacher bias |
| Cross-group rebalancing only | ~20 | ~48 | Effective for group-level calibration |
| Within-group reweighting only | ~18 | ~47 | Effective for gradient balancing |
| LTKD (both components) | 27.21 | 51.08 | Significant synergistic effect |

### Key Findings

- **LTKD surpasses the teacher itself in nearly all settings**: at $\gamma=100$, the teacher achieves only 15.28% Tail accuracy, while the student reaches 27.21%.
- The method remains effective across heterogeneous architecture pairs (WRN-40-2 → ShuffleNetV1, ResNet50 → MobileNetV2).
- The advantage grows with greater imbalance: Tail accuracy improves by +12.12% at $\gamma=100$ and +6.58% at $\gamma=10$.
- DKD's target/non-target decomposition yields only marginal improvements under long-tailed settings.

## Highlights & Insights

- **Math-decomposition-driven design**: The method first uses exact mathematical identities to expose the root cause of failure, then applies targeted corrections accordingly.
- **Counter-intuitive "student outperforms teacher" results**: The teacher's dark knowledge contains useful information that is obscured by bias.
- **Minimal design with substantial gains**: Only the loss function is modified — no architectural changes, no additional modules, and no data augmentation strategies.

## Limitations & Future Work

- The three-group partition is fixed (33% each); adaptive grouping may yield further improvements.
- Validation is limited to CNN architectures; ViT and larger-scale models remain untested.
- The rebalancing factors are estimated from batch statistics, which may be unstable under small batch sizes.
- No combination experiments with long-tail debiasing strategies such as logit adjustment are reported.

## Related Work & Insights

- DKD's target/non-target decomposition served as one source of inspiration, though the decomposition dimension differs.
- Logit adjustment performs calibration at inference time, whereas LTKD calibrates the teacher distribution during training; the two approaches may be complementary.
- The grouping-and-reweighting paradigm is generalizable to any scenario in which the teacher exhibits systematic bias.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The KL decomposition perspective is novel; the loss correction strategy, though simple, is mathematically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 3 datasets × 3 imbalance factors × 4 architecture pairs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain is complete and the mathematical derivations are clear.
- **Value**: ⭐⭐⭐⭐ Addresses a key pain point of KD in realistic imbalanced settings; the method is simple and directly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](../../ACL2026/model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)

</div>

<!-- RELATED:END -->
