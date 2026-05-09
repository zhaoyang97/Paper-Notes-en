---
title: >-
  [Paper Note] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation
description: >-
  [ICCV 2025][Autonomous Driving][Knowledge distillation] This paper proposes ACAM-KD, an adaptive student-teacher cooperative attention masking framework for knowledge distillation. By employing Student-Teacher Cross-Attention Feature Fusion (STCA-FF) and Adaptive Spatial-Channel Masking (ASCM) to dynamically adjust distillation focus, ACAM-KD surpasses the state of the art by up to 1.4 mAP on COCO detection and improves mIoU by 3.09 on Cityscapes segmentation.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Knowledge distillation
  - attention masking
  - cross-attention fusion
  - spatial-channel selection
  - object detection
date: 2026-05-08
content_hash: 5bb6b99e7b148aef
---

# ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: Knowledge distillation, attention masking, cross-attention fusion, spatial-channel selection, object detection

## TL;DR

This paper proposes ACAM-KD, an adaptive student-teacher cooperative attention masking framework for knowledge distillation. By employing Student-Teacher Cross-Attention Feature Fusion (STCA-FF) and Adaptive Spatial-Channel Masking (ASCM) to dynamically adjust distillation focus, ACAM-KD surpasses the state of the art by up to 1.4 mAP on COCO detection and improves mIoU by 3.09 on Cityscapes segmentation.

## Background & Motivation

**Why is attention selection in existing feature distillation methods problematic?** The core issues can be summarized in three aspects:

**Stasis**: For a given image, the distillation focus region remains fixed throughout training. Even if the student has already learned certain regions well by Epoch 12, it is still forced to attend to the same regions at Epoch 24—an inefficient and potentially harmful constraint.

**Unidirectionality**: Knowledge selection is entirely teacher-driven (FKD, MasKD) or relies on hand-crafted heuristics (ground-truth bounding boxes, RPN regions). Regions the teacher deems important are not necessarily what the student needs most at its current learning stage. For example, Figure 1 shows that the student's attention localization at Epoch 12 can even **surpass the teacher's**, yet by Epoch 24 degrades into imitating the teacher's suboptimal attention.

**Spatial-channel disconnect**: Most methods perform feature selection solely along the spatial dimension, ignoring differential contributions across channels.

**Why is "cooperation" necessary?** Knowledge distillation should not be a passive reception process but a **dynamic interaction** between both parties—the teacher provides guidance while the student feeds back its own learning state, and the mask adjusts accordingly.

## Method

### Overall Architecture

ACAM-KD consists of two core modules applied sequentially:

1. **STCA-FF** (Student-Teacher Cross-Attention Feature Fusion): Fuses teacher and student features via cross-attention to produce an interaction feature.
2. **ASCM** (Adaptive Spatial-Channel Masking): Generates adaptive spatial and channel masks from the fused features.

Core philosophy: masks are no longer fixed or unilaterally determined by the teacher, but are dynamically generated from **teacher-student interaction features** and evolve in real time with the student's learning state.

### Key Designs

#### Student-Teacher Cross-Attention Feature Fusion (STCA-FF)

Given teacher and student feature maps $F_T, F_S \in \mathbb{R}^{C \times H \times W}$:

$$Q = W_q F_T, \quad K = W_k F_S, \quad V = W_v F_S$$

where $W_q \in \mathbb{R}^{C_q \times C}$ and $W_k \in \mathbb{R}^{C_q \times C}$ reduce the channel dimension to $C_q = C/2$, and $W_v \in \mathbb{R}^{C \times C}$ preserves the original dimension.

**Why does the teacher serve as Query and the student as Key/Value?** As the more knowledgeable model, the teacher uses Query to guide "where to attend"; the student provides Key/Value, representing its current feature state. The resulting attention matrix thus reflects "the correspondence of teacher-attended positions in the student's feature space." Ablation experiments (Table 9) confirm this configuration outperforms the reverse by 0.2 mAP.

Attention matrix computation:

$$A = \text{softmax}\left(\frac{QK}{\sqrt{C_q}}\right) \in \mathbb{R}^{HW \times HW}$$

Fused feature: $F_{fused} = AV \in \mathbb{R}^{C \times H \times W}$

#### Adaptive Spatial-Channel Masking (ASCM)

**Learnable selection units** are introduced on the fused features to dynamically generate two types of masks:

**Channel mask**: $M^c = \sigma(m_c \cdot v)$
- $m_c \in \mathbb{R}^{M \times 1}$: $M$ channel selection units
- $v \in \mathbb{R}^{1 \times C}$: spatially average-pooled vector of $F_{fused}$

**Spatial mask**: $M^s = \sigma(m_s \cdot z)$
- $m_s \in \mathbb{R}^{M \times C}$: $M$ spatial selection units
- $z \in \mathbb{R}^{C \times HW}$: spatially flattened $F_{fused}$

**Why use $M$ masks rather than a single mask?** Multiple masks can capture different feature importance patterns. For detection, $M=6$; for segmentation, $M=19$ (equal to the number of classes, with each mask corresponding to one semantic category).

**Mask diversity loss**: To prevent all masks from converging to similar patterns, a Dice coefficient-based diversity loss is introduced:

$$L_{div} = \frac{2 \sum_{i=1}^M \sum_{j=1, j \neq i}^M M_i \cdot M_j}{\sum_{i=1}^M M_i^2 + \sum_{j=1}^M M_j^2}$$

### Loss & Training

**Channel distillation loss**:

$$L^c_{distill} = \frac{1}{M} \sum_{m=1}^M \frac{1}{HW} \sum_{k=1}^C M^c_{m,k} \| M^c_m \odot (F_T - f_{align}(F_S)) \|_2^2$$

**Spatial distillation loss**:

$$L^s_{distill} = \frac{1}{M} \sum_{m=1}^M \frac{1}{C} \sum_{p=1}^{H \times W} M^s_{m,p} \| M^s_m \odot (F_T - f_{align}(F_S)) \|_2^2$$

**Total loss**:

$$L = L_{task} + \alpha (L^c_{distill} + L^s_{distill}) + \lambda L_{div}$$

where $\alpha = 1$ and $\lambda = 1$. Feature distillation is applied at the FPN neck for detection and on the predicted segmentation maps for segmentation.

Training details:
- Detection: SGD, momentum=0.9, weight decay=0.0001, with an inheritance strategy to stabilize early training.
- Segmentation: SGD, weight decay=5e-4, polynomial annealing learning rate schedule, initial LR=0.02, 40K iterations.

## Key Experimental Results

### Main Results

**COCO Object Detection — ResNet-101 Teacher (Table 1):**

| Method | RetinaNet mAP | Faster R-CNN mAP | RepPoints mAP |
|--------|--------------|-----------------|---------------|
| Student baseline (R50) | 37.4 | 38.4 | 38.6 |
| FGD | 39.6 | 40.4 | 41.0 |
| MasKD | 39.8 | 40.8 | 41.1 |
| FreeKD | 39.9 | 40.8 | - |
| **ACAM-KD (Ours)** | **41.2** (+1.3) | **41.4** (+0.6) | **42.5** (+1.4) |

**COCO Object Detection — ResNeXt-101 Teacher (Table 2):**

| Method | RetinaNet mAP | Faster R-CNN mAP | RepPoints mAP |
|--------|--------------|-----------------|---------------|
| Student baseline (R50) | 37.4 | 38.4 | 38.6 |
| MasKD | 40.9 | 42.4 | 41.8 |
| FreeKD | 41.0 | 42.4 | 42.4 |
| **ACAM-KD (Ours)** | **41.5** | **42.6** | **42.8** |

From R50 student to X101 teacher, ACAM-KD improves student mAP by up to **4.2 points**.

**Cityscapes Semantic Segmentation (Tables 3–5):**

| Student Model | Baseline mIoU | MasKD | FreeKD | **ACAM-KD** |
|--------------|-------------|-------|--------|-------------|
| DeepLabV3-R18 | 72.96 | 77.00 | 76.45 | **77.53** (+0.53) |
| DeepLabV3-MBV2 | 73.12 | 75.26 | - | **76.21** (+0.95) |
| PSPNet-R18 | 72.55 | 75.34 | - | **75.99** (+0.65) |

### Ablation Study

**Spatial vs. Channel Masking (Table 8):**

| Masking Strategy | mAP | AP_s | AP_m | AP_l |
|-----------------|-----|------|------|------|
| Baseline (R50) | 37.4 | 20.0 | 40.7 | 49.7 |
| Spatial only | 40.9 | **25.4** | 44.3 | 52.3 |
| Channel only | 40.4 | 24.5 | 44.1 | 52.3 |
| **Spatial + Channel** | **41.2** | 24.6 | **45.5** | **54.1** |

Spatial masking contributes most to small-object detection (AP_s, +5.4), while combining both further improves medium and large objects.

**Fixed vs. Adaptive Masking (Table 10):**

| Masking Strategy | mAP | AP_s | AP_m | AP_l |
|-----------------|-----|------|------|------|
| No mask | 37.4 | 20.8 | 40.8 | 50.9 |
| Teacher offline fixed mask (MasKD-style) | 39.8 | 21.5 | 43.9 | 54.0 |
| Teacher adaptive mask | 39.9 | 21.7 | 43.7 | 53.9 |
| **ACAM-KD (cooperative adaptive)** | **41.2** | **24.6** | **45.5** | **54.1** |

Key finding: making the mask learnable alone yields only marginal improvement (39.8 → 39.9), whereas introducing student-teacher interaction produces a substantial gain (39.9 → 41.2), demonstrating that the **cooperative mechanism is the fundamental source of performance improvement**.

**Query Source in Cross-Attention (Table 9):**

| Query Source | mAP | AP_s | AP_m | AP_l |
|-------------|-----|------|------|------|
| From student | 41.0 | 24.4 | 45.1 | 54.0 |
| **From teacher** | **41.2** | **24.6** | **45.5** | **54.1** |

### Key Findings

1. **Cooperation > teacher-driven**: Student-teacher cooperative masking outperforms purely teacher-driven masking by 1.3+ mAP.
2. **Spatial and channel are complementary**: Their combination outperforms either one alone.
3. **Adaptive ≠ learnable**: Making the teacher mask learnable alone provides negligible improvement; the critical factor is incorporating student participation.
4. **Cross-architecture generalization**: Consistently effective across three detectors—RetinaNet, Faster R-CNN, and RepPoints.
5. **Cross-task generalization**: Achieves state-of-the-art on both detection and segmentation tasks.

## Highlights & Insights

- **Rethinking agency in distillation**: The student should not be a passive recipient but an active participant in knowledge selection.
- **Masks that evolve dynamically with training**: Different distillation masks are generated for the same image at different epochs.
- **Clean ablation design**: Table 10 clearly isolates the respective contributions of "learnability" and "cooperation."
- **FPN neck distillation + segmentation map distillation**: Applying distillation at different positions for the two tasks represents a principled adaptation.
- **Diversity loss prevents mask collapse**: A simple yet effective regularization term whose complementary effect is visualized in Figure 3.

## Limitations & Future Work

- The cross-attention module introduces additional computational overhead (extra Q/K/V projections and attention computation); training time overhead is not separately reported.
- Validation is limited to CNN architectures (ResNet family); Transformer-based architectures such as ViT and Swin are not explored.
- The choice of $M$ (6 for detection, 19 for segmentation) appears empirically determined, with no sensitivity analysis provided.
- $\alpha$ and $\lambda$ are both fixed at 1; the effect of varying these hyperparameters is not explored.
- Evaluation is confined to COCO and Cityscapes; other dense prediction tasks (e.g., depth estimation, panoptic segmentation) are not considered.

## Related Work & Insights

Compared with MasKD (ICLR'23, which employs learnable receptive tokens but with offline-fixed masks) and FreeKD (CVPR'24, frequency-domain prompt guidance), the core innovation of ACAM-KD lies in **online cooperation**—masks are updated in real time throughout training. Compared with CWD (channel-wise distillation), ACAM-KD jointly optimizes both spatial and channel dimensions. The key insight from this work is that **giving the student model greater agency in the distillation process** may be a fundamental direction for improving distillation efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ICCV 2025\] PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection](pbcat_patch-based_composite_adversarial_training_against_physically_realizable_a.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)

</div>

<!-- RELATED:END -->
