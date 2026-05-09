---
title: >-
  [Paper Note] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning
description: >-
  [ICCV 2025][3D Vision][Point Cloud Self-Supervised Learning] This paper proposes a domain-adaptive point cloud MAE framework (DAP-MAE) that enables a single cross-domain pre-training to achieve state-of-the-art performance across multiple downstream tasks spanning diverse domains, including object classification, facial expression recognition, part segmentation, and object detection, via two key modules: the Heterogeneous Domain Adapter (HDA) and the Domain Feature Generator (DFG).
tags:
  - ICCV 2025
  - 3D Vision
  - Point Cloud Self-Supervised Learning
  - Masked Autoencoder
  - Cross-Domain Learning
  - Domain Adaptation
  - 3D Point Cloud Analysis
date: 2026-05-08
content_hash: e6d39c7a99641ead
---

# DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning

**Conference**: ICCV 2025
**arXiv**: [2510.21635](https://arxiv.org/abs/2510.21635)
**Code**: [https://github.com/CVI-SZU/DAP-MAE](https://github.com/CVI-SZU/DAP-MAE)
**Area**: 3D Vision / Self-Supervised Learning
**Keywords**: Point Cloud Self-Supervised Learning, Masked Autoencoder, Cross-Domain Learning, Domain Adaptation, 3D Point Cloud Analysis

## TL;DR

This paper proposes a domain-adaptive point cloud MAE framework (DAP-MAE) that enables a single cross-domain pre-training to achieve state-of-the-art performance across multiple downstream tasks spanning diverse domains, including object classification, facial expression recognition, part segmentation, and object detection, via two key modules: the Heterogeneous Domain Adapter (HDA) and the Domain Feature Generator (DFG).

## Background & Motivation

3D point cloud datasets are considerably smaller in scale compared to 2D image datasets, and different application domains (objects, faces, scenes) each maintain their own small independent datasets. Existing point cloud MAE methods (e.g., Point-MAE, PiMAE, 3DFaceMAE) are typically pre-trained on a single domain and transferred to same-domain tasks; performance degrades significantly when transferred to a different domain. A natural approach is to mix data from different domains for joint pre-training—however, experiments reveal that naively mixing heterogeneous domain data introduces inter-domain noise, causing downstream performance to decline rather than improve. For instance, jointly pre-training ReCon-SMC on object, face, and scene data actually yields lower accuracy on facial expression recognition and object detection compared to single-domain pre-training.

This exposes a fundamental tension: **more data from multiple domains is desirable, yet multi-domain information interferes with itself.**

## Core Problem

How can point cloud data from heterogeneous domains (objects, faces, scenes) be effectively exploited for joint pre-training, such that a single pre-trained model consistently benefits multiple downstream tasks across all domains, rather than being hindered by inter-domain discrepancies?

## Method

### Overall Architecture

DAP-MAE is built upon a standard Transformer MAE architecture (using ReCon-SMC as the baseline) and introduces two core components:

1. **Heterogeneous Domain Adapter (HDA)**: handles cross-domain discrepancies at the tokenization stage
2. **Domain Feature Generator (DFG)**: extracts domain features via contrastive learning to guide downstream task adaptation

The overall pipeline consists of pre-training and fine-tuning stages, in which HDA operates under two different modes.

Pre-training data is drawn from three domains: ShapeNet (object domain $\mathbb{O}$, 50K+ point clouds), Enriched FRGCv2 (face domain $\mathbb{F}$, 120K+ 3D faces), and S3DIS (scene domain $\mathbb{S}$, indoor point cloud scenes). Each point cloud is uniformly sampled to 4,096 points.

### Key Designs

**1. Heterogeneous Domain Adapter (HDA) — Two Operating Modes**

HDA consists of three parallel MLPs, each corresponding to one domain. The key design lies in the adoption of entirely different strategies during pre-training and fine-tuning:

- **Pre-training — Adaptation Mode**: Given an input point cloud, the MLP corresponding to its domain is selected to process the tokens. Object-domain point clouds are routed through the object MLP, faces through the face MLP, and scenes through the scene MLP. Each MLP thus specializes in the geometric features of one domain, preventing cross-domain interference.

- **Fine-tuning — Fusion Mode**: The three MLPs are frozen, and the input point cloud is passed through all three simultaneously. Two additionally trained MLPs generate fusion coefficients to perform a weighted linear combination of the three MLP outputs. The output of the domain-specific MLP corresponding to the target downstream task serves as the primary feature, while the other two provide auxiliary features. Fusion is applied in two steps (once after each FC layer within each MLP).

The elegance of this design lies in training each MLP to independently learn a domain-specific representation during pre-training (avoiding interference), while adaptively borrowing cross-domain knowledge via learnable fusion coefficients during fine-tuning.

**2. Domain Feature Generator (DFG)**

DFG maintains one class token and three domain tokens (one each for objects, faces, and scenes). A cross-attention mechanism decomposes the encoder output features into:
- **Domain feature $d$**: encodes the overall domain membership of the point cloud
- **Class feature $c$**: encodes the specific category of the point cloud

During pre-training, only the domain features are trained via a contrastive loss. During fine-tuning, both domain and class features are trained, and all three representations (domain feature, class feature, and point cloud feature) are jointly fed into the downstream task head.

### Loss & Training

**Pre-training total loss**: $\mathcal{L} = w_1 \mathcal{L}_{rec} + w_2 \mathcal{L}_{con}$

- **Reconstruction loss $\mathcal{L}_{rec}$**: standard Chamfer Distance for reconstructing masked point cloud patches
- **Contrastive loss $\mathcal{L}_{con}$**: domain features of same-domain point clouds are pulled together (cosine similarity toward 1), while cross-domain pairs are pushed apart (with margin $a$)

The optimal weight configuration is $w_1 = 100$, $w_2 = 0.001$, indicating that the reconstruction loss dominates (the contrastive loss is prone to overfitting).

**Training details**:
- Pre-training: batch size 512, AdamW optimizer, learning rate 0.0005, cosine decay, 300 epochs
- HDA MLP parameters are frozen during fine-tuning (releasing them causes overfitting, as validated experimentally)
- Hardware: NVIDIA V100 32GB

## Key Experimental Results

| Task / Dataset | Metric | DAP-MAE | Prev. SOTA (single-modal) | Gain |
|---|---|---|---|---|
| ScanObjectNN OBJ_BG | Acc (%) | **95.18** | 95.18 (Point-FEMAE) | — |
| ScanObjectNN PB_T50_RS | Acc (%) | **90.25** | 90.22 (Point-FEMAE) | +0.03 |
| BU-3DFE Expression Recog. | Acc (%) | **89.83** | 89.15 (DrFER) | +0.68 |
| Bosphorus Expression Recog. | Acc (%) | **88.45** | 86.77 (DrFER) | +1.68 |
| ShapeNetPart Part Seg. | mIoU_c (%) | **84.9** | 84.3 (PM-MAE) | +0.6 |
| ScanNetV2 Det. | AP50 | **43.2** | 42.1 (ACT, cross-modal) | +1.1 |
| ScanNetV2 Det. | AP25 | **64.0** | 63.8 (ACT/ReCon) | +0.2 |

**Key comparison** (same baseline ReCon-SMC: single-domain pre-training vs. naive cross-domain mixing vs. DAP-MAE):
- Object classification: 94.15% → 94.32% (naive mixing +0.17) → **95.18%** (DAP-MAE +1.03)
- Expression recog. (BOS): 87.69% → 87.23% (naive mixing −0.46) → **88.45%** (DAP-MAE +0.76)
- Detection AP50: 42.7% → 42.5% (naive mixing −0.2) → **43.2%** (DAP-MAE +0.5)

This comparison is the most compelling: naive multi-domain data mixing is largely ineffective or even harmful, whereas DAP-MAE effectively leverages cross-domain data.

### Ablation Study

1. **Component ablation** (ScanObjectNN OBJ_BG):
   - Baseline (single-domain): 94.15%
   - +Cross-domain data (CD): 94.32% (+0.17)
   - +CD+HDA: 94.66% (+0.51)
   - +CD+DFG: 94.66% (+0.51)
   - +CD+HDA+DFG (full DAP-MAE): **95.18%** (+1.03)
   - HDA and DFG each contribute independently; their combination is strictly superior.

2. **HDA fusion mode ablation**:
   - No fusion (domain-specific MLP only): 94.66%
   - Direct summation fusion: 92.59% (catastrophic drop)
   - FC-predicted coefficient fusion: 94.84%
   - MLP-predicted coefficient fusion (full design): **95.18%**

3. **Feature combination ablation**: Combining domain feature $d$, class feature $c$, and point cloud feature $\mathcal{F}$ yields the best result (95.18%); using each individually gives 93.80%, 93.12%, and 94.49%, respectively.

4. **Computational overhead**: Compared to the baseline ReCon (43.6M parameters, 5.3G FLOPs), DAP-MAE adds only 0.2M parameters and 0.1G FLOPs—essentially negligible.

## Highlights & Insights

1. **Using different strategies for cross-domain information during pre-training and fine-tuning** is the central insight. "Divide and conquer" during pre-training, "unite and fuse" during fine-tuning—this principle is broadly applicable to cross-domain and multi-task learning scenarios.

2. **Naive multi-domain data mixing is no better than single-domain training**, and sometimes worse. This empirically validates the reality of inter-domain interference and provides solid motivation for DAP-MAE's design.

3. **Freezing HDA parameters during fusion fine-tuning** is counterintuitive yet highly effective—releasing the parameters leads to overfitting. This suggests that domain-specific representations learned during pre-training are valuable and must be preserved.

4. The contrastive loss weight must be extremely small (0.001 vs. 100 for the reconstruction loss), indicating that contrastive learning within a MAE framework is prone to overfitting and requires careful balancing.

5. The minimal additional overhead (+0.2M parameters) demonstrates that the performance gains arise primarily from design ingenuity rather than brute-force model scaling.

## Limitations & Future Work

1. **No dynamic extension to new domains**: The current three domains are fixed at pre-training time; adding a new domain requires full retraining. The authors acknowledge this limitation and suggest continual learning strategies as a future direction.

2. **Fixed number of domains**: HDA hard-codes three MLPs for three domains (objects, faces, scenes), lacking flexibility. A dynamic, variable-number adapter design would be more general.

3. **Imbalanced pre-training data**: ShapeNet has ~50K samples, FRGCv2 has ~120K, and S3DIS varies in scale, potentially introducing inter-domain data imbalance.

4. **Point-cloud-only modality**: Although DAP-MAE outperforms some cross-modal methods (PC+Image) on several tasks, a gap remains compared to ReCon-full, which uses images and text (e.g., PB_T50_RS: 90.25 vs. 90.63). Incorporating cross-modal information may yield further improvements.

5. **Evaluation is primarily on indoor/synthetic data**: Validation on outdoor scenarios (e.g., autonomous driving point clouds) is absent.

## Related Work & Insights

| Method | Modality | Cross-Domain Support | Key Distinction |
|---|---|---|---|
| Point-MAE | Point cloud only | Single-domain | Standard MAE; single-domain pre-training and transfer |
| ReCon-SMC | Point cloud only | Single-domain (forced cross-domain yields poor results) | Baseline for DAP-MAE |
| ReCon-full | PC+Image+Text | Single-domain | Cross-modal distillation; high training cost |
| ACT | PC+Image | Single-domain | 2D pre-trained Transformer adapted to 3D |
| Point-FEMAE | Point cloud only | Single-domain | Compact representation MAE; leading on single-scan classification |
| **DAP-MAE** | **Point cloud only** | **Multi-domain** | **First to support multi-domain joint pre-training; strongest generality** |

DAP-MAE is the first work to systematically address the problem of cross-domain MAE pre-training for point clouds, filling a notable gap in the literature.

## Relevance to My Research

- **Continual learning extension**: The explicitly acknowledged limitation of "inability to dynamically add new domains" is a promising research direction; adapter- or prompt-tuning-based approaches could enable incremental domain expansion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to propose a cross-domain point cloud MAE pre-training framework; the dual-mode switching design of HDA is genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 task types across 6 datasets; ablation studies are comprehensive; parameter count comparisons are provided.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear; problem definition is precise; figures and tables are highly informative.
- **Value**: ⭐⭐⭐ The cross-domain adaptation ideas offer useful reference points, though direct relevance to specific research directions is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
