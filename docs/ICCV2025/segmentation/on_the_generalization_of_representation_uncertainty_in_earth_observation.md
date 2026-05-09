---
title: >-
  [Paper Note] On the Generalization of Representation Uncertainty in Earth Observation
description: >-
  [ICCV 2025][Segmentation][Earth Observation] This paper systematically investigates the generalization of pretrained representation uncertainty in Earth Observation (EO), demonstrating that EO-pretrained uncertainty generalizes robustly across geographic locations, EO tasks, and target granularities, while remaining highly sensitive to ground sampling distance (GSD).
tags:
  - ICCV 2025
  - Segmentation
  - Earth Observation
  - Representation Uncertainty
  - Zero-Shot Transfer
  - Semantic Segmentation
  - Uncertainty Generalization
date: 2026-05-08
content_hash: 438026174ef16ac7
---

# On the Generalization of Representation Uncertainty in Earth Observation

**Conference**: ICCV 2025
**arXiv**: [2503.07082](https://arxiv.org/abs/2503.07082)
**Code**: [GitHub](https://github.com/Orion-AI-Lab/EOUncertaintyGeneralization)
**Area**: Image Segmentation
**Keywords**: Earth Observation, Representation Uncertainty, Zero-Shot Transfer, Semantic Segmentation, Uncertainty Generalization

## TL;DR

This paper systematically investigates the generalization of pretrained representation uncertainty in Earth Observation (EO), demonstrating that EO-pretrained uncertainty generalizes robustly across geographic locations, EO tasks, and target granularities, while remaining highly sensitive to ground sampling distance (GSD).

## Background & Motivation

The safety-critical nature of EO applications demands trustworthy deep learning models, and uncertainty estimation is a key mechanism for providing prediction confidence. However, standard uncertainty-aware methods—such as Bayesian neural networks and ensemble learning—impose substantial modeling complexity and computational overhead.

Recent computer vision research has introduced the concept of **pretrained representation uncertainty**, which learns uncertainty in the representation space of large-scale pretrained models to enable zero-shot uncertainty transfer. This is highly promising for EO, yet several challenges remain:

**Unique characteristics of EO data**: EO images exhibit multi-scale, multi-resolution, and multi-spectral properties, and lack a clear notion of "background," as all scene elements carry semantic meaning.

**Cross-domain gaps**: Can uncertainty learned from natural image pretraining transfer directly to EO? Can it transfer across different sensors and tasks within EO?

**Absence of evaluation frameworks**: Existing evaluations are limited to single-label classification (Recall@1), whereas EO tasks predominantly involve multi-label classification or semantic segmentation, necessitating new evaluation metrics.

The paper is motivated by the need to fill the gap in pretrained representation uncertainty research within the EO domain. It defines four **Semantic Factors (SFs)** governing EO imagery—GSD, domain of interest, target granularity, and spatial arrangement—and systematically evaluates the generalization properties of uncertainty along these dimensions.

## Method

### Overall Architecture

The paper adopts the framework of Kirchhof et al.: an uncertainty module (MLP) is trained on top of a frozen pretrained representation space, using the pretraining task loss as a proxy for aleatoric uncertainty. A scalar uncertainty value $u(x)$ is estimated for each input $x$. The core mechanism is that the uncertainty module operates entirely in representation space without modifying the backbone network, enabling lightweight and transferable uncertainty estimation.

### Key Designs

1. **Four Semantic Factors (SFs)**: Four key dimensions governing uncertainty generalization in EO imagery are defined. SF1: Ground Sampling Distance (GSD), determining spatial resolution and object detectability. SF2: Domain of Interest, encompassing geographic, temporal, and thematic/environmental variation. SF3: Target Granularity, ranging from fine-grained ($<1$ m, e.g., tree species classification) to coarse-grained ($>1$ km, e.g., forest detection). SF4: Spatial Arrangement, where the spatial distribution of objects reveals scene formation processes. These four factors constitute the analytical framework for systematically evaluating uncertainty generalization.

2. **Label Agreement@1 Metric Family for Multi-Label and Segmentation Settings**: The existing Recall@1 applies only to single-label tasks. This paper proposes the LA@1 metric family to extend evaluation to EO scenarios. For multi-label classification: One-LA@1 (sharing at least one class), All-LA@1 (containing all classes), and %-LA@1 (proportion of matched classes). For semantic segmentation: All-LA@1 (context metric), Patches-p-LA@1 (spatial patch-level similarity), PD-LA@1 (class distribution distance), and Patches-p-PD-LA@1 (combined spatial and contextual distribution distance). The Coefficient of Predictive Ability (CPA) is used in place of AUROC to evaluate continuous-valued metrics.

3. **Spatial Uncertainty Estimation**: Leveraging the sequence-to-sequence nature of ViTs, each token in the embedded token sequence is independently passed through the uncertainty module to compute a per-patch uncertainty value, yielding a spatial uncertainty map at patch resolution (each token corresponding to a $p \times p$ pixel region).

### Loss & Training

The uncertainty module is trained with a ranking-based loss to predict upstream task loss, avoiding loss-scale issues in downstream tasks. Specifically, four ViT variants (Tiny/Small/Base/Large) are trained on large-scale EO datasets (BigEarthNet and Flair), with their representations frozen prior to uncertainty module training. RGB bands are used to ensure fair comparison with ImageNet-pretrained models.

## Key Experimental Results

### Main Results

**Uncertainty Generalization: EO Pretraining vs. ImageNet Pretraining (ViT-Large, multiple datasets)**

| Pretraining Data | MLRSNet %-LA-CPA | Woody Patches-LA-CPA | MARIDA %-LA-CPA |
|---|---|---|---|
| ImageNet | <0.50 (below random baseline) | <0.50 | <0.50 |
| BigEarthNet | >0.55 | >0.55 | >0.55 |
| **Flair** | >0.55 | **Best** | >0.55 |

EO pretraining consistently outperforms ImageNet pretraining and random baselines across nearly all metrics and model sizes. ImageNet pretraining falls below the random guess baseline on EO datasets, demonstrating that domain gap severely impairs uncertainty generalization.

**Discard Test** — Alignment of Zero-Shot Uncertainty with Downstream Loss

| Dataset | BigEarthNet Pretrain MF | Flair Pretrain MF | ImageNet Pretrain MF |
|---|---|---|---|
| MLRSNet | High ↓ | High ↓ | Low / Non-decreasing |
| Woody | High ↓ | High ↓ | Low / Non-decreasing |
| MARIDA | High ↓ | High ↓ | Low / Non-decreasing |

(MF = Monotonicity Fraction, measuring the frequency with which loss decreases as high-uncertainty samples are removed)

### Ablation Study

**Effect of GSD on Uncertainty Generalization (ViT-Large, BigEarthNet pretrained at varying resolutions)**

| Pretraining Resolution | High-Res Inference Metric | Low-Res Inference Metric | Notes |
|---|---|---|---|
| 120×120 (original) | **Best** | Good | Original-resolution pretraining is optimal for high-res inference |
| 60×60 | Second best | Sufficient | Reduced pretraining resolution still viable |
| 30×30 | Degraded | Sufficient | Suitable for low-resolution inference |
| 16×16 | Severe degradation | Degraded | Extreme downsampling causes global collapse |

**Target Granularity Ablation (BigEarthNet-5 vs. BigEarthNet)**

| Model | BigEarthNet-5 (5 classes) | BigEarthNet (19 classes) |
|---|---|---|
| ViT-Tiny/Small | Consistently outperforms 19-class | Baseline |
| ViT-Base/Large | Comparable | Baseline |

Coarse labels emphasize fundamental semantic patterns rather than subtle distinctions, which is sufficient to maintain generalization capacity.

### Key Findings

- **Domain gap is the primary obstacle to uncertainty generalization**: The uncertainty module trained on ImageNet pretraining fails completely on EO data, despite the encoder producing representations of comparable quality to EO encoders. The degradation originates entirely from the uncertainty module, not the encoder.
- **EO-pretrained uncertainty is robust to geographic drift**: Models pretrained on European data generalize to Chile and New Zealand.
- **GSD alignment is critical**: Uncertainty estimation is optimal when upstream and downstream resolutions are matched.
- **Spatial uncertainty is promising**: ViT token-level uncertainty yields spatial uncertainty maps in which snow- and cloud-covered regions receive higher uncertainty scores.

## Highlights & Insights

- **Pioneering scope**: This is the first systematic study of pretrained representation uncertainty generalization in the EO domain.
- **Comprehensive evaluation framework**: A metric family (LA@1 + CPA) is proposed for evaluating representation-space uncertainty in multi-label classification and semantic segmentation settings.
- **High practical value**: A lightweight uncertainty module (MLP) is shown to provide reliable zero-shot uncertainty estimates.
- **Spatial uncertainty**: The patch token structure of ViTs is elegantly exploited to produce per-region uncertainty estimates without additional training.

## Limitations & Future Work

- Uncertainty is learned only on image-level task representations, not directly trained on dense prediction tasks.
- Spatial uncertainty estimation resolution is constrained by the ViT patch size (16×16).
- Systematic evaluation of multispectral/SAR data is not explored (addressed only briefly in the appendix).
- Aggregating patch-level uncertainty into a single image-level value may be unreliable (e.g., discrepancy between mean loss and maximum pixel loss in Flair).
- Future direction: developing a general-purpose EO uncertainty foundation model.

## Related Work & Insights

- Kirchhof et al. proposed the pretrained visual uncertainty framework and validated it on natural images; this paper extends their approach to the EO domain.
- EO foundation models such as Scale-MAE, CROMA, and DOFA establish the basis for large-scale representation learning.
- Uncertainty estimation remains substantially underexplored in EO; the evaluation framework proposed here is applicable to broader vision domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce the representation uncertainty generalization problem to EO, with a well-defined semantic factor analysis framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 8 datasets, 4 ViT variants, multi-dimensional metrics, discard tests, and noise tests.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured and analytically rigorous, though the metric system is relatively complex.
- **Value**: ⭐⭐⭐⭐ Provides the EO community with a standardized uncertainty evaluation framework and key actionable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification](../../NeurIPS2025/segmentation/torch-uncertainty_a_deep_learning_framework_for_uncertainty_quantification.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](../../CVPR2026/segmentation/brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[ICCV 2025\] Region-based Cluster Discrimination for Visual Representation Learning](region-based_cluster_discrimination_for_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
