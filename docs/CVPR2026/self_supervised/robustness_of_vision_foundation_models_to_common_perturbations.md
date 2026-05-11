---
title: >-
  [Paper Note] Robustness of Vision Foundation Models to Common Perturbations
description: >-
  [CVPR 2026][Self-Supervised Learning][foundation model] This paper presents the first systematic study on the robustness of vision foundation models to common perturbations (e.g., JPEG compression…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "foundation model"
  - "robustness"
  - "common perturbation"
  - "embedding"
  - "CLIP"
  - "DINOv2"
date: 2026-05-08
content_hash: c4d8c3eacc7bdd28
---

# Robustness of Vision Foundation Models to Common Perturbations

**Conference**: CVPR 2026
**arXiv**: [2604.14973](https://arxiv.org/abs/2604.14973)
**Code**: None
**Area**: AI Safety / Robustness
**Keywords**: foundation model, robustness, common perturbation, embedding, CLIP, DINOv2

## TL;DR

This paper presents the first systematic study on the robustness of vision foundation models to common perturbations (e.g., JPEG compression, brightness adjustment). It proposes three robustness metrics, formalizes five mathematical properties, finds that foundation models are generally non-robust, and introduces a fine-tuning method that improves robustness without sacrificing utility.

## Background & Motivation

Vision foundation models produce image embeddings for downstream tasks; however, common editing operations (e.g., JPEG compression, brightness/contrast adjustment) can alter these embeddings. Unlike adversarial perturbations, common perturbations occur frequently in non-adversarial, real-world scenarios. Three core questions are addressed: (1) How robust are foundation models themselves? (2) How robust are downstream applications? (3) How can robustness be improved? Designing appropriate metrics to quantify robustness is identified as a key challenge.

## Method

### Overall Architecture

(1) Three robustness metrics are proposed and their mathematical properties analyzed; (2) six industry-grade foundation models are systematically evaluated under nine categories of common perturbations; (3) a fine-tuning method is proposed to balance robustness and utility.

### Key Designs

1. **DivergenceRadius Metric**: The radius of the minimum enclosing ball in embedding space is used as the robustness metric. It satisfies all five desired mathematical properties (bounded domain, monotonicity, optimal robustness, worst-case robustness, and rotation invariance), outperforming cosine similarity and Euclidean distance metrics, which fail to satisfy the worst-case robustness property. The paper further proves that the cosine similarity metric and Euclidean distance metric are equivalent ($\mathcal{R}_{ed} = \sqrt{\mathcal{R}_{cs}}$).

2. **Linear Robustness–Performance Relationship**: An approximately linear relationship is discovered between downstream classification accuracy / depth estimation MSE and per-image robustness values, enabling accurate prediction of downstream performance on perturbed images via simple linear regression.

3. **Robustness-Aware Fine-Tuning**: The optimization objective is a weighted sum of a robustness loss and a utility loss. The robustness objective minimizes variation among embeddings of perturbed images, while the utility objective preserves performance on the original downstream task. Experiments confirm that the method improves robustness without degrading utility.

### Loss & Training

Fine-tuning loss = utility loss (preserving original representation quality) + $\alpha$ × robustness loss (minimizing embedding variation under perturbation), where $\alpha$ controls the trade-off.

## Key Experimental Results

### Main Results

CLIP (OpenAI, 3 architectures) and DINOv2 (Meta, 3 architectures) are evaluated under 9 perturbation categories:

| Finding | Details |
|---------|---------|
| Generally non-robust | All foundation models exhibit significant embedding changes under common perturbations |
| Architecture effect | ViT architectures are more robust than ResNet architectures |
| Downstream impact | Glass blur reduces zero-shot ImageNet classification accuracy by 9.4% |
| Predictability | Robustness values accurately predict downstream performance (high $R^2$ in linear fit) |

### Ablation Study

- Robustness monotonically decreases as the perturbation parameter range is expanded (validating the monotonicity property)
- Different perturbation types vary substantially in their impact on embeddings
- Fine-tuned models show improved robustness across most perturbation types

### Key Findings

- The robustness problem of foundation models has been severely overlooked — simple JPEG compression can substantially alter embeddings
- The greater robustness of ViT over ResNet may stem from Transformer's global attention mechanism
- The robustness metric can serve as a proxy for predicting downstream performance

## Highlights & Insights

- The formalization of five mathematical properties and the proofs of which metrics satisfy or violate them constitute a rigorous theoretical contribution
- The minimum enclosing ball formulation of DivergenceRadius is both intuitive and mathematically complete
- The linear robustness–performance relationship has direct practical utility

## Limitations & Future Work

- Only nine common perturbation types are considered; the effect of combined perturbations is not analyzed
- The fine-tuning method requires separate training for each perturbation type
- The robustness of multimodal foundation models (e.g., the text encoder of CLIP) is not addressed

## Related Work & Insights

- This work provides an important robustness reference baseline for the deployment of foundation models
- The DivergenceRadius metric is generalizable to other settings requiring quantification of representation stability
- The linear relationship between robustness and downstream performance simplifies quality assessment in practical deployments

## Rating

7/10 — Systematic, theoretically rigorous, and practically valuable; an important baseline contribution to robustness research on foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->
