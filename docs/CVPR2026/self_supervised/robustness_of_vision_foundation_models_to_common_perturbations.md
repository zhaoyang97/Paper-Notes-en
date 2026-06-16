---
title: >-
  [Paper Note] Robustness of Vision Foundation Models to Common Perturbations
description: >-
  [CVPR 2026][Self-Supervised Learning][foundation model] This paper presents the first systematic study of the robustness of vision foundation models (VFMs) to common perturbations (e.g., JPEG compression, brightness adjustment). It proposes three robustness metrics, formalizes five mathematical properties, reveals that foundation models are generally not robust, and introdu
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - foundation model
  - robustness
  - common perturbation
  - embedding
  - CLIP
  - DINOv2
date: 2026-05-08
content_hash: 7d9171656e5b9b7f
---
# Robustness of Vision Foundation Models to Common Perturbations

**Conference**: CVPR 2026  
**arXiv**: [2604.14973](https://arxiv.org/abs/2604.14973)  
**Code**: None  
**Area**: Self-supervised learning  
**Keywords**: foundation model, robustness, common perturbation, embedding, CLIP, DINOv2

## TL;DR

This paper presents the first systematic study of the robustness of vision foundation models (VFMs) to common perturbations (e.g., JPEG compression, brightness adjustment). It proposes three robustness metrics, formalizes five mathematical properties, reveals that foundation models are generally not robust, and introduces a fine-tuning method to improve robustness without sacrificing utility.

## Background & Motivation

Vision foundation models output embedding vectors for downstream tasks, but common editing operations (JPEG compression, image adjustments) alter these embeddings. Unlike adversarial attacks, common perturbations occur frequently in non-adversarial real-world scenarios. This study addresses three core problems: (1) How robust are foundation models themselves? (2) How robust are downstream applications? (3) How can robustness be improved? Designing appropriate metrics to quantify robustness is identified as a key challenge.

## Method

### Overall Architecture

The paper systematically investigates the vulnerability of vision foundation models to common perturbations. The approach consists of three steps: first, proposing metrics to quantify embedding robustness and analyzing their mathematical properties; second, utilizing these metrics to evaluate six industrial-grade foundation models across nine categories of perturbations; and finally, proposing a robustness-aware fine-tuning method to enhance robustness while maintaining utility.

### Key Designs

**1. DivergenceRadius Metric: Quantifying Embedding Robustness via Minimum Enclosing Ball Radius**

To evaluate the robustness of foundation models, a rigorous metric is required. This work defines the minimum enclosing ball of embedding vectors generated under various perturbations and uses its radius as the robustness metric. This metric satisfies all five expected mathematical properties (bounded domain, monotonicity, optimal robustness, worst robustness, rotational invariance). In contrast, cosine similarity and Euclidean distance metrics violate the "worst robustness" property. The authors also prove that Euclidean distance and cosine similarity metrics are equivalent ($\mathcal{R}_{ed} = \sqrt{\mathcal{R}_{cs}}$), indicating they share the same essence, whereas the minimum enclosing ball definition is mathematically complete.

**2. Robustness-Performance Linear Relationship: Predicting Downstream Performance via Robustness Values**

The authors observe that the robustness value is highly informative: downstream classification accuracy and depth estimation MSE exhibit an approximately linear relationship with the image's robustness value. Robustness values can accurately predict the performance of perturbed images on downstream tasks using simple linear regression. This implies that the robustness metric can serve as a proxy for downstream performance, eliminating the need to re-evaluate every downstream task during deployment.

**3. Robustness-Aware Fine-Tuning: Reducing Sensitivity to Perturbations While Preserving Utility**

To address the lack of robustness in foundation models, the paper proposes robustness-aware fine-tuning. The optimization objective is a weighted sum of a robustness loss and a utility loss. The robustness term minimizes the distance between the embeddings of the original image and its perturbed versions to reduce sensitivity. The utility term constrains the fine-tuned model's embeddings on clean images to remain close to those of the original model, preserving the quality of downstream representations. Experiments demonstrate that this approach improves robustness across most perturbation types without compromising utility.

### Loss & Training

Fine-tuning Loss = Robustness Loss $\mathcal{L}_1$ + $\lambda \cdot$ Utility Loss $\mathcal{L}_2$.  
The robustness loss $\mathcal{L}_1 = -\frac{1}{|\mathcal{D}|}\sum_x \cos(f'(x),\, f'(P(x,k)))$ aligns the embeddings of the original image and its perturbed version in the new model.  
The utility loss $\mathcal{L}_2 = -\frac{1}{|\mathcal{D}|}\sum_x \cos(f(x),\, f'(x))$ ensures the new model's embeddings on clean images remain close to the original model. $\lambda$ controls the balance between the two terms (default $\lambda=1$).

## Key Experimental Results

### Main Results

The study evaluates CLIP (OpenAI, 3 architectures) and DINOv2 (Meta, 3 architectures) across 9 perturbation types:

| Finding | Details |
|------|------|
| Generally not robust | All foundation models show significant embedding changes under common perturbations. |
| Architecture impact | ViT architectures are more robust than ResNet architectures. |
| Downstream impact | Glass blur causes a 9.4% drop in zero-shot ImageNet classification accuracy. |
| Predictability | Robustness values accurately predict downstream performance (high linear regression $R^2$). |

### Ablation Study

- Robustness decreases monotonically as the perturbation parameter domain expands, validating the monotonicity property.
- Different types of perturbations affect embeddings to significantly varying degrees.
- Fine-tuned models exhibit improved robustness across most perturbation types.

### Key Findings

- Robustness in foundation models is severely overlooked—simple JPEG compression can significantly change embeddings.
- The superior robustness of ViT over ResNet may be attributed to the global attention mechanism of Transformers.
- Robustness metrics can serve as proxy indicators for predicting downstream performance.

## Highlights & Insights

- Formalization of five mathematical properties and the analysis of which metrics satisfy or violate them provides a rigorous theoretical foundation.
- The DivergenceRadius metric based on the minimum enclosing ball is both intuitive and mathematically complete.
- The linear relationship between robustness and performance offers significant practical value for deployment.

## Limitations & Future Work

- Only nine common perturbations were considered; the effects of combined perturbations remain unanalyzed.
- The fine-tuning method requires separate training for each type of perturbation.
- The robustness of multimodal components, such as CLIP's text encoder, was not investigated.

## Related Work & Insights

- Provides an essential robustness reference baseline for the deployment of vision foundation models.
- The DivergenceRadius metric is generalizable to other scenarios requiring quantification of representation stability.
- The ability to predict performance via robustness simplifies quality assessment in practical applications.

## Rating

7/10 — Systematic, theoretically rigorous, and provides clear practical value. It serves as an important baseline for research into the robustness of foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] Scaling Parallel Sequence Models to Vision Foundation Models](scaling_parallel_sequence_models_to_vision_foundation_models.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[CVPR 2026\] Harnessing the Power of Foundation Models for Accurate Material Classification](harnessing_the_power_of_foundation_models_for_accurate_material_classification.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)

</div>

<!-- RELATED:END -->
