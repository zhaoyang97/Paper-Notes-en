---
title: >-
  [Paper Note] Referring Layer Decomposition
description: >-
  [ICLR 2026][Image Generation][Layer Decomposition] This paper introduces the Referring Layer Decomposition (RLD) task, which predicts complete RGBA layers from a single RGB image given flexible user-provided prompts (spatial, textual, or hybrid). It also constructs the RefLade dataset comprising 1.1 million samples and proposes an automated evaluation protocol.
tags:
  - ICLR 2026
  - Image Generation
  - Layer Decomposition
  - RGBA Layers
  - Multimodal Referring Input
  - Data Engine
  - RefLayer
date: 2026-05-08
content_hash: 306715d18b8e6211
---

# Referring Layer Decomposition

**Conference**: ICLR 2026
**arXiv**: [2602.19358](https://arxiv.org/abs/2602.19358)
**Code**: [https://yaojie-shen.github.io/project/RLD/](https://yaojie-shen.github.io/project/RLD/)
**Area**: Image Decomposition / Image Editing
**Keywords**: Layer Decomposition, RGBA Layers, Multimodal Referring Input, Data Engine, RefLayer

## TL;DR

This paper introduces the Referring Layer Decomposition (RLD) task, which predicts complete RGBA layers from a single RGB image given flexible user-provided prompts (spatial, textual, or hybrid). It also constructs the RefLade dataset comprising 1.1 million samples and proposes an automated evaluation protocol.

## Background & Motivation

Modern generative models typically process images holistically, lacking explicit representations of individual scene elements. This makes selective manipulation, cross-edit consistency maintenance, and semantic alignment particularly challenging. Image layers — transparent visual units in RGBA format — provide a more intuitive framework analogous to layer-based workflows in Photoshop.

Limitations of prior work:
- **MuLAn**: Limited data scale (44K images) with a success rate of only 36%.
- **Text2Layer**: Restricted to separating foreground and background into two layers only.
- **LayerDecomp**: Relies on synthetic supervision and requires target masks.

The core innovation of the RLD task lies in supporting diverse user prompts (points, boxes, masks, and text) to enable on-demand extraction of target RGBA layers.

## Method

### Overall Architecture

RLD comprises three major components:
1. **RefLade Dataset**: 1.1 million image–layer–prompt triplets.
2. **Automated Evaluation Protocol**: Three-axis evaluation covering preservation, completion, and fidelity.
3. **RefLayer Baseline Model**: Conditional generation built upon Stable Diffusion 3.

### Data Engine (6-Stage Pipeline)

1. **Pre-filtering**: Rule-based removal of low-quality images (86.1% retention rate).
2. **Scene Understanding**: Integration of closed-set detection, open-vocabulary detection, and MLLM-based localization.
3. **Layer Completion**: Reconstruction of occluded object regions.
4. **Post-processing**: Mask refinement and alpha matte prediction.
5. **Prompt Generation**: Generation of spatial, textual, and multimodal prompts.
6. **Post-filtering**: Assessment of RGBA layer fidelity, realism, and semantic consistency.

The overall success rate improves from MuLAn's 36% to 70%.

### Evaluation Protocol (Three Dimensions)

**Preservation score $\mathcal{S}_{\text{vis}}$**: LPIPS similarity over originally visible regions.

$$\mathcal{S}_{\text{vis}} = \mathbb{E}_{(p,g)\sim\mathcal{D}}[\text{LPIPS}(g_{\text{rgb}} \odot g_v, p_{\text{rgb}} \odot g_v)]$$

**Completion score $\mathcal{S}_{\text{gen}}$**: Directional similarity based on CLIP features.

$$\mathcal{S}_{\text{gen}} = \mathbb{E}[\cos(f(g_{\text{rgb}}) - f(g_{\text{rgb}} \odot g_v), f(p_{\text{rgb}}) - f(g_{\text{rgb}} \odot g_v))]$$

**Fidelity score $\mathcal{S}_{\text{fid}}$**: FID computed after alpha-compositing the predicted layer onto a background.

**HPA composite score**: A normalized weighted average calibrated via human-preference Elo rankings, which exhibits strong correlation with human judgments.

### RefLayer Model

- Built upon Stable Diffusion 3.
- A VAE encoder encodes the source image and spatial prompts.
- A lightweight convolutional layer compresses the channel dimensions.
- Dual decoders: a standard RGB decoder and a custom alpha decoder.

**Prompt encoding strategy**: All spatial prompts are unified into a colored RGB image format:
- Blue canvas → background
- Green region → bounding box
- Red region → mask
- Gaussian heatmap → point

## Experiments

### Dataset Statistics

| Dataset | Task | #Images | #Categories | #Instances | Occlusion Rate |
|--------|------|-------|-------|-------|--------|
| MuLAn | LD | 44,860 | 759 | 101,269 | 7.7% |
| **RefLade** | **RLD** | **430,488** | **12K** | **871,829** | **60.8%** |

### Evaluation Protocol Validation

The HPA score exhibits strong correlation with human Elo rankings, whereas individual metrics $\mathcal{S}_{\text{vis}}$, $\mathcal{S}_{\text{fid}}$, and $\mathcal{S}_{\text{gen}}$ each fail to consistently reflect human preferences in isolation.

### Quality Assessment

- 74.7% of foreground layers and 70.2% of background layers meet the quality threshold.
- Human annotation was conducted over 43 days by 9 professional annotators.
- Careful curation yields 59K high-quality images and 110K validated layers.

### Key Findings

- Coarse-grained prompts (e.g., a single point) may lead to coarse-grained outputs, whereas precise prompts yield accurate object-level layers.
- RefLayer demonstrates strong zero-shot generalization capability.
- The multi-granularity prompt system supports flexible control ranging from coarse to fine.

## Highlights & Insights

- This work is the first to formally define the layer decomposition task conditioned on multimodal referring inputs.
- The data engine is systematically designed and raises the success rate from 36% to 70%.
- The evaluation protocol aligns closely with human preferences, addressing a critical evaluation bottleneck.
- The RefLade dataset is substantially larger than existing counterparts (430K vs. MuLAn's 44K).

## Limitations & Future Work

- The data engine relies on multiple external models (detection, segmentation, inpainting), making cascading errors unavoidable.
- Human annotation incurs high cost (43 days × 9 annotators).
- Ground-truth layers used in the evaluation protocol may themselves be imperfect.

## Related Work & Insights

- **Image Understanding and Editing**: Detection, segmentation, inpainting, alpha matting, etc.
- **Compositional Image Representation**: RGBA layer methods such as MuLAn and Text2Layer.
- **Referring Expression Segmentation**: Promptable segmentation methods such as SAM output masks only and do not reconstruct occluded content.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The task definition is original and fills a clear research gap.
- Data Contribution: ⭐⭐⭐⭐⭐ — Million-scale dataset + data engine + human annotation.
- Evaluation: ⭐⭐⭐⭐ — Three-dimensional evaluation protocol aligned with human preferences.
- Practicality: ⭐⭐⭐⭐ — Direct applicability to image editing and compositing.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](../../CVPR2026/image_generation/cycle-consistent_tuning_for_layered_image_decomposition.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](../../CVPR2026/image_generation/pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)

<!-- RELATED:END -->
