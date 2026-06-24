---
title: >-
  [Paper Note] Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models
description: >-
  [CVPR 2025][Segmentation][Concept Bottleneck Models] SALF-CBM is proposed to convert any vision network into a spatially-aware concept bottleneck model. By using CLIP visual prompting to generate spatialized concept maps, it provides dual explanations of "where" (heatmaps) and "what" (concepts), achieving performance that even surpasses the original backbone accuracy on ImageNet.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Concept Bottleneck Models"
  - "Spatial Explainability"
  - "Visual Prompting"
  - "Zero-Shot Segmentation"
  - "Model Debugging"
date: 2026-05-08
content_hash: 8c39cfe51f9015fd
---

# Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models

**Conference**: CVPR 2025  
**arXiv**: [2502.20134](https://arxiv.org/abs/2502.20134)  
**Code**: [https://itaybenou.github.io/show-and-tell/](https://itaybenou.github.io/show-and-tell/)  
**Area**: Segmentation/Explainable AI  
**Keywords**: Concept Bottleneck Models, Spatial Explainability, Visual Prompting, Zero-Shot Segmentation, Model Debugging

## TL;DR

SALF-CBM is proposed to convert any vision network into a spatially-aware concept bottleneck model. By using CLIP visual prompting to generate spatialized concept maps, it provides dual explanations of "where" (heatmaps) and "what" (concepts), achieving performance that even surpasses the original backbone accuracy on ImageNet.

## Background & Motivation

Although deep neural networks have achieved human-level performance, they lack the ability to explain decisions in a human-like manner—namely, explaining both "what is seen" and "where it is seen" simultaneously.

Existing explainable AI (XAI) methods suffer from a trade-off:

1. **Attribution Methods (Heatmaps)**: Such as GradCAM, LRP, etc., which show spatially attended regions but lack semantic descriptions. The highlighted regions can be ambiguous, leaving it unclear "what concepts" the model is seeing.
2. **Concept Bottleneck Models (CBMs)**: These project features into an interpretable concept space, but existing CBMs are global—only explaining that "feathers are observed in the entire image" without being able to pinpoint where.
3. **Accuracy Penalty**: The bottleneck layer in existing CBMs typically leads to a drop in classification accuracy, limiting their practical deployment.

Key Challenge: Can a unified framework be constructed to provide both spatial localization and concept-level explanations simultaneously, without sacrificing—or even while improving—classification accuracy?

## Method

### Overall Architecture

Given a pre-trained backbone, SALF-CBM performs the transformation in four steps: (1) GPT automatically generates a task-relevant list of concepts; (2) CLIP visual prompting computes a spatialized concept similarity matrix $P$; (3) A spatially-aware concept bottleneck layer is trained to project features into concept maps; (4) A sparse classification layer is trained on the pooled concept activations.

### Key Designs

**Design 1: Visual-Prompt-Based Local Concept Similarity**

- **Function**: Computes the similarity with each concept at every spatial location of each training image.
- **Mechanism**: A uniform grid of size $\tilde{H} \times \tilde{W}$ is established on the image. At each grid position, a red circle is drawn to generate an augmented image $x_n^{(h,w)}$. CLIP is used to compute the cosine similarity between this augmented image and each concept text: $P[n,m,h,w] = \frac{I_n^{(h,w)} \cdot T_m}{\|I_n^{(h,w)}\| \|T_m\|}$.
- **Design Motivation**: The visual prompting capability of CLIP (where a red circle guides CLIP to focus on a specific region) is cleverly utilized to acquire local semantics without any additional annotation. This is pre-computed only once.

**Design 2: Spatially-Aware Concept Bottleneck Layer**

- **Function**: Projects the black-box feature map of the backbone into an interpretable concept map space.
- **Mechanism**: The spatial information of the backbone features is preserved (avoiding global pooling). A single $1 \times 1$ convolutional layer (with $M$ output channels) maps the feature map to a concept map $c(x) \in \mathbb{R}^{M \times \tilde{H} \times \tilde{W}}$, which is aligned with the pre-computed similarity matrix $P$ using a cubic cosine similarity loss.
- **Design Motivation**: Traditional CBMs pool features globally before projection, losing spatial information. Preserving the spatial dimensions allows the concept maps to inherently possess localization capabilities, while maintaining the same parameter count as non-spatial CBMs.

**Design 3: Interactive Model Exploration and Debugging**

- **Function**: Supports active user queries and interventions in model decisions.
- **Mechanism**: (a) "Explain any region": The user specifies an ROI (point/bounding box/mask), concept activations within this region are aggregated, and the top-k concepts are displayed. (b) "Local intervention": The user can amplify or suppress the activation of a specific concept in a targeted region ($c(x)[m] \leftarrow c(x)[m] + \beta I$) to observe how the predictions change.
- **Design Motivation**: The core advantage of ante-hoc explainability is permitting intervention. Spatialized concept maps make local interventions possible, facilitating counterfactual analysis and model debugging.

### Loss & Training

The bottleneck layer is trained using a cubic cosine similarity loss:

$$\mathcal{L}_{CBL} = -\sum_{m=1}^{M} \sum_{h,w} sim(q[m,h,w], p[m,h,w])$$

The classification layer is trained using cross-entropy loss coupled with Elastic Net regularization (L1 + L2) to ensure sparsity.

## Key Experimental Results

### Classification Accuracy Comparison

| Method | Sparse | CUB-200 | Places365 | ImageNet |
|------|------|---------|-----------|----------|
| Standard Backbone | Yes | 75.96% | 38.46% | 74.35% |
| P-CBM | Yes | 59.60% | N/A | N/A |
| LF-CBM | Yes | 74.31% | 43.68% | 71.95% |
| **SALF-CBM** | **Yes** | **74.35%** | **46.73%** | **75.32%** |
| Standard Backbone | No | 76.70% | 48.56% | 76.13% |
| **SALF-CBM** | **No** | 76.21% | **49.38%** | **76.26%** |

### Zero-Shot Segmentation (ImageNet-Segmentation)

| Method | Pixel Acc.↑ | mIoU↑ | mAP↑ |
|------|------------|-------|------|
| GradCAM | 71.34% | 53.34% | 83.88% |
| FullGrad | 73.04% | 55.78% | 88.35% |
| **SALF-CBM** | **76.94%** | **58.30%** | 85.31% |

### Key Findings

1. SALF-CBM outperforms the original backbone on ImageNet with 75.32%/76.26% (sparse/non-sparse) versus 74.35%/76.13%, demonstrating that spatialized concept representation does not sacrifice accuracy and can even improve it.
2. The pixel accuracy of zero-shot segmentation is +3.9% higher than the best attribution method (FullGrad), and the mIoU is +2.52% higher.
3. The difference between sparse and non-sparse SALF-CBM is less than 1% on ImageNet, indicating that a small number of concepts is sufficient to capture decision-making information.
4. The concept bottleneck layer introduces no additional learnable parameters (compared to non-spatial CBMs), meaning that spatial information comes "for free."

## Highlights & Insights

1. **Unified "Show and Tell" Framework**: First to simultaneously provide spatial and concept-level explanations within a single model, naturally coupling the two.
2. **Creative Application of CLIP Visual Prompting**: Uses red circles to guide CLIP to attend to local regions, obtaining spatialized concept annotations without requiring a segmentation model.
3. **Accuracy Improvement instead of Penalty**: The spatial concept bottleneck layer surprisingly improves classification accuracy, shattering the conventional wisdom that "interpretability requires sacrificing accuracy."

## Limitations & Future Work

1. The pre-computation cost of visual prompting (red circles + CLIP inference) is high, especially under high-resolution grids.
2. The concept list depends on GPT generation, and its quality hinges on prompt design and category coverage.
3. Using red circles as visual prompts may introduce artifacts in specific images (e.g., when the image itself contains red objects).
4. More efficient ways of local concept computation can be explored, such as a combination of SAM and CLIP.

## Related Work & Insights

- **LF-CBM**: A pioneer in annotation-free concept bottleneck models, upon which this work adds a spatial dimension.
- **CLIP + Visual Prompting**: The ability of visual prompting to direct CLIP's focus to local regions is creatively exploited.
- **CRAFT**: A matrix factorization-based concept attribution method, which also provides spatial localization but post-hoc.
- Insight: Spatialized concept representations can serve as intermediate representations for downstream dense prediction tasks.

## Rating

⭐⭐⭐⭐ — Elegantly unifies the two major XAI paradigms of spatial attribution and concept explanation, achieving the ideal outcome of "interpretability boosting accuracy." The utilization of CLIP visual prompting is highly ingenious. Limitations lie in the pre-computation cost and the dependency on CLIP's capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time](../../ECCV2024/segmentation/deep_nets_with_subsampling_layers_unwittingly_discard_useful_activations_at_test.md)
- [\[CVPR 2025\] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)
- [\[CVPR 2026\] Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](../../CVPR2026/segmentation/concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2025\] A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [\[CVPR 2025\] EditAR: Unified Conditional Generation with Autoregressive Models](editar_unified_conditional_generation_with_autoregressive_models.md)

</div>

<!-- RELATED:END -->
