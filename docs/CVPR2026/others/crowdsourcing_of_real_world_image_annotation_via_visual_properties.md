---
title: >-
  [Paper Note] Crowdsourcing of Real-world Image Annotation via Visual Properties
description: >-
  [CVPR 2026][image annotation] This paper proposes an image annotation methodology constrained by visual properties. It constructs an object category hierarchy through knowledge representation and combines an interactive crowdsourcing framework that leverages visual genus and visual differentia to guide the annotation process, thereby reducing annotator subjectivity and the semantic gap problem.
tags:
  - CVPR 2026
  - image annotation
  - crowdsourcing
  - visual properties
  - semantic gap
  - object hierarchy
date: 2026-05-08
content_hash: 5e007e71c50ec72b
---

# Crowdsourcing of Real-world Image Annotation via Visual Properties

**Conference**: CVPR 2026
**arXiv**: [2604.14449](https://arxiv.org/abs/2604.14449)
**Code**: None
**Area**: Dataset Construction / Annotation Methodology
**Keywords**: image annotation, crowdsourcing, visual properties, semantic gap, object hierarchy

## TL;DR

This paper proposes an image annotation methodology constrained by visual properties. It constructs an object category hierarchy through knowledge representation and combines an interactive crowdsourcing framework that leverages visual genus and visual differentia to guide the annotation process, thereby reducing annotator subjectivity and the semantic gap problem.

## Background & Motivation

The construction of existing image datasets (e.g., ImageNet, Open Images) suffers from subjectivity: annotators map images to predefined categories based on personal interpretation, leading to many-to-many mapping issues and annotation inconsistencies. For example, the same image may be assigned to three categories of different granularities in ImageNet, or visually distinct images (real objects, toys, cartoons, figurines) may all be labeled as the same "brown bear" category. The root cause is the Semantic Gap Problem (SGP) introduced by the complexity and ambiguity of natural language.

## Method

### Overall Architecture

A four-step annotation strategy: (1) **Label Definition**: construct a category hierarchy based on knowledge bases and precisely define the visual properties of each category; (2) **Label Disambiguation**: assign a unique concept identifier to each label; (3) **Object Localization**: identify and localize all objects in an image; (4) **Visual Classification**: guide categorization through visual properties by confirming visual genus and visual differentia level by level.

### Key Designs

1. **Visual Property-Based Category Hierarchy**: The visual genus serves as the shared attribute of the parent category (e.g., the visual genus of "goldfinch" is "finch"), while the visual differentia distinguishes sibling categories (e.g., "crimson face and yellow-black wings"). The annotation process requires annotators to verify these concrete visual properties rather than directly matching abstract category names.

2. **Interactive Crowdsourcing Q&A Framework**: Questions are dynamically generated based on the predefined object hierarchy. Annotators begin at the root node and answer yes/no questions of the form "does the object exhibit a given visual differentia attribute," proceeding downward level by level until further subdivision is no longer possible. The VisClassify algorithm implements this recursive hierarchical visual classification process.

3. **Multi-Granularity Label Output**: The resulting dataset contains multi-level labels: fine-grained category labels at different granularities, visual property labels, and natural language descriptions of visual features, supporting diverse tasks such as object recognition, fine-grained classification, zero-shot recognition, and image captioning.

### Core Algorithm: VisClassify

A recursive hierarchical visual classification process (Algorithm 1): starting from the root node of the predefined hierarchy tree $H$, annotators are asked whether the object exhibits the visual differentia of the current node. A "No" response results in the image being discharged; a "Yes" response records the current-level label and continues traversal of child nodes, until the current node has no subcategories or the annotator negates the visual differentia of all children. Questions at each branching point are dynamically generated from predefined visual differentia attributes in the knowledge base, rather than requiring annotators to freely judge category names.

### Loss & Training

This paper presents an annotation methodology and does not involve model training.

## Key Experimental Results

### Main Results

The effectiveness of the methodology is validated through crowdsourcing experiments, with annotator feedback informing directions for optimizing the crowdsourcing setup. Compared to unconstrained free annotation, visual property-constrained annotation significantly improves annotation consistency and accuracy. In the experiments, annotators label bird images by answering hierarchical visual property questions (e.g., distinguishing "goldfinch" from "greenfinch" requires verifying the visual differentia "crimson face"), and results show that annotators from diverse backgrounds achieve higher agreement under visual property guidance. The resulting dataset contains multi-granularity labels, visual property labels, and natural language descriptions, directly supporting object recognition, fine-grained classification, zero-shot recognition, and image captioning.

### Key Findings

- Visual property constraints effectively reduce inter-annotator subjectivity
- The hierarchical Q&A process lowers the cognitive burden of annotation tasks
- Multi-level labels provide richer supervisory signals for various downstream tasks

## Highlights & Insights

- Systematically redesigning the annotation pipeline from the perspective of the semantic gap problem is a well-motivated contribution
- The conceptualization of visual genus and visual differentia reflects philosophical depth
- Multi-granularity label output enhances the general applicability of the resulting dataset
- Each category is precisely defined using knowledge bases such as WordNet and Wikipedia during annotation, eliminating ambiguity introduced by natural language polysemy
- The Label Disambiguation step assigns unique concept identifiers (e.g., "1-1" and "2-5-3") to each label, resolving polysemy issues
- The Object Localization step employs object localization models to automatically crop multi-object images into single-object images, eliminating object ambiguity

## Limitations & Future Work

- Constructing the predefined visual property hierarchy requires domain expert involvement, making scaling costly
- The methodology targets object recognition scenarios; its applicability to tasks such as scene understanding and action recognition is limited
- The experimental scale is relatively small; large-scale validation has not been conducted
- The definition of visual differentia attributes relies on taxonomic canons, and their adaptability to annotators from different cultural backgrounds remains to be examined
- Integration with automated annotation tools (e.g., MLLM-assisted annotation) has not been explored

## Related Work & Insights

- The systematic analysis of annotation quality issues in existing benchmark datasets is of reference value
- The visual property-guided annotation paradigm can be incorporated into active learning and human-machine collaborative annotation
- The hierarchical labeling scheme offers guidance for constructing higher-quality datasets
- Concrete case analyses from ImageNet and Open Images reveal systematic deficiencies in existing annotations

## Rating

5/10 — The problem formulation is valuable, but the work lacks large-scale experimental validation and quantitative improvement metrics.

The four-step annotation strategy (Label Definition → Label Disambiguation → Object Localization → Visual Classification) reflects a complete pipeline design spanning from knowledge representation to crowdsourcing execution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](../../NeurIPS2025/others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)

</div>

<!-- RELATED:END -->
