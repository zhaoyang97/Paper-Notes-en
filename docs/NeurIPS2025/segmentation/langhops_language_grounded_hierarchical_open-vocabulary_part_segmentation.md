---
title: >-
  [Paper Note] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Open-vocabulary part segmentation] LangHOPS is the first open-vocabulary object-part instance segmentation framework based on a multimodal large language model (MLLM). It establishes object-part hierarchical relationships in language space and leverages the knowledge and reasoning capabilities of MLLMs to bridge multi-granularity concepts. It achieves 56.9% AP on PartImageNet, surpassing the previous SOTA by 5.5%, and outperforms by 4.8% in cross-dataset settings.
tags:
  - NeurIPS 2025
  - Segmentation
  - Open-vocabulary part segmentation
  - object-part hierarchy
  - MLLM
  - language-space hierarchical modeling
  - instance segmentation
date: 2026-05-08
content_hash: 7dd5a36e0582e6cb
---

# LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2510.25263](https://arxiv.org/abs/2510.25263)
**Code**: Available (to be released)
**Area**: Segmentation
**Keywords**: Open-vocabulary part segmentation, object-part hierarchy, MLLM, language-space hierarchical modeling, instance segmentation

## TL;DR

LangHOPS is the first open-vocabulary object-part instance segmentation framework based on a multimodal large language model (MLLM). It establishes object-part hierarchical relationships in language space and leverages the knowledge and reasoning capabilities of MLLMs to bridge multi-granularity concepts. It achieves 56.9% AP on PartImageNet, surpassing the previous SOTA by 5.5%, and outperforms by 4.8% in cross-dataset settings.

## Background & Motivation

**Granularity limitations of open-vocabulary segmentation**: Current OVS methods primarily focus on object-level segmentation, while object-part (partonomic) segmentation remains an open problem. Decomposing objects into semantic parts (e.g., "car → wheel, door, hood") is critical for downstream tasks such as robotic manipulation and fine-grained recognition.

**Limitations of prior part segmentation methods**:
- VLPart, PartGLEE, and similar methods rely on **heuristic or learnable visual groupings** to model object-part relationships.
- Visual-space groupings lack semantic prior knowledge (e.g., "birds have wings and beaks"), leading to poor generalization to unseen categories.
- The absence of hierarchical context between objects and parts results in imprecise part parsing.

**Core Motivation**: Migrate object-part hierarchical relationships from visual space to **language space**, leveraging the world knowledge internalized by MLLMs ("what parts should this object have?") to initialize and refine part queries, thereby achieving better cross-category generalization.

## Method

### Overall Architecture

LangHOPS adopts a two-stage architecture:
1. **Object segmentation stage**: Detects and segments object instances in the image, generating object queries $\mathbf{O}^L$.
2. **MLLM-driven part parsing stage**: Feeds object queries together with hierarchically initialized part queries in language space into an MLLM, which refines part queries $\mathbf{P}$ using its reasoning capabilities; these are then passed to a part decoder to produce final segmentation.

The input is an image and a list of candidate object-part categories; the output is hierarchical object and part instance segmentation results.

### Key Designs

**1. Language-Grounded Hierarchies**

Unlike conventional methods that use randomly initialized learnable queries, LangHOPS constructs initial part query representations in language space:

- Given candidate part category names, initial part queries are constructed using the semantic hierarchical relationships between objects and parts.
- For example, for the object "dog," the text representations of associated parts—"head," "leg," "tail"—are used to initialize the corresponding part queries.
- This initialization naturally encodes object-part membership relationships and provides stronger semantic priors than randomly initialized learnable queries.

Ablation validation: Replacing language-grounded hierarchy initialization with $N$ learnable queries ("w/o hierarchy") reduces AP on PartImageNet from 26.7% to 22.5% (−4.2%), demonstrating the importance of language hierarchies.

**2. MLLM-based Object-Part Parsing**

This is the core innovation module of LangHOPS:

- Object queries $\mathbf{O}^L$ output from the object segmentation stage carry visual features of objects.
- $\mathbf{O}^L$ and language-hierarchically initialized part queries $\mathbf{P}^0$ are jointly fed into the MLLM.
- Based on its understanding of object visual features and internalized world knowledge, the MLLM **reasons about which parts the object should contain** and refines the part queries accordingly.
- The output refined part queries $\mathbf{P}$ are sent to the part decoder to generate final part segmentation masks.

Key gradient flow design: Gradients from the part segmentation loss are back-propagated through the MLLM to object queries $\mathbf{O}^L$, enabling **Object-Part Synergy**—optimization of part segmentation simultaneously improves object segmentation quality.

Ablation validation:
- Replacing the MLLM with Q-Former ("w/o MLLM") reduces PartImageNet AP from 26.7% to 23.2% (−3.5%).
- Detaching the gradient flow ("Detached Obj-Part Seg") reduces attention scores for both object and part.

**3. Object-Part Synergy**

LangHOPS demonstrates that joint training of object and part segmentation outperforms training object segmentation alone:

- "Obj Seg" (object segmentation only): PartImageNet object mAP 67.9%.
- "Obj-Part Seg" (joint training): PartImageNet object mAP **68.3%** (+0.4%), part mAP **14.9%**.
- Joint training not only provides part segmentation capability but also **improves object segmentation in return**, confirming that gradient back-propagation through the MLLM parsing module effectively enhances the quality of object queries.

Attention score analysis: Under synergistic training, object attention scores increase from 0.76 to 0.82, and part attention scores from 0.58 to 0.67.

### Loss & Training

- **Two-stage training**: Stage 1 trains on object segmentation; Stage 2 jointly trains object and part segmentation. The two-stage strategy outperforms direct one-stage training in cross-dataset generalization.
- Ablation comparison: Two-stage vs. one-stage in cross-dataset setting yields AP of 26.7 vs. 25.4 (+1.3), though one-stage is marginally better in-domain (58.6 vs. 56.9).
- Training data scalability: Supports progressively incorporating more datasets (PPS-116 → +INS → +INS+PART); LangHOPS achieves the largest gain upon adding part-level annotations.

## Key Experimental Results

### Cross-Dataset Experiment: Trained on PPS-116 → Evaluated on PartImageNet

| Method | PPS-116 obj | PPS-116 part | PPS-116 AP | +INS+PART AP |
|--------|-------------|--------------|------------|--------------|
| PSALM† | 31.6 | 8.27 | 13.4 | 21.9 |
| PartGLEE | 38.4 | 9.20 | 15.6 | 21.0 |
| **LangHOPS** | **44.5** | **8.86** | **16.7** | **26.7** |

LangHOPS gains +10.0 AP upon adding part-level data, far exceeding PartGLEE (+5.4) and PSALM (+8.5).

### In-Domain Experiment: Trained and Evaluated on PartImageNet

| Method | obj mAP | part mAP | AP |
|--------|---------|----------|----|
| PSALM† | 79.2 | 40.1 | 48.7 |
| PartGLEE | 81.4 | 41.5 | 50.4 |
| **LangHOPS** | **83.9** | **49.2** | **56.9** |

LangHOPS surpasses PartGLEE by 6.5% AP in-domain, with a 7.7% improvement in part mAP.

### Zero-Shot Semantic Segmentation

| Method | PPS-116 hIoU | PartImageNet hIoU | ADE20K hIoU |
|--------|-------------|-------------------|-------------|
| PartCLIPSeg | 38.8 | 53.9 | 38.6 |
| PartGLEE | 37.1 | — | 41.8 |
| PartCATSeg | 50.4 | 72.7 | **50.0** |
| **LangHOPS** | **52.1** | **72.8** | 49.5 |

LangHOPS achieves the best hIoU on PPS-116 and PartImageNet, and matches PartCATSeg—designed specifically for semantic segmentation—on ADE20K.

### Ablation Study

| Setting | PartImageNet AP | PPS-116 AP |
|---------|-----------------|------------|
| w/o MLLM (Q-Former) | 23.2 | 18.4 |
| w/o hierarchy | 22.5 | 19.1 |
| **LangHOPS** | **26.7** | **19.8** |

### Key Findings

1. **Data scalability**: LangHOPS achieves the largest gain (+10.0 AP) upon incorporating part-level annotations, whereas PartGLEE exhibits part mAP regression (+5.9→+5.4), indicating that more data is not always beneficial without hierarchical semantic context.
2. **Cross-dataset generalization advantage**: The PartImageNet→PPS-116 direction (more novel categories and finer parts) is more challenging for all methods, yet LangHOPS maintains a consistent advantage.
3. **MLLM reasoning capability**: The MLLM contributes 3.5% AP over Q-Former, serving not merely as a feature extractor but as a semantic reasoner.
4. **Bidirectional gain from synergistic training**: Part segmentation optimization reciprocally improves object segmentation (+0.4% obj mAP).

## Highlights & Insights

1. **Language-space hierarchical modeling**: The first work to migrate object-part hierarchical relationships from visual to language space, naturally leveraging semantic priors.
2. **MLLM-driven part parsing**: Employs MLLM world knowledge to reason about the parts an object should possess, rather than relying purely on visual grouping.
3. **Synergistic training mechanism**: Achieves bidirectional object-part gains through gradient back-propagation—an elegant multi-task design.
4. **Strong cross-dataset generalization**: Consistently outperforms competitors across multiple cross-dataset settings, validating the role of language priors in generalizing to unseen categories.

## Limitations & Future Work

1. **High computational cost**: MLLM integration significantly increases inference overhead, hindering real-time or edge deployment.
2. **Training data constraints**: Primary datasets cover common object/part categories; specialized domains (e.g., industrial components, medical imaging) may require additional fine-tuning.
3. **Extension to 3D**: The current framework is limited to 2D image segmentation; combining it with 2D-to-3D lifting for robotics and other 3D applications is an important future direction.
4. Exploring lightweight language models as MLLM substitutes to reduce computational overhead.
5. Integrating SAM's prompt-based segmentation capability to further enhance part mask quality.

## Related Work & Insights

- Contrasted with VLPart and PartGLEE: the former uses text-guided detection heads; the latter adopts a unified architecture but lacks semantic hierarchy. LangHOPS contributes the missing semantic reasoning dimension.
- The application of MLLMs in segmentation tasks is emerging (e.g., PSALM, LISA); LangHOPS opens the more fine-grained **part-level** direction within this trend.
- The language-space hierarchical modeling paradigm is generalizable to other hierarchical visual tasks (e.g., scene graph parsing, relation reasoning).

## Rating

- Novelty: ⭐⭐⭐⭐ Language-space hierarchical modeling combined with MLLM-based part parsing; a novel research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three evaluation settings (in-domain / cross-dataset / zero-shot) with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation and well-designed experiments.
- Value: ⭐⭐⭐⭐ Pioneers MLLM application for part-level segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [\[NeurIPS 2025\] COS3D: Collaborative Open-Vocabulary 3D Segmentation](cos3d_collaborative_open-vocabulary_3d_segmentation.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](../../CVPR2026/segmentation/pca_seg_cost_aggregation_open_vocabulary_segmentation.md)

</div>

<!-- RELATED:END -->
