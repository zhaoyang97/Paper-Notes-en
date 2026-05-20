---
title: >-
  [Paper Note] Panoptic Captioning: An Equivalence Bridge for Image and Text
description: >-
  [NeurIPS 2025][Segmentation][panoptic captioning] This paper proposes the novel task of Panoptic Captioning, which pursues a *minimum text equivalence* of images—defining a comprehensive structured description along five…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "panoptic captioning"
  - "minimum text equivalence"
  - "PancapScore"
  - "PancapChain"
  - "grounding"
date: 2026-05-08
content_hash: 6abeed277a537b71
---

# Panoptic Captioning: An Equivalence Bridge for Image and Text

**Conference**: NeurIPS 2025
**arXiv**: [2505.16334](https://arxiv.org/abs/2505.16334)  
**Code**: [https://visual-ai.github.io/pancap/](https://visual-ai.github.io/pancap/)  
**Area**: Image Segmentation
**Keywords**: panoptic captioning, minimum text equivalence, PancapScore, PancapChain, grounding

## TL;DR

This paper proposes the novel task of Panoptic Captioning, which pursues a *minimum text equivalence* of images—defining a comprehensive structured description along five dimensions (entity semantic tags, locations via bounding boxes, attributes, relations, and global state)—and introduces the PancapEngine data engine and PancapChain decoupled multi-stage method. A 13B model trained under this framework surpasses InternVL-2.5-78B and GPT-4o.

## Background & Motivation

Representing images with text is a fundamental problem in CV/NLP, yet the most effective format remains an open question:

- **Short captions** (e.g., BLIP-2): lose critical details such as entity attributes and spatial locations.
- **Detailed captions** (e.g., ShareGPT4V): describe locations in free-form text, resulting in verbosity and imprecision.
- **Dense captioning**: generates brief descriptions per region but disregards inter-entity relations.

The core objective is to find the **minimum text equivalence** of an image—capturing all semantic elements as completely as possible with minimal text. Conceptually, this corresponds to aligning images and text in data space, as opposed to embedding space alignment performed by CLIP.

## Method

### Overall Architecture

The paper makes three main contributions: (1) a five-dimensional task definition with the PancapScore evaluation metric; (2) the PancapEngine data engine; and (3) the PancapChain decoupled generation method.

### Key Designs

1. **Five-Dimensional Task Definition**:

    - **Function**: Groups the semantic content of a panoptic caption into five dimensions.
    - **Mechanism**: Semantic Tag (entity category labels) + Location (bounding box coordinates) + Attribute (appearance/state/material) + Relation (spatial/action/part relations between entities) + Global State (lighting/tone/scene style).
    - **Design Motivation**: Bounding box coordinates provide precise localization with only a few numbers, outperforming free-form textual location descriptions. The five-dimensional decomposition ensures both completeness and evaluability.

2. **PancapEngine Data Engine (detect-then-caption)**:

    - **Function**: Automatically generates high-quality panoptic caption data.
    - **Mechanism**: Entity Detection Suite (OLN class-agnostic detection + RAM 6400+ category label assignment + Grounding-DINO/OW-DETR supplementary detection) → Entity-Aware Caption Generation (Gemini-Exp-1121 generation + Qwen2-VL-72B cross-validation for consistency).
    - **Design Motivation**: Conventional detectors are constrained to fixed categories (e.g., 80 COCO classes); the OLN+RAM combination effectively removes this category ceiling.

3. **PancapChain Decoupled Generation Method**:

    - **Function**: Decomposes panoptic captioning into sequential sub-tasks across multiple stages.
    - **Mechanism**: Stage 1: entity localization (bounding boxes) → Stage 2: semantic tag assignment → Stage 3: supplementary entity discovery → Stage 4: comprehensive panoptic caption generation.
    - **Design Motivation**: Requiring a model to generate a complete panoptic caption end-to-end is highly challenging, as it must simultaneously localize, classify, and describe all entities. Decoupling allows each stage to focus on a specific sub-task.

### Loss & Training

PancapChain is trained via SFT across multiple stages. The SA-Pancap benchmark comprises 9,000 training images + 500 validation images (automatically generated captions) + 130 test images (manually annotated captions). PancapScore is defined as: entity matching (tag F1 + location F1) + instance-aware QA (precision/recall/F1 for attributes, relations, and global state).

## Key Experimental Results

### Main Results

| Model | Parameters | Overall PancapScore | Tagging F1 | Location F1 | Attribute F1 | Relation F1 |
|-------|-----------|--------------------:|----------:|---:|---:|---:|
| InternVL-2.5-78B | 78B | 154.66 | - | - | - | - |
| GPT-4o | - | 148.01 | - | - | - | - |
| Gemini-2.0-Pro | - | 157.88 | - | - | - | - |
| **PancapChain-13B** | **13B** | **173.19** | **56.45** | **31.76** | **44.46** | **32.54** |

The 13B model surpasses the 78B open-source model and commercial large models across all dimensions, demonstrating that data quality and method design matter more than model scale.

### Ablation Study

- PancapChain 4-stage decoupled generation vs. direct generation: decoupling yields 6.5%+ improvement in Overall PancapScore.
- Impact of cross-validation in the data engine: removing Qwen validation degrades data quality by ~3%.
- Image retrieval application (DOCCI R@1): PancapChain 61.9 vs. ALIGN 59.9 vs. ShareGPT4V 59.6.

### Key Findings

- **Decoupling is essential**—even with the same 13B backbone, PancapChain's staged generation substantially outperforms end-to-end generation.
- **Location (bounding box prediction) is the greatest bottleneck** for current models—a Location F1 of 31.76 indicates that precise localization remains challenging.
- **Captions generated by PancapChain yield the best text-to-image reconstruction quality**, validating the concept of minimum text equivalence.

## Highlights & Insights

- **A 13B model surpassing 78B+ closed-source models**: a clear victory for data quality and method design.
- **Elegant task definition**: the five-dimensional structured description is both concise (bounding box coordinates require only a few numbers) and complete (covering all semantic elements).
- **PancapScore exhibits high agreement with human judgment**, providing a reliable evaluation metric.
- **Practical value**: text-only retrieval outperforms CLIP-style alignment models (DOCCI R@1: 61.9 vs. 59.9).
- **Conceptual innovation**: advances CLIP's embedding-space alignment to data-space alignment.

## Limitations & Future Work

- The task definition remains an approximation of minimum text equivalence—highly fine-grained details (e.g., ground texture) are not covered.
- Location F1 is only 31.76—bounding box precision is the primary bottleneck.
- Evaluation relies on an LLM judge (Qwen2.5-14B), which may introduce evaluation bias.
- The data engine is bounded by the capabilities of existing detectors and MLLMs.
- Current models already perform reasonably well on Global State; the main room for improvement lies in tagging, location, and relation dimensions.

## Related Work & Insights

- **ShareGPT4V**: provides detailed captioning but describes locations in free-form text, resulting in insufficient spatial precision.
- **GLaMM**: a grounded MLLM that jointly performs captioning and grounding, but requires an additional localization module and produces brief descriptions.
- **Dense Captioning**: generates short per-region descriptions without modeling inter-entity relations.
- **Insight**: panoptic captions can serve as a superior format for multimodal pretraining data, providing more precise spatial information than ShareGPT4V.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — novel task definition, new metric, and new method with high overall completeness.
- Experimental Thoroughness: ⭐⭐⭐⭐ — comparison with multiple state-of-the-art models and downstream application validation.
- Writing Quality: ⭐⭐⭐⭐ — task definition and method pipeline are clearly presented.
- Value: ⭐⭐⭐⭐⭐ — defines a new paradigm for image captioning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[NeurIPS 2025\] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning](srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](../../ICCV2025/segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[ICLR 2026\] VIRTUE: Visual-Interactive Text-Image Universal Embedder](../../ICLR2026/segmentation/virtue_visual-interactive_text-image_universal_embedder.md)
- [\[NeurIPS 2025\] ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](argenseg_image_segmentation_with_autoregressive_image_generation_model.md)

</div>

<!-- RELATED:END -->
