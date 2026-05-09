---
title: >-
  [Paper Note] Relational Visual Similarity
description: >-
  [CVPR 2026][Multimodal VLM][Relational Similarity] This paper formally defines the problem of relational visual similarity — the intrinsic relational or functional correspondence between two images, as opposed to surface-level attribute similarity — constructs a 114K anonymous-description dataset, trains the relsim model, and reveals fundamental deficiencies in existing similarity metrics (CLIP, DINO, etc.) for capturing relational similarity.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Relational Similarity
  - Visual Analogy
  - Anonymous Descriptions
  - Cognitive Science
  - Image Retrieval
date: 2026-05-08
content_hash: e6cad573d8ac5e2c
---

# Relational Visual Similarity

**Conference**: CVPR 2026
**arXiv**: [2512.07833](https://arxiv.org/abs/2512.07833)
**Code**: [https://thaoshibe.github.io/relsim](https://thaoshibe.github.io/relsim)
**Area**: Multimodal VLM
**Keywords**: Relational Similarity, Visual Analogy, Anonymous Descriptions, Cognitive Science, Image Retrieval

## TL;DR
This paper formally defines the problem of relational visual similarity — the intrinsic relational or functional correspondence between two images, as opposed to surface-level attribute similarity — constructs a 114K anonymous-description dataset, trains the relsim model, and reveals fundamental deficiencies in existing similarity metrics (CLIP, DINO, etc.) for capturing relational similarity.

## Background & Motivation
1. **State of the Field**: Visual similarity is a foundational capability in computer vision. Existing methods (LPIPS, CLIP, DINO, etc.) focus on attribute similarity — matching at the pixel, semantic, or descriptive level.
2. **Limitations of Prior Work**: These methods fail to recognize relational similarity — for example, the stages of a burning match and the ripening stages of a banana share the same "temporal gradual change" logic, yet are entirely dissimilar in terms of attributes.
3. **Root Cause**: Cognitive science regards attribute similarity and relational similarity as two core pillars of human perception, yet visual computing has entirely neglected the latter. Relational similarity is considered a key cognitive ability that distinguishes humans from other species.
4. **Paper Goals**: To formalize relational visual similarity as a measurable problem and to construct models capable of capturing relational structure.
5. **Starting Point**: Inspired by cognitive science — humans identify relational similarity through conceptual abstraction mediated by language or prior knowledge. Accordingly, the paper introduces "anonymous descriptions" (describing intrinsic logic rather than concrete objects) as the bridge linking relationally similar images.
6. **Core Idea**: Define anonymous descriptions (e.g., "the change of {subject} over time"), train a model to generate such descriptions, and use them to bring images sharing the same relational logic closer together in representation space.

## Method

### Overall Architecture
A three-stage pipeline: (1) filter 114K images from LAION-2B that likely contain transferable relational structures; (2) train an anonymous-description generation model to produce an anonymous description for each image; (3) train the relsim model on {image, anonymous description} pairs, optimizing to bring closer the representations of images whose descriptions encode similar relational abstractions.

### Key Designs

1. **Data Filtering and Curation**:
    - Function: Extract images containing transferable relational structures from large-scale image corpora.
    - Mechanism: Filter out low-quality, mislabeled, and relationally uninformative images from LAION-2B, retaining images that likely exhibit relational patterns such as temporal sequences, structural analogies, and functional correspondences.
    - Design Motivation: A large proportion of images in LAION-2B are relationally irrelevant (e.g., product photos, selfies); direct use would introduce substantial noise.

2. **Anonymous Description Model**:
    - Function: Generate text that describes the intrinsic relational logic of an image rather than its concrete content.
    - Mechanism: A dedicated captioning model is trained to take an image as input and output an anonymous description — one that refers to no specific visible object but instead captures the relational logic conveyed by the image. For example, the anonymous description for an image of a burning match is "transformation of {subject} over time" rather than "burning matchsticks."
    - Design Motivation: Anonymous descriptions serve as "glue" connecting images with similar intrinsic logic. This is the key step that operationalizes the cognitive science insight that relational similarity requires conceptual abstraction.

3. **relsim Relational Similarity Model**:
    - Function: Learn to bring images sharing the same relational structure closer together in representation space.
    - Mechanism: A vision-language model is fine-tuned on the {image, anonymous description} dataset, with the optimization objective bringing closer the features of images whose anonymous descriptions encode similar relational abstractions.
    - Design Motivation: Standard vision-language contrastive learning (e.g., CLIP) optimizes the matching of images to their concrete descriptions, naturally biasing toward attribute similarity. Replacing concrete descriptions with anonymous descriptions shifts the optimization objective from attribute alignment to relational alignment.

### Loss & Training
Standard vision-language contrastive learning loss, with anonymous descriptions substituted for conventional captions.

## Key Experimental Results

### Main Results

| Model | Attribute Similarity | Relational Similarity | Notes |
|-------|---------------------|-----------------------|-------|
| CLIP | High | Low | Attributes only |
| DINO | High | Low | Attributes only |
| LPIPS | High | Very Low | Pixel-level |
| relsim | Medium-High | High | Relation-aware |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full relsim | Best | Trained with anonymous descriptions |
| Conventional description training | Poor relational similarity | Attribute descriptions cannot encode relations |
| w/o data filtering | Degraded | Noisy data interference |

### Key Findings
- All mainstream visual similarity metrics perform poorly on relational similarity, revealing a critical blind spot in visual computing.
- Anonymous descriptions serve as an effective intermediary for linking relationally similar images.
- relsim demonstrates practical value in applications such as relational image retrieval and analogical image generation.

## Highlights & Insights
- **Opens a fundamentally new dimension of visual understanding**: The shift from attributes to relations constitutes a conceptual breakthrough.
- **Anonymous descriptions** represent an elegant concept: removing concrete objects while retaining abstract logic.
- The interdisciplinary integration of cognitive science and computer vision merits attention.

## Limitations & Future Work
- Evaluation of relational similarity inherently lacks clear ground truth and remains highly subjective.
- The quality of anonymous description generation still has room for improvement.
- Future work may explore broader applications in reasoning and creative generation.

## Related Work & Insights
- **vs. CLIP/DINO**: These methods focus on semantic matching at the attribute level and cannot capture relational similarity. relsim fills this gap through anonymous-description training.
- **vs. NIGHTS**: Focuses on mid-level perceptual similarity, which remains attribute-driven.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel problem formulation with a unique interdisciplinary perspective
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparison and application demonstrations, though large-scale user studies are lacking
- Writing Quality: ⭐⭐⭐⭐⭐ Compelling narrative enriched by cognitive science background
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental blind spot in visual AI with far-reaching implications

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[AAAI 2026\] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations](../../AAAI2026/multimodal_vlm/the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)

<!-- RELATED:END -->
