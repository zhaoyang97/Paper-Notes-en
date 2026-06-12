---
title: >-
  [Paper Note] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching
description: >-
  [ICCV 2025][Information Retrieval & RAG][image-text matching] This paper proposes D2S-VSE, a two-stage training framework that addresses the information density asymmetry in image-text matching. In the first stage…
tags:
  - "ICCV 2025"
  - "Information Retrieval & RAG"
  - "image-text matching"
  - "visual semantic embedding"
  - "dense-to-sparse distillation"
  - "information capacity"
  - "cross-modal retrieval"
date: 2026-05-08
content_hash: 4891bdee7bb8e0b4
---

# Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching

**Conference**: ICCV 2025
**arXiv**: N/A  
**Code**: [Project Page](https://d2s-vse.github.io)  
**Area**: Image-Text Matching / Visual Semantic Embedding
**Keywords**: image-text matching, visual semantic embedding, dense-to-sparse distillation, information capacity, cross-modal retrieval

## TL;DR

This paper proposes D2S-VSE, a two-stage training framework that addresses the information density asymmetry in image-text matching. In the first stage, the model is pre-trained on LLaVA-generated dense captions to enhance information capacity; in the second stage, dense text embeddings are distilled into sparse text embeddings. The method achieves state-of-the-art performance on MS-COCO and Flickr30K.

## Background & Motivation

- **Background**: Conventional visual semantic embedding (VSE) models learn a joint embedding space for cross-modal retrieval.
- **Limitations of Prior Work**: Existing methods overlook the information density gap between modalities: images carry rich visual details (high information capacity), whereas dataset captions are typically short (low information capacity). Methods that learn multiple embeddings to cover diverse perspectives remain limited in per-embedding information capacity and are susceptible to false negatives with partial semantic overlap.
- **Key Challenge**: The asymmetric information capacity between visual and textual representations degrades retrieval quality.
- **Goal**: Learn visual semantic embeddings with higher information capacity for both modalities.

## Method

### Overall Architecture

D2S-VSE consists of three stages: (a) **Dense caption generation** — LLaVA is used to generate detailed descriptions for dataset images; (b) **Pre-training** — contrastive learning on image–dense caption pairs to enhance the information capacity of image embeddings; (c) **Fine-tuning** — jointly optimizing image–sparse text alignment and dense-to-sparse text embedding distillation. At inference time, only the image and sparse text are required, introducing no additional computational overhead.

### Key Designs

1. **Dense Caption Pre-training for Enhanced Information Capacity**: LLaVA generates comprehensive descriptions capturing all visual details. Contrastive learning aligns images with dense captions, encouraging the image encoder to extract more thorough features and produce higher-capacity embeddings. The pre-trained image encoder weights are frozen during fine-tuning.

2. **Dense-to-Sparse Feature Distillation**: A Transformer decoder module is appended after the text encoder. It combines sparse text features with learnable mask tokens to predict the latent context implicit in the sparse text. This is formulated as a masked signal reconstruction task, guiding the sparse text embedding to internalize information present in the dense text embedding. The distillation objective and image–text contrastive alignment are optimized jointly.

3. **Zero-overhead Inference Design**: At inference time, only standard image and sparse text inputs are needed; dense captions are used exclusively during training. The distilled sparse text embeddings already encode higher information capacity, enabling a single visual embedding to match descriptions from diverse perspectives.

### Loss & Training

The pre-training stage employs a contrastive loss to align image–dense caption pairs. The fine-tuning stage jointly optimizes: (1) an image–sparse text contrastive loss, and (2) a dense-to-sparse distillation loss (masked reconstruction).

## Key Experimental Results

### Main Results

| Method | Flickr30K R@1 | MS-COCO R@1 |
|--------|--------------|-------------|
| Prev. SOTA | Baseline | Baseline |
| **D2S-VSE** | **Exceeds SOTA** | **Exceeds SOTA** |

D2S-VSE surpasses the latest state-of-the-art methods across multiple backbone architectures on both Flickr30K and MS-COCO benchmarks.

### Ablation Study

- Dense caption pre-training vs. no pre-training: pre-training substantially improves information capacity.
- Distillation branch vs. no distillation: distillation further enhances sparse text embedding quality.
- Varying dense caption length: more detailed descriptions yield higher information capacity.
- Zero inference overhead confirmed.

### Key Findings

- Information capacity is a critical factor in visual semantic embedding quality.
- Knowledge from dense captions can be effectively distilled into sparse text embeddings.
- Pre-training and distillation are complementary — the former enhances image embeddings, the latter enhances text embeddings.

## Highlights & Insights

- The introduction of "information capacity" as a unifying concept to explain the core challenge of image-text matching offers a novel and compelling perspective.
- Leveraging LLaVA-generated dense captions as free training supervision is a practical and scalable strategy.
- Zero inference overhead — distilled sparse text embeddings implicitly encode dense information without requiring dense inputs at test time.
- The two-stage design is clean and straightforward.

## Limitations & Future Work

- The method depends on the quality of LLaVA-generated dense captions; generation errors may propagate into training.
- Distillation effectiveness is sensitive to the semantic gap between dense and sparse descriptions.
- Validation is limited to image-text retrieval; generalization to other tasks such as VQA has not been explored.
- The Transformer decoder introduces additional computational cost during training.

## Related Work & Insights

- Methods addressing information density, such as A-VSE and PCME++, serve as direct baselines for comparison.
- Works leveraging long-form text, including DreamLip and Long-CLIP, provide relevant reference points.
- The distillation framework is potentially extensible to other cross-modal alignment tasks, such as video-text matching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The information capacity perspective and the dense-to-sparse distillation approach are both original.
- **Technical Depth**: ⭐⭐⭐ — The core methodology (contrastive learning + distillation) is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two major benchmarks with multiple backbones and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation figures are clear and the framework is described fluently.
- **Value**: ⭐⭐⭐⭐ — Zero inference overhead makes the method easy to integrate into existing systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICCV 2025\] MonSTeR: a Unified Model for Motion, Scene, Text Retrieval](monster_a_unified_model_for_motion_scene_text_retrieval.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ICCV 2025\] OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)

</div>

<!-- RELATED:END -->
