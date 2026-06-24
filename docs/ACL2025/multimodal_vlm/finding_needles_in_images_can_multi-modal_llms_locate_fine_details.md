---
title: >-
  [Paper Note] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?
description: >-
  [ACL 2025][Multimodal VLM][Fine-grained document understanding] This paper proposes the NiM benchmark dataset to systematically evaluate the capability of Multimodal Large Language Models (MLLMs) to locate fine-grained information in complex documents, and designs the Spot-IT method to significantly improve model performance in detail extraction tasks through intelligent patch selection and a Gaussian attention mechanism.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Fine-grained document understanding"
  - "Multimodal Large Language Models"
  - "Patch selection"
  - "Gaussian attention"
  - "Benchmark"
date: 2026-05-08
content_hash: 9c32b58fcb2cbb9a
---

# Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?

**Conference**: ACL 2025  
**arXiv**: [2508.05053](https://arxiv.org/abs/2508.05053)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Fine-grained document understanding, Multimodal Large Language Models, Patch selection, Gaussian attention, Benchmark

## TL;DR

This paper proposes the NiM benchmark dataset to systematically evaluate the capability of Multimodal Large Language Models (MLLMs) to locate fine-grained information in complex documents, and designs the Spot-IT method to significantly improve model performance in detail extraction tasks through intelligent patch selection and a Gaussian attention mechanism.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made positive progress in document understanding tasks, enabling success in common tasks such as document Q&A, table understanding, and layout analysis. Models like GPT-4V and Gemini continuously refresh the SOTA on standard document understanding benchmarks.

**Limitations of Prior Work**: Existing evaluations mostly focus on global document understanding capabilities, while neglecting a critical scenario—locating and reasoning over fine-grained information in complex documents. Examples include finding the specific nutritional details of a certain dish in a restaurant menu, or identifying a disclaimer in a long newspaper article. These tasks require the model to accurately locate small but crucial details, akin to "finding a needle in a haystack" within a large amount of information.

**Key Challenge**: MLLMs typically take the entire document image as input, but resolution limitations and attention dispersion make it difficult for models to focus on localized, minute information in the document. Existing models perform outstandingly in global understanding, yet exhibit insufficient capabilities in fine-grained tasks that require "zooming in to find details," and there is a lack of a dedicated benchmark to quantify this gap.

**Goal**: (1) To construct a benchmark, NiM, specifically designed to evaluate the fine-grained document localization capabilities of MLLMs; (2) to propose Spot-IT, a method inspired by human reading behavior to enhance the detail localization capabilities of models.

**Key Insight**: When reading complex documents, humans naturally perform a "zoom-and-focus" process—first scanning the global layout, and then zooming in on regions of interest to inspect details. Existing MLLMs lack such a hierarchical attention mechanism.

**Core Idea**: Emulate the human "zoom-and-focus" reading strategy. By intelligently selecting relevant image patches and applying Gaussian attention weighting, the MLLM can prioritize local details while preserving global understanding.

## Method

### Overall Architecture

The core mechanism of Spot-IT is to preprocess the document image before feeding it into the MLLM: first, the document image is partitioned into multiple patches; then, the most relevant subset of patches is intelligently selected based on the query content; finally, Gaussian attention is applied to the selected patches to highlight focus areas. The pipeline is: input document image + query $\to$ patch partitioning $\to$ relevance evaluation & patch selection $\to$ Gaussian attention weighting $\to$ input to MLLM for answering.

### Key Designs

1. **NiM Benchmark Dataset Construction**:

    - **Function**: Provides standardized evaluation for fine-grained document understanding.
    - **Mechanism**: Meticulously collects document images covering various real-world scenarios such as newspapers, menus, and handouts, and designs Q&A pairs for each image requiring the localization of minute details. The dataset covers multiple document types and layout complexities to ensure evaluating comprehensiveness. The questions are designed such that the model must accurately locate a specific region in the document to answer correctly.
    - **Design Motivation**: Existing document understanding benchmarks cannot effectively evaluate "needle-in-a-haystack" fine-grained localization capabilities, necessitating a specialized benchmark to expose model weaknesses.

2. **Intelligent Patch Selection**:

    - **Function**: Automatically identifies local regions most relevant to the query from the document image.
    - **Mechanism**: Partitions the document image into a regular grid of patches, computes the relevance score between each patch and the query text using a vision-language matching model (such as CLIP), and selects the top-$K$ scoring patches as the focused regions. This step simulates the human "scanning" process to identify regions of interest.
    - **Design Motivation**: Directly inputting high-resolution full-images to MLLMs incurs excessive computational overhead and disperses attention. Filtering before reading in detail substantially improves both efficiency and accuracy.

3. **Gaussian Attention**:

    - **Function**: Applies spatial attention weights on the selected patches to further focus on key regions.
    - **Mechanism**: Uses the center of the selected patch as the mean to construct a 2D Gaussian distribution for spatial attention weighting, where pixels closer to the center receive higher weights. This soft attention mechanism prevents the loss of contextual information caused by hard cropping—information at the patch boundaries does not disappear entirely but is down-weighted. Weighted image patches exhibit a natural "center-focused, edge-faded" effect.
    - **Design Motivation**: Mimics the foveal characteristics of human vision—the center of attention is clearest while the periphery is gradually blurred, retaining local context while highlighting key points.

### Loss & Training

Spot-IT is an training-free, inference-time augmentation method that does not involve extra loss functions or training processes. It directly acts on the input side of existing MLLMs, improving performance by optimizing input quality, highlighting its plug-and-play nature.

## Key Experimental Results

### Main Results

Evaluating several mainstream MLLMs on the NiM benchmark, the Spot-IT method brings significant improvements across all models:

| Model | Baseline Accuracy | +Spot-IT | Gain |
|------|-----------|----------|---------|
| GPT-4V | 52.3% | 61.8% | +9.5% |
| Gemini Pro Vision | 45.7% | 54.2% | +8.5% |
| LLaVA-1.5 | 38.1% | 47.6% | +9.5% |
| InternVL | 41.5% | 50.3% | +8.8% |
| Qwen-VL | 39.6% | 48.9% | +9.3% |

Performance differences across different document types:

| Document Type | Baseline Avg. | +Spot-IT Avg. | Gain |
|---------|---------|-------------|------|
| Newspaper/News | 40.2% | 50.5% | +10.3% |
| Menu | 43.8% | 52.1% | +8.3% |
| Handout/Slide | 47.1% | 55.7% | +8.6% |
| Table-dense | 36.5% | 47.2% | +10.7% |

### Ablation Study

| Configuration | Accuracy | Description |
|------|-------|------|
| Full Spot-IT | 61.8% | Full model (patch selection + Gaussian attention) |
| w/o Gaussian Attention | 57.2% | Patch selection only, no attention weighting, drops by 4.6% |
| w/o Patch Selection | 55.1% | Random patch selection + Gaussian attention, drops by 6.7% |
| Uniform Attention | 56.8% | Patch selection + uniform weighting (non-Gaussian), drops by 5.0% |
| Full-image Input | 52.3% | Baseline, without any preprocessing |

### Key Findings

- The patch selection module contributes the most (dropping by 6.7% when removed), indicating that "finding the correct region" is the key bottleneck in fine-grained understanding.
- Gaussian attention exhibits the most prominent effect on table-dense documents, due to high information density and the need for precise focusing in tables.
- All evaluated MLLMs underperform significantly on the NiM benchmark compared to conventional document understanding benchmarks, validating that fine-grained localization is indeed a major weakness of current models.
- The performance improvements of Spot-IT are more pronounced on documents with more complex layouts, aligning with its design goal of "zoom-and-focus".

## Highlights & Insights

- **Inference-Time Augmentation Without Training**: Spot-IT does not require fine-tuning any model parameters and acts as a plug-and-play preprocessing module, allowing it to adapt to any MLLM with zero cost. The philosophy of "improving input rather than modifying the model" is highly practical for real-world deployment.
- **Human-Cognition-Inspired Design**: Gaussian attention simulates the foveal characteristics of human vision. This design paradigm, borrowed from cognitive science, carries more biological plausibility than purely engineering-driven methods and yields easier interpretability of its effectiveness.
- **NiM Benchmark Exposing Model Blind Spots**: Through this specialized benchmark, it is revealed that even GPT-4V achieves only about 50% accuracy on fine-grained localization, uncovering a significantly overlooked capability flaw in existing MLLMs.

## Limitations & Future Work

- Currently, the NiM benchmark only covers English documents; evaluation of fine-grained understanding in multi-lingual documents needs to be extended.
- Patch selection depends heavily on the matching quality of pre-trained models like CLIP, which may fail when the query content is highly abstract.
- The Gaussian attention window size is fixed, offering limited adaptability to detail info across different scales. Future work could consider multi-scale or adaptive mechanisms.
- Spot-IT increases the computational cost during inference (requiring an additional patch selection step), demanding efficiency optimizations for real-time scenarios.
- The core idea of Spot-IT can be extended to video document understanding and cross-page information localization in multi-page documents.

## Related Work & Insights

- **vs TextMonkey**: TextMonkey employs a sliding-window strategy to handle high-resolution documents. In contrast, the patch selection in this work is more intelligent—adaptively selecting based on query content rather than scanning blindly. Spot-IT is more efficient but depends on the quality of the patch selector.
- **vs DocPedia**: DocPedia compresses high-resolution input via frequency-domain transformation, pursuing an entirely different approach by addressing the resolution bottleneck through signal processing. These two methods are complementary and could be combined.
- **vs UReader**: UReader utilizes a unified visual encoder to handle various documents, focusing on the model's structural architecture. Spot-IT, on the other hand, optimizes from the input end without altering the model architecture, allowing both paradigms to be superimposed.

## Rating

- Novelty: ⭐⭐⭐⭐ The NiM benchmark fills the gap in fine-grained document understanding evaluation; the design of Spot-IT is clever, but technical novelty is relatively limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models and document types with clear ablation studies, though comparisons with more baselines could be enriched.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, vivid and accessible human-cognition analogies, and overall smooth organization.
- Value: ⭐⭐⭐⭐ Points out a neglected capability bottleneck in MLLMs; both the NiM benchmark and the Spot-IT method offer practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can MLLMs Understand the Deep Implication Behind Chinese Images?](can_mllms_understand_the_deep_implication_behind_chinese_images.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[ICLR 2026\] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](../../ICLR2026/multimodal_vlm/modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)

</div>

<!-- RELATED:END -->
