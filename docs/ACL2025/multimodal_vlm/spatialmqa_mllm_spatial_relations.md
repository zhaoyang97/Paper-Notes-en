---
title: >-
  [Paper Note] Can Multimodal Large Language Models Understand Spatial Relations?
description: >-
  [Multimodal VLM] Proposes the SpatialMQA benchmark to evaluate the spatial relation reasoning capability of MLLMs in a multiple-choice format, revealing that the state-of-the-art model only achieves 48.14% accuracy, far below the human performance of 98.40%.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 82ea9dccae1b1466
---

# Can Multimodal Large Language Models Understand Spatial Relations?

- **Conference**: ACL 2025
- **arXiv**: [2505.19015](https://arxiv.org/abs/2505.19015)
- **Code**: [GitHub](https://github.com/ziyan-xiaoyu/SpatialMQA)
- **Area**: Multimodal VLM
- **Keywords**: Spatial relations, multimodal benchmarks, MLLM evaluation, SpatialMQA, perspective shifts

## TL;DR

Proposes the SpatialMQA benchmark to evaluate the spatial relation reasoning capability of MLLMs in a multiple-choice format, revealing that the state-of-the-art model only achieves 48.14% accuracy, far below the human performance of 98.40%.

## Background & Motivation

- **Core Problem**: Existing spatial relation reasoning benchmarks suffer from issues such as relying on bounding boxes, ignoring perspective shifts, and being answerable solely through prior knowledge, failing to truly evaluate MLLMs' understanding of spatial relations in images.
- **Limitations of Prior Work**:
    - **Dependence on bounding boxes**: SpatialVOC2K, Rel3D, SpatialSense+, etc., require bounding box annotations for subjects and objects, but certain entities (such as the "sun") cannot be bounded by bounding boxes.
    - **Annotations not based on the physical world**: SpatialSense annotates "the sky is behind the forest" as "behind", which is inconsistent with human cognitive understanding.
    - **Neglect of perspective shifts**: For instance, only 6% of samples in VSR use a first-person perspective, lacking the capability to evaluate complex scenarios (such as autonomous driving).
    - **Answerable using prior knowledge**: For example, "the book is above the bus" can be answered as "No" solely based on common sense, without needing to understand the image.
- **Design Motivation**: To construct a high-quality, human-annotated benchmark that forces models to understand images to answer, while covering various perspective shift scenarios.

## Method

### Overall Architecture

SpatialMQA is a multiple-choice spatial relation reasoning benchmark based on COCO2017, containing 5,392 samples, 128 subject/object types, and 6 spatial relations (left of / right of / in front of / behind / on/above / below). The task is formulated as: given an image $I$ and a question $Q$, select the correct spatial relation from $k \ (k=2,...,6)$ options.

### Key Designs

1. **Spatial Coordinate System with the Physical World as Reference**: Setting gravity as down, the observer as the origin, the X-axis from left to right, the Y-axis from back to front, and the Z-axis from down to up, ensuring annotations align with human intuitive cognition.
2. **Perspective Shift Mechanism**: Questions are classified into two categories: perspective outside the image (third-party observer) and perspective inside the image (first-person/third-person), where the perspective-inside-image questions account for 60%, requiring models to understand spatial relations under different observer viewpoints.
3. **Three-Round Annotation Quality Control**: First-round annotation $\rightarrow$ Second-round checking (whether answers can be guessed via prior knowledge + clarity of subject/object) $\rightarrow$ Third-round auditing (20% random checks by the lead author), with passing rate thresholds (90%/95%) set for each round.

### Loss & Training

Open-source models are fine-tuned using standard cross-entropy loss, with LoRA applied for parameter-efficient fine-tuning.

## Experiments

### Main Results

| Model | Setting | Acc (%) |
|------|------|---------|
| **SpaceLLaVA** | LoRA | **48.14** |
| LLaVA1.5-7B | LoRA | 46.85 |
| InstructBLIP-3B | LoRA | 42.38 |
| GPT-4o | 0-shot | 40.20 |
| Gemini-1.5-flash | 3-shot | 38.00 |
| BLIP-vqa-base | Full | 33.64 |
| Random Choose | - | 27.20 |
| **Human** | - | **98.40** |

### Ablation Study

| Model | Q1 (Outside) | Q2 (First-Person) | Q3 (Third-Person) | Ax (Left-Right) | Ay (Front-Back) | Az (Up-Down) |
|------|----------|-------------|-------------|----------|----------|----------|
| SpaceLLaVA (LoRA) | 54.87 | 42.37 | 58.82 | 56.00 | 51.85 | 31.41 |
| GPT-4o (0-shot) | 44.09 | 33.74 | 61.76 | 37.08 | 47.50 | 36.00 |
| LLaVA1.5-7B (LoRA) | 53.14 | 40.99 | 64.71 | 55.71 | 29.64 | 48.13 |

### Key Findings

1. **Huge Gap between MLLMs and Humans**: SOTA model (SpaceLLaVA LoRA) at 48.14% vs Human at 98.40%, representing a gap of over 50 percentage points.
2. **Perspective Shift is the Main Difficulty**: The first-person perspective inside the image (Q2) generally yields the lowest accuracy (e.g., SpaceLLaVA is only 42.37%), showing that models struggle with perspective-shifting reasoning.
3. **LoRA Fine-tuning Significantly Boosts Performance**: After instruction tuning, SpaceLLaVA improves from 31.32% to 48.14%, and LLaVA improves from 29.28% to 46.85%.
4. **Unanswerable with Text Only**: Accuracy when providing only text without images is only 24.40%, which is lower than random guess, verifying that the benchmark strictly requires image understanding.
5. **Few-shot may not outperform Zero-shot for GPT-4o**: GPT-4o 0-shot (40.20%) outperforms 3-shot (37.80%), possibly because in-context learning (ICL) examples introduced distraction.

## Highlights & Insights

- The first benchmark for spatial relation reasoning that systematically excludes the interference of prior knowledge and covers both first-person and third-person perspective shifts.
- The rigorous design of the three-round human annotation quality control process ensures that every sample must be evaluated by looking at the image.
- Reveals the huge deficiency of current MLLMs in understanding spatial relations (a gap of over 50 percentage points), pointing out clear directions for future research.
- Provides fine-grained analysis based on perspective types and spatial dimensions, pinpointing the exact weak spots of models.
- The benchmark and code are fully open-sourced, with detailed and reproducible annotation guidelines.

## Limitations & Future Work

- Spatial relations only cover 6 basic types, neglecting more complex spatial descriptions (such as "next to", "between", or "surrounding").
- Based on COCO2017 images, the diversity of scenes is limited by the coverage of this dataset.
- Only evaluates a limited number of closed-source models (GPT-4o and Gemini), failing to cover more of the latest models (e.g., Claude, Qwen-VL).
- The number of third-person perspective samples is small (only 185), which may affect the statistical reliability of the evaluation in this category.
- The distribution of the number of options (2/4/6) is unbalanced (75% are 4-option), leading to insufficient analysis on the difficulty variations across different option sizes.
- Does not explore how the ambiguity of spatial relations (such as boundary cases between "on" and "above") affects annotation consistency.

## Related Work & Insights

- **Spatial Relation Benchmarks**: SpatialVOC2K (Belz et al. 2018), SpatialSense (Yang et al. 2019), Rel3D (Goyal et al. 2020), VSR (Liu et al. 2023a), EmbSpatial (Du et al. 2024), SpatialRGPT (Cheng et al. 2024).
- **MLLMs**: GPT-4o (Achiam et al. 2023), Gemini-1.5-flash, LLaVA (Liu et al. 2024), SpaceLLaVA (Chen et al. 2024), BLIP/BLIP2/InstructBLIP series.
- **Spatial Reasoning Enhancement**: SpaceLLaVA improves understanding via spatial relation instruction tuning; SpatialVLM (Chen et al. 2024) evaluates spatial awareness through open-ended QA.
- **Benchmark Design Methodology**: COCO2017 (Lin et al. 2014) provides multi-entity scene images; the three-round annotation process references strict quality control paradigms.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — For the first time systematically addresses perspective shifts and prior-knowledge bias in spatial relation benchmarks.
- **Value**: ⭐⭐⭐⭐ — Provides the community with a high-quality evaluation tool, exposing the explicit shortcomings of current models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The three-round annotation quality control process is highly solid.
- **Overall Rating**: ⭐⭐⭐⭐

<!-- SpatialMQA: human-annotated, perspective-aware spatial relation reasoning benchmark for MLLMs -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
