---
title: >-
  [Paper Note] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
description: >-
  [NeurIPS 2025][Segmentation][3D part segmentation] This paper presents PartNeXt, a fine-grained hierarchical part annotation dataset comprising 23,519 high-quality textured 3D models across 50 categories. Two benchmarks are established—category-agnostic part segmentation and 3D part question answering—revealing significant deficiencies of current methods in fine-grained part understanding.
tags:
  - NeurIPS 2025
  - Segmentation
  - 3D part segmentation
  - dataset
  - hierarchical annotation
  - crowdsourced annotation
  - 3D vision-language models
date: 2026-05-08
content_hash: 1df14771298cc658
---

# PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2510.20155](https://arxiv.org/abs/2510.20155)
**Code**: [GitHub](https://authoritywang.github.io/partnext)
**Area**: Image Segmentation
**Keywords**: 3D part segmentation, dataset, hierarchical annotation, crowdsourced annotation, 3D vision-language models

## TL;DR

This paper presents PartNeXt, a fine-grained hierarchical part annotation dataset comprising 23,519 high-quality textured 3D models across 50 categories. Two benchmarks are established—category-agnostic part segmentation and 3D part question answering—revealing significant deficiencies of current methods in fine-grained part understanding.

## Background & Motivation

**Background**: The PartNet dataset has advanced 3D part-level understanding by providing 573K part annotations across 26K models and 24 categories.

**Limitations of Prior Work**:
- PartNet's annotation pipeline requires remeshing, which causes texture loss and geometric distortion in some objects, limiting the use of visual cues.
- PartNet's annotation interface demands 3D modeling expertise (manually drawing curves to cut meshes and inspecting cross-sections), making it unsuitable for crowdsourced scaling.
- Existing datasets have limited category coverage (24 categories) and predominantly consist of untextured geometry.

**Key Challenge**: A fundamental tension exists between the demand for high-quality fine-grained 3D part annotations and the need for scalable, accessible annotation pipelines.

**Goal**: To construct a high-quality, scalable next-generation 3D part annotation dataset alongside new evaluation benchmarks.

**Key Insight**: Design a fully web-based crowdsourcing annotation interface combined with AI-assisted hierarchy definition and direct annotation on textured meshes.

**Core Idea**: Enable large-scale, fine-grained textured 3D part annotation through annotation tooling innovation and AI assistance.

## Method

### Overall Architecture

The construction of PartNeXt involves four stages:
1. Data collection and preprocessing (selecting high-quality models from Objaverse, ABO, and 3D-FUTURE)
2. Hierarchy definition and example generation (GPT-4o-assisted with human review)
3. Annotation system design (web-based crowdsourcing platform)
4. Benchmark establishment (part segmentation + part question answering)

### Key Designs

1. **Data Collection and CLIP Filtering**

    - **Function**: Select high-quality, category-consistent models from three large-scale 3D datasets.
    - **Approach**:
        - ABO and 3D-FUTURE are filtered directly by category.
        - Objaverse is large but variable in quality: animated models, meshes exceeding 130K faces, and scanned/architectural models are filtered out.
        - A CLIP text encoder encodes approximately 100 category names and Cap3D-provided descriptions; models are classified by cosine similarity.
        - Models with a maximum similarity below 0.75 are discarded.
        - The 50 categories with the most remaining models are selected.

2. **AI-Assisted Hierarchy Definition**

    - **Function**: Define fine-grained, consistent part hierarchies for each category.
    - **Motivation**: Manually defining detailed hierarchies—especially enumerating diverse part variants—is time-consuming.
    - **Approach**:
        - Five hierarchy design principles are formulated: functionality-awareness, hierarchicality, exhaustive variants, atomicity, and consistency.
        - GPT-4o generates a coarse hierarchy, refined with rendered images, and reviewed by human experts.
        - GPT-4o's image generation capability produces visual reference examples for each part node.
    - Final hierarchy depths range from 4 to 10 levels.

3. **Web-Based Crowdsourcing Annotation System**

    - **Function**: Provide an efficient and accessible 3D part annotation interface.
    - **Three Core Features**:
        - **Hierarchical annotation workflow**: A collapsible tree structure allows annotators to progressively expand and label leaf nodes, with an "Other" node to handle unexpected parts.
        - **Dual-panel interface**: The left panel displays the unsegmented mesh; the right panel displays segmented results from the same viewpoint. Annotated parts are transferred from left to right, each identified by a unique color. This design is particularly suited for annotating occluded internal parts.
        - **Face selection toolset**:
            - Connected-component selection: clicking automatically selects an entire connected region.
            - Bounding-box selection: a 2D box projection selects all visible faces within.
            - Face-by-face selection: enables fine-grained control.
    - **Distinction from PartNet**: Annotation is performed directly on the original textured mesh without remeshing, preserving texture information.

4. **Annotation Quality Control**

    - 35 professional annotators and 5 senior annotators handle data verification.
    - Annotators complete two days of training.
    - Every annotation undergoes at least one review, resulting in 5,211 corrections.
    - Average annotation time per model is approximately 5–6 minutes.

### Dataset Statistics

| Dimension | Value |
|-----------|-------|
| Total models | 23,519 |
| Total part instances | 350,187 |
| Number of categories | 50 |
| Data sources | Objaverse (14,811) + ABO (2,633) + 3D-FUTURE (6,075) |
| Hierarchy depth | 4–10 |

## Key Experimental Results

### Main Results I: Category-Agnostic 3D Part Instance Segmentation

Evaluation covers 250 objects (50 categories × 5), using leaf nodes as ground truth.

| Method | Bed | Bottle | Chair | Knife | Table | Controller | Fan | Glasses | Monitor | Wrench | mIoU |
|--------|-----|--------|-------|-------|-------|-----------|-----|---------|---------|--------|------|
| SAMPart3D | 17.51 | 47.71 | 28.49 | 61.08 | 25.86 | 24.00 | 31.12 | 28.34 | 25.70 | 40.53 | 36.78 |
| PartField | 24.77 | 67.91 | 43.78 | 68.22 | 53.26 | 41.57 | 46.66 | 55.57 | 45.97 | 60.53 | 50.22 |
| SAMesh | **82.59** | 35.63 | **72.57** | 51.19 | **64.81** | **47.71** | **56.72** | 33.38 | 45.16 | 52.17 | **51.57** |

**Key Finding**: All three state-of-the-art methods perform poorly on PartNeXt (best mIoU of only 51.57%), demonstrating that fine-grained part segmentation remains a significant challenge. Each method exhibits distinct characteristics: SAMesh produces fine-grained but over-segmented results, PartField under-segments connected regions, and SAMPart3D struggles with continuity in low-texture areas.

### Main Results II: 3D Part Question Answering

| Task | Metric | 3DLLM | PointLLM | ShapeLLM |
|------|--------|-------|----------|----------|
| Part Count (with category) | MAE↓ | 2.16 | 1.87 | 1.72 |
| Part Count (without category) | MAE↓ | 2.46 | 1.79 | 1.85 |
| Classification (with category) | Acc↑ | — | 0.22 | 0.25 |
| Classification (without category) | Acc↑ | — | 0.18 | 0.08 |
| Grounding (with category) | IoU↑ | — | — | 0.33 |
| Grounding (without category) | IoU↑ | — | — | 0.30 |

**Key Finding**: Current 3D vision-language models are severely limited in part-level reasoning. Part count MAE approaches 2, classification accuracy is approximately 20%, and grounding IoU is approximately 30%.

### Ablation Study: Point-SAM Training Data Ablation

| Eval Set | Training Set | IoU@1 | IoU@3 | IoU@5 | IoU@7 | IoU@10 |
|----------|-------------|-------|-------|-------|-------|--------|
| PartNet-Mobility | PartNet | 39.0 | 53.7 | 58.6 | 60.9 | 62.9 |
| PartNet-Mobility | **PartNeXt** | 40.2 | 57.5 | 63.2 | 65.0 | 67.4 |
| PartNet-Mobility | Mixture | **40.4** | **58.3** | **64.1** | **66.9** | **68.7** |
| PartNeXt | PartNet | 39.9 | 53.9 | 58.4 | 60.4 | 60.3 |
| PartNeXt | **PartNeXt** | 44.3 | 60.1 | 63.2 | 64.8 | 65.9 |
| PartNeXt | Mixture | **45.3** | **61.7** | **65.3** | **66.6** | **67.6** |

**Key Finding**: Point-SAM trained solely on PartNeXt substantially outperforms the PartNet-trained variant (IoU@10: 67.4 vs. 62.9), confirming the dataset's high quality and diversity.

### Key Findings

- Current state-of-the-art part segmentation methods leave substantial room for improvement in fine-grained hierarchical segmentation (best mIoU of only 51.57%).
- 3D LLMs are severely limited in part-level reasoning and localization; 3D part question answering represents a valuable new research direction.
- Richer training data (PartNeXt) directly yields significant gains for interactive segmentation models.

## Highlights & Insights

- **Systematic annotation tooling innovation**: The combination of a dual-panel interface, three face selection tools, and a hierarchical annotation workflow is highly practical and substantially lowers the barrier to annotation.
- **AI-assisted pipeline**: CLIP filtering, GPT-4o hierarchy definition, and GPT-4o reference image generation together form a complete AI-assisted data construction workflow.
- **Texture-preserving annotation**: Annotating directly on textured meshes avoids the remeshing issues of PartNet and supports downstream tasks that require texture information.
- **Two new benchmarks**: The part segmentation and part question answering benchmarks expose the deficiencies of current methods and point the community toward important open problems.

## Limitations & Future Work

- **Dataset scale** remains limited (23.5K models); future work plans to expand from Objaverse-XL.
- **Predefined hierarchies are required**: Each category requires a carefully designed part hierarchy, which constrains open-vocabulary annotation capability.
- **Absence of semantic description annotations**: The current dataset provides only part name annotations without captions or physical attribute annotations.
- The annotation interface still relies on manual operation; future work may explore semi-automatic annotation assisted by vision-language models.
- Evaluation uses only 5 objects per category, so results may be sensitive to sample selection.

## Related Work & Insights

- **Evolution from PartNet to PartNeXt**: The shift from untextured geometry with expert annotation to textured meshes with crowdsourced annotation reflects a paradigm upgrade in 3D dataset construction.
- **AI-assisted data construction**: The application of GPT-4o to hierarchy definition and reference image generation demonstrates the potential of large models to assist data engineering.
- **Exposed limitations of 3D LLMs**: The inadequacy of ShapeLLM, PointLLM, and others on part-level tasks suggests that fine-grained 3D understanding may be a critical direction for next-generation 3D foundation models.
- **Insight**: Dataset quality and diversity represent the most direct path to improving model generalization.

## Rating

- Novelty: ⭐⭐⭐⭐ The annotation system design is innovative, and the part QA benchmark introduces a new task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks for segmentation and QA are comprehensive; the Point-SAM ablation is convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ Structure is clear, figures are well-crafted, and dataset statistics are thorough.
- Value: ⭐⭐⭐⭐⭐ The high-quality dataset offers long-term community value, and the benchmarks identify important research gaps.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[NeurIPS 2025\] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](partonomy_large_multimodal_models_with_part-level_visual_understanding.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](../../ICCV2025/segmentation/partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)

<!-- RELATED:END -->
