---
title: >-
  [Paper Note] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation
description: >-
  [CVPR 2026][LLM Evaluation][Video quality assessment] VGA-Bench proposes a unified AIGC video evaluation benchmark comprising a three-tier taxonomy (aesthetic quality, aesthetic labels, and generation quality), 1,016 prompts, 60,000 videos, and three dedicated evaluation models, enabling automated assessment aligned with human judgment.
tags:
  - CVPR 2026
  - LLM Evaluation
  - Video quality assessment
  - aesthetic evaluation
  - AIGC evaluation
  - multi-task evaluator
  - video generation
date: 2026-05-08
content_hash: 5fc62552e36a3180
---

# VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation

**Conference**: CVPR 2026
**arXiv**: [2604.10127](https://arxiv.org/abs/2604.10127)
**Code**: Available
**Area**: Image/Video Generation Evaluation
**Keywords**: Video quality assessment, aesthetic evaluation, AIGC evaluation, multi-task evaluator, video generation

## TL;DR

VGA-Bench proposes a unified AIGC video evaluation benchmark comprising a three-tier taxonomy (aesthetic quality, aesthetic labels, and generation quality), 1,016 prompts, 60,000 videos, and three dedicated evaluation models, enabling automated assessment aligned with human judgment.

## Background & Motivation

**Background**: AIGC video generation has advanced rapidly (diffusion models, Transformers, etc.), yet evaluation frameworks remain focused on technical fidelity metrics (FVD, CLIP Score), overlooking high-level perceptual qualities such as aesthetic appeal.

**Limitations of Prior Work**: Benchmarks such as V-Bench reduce "video aesthetics" to a single score, heavily relying on external scoring models (MUSIQ/DINO), resulting in insufficient granularity, significant bias, and limited controllability.

**Key Challenge**: Video generation models grow increasingly powerful, yet a comprehensive, fine-grained, and interpretable evaluation framework capable of jointly measuring technical quality and aesthetic quality is lacking.

**Goal**: Establish a three-dimensional unified evaluation framework covering generation quality, aesthetic quality, and visual formal elements.

**Key Insight**: Design a hierarchical taxonomy that decomposes each dimension into fine-grained sub-attributes (composition, color harmony, lighting, motion aesthetics, etc.) and train dedicated evaluation models accordingly.

**Core Idea**: Replace the ad hoc combination of external scoring models with three dedicated neural evaluators, achieving end-to-end, consistent, and scalable automated evaluation.

## Method

### Overall Architecture

A three-tier taxonomy: Aesthetic Quality (composition, color, lighting, motion aesthetics, etc.) + Aesthetic Labels (visual formal elements such as style and scene type) + Generation Quality (temporal consistency, prompt alignment, distortion, etc.). 1,016 prompts → 12 video generation models → 60,000 videos → manually annotated subset → training of three evaluators: VAQA-Net, VTag-Net, and VGQA-Net.

### Key Designs

1. **Three-Tier Taxonomy Evaluation Framework**:

    - Function: Enables systematic and comprehensive video assessment.
    - Mechanism: Decomposes evaluation into three dimensions — Aesthetic Quality (overall aesthetics and fine-grained attributes such as composition and color harmony), Aesthetic Labels (automatic tagging of visual formal elements such as style and scene type), and Generation Quality (technical fidelity including temporal consistency and artifact detection).
    - Design Motivation: V-Bench provides only 1 aesthetic dimension across 16 total dimensions; VGA-Bench substantially expands both the granularity and coverage of evaluation.

2. **Three Dedicated Multi-Task Evaluation Models**:

    - Function: Eliminates dependence on external scoring models.
    - Mechanism: VAQA-Net predicts aesthetic quality scores; VTag-Net performs automatic aesthetic label tagging; VGQA-Net assesses generation and basic quality attributes. All three are trained on human annotations to achieve alignment with human judgment.
    - Design Motivation: External models (e.g., MUSIQ) are not designed for AIGC video and introduce systematic bias.

3. **Large-Scale Diverse Prompt Suite**:

    - Function: Ensures breadth and challenge of evaluation coverage.
    - Mechanism: 1,016 diverse prompts are designed to cover a wide range of scenes, actions, styles, and challenging scenarios. Each of the 12 state-of-the-art video generation models generates approximately 5,000 videos, yielding 60,000 videos in total.
    - Design Motivation: Sufficiently diverse and large-scale test data is necessary for fair cross-model comparison.

### Loss & Training

The three evaluation models are trained separately on human-annotated data. Within a multi-task learning framework, each model handles multiple sub-attributes within its respective dimension.

## Key Experimental Results

### Main Results

| Evaluation Model | Human Alignment | Dimensions Covered |
|---|---|---|
| VAQA-Net | High alignment | Multi-dimensional aesthetic quality |
| VTag-Net | High accuracy | Automated aesthetic labeling |
| VGQA-Net | High alignment | Multi-dimensional generation quality |

### Ablation Study

| Dimension | V-Bench | VGA-Bench |
|---|---|---|
| Total dimensions | 16 | Substantially expanded |
| Aesthetic dimensions | 1 | Multiple fine-grained dimensions |
| Evaluated models | 4 | 12 |
| Number of prompts | ~1,600 | 1,016 (curated) |

### Key Findings

- Dedicated evaluation models significantly outperform general-purpose external models in aligning with human judgment.
- Different video generation models exhibit clear divergence in aesthetic and technical quality.
- Aesthetic quality and generation quality are not always positively correlated — some models achieve high technical fidelity but poor aesthetic performance.

## Highlights & Insights

- **From Technical Fidelity to Aesthetic Intelligence**: VGA-Bench elevates AIGC evaluation from "does it look real?" to "does it look beautiful?"
- **Value of Evaluation Infrastructure**: 60,000 videos, human annotations, and three evaluation models together constitute a complete evaluation ecosystem.
- **Fully Open-Source Commitment**: Includes the taxonomy, prompt templates, annotation data, API, and video dataset.

## Limitations & Future Work

- Aesthetic evaluation is inherently subjective, and human annotations may carry bias.
- The 1,016 curated prompts, while carefully selected, still offer limited coverage.
- Evaluation models may require continuous updates as video generation technology advances.

## Related Work & Insights

- **vs. V-Bench**: V-Bench represents a systematic first attempt but oversimplifies the aesthetic dimension (a single score); VGA-Bench substantially expands upon it.
- **vs. FVD/CLIP Score**: Traditional metrics measure only technical fidelity, whereas VGA-Bench covers both aesthetic and generation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Fine-grained aesthetic quality taxonomy and dedicated evaluators
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models × 60,000 videos × human annotation
- Writing Quality: ⭐⭐⭐⭐ Well-structured and comprehensive framework
- Value: ⭐⭐⭐⭐ Significant contribution to AIGC evaluation infrastructure

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](../../ICLR2026/llm_evaluation/can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)
- [\[ICLR 2026\] Soft Quality-Diversity Optimization](../../ICLR2026/llm_evaluation/soft_quality-diversity_optimization.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)

<!-- RELATED:END -->
