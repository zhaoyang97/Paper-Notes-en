---
title: >-
  [Paper Note] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] The EMMA benchmark is proposed to systematically evaluate concept erasure methods for T2I models across five dimensions (erasing ability, retaining ability, efficiency, quality, and bias) with 12 metrics. Covering 206 concept categories across 5 domains, it reveals for the first time the shallow erasure nature and bias amplification issues of existing methods under implicit prompts.
tags:
  - CVPR 2026
  - Image Generation
  - Concept Erasure
  - Text-to-Image Generation
  - Benchmarking
  - Implicit Prompts
  - Bias Evaluation
date: 2026-05-08
content_hash: d8173a1864781d17
---

# EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories

**Conference**: CVPR 2026  
**arXiv**: [2512.17320](https://arxiv.org/abs/2512.17320)  
**Code**: [https://github.com/lobsterlulu/EMMA](https://github.com/lobsterlulu/EMMA)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Concept Erasure, Text-to-Image Generation, Benchmarking, Implicit Prompts, Bias Evaluation

## TL;DR

The EMMA benchmark is proposed to systematically evaluate concept erasure methods for T2I models across five dimensions (erasing ability, retaining ability, efficiency, quality, and bias) with 12 metrics. Covering 206 concept categories across 5 domains, it reveals for the first time the shallow erasure nature and bias amplification issues of existing methods under implicit prompts.

## Background & Motivation

**Background**: The widespread adoption of Text-to-Image (T2I) generation models (e.g., Stable Diffusion) has raised privacy, bias, and copyright concerns. Concept Erasure has been proposed as a lightweight solution that selectively removes unwanted concepts (e.g., specific celebrity faces, NSFW content, copyrighted brands) from pre-trained models without requiring full retraining.

**Limitations of Prior Work**: Current evaluations of concept erasure methods suffer from significant deficiencies: (1) narrow concept coverage, with most methods tested on only 5–20 concepts; (2) single-dimensional evaluation, primarily using direct prompts containing the explicit name of the target concept (e.g., "a photo of a dog") to verify erasure; (3) lack of assessment regarding bias impact. Consequently, it remains unknown whether these methods truly delete the target concept from the model's representation or merely sever the surface-level association between the concept name and the generated output.

**Key Challenge**: Existing evaluations only detect "whether the target concept can be generated using its name," while ignoring "whether the erasure can be bypassed using indirect descriptions (e.g., descriptive text)." Experiments show that even when a concept name is successfully erased, the model can still generate the concept via descriptive prompts, indicating that the semantic representation of the concept has not been truly removed.

**Goal**: To construct a comprehensive evaluation benchmark for concept erasure to answer a key question: Are current evaluation methods detecting if a concept is truly removed from the model representation, or merely detecting its surface-level hiding?

**Key Insight**: Design multi-level, multi-granularity test prompts (from explicit names to implicit descriptions), combined with retention tests of visually similar concepts and social bias analysis, to build a comprehensive evaluation system with 5 dimensions and 12 metrics.

**Core Idea**: Use implicit prompts (descriptive prompts without using the concept name) to test if a concept is truly erased, while simultaneously evaluating the collateral damage of erasure on similar concepts and its impact on gender/racial bias.

## Method

### Overall Architecture

EMMA consists of two core components: (1) **Concept and Category System**—covering 206 concept categories across 5 domains (Objects, Celebrities, Art Styles, NSFW, Copyright); (2) **Evaluation Dimensions and Metric System**—12 metrics across 5 dimensions (Erasing Ability, Retaining Ability, Efficiency, Quality, Bias). For each concept $\times$ metric combination, EMMA constructs a specific set of test prompts and detects the presence of the concept in the generated images using domain-specific classifiers.

### Key Designs

1. **Multi-level Erasing Ability (EA) Evaluation**:

    - Function: Test whether concept erasure methods truly delete the target concept from the model representation.
    - Mechanism: Design 5 metrics that progress from explicit to implicit: (a) **Name**—direct use of the concept name (e.g., "a photo of a dog"); (b) **Prefix**—adding modifiers before the name (e.g., "cutedog"); (c) **Variant**—using synonyms and aliases (e.g., "kitten" instead of "cat"); (d) **Short**—using short descriptions instead of the name (e.g., "a loyal companion with a wagging tail"); (e) **Long**—using detailed descriptions. The EA score is calculated as $S_{EA} = N_{SE} / N_P$, the ratio of successfully erased images to the total number of prompts.
    - Design Motivation: If an erasure method only severs the "concept name $\rightarrow$ generation" mapping without deleting the semantic representation, the model can still generate the erased concept via descriptive prompts. Incremental testing from explicit to implicit reveals the true depth of the erasure.

2. **Retaining Ability (RA) Test for Visually Similar Concepts**:

    - Function: Test whether concept erasure causes collateral damage to non-target concepts, especially visually similar ones.
    - Mechanism: Evaluation via two metrics—(a) **Random**—randomly selecting non-target concepts from the same domain to verify normal generation; (b) **Similar**—selecting the 5 most similar concepts to the erased one (determined by ChatGPT) to verify if erasure affects them. The RA score is the ratio of successfully retained images to the total prompts.
    - Design Motivation: A good erasure method should remove only the target concept without affecting others. However, whether "motorcycles" can still be generated normally after erasing "bicycles" is the real challenge. Previous evaluations often used completely unrelated concepts for retention tests (e.g., testing "airplane" after erasing "cat"), which is too simple.

3. **Bias Evaluation**:

    - Function: Quantify the impact of concept erasure on the model's gender and racial biases.
    - Mechanism: Construct neutral prompts (e.g., "a person") and attribute-specific prompts (e.g., "a man"/"a woman"/"a Black person"), then compare which attribute group the images generated by neutral prompts more closely "resemble" before and after erasure. CLIP and SSIM are used to calculate the similarity difference between neutral images and each attribute group: $B_a^M = \frac{1}{N}\sum_i [f_{sim}(I_n^M(i), I_{ref}^M(i)) - f_{sim}(I_n^M(i), I_a^M(i))]$.
    - Design Motivation: Concept erasure may unintentionally shift the model's bias tendencies in person generation. If erasing a concept causes neutral prompts to lean toward generating specific genders or races, the method introduces bias. This is critical for practical deployment.

### Loss & Training

EMMA is an evaluation benchmark and does not involve training. The 5 evaluated concept erasure methods fall into two categories: (1) **Concept Remapping** (MACE, ESD, UCE), which modifies cross-attention weights to map target concepts to substitute concepts; (2) **Optimization Methods** (CA, FMN), which use iterative fine-tuning to make the model ignore the target concept.

## Key Experimental Results

### Main Results (Object Domain, partial data)

| Method | Name EA↑ | Long EA↑ | Random RA↑ | Similar RA↑ | FID↓ |
|--------|------|------|----------|------|------|
| SD v1.4 (Original) | 5.7 | 23.2 | 94.4 | 94.5 | 42.85 |
| +CA | 16.5 | 21.2 | 94.4 | 94.2 | 45.04 |
| +ESD | 89.7 | 61.7 | 87.8 | 74.6 | 34.81 |
| +UCE | 82.6 | 73.8 | 93.4 | 86.0 | 34.64 |
| +MACE | 98.6 | 70.5 | 91.6 | 79.4 | 43.90 |

### Performance in the Celebrity Domain

| Method | Name EA↑ | Short EA↑ | Long EA↑ | Random RA↑ | Similar RA↑ |
|------|---------|------|------|------|------|
| +CA | 71.0 | 90.7 | 89.2 | 89.6 | 87.2 |
| +ESD | 96.8 | 99.2 | 97.9 | 78.1 | 68.6 |
| +UCE | 99.7 | 99.7 | 99.7 | 86.6 | 81.4 |
| +MACE | 97.3 | 97.9 | 97.6 | 89.9 | 89.5 |

### Key Findings

- **Concept remapping methods consistently outperform optimization methods**: ESD, UCE, and MACE lead CA and FMN significantly in both EA and RA. FMN's erasure effect in the Object domain is nearly identical to the original unerased model.
- **Erasure effectiveness drops significantly under implicit prompts**: MACE's EA in the Object domain drops from 98.6% (Name) to 70.5% (Long), indicating that "erased" concepts resurface under descriptive prompts.
- **Retaining similar concepts is more difficult**: All methods perform significantly worse at retaining visually similar concepts compared to random concepts; for instance, MACE's RA in the Object domain drops from 91.6% (Random) to 79.4% (Similar).
- **Inference efficiency comes at a high cost**: The inference time for all methods increases by 2–10 times compared to the original model.
- **ESD consistently amplifies gender and racial bias**, while FMN is the only method that mitigates bias (possibly because its base model, SD 2.1, is inherently more biased).

## Highlights & Insights

- **Introduction of implicit prompt testing pioneeringly reveals the shallow nature of concept erasure**: This is the most significant contribution of the paper. It proves that existing methods merely cut the "name $\rightarrow$ generation" mapping rather than deleting semantic representations, posing a serious challenge to safety claims.
- **Retention testing of visually similar concepts**: This is more challenging and practically meaningful than random concept retention tests. Being unable to generate a "motorcycle" after erasing a "bicycle" is unacceptable in real-world applications.
- **Systematic introduction of bias evaluation**: This work quantifies the impact of concept erasure on gender/racial bias for the first time, providing a necessary evaluation framework for safe and compliant deployment.

## Limitations & Future Work

- Selection of visually similar concepts relies on ChatGPT's judgment, which might miss truly challenging similar concepts.
- The root cause of bias amplification remains unclear—why parameter-efficient erasure methods alter the model's bias tendencies in person generation.
- Currently only SD v1.4/v2.1 are evaluated; newer models like SDXL and FLUX are not covered.
- Future work should explore stronger concept erasure methods that can truly delete semantic representations rather than just severing name mappings.

## Related Work & Insights

- **vs UnlearnCanvas**: While UnlearnCanvas focuses on art style erasure, EMMA covers 5 domains and introduces implicit prompt testing and bias evaluation.
- **vs HUB**: HUB covers some evaluation dimensions but lacks implicit prompts and bias analysis; EMMA significantly exceeds it in evaluation comprehensiveness.
- **vs Ring-A-Bell**: Ring-A-Bell focuses on adversarial bypassing but is limited to the NSFW domain; EMMA's implicit prompt testing is more systematic.
- The "shallow erasure" problem revealed here has important implications for AI safety: fine-tuning-based concept erasure may not be secure enough.

## Rating

- Novelty: ⭐⭐⭐⭐ Implicit prompt testing and bias evaluation are major contributions, though the technical innovation of the benchmark itself is relatively incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive: 5 domains $\times$ 5 dimensions $\times$ 12 metrics $\times$ 5 methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, substantial data, and deep analysis.
- Value: ⭐⭐⭐⭐⭐ Serves as a landmark for evaluation standards in the concept erasure field, revealing critical safety issues.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)

</div>

<!-- RELATED:END -->
