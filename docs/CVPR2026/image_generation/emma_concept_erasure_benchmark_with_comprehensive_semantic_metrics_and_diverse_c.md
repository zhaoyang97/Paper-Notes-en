---
title: >-
  [Paper Note] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] The EMMA benchmark is proposed to systematically evaluate concept erasure methods for T2I models across five dimensions (erasing ability, retaining ability, efficiency, quality, and bias) with 12 metrics. Covering 206 concept categories across 5 domains, it reveals for the first time the "shallow erasure" nature of existing methods under implicit prompts and the issue of bias amplification.
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Concept Erasure"
  - "Text-to-Image Generation"
  - "Benchmark"
  - "Implicit Prompt"
  - "Bias Evaluation"
date: 2026-05-08
content_hash: 4ec20b1d4d9612a0
---

# EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories

**Conference**: CVPR 2026  
**arXiv**: [2512.17320](https://arxiv.org/abs/2512.17320)  
**Code**: [https://github.com/lobsterlulu/EMMA](https://github.com/lobsterlulu/EMMA)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Concept Erasure, Text-to-Image Generation, Benchmark, Implicit Prompt, Bias Evaluation

## TL;DR

The EMMA benchmark is proposed to systematically evaluate concept erasure methods for T2I models across five dimensions (erasing ability, retaining ability, efficiency, quality, and bias) with 12 metrics. Covering 206 concept categories across 5 domains, it reveals for the first time the "shallow erasure" nature of existing methods under implicit prompts and the issue of bias amplification.

## Background & Motivation

**Background**: The widespread adoption of Text-to-Image (T2I) generation models (e.g., Stable Diffusion) has raised privacy, bias, and copyright concerns. Concept Erasure has been proposed as a lightweight solution that selectively removes unwanted concepts (e.g., specific celebrity faces, NSFW content, copyrighted brands) from pre-trained models without requiring full retraining.

**Limitations of Prior Work**: Evaluation of current concept erasure methods suffers from significant deficiencies: (1) narrow concept coverage, with most methods tested on only 5–20 concepts; (2) single-dimensional evaluation, primarily using direct prompts containing the explicit target concept name (e.g., "a photo of a dog"); (3) lack of assessment regarding bias impact. Consequently, it remains unclear whether these methods truly delete target concepts from model representations or merely sever the surface-level association between concept names and generated results.

**Key Challenge**: Current evaluations only detect "whether the target concept can be generated using its name," while ignoring "whether the erasure can be bypassed using indirect descriptions." Experiments demonstrate that even when a concept name is successfully erased, the model can still generate the concept via descriptive prompts, indicating that the underlying semantic representation has not been truly removed.

**Goal**: To construct a comprehensive concept erasure evaluation benchmark to answer a critical question: Are current evaluation methods detecting true removal of concepts from model representations, or merely surface-level hiding?

**Key Insight**: Designing multi-level, multi-granular test prompts (ranging from explicit names to implicit descriptions), combined with retention tests for visually similar concepts and social bias analysis, to build a comprehensive evaluation system with 5 dimensions and 12 metrics.

**Core Idea**: Use implicit prompts (descriptive prompts that do not use the concept name) to test whether concepts are truly erased, while simultaneously evaluating collateral damage to similar concepts and the impact on gender/racial bias.

## Method

### Overall Architecture

EMMA addresses a question often ignored by previous evaluations: whether concept erasure deletes the target concept from the model representation or merely severs the "concept name → generation" mapping. To do this, it consists of two components. The first is a **Concept and Category System** covering 206 categories across five domains (Object, Celebrity, Art Style, NSFW, and Copyright), which is significantly larger than the 5–20 concepts typically used. The second is an **Evaluation Dimension and Metric System**, which decomposes five dimensions (erasing ability, retaining ability, efficiency, quality, and bias) into 12 metrics. For each "concept × metric" combination, EMMA generates test prompts based on the metric design, generates images using the evaluated model, utilizes domain-specific classifiers to determine if the target concept is present, and converts hit rates into scores. The core contribution lies in **how prompts are constructed**, which determines whether shallow mappings or deep representations are being tested.

### Key Designs

**1. Multilevel Erasing Ability (EA): Progressive Testing from Explicit Names to Implicit Descriptions**

Previous evaluations relied almost exclusively on direct prompts with concept names (e.g., "a photo of a dog"), only verifying name-based recall. However, if an erasure method only severs the mapping without altering semantic representations, the model can bypass the name using descriptions. EA categorizes test prompts into five levels of increasing difficulty: *Name* (direct name), *Prefix* (name with modifiers like "cute dog"), *Variant* (synonyms or aliases like "kitten" for cat), *Short* (a brief description like "a loyal companion with a wagging tail"), and *Long* (detailed paragraph description). The score is calculated as the ratio of successfully erased images to total prompts: $S_{EA} = N_{SE} / N_P$. A high score in the *Name* tier but a low score in the *Long* tier reveals that erasure is restricted to the name level and does not reach the underlying semantics.

**2. Retaining Ability (RA) for Visually Similar Concepts: Testing Collateral Damage via Nearest Neighbors**

Effective erasure should delete the target concept without affecting others. Prior evaluations often used irrelevant concepts for retention tests (e.g., testing "airplane" after erasing "cat"), which fails to identify issues as unrelated concepts do not share representations. RA splits retention testing into two difficulty levels: *Random* (randomly selected non-target concepts from the same domain) and *Similar* (the top 5 visually similar neighbors suggested by ChatGPT). Both are scored by the ratio of successfully retained images. Testing whether "motorcycles" can still be generated after erasing "bicycles" provides a more discriminative test of precision.

**3. Social Bias Analysis: Quantifying Perturbations in Gender and Racial Tendencies**

Parameter-efficient erasure modifies weights, which may unintentionally shift model bias in person generation—a factor rarely quantified. This is measured by creating both neutral prompts (e.g., "a person") and attribute-specific prompts (e.g., "a man," "a woman," "a Black person"). The bias is measured by whether the neutral prompt outputs "move" closer to specific attribute groups after erasure. CLIP and SSIM are used to calculate the similarity difference between neutral images, attribute groups, and reference images:

$$B_a^M = \frac{1}{N}\sum_i \left[ f_{sim}(I_n^M(i), I_{ref}^M(i)) - f_{sim}(I_n^M(i), I_a^M(i)) \right]$$

If neutral prompts lean significantly toward a specific gender or race after erasure, it indicates that the erasure process introduced bias—a critical side effect for real-world deployment.

### Loss & Training

EMMA is an evaluation benchmark and does not involve training. The five concept erasure methods evaluated are categorized into two types: (1) **Concept Remapping** (MACE, ESD, UCE), which modifies cross-attention weights to map target concepts to surrogates, and (2) **Optimization Methods** (CA, FMN), which employ iterative fine-tuning to make the model ignore the target concept.

## Key Experimental Results

### Main Results (Object Domain, Selected Data)

| Method | Name EA↑ | Long EA↑ | Random RA↑ | Similar RA↑ | FID↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SD v1.4 (Original) | 5.7 | 23.2 | 94.4 | 94.5 | 42.85 |
| +CA | 16.5 | 21.2 | 94.4 | 94.2 | 45.04 |
| +ESD | 89.7 | 61.7 | 87.8 | 74.6 | 34.81 |
| +UCE | 82.6 | 73.8 | 93.4 | 86.0 | 34.64 |
| +MACE | 98.6 | 70.5 | 91.6 | 79.4 | 43.90 |

### Performance in Celebrity Domain

| Method | Name EA↑ | Short EA↑ | Long EA↑ | Random RA↑ | Similar RA↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| +CA | 71.0 | 90.7 | 89.2 | 89.6 | 87.2 |
| +ESD | 96.8 | 99.2 | 97.9 | 78.1 | 68.6 |
| +UCE | 99.7 | 99.7 | 99.7 | 86.6 | 81.4 |
| +MACE | 97.3 | 97.9 | 97.6 | 89.9 | 89.5 |

### Key Findings

- **Concept remapping methods outperform optimization methods**: ESD, UCE, and MACE significantly outperform CA and FMN in both EA and RA. In the object domain, FMN's erasing effect is nearly identical to the original unerased model.
- **Erasure effectiveness drops significantly under implicit prompts**: MACE's EA in the object domain drops from 98.6% for *Name* to 70.5% for *Long*, proving that "erased" concepts reappear under descriptive prompts.
- **Retaining similar concepts is more difficult**: All methods performed significantly worse at retaining visually similar concepts compared to random concepts. For example, MACE's RA in the object domain fell from 91.6% (*Random*) to 79.4% (*Similar*).
- **Inference efficiency comes at a cost**: All methods increased inference time by 2–10x compared to the original model.
- **ESD consistently amplifies gender and racial bias**, while FMN is the only method that mitigates bias (possibly because its base model, SD 2.1, is inherently more biased).

## Highlights & Insights

- **Introduction of implicit prompt testing reveals the shallow nature of concept erasure**: This is the primary contribution of the paper. It proves that current methods only sever "name → generation" mappings rather than deleting semantic representations, posing a serious challenge to safety claims.
- **Retention testing for visually similar concepts**: This provides a more challenging and practical assessment than random concepts. In real-world applications, it is unacceptable for erasing "bicycles" to prevent the generation of "motorcycles."
- **Systematic introduction of bias evaluation**: This is the first work to quantify how concept erasure affects gender/racial bias, providing a necessary framework for safe and compliant deployment.

## Limitations & Future Work

- Selection of visually similar concepts relies on ChatGPT's judgment, which might miss truly challenging similar concepts.
- The fundamental cause of bias amplification remains unclear—specifically, why parameter-efficient erasure shifts model bias in person generation.
- The study only evaluates SD v1.4/v2.1 and does not cover newer generations like SDXL or FLUX.
- Future work should explore more robust concept erasure methods that can truly delete semantic representations rather than just cutting name mappings.

## Related Work & Insights

- **vs. UnlearnCanvas**: UnlearnCanvas focuses on art style erasure, while EMMA covers 5 domains and introduces implicit prompt testing and bias evaluation.
- **vs. HUB**: HUB covers some evaluation dimensions but lacks implicit prompts and bias analysis; EMMA is significantly more comprehensive.
- **vs. Ring-A-Bell**: Ring-A-Bell focuses on adversarial bypasses but is limited to the NSFW domain; EMMA's implicit prompt testing is more systematic.
- The "shallow erasure" problem highlighted in this work offers critical insights for AI safety: fine-tuning-based concept erasure may not be sufficiently secure.

## Rating

- Novelty: ⭐⭐⭐⭐ Implicit prompt testing and bias evaluation are major contributions, though the technical innovation of the benchmark itself is relatively incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive: 5 domains × 5 dimensions × 12 metrics × 5 methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, robust data, and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Sets a new standard for the concept erasure field and exposes significant security vulnerabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MapRoute: Semantic Routing for Precise Concept Erasure with Mapper](maproute_semantic_routing_concept_erasure.md)
- [\[CVPR 2026\] I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models](i2i-bench_a_comprehensive_benchmark_suite_for_image-to-image_editing_models.md)
- [\[CVPR 2026\] Closed-Form Concept Erasure via Double Projections](closed-form_concept_erasure_via_double_projections.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)

</div>

<!-- RELATED:END -->
