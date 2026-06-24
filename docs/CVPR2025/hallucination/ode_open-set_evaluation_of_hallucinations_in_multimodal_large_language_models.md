---
title: >-
  [Paper Note] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models
description: >-
  [CVPR 2025][Hallucination Detection][Hallucination Evaluation] This paper proposes the ODE (Open-set Dynamic Evaluation) protocol, which models real-world object concepts and their distribution associations using a graph structure. It dynamically extracts concept combinations and generates synthetic test images to realize open-set, continuously updated multimodal hallucination evaluation, effectively avoiding the data contamination issues potentially present in current static…
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Hallucination Evaluation"
  - "Open-set Evaluation"
  - "Data Contamination"
  - "Dynamic Testing"
  - "Text-to-Image Generation"
date: 2026-05-08
content_hash: c279d54c2365c3f7
---

# ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2409.09318](https://arxiv.org/abs/2409.09318)  
**Code**: [https://github.com/Iridescent-y/ODE](https://github.com/Iridescent-y/ODE)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Evaluation, Open-set Evaluation, Data Contamination, Dynamic Testing, Text-to-Image Generation

## TL;DR

This paper proposes the ODE (Open-set Dynamic Evaluation) protocol, which models real-world object concepts and their distribution associations using a graph structure. It dynamically extracts concept combinations and generates synthetic test images to realize open-set, continuously updated multimodal hallucination evaluation, effectively avoiding the data contamination issues potentially present in current static benchmarks.

## Background & Motivation

**Background**: The hallucination problem in Multimodal Large Language Models (MLLMs) has attracted widespread attention. The community has proposed a series of evaluation benchmarks: CHAIR measures object accuracy in descriptions, POPE evaluates object existence discrimination, AMBER evaluates across three dimensions (existence/attributes/relations), and HallusionBench focuses on visual commonsense reasoning. These benchmarks have rapidly advanced research on hallucinations.

**Limitations of Prior Work**: Almost all existing benchmarks are static, using fixed test data (such as the COCO2014 subset) with limited distribution. As the scale of model training data continues to expand, the risk of overlap between test and training data increases. The authors found crucial evidence: under the same semantic distribution, model performance on COCO2014 images is significantly better than on recent internet images (which are much less likely to have been seen during training), hinting that correct answers might stem from data contamination rather than genuine understanding.

**Key Challenge**: Static benchmarks cannot distinguish whether a model "genuinely understands visual content" or merely "remembers test samples seen during training." In the LLM domain, data contamination has been widely discussed (mentioned in GPT-4 and LLaMA reports), but there is still no targeted solution in the multimodal field.

**Goal**: (1) How to generate open-set, unseen test samples to evaluate hallucinations; (2) How to systematically test model robustness at different distribution levels; (3) How to leverage dynamic evaluation data to improve model optimization.

**Key Insight**: If test data is newly generated from scratch (synthetic images + dynamic concept combinations), it is impossible for models to have seen it during training, thereby eliminating data contamination. The key innovation is to model the co-occurrence relationships between concepts using a graph structure and select concept combinations based on different frequency distribution standards.

**Core Idea**: Use a graph structure to model object concept associations and dynamically generate synthetic test samples across different distribution levels to achieve open-set hallucination evaluation.

## Method

### Overall Architecture

The ODE protocol consists of four steps: (1) Graph structure modeling—constructing real-world object concepts, attributes, and their co-occurrence relations into a weighted graph $G=(V, A, E, W)$; (2) Semantic scene construction—selecting concept pairs from the graph according to four distribution standards and assigning attributes; (3) Image generation and filtering—using text-to-image models to generate test images followed by quality control; (4) Query template design—automatically generating evaluation questions targeting existence and attribute hallucinations.

### Key Designs

1. **Graph-structured Concept Modeling**:

    - **Function**: Abstracting real-world scenes into an operable graph structure.
    - **Mechanism**: Extracting 337 object categories from the AMBER benchmark as nodes $V$, classified by scene function into environment-level (e.g., grass) and entity-level (e.g., frisbee) classes. Each node is accompanied by attribute nodes $A$ (state, action, count). Edge weights $W$ are determined by the co-occurrence frequency of two concepts in the dataset, reflecting the strength of semantic association. Concepts are further distinguished into two co-occurrence patterns: entity-environment and entity-entity.
    - **Design Motivation**: The graph structure not only represents the association strength between concepts but also facilitates the extraction of concept combinations according to different distribution standards, supporting dynamic updates and domain expansion.

2. **Four-level Distribution Selection Standards**:

    - **Function**: Systematically testing the hallucination performance of models at different semantic distribution levels.
    - **Mechanism**: (1) **Standard**—selecting concept pairs with the highest co-occurrence frequency $(V_i, V_j) \in \arg\max c_{i, j}$ to test the model's understanding of high-frequency combinations; (2) **Long-tail**—selecting pairs with medium co-occurrence frequency $\epsilon < c_{k,l} < \delta$ to test performance under long-tail distributions; (3) **Random**—uniformly and randomly selecting $(V_i, V_j) \sim \text{Uniform}(V \times V)$ with randomly chosen attributes to test robustness; (4) **Fictional**—selecting pairs with zero registered co-occurrences $c_{k,l} = 0$ to test inference capabilities on entirely new concept combinations.
    - **Design Motivation**: Model performance under different distribution frequencies can vary significantly—high-frequency pairs may rely on memory, while low-frequency/fictional pairs are more likely to expose the true comprehension capability.

3. **Synthetic Image Generation and Quality Control**:

    - **Function**: Generating high-quality, unseen test images for models.
    - **Mechanism**: Generating images using FLUX.1-dev or Stable Diffusion 1.5 based on textual descriptions (e.g., "a picture of a black running dog and a yellow frisbee"). Multiple images are generated per test case using different random seeds, then filtered using an open-vocabulary object detection model—discarding cases where the target entities have detection confidence scores below 0.65. High-quality samples are retained, and all detected concepts serve as the ground truth.
    - **Design Motivation**: Synthetic images eliminate the potential for data contamination at the source. CLIP feature analysis shows that synthetic and natural images are highly similar in the feature space, which validates the feasibility of using synthetic images as substitutes.

### Loss & Training

ODE itself is an evaluation protocol and does not involve training. However, the authors demonstrate that the data generated by ODE can be used for model fine-tuning; targeted fine-tuning on error samples identified by ODE can effectively reduce hallucinations.

## Key Experimental Results

### Main Results (ODE vs. AMBER Static Benchmark Comparison)

| Model | AMBER-Exist F1 | ODE-Standard Exist F1 | AMBER-Attr F1 | ODE-Standard Attr F1 |
|------|---------------|----------------------|--------------|---------------------|
| LLaVA-1.5 | 83.0 | 70.7 | 64.8 | 44.8 |
| CogVLM | 34.5 | 41.5 | 29.7 | 50.8 |
| InstructBLIP | 80.5 | 67.4 | 71.4 | 36.6 |
| MiniGPT-4 | 98.4 | 64.3 | 56.6 | 19.0 |

### Impact of Different Image Generation Models

| Model | ODE-SD Exist Acc | ODE-Flux Exist Acc | Δ |
|------|----------------|--------------------|---|
| LLaVA-1.5 | 94.3 | 51.3 | +43.0 |
| CogVLM | 92.8 | 41.4 | +51.4 |
| MiniGPT-4 | 66.7 | 67.1 | -0.4 |

### Key Findings

- Most models perform significantly worse on ODE-generated samples compared to static benchmarks (e.g., MiniGPT-4's existence F1 drops from 98.4% on AMBER to 64.3% on ODE-Standard), strongly implying the presence of data contamination in static benchmarks.
- The hallucination rate rises noticeably under Random and Fictional distributions, especially in attribute recognition tasks, indicating that models highly depend on co-occurrence patterns learned during training.
- Differences in synthetic image quality produced by different generative models (FLUX vs. SD1.5) lead to massive discrepancies in evaluation results (with CogVLM varying by 51.4 points), introducing a new uncontrollable variable.
- While models perform reasonably well on high-frequency concepts in generative tasks, high-frequency concepts in discriminative tasks may conversely become unstable due to overfitting/over-memorization.

## Highlights & Insights

- **Eliminating data contamination with synthetic images**: The core insight is simple yet profound—if the test images are newly generated, it is impossible for models to have seen them during training. Similarity validation between synthetic and natural images via CLIP features provides a methodological foundation for future work.
- **Design of four-level distribution standards**: The transition from Standard to Fictional provides a continuous spectrum evaluation from "familiar to the model" to "unfamiliar to the model." This reveals performance differences across varying cognitive difficulty levels, offering much more informative feedback than a single-distribution evaluation.
- **Evaluation-optimization closed loop**: ODE is not only used for evaluation; its generated data can also be directly utilized for model fine-tuning to mitigate hallucinations, achieving closed-loop evaluation and improvement.

## Limitations & Future Work

- Synthetic image quality is limited by the text-to-image models; FLUX vs. SD1.5 causes extreme variances in evaluation outcomes (with some models showing a difference of 40+ points), introducing a new uncontrollable variable.
- Currently, it only supports combination scenes with two objects, failing to evaluate more complex multi-object scenarios.
- The concept graph contains only 337 classes (sourced from AMBER), offering limited coverage.
- Filtering with object detection (confidence threshold of 0.65) might be too conservative or imprecise, affecting ground-truth quality.
- Relational hallucinations (spatial or interactive relations between objects) are not evaluated; only existence and attribute hallucinations are covered.

## Related Work & Insights

- **vs POPE**: POPE evaluates existence hallucinations based on fixed COCO images and suffers from data contamination; ODE dynamically generates images to avoid this issue and reveals that POPE might overestimate model performance.
- **vs AMBER**: AMBER provides a multi-dimensional hallucination evaluation framework but remains static; ODE can be viewed as a dynamic extension of AMBER.
- **vs DyVal (dynamic evaluation for LLMs)**: DyVal dynamically synthesizes mathematical reasoning samples via directed acyclic graphs but is constrained to specific algorithms. ODE extends the idea of dynamic evaluation to the multimodal domain, fulfilling cross-modal dynamic testing through concept graphs and image generation.
- Data contamination is an underestimated problem—does the community need to routinely include out-of-distribution (OOD) evaluations when reporting model performance?

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ It is the first to systematically address the data contamination problem in multimodal hallucination evaluation, offering an ingenious approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, distributions, and tasks, though the quality variance of synthetic images remains an open issue.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear and the methodology is well-structured, although some details (e.g., choice of filtering thresholds) could be more thoroughly discussed.
- **Value**: ⭐⭐⭐⭐⭐ Proposes a sustainably updatable evaluation paradigm that will have a long-term impact on the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/hallucination/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[AAAI 2026\] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](../../AAAI2026/hallucination/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/hallucination/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[AAAI 2026\] Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models](../../AAAI2026/hallucination/verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la.md)
- [\[CVPR 2025\] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
