---
title: >-
  [Paper Note] GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing
description: >-
  [CVPR 2025][Image Generation][Image Editing Benchmark] This paper introduces GRADE, the first benchmark designed to evaluate discipline-informed reasoning in image editing. Spanning 520 samples across 10 academic disciplines, it establishes a multidimensional evaluation protocol that reveals significant deficiencies in 20 state-of-the-art multimodal models on knowledge-intensive editing tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing Benchmark"
  - "Discipline-Informed Reasoning"
  - "Multimodal Model Evaluation"
  - "Knowledge-Intensive Editing"
  - "Evaluation Protocol"
date: 2026-05-08
content_hash: 720ee410ded54b3c
---

# GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2603.12264](https://arxiv.org/abs/2603.12264)  
**Code**: [https://github.com/VisionXLab/GRADE](https://github.com/VisionXLab/GRADE)  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Image Editing Benchmark, Discipline-Informed Reasoning, Multimodal Model Evaluation, Knowledge-Intensive Editing, Evaluation Protocol

## TL;DR
This paper introduces GRADE, the first benchmark designed to evaluate discipline-informed reasoning in image editing. Spanning 520 samples across 10 academic disciplines, it establishes a multidimensional evaluation protocol that reveals significant deficiencies in 20 state-of-the-art multimodal models on knowledge-intensive editing tasks.

## Background & Motivation

**Background**: Unified multimodal models (e.g., GPT-4o, Gemini) strive for unified capabilities in understanding, reasoning, and generation, with image editing serving as a crucial application scenario. Multiple image editing benchmarks have been established to evaluate model performance.

**Limitations of Prior Work**: Existing image editing benchmarks (e.g., PIE-Bench, EditBench) primarily focus on natural images and shallow common-sense reasoning, such as "changing a cat to a dog" or "altering the color of the sky." These evaluations fail to assess whether models possess structured domain knowledge and reasoning capabilities—such as correcting errors in physics equations, adjusting chemical molecular structures, or modifying schematics according to historical conventions.

**Key Challenge**: Genuine image editing capabilities extend beyond simple visual manipulations to encompass deep understanding of academic disciplines and logical reasoning. However, a systematic evaluation framework to measure such "knowledge-intensive" editing capabilities is currently lacking.

**Goal**: To construct the first interdisciplinary, knowledge-driven image editing benchmark and design a multidimensional evaluation protocol to comprehensively expose the limitations of existing models.

**Key Insight**: Starting from academic disciplines (ranging from natural sciences to social sciences), this work designs editing tasks that require specific domain knowledge to complete successfully, elevating image editing from simple "image-to-image modification" to a comprehensive "understanding-reasoning-editing" capability test.

**Core Idea**: To construct the GRADE benchmark across 10 academic disciplines and propose a three-dimensional evaluation protocol encompassing "discipline-informed reasoning, visual consistency, and logical readability" to systematically evaluate the knowledge-intensive editing capabilities of multimodal models.

## Method

### Overall Architecture
The construction workflow of the GRADE benchmark consists of: (1) meticulously curating editing samples across 10 academic disciplines; (2) providing input images, implicit editing instructions (instructions requiring disciplinary knowledge to comprehend), and reference answers for each sample; and (3) scoring the models across three dimensions using a multidimensional evaluation protocol. The evaluation targets 20 mainstream open-source and closed-source multimodal models.

### Key Designs

1. **Interdisciplinary Sample Construction**:

    - **Function**: Covers a broad range of academic fields from natural sciences to social sciences to ensure evaluation comprehensiveness.
    - **Mechanism**: GRADE contains 520 carefully curated samples spanning 10 academic disciplines (e.g., physics, chemistry, biology, mathematics, computer science, economics, history). The editing instructions for each sample are implicit—instead of directly telling the model to "change a color" or "replace an object," they require the model to comprehend disciplinary knowledge to infer the necessary editing actions. For instance, given the instruction "correct the error in this mechanics diagram," the model must understand mechanical principles to identify and fix the mistake.
    - **Design Motivation**: Implicit instructions force models into deep reasoning rather than simple instruction-following, genuinely testing their grasp of knowledge.

2. **Multimensional Evaluation Protocol**:

    - **Function**: Comprehensively evaluates editing quality across three complementary dimensions.
    - **Mechanism**: (1) **Discipline Reasoning (DR)** evaluates the accuracy of disciplinary reasoning, assessing whether the edited result reflects correct scientific knowledge and logical reasoning; (2) **Visual Consistency (VC)** evaluates the consistency between the edited and original images in non-edited regions, as well as the overall visual quality; (3) **Logical Readability (LR)** evaluates whether the representation in the edited result is clear and logically coherent.
    - **Design Motivation**: A single metric cannot capture the multi-faceted nature of knowledge-intensive editing. A model might reason correctly but produce visual clutter, or generate visually perfect images with incorrect disciplinary knowledge.

3. **Implicit vs. Explicit Editing Settings**:

    - **Function**: Distinguishes between different levels of difficulty in editing capability.
    - **Mechanism**: In the implicit setting, only high-level goals are provided, requiring the model to infer specific operations; the explicit setting provides detailed editing instructions. The reasoning capability of models is measured by comparing the performance gap between these two settings.
    - **Design Motivation**: The implicit setting is closer to real-world applications where users typically describe high-level needs rather than providing pixel-level instructions.

### Loss & Training
As this is a benchmark paper, no training is involved. Evaluation is conducted using GPT-4o as an automatic evaluator to score across the three dimensions, with alignment validated against human evaluations.

## Key Experimental Results

### Main Results
Comprehensive evaluation of 20 SOTA models (including open-source and closed-source) on GRADE:

| Model Type | Representative Model | DR Score | VC Score | LR Score | Overall |
|---------|---------|--------|--------|--------|------|
| Best Closed-source | GPT-4o | Top tier | High | High | Leading, but still has clear gaps |
| Closed-source | Gemini series | Mid-High | Mid-High | Mid-High | Behind GPT-4o |
| Best Open-source | Representative open-source | Moderate | Moderate | Moderate | Significantly behind closed-source |
| Weaker Open-source | Small-parameter models | Low | Mid-Low | Mid-Low | Substantially behind |

### Discipline-wise Analysis

| Academic Discipline | Average Model Performance | Difficulty | Description |
|---------|------------|------|------|
| Mathematics / Physics | Poor | High | Requires precise mathematical and physical formula reasoning |
| Chemistry / Biology | Poor | High | Requires expert knowledge like molecular structures |
| Computer Science | Moderate | Medium | Editing of code / architecture diagrams |
| Social Sciences | Relatively good | Low | Involves more common-sense knowledge |

### Key Findings
- **Huge gap between implicit and explicit**: Under the implicit editing setting, all models suffer a substantial drop in performance, suggesting that current models heavily rely on explicit instructions and lack autonomous reasoning.
- **Discipline-informed reasoning is the primary bottleneck**: Scores in the DR dimension are generally the lowest; models often maintain visual consistency but fail on academic accuracy.
- **Significant gap between open- and closed-source**: Closed-source models (especially GPT-4o) exhibit prominent advantages on knowledge-intensive tasks, yet even the best-performing models fall far short of satisfactory levels.
- **Natural sciences are harder than social sciences**: Editing tasks in rigorous disciplines like physics and chemistry pose the greatest challenges to the models.

## Highlights & Insights
- **Advancing from shallow editing to knowledge-intensive editing** represents a valuable direction for benchmark evolution. GRADE fills the vacancy in existing benchmarks that focus only on simple visual operations, pushing forward the evaluation of the "depth of understanding" in multimodal models.
- **The design of the three-dimensional evaluation protocol** is transferable to other generation tasks requiring domain/specialized knowledge, such as scientific chart generation or educational content creation.
- **The implicit editing setting** reveals a crucial insight: a model's "editing capability" can be decoupled from its "reasoning capability," pointing to targeted enhancements in the reasoning phase for future work.

## Limitations & Future Work
- The sample size of 520 is relatively limited; approximately 50 samples per discipline may not comprehensively cover the vast spectrum of academic knowledge.
- Utilizing GPT-4o as an automatic evaluator may introduce biases, particularly when assessing competing models.
- More complex settings, such as video editing and 3D editing in knowledge-intensive scenarios, are not yet covered.
- The benchmark creation process relies on manual curation, making scaling up to more academic disciplines costly.
- Future research can explore coupling GRADE's evaluation protocol with automated knowledge graph verification.

## Related Work & Insights
- **vs. PIE-Bench**: PIE-Bench focuses on general image editing (e.g., object modification, color changes), whereas GRADE specializes in knowledge-intensive editing, making them complementary.
- **vs. EditBench**: EditBench evaluates attribute-control capabilities in text-guided editing but does not address academic reasoning.
- **vs. RISE-Video, etc.**: Emerging physical-rule evaluation benchmarks for video generation share similar philosophy with GRADE, but GRADE covers a much broader spectrum of academic disciplines.

## Rating
- Novelty: ⭐⭐⭐⭐ First discipline-informed reasoning benchmark in image editing with a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 20 models with rich ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Meticulously structured and detailed 49-page paper.
- Value: ⭐⭐⭐⭐ Holds significant value in advancing the evaluation of deep reasoning capabilities in multimodal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WiseEdit: Benchmarking Cognition- and Creativity-Informed Image Editing](../../CVPR2026/image_generation/wiseedit_benchmarking_cognition-_and_creativity-informed_image_editing.md)
- [\[CVPR 2025\] Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models](six-cd_benchmarking_concept_removals_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] MotionEdit: Benchmarking and Learning Motion-Centric Image Editing](../../CVPR2026/image_generation/motionedit_benchmarking_and_learning_motion-centric_image_editing.md)
- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] Concept Lancet: Image Editing with Compositional Representation Transplant](concept_lancet_image_editing_with_compositional_representation_transplant.md)

</div>

<!-- RELATED:END -->
