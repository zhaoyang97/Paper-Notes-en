---
title: >-
  [Paper Note] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers
description: >-
  [ACL 2025 Findings][Multimodal VLM][Schematic Diagram Understanding] This paper introduces MISS-QA, the first benchmark specifically designed to evaluate the capability of multimodal foundation models to understand schematic diagrams in scientific papers. It contains 1,500 expert-annotated samples and reveals a significant performance gap between the state-of-the-art models and human experts.
tags:
  - "ACL 2025 Findings"
  - "Multimodal VLM"
  - "Schematic Diagram Understanding"
  - "Scientific Literature QA"
  - "Multimodal Benchmark"
  - "Vision-Language Models"
  - "Information Retrieval"
date: 2026-05-08
content_hash: 786fa2521049d011
---

# Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers

**Conference**: ACL 2025 Findings  
**arXiv**: [2507.10787](https://arxiv.org/abs/2507.10787)  
**Code**: [https://github.com/yilunzhao/MISS-QA](https://github.com/yilunzhao/MISS-QA)  
**Area**: Multimodal VLM  
**Keywords**: Schematic Diagram Understanding, Scientific Literature QA, Multimodal Benchmark, Vision-Language Models, Information Retrieval

## TL;DR

This paper introduces MISS-QA, the first benchmark specifically designed to evaluate the capability of multimodal foundation models to understand schematic diagrams in scientific papers. It contains 1,500 expert-annotated samples and reveals a significant performance gap between the state-of-the-art models and human experts.

## Background & Motivation

**Background**: Multimodal foundation models (e.g., GPT-4o, Gemini, Qwen2.5-VL) have achieved remarkable progress in tasks such as chart understanding and natural image QA. However, most evaluations focus on structured charts (bar charts, line graphs, tables) or natural scene images.

**Limitations of Prior Work**: Schematic diagrams widely used in scientific papers—such as method flowcharts, system architectures, and model overviews—are key vehicles for conveying the core ideas of research. However, there is currently a lack of benchmarks specifically evaluating models' ability to understand these unstructured, information-dense scientific schematic diagrams. Existing benchmarks either focus on simple charts or only on image captioning, failing to measure the deep understanding of complex scientific concepts.

**Key Challenge**: Schematic diagrams are fundamentally different from general charts—they typically contain complex spatial relationships, symbolic representations, arrow connections, etc., and require integration with paper context to be correctly understood. Existing evaluation methods overlook this challenge of multimodal scientific reasoning.

**Goal**: Build a high-quality benchmark to systematically evaluate models' information-seeking QA capability on scientific schematic diagrams, and analyze the capability boundaries and weaknesses of current mainstream models.

**Key Insight**: The authors notice that when reading papers, researchers often use schematic diagrams to quickly grasp the core process and design ideas of a method. Therefore, they design an "information-seeking" oriented QA task—given a schematic diagram and one of its visual elements, the model needs to answer questions regarding the design rationale, implementation details, literature background, or experimental results represented by that element.

**Core Idea**: Build the first multimodal QA benchmark, MISS-QA, targeted at schematic diagrams in scientific papers. Through expert annotation and a carefully designed evaluation protocol, it reveals the deficiencies of current multimodal models in understanding complex scientific visual information.

## Method

### Overall Architecture

The construction pipeline of the MISS-QA benchmark includes: (1) collecting schematic diagrams from 465 AI-related papers on arXiv; (2) asking researchers to propose free-form information-seeking questions for specific visual elements (annotated with bounding boxes) in each diagram; (3) having human experts annotate answers or mark questions as unanswerable. During evaluation, the model receives the schematic diagram, the highlighted visual element, and the question, and is required to either generate an answer or determine that the question is unanswerable.

### Key Designs

1. **Benchmark Data Construction Pipeline**:

    - Function: Generate a high-quality scientific schematic diagram QA dataset.
    - Mechanism: Each sample includes a schematic diagram from a scientific paper, a visual element highlighted with a bounding box, a free-form information-seeking question, corresponding paper context, and a human-annotated answer. Questions cover five major categories of information-seeking scenarios—Design Rationale, Implementation Details, Literature Background, Experimental Results, and Others (e.g., limitations, ethics).
    - Design Motivation: Cover the information dimensions that researchers care about most when reading papers, ensuring the benchmark reflects real scientific literature reading scenarios.

2. **Unanswerable Question Design**:

    - Function: Evaluate the model's ability to recognize situations with insufficient information.
    - Mechanism: Some questions are designed to be unanswerable based solely on the schematic diagram (requiring additional information from the full paper text). The model needs to judge and refuse to answer, rather than hallucinating answers.
    - Design Motivation: In real-world scenarios, schematic diagrams do not always contain all information. This tests whether the model has the ability to "know what it doesn't know," which is a crucial characteristic of reliable AI systems.

3. **Automatic Evaluation Protocol**:

    - Function: Achieve automatic scoring highly consistent with human judgment.
    - Mechanism: Train an automatic evaluation model based on human-graded data to score the quality of the model's generated free-form answers, avoiding the use of crude metrics like simple string matching.
    - Design Motivation: Free-form QA is not suitable for exact match evaluation; a smarter evaluation method is needed to capture semantic equivalence.

### Loss & Training

This is a benchmark paper and does not involve model training. Regarding the evaluation protocol, an automatic scorer calibrated with human grading data is used to ensure high correlation between automatic scores and human judgments.

## Key Experimental Results

### Main Results

| Model | Overall Accuracy | Answerable Questions | Unanswerable Questions |
|------|-----------|-----------|-------------|
| Human Expert | ~85% | ~87% | ~80% |
| o4-mini | ~65% | ~68% | ~55% |
| Gemini-2.5-Flash | ~62% | ~64% | ~52% |
| Qwen2.5-VL-72B | ~58% | ~61% | ~48% |
| GPT-4o | ~55% | ~58% | ~45% |
| InternVL2-76B | ~52% | ~55% | ~42% |
| Open-source Small Models (7B) | ~35-42% | ~38-45% | ~25-32% |

### Ablation Study

| Information-Seeking Scenario | Best Model Performance | Human Performance | Gap |
|-------------|------------|---------|------|
| Design Rationale | ~60% | ~83% | ~23% |
| Implementation Details | ~58% | ~85% | ~27% |
| Literature Background | ~52% | ~80% | ~28% |
| Experimental Results | ~65% | ~88% | ~23% |
| Other | ~55% | ~82% | ~27% |

### Key Findings

- There is a significant performance gap (about 20-50 percentage points) between all 18 evaluated models and human experts, indicating that understanding scientific schematic diagrams remains a primary challenge for multimodal models.
- Models perform particularly poorly on unanswerable questions, tending to generate answers overconfidently rather than refusing, which exposes a lack of reliable "uncertainty awareness" in current models.
- Closed-source large models (o4-mini, Gemini-2.5-Flash) significantly outperform open-source models, but remain far below human levels.
- Models perform worst on "Literature Background" questions, showing that models struggle to associate visual elements in schematic diagrams with broader academic knowledge.

## Highlights & Insights

- **Pioneering Evaluation Dimension**: MISS-QA is the first to introduce scientific schematic diagram understanding as an independent multimodal evaluation dimension, filling a critical gap in existing benchmarks. This reminds us that evaluating multimodal capabilities should not be restricted to natural images and structured charts.
- **Design of Unanswerable Questions**: Evaluating the "self-awareness" of models by introducing unanswerable questions. This practice can be transferred to any QA benchmark design, helping to distinguish true comprehension from surface-level matching.
- **Classification of Information-Seeking Scenarios**: Categorizing questions based on researchers' actual information needs (design rationale, implementation details, literature background, etc.) provides a structured analytical framework for future research.

## Limitations & Future Work

- The dataset only covers papers in the AI field. The forms and complexities of scientific schematic diagrams in other disciplines (such as biology, chemistry, physics) differ greatly, and generalization remains to be verified.
- Although the scale of 1,500 samples ensures annotation quality, it is relatively small for model training.
- The reliability of the automatic evaluation protocol depends on the quality and coverage of the calibration data, potentially introducing systematic biases.
- Future work could consider extending the benchmark to multi-disciplinary and multi-lingual scenarios, and exploring performance improvements on this task through supervised training data enhancement.

## Related Work & Insights

- **vs ChartQA/PlotQA**: These benchmarks focus on structured charts (bar charts, line graphs, etc.), whereas this work focuses on more complex scientific schematic diagrams, which require more domain knowledge and spatial reasoning abilities.
- **vs DocVQA**: While DocVQA targets information extraction from document images, this work focuses on the joint understanding of schematic diagrams and paper context in scientific papers, presenting a more challenging task.
- **vs SciGraphQA**: Although that benchmark also focuses on scientific images, it is mainly targeted at data reading from scientific charts, rather than the deep understanding of schematic diagrams like method flowcharts.

## Rating

- Novelty: ⭐⭐⭐⭐ First specialized benchmark for scientific schematic diagram understanding, filling an important evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 18 mainstream models, covering multiple information-seeking scenarios, but lacks deep experiments on model improvement directions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined problems, and rich analytical dimensions.
- Value: ⭐⭐⭐⭐ As a benchmark paper, it holds significant reference value for driving the development of the multimodal scientific literature understanding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](../../NeurIPS2025/multimodal_vlm/what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)

</div>

<!-- RELATED:END -->
