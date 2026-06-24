---
title: >-
  [Paper Note] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Evaluation] This paper proposes FIHA, an automated, fine-grained hallucination evaluation framework that requires neither LLMs nor human annotations. By extracting entities, attributes, and relations from images and descriptions to generate Q&A pairs, and introducing Davidson Scene Graphs (DSG) to model inter-question dependencies, the authors construct the FIHA-v1 benchmark to comprehensively evaluate the hallucination levels…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Evaluation"
  - "Large Vision-Language Models"
  - "Davidson Scene Graph"
  - "LLM-free Evaluation"
  - "Fine-grained Evaluation"
date: 2026-05-08
content_hash: 9a2a401e098f30e6
---

# FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs

**Conference**: ACL 2025  
**arXiv**: [2409.13612](https://arxiv.org/abs/2409.13612)  
**Code**: [https://github.com/confidentzzzs/FIHA](https://github.com/confidentzzzs/FIHA)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Evaluation, Large Vision-Language Models, Davidson Scene Graph, LLM-free Evaluation, Fine-grained Evaluation

## TL;DR
This paper proposes FIHA, an automated, fine-grained hallucination evaluation framework that requires neither LLMs nor human annotations. By extracting entities, attributes, and relations from images and descriptions to generate Q&A pairs, and introducing Davidson Scene Graphs (DSG) to model inter-question dependencies, the authors construct the FIHA-v1 benchmark to comprehensively evaluate the hallucination levels of mainstream Large Vision-Language Models.

## Background & Motivation

Large Vision-Language Models (LVLMs) such as LLaVA and MiniGPT-4 have demonstrated powerful capabilities in visual understanding. However, they commonly suffer from hallucination issues, where models may describe non-existent objects, incorrect attributes, or inaccurate relations in images. Accurately evaluating the degree of model hallucination is crucial for improving model trustworthiness.

**Limitations of Prior Work**: Current hallucination evaluation methods face two core challenges:

**Neglecting dependencies between questions**: Existing methods evaluate each question independently, but in reality, logical dependencies exist between questions. For example, if a model incorrectly answers "Is there a bicycle in the picture?", answering dependent questions such as "What color is the bicycle?" becomes meaningless. Ignoring such dependencies leads to unreliable evaluation results, as weaker models might achieve artificially inflated scores by guessing dependent questions correctly.

**Reliance on expensive human annotation or LLMs**: Most existing benchmarks require human annotation (e.g., AMBER) or rely on LLMs to generate Q&A pairs (e.g., Hal-Eval), resulting in high costs and poor scalability.

**Key Challenge**: There is a need for an evaluation framework that comprehensively covers multiple hallucination types (objects, attributes, and relations) while remaining low-cost and reliable.

**Key Insight**: Leverage mature vision tools (object detection, relation extraction) to automatically generate Q&A pairs, avoiding reliance on LLMs and human annotation; introduce DSG to model the dependency structure between questions, thereby improving evaluation reliability.

## Method

### Overall Architecture
FIHA provides two parallel Q&A generation paths: image-based and description-based. Both paths extract entity information separately to generate diverse questions, which are ultimately organized via DSG based on their dependency relationships before being fed into the models for evaluation.

### Key Designs

1. **Description-based Information Extraction**:

    - If no ready-made description is available, BLIP-2 is used to generate image descriptions (a smaller model is selected to minimize hallucinations).
    - SpaCy POS tagging is utilized to extract objects and their attributes (color, quantity, size, etc.), yielding $G^C_{O,A} = \{o_1:A_1, \ldots, o_n:A_n\}$.
    - Stanford CoreNLP is used to extract relations between objects, obtaining $G^C_R = \{R_1(o^1_{R_1}, o^2_{R_1}), \ldots\}$.
    - Design Motivation: Description text provides a human-perspective summary of information, and the extraction methods are mature and reliable.

2. **Image-based Information Extraction**:

    - Grounding DINO is employed for object detection, extracting objects and attributes (color, size, shape).
    - RelTR is used to generate sparse scene graphs, extracting spatial and action relations between objects.
    - Design Motivation: Images contain richer detailed information than descriptions, allowing the two paths to serve as complementary coverage.

3. **Multi-type Q&A Pair Generation**:

    - **Yes-No Questions**: Check object existence (e.g., "Is there a {obj} in the image?") and relations (e.g., "Is there a {obj2} near the {obj1}?").
    - **Wh- Questions**: Use question words like what/who/which/where/how many, requiring free-text answers (no more than three words).
    - **Negative Questions**: Replace real objects, attributes, or relations with non-existent ones to detect whether the model produces false confirmations.
    - Design Motivation: Multi-type questions provide a more comprehensive hallucination evaluation than relying solely on Yes-No questions.

4. **Davidson Scene Graph (DSG) Dependency Modeling**:

    - Q&A pairs are organized into a tree-like structure, where object existence questions act as root nodes, while related attribute and relation questions serve as leaf nodes.
    - During evaluation, the root node is assessed first: if the object existence answer is incorrect, all leaf node questions are directly classified as hallucinations.
    - Design Motivation: Avoids weaker models gaining inflated scores by guessing dependent questions correctly, making the evaluation more rigorous and reliable.

## Key Experimental Results

### Main Results — MSCOCO Dataset (Image-based Q&A)

| Model | Accuracy | Precision | Recall | F1 | F1(Gen) |
|------|----------|-----------|--------|-----|---------|
| mPLUG-Owl | 42.1 | 70.2 | 61.4 | 43.7 | 15.2 |
| MiniGPT-4 | 23.5 | 27.5 | 22.2 | 22.1 | 21.6 |
| LLaVA-1.5-7B | 77.8 | 77.0 | 65.9 | 67.7 | 21.4 |
| LLaVA-1.5-13B | 78.9 | 80.9 | 66.4 | 68.3 | 20.9 |
| InstructBLIP | 84.7 | 83.3 | 78.6 | 80.4 | 21.8 |
| GPT-4V | **87.2** | **81.4** | **86.3** | **85.5** | **25.2** |

### Ablation Study — Performance Changes After Introducing DSG

| Model | Acc. Drop | F1 Drop | Description |
|------|---------|--------|------|
| GPT-4V | 6.0% | 9.9%→8.4% | Strong models are minimally affected by DSG, showing stronger contextual reasoning. |
| LLaVA-1.5-13B | 2.7% | 3.6% | Fewer cascade errors. |
| mPLUG-Owl | 29.6% | 28.7% | Frequent propagation of root node errors in weaker models. |
| MiniGPT-4 | 62.6% | 61.2% | The weakest model exposes a large amount of basic hallucinations. |

### Fine-grained Results (MSCOCO Caption-based)

| Hallucination Type | GPT-4V F1 | InstructBLIP F1 | Description |
|----------|-----------|-----------------|------|
| Object Existence | 88.6 | 84.2 | Models generally perform well. |
| Attribute Recognition | 79.8 | 55.6 | Judging attributes like color/size is significantly more difficult. |
| Relation Judgment | 58.3 | 52.1 | Most challenging, involving interactions among multiple objects. |

### Key Findings
- GPT-4V performs best across all dimensions, with InstructBLIP closely following.
- All models perform worst on relation hallucinations (GPT-4V's F1 is only 58.3%), as relations involve multiple objects.
- Attribute hallucination represents medium difficulty; model performance drops significantly on blurred/foggy images.
- Performance scales with model parameter size (LLaVA-1.5-13B outperforms 7B).
- The human-verified accuracy of Q&A pairs generated by FIHA reaches 96-98.2%.

## Highlights & Insights
- The first LVLM hallucination evaluation framework that is simultaneously LLM-free and annotation-free, enabling large-scale deployment at minimal cost.
- DSG dependency modeling is a simple yet effective innovation that exposes how weaker models are overestimated by traditional evaluations.
- It simultaneously supports both discriminative and generative question evaluation, offering broader coverage than previous works.
- Tests on fogged images demonstrate the evaluation framework's ability to analyze model robustness against noisy images.

## Limitations & Future Work
- Information extraction relies on the accuracy of tools like Grounding DINO and RelTR; errors from these tools propagate to the evaluation.
- Only 7 models have been tested so far, lacking newer models such as LLaVA-NeXT, Qwen-VL, etc.
- Wh-questions are evaluated using BERTScore, which may not be precise enough for matching free-text answers.
- The construction of negative questions is based on random replacement, which may not be challenging enough (models can easily answer via elimination).
- More complex hallucination types, such as temporal relations and causal reasoning, are not yet considered.

## Related Work & Insights
- Compared with POPE (which only detects object hallucinations) and AMBER (which requires human annotation), FIHA provides more comprehensive coverage at a lower cost.
- The introduction of DSG is inspired by Cho et al. (2023), transferring it from text generation evaluation to the domain of hallucination evaluation.
- The complementary design philosophy of the two information extraction paths (image + description) can be extended to other visual evaluation tasks.
- It suggests a future direction: leveraging combinations of existing mature tools to accomplish complex evaluation tasks, rather than relying solely on end-to-end large models.

## Rating
- Novelty: ⭐⭐⭐⭐ DSG dependency modeling and LLM-free evaluation are major highlights, though the information extraction methods themselves are less novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-model, fine-grained analysis, and reliability verification are all well-engineered, though model coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich figures/tables, but some symbolic definitions appear slightly redundant.
- Value: ⭐⭐⭐⭐ Provides a practical, low-cost hallucination evaluation tool with immediate value for LVLM developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[ICLR 2026\] FREAK: A Fine-grained Hallucination Evaluation Benchmark for Advanced MLLMs](../../ICLR2026/hallucination/freak_a_fine-grained_hallucination_evaluation_benchmark_for_advanced_mllms.md)
- [\[ACL 2025\] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](../../CVPR2025/hallucination/ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
