---
title: >-
  [Paper Note] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?
description: >-
  [ACL 2025 (Short Paper)][LLM (Other)][Medical QA] This paper explores using LLMs to automatically generate answer keys (information nuggets) for medical QA. By comparing various generation methods with human expert annotations, the study finds that the few-shot answer-based extraction method performs best. However, the capability of LLMs to extract atomic facts remains limited, with Llama 3.3 showing the best performance.
tags:
  - "ACL 2025 (Short Paper)"
  - "LLM (Other)"
  - "Medical QA"
  - "Information Nugget"
  - "Answer Keys"
  - "LLM Evaluation"
  - "Fact Extraction"
date: 2026-05-08
content_hash: 746f0e4a22b5cc2e
---

# Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?

**Conference**: ACL 2025 (Short Paper)  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-short.28/)
**Code**: None  
**Area**: Medical NLP / QA Systems / Factuality Evaluation  
**Keywords**: Medical QA, Information Nugget, Answer Keys, LLM Evaluation, Fact Extraction

## TL;DR

This paper explores using LLMs to automatically generate answer keys (information nuggets) for medical QA. By comparing various generation methods with human expert annotations, the study finds that the few-shot answer-based extraction method performs best. However, the capability of LLMs to extract atomic facts remains limited, with Llama 3.3 showing the best performance.

## Background & Motivation

**Background**: Evaluating the factuality of LLM-generated text is a core challenge in NLP, particularly critical in the medical QA domain where misinformation can directly impact patient health. "Information nugget" is an evaluation method that decomposes correct answers into atomic facts and checks whether the evaluated text contains these facts.

**Limitations of Prior Work**: Manual extraction of nuggets is costly and time-consuming, making it unsuitable for large-scale evaluation. Recent shared tasks on RAG evaluation have begun exploring LLM-based automated nugget extraction, but its efficacy in the medical domain has not been systematically evaluated. Medical nuggets require extreme precision, as a vague or erroneous nugget can lead to misjudgments of medical answers.

**Key Challenge**: While automated nugget generation can significantly reduce costs, the medical domain demands exceptionally high precision, and LLMs may not reliably extract precise atomic facts from medical texts.

**Goal**: (1) To evaluate various LLM-based methods for automatically generating medical nuggets; (2) To quantify the reliability of automated methods by comparing them against human expert annotations.

**Key Insight**: Grounded in RAG evaluation practices, this work systematically compares different nugget generation strategies (such as question-based vs. answer-based, few-shot vs. zero-shot settings) to establish a reliability benchmark in the medical QA scenario.

**Core Idea**: Automated evaluation of medical QA requires reliable answer keys; this work systematically evaluates the capability boundaries of LLMs on this task.

## Method

### Overall Architecture

The input consists of a medical question and its reference answer, and the output is a set of information nuggets (a list of atomic facts). The evaluation metric is the alignment between the automatically generated nuggets and human expert annotations.

### Key Designs

1. **Diverse Nugget Generation Strategies**:

    - Function: Exploring different nugget generation methods.
    - Mechanism: Several generation strategies are designed: (a) Question-based generation—generating key factual points that should be included given only the question; (b) Answer-based extraction—extracting atomic facts from a given reference answer; (c) Hybrid method—initially generating candidates from the question and then verifying and supplementing them using the answer. Each method is evaluated in both zero-shot and few-shot (with manual nugget examples) settings.
    - Design Motivation: Different strategies have distinct pros and cons—question-based generation might omit specific details present in the reference answer, whereas answer-based extraction can be biased by the phrasing of the answer.

2. **Human Expert Nugget Annotation**:

    - Function: Establishing the gold standard for evaluation.
    - Mechanism: Medical informatics experts (researchers from NIH/NLM) manually created nuggets for the medical questions. Each nugget is a short, factual statement that is "atomic" (i.e., cannot be further decomposed into smaller independent facts). The annotation process followed strict guidelines, which included granularity control (neither too coarse nor too fine-grained) and medical accuracy auditing.
    - Design Motivation: High-quality human annotation is an indispensable reference for evaluating automated methods.

3. **Alignment Evaluation Framework**:

    - Function: Quantifying the alignment between automated nuggets and human nuggets.
    - Mechanism: Two assessment approaches are used: (a) Semantic similarity matching to compute the coverage and precision between the automated nugget set and the human nugget set; (b) Human judgment to determine whether each automatically generated nugget is semantically equivalent to a human gold nugget. Precision (how many generated nuggets are correct), Recall (how many human nuggets are covered), and F1 are reported.
    - Design Motivation: Purely automated semantic matching might lack precision due to the complex synonymous expressions of medical terminology, thereby requiring human assessment for validation.

### Experimental Setup

Several models such as GPT-4, GPT-3.5, and Llama 3.3 were evaluated. Medical QA data from the TREC Health Misinformation Track and BioASQ datasets were utilized.

## Key Experimental Results

### Main Results

| Method | Model | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| Answer-based + Few-shot | Llama 3.3 | Best | High | Best |
| Answer-based + Zero-shot | GPT-4 | High | Medium | Second Best |
| Question-based + Few-shot | GPT-4 | High-Medium | Low | Medium |
| Question-based + Zero-shot | GPT-4 | Medium | Low | Low-Medium |
| Hybrid | GPT-4 | Medium | Medium | Medium |

### Ablation Study

| Factor | Impact | Description |
|------|------|------|
| Few-shot vs. Zero-shot | Few-shot significantly outperforms zero-shot | Examples help the model comprehend nugget granularity |
| Answer-based vs. Question-based | Answer-based is superior | Reference answers contain specific facts |
| Nugget Granularity Control | Significant variance | Too coarse or too fine-grained nuggets reduce alignment |
| Llama 3.3 vs. GPT-4 | Llama 3.3 is slightly better | Potentially related to the specific prompt style |

### Key Findings
- The strategy of providing examples and extracting nuggets from the reference answer performs best—examples help the model grasp the correct granularity.
- LLM-generated nuggets exhibit systematic issues: (1) Inconsistent granularity—some are overly detailed while others are too generic; (2) Merging multiple facts into a single nugget; (3) Occasionally introducing information not present in the original reference answer.
- Llama 3.3 demonstrates the best performance on this task, possibly due to more precise instruction-following.
- Overall, the capability of LLMs to extract atomic facts from text remains **limited**—indicating an avenue that warrants further investigation.

## Highlights & Insights
- **Treating the nugget generation task as a testbed for LLM capabilities** provides an insightful perspective—keyword/fact extraction is a fundamental capability for LLMs performing fact-checking and text evaluation.
- **The high-precision requirements in the medical domain** make this task an excellent benchmark for testing LLM reliability—unlike in general domains, errors in medical nuggets are unacceptable.
- The finding that "answer-based extraction outperforms question-based generation" offers practical implications—suggesting that answer-driven approaches should be preferred when designing evaluation workflows.

## Limitations & Future Work
- As a short paper, the experimental scale is limited (small dataset size, limited number of models evaluated).
- The practical impact of nugget quality on downstream evaluation tasks (such as RAG system assessment) has not been evaluated.
- The definition and granularity standards for medical nuggets may vary depending on the scenario (e.g., patient-facing vs. clinician-facing).
- Future work can explore leveraging multi-turn iterations with LLMs to refine and optimize nugget quality.

## Related Work & Insights
- **vs. ARES**: ARES employs LLMs to evaluate RAG systems without explicitly extracting nuggets, whereas the nugget-based approach in this study provides a more fine-grained evaluation.
- **vs. FActScore**: FActScore proposes a framework to extract and verify atomic claims from generated text; this work applies a similar concept to the evaluation of medical QA.
- **vs. BioASQ**: BioASQ is a dominant benchmark for medical QA evaluation; the nugget method in this study can serve as a novel evaluation complement to BioASQ.

## Rating
- Novelty: ⭐⭐⭐ Methodologically, this is a systematic comparison of existing nugget extraction methods, presenting moderate innovation.
- Experimental Thoroughness: ⭐⭐⭐ The short paper format limits the dataset scale, but the dimensions of comparison are reasonably designed.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and thorough experimental analysis.
- Value: ⭐⭐⭐⭐ It holds practical significance for researching the reliability of medical NLP evaluation methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)

</div>

<!-- RELATED:END -->
