---
title: >-
  [Paper Note] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study
description: >-
  [ACL 2025][LLM (Other)][Dialogue Quality Assessment] This paper constructs a large-scale, multi-dimensional conversational quality assessment corpus covering multiple quality dimensions such as fluency, consistency, informativeness, and engagingness. Based on this corpus, a comprehensive benchmark and analysis of existing conversational evaluation methods are conducted.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Dialogue Quality Assessment"
  - "Large-Scale Corpus"
  - "Multi-Dimensional Evaluation"
  - "Dialogue System Evaluation"
  - "Human Annotation"
date: 2026-05-08
content_hash: 9b7328470d274be7
---

# Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study

**Conference**: ACL 2025  
**Area**: NLP Understanding  
**Keywords**: Dialogue Quality Assessment, Large-Scale Corpus, Multi-Dimensional Evaluation, Dialogue System Evaluation, Human Annotation

## TL;DR
This paper constructs a large-scale, multi-dimensional conversational quality assessment corpus covering multiple quality dimensions such as fluency, consistency, informativeness, and engagingness. Based on this corpus, a comprehensive benchmark and analysis of existing conversational evaluation methods are conducted.

## Background & Motivation

**Background**: The automatic evaluation of dialogue systems has long been a core challenge in the NLP field. Traditional word-overlap-based metrics such as BLEU and METEOR exhibit low correlation with human judgments. While model-based evaluation methods (e.g., BERTScore, GPT-4 scoring) yield better results, they lack a unified evaluation framework and standardized benchmark datasets.

**Limitations of Prior Work**: (1) Existing dialogue evaluation datasets are small in scale and limited in dimensions, typically annotating only binary "good/bad" classifications or single-dimensional scores. (2) The performance of different evaluation methods varies significantly across datasets, lacking a systematic horizontal comparison. (3) Quality assessment for multi-turn dialogues is even scarcer, as most existing works focus on single-turn response quality, neglecting holistic dialogue-level characteristics such as coherence, logic, and engagingness.

**Key Challenge**: Dialogue quality is a multi-dimensional concept involving linguistic quality, content quality, and interaction quality. However, existing datasets and evaluation methods typically focus on only one or two dimensions, leading to biased evaluation results and a lack of comparability among different approaches.

**Goal**: (1) To construct a large-scale, multi-dimensional, and multi-scenario dialogue quality assessment corpus; (2) To establish a unified evaluation framework and metric system; (3) To conduct a comprehensive benchmark and analysis of existing mainstream evaluation methods.

**Key Insight**: The authors propose that dialogue quality should be evaluated across five core dimensions: Fluency, Consistency, Informativeness, Engagingness, and Safety. Detailed annotation guidelines and Likert scales are designed for each dimension.

**Core Idea**: Build a dialogue evaluation corpus, ConvQA, spanning five quality dimensions through large-scale crowdsourced annotation. Based on this, a unified evaluation of existing methods is conducted to reveal their strengths and weaknesses across different dimensions.

## Method

### Overall Architecture
The study is divided into three phases: (1) Corpus construction: collecting dialogue data from various sources, designing annotation schemes, and conducting large-scale crowdsourced annotation; (2) Benchmarking: assessing the correlation between multiple automatic evaluation methods and human judgments on ConvQA; (3) In-depth analysis: exploring factors that influence evaluation quality, such as dialogue length, domain, and error types.

### Key Designs

1. **Multi-dimensional Annotation Scheme**:

    - **Function**: Provides a comprehensive and actionable evaluation framework for dialogue quality.
    - **Mechanism**: Five core dimensions are defined, with each dimension using a 1-5 Likert scale. Each score level is equipped with detailed descriptive anchors (e.g., "Fluency 5 = completely natural and fluent, without any grammatical or vocabulary errors"). Independent verification testing across dimensions is also implemented to ensure that annotators' judgments on each dimension remain mutually independent. The annotation guidelines include abundant examples and common error cases.
    - **Design Motivation**: Independent multi-dimensional evaluation characterizes different aspects of dialogue quality more accurately. Detailed anchor descriptions and examples enhance inter-annotator agreement (IAA).

2. **Dialogue Data Collection and Sampling Strategy**:

    - **Function**: Ensures that the corpus covers diverse dialogue types and quality levels.
    - **Mechanism**: Dialogue data is collected from four sources: open-domain chitchat (generated by models like BlenderBot and DialoGPT), task-oriented dialogues (from datasets like MultiWOZ), knowledge-grounded dialogues (from Wizard of Wikipedia), and real human-machine dialogues (from online dialogue system logs). A stratified sampling strategy is employed to guarantee sufficient samples for each dialogue type and quality level. The final corpus contains approximately 10,000 dialogues, with each dialogue independently evaluated by 3-5 annotators.
    - **Design Motivation**: Multi-source sampling prevents data bias, and the stratified strategy ensures a balanced distribution of low-quality and high-quality samples, allowing evaluation methods to be thoroughly tested across different quality intervals.

3. **Consistency Control and Annotation Quality Assurance**:

    - **Function**: Ensures the quality and consistency of large-scale annotations.
    - **Mechanism**: A multi-stage quality control system is adopted: qualification tests (only annotators with a pass rate >80% can participate in formal annotation), gold questions (10% of tasks with known answers are mixed into each batch as quality checkpoints), agreement monitoring (real-time calculation of Krippendorff's alpha among annotators, with low-agreement batches re-annotated), and bias calibration (regular calibration meetings to align annotators' understanding).
    - **Design Motivation**: The greatest challenge in crowdsourced annotation is quality control, particularly for highly subjective evaluation tasks. Multi-stage quality inspection ensures that the annotated results are reliable and usable.

### Loss & Training
This paper is primarily a resource and analysis paper and does not involve training new models. In the benchmarking experiments, the evaluation methods that require training are trained, validated, and tested using a unified 80/10/10 split of the corpus.

## Key Experimental Results

### Main Results

| Method | Fluency (ρ) | Consistency (ρ) | Informativeness (ρ) | Engagingness (ρ) | Overall (ρ) |
|---------|----------|----------|----------|----------|---------|
| BLEU | 0.21 | 0.08 | 0.15 | 0.12 | 0.14 |
| BERTScore | 0.38 | 0.25 | 0.31 | 0.22 | 0.29 |
| UniEval | 0.62 | 0.55 | 0.48 | 0.41 | 0.52 |
| GPT-4 Score | 0.71 | 0.63 | 0.58 | 0.52 | 0.61 |
| Our Fine-tuned Model | 0.75 | 0.68 | 0.63 | 0.57 | 0.66 |

### Annotation Quality Statistics

| Dimension | Krippendorff's α | IAA (Pearson) | Distribution Skewness | Description |
|------|------------------|--------------|---------|------|
| Fluency | 0.72 | 0.81 | -0.3 | Highest agreement, low subjectivity |
| Consistency | 0.65 | 0.74 | -0.1 | Good agreement |
| Informativeness | 0.58 | 0.68 | 0.2 | Higher subjectivity |
| Engagingness | 0.51 | 0.62 | 0.4 | Most subjective dimension |
| Safety | 0.78 | 0.85 | -1.2 | High agreement but skewed distribution |

### Key Findings
- Traditional metrics based on word overlap (e.g., BLEU) exhibit low correlation with human judgments across all dimensions, being close to random especially in consistency and engagingness.
- GPT-4 performs well across various dimensions but still shows a notable gap in "Engagingness," which is the most subjective dimension.
- Dedicated evaluation models fine-tuned on ConvQA outperform GPT-4, demonstrating the value of domain-specific data.
- Inter-annotator agreement varies significantly across dimensions: engagingness is the most difficult to reach consensus on, whereas safety is the easiest.

## Highlights & Insights
- The design of the five-dimensional evaluation framework is systematic and complete, covering major aspects of dialogue quality with good independence among dimensions, allowing them to be applied to different downstream requirements.
- The large-scale, high-quality corpus itself constitutes a significant contribution, comprising 10,000 dialogues with multi-source and multi-dimensional annotations, filling a long-standing gap in data scarcity in this field.
- The analysis regarding the evaluability of each dimension is inspiring, revealing the fundamental difficulties in automatically assessing subjective dimensions like engagingness.

## Limitations & Future Work
- The corpus primarily covers English dialogues. Frameworks and datasets for multi-lingual dialogue quality evaluation remain to be established.
- The five dimensions might still be insufficient; dimensions such as "creativity" and "personalization" are not yet integrated.
- Although the quality of crowdsourced annotations is strictly controlled, it is inevitably influenced by annotator backgrounds and cultural differences.
- Future work can focus on training stronger multi-dimensional automatic evaluation models based on this corpus, serving as standardized evaluation tools for dialogue system development.

## Related Work & Insights
- **vs FED (Mehri & Eskenazi)**: FED provides a multi-dimensional framework for dialogue evaluation but has a small data scale (approx. 500 dialogues). Ours is 20 times larger and offers higher annotation quality.
- **vs USR (Mehri & Eskenazi)**: USR focuses on reference-free evaluation but features fewer dimensions, whereas ours provides more comprehensive dimensional coverage.
- **vs ChatEval**: ChatEval is an evaluation platform rather than a dataset, whereas ours provides unified benchmark data for model comparison.

## Rating
- Novelty: ⭐⭐⭐ The framework design is solid but the paradigm is not highly novel; the core contribution lies in resource construction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale annotation, comprehensive benchmarking, and detailed statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ The corpus construction process is described in detail, and the analysis is in-depth.
- Value: ⭐⭐⭐⭐⭐ Holds significant infrastructural value for the dialogue system evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)
- [\[ACL 2025\] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)

</div>

<!-- RELATED:END -->
