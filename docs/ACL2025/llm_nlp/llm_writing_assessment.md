---
title: >-
  [Paper Note] LLMs can Perform Multi-Dimensional Analytic Writing Assessments
description: >-
  [ACL 2025][LLM (Other)][Automated Writing Evaluation] Using an L2 postgraduate literature review corpus, this study systematically evaluates the capabilities of LLMs in multi-dimensional analytic writing assessment (scoring + commenting) and proposes ProEval, an explainable feedback quality evaluation framework.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Automated Writing Evaluation"
  - "Multi-Dimensional Analytic Scoring"
  - "Feedback Comment Generation"
  - "L2 Academic Writing"
  - "LLM-as-judge"
date: 2026-05-08
content_hash: 557ccffa0d941dc6
---

# LLMs can Perform Multi-Dimensional Analytic Writing Assessments

**Conference**: ACL 2025  
**arXiv**: [2502.11368](https://arxiv.org/abs/2502.11368)  
**Code**: [GitHub](https://github.com/jaaack-wang/multi-dimensional-analytic-writing-assessments)  
**Area**: LLM/NLP  
**Keywords**: Automated Writing Evaluation, Multi-Dimensional Analytic Scoring, Feedback Comment Generation, L2 Academic Writing, LLM-as-judge  

## TL;DR

Using an L2 postgraduate literature review corpus, this study systematically evaluates the capabilities of LLMs in multi-dimensional analytic writing assessment (scoring + commenting) and proposes ProEval, an explainable feedback quality evaluation framework.

## Background & Motivation

**High cost of multi-dimensional analytic assessment**: Human grading and commenting across multiple dimensions simultaneously imposes a heavy cognitive load, is time-consuming, and is expensive, resulting in a severe scarcity of high-quality annotated corpora in the L2 writing field.

**Emergence of LLMs in writing assessment**: Prior studies have explored LLMs for holistic scoring or single-dimension commenting, but the ability to **jointly perform multi-dimensional scoring and commenting** has not been systematically investigated.

**Limitations of prior work in evaluation**: Evaluation of comment quality typically relies on human Likert scale ratings, which are costly, non-reproducible, and difficult to scale.

**Lack of suitable evaluation corpora**: Publicly available L2 writing corpora that concurrently contain multi-dimensional scores and comments are virtually non-existent.

**Core Problem**: Can LLMs provide "reasonably good" multi-dimensional analytic assessments? How do different interaction modes and prompting conditions affect the quality?

**Value**: If LLM assessment is reliable, it can significantly reduce feedback costs in L2 academic writing pedagogy, benefiting both learners and instructors.

## Method

### Overall Architecture

The research pipeline consists of three components: (1) constructing an L2 writing corpus annotated with human multi-dimensional assessments; (2) prompting LLMs to execute the same assessment task under various conditions; and (3) proposing the ProEval framework to automatically evaluate the quality of feedback comments.

### Corpus Construction

- **Scale**: 141 literature reviews written by 51 L2 graduate students, with an average length of 1,321 words (930 words excluding citations), covering 5 humanities and social science topics.
- **Assessment Rubric**: Each paper is graded on a 10-point scale and reviewed by 2–3 independent experts across 9 analytic dimensions (C1–C9). C1: Material Selection, C2: Citation Integration, C3: Quality of Key Elements, C4: Structural Logic, C5: Content Clarity, C6: Coherence, C7: Use of Cohesive Devices, C8: Grammar and Syntax, C9: Academic Vocabulary.
- **No Data Contamination**: The corpus was created prior to the release of ChatGPT and has never been made public, making it suitable for LLM evaluation.

### ProEval Feedback Comment Quality Evaluation Framework

A three-step pipeline evaluation:

1. **Problem Extraction**: Uses GPT-4o to extract identified writing problems and their contextual information (explanation/suggestion/correction) from the feedback comments.
2. **Problem Classification**: Characterizes each problem along three dimensions—whether it references a specific location in the paper, whether it includes suggestions for improvement, and whether it provides a directly applicable concrete correction.
3. **Correction Relevance Check**: Uses GPT-4-Turbo to verify whether the identified problems are genuine, whether they relate to the assessment rubric, and whether the corrections are accurate.

Framework Validation: Two annotators independently labeled 200+ samples, achieving generally high Cohen's Kappa. The LLM obtained an $F_1 = 0.92$ in problem extraction and a classification accuracy of $\ge 5\%$.

### LLM Evaluation Experimental Setup

- **Models**: GPT-4o, Gemini-1.5-flash, Llama-3 70B-Instruct.
- **Three Interaction Modes**: IM1 (one-shot prompting for 9 questions), IM2 (multi-turn conversation questioning dimension-by-dimension), IM3 (9 independent prompts questioned separately).
- **Default Prompt Settings**: System prompt containing L2 context and assessment guidelines, input containing references, score-first comment-second, and temperature = 0.
- **Reliability Testing**: Single-factor comparisons including altering model versions, simplifying system prompts, removing reference lists, comment-first, and temperature = 1.

## Key Experimental Results

### Table 1: Scoring Agreement (Figure 3 Heatmap)

| Comparison | QWK Range | AAR1 Range |
|------|----------|-----------|
| Human-Human | Higher | Higher |
| LLM-LLM | Highest | Highest |
| Human-LLM (Best) | 0.59-0.88 (AAR1) | Majority $>0.5$ |

**Key Findings**: (a) Humans exhibit a human-like scoring pattern, whereas LLMs cluster with other LLMs; (b) LLM scores usually differ from human scores by $\le 1$ point; (c) Human-LLM agreement is highest under the IM3 interaction mode; (d) Agreement is higher for technical/objective dimensions such as C1/C2/C8/C9 and worst for C7 (cohesive devices).

### Table 2: Statistics on Feedback Comments (Table 2)

| Evaluator | Comment Rate | Avg Length | Problem Detection Rate | Avg Problem Count |
|--------|--------|----------|------------|------------|
| Human B | 0.24 | 104 $\pm$ 85 | 0.97 | 3.8 $\pm$ 3.5 |
| Human C | 1.00 | 62 $\pm$ 85 | 0.56 | 1.3 $\pm$ 1.8 |
| GPT-4o IM1 | 1.00 | 65 $\pm$ 14 | 1.00 | 2.1 $\pm$ 0.9 |
| GPT-4o IM3 | 1.00 | 381 $\pm$ 65 | 1.00 | 6.1 $\pm$ 2.0 |
| Gemini IM3 | 1.00 | 571 $\pm$ 182 | 1.00 | 8.2 $\pm$ 3.3 |

**Key Findings**: (a) LLMs consistently provide comments and detect problems, whereas humans sometimes omit them; (b) Comments generated under IM2/IM3 are significantly longer and more detailed than those from IM1; (c) Under IM3, LLMs provide more concrete corrections in subjective dimensions (C3–C6) compared to humans; (d) Scores and comments display the expected negative correlation, verifying assessment validity.

### Reliability Testing (Table 4)

After altering model versions or prompting conditions, AAR1 remains $\ge 0.81$ (majority $>0.9$) and BERTScore remains $\ge 0.67$, indicating that LLM assessment exhibits strong stability and robustness.

## Highlights & Insights

- **Deft design of the ProEval framework**: It decomposes subjective comment quality assessment into a verifiable pipeline of subtasks, achieving interpretability, extensibility, and reproducibility, and outperforming direct Likert scale ratings.
- **Comprehensive experimental design**: Evaluating 3 interaction modes $\times$ multiple ablation conditions $\times$ 3 LLMs, covering both scoring and commenting dimensions.
- **Corpus contribution**: The first publicly available L2 academic writing corpus that concurrently includes multi-dimensional scores and comments, with zero data contamination.
- **High practical value**: Under the IM3 mode, LLMs can provide concrete corrective recommendations even in subjective dimensions, compensating for the lack of detailed human commentary in such dimensions.

## Limitations & Future Work

- **Domain limitations**: It only covers English literature reviews, leaving other genres (such as technical reports or creative writing) and other languages unverified.
- **Indirect evaluation**: The evaluation of comment quality is indirect (via problem extraction rather than direct human judgment) and lacks large-scale human validation.
- **Assumption limitations of ProEval**: It does not account for factors influencing perceived comment quality, such as politeness or logical coherence.
- **Limited ablation studies**: Only one condition is changed at a time, leaving multi-factor interaction effects unexplored.
- **Model timeliness**: Only GPT-4o, Gemini-1.5-flash, and Llama-3 70B were tested, without covering more recent models.

## Related Work & Insights

- **Traditional AWE/AES systems**: From the 1960s to the present, primarily focused on holistic scoring, utilizing DNNs for scoring (Taghipour & Ng 2016) and sentence-level error correction (Nagata 2019).
- **LLMs for AWE**: Holistic scoring (Mizumoto & Eguchi 2023; Yancey et al. 2023), multi-dimensional scoring (Yavuz et al. 2024; Banno et al. 2024), and feedback generation (Han et al. 2024; Behzad et al. 2024). Stahl et al. (2024) is the only study investigating joint scoring and commenting, but focuses solely on holistic assessment.
- **L2 writing corpora**: TOEFL11 (scored without comments), CLC-FCE (error-annotated), and LEAF (personalized feedback)—none of which contain joint multi-dimensional score and comment annotations.

## Rating

- **Novelty**: ⭐⭐⭐ — The task definition (joint multi-dimensional scoring and commenting) is innovative, and the ProEval framework is novel, though the core remains LLM prompting evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The experiments are comprehensive with thorough ablations, and ProEval is doubly validated through human annotation and LLM-as-judge.
- **Value**: ⭐⭐⭐⭐ — The corpus and framework have direct practical value for L2 education and automated assessment research.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured, with high-density charts/tables and detailed methodological descriptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [\[ACL 2025\] Masking in Multi-hop QA: How LMs Perform with Context Permutation](masking_in_multi-hop_qa_an_analysis_of_how_language_models_perform_with_context_.md)
- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->
