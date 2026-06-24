---
title: >-
  [Paper Note] QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation
description: >-
  [ACL 2025][LLM (Other)][Question Generation Evaluation] QG-SMS proposes simulating student populations with varying comprehension levels using a single LLM. Through a three-step workflow of student profile generation, performance prediction, and analysis, it addresses the severe limitations of existing LLM evaluators in post-test analysis dimensions (item difficulty, discrimination, and distractor efficiency), achieving the highest consistency accuracy across multiple dataset…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Question Generation Evaluation"
  - "Test Item Analysis"
  - "Student Modeling and Simulation"
  - "LLM Evaluator"
  - "Item Difficulty and Discrimination"
date: 2026-05-08
content_hash: 55d6518771d69e31
---

# QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation

**Conference**: ACL 2025  
**arXiv**: [2503.05888](https://arxiv.org/abs/2503.05888)  
**Code**: [Yes](https://github.com/bnguyen5/qg-sms)  
**Area**: Other  
**Keywords**: Question Generation Evaluation, Test Item Analysis, Student Modeling and Simulation, LLM Evaluator, Item Difficulty and Discrimination

## TL;DR

QG-SMS proposes simulating student populations with varying comprehension levels using a single LLM. Through a three-step workflow of student profile generation, performance prediction, and analysis, it addresses the severe limitations of existing LLM evaluators in post-test analysis dimensions (item difficulty, discrimination, and distractor efficiency), achieving the highest consistency accuracy across multiple datasets.

## Background & Motivation

### Problem Background

The Question Generation (QG) task in Natural Language Processing is increasingly applied in educational assessment. Current QG evaluation methods are mainly divided into:
- **Reference-based metrics** (ROUGE, BLEU, BERTScore): Measure the grammatical/semantic similarity between generated questions and human references.
- **Reference-free metrics** (KDA, QSalience): Evaluate question quality independently.
- **LLM Evaluators** (Vanilla, CoT, ChatEval, etc.): Evaluate through pairwise comparison.

### Core Motivation

Evaluating question quality in educational testing consists of two stages:

**Pre-test Analysis**: Evaluating the alignment of questions with learning objectives (e.g., topic coverage).

**Post-test Analysis**: Evaluating difficulty, discrimination, and distractor efficiency based on student performance.

Existing methods excel in pre-test analysis (topic coverage) with an average of 95.6%, but fall drastically short in post-test analysis dimensions (difficulty 49.1%, discrimination 44.5%, distractor efficiency 53.3%). The root cause is that these methods only analyze question content and lack modeling from the student's perspective.

### Classic Counter-intuitive Case

Q1 tests computer vision applications (application-level question), while Q2 tests specific statistical numbers (recall-level question). Existing methods select Q1 as having higher discrimination, whereas actual student data shows Q2 is better—because CV applications fall under common sense, while precise statistical figures can only be answered correctly by students who paid close attention in class.

## Method

### Overall Architecture

QG-SMS consists of three steps (Figure 2):

**Step 1 → Step 2 → Step 3**

Input remains unchanged: Learning material $L$, question pair $\{Q_1, Q_2\}$, and evaluation dimension requirements $R_d$.

### Key Designs

1. **Student Profile Generation (Step 1)**:

    - Given the learning material $L$, the LLM is prompted to generate at least 10 student profiles with varying comprehension levels.
    - Key constraint: Simulate only differences in understanding of the learning material, avoiding any personally identifiable information to prevent social bias.
    - Examples include "Alice - The Attentive" (high attention to detail) and "Bob - The Beginner" (only masters the basics).

2. **Student Performance Prediction (Step 2)**:

    - Based on the generated student profiles, predict the response (correct/incorrect) of each simulated student for $Q_1$ and $Q_2$.
    - This step simulates the performance distribution of a student population on two questions within a real classroom.

3. **Comprehensive Evaluation (Step 3)**:

    - Provide both the question content and the simulated student performance to the LLM to make the final determination.
    - The LLM makes an informed judgment by combining its understanding of question semantics with simulated data.
    - More robust than solely calculating statistical metrics: directly calculating DF benefits difficulty (+4.56) but severely harms discrimination (-9.56).

### Task Definition

Given educational material $L$, dimension $d$ (Topic Coverage TC / Difficulty DF / Discrimination DC / Distractor Efficiency DE), and a question pair $(Q_1, Q_2)$, determine which question better meets the dimension requirements $R_d$. It is required that the difference in statistical values of the two questions on $d$ exceeds a threshold $\alpha$.

### Statistical Metrics

| Dimension | Formula | Meaning |
|------|------|------|
| Topic Coverage TC | Binary variable | Whether the question covers the target topic |
| Item Difficulty DF | $\frac{\sum x_s}{\|S\|}$ | Proportion of students answering correctly |
| Discrimination DC | $\frac{Cov(X,T)}{\sigma_X \sigma_T}$ | Correlation between single-item score and total score |
| Distractor Efficiency DE | Number of distractors chosen by $\ge 5\%$ of students | Number of effective distractors |

## Key Experimental Results

### Main Results (Table 2, Consistency Accuracy CA)

| Method | TC (EduAgent) | DF (EduAgent) | DC (EduAgent) | DE (EduAgent) |
|------|--------------|--------------|--------------|--------------|
| Vanilla| 95.39 | 50.80 | 49.18 | 64.00 |
| CoT | 92.63 | 32.26 | 32.79 | 28.00 |
| ChatEval | 95.85 | 51.61 | 42.56 | 56.00 |
| Swap | 95.85 | 54.84 | 45.90 | 53.33 |
| **QG-SMS** | **98.62** | **65.32** | **55.74** | **74.67** |

### Human Evaluation (Table 5)

| Method | Human Items AA | Generated Items CA |
|------|-----------|-----------|
| Vanilla | 70.83 | 58.33 |
| ChatEval | 69.17 | 56.67 |
| **QG-SMS** | **76.67** | **63.33** |
| Human Evaluators | **78.33** | - |

### Key Findings

1. **QG-SMS significantly outperforms all baselines in consistency accuracy**: The DF dimension is 10.48% higher than Swap, and the DE dimension is 9.34% higher than KDA.
2. **Simulated student profiles are robust**: The simulated student performance distribution remains consistent across 5 independent runs.
3. **The Step 3 evaluation step is indispensable**: Directly using simulated data to calculate statistical metrics benefits difficulty but severely harms discrimination.
4. **QG-SMS approaches human evaluators**: Human AA is 78.33%, QG-SMS is 76.67%, and it surpasses humans in discrimination.
5. **Ranking scores derived from QG-SMS show the highest correlation with real DE values** (Table 4: Spearman 0.48 vs Vanilla 0.34).

## Highlights & Insights

- **Core Insight**: Introducing a "simulated student population" perspective bridges the gap between question content analysis and student performance, which is a simple yet powerful design.
- **Single-LLM Multi-Role Simulation**: No need for multiple LLMs to simulate students of different levels; a single GPT-4o can achieve this, making the pipeline more efficient and scalable.
- **Automated Post-test Analysis**: Predicting post-test analysis results during the pre-test stage, saving the time required to wait for actual test implementation.
- **The Challenge of Discrimination Evaluation**: Even human evaluators achieve only 53.33% accuracy in discrimination evaluation, whereas QG-SMS surpasses human performance in this dimension.

## Limitations & Future Work

- Currently, only individual item-level quality is evaluated, without considering diversity and balance at the test-assembly level.
- Statistical significance relies on a relaxed threshold of $p = 0.1$.
- Simulated student profiles may contain implicit biases (e.g., names leaning towards European styles).
- Future Work: QG-SMS can be integrated into reward-based optimization pipelines to better align generated questions with educational objectives.
- Future Work: Extending to research question evaluation, automating assessments using a simulated researcher's perspective.

## Related Work & Insights

- Extends the line of work on LLM evaluators (ChatEval, G-Eval, etc.), introducing test item analysis to QG evaluation for the first time.
- Inspired by generative agents simulating human behavior (Park et al., 2023), applying simulation concepts to educational scenarios.
- Unlike KDA (Moon et al., 2022), QG-SMS uses the same LLM to simulate multiple student levels.
- Directly valuable for the AI in Education (AIED) field: automated pre-test quality evaluation systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing student simulation to QG evaluation is a novel and intuitive idea, simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing two datasets, 7 baselines, multi-dimension evaluations, human evaluations, and robustness analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, vivid case studies, and rigorous experimental design.
- **Value**: ⭐⭐⭐⭐ Holds practical instructional significance for both AI in education and LLM evaluators, with a simple and easily deployable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues](enhancing_spoken_discourse_modeling_in_language_models_using_gestural_cues.md)
- [\[ACL 2025\] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] Enabling LLM Knowledge Analysis via Extensive Materialization](enabling_llm_knowledge_analysis_via_extensive_materialization.md)

</div>

<!-- RELATED:END -->
