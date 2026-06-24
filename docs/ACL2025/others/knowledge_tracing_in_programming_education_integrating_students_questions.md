---
title: >-
  [Paper Note] Knowledge Tracing in Programming Education Integrating Students' Questions
description: >-
  [ACL 2025][Knowledge Tracing] This paper proposes the SQKT (Students' Question-based Knowledge Tracing) model, which is the first to integrate students' questions and automatically extracted skill information into knowledge tracing. It predicts students' completion of subsequent programming problems in programming education, achieving an in-domain AUC improvement of up to 33.1%.
tags:
  - "ACL 2025"
  - "Knowledge Tracing"
  - "Programming Education"
  - "Student Questions"
  - "Skill Extraction"
  - "Adaptive Learning"
date: 2026-05-08
content_hash: 156e575dfa599e25
---

# Knowledge Tracing in Programming Education Integrating Students' Questions

**Conference**: ACL 2025  
**arXiv**: [2502.10408](https://arxiv.org/abs/2502.10408)  
**Code**: None (to be released after paper publication)  
**Area**: Others (AI in Education / Knowledge Tracing)  
**Keywords**: Knowledge Tracing, Programming Education, Student Questions, Skill Extraction, Adaptive Learning

## TL;DR

This paper proposes the SQKT (Students' Question-based Knowledge Tracing) model, which is the first to integrate students' questions and automatically extracted skill information into knowledge tracing. It predicts students' completion of subsequent programming problems in programming education, achieving an in-domain AUC improvement of up to 33.1%.

## Background & Motivation

Knowledge Tracing (KT) aims to monitor students' knowledge states and predict their future performance. In programming education, KT faces unique challenges:

**Complexity of Code Submissions**: Programming tasks are complex, with multiple correct solutions for the same problem; evaluating student ability requires extracting signals from unstructured and noisy source code.

**Limitations of Prior Work in traditional Q-matrix methods**: Manually annotating the skills (knowledge components) required for each problem is time-consuming and struggles to capture the full spectrum of skills actually utilized by students.

**Neglected Informational Value of Student Questions**: Student questions directly reflect areas of confusion and depth of understanding, providing a clearer indication of learning states than code submissions alone.

With the growing popularity of online learning platforms (e.g., Moodle, Canvas), data on student queries and teacher-student interactions is increasingly abundant. However, existing KT models completely ignore this valuable signal. Particularly in programming education, student questions often reveal conceptual understanding and reasoning processes that are difficult to capture through code submissions alone.

## Method

### Overall Architecture

The SQKT model takes the following input sequences to predict whether a student will correctly solve the next programming problem:

- **Historical Part**: Descriptions of previous problems attempted by the student, code submissions, student questions, and extracted skill information.
- **Target Part**: The description of the next problem and the required skills.

The model architecture consists of four core components:
1. Multi-feature input embedding layer
2. Skill extraction system
3. Fusion layer
4. Multi-head self-attention prediction layer

### Key Designs

**Student Question Embedding (Component A)**:
- Uses the CodeT5 model to encode student questions, as it understands both natural language and code syntax.
- Student questions contain two types of information: (a) natural language questions—clarifying concepts or strategies, and (b) code questions—targeting specific lines of code or errors.
- CodeT5 is fine-tuned through an auxiliary task (generating educator responses) to ensure that the question embeddings capture the core of the student's confusion and how an educator would respond.
- If no question exists, a zero vector is used.

**Automatic Skill Extraction System (Component B)**:
- Defines a skill framework comprising **36 core Python concepts + 19 Python error types**.
- Motivation for including error types: Errors reveal students' understanding and misconceptions, which are directly related to learning gaps.
- Uses GPT-4o to generate rule-based skill extraction scripts: GPT-4o is provided with around 20 annotated examples and a predefined skill list to output extraction scripts that can be applied in batch.
- Reason for choosing a rule-based method over online GPT API calls: High precision and consistency.
- Validation: Achieves Precision 0.85, Recall 0.88, and F1 0.86 on 100 samples, with an inter-annotator agreement (Cohen's kappa) of 0.98.

**Bidirectional Skill Application**:
- Extracts skills that a student is struggling with from their questions.
- Extracts skills required for the target problem from reference answers.
- Aligns both to enhance prediction accuracy.

**Code Embedding (Component C)**: Encodes student code submissions using CodeBERT.

**Problem Embedding (Component D)**: Encodes problem descriptions using BERT-base.

**Fusion Layer (Component H)**:
- Projects all embeddings into a unified 512-dimensional space.
- Employs a triplet loss to pull embeddings from the same submission closer and push embeddings from different submissions or concepts further apart.

### Loss & Training

**Triple Loss Function**:

$$L_{total} = L_{pred} + L_{question} + \lambda L_{triplet}$$

1. **Prediction Loss $L_{pred}$**: Binary cross-entropy, predicting student success/failure on the target problem.
2. **Question Auxiliary Loss $L_{question}$**: Negative log-likelihood, fine-tuning CodeT5 to generate educator responses.
3. **Triplet Loss $L_{triplet}$**: Unifies the heterogeneous embedding space.
    - Anchor: Code embedding of the current problem.
    - Positive: Description or student question embedding of the current concept/problem.
    - Negative: Random problem description or question embedding.

**Training Configurations**:
- Optimizer: Adam with a learning rate of 3e-5.
- Batch size: 16, Dropout: 0.1.
- Triplet loss weight $\lambda = 1.0$.
- 6 self-attention layers, with max-pooling followed by a classification head.
- GPU: NVIDIA A100 80GB, training took approximately 1.5 hours for in-domain scenarios and 3 hours for cross-domain scenarios.

## Key Experimental Results

### Main Results

**Datasets**: Gathered from a South Korean online programming education platform (Jan 2022 to Apr 2024), containing 4 Python courses:
- Python Basic (PB): 48 problems, 160 students
- First Python (FP): 60 problems, 8,141 students
- Algorithm (Algo): 32 problems, 77 students (data-scarce)
- Python Introduction (PI): 227 problems, 1,092 students

**In-Domain Experimental Results (AUC%)**:

| Model | Python Introduction | First Python | Python Basic |
|------|-------------------|-------------|-------------|
| KTMFF | 70.2 | 69.4 | 78.0 |
| KTMFF+ | 72.6 | 71.7 | 80.7 |
| OKT | 60.3 | 65.8 | 65.0 |
| OKT+ | 66.7 | 66.7 | 78.4 |
| **SQKT** | **93.4** | **90.3** | **93.3** |

SQKT achieves an absolute AUC improvement of 12.6–20.8 percentage points compared to the best baseline (KTMFF+).

### Ablation Study

**Component Ablation (Python Introduction Course)**:

| Configuration | AUC (%) | ACC (%) | F1 (%) |
|------|---------|---------|--------|
| SQKT Full | 93.4 | 89.2 | 88.4 |
| - Questions (All-ones vector) | 91.3 | 86.3 | 89.9 |
| - Questions (Skills only) | 90.9 | 86.2 | 88.7 |
| - Skills (Questions only) | 89.7 | 81.3 | 83.1 |
| - Questions and Skills | 85.4 | 80.7 | 82.7 |

Key Findings:
- Replacing actual question content with an all-ones vector decreases performance $\rightarrow$ the specific content of the questions matters.
- Utilizing only skills or only questions is less effective than combining both $\rightarrow$ the two signals exhibit synergistic effects.
- Removing both leads to an 8 percentage point drop in AUC $\rightarrow$ both questions and skills contribute significantly.

**Influence of Auxiliary Losses**:
- Removing the question loss: AUC drops by 0.7 / 0.5 / 1.4 percentage points on the three courses, respectively.
- Removing the triplet loss: AUC drops by 1.0 / 2.9 / 1.8 percentage points (with the largest impact observed on First Python).

### Key Findings

1. **Student Questions act as High-quality Predictive Signals**: Even when added to baseline models (KTMFF+ vs. KTMFF, OKT+ vs. OKT), they consistently yield improvements.
2. **Strong Cross-domain Generalization**: In content structure migration scenarios, models utilizing question data achieve a 45.3% absolute AUC improvement.
3. **Alleviating Data Scarcity**: On the Algorithm course (containing only 300 test samples), the cross-domain model outperforms the in-domain model by 11.4% AUC.
4. **Content Matters More than Question Presence**: An all-ones vector (indicating only "question existence") is less effective than actual question embeddings.

**Error Analysis**: Among 60 mispredictions:
- Complexity (55.6%): Challenges in parsing mixed-language syntax.
- Confusion (40.7%): Code errors are unrelated to student questions.
- Ambiguity (22.2%) and Incompleteness (29.6%): Clearer context is required.

## Highlights & Insights

- **First to Integrate Student Questions into Knowledge Tracing**: Fills a long-overlooked gap in the information sources utilized by the KT field.
- **GPT-driven Automated Skill Extraction**: Replaces manual annotation with GPT-4o-generated rule scripts, balancing interpretability and scalability.
- **Ingenious Auxiliary Task Design**: Enhances the informational density of question embeddings by predicting educator responses.
- **Comprehensive Cross-domain Evaluation**: Two cross-domain scenarios (content/structure migration and data-scarce generalization) validate the practical utility of the model.

## Limitations & Future Work

- **Lack of Question Preprocessing**: Real-world classroom questions are often highly noisy and have not been preprocessed.
- **Rule-based Skill Extractor**: While ensuring interpretability, scalability remains limited; a hybrid approach might have been more optimal.
- **Dataset Limited to Korean and Python**: The generalization to other languages and programming languages remains unverified.
- **Text-only Queries**: Multimodal questions such as screenshots and images are not processed.
- **Limited Data Scale**: The Algorithm course consists of only 77 students and 32 problems.

## Related Work & Insights

- Inherits the tradition of knowledge tracing from DKT (Piech et al., 2015) and BKT (Corbett and Anderson, 1994).
- Compared with Code-DKT (Shi et al., 2022), it introduces student questions in addition to utilizing code features.
- GPT-automated skill annotation offers a scalable alternative to traditional Q-matrix methods.
- Insights for adaptive learning systems: Teacher-student interaction data should be heavily utilized to personalize learning paths.

## Rating

- **Novelty**: ★★★★☆ (Integrating student questions is a novel approach, though the overall architecture is relatively standard)
- **Experimental Thoroughness**: ★★★★☆ (In-domain, cross-domain, and ablation studies are comprehensive, though the data scale is small)
- **Value**: ★★★★☆ (Direct practical value for online programming education platforms)
- **Writing Quality**: ★★★★☆ (Clear structure and rich tables/figures)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems](../../ACL2026/others/automated_knowledge_component_generation_for_interpretable_knowledge_tracing_in_.md)
- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)

</div>

<!-- RELATED:END -->
