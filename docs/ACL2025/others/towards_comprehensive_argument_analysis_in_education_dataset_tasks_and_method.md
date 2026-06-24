---
title: >-
  [Paper Note] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method
description: >-
  [ACL 2025][Argument Mining] This paper proposes an annotation scheme for Chinese high school argumentative essays, featuring 14 fine-grained argumentation relation types across two dimensions: vertical (argumentation relations) and horizontal (discourse relations). It establishes a comprehensive benchmark covering three tasks: argumentative component detection, relation prediction, and automated essay scoring.
tags:
  - "ACL 2025"
  - "Argument Mining"
  - "Argumentation Relations"
  - "Educational Evaluation"
  - "Argumentative Essay Analysis"
  - "Fine-grained Annotation"
date: 2026-05-08
content_hash: 38b53556d8fb656f
---

# Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method

**Conference**: ACL 2025  
**arXiv**: [2505.12028](https://arxiv.org/abs/2505.12028)  
**Code**: None  
**Area**: Other  
**Keywords**: Argument Mining, Argumentation Relations, Educational Evaluation, Argumentative Essay Analysis, Fine-grained Annotation

## TL;DR

This paper proposes an annotation scheme for Chinese high school argumentative essays, featuring 14 fine-grained argumentation relation types across two dimensions: vertical (argumentation relations) and horizontal (discourse relations). It establishes a comprehensive benchmark covering three tasks: argumentative component detection, relation prediction, and automated essay scoring.

## Background & Motivation

Argument Mining aims to automatically extract structured argumentative information from unstructured text. However, existing research on argumentative relations remains limited to simple binary classifications of "support" and "attack," which fail to capture the complex argumentation strategies and patterns in real argumentative essays. Specific problems include:

**Relation types are overly simplistic**: Most argument mining studies only classify relations into support and attack, failing to characterize argumentation strategies (e.g., argumentative examples, argumentative citations) and argumentation modes (e.g., hypothetical argumentation, metaphorical argumentation).

**Fragmented domains**: Most existing research is concentrated in non-educational domains (e.g., online forums, academic literature) and primarily targets English and German.

**Lack of connection between tasks**: Argumentative component detection, relation prediction, and quality assessment are often studied independently, lacking a systematic exploration of their connections.

## Method

### Overall Architecture

Based on the CEAMC corpus (226 Chinese high school argumentative essays), the authors propose a fine-grained relation annotation scheme from both vertical and horizontal dimensions, and conduct experiments on three core tasks.

### Key Designs

1. **Vertical Dimension (Argumentation Relations) — 10 Types**:

    - **Stance-based** (3 types): Positive, Negative, Comparative
    - **Evidence-based** (2 types): Example, Citation
    - **Discourse-based** (5 types): Background, Detail, Restatement, Hypothetical, Metaphorical
    - **Design Motivation**: Attack relations rarely occur in argumentative essays within educational scenarios (students aim to argue for their own views rather than attack others). Therefore, they are replaced by three stance-based relations: Positive, Negative, and Comparative.

2. **Horizontal Dimension (Discourse Relations) — 4 Types**:

    - Coherence, Progression, Contrast, Concession
    - **Design Motivation**: Focus on the logical relations between argumentative components of the same category, such as how multiple premises jointly support a main claim.

3. **Three Experimental Tasks**:

    - **Argumentative Component Detection**: Sentence-level classification using IOB tagging to represent span information.
    - **Relation Prediction**: Argument pair classification, predicting the relation type between two argumentative components (multi-label classification).
    - **Automated Essay Scoring**: A four-class classification task evaluating the overall quality of argumentative essays.

### Loss & Training

- PLMs use BERT-Base-Chinese and Chinese-RoBERTa-wwm-ext with the AdamW optimizer and a learning rate of $2\times 10^{-5}$.
- LLMs use Qwen2-7B, DeepSeek-R1-Distill-Qwen-7B, and ChatGLM-4-9b, with LoRA fine-tuning ($\text{rank}=8$, $\text{dropout}=0.1$, learning rate $5\times 10^{-5}$).
- A negative sampling strategy is used in relation prediction, where several unrelated arguments are randomly selected as negative samples for each argumentative component.
- All experiments are conducted on a single NVIDIA RTX 3090 GPU.

## Key Experimental Results

### Argumentative Component Detection (Table)

| Model | P(%) | R(%) | F1(%) |
|------|------|------|-------|
| BERT | 40.05 | 47.83 | 43.59 |
| RoBERTa | 46.34 | 51.30 | 48.69 |
| Qwen (SFT) | 57.40 | 56.23 | 56.81 |
| DeepSeek (SFT) | 53.23 | 50.14 | 51.64 |
| ChatGLM (SFT) | **58.17** | **58.84** | **58.50** |
| GPT-4 (0-shot) | 29.50 | 34.20 | 31.68 |
| GPT-4 (3-shot) | 32.66 | 33.04 | 32.85 |

### Relation Prediction (1 negative sample/argument, Table)

| Model | Micro-F1 | Macro-F1 | Pos.-F1 |
|------|----------|----------|---------|
| BERT | 67.67 | 16.45 | - |
| RoBERTa | - | - | - |
| Qwen (SFT) | Comparable | Significantly higher | Significantly higher |
| ChatGLM (SFT) | Comparable | Significantly higher | Significantly higher |
| GPT-4 (0-shot) | Very low | Very low | - |

### Key Findings

1. **LLM SFT significantly outperforms PLM**: In argumentative component detection, ChatGLM-9B SFT improves the F1-score by approximately 10 percentage points compared to RoBERTa, validating the scaling effect.
2. **Poor zero/few-shot performance of GPT-4**: GPT-4 lags significantly behind SFT methods in both tasks, highlighting the importance of domain-specific fine-tuning data.
3. **GPT-4 bias in relation prediction**: It tends to misclassify a large number of negative samples as positive, as its pre-trained knowledge of "relations" exceeds the scope of the argumentative relations defined in this paper.
4. **Varying impact of negative sample size**: ChatGLM achieves optimal performance with 3 negative samples, whereas RoBERTa performs the worst under this configuration.
5. **Fine-grained argumentative annotation benefits scoring**: Incorporating argumentative components and relation information into the essay scoring input improves scoring performance.

## Highlights & Insights

- Extends argumentative relations from simple support/attack to 14 fine-grained types, substantially enriching the expressive capability of argumentative structures.
- The empirical observation that "attack" relations rarely occur in educational scenarios is valuable—students write argumentative essays using more positive, negative, and contrastive strategies.
- The two-dimensional (vertical and horizontal) design integrates the advantages of both argumentative analysis and discourse relation analysis.
- The annotation consistency is reasonable (relation annotation IAA $\kappa = 0.68$). The dataset size of 226 essays and 4,837 relations is above average in educational NLP.
- Explores the bidirectional influence between writing quality and argumentative component detection / relation prediction.

## Limitations & Future Work

- The data scale is relatively small (only 226 essays), which may limit the generalization ability of models.
- Only covers Chinese high school argumentative essays, and cross-lingual and cross-domain applicability remains unverified.
- Severe class imbalance exists for some relation types (e.g., only 6 cases of hypothetical argumentation and 31 cases of metaphorical argumentation).
- The choice of negative sampling strategy in relation prediction has a significant impact on experimental results but remains under-explored.
- An end-to-end argumentative structure parsing system has not been provided.

## Related Work & Insights

- Compared to the argumentative structure prediction by Stab & Gurevych (2017), this work significantly enhances the granularity of relation types.
- Combined with the discourse relation theory of RST (Mann & Thompson, 1988), it introduces discourse analysis tools to argument mining.
- Echoing the Chinese discourse relation framework of Wu et al. (2023), some relations from their four-level, thirteen-label framework are integrated into argumentation analysis.
- Insight: Argumentation analysis in educational scenarios needs to consider cultural and linguistic differences, as the argumentative patterns of Chinese essays differ significantly from English ones.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The proposal of 14 fine-grained relation types is innovative, and the design of the vertical and horizontal dimensions is reasonable.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments are conducted on all three tasks, but the dataset size is small, with extremely few samples for certain relation types.
- **Writing Quality**: ⭐⭐⭐⭐ — The annotation scheme is clearly described, and the experimental design is systematic.
- **Value**: ⭐⭐⭐ — Provides reference value for educational NLP and argument mining, though the data scale limits its overall impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)
- [\[ACL 2025\] Knowledge Tracing in Programming Education Integrating Students' Questions](knowledge_tracing_in_programming_education_integrating_students_questions.md)
- [\[ACL 2025\] DeepRTL2: A Versatile Model for RTL-Related Tasks](deeprtl2_a_versatile_model_for_rtl-related_tasks.md)

</div>

<!-- RELATED:END -->
