---
title: >-
  [Paper Note] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions
description: >-
  [ACL 2025][LLM (Other)][Unanswerable Questions] This paper proposes UAQFact (13,970 questions), a bilingual dataset of unanswerable questions (UAQs), where each question is annotated with factual knowledge from knowledge graphs. It defines three evaluation tasks to measure the capability of LLMs to distinguish UAQs from answerable questions (ABQs), and to utilize internal/external factual knowledge to handle UAQs. Experiments reveal that LLMs struggle to utilize relevant know…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Unanswerable Questions"
  - "Factual Knowledge"
  - "Knowledge Graphs"
  - "LLM Evaluation"
  - "Bilingual Benchmark"
  - "Refusal Rate"
date: 2026-05-08
content_hash: cc2298e3e0b51c09
---

# UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions

**Conference**: ACL 2025  
**arXiv**: [2505.23461](https://arxiv.org/abs/2505.23461)  
**Authors**: Chuanyuan Tan, Wenbiao Shao, Hao Xiong, Tong Zhu, Zhenhua Liu, Kai Shi, Wenliang Chen (Soochow University & OPPO AI Center)  
**Code**: [github.com/cytan17726/UAQ_Fact](https://github.com/cytan17726/UAQ_Fact)  
**Area**: LLM/NLP  
**Keywords**: Unanswerable Questions, Factual Knowledge, Knowledge Graphs, LLM Evaluation, Bilingual Benchmark, Refusal Rate  

## TL;DR

This paper proposes UAQFact (13,970 questions), a bilingual dataset of unanswerable questions (UAQs), where each question is annotated with factual knowledge from knowledge graphs. It defines three evaluation tasks to measure the capability of LLMs to distinguish UAQs from answerable questions (ABQs), and to utilize internal/external factual knowledge to handle UAQs. Experiments reveal that LLMs struggle to utilize relevant knowledge effectively even when it is already stored.

## Background & Motivation

### Background
LLMs excel in traditional question-answering tasks, but in real-world scenarios, user questions may not have deterministic factual answers—known as unanswerable questions (UAQs). For example, "Who among Nero Caesar's siblings was also the father of Seti I?"—such questions have no factual answers that satisfy the conditions. If LLMs generate hallucinated answers to UAQs, they will mislead users and cause negative consequences.

### Limitations of Prior Work

**Lack of Factual Knowledge Support**: Existing UAQ datasets (e.g., SelfAware, FalseQA, UnknownBench) fetch questions from web scraping, brainstorming, or entity substitution. They only provide answers/labels without supporting factual knowledge, failing to evaluate LLMs' ability to utilize knowledge to handle UAQs.

**English-Only Support**: Existing datasets are solely in English, making them unable to evaluate cross-lingual generalization capabilities.

**Single Evaluation Dimension**: Existing datasets only support UAQ/ABQ binary classification tasks, lacking a deep evaluation of internal and external knowledge utilization.

### Core Idea
Build a bilingual UAQ dataset with auxiliary factual knowledge and define new tasks to deeply evaluate whether LLMs can effectively utilize their stored and externally provided factual knowledge to correctly handle unanswerable questions.

## Method

### Dataset Construction Pipeline

#### 1. Question Type Definition
Define three question types (QTypes):
- **Inter (Intersection)**: Requires the LLM to return the intersection of two sets. For a UAQ, the intersection of the answer sets of the two sub-questions is empty. Example: "Who was both the editor of Enneads and a cast member of The Sixth Sense?"
- **Time (Time-constrained)**: Requires the LLM to answer within given temporal constraints. For a UAQ, the temporal constraint is unsatisfiable. Example: "With which city was Erfurt twinned between 1957 and 1962?" (The actual starting year was 1971)
- **Dilemma (Candidate Answer)**: Provides candidate answers for the LLM to choose from. For a UAQ, all candidate answers are incorrect. Example: "Does Segestes belong to the Mohawk or the Khamti tribe?" (The correct answer Cherusci is not in the options)

#### 2. Factual Triplet Sampling
Sample bilingual factual triplets from the Wikidata knowledge graph as supporting knowledge:
- Query and retrieve 724 properties (such as editor, cast member) and their descriptions.
- Filter properties based on three criteria: comprehensibility, occurrence frequency $\ge 5$, and capability to provide factual knowledge.
- Construct different queries for each QType to retrieve related entities and bilingual labels.

#### 3. Template Generation and Filling
Generate question templates using GPT-3.5 based on property descriptions, manually inspect all 864 templates (per language) to correct or discard erroneous ones, and finally fill the entities into the templates to generate questions.

### Three Evaluation Tasks

#### Task 1: UAQ/ABQ Discrimination
Directly present questions to the LLM to evaluate its fundamental capability to distinguish between UAQs and ABQs.

#### Task 2: Internal Knowledge Utilization Evaluation
First probe whether the LLM stores factual knowledge related to UAQs using multiple-choice questions (Knowledge Passage Rate, KPR), then combine this with Task 1 results to calculate the Knowledge-aware Refusal Rate (KRR):

$$\text{KRR} = \left(1 + e^{-R_\Delta \cdot \text{KPR}^{-1}}\right)^{-1}$$

Where $R_\Delta = R_{ua} - R_{ab}$, with $R_{ua}$ representing the UAQ refusal rate and $R_{ab}$ representing the ABQ refusal rate. A higher KRR indicates a stronger knowledge utilization capability in the LLM.

#### Task 3: External Knowledge Utilization Evaluation
Provide LLMs with carefully designed Chain-of-Thought (CoT) reasoning clues as external knowledge, which include: (1) decomposing the question into sub-questions, (2) providing relevant factual knowledge, and (3) answering based on the prior information. Evaluate whether the LLM can leverage external knowledge to correctly handle UAQs.

### Dataset Statistics
- Total of 13,970 questions (6,985 UAQs + 6,985 ABQs)
- 9,021 entities, 724 properties
- 8,686 knowledge-probing questions + 13,970 reasoning clues
- Entirely bilingual (English + Chinese)
- Manual verification pass rate of 99.2%

## Key Experimental Results

### Task 1: UAQ/ABQ Discrimination Results

| Model | EN $R_{ua}$↑ | EN $R_{ab}$↓ | EN $R_\Delta$↑ | EN Acc↑ | ZH $R_\Delta$↑ | ZH Acc↑ |
|------|--------|--------|---------|------|---------|------|
| Llama3 | 38.80 | 17.91 | 20.89 | 53.21 | 9.95 | 35.88 |
| Mistral0.2 | 62.15 | 31.27 | 30.88 | 54.32 | 7.92 | 19.31 |
| Qwen2.5 | 59.10 | 29.03 | 30.07 | 47.73 | 16.63 | 43.77 |
| GLM4 | 55.05 | 31.22 | 23.83 | 49.63 | 14.13 | 41.53 |
| Gemini-1.5-pro | 66.50 | 12.25 | **54.24** | **69.62** | **38.40** | **53.98** |
| GPT-4o-mini | 85.05 | 42.15 | 42.91 | 50.51 | 18.14 | 47.02 |
| GPT-4 | 85.70 | 22.43 | **63.26** | 66.79 | 34.52 | 51.02 |

- All LLMs achieve a positive $R_\Delta$, indicating a basic ability to distinguish between UAQs and ABQs.
- The best EN $R_\Delta$ is only 63.26 (GPT-4) and ZH is only 38.40 (Gemini-1.5-pro), highlighting the high difficulty of UAQFact.
- Overall $R_\Delta$ in Chinese is lower than in English, suggesting that Chinese UAQs are harder to discriminate.

### Task 2: Internal Knowledge Utilization Results

| Model | EN KPR↑ | EN $R_\Delta$↑ | EN KRR↑ | ZH KPR↑ | ZH KRR↑ |
|------|---------|---------|---------|---------|---------|
| Llama3 | 73.11 | 20.89 | 57.10 | 52.71 | 54.71 |
| Mistral0.2 | 68.92 | 30.88 | 61.02 | 40.21 | 54.91 |
| Qwen2.5 | 70.57 | 30.07 | 60.49 | 60.44 | 56.84 |
| GPT-4 | **81.80** | **63.26** | 68.42 | **83.21** | 60.23 |
| Gemini-1.5-pro | 69.03 | 54.24 | **68.69** | 76.74 | **62.26** |

- LLMs HTML generally achieve a high KPR (68-82% in English), showing they have already stored a large amount of factual knowledge.
- However, the KRR ranges only between 54% and 69%, exposing a significant gap between knowledge storage and knowledge utilization.
- While Gemini-1.5-pro does not have the highest KPR, it achieves the best KRR, demonstrating the highest knowledge utilization efficiency.

### Task 3: External Knowledge Utilization Results

After providing CoT reasoning clues, all LLMs show performance improvements:
- Llama3 shows the most significant improvement: its EN $R_\Delta$ leaps from 20.89 to 74.54 (+53.65).
- Open-source models benefit more than closed-source models (higher relative improvement).
- Closed-source models improve more prominently in the Chinese setting, with average ZH $R_\Delta$ even surpassing EN (73.08 vs 64.96).
- However, even with verified external knowledge, the best EN $R_\Delta$ is only around 75%, falling far short of ideal performance.

## Key Findings

1. **Knowledge Storage $\neq$ Knowledge Utilization**: LLMs store rich factual knowledge (high KPR) but fail to effectively retrieve and apply it when facing UAQs (low KRR), highlighting a clear gulf between "possessing" and "utilizing" knowledge.
2. **Chinese is More Challenging**: Almost all models show a significantly lower $R_\Delta$ in Chinese UAQ discrimination compared to English. Even with similar KPRs, the Chinese KRR generally decreases.
3. **Non-uniform Scaling Effect**: As the Qwen2.5 series scales from 0.5B to 72B, $R_\Delta$ shows an upward trend, but $R_{ua}$ and $R_{ab}$ fluctuate synchronously—meaning that while refusing more UAQs, the models also mistakenly refuse more ABQs.
4. **External Knowledge Helps but is Insufficient**: CoT clues improve performance across all models, but closed-source models tend to give more definitive answers (both $R_{ua}$ and $R_{ab}$ drop simultaneously) when provided with knowledge, which does not necessarily yield a net benefit.
5. **Decoupling of Knowledge Utilization and Quantity**: Mistral0.2 has a lower EN KPR than Qwen2.5 but a higher KRR, suggesting that models with less stored knowledge but higher utilization efficiency can perform better in UAQ discrimination.

## Highlights

- **First Knowledge-Enhanced UAQ Benchmark**: Incorporates factual knowledge from the Wikidata knowledge graph for every question, enabling deep evaluation rather than simple classification.
- **Three-Task Evaluation Framework**: Basic discrimination + internal knowledge utilization + external knowledge utilization, progressively diagnosing the bottleneck in LLMs' capability to handle UAQs.
- **Knowledge-aware Refusal Rate (KRR)**: Innovatively integrates knowledge-probing results with refusal performance into a single metric, enabling fair comparisons across models.
- **Bilingual Design**: Full coverage of English and Chinese, revealing cross-lingual performance differences and offering a new perspective on multilingual LLM evaluation.
- **High-Quality Data**: Based on rigorous Wikidata sampling + GPT-3.5 template generation + manual verification (99.2% pass rate), ensuring highly controlled question quality.

## Limitations & Future Work

- **Lexical Matching Evaluation**: Refusal rates are calculated using keyword matching (detecting refusal, apology, or abstention keywords). Although manual evaluation shows a Cohen's Kappa of 94.90, discrepancies with human judgment remain.
- **Single Source of Knowledge**: Relying solely on Wikidata limits the coverage to its structured attributes, leaving out UAQs that require common-sense reasoning or domain-specific expertise.
- **Limited Question Types**: The three QTypes (Inter/Time/Dilemma) have limited coverage, whereas real-world UAQs exhibit more diverse forms (e.g., false premises, over-specification).
- **Template Generation Quality**: Templates generated by GPT-3.5 suffer from semantic errors and missing slots, making them heavily reliant on manual inspection.
- **Open-Source Model Scale**: Evaluation of open-source LLMs is concentrated around 7B parameters, lacking comparisons with models at the scale of closed-source systems.
- **Knowledge Probing Design**: Probing knowledge via multiple-choice questions may not fully reflect the true depth of knowledge mastery in LLMs.

## Related Work & Insights

- **SelfAware (Yin et al. 2023)**: 3,369 questions, English only, no knowledge support, single-task. In contrast, UAQFact is larger (13,970 questions), bilingual, and contains supporting knowledge.
- **FalseQA (Hu et al. 2023)**: 4,730 questions, based on brainstorming, provides acceptable responses rather than factual answers. In contrast, UAQFact is constructed based on KGs with answers verified by knowledge graphs.
- **UnknownBench (Liu et al. 2023)**: 6,323 questions, constructs UAQs by substituting entities with fictitious ones. In contrast, UAQFact preserves all real-world entities and constructs unanswerability through compositional relations.
- **CREPE (Yu et al. 2022)**: 8,466 questions, web-sourced, without supporting knowledge. In contrast, every question in UAQFact is supported by Wikidata triplets.
- **MMLU/C-Eval**: Evaluates LLMs' internal knowledge but not under UAQ scenarios. UAQFact focuses specifically on knowledge utilization in UAQ settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce factual knowledge into UAQ evaluation; the three-task framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 7 model series, includes full-parameter scaling analysis of Qwen2.5, and provides bilingual comparisons, though it lacks evaluations on some of the latest frontier models.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, clear task definitions, and exceptionally standard table and figure layouts.
- Value: ⭐⭐⭐⭐ — Uncovers the critical insight of the gap between knowledge storage and utilization, providing practical guidance for the research of LLM honesty and reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations](factual_knowledge_in_language_models_robustness_and_anomalies_under_simple_tempo.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)

</div>

<!-- RELATED:END -->
