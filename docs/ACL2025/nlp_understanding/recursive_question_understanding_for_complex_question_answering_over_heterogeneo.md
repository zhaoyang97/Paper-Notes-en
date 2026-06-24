---
title: >-
  [Paper Note] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data
description: >-
  [ACL2025][NLP Understanding][personal data QA] This paper proposes the ReQAP method, which constructs executable operator trees via recursive question decomposition to achieve complex question answering over heterogeneous (structured + unstructured) personal data, supporting lightweight on-device deployment.
tags:
  - "ACL2025"
  - "NLP Understanding"
  - "personal data QA"
  - "heterogeneous data"
  - "question decomposition"
  - "operator tree"
  - "on-device deployment"
date: 2026-05-08
content_hash: f1dd9019c1532b7a
---

# Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data

**Conference**: ACL2025  
**arXiv**: [2505.11900](https://arxiv.org/abs/2505.11900)  
**Authors**: Philipp Christmann, Gerhard Weikum (Max Planck Institute for Informatics)  
**Code**: [reqap.mpi-inf.mpg.de](https://reqap.mpi-inf.mpg.de/)  
**Area**: NLP Understanding  
**Keywords**: personal data QA, heterogeneous data, question decomposition, operator tree, on-device deployment  

## TL;DR

This paper proposes the ReQAP method, which constructs executable operator trees via recursive question decomposition to achieve complex question answering over heterogeneous (structured + unstructured) personal data, supporting lightweight on-device deployment.

## Background & Motivation

- **Growing demand for personal data management**: User devices generate massive amounts of daily data (calendars, fitness logs, shopping records, streaming history, etc.), and users require convenient querying capabilities over this heterogeneous data.
- **Data privacy as a core constraint**: Sensitive personal data must be processed entirely on local devices, which strictly limits computational power and memory resources.
- **Limitations of two dominant paradigms in existing methods**:
    - **Verbalization (RAG)**: Serializes all data into text to feed into LLMs. However, it exceeds context windows when relevant events reach hundreds, and LLMs struggle with complex operations like aggregation and grouping.
    - **Translation (NL2SQL/CodeGen)**: Translates questions into SQL queries. However, it cannot handle unstructured text (e.g., email bodies, social media posts) and relies on complete database schemas.
- **Key Challenge**: How to achieve complex analytical question-answering capabilities over heterogeneous data (structured and unstructured) while ensuring lightweight on-device deployment.

## Method

### Overall Architecture: ReQAP

ReQAP bridges the verbalization and translation paradigms, operating in two phases:

**Phase I: QUD (Question Understanding and Decomposition)**

- Performs **recursive decomposition** of user questions to generate executable operator trees.
- **Key Innovation**: Instead of generating the entire operator tree at once (one-shot, which is error-prone), it makes multiple LLM calls. Each call generates a partial operator tree, leaving sub-questions for subsequent recursive calls.
- **Training Pipeline**: First, a large model (GPT-4o) is used with In-Context Learning (8-shot) to generate (question, operator tree) pairs, which are then distilled into a 1B small model (LLaMA-3.2-1B) for on-device inference.

**Phase II: OTX (Operator Tree Execution)**

- Executes the generated operator tree from bottom to top. Core operators include:

| Operator | Function |
|------|------|
| **RETRIEVE** | Retrieves matching events from personal data sources (core operator) |
| **EXTRACT** | Extracts specified key-value pairs from events (core operator) |
| JOIN | Connects two sets of event lists based on conditions |
| GROUP_BY | Groups events by a specified key |
| FILTER | Filters events based on conditions |
| MAP / APPLY | Applies function transformations |
| ARGMIN/MAX, SUM, AVG | Aggregates operations |

### RETRIEVE Operator (5-Step Pipeline)

1. **SPLADE Pre-filtering**: Sparse retrieval to obtain a candidate event pool (aiming for high recall).
2. **Schema Discovery**: Identifies high-frequency key-value patterns in the candidate pool.
3. **Schema Classification**: Evaluates schema relevance using a Cross-encoder to classify patterns as fully relevant, fully irrelevant, or partially relevant.
4. **Event Classification**: Conducts binary classification on each event in the partially relevant schemas.
5. **De-duplication**: Merges events with overlapping times (preventing double-counting the same event across calendar, email, or social media).

### EXTRACT Operator

- Uses a small-scale seq2seq model (BART-base) to extract semantic information from unstructured text.
- *Example*: Extracting `cuisine="Italian"` from the email body text containing "pizza oven".
- **Efficiency Optimization**: Constructs a static lookup map for keys that cover $\ge 70\%$ of inputs to avoid redundant inference.

### Data Model

All data sources are unified into a **chronologically sorted event list**, where each event is a key-value dictionary covering calendar, email, social media, fitness, shopping, streaming, etc.

## Key Experimental Results

### PerQA Benchmark

The authors constructed the PerQA dataset: containing 20 fictional characters with approximately 40K events each, and 3,567 complex questions.

**Main Results (Table 3): Hit@1 / Rlx-Hit@1 on PerQA Test Set**

| Method | GPT-4o (>100B) | LLaMA-3.3 (70B) | SFT (1B) |
|------|---------------|-----------------|----------|
| RAG | 0.149 / 0.20 | 0.123 / 0.18 | 0.029 / 0.06 |
| CodeGen | 0.319 / 0.44 | 0.239 / 0.33 | 0.315 / 0.47 |
| **ReQAP** | **0.386 / 0.52** | **0.322 / 0.46** | **0.380 / 0.53** |

- All ReQAP variants significantly outperform the baselines (McNemar test, $p < 0.05$).
- The 1B SFT variant achieves performance close to GPT-4o (Hit@1: 0.380 vs 0.386) and even achieves the best Rlx-Hit@1 (0.53).

**Ablation Study (Table 5): PerQA Dev Set**

| Variant | Hit@1 | Rlx-Hit@1 |
|------|-------|-----------|
| ReQAP (SFT) | **0.396** | **0.54** |
| w/o Recursive Decomposition (one-shot) | 0.356 | 0.50 |
| w/o Cross-encoder (SPLADE-only) | 0.269 | 0.36 |
| w/o EXTRACT (key matching only) | 0.138 | 0.23 |

The removal of the EXTRACT operator causes the largest performance drop (~-65%), demonstrating that extracting information from unstructured text is a crucial capability.

**Performance Across Different Question Complexities (Table 4, GPT-4o)**

| Question Type | RAG | CodeGen | ReQAP |
|----------|-----|---------|-------|
| Ordering | 0.167 | 0.440 | **0.529** |
| Grouping | 0.172 | 0.444 | **0.537** |
| Temporal | 0.129 | 0.290 | **0.417** |
| Aggregation | 0.130 | 0.228 | **0.296** |
| Join | 0.073 | 0.176 | **0.236** |
| Multi-source | 0.196 | 0.237 | **0.365** |

### User Study

20 undergraduate students used their actual personal data in an offline Docker environment. 28% of the answers were completely correct, and 41% were nearly correct. 94% of use cases mapped to isomorphic operator trees in PerQA, validating the representativeness of the benchmark design.

## Highlights & Insights

- **Elegant recursive decomposition strategy**: Mitigates error accumulation in one-shot generation of complex operator trees by breaking the task down into multiple recursive steps, each producing simpler subtrees.
- **On-device deployability**: The 1B variant achieves competitive performance with GPT-4o, with the entire pipeline designed considering strict computational constraints.
- **RETRIEVE 5-step pipeline**: Bridges high recall and high efficiency, significantly reducing computational load via source pruning and schema classification.
- **EXTRACT operator bridging the structured/unstructured divide**: Extracts structured fields online from unstructured texts like emails and social media posts.
- **Robust privacy protection**: During the user study, the system was deployed offline with Docker, ensuring personal data never left local devices.

## Limitations & Future Work

- **Limited data modalities**: Currently supports only specific structured/text sources (calendar, email, streaming, etc.), lacking support for multimodal data such as photos and location trajectories.
- **QUD as the primary bottleneck**: Error analysis reveals that 50% of the failures originate from incorrect operator tree generation.
- **Evaluation limitations**: Assessed only on the synthetic PerQA dataset and a small-scale user study (20 participants), lacking large-scale validation in real-world scenarios.
- **Room for improvement in analytical queries**: The best Hit@1 for aggregation tasks remains below 40%, and join-heavy queries achieve only 23.6%.

## Related Work & Insights

- **TimelineQA** (Tan et al., 2023): The closest pioneering work, but limited by templates (42 tasks) and insufficient data diversity. ReQAP achieves a substantial performance advantage on this benchmark (SFT: 0.313 vs 0.135).
- **Text2SQL** (Li et al., 2024; Liu et al., 2024b): Performs well in purely structured scenarios but fails to handle unstructured text.
- **RAG-based methods** (Oguz et al., 2022; Badaro et al., 2023): Effective given sparse evidence, but underperform on intensive aggregation and numerical computation tasks.
- **Question decomposition methods** (Jia et al., 2024; Saeed et al., 2024): Designed for localized, specific scenarios and not generalized to heterogeneous data QA.

## Rating

- Novelty: ⭐⭐⭐⭐ (Recursive decomposition + operator tree + RETRIEVE/EXTRACT bridging the two major paradigms)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Self-created benchmark + ablations + multi-scale model comparison + user study; lacks extensive external datasets)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear organization, unified running example, and intuitive diagrams)
- Value: ⭐⭐⭐⭐ (Personal data QA is a highly practical demand, and the edge-deployment strategy offers solid industrial value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Synthesizing Data for Context Attribution in Question Answering](on_synthesizing_data_for_context_attribution_in_question_answering.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)

</div>

<!-- RELATED:END -->
