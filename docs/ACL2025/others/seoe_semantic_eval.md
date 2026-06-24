---
title: >-
  [Paper Note] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection
description: >-
  [ACL 2025][Open Domain Event Detection] To address two major pain points in the evaluation of Open Domain Event Detection (ODED)—namely, the lack of real-world representativeness in limited benchmarks and the inability of token-level matching metrics to capture semantic similarity—this work proposes the SEOE framework. It constructs a scalable benchmark containing 564 event types across 7 major domains and introduces an LLM-based semantic $F_1$ evaluation metric.
tags:
  - "ACL 2025"
  - "Open Domain Event Detection"
  - "Semantic Evaluation"
  - "LLM-as-Judge"
  - "Benchmark Construction"
  - "Event Extraction"
date: 2026-05-08
content_hash: e3580b5243258119
---

# SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection

**Conference**: ACL 2025  
**arXiv**: [2503.03303](https://arxiv.org/abs/2503.03303)  
**Code**: [https://github.com/Lyfralston/SEOE](https://github.com/Lyfralston/SEOE)  
**Area**: Others  
**Keywords**: Open Domain Event Detection, Semantic Evaluation, LLM-as-Judge, Benchmark Construction, Event Extraction  

## TL;DR

To address two major pain points in the evaluation of Open Domain Event Detection (ODED)—namely, the lack of real-world representativeness in limited benchmarks and the inability of token-level matching metrics to capture semantic similarity—this work proposes the SEOE framework. It constructs a scalable benchmark containing 564 event types across 7 major domains and introduces an LLM-based semantic $F_1$ evaluation metric.

## Background & Motivation

### Evaluation Dilemma of Open Domain Event Detection

The ODED task requires models to extract events from text, identify event types, and generate type definitions without predefined event schemas. Unlike closed-domain ED, the core challenge of ODED is that the event type space is extremely large and non-exhaustible.

### Two Major Limitations of Existing Evaluation Frameworks

**Problem 1: Insufficient Benchmark Representativeness**
- Existing evaluation benchmarks usually contain limited event types and domain coverage.
- Evaluating ODED models in a zero-shot manner, using predefined ontologies to constrain predicted outputs.
- Failing to accurately reflect model performance in real open-domain scenarios.

**Problem 2: Semantic Insensitivity of Evaluation Metrics**
- Existing metrics are based on token-level matching rules (strict match, partial match, head noun match).
- They cannot capture relationships between semantically similar event types.
- For example, "Attack" and "Military_attack" are semantically close, but token matching classifies them as different.

### Design Motivation

A more representative evaluation benchmark and semantic-level evaluation metrics are required to truly assess the capabilities of ODED models.

## Method

### Overall Architecture

SEOE consists of two components: a **scalable evaluation benchmark** and **semantic evaluation metrics**.

#### Part 1: Evaluation Benchmark Construction (Four-Step Process)

**Step 1: Ontology and Data Integration**
- Integrate the ontologies of mainstream event datasets into a comprehensive ontology.
- Uniformly sample data subsets of each type.
- Problem: Data from one dataset is not checked against the ontologies of other datasets.

**Step 2: Fine-Grained Definition Generation**
- Use GPT-4o to generate fine-grained definitions for each event type.
- Averaging about 109 words, containing detailed descriptions of the event types, role explanations, and requirements for examples.

**Step 3: Potential Event Type Identification**
- GPT-4o identifies possible event types and their definitions in the text.
- A text similarity model computes the similarity between identified types and those in the integrated ontology.
- Types in the Top-$k$ ($k=5$) or with a similarity exceeding 0.8 are considered potential event types.
- **Significantly reduces annotation costs**: Filters out event types unrelated to the text.

**Step 4: Supplementary Annotation**
- Provide potential but unannotated event types, along with their definitions, to GPT-4o for supplementary annotation.
- Ensure each data instance is checked by the integrated ontology.

### Key Designs

#### Enhancing Benchmark Reliability via Nucleus Sampling

Inspired by nucleus sampling in text generation:
1. Repeat LLM annotations for multiple rounds.
2. Sort trigger words in descending order of frequency.
3. Select trigger words whose cumulative frequency reaches a threshold $p$.
4. Use GPT-4o to merge semantically redundant trigger words.

**Experimental Validation** (200 documents, 2277 events, 3 human annotators):

| Strategy | Num of Supp. Triggers | Accuracy |
|------|-------------|--------|
| 1 round | 310 | 80.43% |
| 10 rounds, $p=0.3$ | 290 | **86.90%** |
| 10 rounds, $p=0.5$ | 404 | 85.40% |
| 10 rounds, $p=0.7$ | 522 | 81.80% |

It is found that increasing annotation rounds improves both the quantity and accuracy; a larger $p$ increases the quantity but decreases accuracy (reflecting a trade-off between diversity and accuracy).

#### Semantically Similar Definition Grouping

Utilize fine-grained definitions to calculate the similarity between pairs of event types, grouping those that exceed the threshold into the same group. During evaluation, all types within the group are provided as ontology information, helping the LLM-as-Judge understand semantic correlations.

#### Semantic F1 Evaluation

- The LLM acts as an automatic evaluation agent, taking as inputs: {text, predicted set, predicted definitions, gold set, gold definitions + group information}.
- It outputs a semantic correspondence set $C = \{(p, g)\}$, where prediction $p$ and gold $g$ are semantically matched.
- Calculate semantic precision, recall, and $F_1$.

### Loss & Training

This paper proposes an evaluation framework and does not involve model training. The core "training strategy" is reflected in the pipeline design of benchmark construction, balancing efficiency and reliability through multi-round annotation and nucleus sampling.

## Key Experimental Results

### Main Results

**Correlation between LLM and Human Evaluation** (791 predicted-gold event pairs, 3 human annotators):

| Evaluator | Percent Agreement | Spearman | Cohen's Kappa |
|--------|-------------------|----------|---------------|
| 3 Humans | 95.32 | 79.92 | 79.54 |
| GPT-4o | 94.41±0.14 | 77.50±0.39 | 77.03±0.44 |
| GPT-4o (w/o Groups) | 93.37±0.20 | 74.63±0.52 | 73.73±0.61 |
| GPT-4o (w/o Defs) | 92.97±0.28 | 73.85±0.94 | 72.67±1.00 |
| Claude3-Opus | 93.95±0.03 | 76.03±0.35 | 75.43±0.27 |

**Key Statistics**:
- After supplementary annotation, the number of benchmark events increased by **2.29 times**.
- It ultimately contains **564 event types**, covering **7 major domains**.
- Three versions ($p=0.3/0.5/0.7$) are released, biasing towards accuracy and diversity, respectively.

### Key Findings

1. **High alignment between GPT-4o and human evaluation**: The Percent Agreement reaches 94.41%, which is only about 1% lower than inter-human agreement.
2. **Effectiveness of the grouping module**: Removing Groups drops the Spearman correlation from 77.50 to 74.63 (-2.87).
3. **Importance of fine-grained definitions**: Removing definitions leads to further degradation across all metrics.
4. **ODED remains highly challenging**: Even state-of-the-art models face difficulties in balancing prediction accuracy and diversity.
5. **Benchmark scalability**: The marginal cost of adding new data and event types grows approximately linearly.

## Highlights & Insights

1. **Accurate Problem Identification**: Clearly pinpoints the two core problems of current ODED evaluation, with precisely targeted solutions.
2. **Ingenious Nucleus Sampling Strategy**: Transfers concepts from NLG to annotation quality control, providing an adjustable trade-off between diversity and accuracy.
3. **Excellent Cost-Effectiveness**: Pre-filters irrelevant types using a text similarity model, avoiding astronomical costs associated with annotating all 564 types for every data instance.
4. **Farsighted Multi-Version Release**: Allows future researchers to select versions biased towards accuracy or diversity based on their needs.
5. **Design of the Definition Grouping Module**: Compensates for the lack of hierarchical relationship information in the integrated ontology.

## Limitations & Future Work

1. **Dependency on GPT-4o's Annotation Quality**: LLM annotations may inherit bias, particularly for rare or domain-specific events.
2. **English-Centric**: The benchmark primarily covers English data and event ontologies, leaving multilingual scenarios unexplored.
3. **Limited Event Types**: 564 event types remain a finite subset compared to a true "open domain."
4. **Evaluation Cost**: Utilizing GPT-4o for semantic evaluation incurs API costs.
5. **Selection of Grouping Thresholds**: The optimal similarity threshold may vary across different domains, and adaptive methods are currently lacking.
6. **Event Argument Evaluation Not Covered**: The evaluation focuses strictly on event detection (triggers + types) and has not been extended to event arguments.

## Related Work & Insights

- **LLM-as-Judge Paradigm**: Echoes the trend shown by Zheng et al. (2023), introducing LLM evaluation to more complex IE tasks.
- **UniversalNER (Zhou et al., 2023)**: A similar approach of multi-dataset ontology integration, translated here to the event detection domain.
- **RAEE (Lu et al., 2024a)**: A closed-domain event evaluation framework proposed in previous work, where SEOE serves as its open-domain extension.
- **Benchmark Construction Methodology**: Provides a paradigm reference for constructing evaluation benchmarks in other open-domain IE tasks (e.g., Open NER, Open RE).
- **Insight**: Semantic evaluation metrics should become a standard for evaluation in all open-domain IE tasks.

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 8 | The evaluation framework is systematically and comprehensively designed, with a novel nucleus sampling strategy. |
| Experimental Thoroughness | 8 | Thoroughly validated via human evaluation, ablation studies, and multi-model evaluation. |
| Writing Quality | 8 | Clear structure, with precise correspondences between problem definitions and solutions. |
| Value | 9 | Provides an infrastructure-level contribution to the ODED field. |
| Overall Score | 8 | High-quality evaluation framework work that significantly advances the development of the field. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)
- [\[ACL 2025\] Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[ACL 2025\] RoToR: Towards More Reliable Responses for Order-Invariant Inputs](rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)

</div>

<!-- RELATED:END -->
