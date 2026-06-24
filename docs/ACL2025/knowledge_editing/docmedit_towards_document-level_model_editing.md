---
title: >-
  [Paper Note] DocMEdit: Towards Document-Level Model Editing
description: >-
  [ACL 2025][Knowledge Editing][Model Editing] This paper proposes the document-level model editing task for the first time and constructs the DocMEdit benchmark containing 37,990 data items and 105,652 editing facts, revealing the severe shortcomings of existing editing methods in long-context, multi-fact parallel editing scenarios.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Model Editing"
  - "Document-Level Task"
  - "Knowledge Updating"
  - "LLM"
  - "Knowledge Graphs"
date: 2026-05-08
content_hash: 3c5adefb44901407
---

# DocMEdit: Towards Document-Level Model Editing

**Conference**: ACL 2025  
**arXiv**: [2505.19572](https://arxiv.org/abs/2505.19572)  
**Code**: [Yes](https://github.com/BITHLP/DocMEdit)  
**Area**: NLP / Knowledge Editing  
**Keywords**: Model Editing, Document-Level Task, Knowledge Updating, LLM, Knowledge Graphs

## TL;DR

This paper proposes the document-level model editing task for the first time and constructs the DocMEdit benchmark containing 37,990 data items and 105,652 editing facts, revealing the severe shortcomings of existing editing methods in long-context, multi-fact parallel editing scenarios.

## Background & Motivation

Model editing aims to correct errors or outdated knowledge in LLMs at a minimal cost. The core issues of existing datasets (ZsRE, CounterFact, MQuAKE) lie in:

**Overly fine output granularity**: Only requiring models to output phrases or sentences, which is severely disconnected from real-world scenarios (generating biographies, updating Wikipedia articles, long-chain reasoning).

**Lack of extrapolation**: Answers can be directly derived from editing facts, without requiring the model to integrate existing knowledge.

**Insufficient context length**: The average target length of existing datasets is only 6-131 words.

These issues cast doubt on the practical usability of existing editing methods. The authors compare existing benchmarks as follows:

| Benchmark | Document-Level | Extrapolation | Multi-Edit | Locality | Avg. Target Length |
|------|--------|--------|--------|--------|-------------|
| ZsRE | ✗ | ✗ | ✗ | ✔ | 12.12 |
| CounterFact | ✗ | ✗ | ✗ | ✔ | 6.65 |
| MQuAKE | ✗ | ✗ | ✔ | ✗ | 10.94 |
| **DocMEdit** | **✔** | **✔** | **✔** | **✔** | **867.62** |

The average target length of DocMEdit is 6.6 times that of the longest existing benchmark.

## Method

### Overall Architecture

The construction of DocMEdit consists of three steps: document change computation, facts collection, and knowledge graph extraction.

### Key Designs

1. **Document Change Computation**:

    - Collect Wikipedia dumps from two timestamps (20231101 and 20241101).
    - Extract the INTRODUCTION section of each document as $y$ and $y'$ before and after the update.
    - Filter out updates that contain only stylistic modifications, retaining meaningful updates that include at least one new entity.
    - Design Motivation: Most updates on Wikipedia are stylistic rather than factual changes.

2. **Facts Collection**:

    - For each sentence, if the entity mentioned is newly introduced in the document update, the sentence is regarded as a fact supporting the update of that entity.
    - Facts are extracted directly from unstructured Wikipedia data (non-triplets), which is closer to real editing scenarios.
    - Each data item contains an average of 2.78 editing facts.

3. **Knowledge Graph Extraction**:

    - Extract knowledge graphs from the source document, target document, and supporting facts, respectively.
    - Constrain relations in the triplets to must be existing relations in Wikidata.
    - Extracted 568,652 entities, 4,804 relations, and 1,411,057 triplets.
    - Convenient for conducting experiments with RAG-based methods (such as IKE and SKEME).

4. **Evaluation Metrics Design**:

    - **Accuracy**: Document-ROUGE (DR), Document-Entity (DE), Edit-ROUGE (ER), Edit-Entity (EE).
    - **Locality**: ROUGE Side Effect (RSE), Entity Side Effect (ESE) — measuring the preservation of unedited parts.
    - **Quality**: Human evaluation of semantic coherence (SC) with a 3-level score.
    - **Efficiency**: Time consumption ($T_i$), memory requirement ($M_e$).

### Problem Formulation

Edit model $M$ to $M'$ such that $M'(x) = y'$, where $y'$ contains original sentences plus new sentences supported by editing facts, while sentences unrelated to the facts remain unchanged.

## Key Experimental Results

### Main Results

| Model | Method | DR↑ | DE↑ | ER↑ | EE↑ | RSE↑ | ESE↑ | SC↑ |
|------|------|-----|-----|-----|-----|------|------|-----|
| Llama2 | w/o Edit | 26.11 | 18.97 | 15.77 | 0.50 | 53.91 | 55.37 | 1.05 |
| Llama2 | FT | 24.78 | 17.95 | 14.65 | 7.17 | 53.76 | 39.22 | 0.60 |
| Llama2 | MEMIT | 19.63 | 9.62 | 15.16 | 2.50 | 40.54 | 34.86 | 0.62 |
| Llama2 | IKE | 19.79 | 26.30 | 22.77 | 12.20 | 43.27 | 35.80 | 1.03 |
| Llama2 | SKEME | 21.08 | **29.34** | **25.75** | **23.92** | **47.31** | **49.22** | 1.00 |
| DeepSeek | SKEME | **37.71** | **37.05** | **29.64** | **54.49** | 59.04 | 88.55 | 1.99 |

### Analysis of Context Length Effects

| Context Length | Short (0-512) | Medium (512-1024) | Long (1024+) |
|-----------|-----------|-------------|-----------|
| Base Model | Partially workable | Difficult | Mostly failed |
| FT/MEMIT | Degrading performance | Failed | Completely failed |
| IKE/SKEME | Further improved | Effective | Almost failed |

### Key Findings

- **All methods perform poorly**: The high DR value of unedited models stems from hallucination rather than genuine knowledge updates.
- **Parameter modification methods (FT, MEMIT)** have severe side effects: significantly degrading the generation quality of LLMs (SC drops from 1.05 to 0.60).
- **RAG-based methods generally outperform parameter modification methods**: SKEME is more robust based on entity retrieval, while IKE's dense retrieval performance degrades rapidly on long facts.
- **All models suffer from severe side effects**: ESE is below 60 across the board, which means more than 40% of entity information is lost.
- **RAG-based methods degrade rapidly as the number of facts increases**: When the number of facts $\ge 5$, FT surprisingly outperforms RAG-based methods.
- **Error Analysis**: 78.4% are hallucinations, 8.6% ignore factual updates, 7.7% exhibit unexpected style changes, and 5.3% misunderstand the facts.

## Highlights & Insights

- **Valuable problem definition**: It extends model editing to the document level for the first time, filling a gap in research.
- The metric design balances accuracy and side effects; the combination of DR/DE and RSE/ESE is more convincing than single metrics.
- Factual update experiments (RQ2b) reveal the degradation issues in sequential editing: parameter modification methods collapse as internal parameters drift away from their initial states.
- The data construction method derived from real Wikipedia updates makes the benchmark closer to practical applications.

## Limitations & Future Work

- Long input and output impose high requirements on LLMs' context windows and computational resources.
- Only the INTRODUCTION sections of Wikipedia are used; longer complete documents have not yet been tested.
- All existing methods perform poorly, and there is a lack of effective proposed solutions.
- Explorable directions: task decomposition, prompt structure and fact positioning adjustments, simultaneous attention to shallow and deep neurons, and management of conflicts between internal and external knowledge.

## Related Work & Insights

- Complementary to FAME (multi-task editing): FAME focuses on multi-hop reasoning, whereas DocMEdit focuses on document-level long outputs.
- The idea of credible update text generation from FRUIT can be extended to model editing scenarios.
- Challenges in document-level NLP tasks (translation, relation extraction, QA) also exist in model editing, and are even more severe.
- Provides subsequent research with a clear benchmark and evaluation protocol.

## Rating

- **Novelty**: 8/10 — The task definition is novel, and the benchmark design is reasonable.
- **Experimental Thoroughness**: 8/10 — Multiple models, multiple methods, and in-depth analysis across four RQs.
- **Writing Quality**: 7/10 — Detailed content, but the formulation expressions are slightly tedious.
- **Value**: 8/10 — Provides an important benchmark and a new perspective for the field of model editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [\[ACL 2025\] The Mirage of Model Editing: Revisiting Evaluation in the Wild](the_mirage_of_model_editing_revisiting_evaluation_in_the_wild.md)
- [\[NeurIPS 2025\] Rethinking Residual Distribution in Locate-then-Edit Model Editing](../../NeurIPS2025/knowledge_editing/rethinking_residual_distribution_in_locate-then-edit_model_editing.md)
- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ACL 2025\] Towards a Principled Evaluation of Knowledge Editors](towards_a_principled_evaluation_of_knowledge_editors.md)

</div>

<!-- RELATED:END -->
