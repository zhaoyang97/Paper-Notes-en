---
title: >-
  [Paper Note] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][Non-Factoid QA] The proposed Typed-RAG framework achieves type-aware decomposition for non-factoid questions (NFQs). By decomposing complex multi-aspect questions into single-aspect sub-queries, it designs differentiated retrieval and generation strategies tailored to distinct question types (debate, experience, comparison, etc.), thereby significantly improving the performance of RAG in NFQA.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Non-Factoid QA"
  - "RAG"
  - "question type awareness"
  - "multi-aspect decomposition"
  - "Wiki-NFQA"
date: 2026-05-08
content_hash: 7cf49f0b438f5622
---

# Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2503.15879](https://arxiv.org/abs/2503.15879)  
**Code**: Yes ([https://github.com/TeamNLP/Typed-RAG](https://github.com/TeamNLP/Typed-RAG))  
**Area**: NLP / Question Answering Systems / Retrieval-Augmented Generation  
**Keywords**: Non-Factoid QA, RAG, question type awareness, multi-aspect decomposition, Wiki-NFQA

## TL;DR

The proposed Typed-RAG framework achieves type-aware decomposition for non-factoid questions (NFQs). By decomposing complex multi-aspect questions into single-aspect sub-queries, it designs differentiated retrieval and generation strategies tailored to distinct question types (debate, experience, comparison, etc.), thereby significantly improving the performance of RAG in NFQA.

## Background & Motivation

Traditional QA systems primarily deal with factoid questions (e.g., "When was Google founded?"). In contrast, a vast number of real-world information needs are **non-factoid** (NFQ), requiring subjective, multi-faceted, and comprehensive answers. Non-factoid question answering (NFQA) faces three primary challenges:

**High question heterogeneity**: NFQs span multiple types including debate, experience, comparison, reason, and instruction, with each type requiring distinct processing strategies.

**Poor generalization of single-method approaches**: Approaches tailored to specific NFQ types (e.g., handling only "how-to" questions) fail to generalize across all types.

**Homogenization of RAG**: Standard RAG systems generate overly homogeneous responses, lacking the multi-faceted depth required for NFQs.

**Key Challenge**: NFQs are inherently multi-faceted (e.g., a query about whether a product is good may require evaluation from price, quality, and user experience perspectives), whereas existing RAG systems retrieve documents based on a single query and fail to capture this multi-faceted nature.

## Method

### Overall Architecture

The workflow of Typed-RAG consists of:
1. **Type Classifier**: Classifies the NFQ into one of six types.
2. **Multi-aspect Decomposer**: Decomposes the question into single-aspect sub-queries according to its type.
3. **Retrieval + Generation**: Conducts independent retrieval and generation for each sub-query.
4. **Answer Aggregator**: Integrates sub-answers into the final generated response.

### Key Designs

1. **Six NFQ Types and Their Handling Strategies**:

    - **Evidence-based**: Single-aspect and direct; simple RAG is applied with no decomposition.
    - **Comparison**: Extracts comparison purposes (difference/similarity/superior) and keywords, retrieves info separately, and conducts a comparison.
    - **Experience**: Extracts core topic keywords and retrieves diverse viewpoints and experiences.
    - **Reason**: Decomposes into single-aspect sub-queries, retrieves and generates sub-answers individually, and then aggregates (noting potential contradictions between different explanations).
    - **Instruction**: Similar to Reason, but procedural steps typically display higher consistency.
    - **Debate**: Decomposes into sub-queries representing opposing stances and synthesizes a balanced response using a "debate mediator" persona.
    - **Design Motivation**: Distinct question types display fundamental differences in intent, aspect directionality, and contrast of perspectives.

2. **Two Modules of the Multi-aspect Decomposer**:

    - **Single-aspect Query Generator**: Deconstructs multi-aspect NFQs into focused, single-aspect sub-queries.
    - **Keyword Extractor**: Extracts core entities and comparison objectives.
    - These two modules are selectively activated based on the question type (e.g., the decomposer is bypassed for Evidence-based questions).

3. **Wiki-NFQA Dataset**:

    - Constructed from Wikipedia, filtering NFQs based on 6 existing QA datasets.
    - Annotated with a pre-trained NFQ type classifier.
    - Comprises 945 NFQs spanning six types.
    - Evaluated using LINKAGE (an LLM-based ranking framework).

### Loss & Training

- The type classifier leverages the pre-trained classifier from Bolotova et al. (2022).
- The multi-aspect decomposer relies on few-shot prompting and requires no additional training.
- BM25 and Contriever are utilized as retrievers.
- The generator utilizes Llama-3.2-3B and Mistral-7B.

## Key Experimental Results

### Main Results (LINKAGE scores on Wiki-NFQA)

| Method | Evaluation LLM | NQ-NF | SQD-NF | TQA-NF | 2WMH-NF | HQA-NF | MSQ-NF |
|------|---------|-------|--------|--------|---------|--------|--------|
| LLM (Llama-3.2-3B) | Mistral-7B | 0.589 | 0.512 | 0.619 | 0.357 | 0.483 | 0.426 |
| RAG (Llama-3.2-3B) | Mistral-7B | 0.529 | 0.494 | 0.547 | 0.415 | 0.453 | 0.405 |
| **Typed-RAG** (Llama-3.2-3B) | Mistral-7B | **0.766** | **0.649** | **0.706** | **0.454** | **0.562** | **0.536** |
| LLM (Mistral-7B) | GPT-4o mini | 0.466 | 0.422 | 0.592 | 0.318 | 0.397 | 0.338 |
| RAG (Mistral-7B) | GPT-4o mini | 0.441 | 0.382 | 0.545 | 0.289 | 0.356 | 0.308 |
| **Typed-RAG** (Mistral-7B) | GPT-4o mini | **0.841** | **0.744** | **0.777** | **0.399** | - | - |

### Performance Variations Across NFQ Types (Table)

| NFQ Type | Proportion | Characteristics | Handling Strategy |
|---------|------|------|---------|
| Evidence-based | 58.73% | Single-aspect, explicit information | Direct RAG |
| Comparison | 3.81% | Multi-aspect, requires comparison | Keyword extraction + separate retrieval |
| Experience | 4.34% | Multi-aspect, highly subjective | Keyword extraction + diverse retrieval |
| Reason | 23.07% | Multi-aspect, potentially contradictory | Sub-query decomposition + aggregation |
| Instruction | 5.19% | Multi-aspect, consistent procedural steps | Sub-query decomposition + aggregation |
| Debate | 4.87% | Opposing perspectives | Sub-query decomposition + debate mediation |

### Key Findings

1. **Typed-RAG consistently outperforms LLMs and standard RAG**: Achieving superior results across all datasets and evaluation metrics.
2. **Standard RAG does not necessarily outperform pure LLMs**: On several subsets, standard RAG performs worse than pure LLMs, suggesting retrieved documents for NFQs may introduce noise.
3. **Type-aware decomposition is crucial**: Splitting NFQs into single-aspect sub-queries drastically improves the retrieval precision of each sub-query.
4. **Debate type questions exhibit the highest challenge**: Balancing opposing viewpoints remains difficult, resulting in relatively limited improvements even with Typed-RAG.
5. **The performance margin widens under stronger evaluator LLMs (GPT-4o mini)**: Indicating the advantages of Typed-RAG answers are more pronounced under rigorous evaluation.

## Highlights & Insights

- **Simple and effective type-aware approach**: Transitioning NFQA from a "one-size-fits-all" method to high-precision, type-specific pipelines offers a clear and structured methodology.
- **Decomposition into single-aspect sub-queries** is reminiscent of query decomposition in multi-hop QA, but tailored specifically to the diverse multi-faceted nature of NFQs.
- **The debate mediator mechanism** is elegantly designed—allowing the LLM to assume a neutral mediator persona to systematically balance opposing viewpoints in Debate-type questions.
- **The Wiki-NFQA dataset** addresses the absence of a unified benchmark in the NFQA field.
- **The underperformance of standard RAG on NFQA** underscores the necessity of more deliberate and refined architectures for retrieval-augmented generation in specialized scenarios.

## Limitations & Future Work

- The accuracy of the type classifier directly dictates subsequent paths; misclassifications propagate strategy mismatches.
- Wiki-NFQA is relatively small (945 questions) with highly imbalanced type distributions (Evidence-based accounts for 59%).
- Only small open-source LLMs (3B and 7B) have been evaluated; effectiveness on larger-scale models remains unverified.
- The quality of multi-aspect decomposition depends heavily on the design of few-shot prompts.
- Comparisons with alternative state-of-the-art query reformulation/decomposition methods (e.g., Query2Doc, HyDE) were omitted.
- The handling of Debate-type questions requires further refinement.

## Related Work & Insights

- Inherits directly from the NFQ taxonomy classification of Bolotova et al. (2022), from which TUMLU adopts its classifier.
- Complementary to An et al. (2024)'s specialized RAG for "how-to" questions, where Typed-RAG provides a unified overarching framework.
- Integrating the LINKAGE (Yang et al., 2024) evaluation confirms its utility and applicability for evaluating NFQA systems.
- Insight: RAG designs must account for the heterogeneity of query intents; generalized solutions struggle compared to specialized pipelines tailored to question types.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of type-aware decomposition is clear and effective, with a unified framework that surpasses single-type methods.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers multiple datasets but is limited to smaller model scales and lacks comparisons against a broader set of baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — The processing logic for the six types is well-articulated, supplemented with well-designed diagrams/tables.
- **Value**: ⭐⭐⭐⭐ — Delivers both a practical framework and a valuable benchmark for the NFQA research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](../../ACL2026/information_retrieval/disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)

</div>

<!-- RELATED:END -->
