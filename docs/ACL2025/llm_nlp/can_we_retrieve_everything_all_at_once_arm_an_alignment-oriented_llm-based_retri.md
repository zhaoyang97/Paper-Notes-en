---
title: >-
  [Paper Note] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method
description: >-
  [ACL 2025][LLM (Other)][Retrieval-Augmented Generation] Proposes ARM (Alignment-oriented Retrieval Method), which integrates three modules—information alignment (N-gram constrained decoding), structure alignment (MIP solver to reason about relationships between data objects), and self-verification aggregation—into the LLM decoding process to retrieve all required data objects "all at once." It significantly outperforms standard RAG (up to +5.2pt) and agentic RAG/ReAct (up to…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Retrieval-Augmented Generation"
  - "Complex QA"
  - "Alignment-based Retrieval"
  - "Constrained Decoding"
  - "Mixed Integer Programming"
date: 2026-05-08
content_hash: a0334f3e01fbb4d6
---

# Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method

**Conference**: ACL 2025  
**arXiv**: [2501.18539](https://arxiv.org/abs/2501.18539)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Retrieval-Augmented Generation, Complex QA, Alignment-based Retrieval, Constrained Decoding, Mixed Integer Programming

## TL;DR

Proposes ARM (Alignment-oriented Retrieval Method), which integrates three modules—information alignment (N-gram constrained decoding), structure alignment (MIP solver to reason about relationships between data objects), and self-verification aggregation—into the LLM decoding process to retrieve all required data objects "all at once." It significantly outperforms standard RAG (up to +5.2pt) and agentic RAG/ReAct (up to +19.3pt) on the Bird and OTT-QA datasets.

## Background & Motivation

Real-world complex open-domain questions often require information from multiple heterogeneous sources (text, tables, etc.) to answer. For example, "What is the highest proportion of K-12 students eligible for free lunch among the most populous counties in California?" might require 1 article + 3 joinable tables.

**Limitations of Prior Work**:

**Standard RAG + Query Decomposition**: The LLM decomposes the question into sub-queries and retrieves for each sub-query. However, the decomposition process is blind to what data exists in the collection and how they are organized, easily missing key objects (such as bridging tables) that are not explicitly mentioned.

**Agentic RAG (ReAct)**: Iteratively decides the next step based on previous retrieval results in each round. The formulation suffers from: (a) search driven by "what information is missing" rather than "what data is available," which is inefficient; (b) inability to jointly optimize sequential steps, where a single error can lead to reasoning derailment; and (c) high cost due to numerous LLM calls.

**Core Idea**: Retrieving answers for complex questions requires aligning the question with the organizational structure of the data collection—performing not only semantic matching but also reasoning over the connection relationships between data objects.

## Method

### Overall Architecture

ARM models retrieval as a generative problem: the LLM outputs a "reasoning process" through a single decoding pass to locate all required data objects. The decoding process consists of three alignment steps:
1. **Information Alignment**: Extracts key information and aligns it with N-grams in the data collection.
2. **Structure Alignment**: Reasons about the connection relationships of the identified data objects and supplements missing bridging objects.
3. **Self-Verification & Aggregation**: The LLM verifies the relevance of the objects and aggregates the results via beam search.

### Key Designs

**Indexing**: Unifies tables and articles as textual data objects, where each object is chunked, embedded, and mapped to a set of N-grams ($N=1\sim3$). Tables are serialized using name + header + description + rows.

**Information Alignment — Constrained Beam Decoding**:
- The LLM first extracts keywords from the question.
- It then aligns the keywords with the dataset's N-grams using constrained decoding.
- A suffix tree is used to track valid token continuations.
- Decoded N-grams are used with BM25 to search for relevant chunks, and final scores are calculated by combining semantic embedding similarity.

**Structure Alignment — Mixed Integer Programming (MIP)**:

$$\arg\max \sum_i R_i b_i + \sum_{i,j} C_{ij} c_{ij}$$

where $R_i$ is the relevance (cosine similarity) of an object to the question, $C_{ij}$ is the compatibility (semantic + exact value matching) between objects, and $b_i, c_{ij} \in \{0,1\}$ are decision variables. Constraints ensure that $k$ connected objects are selected. A Gurobi solver is utilized.

**Multi-Draft Expansion**: Since the initial set from information alignment might be incomplete, the candidate object set is iteratively expanded (adding the top-$k$ most compatible objects at each step, repeated for $l$ steps). MIP is then run on different expanded sets to generate multiple drafts.

**Self-Verification — LLM as a Verifier**:
- Drafts are injected into the LLM decoding process (constrained decoding forces the output to be draft contents).
- The LLM comprehensively evaluates all decoded information to select the final data objects.
- Constrained decoding ensures that selected objects strictly exist in the drafts.

**Beam Search Aggregation**: Multiple beams produce multiple reasoning paths, which are aggregated through weighted voting: voting weight = average token logits of the object name, and vote count = occurrence frequency across beams (normalized using softmax). The final confidence is the weighted sum of both forces.

### Loss & Training

ARM does not require extra training for the LLM—it directly uses Llama-3.1-8B-Instruct for inference. The core innovation lies in the decoding strategy rather than training. A 3-shot ICL prompt guides the LLM to perform retrieval reasoning.

## Key Experimental Results

### Main Results

**Retrieval Performance** (Table 2):

| Method | Bird Recall | Bird PR | OTT-QA Recall | OTT-QA PR | LLM Calls↓ |
|------|-----------|---------|-------------|-----------|-----------|
| Dense Retrieval @5 | 89.0 | 78.4 | 75.2 | 53.8 | 0 |
| ReAct (Llama3.1) | 96.7 | 93.5 | 76.0 | 55.1 | 5.26 |
| ReAct (GPT4o-mini) | 97.0 | 93.3 | 80.6 | 62.7 | 3.16 |
| **ARM** | **96.5** | **92.7** | **79.8** | **62.5** | **1** |

ARM achieves recall comparable to ReAct (5+ calls) with only 1 LLM call, while improving precision from 15% to 42.7% (Bird), significantly reducing retrieval noise.

**End-to-End Performance** (Table 3, average of two models):

| Method | Bird Acc | OTT-QA Exact | OTT-QA F1 |
|------|---------|-------------|-----------|
| Dense Retrieval @5 | 23.6 | 37.1 | 44.2 |
| DRR + Decomposition @5 | 21.2 | 42.2 | 50.9 |
| ReAct | 15.1 | 33.9 | 40.9 |
| **ARM** | **26.2** | **46.6** | **55.5** |

ARM is 2.6pt higher than standard RAG and 11.1pt higher than ReAct on Bird; and 4.4pt F1 higher than standard RAG and 14.6pt F1 higher than ReAct on OTT-QA.

### Ablation Study

**Incremental Module Addition** (Figure 2, average across two datasets):
- Dense Retrieval + Decomposition (Baseline)
- + Information Alignment: Recall +12.5pt, PR +19.8pt
- + Structure Alignment: Recall another +1.28pt, PR +4.02pt
- + Self-Verification & Aggregation: Recall another +5.72pt, PR +9.18pt

Each module contributes significantly, with Information Alignment making the largest contribution.

**Contribution of Keyword Matching in Information Alignment** (Figure 3): Comparing embedding similarity only vs. embedding + keyword matching, the latter improves Recall by +2.15pt and PR by +3.65pt.

### Key Findings

1. Complex retrieval can be completed in a single decoding pass without multi-round iterations—this is the core efficiency advantage of ARM over ReAct.
2. Two major error modes of ReAct: (a) forgetting information from previous iterations; (b) searching similar keywords in loops.
3. The structure alignment module reasons about connection relationships between tables (e.g., joinable columns, bridging entities) via the MIP solver, which cannot be achieved by pure semantic retrieval.
4. ARM retrieves fewer but higher-quality objects, reducing the noise processed by the downstream LLM.

## Highlights & Insights

- **Paradigm Shift**: From "iterative retrieval" to "all-at-once alignment retrieval", drastically improving efficiency (1 vs. 5+ LLM calls).
- **Reasoning in Retrieval**: Integrates structural reasoning (table joins, bridging entities) directly into the retrieval process rather than leaving it to the downstream LLM.
- **Innovative Use of Constrained Decoding**: Achieves information alignment through N-gram constrained decoding, ensuring generated keywords strictly match the actual dataset.
- **Flexibility from MIP**: Enables the injection of domain-specific business logic into retrieval objectives.

## Limitations & Future Work

- Relies on pre-built N-gram indices and embeddings; the index-building cost is not fully discussed.
- The scalability of MIP solving on large-scale datasets may be limited (the search space correlates with the number of objects).
- ARM is only evaluated with Llama3.1-8B-Instruct; it remains unknown whether larger models can further improve performance.
- Compatibility scores in structure alignment depend on embedding similarity and exact value matches, potentially missing more complex semantic relationships.
- The number of steps for draft expansion and the amount added per step require manual configuration.
- The potential of combining with stronger retrievers (e.g., ColBERT) or superior embedding models is not explored.

## Related Work & Insights

- Extends the "retrieve all tables at once" idea from Chen et al. (2024c) to multi-modal datasets containing both tables and text.
- Shares a similar concept of "retrieve-while-generate" with RAG to Riches (Jain et al., 2024), but ARM additionally introduces structural reasoning.
- A rational critique of agentic RAG: although iterative methods offer interactivity, they are limited by being "based on past decisions" rather than "based on data organization".
- Insight: Complex retrieval must consider both information relevance and data structure; pure semantic matching is insufficient.

## Rating

- **Novelty**: 9/10 — The "all-at-once alignment retrieval" paradigm and the application of MIP in retrieval are highly novel.
- **Technical Depth**: 9/10 — The three modules—constrained decoding, MIP solving, and self-verification—contain high technical depth.
- **Experimental Thoroughness**: 8/10 — Two representative datasets, with comprehensive ablation studies and baseline comparisons.
- **Value**: 8/10 — Significantly reduces LLM call costs, but relies on specific indices and solvers.
- **Overall Rating**: 9/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)

</div>

<!-- RELATED:END -->
