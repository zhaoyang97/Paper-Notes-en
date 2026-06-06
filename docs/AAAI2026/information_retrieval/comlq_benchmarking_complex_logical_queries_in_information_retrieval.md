---
title: >-
  [Paper Note] ComLQ: Benchmarking Complex Logical Queries in Information Retrieval
description: >-
  [AAAI 2026][Information Retrieval & RAG][Complex logical queries] This paper introduces ComLQ, the first IR benchmark targeting complex logical queries spanning 14 query types (conjunction, disjunction, negation…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "Complex logical queries"
  - "negation handling"
  - "subgraph-guided prompting"
  - "IR benchmark"
  - "first-order logic"
date: 2026-05-08
content_hash: 5787dc153472d8ee
---

# ComLQ: Benchmarking Complex Logical Queries in Information Retrieval

**Conference**: AAAI 2026  
**arXiv**: [2511.12004](https://arxiv.org/abs/2511.12004)  
**Code**: [https://github.com/xgl-git/ComLQR-main](https://github.com/xgl-git/ComLQR-main)  
**Area**: Information Retrieval / Benchmarking  
**Keywords**: Complex logical queries, negation handling, subgraph-guided prompting, IR benchmark, first-order logic

## TL;DR
This paper introduces ComLQ, the first IR benchmark targeting complex logical queries spanning 14 query types (conjunction, disjunction, negation, and their combinations). It proposes a subgraph-guided LLM data synthesis pipeline and a negation consistency metric LSNC, revealing that existing retrievers suffer severely in logical reasoning—particularly in negation modeling.

## Background & Motivation
**Background**: Information retrieval systems are foundational for managing information overload and are widely deployed in recommendation, question answering, and related applications. Existing IR benchmarks (MS-MARCO, TREC, BEIR, etc.) primarily focus on simple queries semantically akin to single-hop or multi-hop relational lookups.

**Limitations of Prior Work**:
   - Real-world user queries frequently involve compound logical reasoning (conjunction ∧, disjunction ∨, negation ¬, and their combinations), yet **over 93% of queries in existing benchmarks are simple queries**.
   - Complex logical queries have been studied in knowledge base question answering (KBQA), but remain largely unexplored in IR—where set operations must be performed over **unstructured text**, making the task substantially harder than reasoning over structured triples.
   - Existing retrievers rely heavily on term co-occurrence and tend to retrieve irrelevant passages containing negated keywords (e.g., "American") when confronted with negation queries.

**Key Challenge**: The IR community lacks a benchmark capable of **systematically and fine-grainedly evaluating the logical reasoning ability of retrievers**, with no means to quantify negation handling in particular.

**Goal**: Construct an IR benchmark covering 14 first-order logic query types and propose appropriate evaluation metrics.

**Key Insight**: Leverage LLM generation, subgraph-guided prompt design, and expert validation to ensure that each query's logical structure is precise and well-grounded.

**Core Idea**: Use subgraph indicators to guide LLMs in generating queries of specific logical structures from passages, thereby constructing the first comprehensive IR benchmark for complex logical queries and exposing the logical reasoning deficiencies of existing retrievers.

## Method

### Overall Architecture
ComLQ is constructed in three stages: (1) **Passage selection**—relevant passages are selected from a Wikipedia dump; (2) **Subgraph-guided query generation**—prompts containing subgraph indicators are designed to guide LLMs in generating queries conforming to specified logical structures; (3) **Expert validation**—three annotators verify each query–passage pair for structural consistency and evidence distribution.

### Key Designs
1. **Query Type Taxonomy**:

    - Standard first-order logic query definitions are adopted, with four primitive operations: projection $p$ (relational path traversal), intersection $i$ (conjunction), union $u$ (disjunction), and negation $n$.
    - A total of 14 query types are defined: 9 negation-free (1p/2p/3p/2i/3i/pi/ip/2u/up) and 5 negation-containing (2in/3in/inp/pin/pni).
    - The dataset comprises 2,909 queries and 11,251 candidate passages; negation queries account for 33.8%.
    - **Design Motivation**: Full coverage of projection, intersection, union, negation, and their combinations makes this the most fine-grained IR logical query benchmark to date.

2. **Subgraph-Guided Prompting**:

    - LLMs alone struggle to consistently generate queries that conform to specific logical structures from natural language descriptions.
    - **Subgraph indicators**—symbolic subgraph patterns that explicitly encode the target query structure—are incorporated into prompts. For example, the subgraph indicator for a pni-type query is: `{?z | (?x, R1, ?y) ∧ (?y, R2, ?z)} ∩ {?z | ¬(?w, R3, ?z)}`.
    - Each complete prompt consists of three components: query definition + subgraph indicator + demonstrations.
    - **Design Motivation**: Symbolic subgraphs provide a precise structural blueprint for the LLM, and when combined with its natural language generation capability, enable structure-controlled query generation.

3. **Dual Expert Validation Criteria**:

    - **Structural consistency**: Three annotators verify whether each generated query strictly conforms to the intended logical structure, aided by auxiliary triples. For instance, a negative example: "Processors considered but not used by the IBM PC" appears to follow a pin structure but does not strictly satisfy it.
    - **Evidence distribution**: For queries generated from multiple passages, annotators verify that supporting evidence is genuinely distributed across different passages rather than concentrated in a single one.
    - Both steps use majority voting to reach consensus.
    - Annotation scheme: 3-level relevance scores (0—irrelevant; 1—partially relevant / evidence distributed across multiple passages; 2—fully relevant / a single passage contains the complete answer).

4. **Negation Consistency Metric LSNC@K**:

    - Standard nDCG and mAP measure overall relevance but cannot specifically assess a retriever's handling of negation conditions.
    - $\text{LSNC@K} = -\log\left(\frac{\sum V(d) + 1}{K + 1}\right) / \log(K + 1)$, where $V(d)$ is an indicator function equal to 1 when a retrieved passage violates the negation condition.
    - Higher LSNC@K indicates fewer top-$K$ passages violating the negation condition.
    - **Design Motivation**: Directly quantifies whether a retriever ranks excluded content highly, filling the gap left by existing metrics in negation evaluation.

### Loss & Training
This paper is a **benchmark construction and evaluation study** and requires no model training. All experiments are conducted in a zero-shot setting across multiple retrieval models.

## Key Experimental Results

### Main Results (nDCG@10, %)

| Model | 1p | 2i | 3i | pin | pni | Overall |
|------|-----|-----|-----|-----|-----|------|
| BM25 | 66.1 | 63.5 | 60.3 | 32.4 | 31.7 | 50.5 |
| BGE | 66.3 | 60.1 | 58.2 | 33.3 | 34.8 | 47.4 |
| InteR | 71.8 | 63.6 | 62.6 | 34.6 | 37.5 | **55.7** |
| Contriever | 70.2 | 60.7 | 61.2 | 32.1 | 35.5 | 53.4 |
| AGR | 74.3 | 62.3 | 62.7 | 35.5 | 33.8 | 54.3 |

All retrievers exhibit substantial performance degradation on negation-containing query types (2in/3in/inp/pin/pni). No single model consistently dominates across all query types.

### LSNC@100 Evaluation (Negation Queries)

| Model | 2in | 3in | inp | pin | pni |
|------|------|------|------|------|------|
| BM25 | 32.2 | 29.0 | 30.5 | 29.3 | 27.4 |
| BGE | 30.2 | 31.8 | 26.6 | 27.2 | 25.7 |
| HyDE | 27.8 | 26.0 | 23.1 | 24.4 | 24.9 |

All models achieve low LSNC@100 scores (mostly in the 25–35% range), indicating that a large proportion of the top-100 retrieved passages violate negation conditions.

### Key Findings
1. **Performance degrades with increasing complexity**: Performance consistently declines along 1p→2p→3p and 2i→3i, exposing the limitations of retrievers in handling compound reasoning.
2. **Negation is the most critical weakness**: Negation-containing queries consistently underperform their negation-free structural counterparts.
3. **Operation order matters**: Projection-then-intersection (pi/pin/pni) is harder than intersection-then-projection (ip/inp), suggesting that retrievers struggle with compositional reasoning over intermediate steps.
4. **Sparse retrievers remain competitive**: BM25 matches or surpasses dense retrievers on multiple query types, challenging the assumption that dense models uniformly outperform sparse ones.

## Highlights & Insights
- **Fills an important gap**: The first IR benchmark to systematically cover 14 logical query types, offering unique value to the IR community.
- **Elegant subgraph-guided prompt design**: The pipeline of symbolic subgraphs + LLM generation + human validation ensures both quality and scalability.
- **Practical LSNC metric**: Existing metrics genuinely cannot evaluate negation handling in isolation; the proposed metric is well-motivated and exposes a serious shortcoming.
- **Constructive findings**: Beyond identifying problems, the results implicitly point to improvement directions—retrieval strategies tailored to specific logical structures are needed.

## Limitations & Future Work
- Only Wikipedia is used as the corpus, limiting domain coverage (though the construction pipeline is claimed to generalize to other domains).
- Queries are LLM-generated and subsequently human-validated rather than sourced entirely from real users, which may introduce distributional bias.
- Only zero-shot evaluation is conducted; the potential gains from fine-tuning retrievers on complex logical queries remain untested.
- The dataset is relatively small (2,909 queries); larger-scale, multi-domain versions would be more compelling.
- LSNC targets negation exclusively; dedicated metrics for conjunction and disjunction are absent.

## Related Work & Insights
- BEIR (Thakur et al. 2021) is the standard for multi-domain, multi-task IR evaluation but lacks logical query types.
- NegConstraint (Xu et al. 2025) addresses only negation-constrained queries and constitutes a subset of ComLQ.
- BetaE (Ren et al. 2020) and Query2Box (Ren et al. 2020) study complex logical queries in KBQA; this paper transfers that paradigm to unstructured text retrieval.
- The work directly informs researchers working on RAG and agentic retrieval by highlighting the need to attend to retrievers' handling of compound logical conditions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering benchmark with a clearly defined problem and an original subgraph-guided prompting approach
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven retrieval models evaluated across 14 query types, though fine-tuning experiments are absent
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is thoroughly articulated, examples are intuitive, and metric design is rigorous
- Value: ⭐⭐⭐⭐⭐ Opens a new track for complex logical query evaluation in the IR community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](../../ACL2026/information_retrieval/benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)
- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](../../CVPR2026/information_retrieval/pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)
- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](../../ACL2026/information_retrieval/authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[ACL 2026\] Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers](../../ACL2026/information_retrieval/code-switching_information_retrieval_benchmarks_analysis_and_the_limits_of_curre.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](../../ACL2026/information_retrieval/unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)

</div>

<!-- RELATED:END -->
