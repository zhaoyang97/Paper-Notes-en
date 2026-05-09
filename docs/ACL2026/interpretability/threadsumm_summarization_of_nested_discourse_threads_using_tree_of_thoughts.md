---
title: >-
  [Paper Note] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts
description: >-
  [ACL 2026][nested discourse thread summarization] This paper proposes ThreadSumm, a multi-stage LLM pipeline framework that models nested discourse thread summarization as a hierarchical reasoning problem. It first extracts aspects and atomic content units (ACUs) for content planning, then constructs a thread-aware sequence via sentence ordering, and finally applies Tree of Thoughts search to generate and score multiple paragraph candidates. The approach outperforms baselines on Reddit and StackExchange datasets.
tags:
  - ACL 2026
  - nested discourse thread summarization
  - Tree of Thoughts
  - atomic content units
  - multi-stage LLM pipeline
  - coherence and coverage
date: 2026-05-08
content_hash: 82eaf06760b3ae14
---

# ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts

**Conference**: ACL 2026
**arXiv**: [2604.17648](https://arxiv.org/abs/2604.17648)
**Code**: None
**Area**: Interpretability
**Keywords**: nested discourse thread summarization, Tree of Thoughts, atomic content units, multi-stage LLM pipeline, coherence and coverage

## TL;DR

This paper proposes ThreadSumm, a multi-stage LLM pipeline framework that models nested discourse thread summarization as a hierarchical reasoning problem. It first extracts aspects and atomic content units (ACUs) for content planning, then constructs a thread-aware sequence via sentence ordering, and finally applies Tree of Thoughts search to generate and score multiple paragraph candidates. The approach outperforms baselines on Reddit and StackExchange datasets.

## Background & Motivation

**Background**: The nested thread structure in discussion forums—where replies, citations, and reposts are interleaved—makes summarization far more complex than standard document summarization. Existing LLM-based summarization methods primarily handle linear documents or relatively structured dialogues.

**Limitations of Prior Work**: (1) The tree/graph structure of nested threads causes off-topic and on-topic replies to be interleaved, burying key content; (2) existing methods fail to balance diverse perspectives, tending to favor the most frequent topics while neglecting minority but important viewpoints; (3) turn overlap and interruptions in multi-speaker settings render simple linear adjacency models incapable of inferring reply relations.

**Key Challenge**: The graph structure of threads versus the linear output of summaries—maintaining coherence while covering diverse topics distributed across different branches is inherently challenging.

**Goal**: (1) Address discourse coverage (balanced representation of multiple interleaved topics); (2) address coherence (generating coherent summaries even without a predefined thread order).

**Key Insight**: Decompose summarization into two independent reasoning levels: content planning (aspect extraction + ACU generation) and text realization (sentence ordering + paragraph writing + ToT search).

**Core Idea**: Use structured intermediate representations (aspects + ACUs) to explicitly control coverage, and apply Tree of Thoughts search to find the optimal balance between coherence and coverage.

## Method

### Overall Architecture

A five-stage pipeline: (1) **Aspect Extraction**—identify who/what/where elements in the document; (2) **ACU Generation**—generate atomic, indivisible semantic units for each aspect; (3) **Sentence Ordering**—rearrange ACUs into a logically coherent sequence; (4) **Paragraph Writing**—compose the ordered ACUs into fluent paragraphs; (5) **Tree of Thoughts**—iteratively generate multiple paragraph candidates, score them by coherence and coverage, and select the best.

### Key Designs

1. **Content Planning Layer with Aspects + ACUs**:

    - Function: Ensure the summary covers all important aspects of the source document.
    - Mechanism: Few-shot prompting is first used to extract aspects (who/what/where), then atomic content units—indivisible, self-contained semantic statements—are generated for each aspect. ACUs are more fine-grained than the original text, enabling precise coverage control.
    - Design Motivation: Direct summarization tends to drift toward the most salient topics. Explicitly extracting aspects and generating per-aspect ACUs enforces balanced coverage.

2. **LLM-Driven Sentence Ordering**:

    - Function: Reorganize an unordered set of ACUs into a logically coherent narrative sequence.
    - Mechanism: Zero-shot prompting is used to instruct the LLM to reorder the ACU list to follow a logical and coherent flow.
    - Design Motivation: Important content in nested threads is distributed across different branches and depths, making position/timestamp heuristics inadequate. LLM-based ordering addresses more global discourse coherence.

3. **Tree of Thoughts Multi-Candidate Search**:

    - Function: Find the optimal balance between coherence and coverage.
    - Mechanism: Given the ordered ACUs, multiple paragraph candidates are generated. An LLM evaluates each candidate on coherence (idea connectivity and logical flow) and coverage (inclusion of important information from the source), and the highest-scoring candidate is selected. The process iterates over multiple steps, with the ordering scheme of the best candidate carried into the next step.
    - Design Motivation: Single-pass generation is prone to local optima. ToT's multi-candidate generation and iterative refinement enable systematic search over a larger summarization space.

### Loss & Training

The pipeline is training-free. Three LLMs are used: GPT-4, Claude-3, and LLaMA-3-70B. Evaluation is conducted on Reddit (250 instances), StackExchange (117 instances), and a Bitcoin forum (1 instance case study).

## Key Experimental Results

### Main Results

**Reddit Dataset (QAGS Consistency / ROUGE-1)**

| Model–Method | QAGS | ROUGE-1 |
|---|---|---|
| Claude-Vanilla | 38.34 | 30.88 |
| Claude-CHRONOS | 45.43 | 26.35 |
| Claude-ThreadSumm | **55.66** | **34.37** |
| GPT-4-Vanilla | 36.46 | 30.54 |
| GPT-4-ThreadSumm | **50.34** | **33.30** |

### Key Findings

- ThreadSumm significantly outperforms all baselines on QAGS (factual consistency)—explicit content planning via ACUs effectively prevents hallucination.
- ROUGE-1 improvements are consistent, indicating genuine gains in coverage.
- ToT iterative refinement yields notable coherence improvements—multi-candidate search produces higher quality than single-pass generation.
- Consistent trends across different LLMs demonstrate the model-agnostic nature of the framework.

## Highlights & Insights

- ACUs as intermediate representations are particularly well-suited for thread summarization—they naturally support cross-branch content aggregation and balanced coverage.
- Applying ToT search to summarization is novel—the dual optimization of coherence and coverage as search objectives is a meaningful contribution.
- The sentence ordering step is an underappreciated but critical component—well-ordered ACUs are a prerequisite for coherent paragraph generation.

## Limitations & Future Work

- Multi-step LLM calls increase latency and cost.
- Evaluation is limited to English-language discussion forums.
- The Bitcoin forum case study involves only one instance, which is insufficient for statistical significance.
- No direct comparison with recent long-context LLMs is provided.

## Related Work & Insights

- **vs. CHRONOS**: CHRONOS processes threads in chronological order; ThreadSumm uses LLM reasoning to handle arbitrary thread structures.
- **vs. arg-graph**: Argument graph methods structure dialogues via argumentation graphs; ThreadSumm uses ACUs for a more general content decomposition.
- **vs. mRedditSumm**: A multi-document summarization baseline; ThreadSumm achieves superior coherence through ToT search.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of ACUs and ToT search is novel in the context of thread summarization.
- Experimental Thoroughness: ⭐⭐⭐ Covers 3 models and 3 datasets, though the scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Research questions are clearly stated and the framework is thoroughly described.
- Value: ⭐⭐⭐⭐ Provides a practical solution for nested discourse summarization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](../../CVPR2026/interpretability/cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](similarity-distance-magnitude_activations.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[ACL 2026\] HistLens: Mapping Idea Change across Concepts and Corpora](histlens_mapping_idea_change_across_concepts_and_corpora.md)

</div>

<!-- RELATED:END -->
