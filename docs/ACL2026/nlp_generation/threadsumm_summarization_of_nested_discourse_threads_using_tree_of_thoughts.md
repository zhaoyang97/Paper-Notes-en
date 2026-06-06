---
title: >-
  [Paper Note] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts
description: >-
  [ACL 2026][Text Generation][Nested Discourse Thread Summarization] This paper proposes ThreadSumm, a multi-stage LLM pipeline framework that models nested discourse thread summarization as a hierarchical reasoning proble…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Nested Discourse Thread Summarization"
  - "Tree of Thoughts"
  - "Atomic Content Units"
  - "Multi-stage LLM Pipeline"
  - "Coherence and Coverage"
date: 2026-05-08
content_hash: c3d8df91598b9541
---

# ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts

**Conference**: ACL 2026  
**arXiv**: [2604.17648](https://arxiv.org/abs/2604.17648)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Nested Discourse Thread Summarization, Tree of Thoughts, Atomic Content Units, Multi-stage LLM Pipeline, Coherence and Coverage

## TL;DR

This paper proposes ThreadSumm, a multi-stage LLM pipeline framework that models nested discourse thread summarization as a hierarchical reasoning problem. It first extracts aspects and Atomic Content Units (ACUs) for content planning, then constructs thread-aware sequences through sentence ordering, and finally generates and scores multiple paragraph candidates using Tree of Thoughts (ToT) search, outperforming baselines on Reddit and StackExchange datasets.

## Background & Motivation

**Background**: The nested thread structures in discussion forums (interweaving replies, quotes, and forwards) make summarization significantly more complex than standard document summarization. Existing LLM summarization methods primarily handle linear documents or relatively structured dialogues.

**Limitations of Prior Work**: (1) The tree or graph-like structure of nested threads causes off-topic replies to interweave with thematic ones, burying key content; (2) current methods fail to balance diverse viewpoints, tending toward the most frequent topics while ignoring minority but important perspectives; (3) overlapping turns and interruptions in multi-speaker scenarios make simple linear adjacency models incapable of inferring response relationships.

**Key Challenge**: The graph structure of threads vs. the linear output of summaries—there is a need to maintain coherence while covering diverse topics distributed across different branches.

**Goal**: (1) Address discourse coverage (balanced representation of multiple interwoven topics); (2) address coherence (generating coherent summaries even without a predefined thread order).

**Key Insight**: Decompose summarization into two independent reasoning levels: content planning (aspect extraction + ACU generation) and textual realization (sentence ordering + paragraph writing + ToT search).

**Core Idea**: Use a structured intermediate representation (Aspects + ACUs) to explicitly control coverage, and employ Tree of Thoughts search to find the optimal balance between coherence and coverage.

## Method

### Overall Architecture

A five-step pipeline: (1) Aspect Extraction—identifying who/what/where elements in the document; (2) ACU Generation—generating irreducible semantic units for each aspect; (3) Sentence Ordering—rearranging ACUs into a logically coherent sequence; (4) Paragraph Writing—transforming ordered ACUs into fluent paragraphs; (5) Tree of Thoughts—iteratively generating multiple paragraph candidates and selecting the best based on coherence and coverage scores.

### Key Designs

1.  **Content Planning Layer with Aspects + ACUs**:
    - **Function**: Ensures the summary covers all important aspects of the source document.
    - **Mechanism**: First, few-shot prompting is used to extract aspects (who/what/where), then Atomic Content Units—irreducible, independent semantic statements—are generated for each aspect. ACUs are more granular than the original text, supporting precise coverage control.
    - **Design Motivation**: Direct summarization easily "drifts" toward the most salient topics. Explicitly extracting aspects and then generating ACUs for each forces balanced coverage.

2.  **LLM-driven Sentence Ordering**:
    - **Function**: Reorganizes an unordered set of ACUs into a logically coherent narrative sequence.
    - **Mechanism**: Uses zero-shot prompting to let the LLM rearrange the ACU list to follow a logical and coherent flow.
    - **Design Motivation**: Important content in nested threads is distributed across different branches and depths, making position or timestamp heuristics inapplicable. LLM ordering handles more global discourse coherence issues.

3.  **ToT Multi-candidate Search**:
    - **Function**: Finds the optimal balance between coherence and coverage.
    - **Mechanism**: Given ordered ACUs, multiple paragraph candidates are generated. The LLM evaluates each candidate's coherence (idea connectivity and logical flow) and coverage (inclusion of important source information), selecting the highest-scoring candidate. This iterates over multiple steps, where the ordering scheme of the best candidate from each step is carried into the next.
    - **Design Motivation**: Single-pass generation often falls into local optima. The multi-candidate approach and iterative refinement of ToT allow for a systematic search of a larger summarization space.

### Loss & Training

A training-free pipeline. Utilizes three LLMs: GPT-4, Claude-3, and LLaMA-3-70B. Evaluated on Reddit (250 instances), StackExchange (117 instances), and a Bitcoin forum (1 instance case study).

## Key Experimental Results

### Main Results

**Reddit Dataset (QAGS Consistency / ROUGE-1)**

| Model-Method | QAGS | ROUGE-1 |
| :--- | :--- | :--- |
| Claude-Vanilla | 38.34 | 30.88 |
| Claude-CHRONOS | 45.43 | 26.35 |
| Claude-ThreadSumm | **55.66** | **34.37** |
| GPT-4-Vanilla | 36.46 | 30.54 |
| GPT-4-ThreadSumm | **50.34** | **33.30** |

### Key Findings

- ThreadSumm significantly outperforms all baselines in QAGS (factual consistency), as explicit content planning with ACUs effectively prevents hallucinations.
- ROUGE-1 gains are consistent, indicating improved coverage.
- Iterative refinement in ToT markedly improves coherence; multi-candidate search yields higher quality than single-pass generation.
- Consistent trends across different LLMs demonstrate the model-agnostic nature of the framework.

## Highlights & Insights

- ACUs as an intermediate representation are well-suited for thread summarization, naturally supporting cross-branch content aggregation and balanced coverage.
- The application of ToT search in summarization tasks is novel, treating coherence and coverage as dual optimization targets for the search.
- The sentence ordering step is an undervalued but crucial link; good ordering is a prerequisite for coherent paragraphs.

## Limitations & Future Work

- Multi-step LLM calls increase latency and cost.
- Validated only on English discussion forums.
- The Bitcoin forum includes only one instance as a case study, lacking statistical significance.
- No direct comparison with the latest long-context LLMs.

## Related Work & Insights

- **vs. CHRONOS**: Processes threads based on chronological order; ThreadSumm uses LLM reasoning to handle arbitrary thread structures.
- **vs. arg-graph**: Structures dialogue with argumentation graphs; ThreadSumm uses ACUs for a more general content decomposition.
- **vs. mRedditSumm**: A multi-document summarization baseline; ThreadSumm achieves better coherence through ToT search.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of ACU and ToT search is novel for thread summarization.
- Experimental Thoroughness: ⭐⭐⭐ Uses 3 models and 3 datasets but on a relatively small scale.
- Writing Quality: ⭐⭐⭐⭐ Clear research questions and comprehensive framework description.
- Value: ⭐⭐⭐⭐ Provides a practical solution for nested discourse summarization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)
- [\[ACL 2026\] In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)

</div>

<!-- RELATED:END -->
