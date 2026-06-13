---
title: >-
  [Paper Note] CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation
description: >-
  [ACL 2026][Information Retrieval & RAG][Citation Attribution] CiteGuard proposes a retrieval-augmented agent framework that provides a more faithful foundation for scientific citation attribution through expanded retriev…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Citation Attribution"
  - "Retrieval-Augmented Validation"
  - "Scientific Writing"
  - "Hallucination Mitigation"
  - "Agent"
date: 2026-05-08
content_hash: 9878d031c4cce5b9
---

# CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation

**Conference**: ACL 2026  
**arXiv**: [2510.17853](https://arxiv.org/abs/2510.17853)  
**Code**: [https://github.com/KathCYM/CiteGuard](https://github.com/KathCYM/CiteGuard)  
**Area**: Scientific Citation Attribution  
**Keywords**: Citation Attribution, Retrieval-Augmented Validation, Scientific Writing, Hallucination Mitigation, Agent

## TL;DR

CiteGuard proposes a retrieval-augmented agent framework that provides a more faithful foundation for scientific citation attribution through expanded retrieval actions (including full-text search and context retrieval). It achieves a 10 percentage point improvement over baselines on the CiteME benchmark, reaching 68.1% accuracy, which is close to human performance (69.2%).

## Background & Motivation

**Background**: LLMs are increasingly used for scientific writing assistance, but the problem of citation hallucination is severe (LLMs can generate up to 78-90% fictitious citations). More than 50 citation hallucinations were discovered among 300 papers submitted to ICLR 2026.

**Limitations of Prior Work**: (1) LLM-as-a-Judge exhibits extremely low recall in citation validation (only 16-17%), as LLMs are overly sensitive to minor terminology variations; (2) the accuracy of existing methods like CiteAgent remains significantly lower than human performance; (3) existing methods lack the capability to search the full-text content of papers.

**Key Challenge**: Retrieval based solely on titles and abstracts is insufficient to confirm citation relationships, which often requires deep cross-validation within the full-text content of papers.

**Goal**: To design a more faithful and generalizable citation attribution Agent.

**Key Insight**: Expanding the retrieval action set, specifically by incorporating full-text search and context retrieval capabilities.

**Core Idea**: Citation validation needs to transcend title/abstract-level information by establishing a stronger evidentiary basis through full-text search and contextual retrieval.

## Method

### Overall Architecture

CiteGuard is an LLM-based Agent that extends CiteAgent with three new actions: `find_in_text` (full-text paper search), `ask_for_more_context` (retrieval of context from source papers), and `search_text_snippet` (cross-paper full-text snippet search). It supports iterative retrieval to recommend multiple references.

### Key Designs

1.  **Expanded Retrieval Action Set**:
    - **Function**: Provides deeper evidence compared to titles and abstracts.
    - **Mechanism**: Introduces `find_in_text` (searching queries within a specific full-text paper), `ask_for_more_context` (retrieving 3 paragraphs preceding and following an excerpt), and `search_text_snippet` (searching full-text snippets across databases).
    - **Design Motivation**: Citation relationships are often embedded within the body of a paper; relying exclusively on titles and abstracts can lead to misjudgments.

2.  **Iterative Retrieval for Multi-citation Recommendation**:
    - **Function**: Recommends multiple relevant references.
    - **Mechanism**: Each execution recommends one reference, and subsequent runs exclude previously selected papers to search for new ones. Redundant recommendations are avoided by filtering against the selected set $E_k$.
    - **Design Motivation**: Many academic claims comprise multiple valid citations, making a single reference insufficient.

3.  **Cross-domain Generalization**:
    - **Function**: Evaluates the usability of the method in fields beyond Computer Science.
    - **Mechanism**: Collected the CiteMulti extension benchmark, covering Biomedicine, Physics, and Mathematics, as well as long-paragraph scenarios.
    - **Design Motivation**: To verify the generalizability of the proposed method.

### Loss & Training

No model training is involved. The Agent utilizes GPT-4o or DeepSeek-R1 as the backbone model.

## Key Experimental Results

### Main Results

**CiteME Benchmark Results**

| Method | Accuracy (All Difficulties) |
| :--- | :--- |
| CiteAgent + GPT-4o | 35.4% |
| CiteGuard + GPT-4o | 45.4% (+10pp) |
| CiteGuard + DeepSeek-R1 | **68.1%** |
| Human Performance | 69.2% |

### Ablation Study

- CiteGuard identifies alternative valid citations not covered by the baseline.
- The new retrieval actions (especially `find_in_text`) contribute most significantly to performance gains.
- Cross-domain experiments demonstrate the potential for generalization.

### Key Findings

- Full-text search capability is essential for citation validation.
- The accuracy of 68.1%, which is close to human performance, proves the effectiveness of the method.
- LLM-as-a-Judge is unreliable for citation validation and requires retrieval augmentation.

## Highlights & Insights

- Addresses a real pain point in scientific writing with high practical value.
- Achieving performance close to humans represents a significant milestone.
- The expanded CiteMulti benchmark fills the gap in cross-domain evaluation.

## Limitations & Future Work

- Dependency on the Semantic Scholar API may limit coverage in certain domains.
- Full-text search requires papers to be accessible, which may not be possible for all publications.
- Iterative retrieval increases inference costs.
- Future work may explore integrating the method into academic writing workflows.

## Related Work & Insights

- The extension of CiteAgent reveals the critical value of full-text searching.
- Provides a practical tool for quality control in scientific citations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Full-text search and iterative multi-citation recommendation are practical innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Cross-domain evaluation + manual annotation + multi-model comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and sound experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ICLR 2026\] Attribution-Guided Decoding](../../ICLR2026/information_retrieval/attribution-guided_decoding.md)

</div>

<!-- RELATED:END -->
