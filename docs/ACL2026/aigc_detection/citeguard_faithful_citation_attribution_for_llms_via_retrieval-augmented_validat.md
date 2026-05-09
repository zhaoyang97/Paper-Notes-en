---
title: >-
  [Paper Note] CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation
description: >-
  [ACL 2026][AIGC Detection][citation attribution] CiteGuard proposes a retrieval-augmented agent framework with extended retrieval actions (including full-text search and context retrieval) to provide a more faithful basis for scientific citation attribution, achieving 68.1% accuracy on the CiteME benchmark — a 10-point improvement over baselines, approaching human performance (69.2%).
tags:
  - ACL 2026
  - AIGC Detection
  - citation attribution
  - retrieval-augmented validation
  - scientific writing
  - hallucination mitigation
  - agent
date: 2025-05-08
content_hash: a3be52cdf3a3747e
---

# CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation

**Conference**: ACL 2026
**arXiv**: [2510.17853](https://arxiv.org/abs/2510.17853)
**Code**: [https://github.com/KathCYM/CiteGuard](https://github.com/KathCYM/CiteGuard)
**Area**: AIGC Detection
**Keywords**: citation attribution, retrieval-augmented validation, scientific writing, hallucination mitigation, agent

## TL;DR

CiteGuard proposes a retrieval-augmented agent framework with extended retrieval actions (including full-text search and context retrieval) to provide a more faithful basis for scientific citation attribution, achieving 68.1% accuracy on the CiteME benchmark — a 10-point improvement over baselines, approaching human performance (69.2%).

## Background & Motivation

**Background**: LLMs are increasingly used for scientific writing assistance, but citation hallucination is a severe problem (LLMs can generate up to 78-90% fabricated citations). Over 50 citation hallucinations were found among 300 ICLR 2026 submissions.

**Limitations of Prior Work**: (1) LLM-as-a-Judge has extremely low recall in citation verification (only 16-17%), as LLMs are overly sensitive to minor terminology changes; (2) existing methods like CiteAgent still fall far below human accuracy; (3) existing methods lack the ability to search full-text content of papers.

**Key Challenge**: Retrieval based solely on titles and abstracts is insufficient to confirm citation relationships; it often requires delving into the full text for cross-validation.

**Goal**: Design a more faithful and generalizable citation attribution agent.

**Key Insight**: Extend the set of retrieval actions, particularly by adding full-text search and context retrieval capabilities.

**Core Idea**: Citation verification requires information beyond the title/abstract level; full-text search and context retrieval provide a stronger evidence base.

## Method

### Overall Architecture

CiteGuard is an LLM-based agent that extends CiteAgent with three new actions: find_in_text (full-text search within a paper), ask_for_more_context (source paper context retrieval), and search_text_snippet (cross-paper full-text snippet search). It supports iterative retrieval to recommend multiple references.

### Key Designs

1. **Extended Retrieval Action Set**:

    - Function: Provide deeper evidence than title/abstract
    - Mechanism: Three new actions — find_in_text (search queries within a specific paper's full text), ask_for_more_context (retrieve 3 paragraphs before and after an excerpt's context), and search_text_snippet (cross-database full-text snippet search)
    - Design Motivation: Citation relationships are often hidden in the body of papers; looking only at titles and abstracts may lead to misjudgment

2. **Iterative Multi-Citation Recommendation**:

    - Function: Recommend multiple relevant references
    - Mechanism: Each run recommends one reference; subsequent runs exclude already-selected papers to search for new references. Filtering through the exclusion set $E_k$ ensures no duplicate recommendations
    - Design Motivation: Many academic claims have multiple valid citations; a single reference is insufficient

3. **Cross-Domain Generalization**:

    - Function: Evaluate method usability beyond computer science
    - Mechanism: Collect the CiteMulti extended benchmark covering biomedical, physics, and mathematics domains, as well as long-paragraph scenarios
    - Design Motivation: Verify the generality of the method

### Loss & Training

No model training involved. The agent uses GPT-4o or DeepSeek-R1 as the base model.

## Key Experimental Results

### Main Results

**CiteME Benchmark Results**

| Method | All-Difficulty Accuracy |
|---|---|
| CiteAgent + GPT-4o | 35.4% |
| CiteGuard + GPT-4o | 45.4% (+10pp) |
| CiteGuard + DeepSeek-R1 | **68.1%** |
| Human Performance | 69.2% |

### Ablation Study

- CiteGuard can identify alternative valid citations not covered by the benchmark
- The new retrieval actions (especially find_in_text) contribute most to performance improvement
- Cross-domain experiments show the method has generalization potential

### Key Findings

- Full-text search capability is critical for citation verification
- The 68.1% accuracy approaching human performance demonstrates the method's effectiveness
- LLM-as-a-Judge is unreliable for citation verification and requires retrieval augmentation

## Highlights & Insights

- Addresses a real pain point in scientific writing with high practical value
- Approaching human performance is an important milestone
- The extended CiteMulti benchmark fills the gap in cross-domain evaluation

## Limitations & Future Work

- Depends on the Semantic Scholar API, which may not cover all domains
- Full-text search requires paper accessibility; some papers may be unavailable
- Iterative retrieval increases inference cost
- Future work could explore integrating the method into academic writing workflows

## Related Work & Insights

- The extension of CiteAgent demonstrates the critical value of full-text search
- Provides a practical tool for scientific citation quality control

## Rating

- Novelty: ⭐⭐⭐⭐ Full-text search and iterative multi-citation recommendation are practical innovations
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-domain evaluation + human annotation + multi-model comparison
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, well-designed experiments

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](../../ICLR2026/aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)

<!-- RELATED:END -->
