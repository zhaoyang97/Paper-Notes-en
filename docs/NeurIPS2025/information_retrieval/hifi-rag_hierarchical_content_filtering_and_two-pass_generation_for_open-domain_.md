---
title: >-
  [Paper Note] HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG
description: >-
  [NeurIPS 2025][Retrieval-Augmented Generation] By decoupling the filtering capability of a lightweight Flash model from the reasoning capability of a Pro model, the paper constructs a multi-stage pipeline (query optimization → hierarchical filtering → two-pass generation → citation verification) that achieves SOTA performance in the MMU-RAGent competition.
tags:
  - NeurIPS 2025
  - Retrieval-Augmented Generation
  - hierarchical filtering
  - two-pass generation
  - multi-LLM strategy
  - Flash-Pro cascade
date: 2026-05-08
content_hash: d7d6ab1a8bee26e3
---

# HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG

**Conference**: NeurIPS 2025
**arXiv**: [2512.22442](https://arxiv.org/abs/2512.22442)
**Code**: None
**Area**: LLM Agent / RAG
**Keywords**: Retrieval-Augmented Generation, hierarchical filtering, two-pass generation, multi-LLM strategy, Flash-Pro cascade

## TL;DR
By decoupling the filtering capability of a lightweight Flash model from the reasoning capability of a Pro model, the paper constructs a multi-stage pipeline (query optimization → hierarchical filtering → two-pass generation → citation verification) that achieves SOTA performance in the MMU-RAGent competition.

## Background & Motivation

**Background**: RAG faces information retrieval challenges in open-domain settings, where standard vector similarity search is prone to retrieving irrelevant content.

**Limitations of Prior Work**:
   - Retrieved documents contain substantial irrelevant information (garbage in, garbage out)
   - Generated answers are misaligned with user intent
   - Performance degrades when handling post-knowledge-cutoff data

**Key Challenge**: Simultaneous optimization of retrieval precision, generation quality, and computational cost is inherently difficult.

**Goal**: Achieve high-quality content retrieval and answer generation in open-domain RAG.

**Key Insight**: Adopt a cascaded model strategy that employs a lightweight Flash model for semantic filtering and a high-capability Pro model for reasoning.

**Core Idea**: A multi-stage pipeline system that incrementally integrates query optimization, hierarchical filtering, two-pass generation, and citation verification.

## Method

### Overall Architecture
HiFi-RAG comprises five stages: (1) Query Planning — Flash refines user queries; (2) Retrieval & URL Filtering — Google Search combined with Flash cascade filtering; (3) Hierarchical Content Parsing & Filtering — HTML parsing with LLM-based reranking; (4) Two-Pass Generation — Draft followed by Refinement (both using Pro); (5) Citation Verification — Flash validates source attribution.

### Key Designs

1. **Hierarchical Content Filtering**:

    - Function: Groups HTML content into hierarchical chunks by headings (h1–h4); Flash evaluates the relevance of each chunk with respect to the query.
    - Mechanism: Only the heading and the first 200 characters are used for judgment, removing an average of 60.5% of chunks.
    - Design Motivation: LLM-based chunk filtering offers superior semantic understanding compared to full-text vector similarity.

2. **Two-Pass Generation**:

    - Function: Separates factuality from style — Pro first generates a comprehensive Draft, then performs Refinement to adjust style and length.
    - Design Motivation: Single-pass generation struggles to simultaneously optimize factual accuracy and presentation style.

3. **Flash-Pro Cascaded Model**:

    - Function: Flash handles high-throughput tasks (4–6× lower cost); Pro is reserved for reasoning-intensive tasks.
    - Design Motivation: Exploits the capability gradient between models, achieving an ideal balance between cost and performance.

## Key Experimental Results

### Main Results

| System Configuration | ROUGE-L (F1) | DeBERTaScore (F1) | Gain |
|---|---|---|---|
| Baseline (original query) | 0.2291 | 0.6375 | - |
| + Query rewriting | 0.2591 | 0.6667 | +13% |
| + RAG search | 0.2664 | 0.6677 | +16.3% |
| + Filtering | 0.2695 | 0.6712 | +17.7% |
| Full system (final) | **0.2739** | **0.6772** | +19.6% |

### Post-Cutoff Data Analysis

| Configuration | Test2025 ROUGE-L | Test2025 DeBERTaScore |
|---|---|---|
| Baseline (no search) | 0.2766 | 0.6574 |
| RAG + Filtering | 0.3031 | 0.6840 |
| Full system | **0.3182** | **0.7092** |

### Key Findings
- A 57.4% improvement in ROUGE-L on post-cutoff data demonstrates the substantial value of RAG in addressing LLM knowledge staleness.
- URL filtering reduces the number of URLs by 33.5%, significantly lowering latency.
- Embedding-based filtering underperforms LLM-based filtering, likely due to topological noise.
- DSPy optimization tends to overfit to the validation set.

## Highlights & Insights
- **Elegance of the Flash-Pro cascade architecture**: Effectively exploits the model capability gradient with strong cost efficiency.
- **Effectiveness of hierarchical filtering**: A 60.5% chunk removal rate indicates that the majority of retrieved content constitutes noise.
- **Two-pass separation of factuality and style**: A simple yet effective design decision.

## Limitations & Future Work
- Although Flash is cost-efficient, the system still requires multiple calls, and overall latency may be unacceptable in certain scenarios.
- The two-pass generation relies on manually selected demonstrations, limiting generalizability.
- URL filtering may inadvertently discard valuable information.

## Related Work & Insights
- **vs. Traditional RAG**: Conventional vector retrieval has limited semantic understanding; HiFi-RAG employs LLM-based filtering for greater precision.
- **vs. Full Agent Approaches**: Agent-based solutions incur 10× higher cost with no performance gain.

## Rating
- Novelty: ⭐⭐⭐⭐ Combinatorial innovation in cascaded model strategy and multi-stage pipeline design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Competition-winning results with comprehensive ablation study
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and design
- Value: ⭐⭐⭐⭐⭐ Directly addresses industrial-scale RAG challenges

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [\[NeurIPS 2025\] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition](rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
