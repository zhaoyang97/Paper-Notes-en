---
title: >-
  [Paper Note] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs
description: >-
  [ACL 2026][Information Retrieval & RAG][utility judgment] Inspired by Schutz's philosophical theory of relevance, this paper proposes ITEM, an iterative utility judgment framework that enables the three core RAG componen…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "utility judgment"
  - "philosophical relevance theory"
  - "iterative framework"
  - "RAG optimization"
  - "LLM reasoning"
date: 2026-05-08
content_hash: ca5bab06d42aa114
---

# An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs

**Conference**: ACL 2026
**arXiv**: [2406.11290](https://arxiv.org/abs/2406.11290)  
**Code**: [GitHub](https://github.com/Trustworthy-Information-Access/ITEM)  
**Area**: Information Retrieval / RAG
**Keywords**: utility judgment, philosophical relevance theory, iterative framework, RAG optimization, LLM reasoning

## TL;DR

Inspired by Schutz's philosophical theory of relevance, this paper proposes ITEM, an iterative utility judgment framework that enables the three core RAG components—relevance ranking, utility judgment, and answer generation—to mutually and dynamically enhance one another, yielding improvements over baselines across retrieval, utility judgment, and QA tasks.

## Background & Motivation

**Background**: In RAG settings, LLMs operate under limited input bandwidth, necessitating the prioritization of high-utility (rather than merely high-relevance) retrieved documents. Relevance measures whether a document is *about* the topic, whereas utility measures whether it *helps answer* the question.

**Limitations of Prior Work**: (1) Existing RAG approaches primarily optimize for topical relevance while neglecting the higher standard of utility; (2) Zhang et al. introduced LLM-based utility judgment but only conducted preliminary exploration; (3) the three RAG components (retrieval, judgment, generation) are typically optimized independently, lacking joint enhancement.

**Key Challenge**: Topically relevant documents are not necessarily useful—a document discussing the same topic but containing no specific answer is relevant yet unhelpful. Existing methods struggle to distinguish between the two.

**Goal**: Improve LLM utility judgment through iterative interaction among the three RAG components.

**Key Insight**: The paper maps RAG onto Schutz's philosophical "system of relevances"—topical relevance, interpretational relevance (utility), and motivational relevance (answer) correspond to three cognitive levels that can mutually reinforce one another.

**Core Idea**: The three RAG components reflect three cognitive levels of LLMs in question answering (aboutness → value → answer), and iterating among them enables mutual enhancement.

## Method

### Overall Architecture

ITEM has two variants: ITEM-A (iterative answer generation + utility judgment) and ITEM-AR (iterative answer generation + relevance ranking + utility judgment). In each iteration, the LLM first generates a pseudo-answer, then refines utility judgment or ranking based on that pseudo-answer, and subsequently regenerates the answer in a cyclic manner.

### Key Designs

1. **Iterative Utility Judgment Mechanism**:

    - **Function**: Progressively improves utility judgment quality through multiple rounds of iteration.
    - **Mechanism**: In each round, the LLM generates a pseudo-answer (serving as a cognitive anchor), performs utility judgment (pointwise or listwise) conditioned on the pseudo-answer, and then updates the pseudo-answer. Judgment quality improves incrementally across rounds.
    - **Design Motivation**: Single-pass judgment is susceptible to noise; iteration allows the LLM to gradually accumulate understanding of both the question and the documents.

2. **Mapping Philosophical Theory to RAG**:

    - **Function**: Provides theoretical grounding for the iterative framework.
    - **Mechanism**: Schutz's three types of relevance—topical relevance (focusing on an object) → interpretational relevance (understanding the object) → motivational relevance (acting on that understanding)—map one-to-one onto RAG's retrieval → utility judgment → answer generation pipeline.
    - **Design Motivation**: Philosophical theory predicts mutual reinforcement among these three dimensions, offering a theoretical basis for the iterative design.

3. **Comparison of Two Iterative Variants**:

    - **Function**: Explores the applicability of different iterative strategies.
    - **Mechanism**: ITEM-A iterates only over answer generation and judgment (fewer components, more rounds); ITEM-AR incorporates a ranking component (more components, richer per-round updates). Different levels of task complexity call for different strategies.
    - **Design Motivation**: Harder tasks benefit from more components and more rounds; simpler tasks are better served by a more lightweight strategy.

### Loss & Training

No training is required. The framework relies entirely on in-context learning with LLMs. Prompt design governs the task assigned at each round (answer generation / utility judgment / ranking).

## Key Experimental Results

### Main Results

| Task | Dataset | ITEM Improvement | Notes |
|------|---------|-----------------|-------|
| Retrieval Ranking | TREC DL | Outperforms baseline | Utility judgment in turn improves ranking |
| Utility Judgment | GTI-NQ | Outperforms baseline | Iteration substantially improves judgment quality |
| QA | NQ | Outperforms baseline | Higher-utility documents yield better answers |
| Non-factoid Retrieval | WebAP | Outperforms baseline | Greater gains on harder tasks |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| 1 round vs. multiple rounds | Multiple rounds superior | Iteration is genuinely effective |
| ITEM-A vs. ITEM-AR | Task-dependent | Harder tasks require ITEM-AR |
| vs. long-reasoning mode | Comparable performance, much lower cost | Iteration is more efficient than single-pass extended reasoning |

### Key Findings
- For harder tasks (e.g., non-factoid retrieval on WebAP) and complex candidate lists (e.g., GTI-NQ), more components combined with more iterations prove most effective.
- ITEM achieves performance comparable to long-reasoning modes at substantially lower computational cost.
- For simpler factoid QA tasks, fewer components with more iterations yield the best results.

## Highlights & Insights
- A creative mapping from philosophical theory to engineering methodology—Schutz's system of relevances offers a novel perspective for RAG optimization.
- The paper reveals a relationship between task complexity and optimal iterative strategy.
- RAG quality is improved without any training, making the approach highly practical.

## Limitations & Future Work
- Iteration increases inference cost due to multiple LLM calls, and latency may be unacceptable in latency-sensitive settings.
- The quality of pseudo-answers may impose an upper bound on the gains achievable through iteration.
- Evaluation is conducted exclusively on English datasets.
- Future work could combine the framework with fine-tuned retrievers for further improvement.

## Related Work & Insights
- **vs. single-pass utility judgment**: The iterative framework substantially improves judgment quality through multi-round cognitive accumulation.
- **vs. multi-round retrieval RAG**: Rather than modifying retrieval itself, the framework iteratively refines utility judgment over already-retrieved results.
- **vs. long-reasoning / chain-of-thought**: Achieves comparable performance at considerably lower cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative perspective mapping philosophical theory to RAG
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multi-task evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical framework, well-organized experiments
- Value: ⭐⭐⭐⭐ Practically informative for RAG optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/information_retrieval/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2026\] CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation](citeguard_faithful_citation_attribution_for_llms_via_retrieval-augmented_validat.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)

</div>

<!-- RELATED:END -->
