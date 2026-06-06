---
title: >-
  [Paper Note] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs
description: >-
  [ACL 2026][Information Retrieval & RAG][Utility Judgment] Inspired by Schutz’s philosophical relevance theory, this paper proposes the ITEM iterative utility judgment framework. By enabling dynamic interaction and mutual…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Utility Judgment"
  - "Philosophical Relevance Theory"
  - "Iterative Framework"
  - "RAG Optimization"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: f9a8529e72ca5081
---

# An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs

**Conference**: ACL 2026  
**arXiv**: [2406.11290](https://arxiv.org/abs/2406.11290)  
**Code**: [GitHub](https://github.com/Trustworthy-Information-Access/ITEM)  
**Area**: Information Retrieval / RAG  
**Keywords**: Utility Judgment, Philosophical Relevance Theory, Iterative Framework, RAG Optimization, LLM Reasoning

## TL;DR

Inspired by Schutz’s philosophical relevance theory, this paper proposes the ITEM iterative utility judgment framework. By enabling dynamic interaction and mutual enhancement among three RAG components (relevance reranking, utility judgment, and answer generation), it outperforms baselines in retrieval, utility judgment, and QA tasks.

## Background & Motivation

**Background**: In RAG scenarios, LLM input bandwidth is limited, necessitating the prioritization of high-utility (rather than just high-relevance) retrieval results. Relevance measures "whether it is about the topic," while utility measures "whether it helps answer the question."

**Limitations of Prior Work**: (1) Existing RAG methods primarily optimize topical relevance, ignoring the higher standard of utility; (2) While Zhang et al. proposed LLM utility judgment, it remains a preliminary exploration; (3) The three components of RAG (retrieval, judgment, and generation) are typically optimized independently, lacking joint enhancement.

**Key Challenge**: Topically relevant documents are not necessarily useful—a document discussing the same topic but lacking specific answers is relevant but useless. Existing methods struggle to distinguish between the two.

**Goal**: Enhance the utility judgment capability of LLMs through the iterative interaction of the three RAG components.

**Key Insight**: The paper maps RAG to Schutz’s philosophical "relevance system"—topical relevance, interpretational relevance (utility), and motivational relevance (answer) correspond to three cognitive levels, which can mutually enhance one another.

**Core Idea**: The three components of RAG reflect three cognitive levels of LLMs in question answering (aboutness → value → answer). They are mutually enhanced through iteration.

## Method

### Overall Architecture

ITEM offers two variants: ITEM-A (iterative answer generation + utility judgment) and ITEM-AR (iterative answer generation + relevance reranking + utility judgment). In each iteration, the LLM first generates a pseudo-answer, then refines utility judgment or reranking based on that pseudo-answer, and finally regenerates the answer in a continuous loop.

### Key Designs

1.  **Iterative Utility Judgment Mechanism**:
    - **Function**: Gradually improves the quality of utility judgment through multi-round iterations.
    - **Mechanism**: In each round, the LLM first generates a pseudo-answer (serving as a cognitive anchor), then performs utility judgment (pointwise or listwise) based on that pseudo-answer, and finally updates the pseudo-answer. The judgment quality improves over multiple rounds.
    - **Design Motivation**: Single-shot judgments are easily influenced by noise; iteration allows the LLM to progressively accumulate understanding of both the question and the documents.

2.  **Mapping Philosophical Theory to RAG**:
    - **Function**: Provides a theoretical foundation for the iterative framework.
    - **Mechanism**: Schutz’s three types of relevance—topical relevance (focusing on the object) → interpretational relevance (understanding the object) → motivational relevance (acting based on understanding)—correspond one-to-one with RAG's retrieval → utility judgment → answer generation.
    - **Design Motivation**: Philosophical theory predicts that these three will mutually enhance each other, providing a theoretical basis for the iterative design.

3.  **Comparison of Two Iterative Variants**:
    - **Function**: Explores the applicability of different iterative strategies.
    - **Mechanism**: ITEM-A iterates only on answer + judgment (fewer components, more rounds); ITEM-AR adds the reranking component (more components, richer information per round). Different task complexities require different strategies.
    - **Design Motivation**: Difficult tasks require more components and more rounds, while simple tasks can be handled with lightweight strategies.

### Loss & Training

No training required. The framework is entirely based on LLM in-context learning. Task execution for each round (answer generation/utility judgment/ranking) is controlled via prompt design.

## Key Experimental Results

### Main Results

| Task | Dataset | ITEM Gain | Description |
|------|--------|---------|------|
| Retrieval Reranking | TREC DL | Superior to baselines | Utility judgment in turn improves reranking |
| Utility Judgment | GTI-NQ | Superior to baselines | Iteration significantly improves judgment quality |
| QA | NQ | Superior to baselines | High-utility documents lead to better answers |
| Non-factoid Retrieval | WebAP | Superior to baselines | Greater gains in difficult tasks |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| 1 Round vs Multi-round | Multi-round is better | Iteration is indeed effective |
| ITEM-A vs ITEM-AR | Task-dependent | Difficult tasks require ITEM-AR |
| vs Long-reasoning mode | Comparable performance, much lower cost | Iteration is more efficient than one-shot long reasoning |

### Key Findings
- In difficult tasks (e.g., WebAP non-factoid answer retrieval) and complex candidate lists (e.g., GTI-NQ), using more components and more iterations is most effective.
- ITEM achieves performance comparable to long-reasoning modes but at a significantly lower computational cost.
- In simple factoid QA tasks, using fewer components with more iterations actually yields the best results.

## Highlights & Insights
- Creative mapping of philosophical theory to engineering methods—Schutz’s relevance system provides a new perspective for RAG optimization.
- Identifies the relationship between task complexity and optimal iteration strategies.
- Improves RAG quality without training, demonstrating strong practical utility.

## Limitations & Future Work
- Iteration increases inference costs (multiple LLM calls), and latency may be unacceptable for some applications.
- The quality of pseudo-answers may limit the upper bound of iteration gains.
- Evaluated only on English datasets.
- Future work could combine this with fine-tuned retrievers for further enhancement.

## Related Work & Insights
- **vs Single-shot Utility Judgment**: The iterative framework significantly improves judgment quality through multi-round cognitive accumulation.
- **vs Multi-round Retrieval RAG**: Does not change retrieval itself but iteratively improves utility judgment on already retrieved results.
- **vs Long-reasoning/Chain-of-Thought**: Achieves comparable effects at a lower cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative perspective mapping philosophical theory to RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 4 datasets and multiple tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical framework and well-organized experiments.
- Value: ⭐⭐⭐⭐ Practical guiding significance for RAG optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](../../NeurIPS2025/information_retrieval/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2026\] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](utility-oriented_visual_evidence_selection_for_multimodal_retrieval-augmented_ge.md)
- [\[ACL 2026\] CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation](citeguard_faithful_citation_attribution_for_llms_via_retrieval-augmented_validat.md)

</div>

<!-- RELATED:END -->
