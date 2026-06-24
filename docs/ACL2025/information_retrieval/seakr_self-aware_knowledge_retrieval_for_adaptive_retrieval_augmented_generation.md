---
title: >-
  [Paper Note] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][Adaptive Retrieval] SeaKR leverages the self-aware uncertainty of LLMs' internal hidden layers (measured by the Gram determinant of hidden representations from multiple EOS token samplings) to adaptively decide when to retrieve, how to re-rank retrieval results, and which reasoning strategies to select. It improves F1 in multi-hop QA by 6% compared to DRAGIN and 9.5% compared to IRCoT.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Adaptive Retrieval"
  - "Self-aware Uncertainty"
  - "Internal States"
  - "Knowledge Re-ranking"
  - "Iterative Reasoning"
date: 2026-05-08
content_hash: 1dbd879b1d32fc4e
---

# SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2406.19215](https://arxiv.org/abs/2406.19215)  
**Code**: Open-sourced (GitHub)  
**Area**: RAG / Retrieval-Augmented Generation  
**Keywords**: Adaptive Retrieval, Self-aware Uncertainty, Internal States, Knowledge Re-ranking, Iterative Reasoning

## TL;DR

SeaKR leverages the self-aware uncertainty of LLMs' internal hidden layers (measured by the Gram determinant of hidden representations from multiple EOS token samplings) to adaptively decide when to retrieve, how to re-rank retrieval results, and which reasoning strategies to select. It improves F1 in multi-hop QA by 6% compared to DRAGIN and 9.5% compared to IRCoT.

## Background & Motivation

**Background**: RAG mitigates hallucination issues by injecting external knowledge into the LLM context. Adaptive RAG further optimizes this by retrieving only when necessary to avoid interference from irrelevant knowledge. Existing adaptive methods, such as FLARE and DRAGIN, determine when to retrieve using output token probabilities, while Self-RAG utilizes fine-tuning to make the model generate special tokens indicating retrieval needs.

**Limitations of Prior Work**: (1) Existing methods judge retrieval needs solely based on LLM outputs, but LLMs exhibit systematic overconfidence bias—they confidently output incorrect answers even when parametric knowledge is insufficient, meaning output probabilities do not reliably reflect actual knowledge gaps. (2) Existing methods overlook the integration of retrieved knowledge—how to select the most useful pieces after retrieving multiple results, and how to integrate multiple pieces of knowledge for reasoning? These issues remain largely unresolved systematically.

**Key Challenge**: Adaptive RAG requires an accurate perception of "what is unknown", but LLM outputs are filtered by discretization and overconfidence bias, resulting in severe information loss. In theory, lower-level internal states carry richer uncertainty information, but they have not yet been utilized in RAG.

**Goal**: To leverage the self-aware uncertainty of LLMs' internal hidden states to drive uncertainty-driven decisions across the three critical stages of adaptive RAG: when to retrieve, how to integrate, and how to reason.

**Key Insight**: Based on findings in cognitive science that "multiple samplings of uncertain content by LLMs exhibit inconsistency in the hidden space," SeaKR measures the consistency of multiple generated EOS hidden representations using the Gram determinant as an uncertainty score. High uncertainty triggers retrieval, while knowledge segments with low uncertainty are prioritized.

**Core Idea**: To use the consistency of multiple sampled EOS token hidden representations from the intermediate layer of the LLM as a self-aware uncertainty metric, driving retrieval triggering, knowledge re-ranking, and reasoning strategy selection.

## Method

### Overall Architecture

SeaKR adopts a CoT-style iterative reasoning framework. For each step: (1) candidate reasoning steps are generated, and a self-aware uncertainty evaluator assesses whether retrieval is required; (2) if retrieval is necessary, the knowledge that best reduces uncertainty is selected from the top-N snippets returned by the search engine; (3) after the iterative process, self-aware reasoning chooses the result with lower uncertainty between two strategies: "direct generation" and "CoT reasoning based on all retrieved knowledge."

### Key Designs

1. **Self-aware Uncertainty Evaluator**:

    - **Function**: Quantifies the LLM's confidence level in the currently generated content based on its internal states.
    - **Mechanism**: For the same input context $\mathbf{c}$, generation is sampled $k=20$ times. The hidden representations $\mathbf{H}^{(l)}_{\langle EOS \rangle}$ of the EOS token in the intermediate layer $l=L/2$ are collected for each generation. The determinant of the normalized Gram matrix of these $k$ vectors is calculated as the uncertainty score $U(\mathbf{c})$. A larger determinant indicates more dispersed vectors and higher model uncertainty.
    - **Design Motivation**: Compared to output token probabilities, consistency measurement at the internal state level avoids interference from natural language polysemy—different surface expressions can represent the same semantics and cause low token probabilities, whereas hidden states remain consistent in the semantic space.

2. **Self-aware Re-ranking**:

    - **Function**: Selects the knowledge snippet from the top-N search engine results that minimizes the LLM's uncertainty.
    - **Mechanism**: Appends each candidate knowledge snippet to the input context individually, independently calculates the uncertainty $U(\mathbf{c} + \mathbf{k}_i)$ for the $N$ augmented contexts, and selects the snippet that yields the lowest uncertainty. This shifts the perspective from "retrieval relevance" to "the most helpful to the model."
    - **Design Motivation**: Traditional RAG ranks knowledge by query relevance, but high relevance does not equate to high utility—it might conflict with the model's existing knowledge or fail to reduce specific uncertainty.

3. **Self-aware Reasoning**:

    - **Function**: Selects the optimal knowledge integration strategy to generate the final answer after iterative reasoning finishes.
    - **Mechanism**: Provides two reasoning strategies: (1) direct generation of the answer from the final step of the CoT chain; (2) appending all re-ranked retrieved knowledge as a context to perform a new CoT reasoning. The uncertainty of the answers from both strategies is calculated, and the one with higher confidence is selected.
    - **Design Motivation**: The optimal reasoning path varies across different questions—for some, the CoT reasoning chain is already sufficient and extra knowledge acts as interference; for others, comprehensive reasoning integrating all knowledge is required.

### Loss & Training

SeaKR is a tuning-free method. All components are based on the original reasoning capabilities and internal states of LLaMA-2-7B-chat without requiring additional fine-tuning. The uncertainty threshold $\delta$ is determined by searching on a small subset of the NQ training set.

## Key Experimental Results

### Main Results

Multi-hop QA (Multi-hop reasoning, LLaMA-2-7B-chat + BM25):

| Method | 2WikiMultiHop F1 | HotpotQA F1 | IIRC F1 |
|------|------------------|-------------|---------|
| CoT (No Retrieval) | 22.3 | 27.5 | 17.3 |
| IRCoT (Retrieval per Step) | 26.5 | 30.4 | 21.6 |
| Self-RAG (Fine-tuned) | 19.6 | 17.5 | 5.7 |
| FLARE | 21.3 | 22.1 | 16.4 |
| DRAGIN | 30.0 | 34.2 | 22.9 |
| **SeaKR** | **36.0** | **39.7** | **23.5** |

### Ablation Study

| Ablation Configuration | 2Wiki F1 | HotpotQA F1 | Explanation |
|---------|----------|-------------|------|
| SeaKR (Full) | **36.0** | **39.7** | Baseline |
| w/o Self-aware Retrieval (Retrieval per Step) | 33.5 | 37.1 | -2.5/-2.6, validates the effectiveness of adaptive retrieval |
| w/o Self-aware Re-ranking (Using top-1) | 31.2 | 35.6 | -4.8/-4.1, re-ranking contributes the most |
| w/o Self-aware Reasoning (Direct generation) | 33.8 | 37.9 | -2.2/-1.8 |
| Replace internal states with output probabilities | 27.8 | 33.1 | -8.2/-6.6, validates that internal states outperform output probabilities |

### Key Findings

- **Self-aware re-ranking contributes the most**: Ablations show that removing re-ranking drops F1 by 4-5%, which is larger than the drops from removing self-aware retrieval (-2.5%) and reasoning (-2.2%), indicating that "how to integrate knowledge" is more critical than "when to retrieve".
- **Internal states far outperform output probabilities**: Replacing internal states with output probabilities results in a 6-8% F1 drop, verifying that internal states carry richer uncertainty information.
- **Self-RAG collapses in multi-hop QA**: Self-RAG fine-tuned on NQ achieves an F1 of only 17.5-19.6% on multi-hop QA, which is even lower than CoT without retrieval, indicating severe out-of-distribution (OOD) shift issues in fine-tuning-based methods.
- **Adaptive retrieval vs. full retrieval**: SeaKR retrieves only about 60% as many times as IRCoT but achieves a 9.5% higher F1, proving that reducing unnecessary retrieval indeed boosts performance.

## Highlights & Insights

- **Internal state-driven RAG decision-making is a new paradigm**: SeaKR is the first to introduce the self-aware uncertainty of LLMs' internal states into every stage of RAG (retrieval, re-ranking, and reasoning), establishing an "uncertainty-driven adaptive RAG" paradigm.
- **Gram determinant as an uncertainty metric**: Quantifying uncertainty based on the consistency of multiple sampled hidden representations strikes a balance between theoretical elegance (the Gram determinant measures linear independence) and practical effectiveness (tuning-free).
- **Generalizes without training**: Unlike the fine-tuning scheme of Self-RAG which collapses under distribution shift, SeaKR's tuning-free design remains robust across different types of QA tasks.

## Limitations & Future Work

- **High inference overhead**: Each uncertainty evaluation requires 20 samplings, and each re-ranking requires N×20 inferences, leading to a computational cost significantly higher than that of baseline methods.
- **Only validated on LLaMA-2-7B**: Lacks evaluation on larger or more recent LLMs.
- **BM25 retriever limitation**: Using BM25 instead of dense retrievers makes the retrieval quality a potential performance bottleneck.
- **Less pronounced advantages in simple QA**: Self-RAG still performs well in simple QA, indicating that SeaKR's advantages lie primarily in multi-hop reasoning scenarios.

## Related Work & Insights

- **vs. DRAGIN**: DRAGIN also performs adaptive retrieval but only uses output token probabilities; SeaKR uses internal states, achieving a 6% higher F1.
- **vs. Self-RAG**: Self-RAG fine-tunes the model to judge retrieval needs, but suffers from severe distribution shift issues; SeaKR requires no fine-tuning and offers better generalization.
- **vs. INSIDE (Chen et al. 2023a)**: SeaKR scales the internal state uncertainty measurement of INSIDE from hallucination detection to the entire RAG pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce internal state self-awareness into the retrieval, re-ranking, and reasoning stages of RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets with detailed ablations and clear analysis of component contributions.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the framework architecture is well-structured.
- Value: ⭐⭐⭐⭐ Pioneered the uncertainty-driven adaptive RAG paradigm, yielding significant improvements particularly for multi-hop QA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)

</div>

<!-- RELATED:END -->
