---
title: >-
  [Paper Note] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport
description: >-
  [ACL 2025][Text Generation][MBR Decoding] Proposes MBR-OT, which introduces Optimal Transport (Wasserstein distance) into Minimum Bayes Risk (MBR) decoding to evaluate document-level output quality using sentence-level utility functions. It significantly outperforms standard MBR decoding on document-level machine translation, text simplification, and dense image captioning tasks.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "MBR Decoding"
  - "Optimal Transport"
  - "Document-Level Generation"
  - "Wasserstein Distance"
  - "Machine Translation"
date: 2026-05-08
content_hash: 04fb47794f1aa876
---

# Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport

**Conference**: ACL 2025  
**arXiv**: [2505.23078](https://arxiv.org/abs/2505.23078)  
**Code**: [https://github.com/jinnaiyuu/mbr-optimal-transport](https://github.com/jinnaiyuu/mbr-optimal-transport)  
**Area**: Text Generation  
**Keywords**: MBR Decoding, Optimal Transport, Document-Level Generation, Wasserstein Distance, Machine Translation  

## TL;DR

Proposes MBR-OT, which introduces Optimal Transport (Wasserstein distance) into Minimum Bayes Risk (MBR) decoding to evaluate document-level output quality using sentence-level utility functions. It significantly outperforms standard MBR decoding on document-level machine translation, text simplification, and dense image captioning tasks.

## Background & Motivation

**Background**: MBR decoding has achieved excellent performance in sentence-level text generation tasks by selecting the candidate output with the highest expected utility to replace greedy/beam search. However, the performance of MBR remains limited in document-level generation tasks (e.g., full-document translation, long text simplification).

**Limitations of Prior Work**: MBR relies on a utility function to measure the quality of candidate outputs, but most utility functions (e.g., BLEU, BERTScore, COMET) are designed for the sentence level. Directly using sentence-level metrics to evaluate an entire document ignores structural variations in the document (such as sentence reordering, merging/splitting).

**Key Challenge**: Sentence-level utility functions assume a one-to-one alignment between sentences in the source and target texts. However, sentences are frequently reordered, merged, or split in document-level translation (especially for language pairs with large structural differences like English-Japanese), leading to inaccurate utility assessment.

**Goal**: How to upgrade well-established sentence-level utility functions to document-level utility functions while maintaining robustness against document structural changes.

**Key Insight**: Optimal Transport theory provides a mathematical framework for comparing differences between two distributions, naturally supporting flexible matching between elements—allowing the "quality" of a single source sentence to be distributed across multiple target sentences.

**Core Idea**: Aggregate sentence-level utility functions into a document-level utility function using Wasserstein distance, enabling robust evaluation of sentence reordering/merging.

## Method

### Overall Architecture
Taking a source document as input, the model generates multiple candidate documents. MBR-OT utilizes an optimal transport-based utility function to evaluate the relative quality among the candidates, selecting the candidate with the highest expected utility as the final output.

### Key Designs

1. **Document Segmentation and Sentence-Level Utility Computation**:

    - **Function**: Segments candidate and reference documents into sets of sentences.
    - **Mechanism**: Treats documents as distributions of sentences (rather than sequences) and uses a sentence-level utility function $u(h_i, y_j)$ to compute the utility between any two sentences.
    - **Design Motivation**: Free from the reliance on fixed sentence alignment, enabling flexible matching.

2. **Document Utility Function based on Wasserstein Distance**:

    - **Function**: Aggregates sentence-level utilities into a document-level utility using optimal transport.
    - **Mechanism**: Treats documents $\mathbf{h}$ and $\mathbf{y}$ as two discrete distributions $p_\mathbf{h}$ and $p_\mathbf{y}$, and uses the Wasserstein distance $\text{WD}_C[p_\mathbf{h} \| p_\mathbf{y}] = \inf_{\gamma \in \Gamma(p_\mathbf{h}, p_\mathbf{y})} \sum_{(i,j)} \gamma(h_i, y_j) C(h_i, y_j)$ to compute the minimum transport cost. Unlike Linear Assignment (LA), WD allows the weight of one source sentence to be distributed across multiple target sentences.
    - **Design Motivation**: Handles sentence merging/splitting scenarios—such as associating "I like cats and dogs" with "I like cats. I like dogs.", where WD can split the source sentence weight across two target sentences.

3. **Various OT Variants**:

    - **Linear Assignment (LA)**: One-to-one matching, restricted but simple.
    - **Wasserstein Distance (WD)**: Many-to-many matching, more flexible.
    - **Entropy-Regularized WD (EWD)**: Adds KL regularization $\epsilon$ to smooth the optimization and improve computational efficiency (Sinkhorn algorithm).
    - A sentence length-weighted version (subscript $L$) is also provided, allocating weights based on sentence lengths.

### Loss & Training
- **Training-Free**: Purely inference-time method, modifying only the utility function in MBR decoding.
- Using MetricX-23 as the sentence-level utility function yields the best performance.
- Samples 32 candidate outputs for MBR selection.

## Key Experimental Results

### Main Results (Document-Level Machine Translation, WMT24)

| Method | MetricX En-Ja | MetricX En-De |
|------|-------------|-------------|
| Beam Search | 61.57 | 79.07 |
| MBR (MetricX) | 68.81 | 82.02 |
| MBR-LA (MetricX) | 70.01 | 80.77 |
| **MBR-WD (MetricX)** | **75.29** | **83.40** |
| MBR-WD$_L$ (MetricX) | 72.38 | 83.24 |
| MBR-EWD$_L$ (MetricX) | 70.67 | 83.24 |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| LA vs WD | WD significantly outperforms LA | Many-to-many matching is better suited for the document level |
| Uniform vs Length Weights | Task-dependent | Uniform is better for En-Ja, while length is better for En-De |
| Different $\epsilon$ values | $\epsilon=0$ (pure WD) is best | Regularization is not necessary for this task |
| Different sentence-level utility functions | MetricX-23 >> COMET >> BERTScore | The quality of the utility function is key |

### Key Findings
- MBR-WD improves MetricX from 68.81 to 75.29 on En-Ja translation—a 6.5-point gain over standard MBR.
- WD aligns well with human evaluation in terms of WMT system-level correlation (>0.88 in most configurations).
- Sentence merging/splitting is very common in document-level translation—with an average of 3.8 sentences in En-Ja candidates compared to 5.5 sentences in the references.
- It is equally effective in text simplification and dense image captioning tasks, demonstrating the framework's universality.
- The computational overhead mainly lies in the number of calls to the sentence-level utility function ($O(mn)$), but it can be accelerated using the Sinkhorn algorithm.

## Highlights & Insights

- **Bridging sentence-level and document-level metrics using optimal transport** is an elegant theoretical contribution—Wasserstein distance is naturally suited for handling flexible matching between distributions.
- The perspective shift of **"treating a document as a distribution of sentences rather than a sequence"** is a key insight—by abandoning the assumption of fixed alignment, the method becomes robust to structural changes.
- The approach is independent of the specific sentence-level utility function—as sentence-level metrics improve (such as the MetricX series), the performance of MBR-OT will automatically scale up.
- This OT idea can be transferred to any scenario that requires aggregating local metrics into global metrics (e.g., paragraph-level summarization evaluation).

## Limitations & Future Work

- The calculation of the utility matrix scales as $O(mn)$ ($m$, $n$ being the number of sentences in the two documents), which becomes computationally expensive for long documents.
- Currently verified only on small-to-medium scale LLMs; performance on large-scale models remains unexplored.
- WD assumes that sentence order within the document does not matter—but order can be crucial in certain tasks.
- Combining WD with other decoding strategies (such as speculative decoding) has not been explored.
- Relies only on automatic metrics evaluation, lacking human evaluation.

## Related Work & Insights

- **vs Standard MBR**: Standard MBR treats the entire document as a single string to compute utility, failing to handle structural variations; MBR-OT decomposes documents to the sentence level and performs flexible matching.
- **vs Vernikos et al. (2022) Document-level Metric**: They assume sentences align sequentially, which fails in sentence-reordering scenarios; WD does not require this assumption.
- **vs Word Mover's Distance**: WMD performs OT at the word level, while this work operates at the sentence level, which is a granularity better suited for document-level tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing optimal transport for MBR decoding is a natural yet effective combination—Wasserstein distance naturally handles document structural changes, and the mathematical justification is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks (machine translation/text simplification/dense image captioning) + multiple OT variants + comparison across various utility functions + WMT system-level correlation verification, though they lack human evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical formulation, clear intuitive explanations (the sentence-splitting example in Figure 1 explains why WD is superior to LA), with a natural progression from LA $\rightarrow$ WD $\rightarrow$ EWD.
- Value: ⭐⭐⭐⭐ Provides a general tool for document-level text generation decoding and evaluation—with the advancement of sentence-level metrics (like MetricX), the performance of MBR-OT will scale up automatically.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation](../../ACL2026/nlp_generation/losses_that_cook_topological_optimal_transport_for_structured_recipe_generation.md)
- [\[ACL 2025\] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)
- [\[ACL 2025\] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)

</div>

<!-- RELATED:END -->
