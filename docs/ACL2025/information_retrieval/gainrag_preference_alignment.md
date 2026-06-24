---
title: >-
  [Paper Note] GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] It is discovered that there is a systematic deviation between the "relevance" optimized by the retriever and the "gain" actually needed by the LLM in RAG—passages containing the gold answer still have a nearly 50% probability of causing incorrect generation, whereas indirectly relevant passages are often more effective. This paper proposes GainRAG, which defines the "gain" signal based on contrastive decoding perplexity and trains…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Preference Alignment"
  - "Contrastive Decoding"
  - "Gain Signal"
  - "Passage Selection"
date: 2026-05-08
content_hash: df13eb6b2fccdd78
---

# GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis

**Conference**: ACL 2025  
**arXiv**: [2505.18710](https://arxiv.org/abs/2505.18710)  
**Code**: [https://github.com/liunian-Jay/GainRAG](https://github.com/liunian-Jay/GainRAG)  
**Area**: Information Retrieval / RAG  
**Keywords**: RAG, Preference Alignment, Contrastive Decoding, Gain Signal, Passage Selection

## TL;DR

It is discovered that there is a systematic deviation between the "relevance" optimized by the retriever and the "gain" actually needed by the LLM in RAG—passages containing the gold answer still have a nearly 50% probability of causing incorrect generation, whereas indirectly relevant passages are often more effective. This paper proposes GainRAG, which defines the "gain" signal based on contrastive decoding perplexity and trains a lightweight selector to perform gain-oriented passage filtering between the retriever and the LLM. It comprehensively outperforms Standard RAG and Rerank baselines across six QA datasets.

## Background & Motivation

**Background**: RAG (Retrieval-Augmented Generation) is currently the dominant framework for enhancing LLM factuality. The standard process is "retrieval -> concatenation -> generation", where the retriever ranks passages based on semantic relevance and injects the top-k passages into the LLM context. This paradigm relies on an implicit assumption: **semantically relevant passages are helpful for the LLM to generate the correct answer**.

**Limitations of Prior Work**: Statistical experiments conducted by the authors on HotpotQA and 2WikiMultiHopQA reveal a severe issue with this assumption: (a) **Even if a passage contains the gold answer (gold answer), there is still a nearly 50% probability that it leads the LLM to generate an incorrect answer**—complex contexts or contradictory information interfere with the reasoning chain; (b) **Most passages used in correct generations do not directly contain the answer**—indirect clues or logical implications are more effective at steering the LLM toward correct reasoning. This indicates a systematic "preference gap" between the retriever and the LLM.

**Key Challenge**: The retriever optimizes the "semantic matching degree between the passage and the query" (**relevance**), while the LLM genuinely requires the "actual contribution of the passage to correct generation" (**gain**). Existing works (such as REPLUG and RA-DIT) bridge this preference by fine-tuning the retriever or joint training, which requires a large amount of high-quality data and is costly to implement. Although BGM and DPA-RAG introduce middleware, their measurement of preferences is too coarse-grained.

**Key Insight**: A "gain" metric based on contrastive decoding perplexity is defined to precisely quantify LLM preferences. A lightweight selector is trained using only a few samples to act as a plug-and-play middleware for preference alignment.

**Core Idea**: Quantify passage gain using contrastive decoding PPL, and train a selector to replace relevance-based ranking, thereby achieving preference alignment between the retriever and the LLM in RAG.

## Method

### Overall Architecture

GainRAG inserts a "gain selector" middleware between the retriever and the LLM. Inference pipeline: (1) The retriever retrieves the top-100 passages from the corpus; (2) The LLM generates a pseudo-passage as a candidate; (3) The selector predicts a gain score for each candidate passage; (4) The passage (or pseudo-passage) with the highest gain is selected and injected into the LLM to generate the final response. During training, the LLM is first used to synthesize gain labels via contrastive decoding PPL, which are then distilled into the lightweight selector.

### Key Designs

**1. Gain Signal Synthesis**

- **Function**: Precisely quantify the actual contribution of a passage to correct generation by the LLM.
- **Mechanism**: Contrastive decoding is introduced to calculate perplexity. Given a query $q$, passage $c$, and gold answer $a$, the logits are calculated with and without the passage, respectively. The contrastive probability distribution is obtained as $(1+\alpha) \cdot \text{logit}(a_t|c,q) - \alpha \cdot \text{logit}(a_t|q)$, and its perplexity is then calculated as the gain metric $\mathcal{M}(c,a|q)$.
- **Design Motivation**: Directly using PPL would be dominated by the intrinsic knowledge of the LLM (PPL can be low even without providing the passage). Contrastive decoding removes the influence of model priors and focuses on the knowledge increment of the passage itself. The hyperparameter is set to $\alpha=0.5$ (following CAD).

**2. Selector Distillation**

- **Function**: Distill the gain-awareness capability of the LLM into a lightweight selector.
- **Mechanism**: From HotpotQA (20k samples) and WebQuestions (4k samples), 20 retrieved passages + 1 pseudo-passage are extracted for each query. Gain labels are calculated using LLaMA3-8B (processed via $v=-\log(v+1)$ to handle long-tail distributions). BGE-reranker-base is fine-tuned as the selector using KL divergence distillation loss for only 2 epochs.
- **Design Motivation**: The overhead of using the LLM for forward propagation to calculate gain for each passage during inference is computationally prohibitive. Distilling it to a small model allows for highly efficient inference. Since BGE-reranker-base already possesses passage semantic understanding capabilities, it can rapidly learn gain-based ranking after fine-tuning.

**3. Pseudo-passage Strategy**

- **Function**: Prevent degradation issues when the gain of all retrieved passages is negative.
- **Mechanism**: Before inference, the LLM first generates a pseudo-passage $c_0$ based on the query, which is then added to the selector candidate list. If the pseudo-passage receives the highest gain score, it suggests that the retrieved passages are less useful than the LLM's internal knowledge. In this case, the pseudo-passage is processed as the context.
- **Design Motivation**: External retrieved passages for some queries are indeed less reliable than the model's own knowledge (e.g., 50% of the queries on 2WikiMultiHopQA select the pseudo-passage). This strategy enables dynamic switching between internal and external knowledge, avoiding the forced use of harmful passages.

### Training Details

| Configuration | Setting |
|--------|------|
| Gain Generator | LLaMA3-8B |
| Selector Backbone | BGE-reranker-base |
| Training Data | ~10k samples (after filtering) |
| Retriever | Contriever, k=100 |
| Distillation Loss | KL divergence |
| Training Epochs | 2 epochs |
| Hardware | Single A100 80GB GPU |
| Contrastive Decoding $\alpha$ | 0.5 |

## Key Experimental Results

### Main Results (6 QA Datasets)

| Dataset | Method | EM | F1 | Avg |
|--------|------|-----|-----|-----|
| HotpotQA | Standard RAG | 31.80 | 33.23 | 32.51 |
| HotpotQA | Rerank | 35.80 | 37.45 | 36.62 |
| HotpotQA | **GainRAG** | **39.60** | **41.99** | **40.79** |
| 2WikiMQA | Standard RAG | 23.40 | 21.81 | 22.61 |
| 2WikiMQA | **GainRAG** | **31.40** | **28.92** | **30.16** |
| WebQuestions | Naive (No Retrieval) | 44.39 | 35.90 | 40.14 |
| WebQuestions | Standard RAG | 35.04 | 33.26 | 34.15 |
| WebQuestions | **GainRAG** | **42.51** | **39.17** | **40.84** |
| NaturalQA | Standard RAG | 38.14 | 36.82 | 37.48 |
| NaturalQA | **GainRAG** | **41.97** | **41.27** | **41.62** |
| TriviaQA | Standard RAG | 62.16 | 61.87 | 62.02 |
| TriviaQA | **GainRAG** | **67.29** | **66.73** | **67.01** |

### Ablation Study

| Variant | HotpotQA Avg | 2WikiMQA Avg | NaturalQA Avg |
|------|-------------|-------------|--------------|
| w/o all (vanilla reranker) | 36.62 | 23.57 | 30.73 |
| w/o pseudo (without pseudo-passages) | 39.23 | 26.04 | 41.09 |
| w/o distillation (without distillation) | 35.02 | 28.14 | 31.90 |
| **GainRAG (Full)** | **40.79** | **30.16** | **41.62** |

### Key Findings

- GainRAG achieves SOTA across all 6 datasets, improving by **5-8 percentage points** on average compared to Standard RAG.
- On WebQuestions, all RAG methods perform worse than the no-retrieval baseline (Naive), indicating that blind retrieval can be harmful; however, GainRAG still achieves the optimal Avg through the pseudo-passage strategy.
- Ablation. studies demonstrate that distillation fine-tuning and the pseudo-passage strategy are complementary and indispensable: omitting distillation drops HotpotQA performance by 5.77, and omitting pseudo-passages drops 2WikiMQA performance by 4.12.
- Removing contrastive decoding for signal synthesis (using standard PPL instead) leads to drops of 1.00 and 1.90 on HotpotQA and 2WikiMQA, respectively, validating the necessity of contrastive debiasing.
- Selecting only the top-1 passage achieves peak performance; increasing the $K$ value does not improve downstream generation quality. While recall increases, accuracy remains unchanged, further verifying that "relevance $\neq$ utility".

## Highlights & Insights

- **Empirical analysis of "Relevance $\neq$ Utility"** breaks the intuitive assumption of RAG—the statistic showing that 50% of gold passages still cause incorrect generation is highly convincing and provides solid motivation for "gain-oriented selection."
- **Using contrastive decoding PPL as a gain metric** is a core innovation. It does not require human annotations and cleverly utilizes the probability difference with and without the passage to quantify the passage's contribution while eliminating any bias from the model's prior knowledge.
- **The pseudo-passage strategy** achieves the capability of "knowing when not to use external knowledge"—with 50% of queries on 2WikiMQA selecting the pseudo-passage, proving that internal knowledge is sometimes more reliable.
- **Extremely high data efficiency**—training with only ~10k samples (after filtering) for 2 epochs yields a selector that generalizes well across datasets, indicating a strong signal-to-noise ratio for the gain signal.
- The entire method is designed as a plug-and-play middleware that **does not require modifications to the retriever or generator**, making it engineering-friendly for real-world deployment.

## Limitations & Future Work

- Synthesizing gain signals requires forward propagation through the LLM for every passage, leading to high initial annotation computation costs.
- It has only been validated on QA tasks; its applicability to scenarios like long-text summarization and conversational retrieval remains unexplored.
- The method only selects the top-1 passage; whether multi-passage combinations could yield higher gains is yet to be investigated.
- Whether signal generation can be accelerated by replacing large models with smaller ones requires further experimentation.
- The capacity limitations of BGE-reranker-base might affect its ability to capture subtle gain differences in more complex scenarios.

## Related Work & Insights

- **vs Rerank (BGE-reranker)**: Traditional reranking is still based on semantic relevance ranking, while GainRAG ranks by gain. The fundamental difference lies in shifting the optimization objective from "matching degree" to "actual utility to the LLM".
- **vs BGM / DPA-RAG**: While both serve as middleware between the retriever and the LLM, the former uses coarse-grained labels (helpful/unhelpful), whereas GainRAG utilizes continuous gain scores to achieve finer-grained preference awareness.
- **vs Self-RAG**: Self-RAG requires large amounts of annotated data to fine-tune the LLM to learn self-reflection. In contrast, GainRAG only requires a small amount of data to train an external selector, incurring significantly lower costs.
- **vs Replug / RA-DIT**: These methods fine-tune the retriever or perform joint training, modifying the entire RAG stack. GainRAG's selector remains independent of both ends, enabling plug-and-play usage.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Defining gain signals via contrastive decoding PPL is novel, and the empirical analysis of the preference gap is solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage with 6 datasets + ablation studies + signal analysis + pseudo-passage analysis, though lacking validation on non-QA tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Data-driven motivations, clear method formulations, and systematic experimental analysis.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play middleware design provides direct guidance for RAG implementation, and the principle of "selecting passages by gain rather than relevance" has universal significance.

## Technical Details (Supplementary)
- Gain signals are based on contrastive decoding: comparing the change in PPL with and without the passage to eliminate intrinsic knowledge bias.
- The selector is fine-tuned based on BERT-base; it takes (query, passage) as input and outputs a gain score, optimized with MSE loss.
- Pseudo-passage: adding empty content to the candidates; if it scores the highest, external knowledge is not used.
- An effective selector can be trained using only 200 queries (with 100 retrieved passages each), amounting to approximately 20k passage-level annotations.
- Yields an average improvement of 5-8 EM points across 6 QA datasets, with the largest improvement on multi-hop reasoning tasks.

## Technical Details (Supplementary)
- Gain signals are based on contrastive decoding: comparing the change in PPL with and without the passage to eliminate intrinsic knowledge bias.
- The selector is fine-tuned based on BERT-base; it takes (query, passage) as input and outputs a gain score, optimized with MSE loss.
- Pseudo-passage: adding empty content to the candidates; if it scores the highest, external knowledge is not used.
- An effective selector can be trained using only 200 queries (with 100 retrieved passages each), amounting to approximately 20k passage-level annotations.
- Yields an average improvement of 5-8 EM points across 6 QA datasets, with the largest improvement on multi-hop reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
