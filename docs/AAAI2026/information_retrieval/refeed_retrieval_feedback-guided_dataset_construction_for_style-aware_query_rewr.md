---
title: >-
  [Paper Note] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting
description: >-
  [AAAI 2026][Information Retrieval & RAG][query rewriting] This paper proposes a retrieval feedback-driven dataset construction framework that automatically builds high-quality style-aware query rewriting datasets through…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "query rewriting"
  - "retrieval feedback"
  - "style-aware"
  - "data-centric IR"
  - "RAG"
date: 2026-05-08
content_hash: 012b1725395e6367
---

# ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting

**Conference**: AAAI 2026
**arXiv**: [2603.01417](https://arxiv.org/abs/2603.01417)
**Code**: None
**Area**: Alignment RLHF / Information Retrieval & RAG
**Keywords**: query rewriting, retrieval feedback, style-aware, data-centric IR, RAG

## TL;DR
This paper proposes a retrieval feedback-driven dataset construction framework that automatically builds high-quality style-aware query rewriting datasets through a closed-loop pipeline of three steps: identifying retrieval failure cases, LLM-based stylistic rewriting, and re-retrieval verification. The resulting dataset provides a data foundation for training retrieval-aligned rewriting models.

## Background & Motivation
**Background**: Retrieval-Augmented Generation (RAG) systems are widely deployed in practice, yet a significant gap exists between the style of user queries (colloquial, informal) and that of domain documents (formal, specialized terminology), leading to retrieval failures.

**Limitations of Prior Work**: Existing query rewriting methods focus primarily on semantic fidelity while neglecting the stylistic characteristics of the target corpus (wording, tone, structure), resulting in rewritten queries that still deviate from the document distribution and yield suboptimal retrieval performance.

**Key Challenge**: Style-aware query rewriting requires large amounts of high-quality training data, yet existing datasets (e.g., CANARD, QReCC) contain only semantic rewrites and lack retrieval feedback and style variation information. Furthermore, existing methods apply feedback solely as a reinforcement learning signal during training, rather than during data construction.

**Goal**: To automatically construct a high-quality query rewriting dataset that jointly encodes retrieval feedback and document style alignment information.

**Key Insight**: Retrieval feedback is repurposed as a data filtering signal rather than a training signal, and datasets are constructed through a closed-loop pipeline of "failure identification → LLM rewriting → verification filtering."

**Core Idea**: Retrieval failure cases serve as the starting point for rewriting; an LLM rewrites queries in accordance with the style of the ground-truth document, and only rewritten pairs that pass re-retrieval verification are retained.

## Method

### Overall Architecture
The ReFeed framework consists of four stages: initial retrieval → LLM-guided rewriting → re-retrieval verification → dataset assembly. The overall design follows a closed-loop paradigm of "identify problem → fix problem → verify fix."

### Key Designs
1. **Initial Retrieval and Failure Identification**

   - **Function**: Dense retrieval is performed for each query in the QA dataset, and retrieval failure cases are flagged (i.e., the ground-truth document does not appear in the top-$k$ results).
   - **Mechanism**: An e5-base-v2 embedding model combined with a FAISS index is used to retrieve the top-3 documents; queries for which the ground-truth document is absent are labeled as misses.
   - **Design Motivation**: Failure cases represent the most valuable rewriting opportunities, and targeting these cases yields the greatest retrieval gains.

2. **LLM-Guided Stylistic Rewriting**

   - **Function**: An LLM rewrites queries for each miss case in a style-aware manner.
   - **Mechanism**: The prompt contains three key elements — the original query $q_i$, the incorrectly retrieved document $D_{\text{neg}}$, and the ground-truth document $D_{\text{pos}}$. By exposing the LLM to both positive and negative samples, it implicitly learns the linguistic and stylistic characteristics of the target document.
   - **Design Motivation**: By contrasting the stylistic differences between positive and negative documents, the rewritten query naturally aligns with the linguistic patterns of the target document, rather than performing simple semantic paraphrasing.

3. **Verification via Re-Retrieval**

   - **Function**: The rewritten query is submitted to retrieval again to verify whether the ground-truth document appears in the top-$k$ results.
   - **Mechanism**: Only successfully verified rewriting pairs are retained, ensuring that every data point is empirically validated.
   - **Design Motivation**: This establishes a closed-loop quality assurance mechanism, ensuring that every (original, rewritten) pair in the dataset carries a measurable retrieval gain.

### Loss & Training
ReFeed is a data generation framework rather than a model training method. The generated dataset can be used for:
- **Few-shot prompting**: The 5 most relevant rewriting examples are retrieved as in-context demonstrations.
- **Supervised Fine-Tuning (SFT)**: The dataset is used directly to train lightweight rewriting models.

## Key Experimental Results

### Main Results — Dataset Construction Statistics

| Metric | Value |
|--------|-------|
| SQuAD training set size | ~87k |
| Initial miss cases | ~16k (18.7%) |
| LLM rewriting success rate | 67.5% |
| Final verified pairs | 11,044 |
| Retrieval model | e5-base-v2 |
| LLM | GPT-5 (temperature=1.0) |

### Few-shot Validation Results

| Original Query | Rewritten Query | Rank Change |
|----------------|-----------------|-------------|
| Time Lord adversary query | More specialized formulation | Not in top-10 → Top-2 |
| Tribal site query | Expanded to archaeological phrasing | Top-8 → Top-1 |
| Missing ceremony content query | Semantically streamlined | Top-5 → Top-2 |
| Imperialist transportation query | Rephrased as imperial expansion | Not in top-10 → Top-1 |

### Key Findings
- The LLM's rewriting strategy is context-adaptive: simple queries are expanded for clarity, while complex queries are condensed to align with document style.
- Even without fine-tuning, few-shot usage alone improves retrieval rankings, demonstrating the standalone utility of the dataset.
- Rewriting is most effective for queries with large stylistic gaps (descriptive expressions, implicit entities, colloquial phrasing).
- A miss rate of 18.7% on SQuAD indicates that even classic datasets exhibit significant style mismatch problems.
- A rewriting success rate of 67.5% suggests that not all retrieval failures can be resolved through stylistic rewriting, as some may involve deeper semantic mismatches.

## Highlights & Insights
- **Data-centric IR perspective**: Retrieval feedback is repositioned from a "training signal" to a "data generation signal," opening a new methodological direction.
- **Style vs. semantics**: This work is among the first to explicitly distinguish style alignment from semantic fidelity in query rewriting, revealing a blind spot in existing methods.
- **Adaptive rewriting**: Rather than uniformly expanding or compressing queries, the LLM performs differentiated adjustments based on retrieval context — an emergent behavior driven by data.
- **Closed-loop verification**: Re-retrieval ensures data quality, avoiding the lack of quality guarantees common in conventional generative pipelines.
- **Prompt design consideration**: The LLM is explicitly instructed not to copy specific content from the ground-truth document and to rewrite solely based on the original query's intent, preventing information leakage.

## Limitations & Future Work
- Validation is conducted only on SQuAD (simple factoid QA); evaluation on complex domains (e.g., technical manuals, customer service logs) is absent.
- Few-shot validation is qualitative; large-scale quantitative evaluation (e.g., retrieval metrics over a full test set) is lacking.
- No SFT training experiments are conducted to verify the dataset's effectiveness for training lightweight rewriting models.
- The retrieval model is fixed as e5-base-v2; generalizability to different retrievers is not explored.
- The number of rewriting iterations and rewriting strategies could be further optimized.
- Multi-turn conversational query rewriting scenarios are not considered.
- Rewriting costs are high (GPT-5 is used); lightweight alternative rewriting models are not explored.
- The top-$k$ threshold is fixed at 3; the effect of different $k$ values on miss rate and data quality is not analyzed.

## Related Work & Insights
- **MaFeRw** (Wang et al. 2024): Integrates multi-faceted feedback (retrieval, generation, document similarity) into rewriting optimization → complementary to ReFeed's data-driven approach.
- **REPLUG** (Shi et al. 2023): Augments black-box LMs with retrieval feedback → ReFeed's dataset construction pipeline can provide better rewriting training data for such methods.
- **RaFe** (Mao et al. 2024): Fine-tunes a rewriter using reranker ranking signals → feedback is applied during training, whereas ReFeed moves feedback upstream to data construction.
- **CANARD/QReCC**: Large-scale conversational rewriting datasets lacking retrieval feedback and style information → ReFeed's dataset serves as a complementary resource.
- **Implications for RAG systems**: This framework can be used prior to RAG deployment to automatically discover and remediate retrieval gaps, forming a continuously improving closed loop.
- **Future directions**: The authors plan to train a small language model (SLM) rewriter on the ReFeed dataset and to integrate a selective rewriting module that determines whether rewriting is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of retrieval feedback-driven data construction is novel, though the overall pipeline is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐ The data construction pipeline is well-documented, but downstream validation is insufficient, with no SFT training or multi-domain evaluation.
- Writing Quality: ⭐⭐⭐⭐ The pipeline logic is clear and the motivation is well-articulated.
- Value: ⭐⭐⭐⭐ Directly applicable to real-world RAG systems, with broader methodological generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis](../../ICLR2026/information_retrieval/multimodal_dataset_distillation_made_simple_by_prototype-guided_data_synthesis.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](../../ACL2026/information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)

</div>

<!-- RELATED:END -->
