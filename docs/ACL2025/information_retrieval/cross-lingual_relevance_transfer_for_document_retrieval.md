---
title: >-
  [Paper Note] Cross-Lingual Relevance Transfer for Document Retrieval
description: >-
  [ACL 2025][Information Retrieval & RAG][Cross-Lingual Retrieval] This paper proposes a cross-lingual relevance transfer method that transfers relevance judgment capability to low-resource languages using a retrieval model trained on high-resource languages (e.g., English), significantly outperforming existing methods on multiple cross-lingual document retrieval benchmarks.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Cross-Lingual Retrieval"
  - "Relevance Transfer"
  - "Document Retrieval"
  - "Multilingual Representation"
  - "Zero-Shot Transfer"
date: 2026-05-08
content_hash: 940ae13603122f10
---

# Cross-Lingual Relevance Transfer for Document Retrieval

**Conference**: ACL 2025  
**Area**: NLP Understanding / Cross-Lingual Information Retrieval  
**Keywords**: Cross-Lingual Retrieval, Relevance Transfer, Document Retrieval, Multilingual Representation, Zero-Shot Transfer

## TL;DR

This paper proposes a cross-lingual relevance transfer method that transfers relevance judgment capability to low-resource languages using a retrieval model trained on high-resource languages (e.g., English), significantly outperforming existing methods on multiple cross-lingual document retrieval benchmarks.

## Background & Motivation

**Background**: Cross-Lingual Information Retrieval (CLIR) aims to retrieve documents in one language using queries in another language. Currently, mainstream approaches include machine translation-based pipeline methods and end-to-end methods based on multilingual pre-trained models (e.g., mBERT, XLM-R).

**Limitations of Prior Work**: Translation-based methods are limited by translation quality, suffering from severe error cascade especially on low-resource language pairs. Although end-to-end multilingual retrieval models avoid translation bottlenecks, they often perform worse than monolingual models in fine-grained relevance judgment due to the uneven quality of semantic alignment across different languages in the multilingual representation space.

**Key Challenge**: High-resource languages have rich labeled retrieval data (e.g., MS MARCO), whereas low-resource languages lack high-quality relevance annotations. Directly training in the multilingual space leads to a major degradation in retrieval performance for low-resource languages. The core challenge is how to leverage supervision signals from high-resource languages to effectively enhance the retrieval capability of low-resource languages.

**Goal**: Design a cross-lingual relevance transfer framework that does not require labeled data in target languages, effectively transferring the relevance judgment capability of an English retrieval model to other languages.

**Key Insight**: The authors observe that while cross-lingual alignment at the token level is already mature in multilingual pre-trained models, a gap still exists in document-level relevance semantics. By explicitly aligning the relevance distribution of query-document pairs rather than purely aligning representations, the retrieval capability can be better transferred.

**Core Idea**: Replace traditional representation space alignment with relevance distribution alignment, transferring the relevance ranking knowledge of high-resource language retrieval models to cross-lingual retrieval models via knowledge distillation.

## Method

### Overall Architecture

The system adopts a teacher-student architecture: the teacher model is a high-performance monolingual retrieval model pre-trained on English MS MARCO, and the student model is a cross-lingual retriever based on a multilingual pre-trained model. The inputs are source language queries and target language documents, and the output is relevance scores. During training, cross-lingual training samples are constructed via translation-aligned parallel corpora, and the relevance distribution of the teacher model guides the learning of the student model.

### Key Designs

1. **Relevance Distribution Distillation**:

    - **Function**: Transfer the fine-grained relevance ranking knowledge of the teacher model to the student model.
    - **Mechanism**: Given a query $q$ and a set of candidate documents $\{d_1, ..., d_k\}$, the teacher model generates a relevance distribution $P_T = \text{softmax}(s_T / \tau)$ in the English space, and the student model generates a distribution $P_S$ in the cross-lingual space. The two are aligned via KL divergence $\mathcal{L}_{KD} = KL(P_T \| P_S)$. The temperature parameter $\tau$ controls the smoothness of the distribution, with higher temperatures preserving more ranking information.
    - **Design Motivation**: Compared to directly aligning representation vectors, aligning relevance distributions preserves the relative ranking relations among documents, which is more critical for retrieval tasks.

2. **Cross-Lingual Hard Negative Mining**:

    - **Function**: Construct high-quality cross-lingual training sample pairs.
    - **Mechanism**: Utilize a multilingual model for initial retrieval to mine hard negatives from the target language document collection that are semantically close to the query but actually irrelevant. Combine BM25 static negatives and dynamic negatives from the model to build training batches for contrastive learning. Each training sample contains 1 positive document and $N$ negative documents.
    - **Design Motivation**: Random negative sampling has limited utility for retrieval model training. Hard negatives force the model to learn finer semantic discrimination capabilities, which is especially important in cross-lingual scenarios.

3. **Progressive Language Expansion**:

    - **Function**: Scale retrieval capabilities from high-resource to low-resource languages in stages.
    - **Mechanism**: Training is divided into three stages: the first stage warms up on English monolingual data, the second stage incorporates high-resource language pairs (e.g., English-German, English-French), and the third stage progressively adds low-and-middle-resource languages. Each stage utilizes curriculum learning to first process target languages that are linguistically close to English before moving on to those further away.
    - **Design Motivation**: Directly training on all languages jointly leads to "language conflicts", where performance on high-resource languages deteriorates. Progressive expansion allows the model to adapt incrementally, reducing cross-lingual interference.

### Loss & Training

The total loss is a weighted combination of relevance distillation loss and contrastive learning loss: $\mathcal{L} = \lambda \mathcal{L}_{KD} + (1-\lambda) \mathcal{L}_{CL}$, where the contrastive learning loss $\mathcal{L}_{CL}$ takes the InfoNCE form. Training uses the AdamW optimizer with a linear learning rate decay after warmup and a batch size of 128.

## Key Experimental Results

### Main Results

| Dataset | Language Pair | Metric (nDCG@10) | Ours | mDPR | ColBERT-X | Gain |
|--------|--------|---------------|---------|------|-----------|------|
| CLEF 2003 | en→de | nDCG@10 | 52.3 | 44.1 | 47.6 | +4.7 |
| CLEF 2003 | en→fr | nDCG@10 | 55.8 | 46.3 | 50.2 | +5.6 |
| CLEF 2003 | en→it | nDCG@10 | 48.7 | 40.5 | 44.1 | +4.6 |
| XOR-TyDi | Multilingual | R@5kt | 47.2 | 38.6 | 42.8 | +4.4 |
| MIRACL | Multilingual | nDCG@10 | 51.6 | 42.3 | 46.9 | +4.7 |

### Ablation Study

| Configuration | nDCG@10 (CLEF avg) | Description |
|------|-------------------|------|
| Full model | 52.3 | Full model |
| w/o Relevance Distillation | 47.1 | Dropped by 5.2 without distillation, making it the most critical component |
| w/o Hard Negatives | 49.5 | Dropped by 2.8 with random negative sampling |
| w/o Progressive Training | 50.1 | Dropped by 2.2 with direct joint training |
| Translation pipeline only | 45.6 | Traditional translation methods have the largest gap |

### Key Findings
- Relevance distribution distillation is the most important component, contributing approximately half of the performance gain, demonstrating that transfer of ranking knowledge is more effective than representation alignment.
- The gain is more significant on language pairs with large linguistic distances (e.g., English-Arabic), indicating that the method is especially effective for low-resource scenarios.
- The temperature parameter $\tau$ performs best between 3 and 5; too high loses ranking information, while too low leads to unstable gradients.

## Highlights & Insights
- **Relevance distribution alignment outperforms representation alignment**: This is a valuable insight—in retrieval scenarios, the relative ranking of documents is more important than absolute vector positions. Retaining ranking signals using distribution distillation is a natural and effective design.
- **Progressive language expansion**: Leveraging the idea of curriculum learning to handle multilingual conflict issues, which can be transferred to other multilingual NLP tasks.
- The cross-lingual hard negative mining strategy can be reused in tasks such as cross-lingual question answering and cross-lingual fact verification.

## Limitations & Future Work
- It still relies on parallel corpora to construct training signals, and its effectiveness on true zero-resource languages (without parallel corpora) remains unknown.
- The quality ceiling of the teacher model determines the upper bound of the student model. If the English retrieval model itself is biased, the bias will be propagated.
- Cross-lingual capability transfer under the RAG paradigm combining retrieval and generation has not been explored.
- Exploring the use of LLMs to generate synthetic cross-lingual query-document pairs to reduce reliance on parallel corpora is a potential direction.

## Related Work & Insights
- **vs mDPR**: mDPR is trained directly on multilingual DPR and lacks an explicit relevance transfer mechanism, resulting in significantly poorer performance on low-resource languages.
- **vs ColBERT-X**: ColBERT-X improves retrieval precision with token-level late interaction, but its cross-lingual alignment is inferior to the distribution distillation method of this work.
- **vs Translate-Train**: The pipeline method of translating first and training later accumulates translation errors, whereas the end-to-end approach of this work is superior.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of relevance distribution distillation is novel, but the overall framework still falls under the KD paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple CLIR benchmarks and language pairs with sufficient ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed method description.
- Value: ⭐⭐⭐⭐ Practical application value for low-resource CLIR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)
- [\[ICLR 2026\] Improving Semantic Proximity in Information Retrieval through Cross-Lingual Alignment](../../ICLR2026/information_retrieval/improving_semantic_proximity_in_information_retrieval_through_cross-lingual_alig.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder](prism_political_bias_embeddings.md)

</div>

<!-- RELATED:END -->
