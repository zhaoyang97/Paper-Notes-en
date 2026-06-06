---
title: >-
  [Paper Note] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders
description: >-
  [ACL 2026][Information Retrieval & RAG][Medical text retrieval] Ours proposes CMedTEB (Chinese Medical Text Embedding Benchmark) and CARE (an asymmetric retrieval framework). The former builds a high-quality Chinese medi…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Medical text retrieval"
  - "asymmetric encoders"
  - "Chinese medical benchmark"
  - "embedding models"
  - "RAG"
date: 2026-05-08
content_hash: 3993b46f3b0ed9b5
---

# Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders

**Conference**: ACL 2026  
**arXiv**: [2604.10937](https://arxiv.org/abs/2604.10937)  
**Code**: [GitHub](https://github.com/PhilipGAQ/CARE)  
**Area**: Medical Imaging  
**Keywords**: Medical text retrieval, asymmetric encoders, Chinese medical benchmark, embedding models, RAG

## TL;DR
Ours proposes CMedTEB (Chinese Medical Text Embedding Benchmark) and CARE (an asymmetric retrieval framework). The former builds a high-quality Chinese medical retrieval/reranking/STS benchmark via multi-LLM voting and expert verification. The latter employs an asymmetric architecture with a lightweight BERT for query encoding and a large LLM for document encoding, achieving LLM-level retrieval accuracy with BERT-level online latency through a two-stage progressive alignment strategy.

## Background & Motivation

**Background**: Text embedding models are fundamental NLP infrastructure, particularly critical in RAG systems. Recently, LLM-based embedding models (e.g., Qwen3-Embedding, NV-Embed) have excelled on general benchmarks, but Chinese medical text embedding remains under-addressed.

**Limitations of Prior Work**: (1) **Poor benchmark quality**: Existing Chinese medical retrieval benchmarks (CmedqaRetrieval, MedicalRetrieval) suffer from severe false negative issues—the "topical density" of the medical field leads to many semantically relevant but unlabeled documents being mislabeled as irrelevant (averaging 9-19 false negatives per query). (2) **Accuracy-efficiency trade-off**: LLM-based models offer high precision but high latency, making them unsuitable for latency-sensitive scenarios like real-time medical QA; BERT-style models offer low latency but insufficient accuracy.

**Key Challenge**: High accuracy requires large models, while real-time scenarios require low latency—an seemingly irreconcilable trade-off between precision and efficiency.

**Goal**: (1) Construct a high-quality Chinese medical embedding benchmark; (2) Design a retrieval framework that breaks the accuracy-latency trade-off.

**Key Insight**: In retrieval, query encoding occurs online (requiring low latency), whereas document encoding can be pre-computed offline (suitable for large models). This natural asymmetry can be exploited by using different-sized models to encode queries and documents respectively.

**Core Idea**: Combine a lightweight BERT for online queries with an LLM for offline documents. A two-stage progressive alignment (aligning the query encoder by freezing the document encoder first, followed by joint fine-tuning) is used to bridge the semantic gap between heterogeneous encoders.

## Method

### Overall Architecture
CMedTEB Benchmark: A multi-LLM consensus annotation pipeline constructs Retrieval, Reranking, and STS tasks. CARE Framework: Initialize two encoders $\rightarrow$ Stage I: Freeze the document encoder and align the query encoder using unsupervised self-contrastive learning $\rightarrow$ Stage II: Unfreeze both encoders for joint fine-tuning. During inference, queries use the BERT model (0.3B) while documents use pre-computed LLM embeddings.

### Key Designs

1.  **CMedTEB Benchmark Construction (Multi-LLM Consensus + Expert Verification)**:
    *   **Function**: Provides a high-fidelity evaluation standard for Chinese medical embeddings.
    *   **Mechanism**: Uses three LLMs (DeepSeek-V3, Doubao-1.5-Pro, GPT-4o) to score query-document pairs on a 5-point scale, retaining positive samples only when all three agree. Experts independently re-annotated 5,000 pairs with a 93.3% agreement rate. A Fleiss' Kappa of 0.731 indicates high annotation reliability.
    *   **Design Motivation**: Single-LLM annotation (e.g., CMIRB using only ChatGPT) cannot guarantee quality; multi-model consensus combined with expert verification provides a more reliable ground truth.

2.  **Two-Stage Asymmetric Alignment Strategy**:
    *   **Function**: Bridges the semantic gap between the lightweight query encoder and the large-scale document encoder.
    *   **Mechanism**: **Stage I** (Query Encoder Alignment): Freeze the document encoder and align using a "self-contrastive" strategy—the same text's embeddings across both encoders act as mutual positive samples. Loss = Asym-InfoNCE (soft rank alignment) + MSE (hard structural alignment). **Stage II** (Joint Fine-tuning): Unfreeze both for end-to-end optimization of retrieval boundaries using query-document pairs.
    *   **Design Motivation**: Direct joint training of heterogeneous encoders leads to unstable convergence. A progressive strategy first establishes a spatial mapping foundation (Stage I using unlabeled data) before optimizing task performance (Stage II using labeled data).

3.  **Medical Domain Training Data Construction (Diversity-Aware De-duplication + False Negative Cleaning)**:
    *   **Function**: Resolves the false negative problem caused by "topical density" in medical fields.
    *   **Mechanism**: Initialize a vector index with 5,000 seeds; discard new candidates if their similarity to existing samples is too high (ensuring diversity). Then use GPT-4o to verify top-50 retrieval results to distinguish true hard negatives from false negatives. This generates 500K high-quality triplets.
    *   **Design Motivation**: Standard hard negative mining fails in medical domains because many semantically relevant documents are unlabeled, causing mined "negatives" to actually be positives.

## Key Experimental Results

### Main Results (CMedTEB Composite Scores)

| Model | Params (Q/D) | Retrieval nDCG@10 | Rerank MAP@10 | STS Pearson | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bge-large-zh-v1.5 | 326M/326M | 50.32 | 67.55 | 78.95 | 73.04 |
| Conan-v1 | 326M/326M | 52.75 | 69.31 | 81.49 | 76.44 |
| gte-Qwen2-1.5B | 1.78B/1.78B | 55.39 | 72.35 | 85.50 | 77.61 |
| **CARE-0.3B-4B** | **305M/4.02B** | **55.91** | **72.84** | **88.53** | **78.13** |
| **CARE-0.3B-8B** | **305M/8.19B** | **56.75** | **73.67** | **87.07** | **78.94** |

### Ablation Study (Asymmetric vs. Symmetric vs. Other Efficient Methods)

| Method | Type | Retrieval | Rerank | Avg |
| :--- | :--- | :--- | :--- | :--- |
| KALE | Asymmetric | 42.67 | 67.42 | 55.05 |
| ScalingNote | Asymmetric | 34.81 | 64.17 | 49.49 |
| **CARE-0.3B-4B** | **Asymmetric** | **55.91** | **72.84** | **64.38** |
| Med-Emb-8B (Symm) | Symmetric | 56.42 | 74.84 | 65.63 |

### Key Findings
*   **CARE breaks the accuracy-latency trade-off**: CARE-0.3B-8B trails the fully symmetric 8B model by only 0.6% in accuracy while using 27x fewer parameters for online inference.
*   **CMedTEB is significantly more difficult than existing benchmarks**: General models average 85.15 on CMedQA but only 57.85 on new CMedTEB tasks.
*   **Two-stage training significantly outperforms other asymmetric methods**: CARE exceeds KALE by 9.33pp and ScalingNote by 14.89pp.
*   **Performance scales with document encoder size without increasing online costs**: The average score improved by 0.81 when moving from 4B to 8B.
*   **False negative issues in existing benchmarks are severe**: LLM-annotated false negatives were confirmed by human verification at a rate of 92%.

## Highlights & Insights
*   **The exploitation of natural asymmetry in retrieval tasks** is the core insight—leveraging the fact that queries are online and documents are offline. This logic is transferable to any query-document matching scenario.
*   **Self-contrastive alignment** (where representations of the same text across two encoders are mutual positives) serves as an elegant unsupervised solution to establish cross-model spatial mapping without extra labels.
*   **CMedTEB construction methodology** (multi-LLM consensus + expert verification + false negative analysis) provides a reusable paradigm for building domain-specific benchmarks.

## Limitations & Future Work
*   Document encoders require offline pre-computation, making the approach less suitable for scenarios with frequent document updates (e.g., real-time news retrieval).
*   MRL (Matryoshka Representation Learning) in Stage I truncates high-dimensional LLM embeddings to 768 dimensions, potentially losing information.
*   CMedTEB only covers Chinese; cross-lingual medical retrieval remains unexplored.
*   Validation is limited to the medical domain; generalizability to other professional fields like law or finance remains to be confirmed.
*   Future work could explore online distillation or progressive knowledge transfer to further downsize the query encoder.

## Related Work & Insights
*   **vs KALE/ScalingNote**: These methods also utilize asymmetric retrieval but use simpler alignment strategies (layer pruning or direct training); the two-stage progressive alignment in this work is significantly more effective.
*   **vs Symmetric LLM Embeddings**: Models like Qwen3-Embedding lead in accuracy but suffer from 10x+ higher latency; CARE nearly matches their accuracy while maintaining BERT-level latency.
*   **vs CMIRB Benchmark**: CMIRB uses single-LLM annotation and focuses only on retrieval, whereas CMedTEB is more comprehensive with multi-LLM consensus and triple-task coverage.

## Rating
*   Novelty: ⭐⭐⭐⭐ Asymmetric architectures are not new, but the two-stage self-contrastive alignment strategy is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis involving benchmarks, models, ablations, and efficiency, with expert-validated benchmark construction.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure with figures and tables effectively conveying core information.
*   Value: ⭐⭐⭐⭐⭐ Full open-source release of benchmarks, models, code, and data provides a direct boost to Chinese medical NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](../../ICLR2026/information_retrieval/efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[AAAI 2026\] ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](../../AAAI2026/information_retrieval/comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)

</div>

<!-- RELATED:END -->
