---
title: >-
  [Paper Note] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders
description: >-
  [ACL 2026][Medical Imaging][Medical text retrieval] This paper proposes CMedTEB (Chinese Medical Text Embedding Benchmark) and CARE (asymmetric retrieval framework). CMedTEB constructs a high-quality Chinese medical retrieval/reranking/STS benchmark via multi-LLM voting with expert validation, while CARE adopts an asymmetric architecture that encodes queries with a lightweight BERT and documents with a large LLM. Through a two-stage progressive alignment strategy, CARE achieves LLM-level retrieval accuracy at BERT-level online latency.
tags:
  - ACL 2026
  - Medical Imaging
  - Medical text retrieval
  - asymmetric encoders
  - Chinese medical benchmark
  - embedding models
  - RAG
date: 2026-05-08
content_hash: e7e83c337a9308be
---

# Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders

**Conference**: ACL 2026
**arXiv**: [2604.10937](https://arxiv.org/abs/2604.10937)
**Code**: [GitHub](https://github.com/PhilipGAQ/CARE)
**Area**: Medical Imaging
**Keywords**: Medical text retrieval, asymmetric encoders, Chinese medical benchmark, embedding models, RAG

## TL;DR
This paper proposes CMedTEB (Chinese Medical Text Embedding Benchmark) and CARE (asymmetric retrieval framework). CMedTEB constructs a high-quality Chinese medical retrieval/reranking/STS benchmark via multi-LLM voting with expert validation, while CARE adopts an asymmetric architecture that encodes queries with a lightweight BERT and documents with a large LLM. Through a two-stage progressive alignment strategy, CARE achieves LLM-level retrieval accuracy at BERT-level online latency.

## Background & Motivation

**Background**: Text embedding models are fundamental infrastructure in NLP, playing a particularly critical role in RAG systems. Recent LLM-based embedding models (e.g., Qwen3-Embedding, NV-Embed) have demonstrated strong performance on general benchmarks, yet Chinese medical text embedding remains underexplored.

**Limitations of Prior Work**: (1) **Poor benchmark quality**: Existing Chinese medical retrieval benchmarks (CmedqaRetrieval, MedicalRetrieval) suffer from severe false-negative issues—the "topic density" of the medical domain causes numerous semantically relevant but unannotated documents to be incorrectly labeled as irrelevant (averaging 9–19 false negatives per query). (2) **Accuracy-efficiency trade-off**: LLM-based embedding models achieve high accuracy but incur substantial latency, rendering them impractical for latency-sensitive scenarios such as real-time medical Q&A; BERT-style models offer low latency but insufficient accuracy.

**Key Challenge**: High accuracy demands large models, while real-time scenarios demand low latency—an apparently irreconcilable trade-off between accuracy and efficiency.

**Goal**: (1) Construct a high-quality Chinese medical embedding benchmark; (2) Design a retrieval framework that breaks the accuracy-latency trade-off.

**Key Insight**: In retrieval, query encoding is online (requiring low latency), whereas document encoding can be precomputed offline (permitting the use of large models). Exploiting this natural asymmetry, different-sized models are used to encode queries and documents separately.

**Core Idea**: A lightweight BERT encodes online queries while a large LLM encodes offline documents. A two-stage progressive alignment strategy—first freezing the document encoder to align the query encoder, then jointly fine-tuning both—bridges the semantic gap between heterogeneous encoders.

## Method

### Overall Architecture
CMedTEB benchmark: a multi-LLM consensus annotation pipeline for constructing retrieval, reranking, and STS tasks. CARE framework: initialize two encoders → Stage I: freeze the document encoder and align the query encoder via unsupervised self-contrastive learning → Stage II: unfreeze both encoders for joint fine-tuning. At inference, queries are processed by the BERT encoder (0.3B), while document embeddings are precomputed by the LLM.

### Key Designs

1. **CMedTEB Benchmark Construction (Multi-LLM Consensus + Expert Validation)**:

    - Function: Provides a high-fidelity evaluation standard for Chinese medical text embedding.
    - Mechanism: DeepSeek-V3, Doubao-1.5-Pro, and GPT-4o independently score query-document pairs on a 5-point scale; a pair is retained as a positive sample only when all three models unanimously agree. Experts independently re-annotate 5,000 pairs, achieving a 93.3% agreement rate. Fleiss' Kappa = 0.731 confirms annotation reliability.
    - Design Motivation: Single-LLM annotation (e.g., CMIRB using only ChatGPT) cannot guarantee quality; multi-model consensus combined with expert validation provides a more reliable gold standard.

2. **Two-Stage Asymmetric Alignment Strategy**:

    - Function: Bridges the semantic gap between the lightweight query encoder and the large document encoder.
    - Mechanism: **Stage I** (query encoder alignment): The document encoder is frozen, and a "self-contrastive" strategy aligns the query encoder—embeddings of the same text produced by the two encoders serve as mutual positives. Loss = Asym-InfoNCE (soft ranking alignment) + MSE (hard structural alignment). **Stage II** (joint fine-tuning): Both encoders are unfrozen, and Asym-InfoNCE on query-document pairs is used for end-to-end optimization of retrieval boundaries.
    - Design Motivation: Directly jointly training heterogeneous encoders leads to unstable convergence. The progressive strategy first establishes a spatial mapping foundation (Stage I using unlabeled data) and then optimizes task performance (Stage II using annotated data).

3. **Medical Domain Training Data Construction (Diversity-Aware Deduplication + False-Negative Cleaning)**:

    - Function: Addresses the false-negative problem caused by topic density in the medical domain.
    - Mechanism: A vector index is initialized with 5,000 seed samples; new candidates are discarded if their similarity to existing samples exceeds a threshold (ensuring diversity). GPT-4o then verifies the top-50 retrieved results to distinguish true hard negatives from false negatives. This process ultimately yields 500K high-quality triplets.
    - Design Motivation: Standard hard-negative mining fails in the medical domain—because a large number of semantically related documents remain unannotated, mined "negatives" are in fact positives.

## Key Experimental Results

### Main Results (CMedTEB Comprehensive Scores)

| Model | Params (Q/D) | Retrieval nDCG@10 | Rerank MAP@10 | STS Pearson | Avg |
|-------|-------------|-------------------|---------------|-------------|-----|
| bge-large-zh-v1.5 | 326M/326M | 50.32 | 67.55 | 78.95 | 73.04 |
| Conan-v1 | 326M/326M | 52.75 | 69.31 | 81.49 | 76.44 |
| gte-Qwen2-1.5B | 1.78B/1.78B | 55.39 | 72.35 | 85.50 | 77.61 |
| **CARE-0.3B-4B** | **305M/4.02B** | **55.91** | **72.84** | **88.53** | **78.13** |
| **CARE-0.3B-8B** | **305M/8.19B** | **56.75** | **73.67** | 87.07 | **78.94** |

### Ablation Study (Asymmetric vs. Symmetric vs. Other Efficient Methods)

| Method | Type | Retrieval | Rerank | Avg |
|--------|------|-----------|--------|-----|
| KALE | Asymmetric | 42.67 | 67.42 | 55.05 |
| ScalingNote | Asymmetric | 34.81 | 64.17 | 49.49 |
| **CARE-0.3B-4B** | **Asymmetric** | **55.91** | **72.84** | **64.38** |
| Med-Emb-8B (symmetric) | Symmetric | 56.42 | 74.84 | 65.63 |

### Key Findings
- **CARE breaks the accuracy-latency trade-off**: CARE-0.3B-8B trails the fully symmetric 8B model by only 0.6% in accuracy while reducing online inference parameter count by 27×.
- **CMedTEB is substantially more challenging than existing benchmarks**: General models average 85.15 on CMedQA but only 57.85 on the new CMedTEB tasks.
- **The two-stage training strategy substantially outperforms other asymmetric methods**: CARE exceeds KALE by 9.33 pp and ScalingNote by 14.89 pp.
- **Scaling the document encoder yields continuous performance gains without increasing online cost**: Expanding from 4B to 8B improves the average score by 0.81.
- **False-negative issues in existing benchmarks are severe**: 92% of LLM-annotated false negatives were confirmed by manual verification.

## Highlights & Insights
- **The asymmetric architecture exploits the natural asymmetry of the retrieval task**—the fact that queries are online and documents are offline is elegantly leveraged. This paradigm is transferable to any query-document matching scenario.
- **Self-contrastive alignment** (treating embeddings of the same text from the two encoders as mutual positives) is an elegant unsupervised solution that establishes cross-model spatial mappings without additional annotation.
- **The CMedTEB construction methodology** (multi-LLM consensus + expert validation + false-negative analysis) provides a reusable paradigm for domain-specific benchmark construction.

## Limitations & Future Work
- The document encoder requires offline precomputation, making the approach less suitable for scenarios with frequent document updates (e.g., real-time news retrieval).
- Stage I's MRL (Matryoshka Representation Learning) truncates high-dimensional LLM embeddings to 768 dimensions, which may incur information loss.
- CMedTEB covers Chinese only; cross-lingual medical retrieval is not addressed.
- Validation is conducted solely in the medical domain; generalizability to other specialized domains such as law and finance remains to be confirmed.
- Online distillation or progressive knowledge transfer could be explored to further close the gap of the query encoder.

## Related Work & Insights
- **vs. KALE/ScalingNote**: These methods also pursue asymmetric retrieval but employ simpler alignment strategies (layer pruning or direct training); the proposed two-stage progressive alignment is markedly more effective.
- **vs. symmetric LLM embeddings**: Models such as Qwen3-Embedding lead in accuracy but incur 10×+ latency; CARE nearly matches their accuracy while maintaining BERT-level latency.
- **vs. CMIRB benchmark**: CMIRB relies on single-LLM annotation and covers only retrieval; CMedTEB offers broader coverage with multi-LLM consensus and three task types.

## Rating
- Novelty: ⭐⭐⭐⭐ The asymmetric architecture is not new, but the two-stage self-contrastive alignment strategy is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of benchmark construction, model evaluation, ablation studies, and efficiency analysis, with expert validation for benchmark quality.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; figures and tables effectively convey core information.
- Value: ⭐⭐⭐⭐⭐ Full open-sourcing of benchmark, model, code, and data provides a direct contribution to Chinese medical NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)

</div>

<!-- RELATED:END -->
