---
title: >-
  [Paper Note] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA
description: >-
  [Medical LLM] MedBioRAG proposes a retrieval-augmented generation framework that combines semantic search, document retrieval, and fine-tuned LLMs, comprehensively outperforming GPT-4o baselines and prior SOTAs on three types of biomedical QA tasks: text retrieval, closed-book QA, and long-text QA.
tags:
  - "Medical LLM"
date: 2026-05-08
content_hash: 2d41539f4fe4225a
---

# MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA

| Project | Content |
|------|------|
| Author | Seonok Kim |
| Institution | Mazelone, Seoul |
| Conference | ACL 2025 |
| arXiv | 2512.10996 |
| Topic | Retrieval-Augmented Generation for Biomedical QA |

## TL;DR

MedBioRAG proposes a retrieval-augmented generation framework that combines semantic search, document retrieval, and fine-tuned LLMs, comprehensively outperforming GPT-4o baselines and prior SOTAs on three types of biomedical QA tasks: text retrieval, closed-book QA, and long-text QA.

## Background & Motivation

- **Domain Challenges**: Biomedical QA demands extremely high factual accuracy. General-domain LLMs (such as GPT-4o) rely on static pre-training data and are prone to hallucinations and outdated information.
- **Limitations of Prior Work**: Traditional keyword retrieval (BM25, TF-IDF) cannot handle synonyms (e.g., "heart attack" vs "myocardial infarction") and polysemy in medical terminology, leading to incomplete retrieval.
- **RAG Bottlenecks**: Although retrieval-augmented generation can dynamically introduce external knowledge, its effectiveness is highly dependent on retrieval quality, document ranking, and the degree of model fine-tuning.
- **Core Motivation**: To design an end-to-end biomedical QA framework that integrates semantic search (for improved retrieval accuracy) and fine-tuned LLMs (for improved generation quality).

## Method

### Overall Architecture

MedBioRAG consists of three core stages:

1. **Hybrid Retrieval Module**: Utilizes both lexical and semantic search, with semantic search playing the leading role.
2. **LLM-Based Answer Generation**: The fine-tuned LLM integrates the retrieved information into coherent answers.
3. **Prompt Engineering and Content Filtering**: Optimizes prompt structures to guide the model to generate factually accurate outputs.

### Retrieval Mechanism

**Lexical Search**: A classic term-frequency ranking method based on BM25, computing the matching score between documents and queries via IDF and TF.

**Semantic Search**: Map query $Q$ and document $D$ to dense vector representations through encoder $\phi$, computing semantic relevance using cosine similarity:

$$\text{Sim}(Q, D_i) = \frac{v_Q \cdot v_{D_i}}{\|v_Q\| \|v_{D_i}\|}$$

The retrieval system ranks according to similarity scores and selects the Top-K documents. The core advantage of semantic search lies in its ability to retrieve semantically relevant documents even without exact keyword matches.

### Fine-Tuning and Generation

- **Supervised Fine-Tuning**: Trained using (x, y) pairs, where x represents the query + retrieved document context, and y represents the expected answer, optimizing the standard language model loss.
- **Confidence Filtering**: The model electrics a confidence score to the generated response, dropping or iteratively correcting responses below the threshold.
- **Prompt Engineering**: System prompts are tailored for closed-book QA (requiring only option letters), long-text QA (generating structured answers), and short-text QA (concise answers), using different max tokens, temperature, and top-p parameters.

## Experiments

### Evaluation Settings

- **Retrieval Evaluation**: NFCorpus, TREC-COVID, with metrics such as NDCG@10, MRR@10, Precision@10, etc.
- **Closed-Book QA**: MedQA, PubMedQA, BioASQ, evaluated by accuracy.
- **Long-Text QA**: LiveQA, MedicationQA, PubMedQA, BioASQ, with metrics including ROUGE, BLEU, BERTScore, and BLEURT.

### Table 1: Closed-Book QA Performance Comparison

| Method | MedQA | PubMedQA | BioASQ |
|------|-------|----------|--------|
| GPT-3.5 + MedBioRAG | 45.36 | 38.60 | 66.91 |
| GPT-4 + MedBioRAG | 78.79 | 72.81 | 97.79 |
| GPT-4o | 81.82 | 44.74 | 96.12 |
| GPT-4o + MedBioRAG | 86.86 | 66.67 | 97.06 |
| GPT-4o-mini + MedBioRAG | 70.71 | 76.32 | 97.06 |
| Fine-Tuned GPT-4o | 87.88 | 80.70 | 97.06 |
| **Fine-Tuned GPT-4o + MedBioRAG** | **89.47** | **85.00** | **98.32** |

**Key Point**: Fine-tuned GPT-4o + MedBioRAG achieves the best performance across all datasets, improving PubMedQA from the GPT-4o baseline of 44.74% to 85.00%, an increase of over 40 percentage points.

### Table 2: Retrieval Performance Comparison (Semantic vs Lexical Search)

| Metric | NFCorpus Lexical | NFCorpus Semantic | TREC-COVID Lexical | TREC-COVID Semantic |
|------|-------------|-------------|----------------|----------------|
| NDCG@10 | 31.34 | **37.91** | 48.35 | **61.02** |
| MRR@10 | 51.63 | **64.29** | 82.50 | **89.17** |
| Precision@10 | 23.04 | **27.88** | 49.60 | **64.20** |
| MAP@10 | 46.01 | **56.15** | 72.31 | **82.19** |

**Key Point**: Semantic search outperforms lexical search across all metrics, with NDCG@10 improving by approximately 6.6 points on NFCorpus and 12.7 points on TREC-COVID.

## Highlights & Insights

1. **Systematic and Comprehensive Evaluation**: Covers three types of tasks (retrieval, closed-book QA, long-text QA) using up to 7 evaluation metrics, demonstrating a very thorough experimental design.
2. **Remarkable Advantage of Semantic Search**: Clearly demonstrates the comprehensive advantages of semantic search over lexical search in the biomedical domain using experimental data.
3. **Synergy of Fine-Tuning + RAG**: Proves that neither pure fine-tuning nor pure RAG is as effective as their combination, providing a clear technical solution for biomedical AI applications.
4. **Trade-off Analysis of Top-K Retrieval**: Reveals that retrieving more documents is not always better; excessive documents introduce noise and conflicting information, which degrades performance.

## Limitations & Future Work

1. **Lack of Medical Specialist Validation**: Model outputs have not been reviewed by clinicians, making it impossible to confirm alignment with expert reasoning.
2. **Insufficient Handling of Retrieval Contradictions**: The model lacks an efficient conflict-resolution mechanism when retrieved documents contain factual contradictions.
3. **High Computational Overhead**: Real-time retrieval increases inference latency, limiting its application in time-sensitive scenarios (e.g., emergency decision-making).
4. **Limited Domain Generalization**: Performance in specific clinical sub-domains (e.g., clinical diagnosis, electronic health records) remains to be verified.
5. **Limited Baseline Models**: Primarily based on the GPT series, lacking in-depth comparison with open-source biomedical models (such as MEDITRON-70B, BioMistral).

## Related Work & Insights

- **Biomedical LLMs**: Med-PaLM 2, BioGPT, MEDITRON-70B, BiomedGPT, etc., enhance biomedical reasoning capabilities through domain-specific fine-tuning.
- **RAG Frameworks**: Hybrid retrieval strategies like BlendedRAG combine lexical and semantic search; LLM4IR explores the application of LLMs in information retrieval.
- **Embedding Models**: Semantic retrieval methods based on pre-trained embeddings such as SGPT provide the foundation for biomedical semantic search.
- **Domain Fine-Tuning**: Prompt optimization methods such as Medprompt and supervised fine-tuning strategies enhance the domain adaptation capabilities of LLMs.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 2 | The combination of semantic search + RAG + fine-tuning is relatively conventional and does not introduce novel technical contributions. |
| Experimental Thoroughness | 4 | Covers three types of tasks, multiple benchmark datasets, and various metrics, with a comprehensive design. |
| Writing Quality | 3 | Structured clearly but the descriptions are somewhat verbose, with some redundant content. |
| Value | 3 | Provides a referenceable RAG framework for biomedical QA, but relies heavily on closed-source models. |
| Overall Score | 3.0 | An engineering integration work with a solid experimental design but lacking in methodological novelty. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_nlp/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[NeurIPS 2025\] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](../../NeurIPS2025/medical_nlp/llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_nlp/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)

</div>

<!-- RELATED:END -->
