---
title: >-
  [Paper Note] MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RAG] This paper proposes MITRA, a locally deployed RAG system for large physics experiment collaborations (e.g., CERN CMS), featuring a two-tier vector database architecture (abstract store + full-text store) and a fully on-premise deployment strategy. MITRA substantially outperforms traditional keyword-based search (BM25) on semantic retrieval tasks, improving Precision@1 from 0.13 to 0.75.
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Scientific Collaboration"
  - "Local Deployment"
  - "Knowledge Retrieval"
  - "Physics Experiments"
date: 2026-05-08
content_hash: ae1595e39a5e9ffb
---

# MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations

**Conference**: NeurIPS 2025
**arXiv**: [2603.09800](https://arxiv.org/abs/2603.09800)  
**Code**: To be confirmed  
**Area**: Information Retrieval
**Keywords**: RAG, Scientific Collaboration, Local Deployment, Knowledge Retrieval, Physics Experiments

## TL;DR

This paper proposes MITRA, a locally deployed RAG system for large physics experiment collaborations (e.g., CERN CMS), featuring a two-tier vector database architecture (abstract store + full-text store) and a fully on-premise deployment strategy. MITRA substantially outperforms traditional keyword-based search (BM25) on semantic retrieval tasks, improving Precision@1 from 0.13 to 0.75.

## Background & Motivation

Large scientific collaborations (e.g., the CMS experiment at CERN) involve thousands of members and generate massive volumes of internal documents—analysis notes, internal wikis, operational guides, and more. For newly joining PhD students or experts seeking to quickly understand a specific measurement, retrieving precise information from such documents is a time-consuming and frustrating task.

Existing problems:

**Limitations of traditional keyword search**: Unable to understand the semantic context of queries; heavily reliant on exact phrase matching.

**Cross-analysis confusion**: The same question (e.g., "What is the most important background?") can have entirely different answers in a Higgs → di-muon analysis versus a dark matter search.

**Data privacy requirements**: Internal analysis results and unpublished data of a collaboration cannot be transmitted to external servers.

The ATLAS collaboration is concurrently developing a similar system (chATLAS), but it relies on an external API (GPT-4o mini). MITRA offers a fully local, privacy-first alternative.

## Method

### Overall Architecture

MITRA's workflow is divided into **offline database creation** and **online inference**:

1. **Document retrieval**: Uses Selenium to automatically log in to internal databases and download analysis notes (PDF format).
2. **Text extraction**: Uses OCR engines (Surya/Tesseract) for high-fidelity text extraction, preserving main content, figure captions, tables, and other structural elements.
3. **Embedding and storage**: Documents are chunked by paragraph, encoded into 768-dimensional vectors using a DPR model (facebook/dpr-question_encoder-multiset-base), and stored in Chroma DB.
4. **Retrieval and re-ranking**: Cosine similarity is used for initial top-k retrieval, followed by precise re-ranking using a cross-encoder (ms-marco-MiniLM-L-6-v2).
5. **Generation**: A 4-bit quantized Mistral-7B model is served locally via Ollama and integrated through LangChain.

### Key Designs

**Two-Tier Vector Database Architecture**

This is the most central design in MITRA, addressing cross-analysis context confusion:

- **Tier 1 (Abstract Store)**: Contains only the abstracts of all analysis documents. Upon the user's initial query, the system searches the abstract store to identify the most relevant analysis. The user is prompted to confirm the selection (a human-in-the-loop validation step).
- **Tier 2 (Full-Text Store)**: Once confirmed, the system "locks in" to the dedicated full-text database for that analysis. All subsequent RAG operations are confined to this single analysis database. Users may start a new conversation at any time to switch analyses.

This design effectively isolates context and prevents the model from confusing information across different analyses.

**Fully Local Deployment**

All components (embedding model, LLM) are deployed on a local GPU server (NVIDIA Tesla T4, 15 GB VRAM):
- Avoids the cumulative cost of API services for large collaborations.
- Ensures sensitive data never leaves the secure network.
- The RAG architecture allows the knowledge base to be refreshed by re-embedding updated documents, without retraining the model.

**Hallucination-Resistant Design**: The LLM is explicitly prompted to answer only based on the retrieved context. In qualitative tests, when locked to a dark matter analysis and asked "How many Higgs bosons were found?", the system correctly declined to answer and indicated that the current context is a dark matter search analysis.

### Loss & Training

This paper focuses on system design rather than model training. Both DPR and the cross-encoder use pre-trained models, and Mistral-7B is applied with 4-bit quantization. OCR-based text extraction serves as the critical data quality assurance step.

## Key Experimental Results

### Main Results

Evaluation was conducted using two query sets designed by domain experts:

**Set 1 (Exact keyword queries) — using verbatim phrasing from documents:**

| System | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 |
|--------|-----|-----|-----|-----|-----|-----|
| BM25 | 1.00 | 0.85 | 0.40 | 0.90 | **0.32** | **1.00** |
| MITRA | 1.00 | 0.85 | 0.40 | 0.90 | 0.24 | 0.90 |

**Set 2 (Semantic queries) — using synonyms/paraphrased phrasing:**

| System | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 |
|--------|-----|-----|-----|-----|-----|-----|
| BM25 | 0.13 | 0.03 | 0.25 | 0.56 | 0.18 | 0.59 |
| **MITRA** | **0.75** | **0.66** | **0.33** | **0.81** | **0.20** | **0.81** |

**Ranking quality evaluation (MRR and NDCG):**

| Query Set | System | MRR | NDCG@3 | NDCG@5 |
|-----------|--------|-----|--------|--------|
| Set 1 | BM25 | 1.00 | 1.00 | 0.98 |
| Set 1 | MITRA | 1.00 | 1.00 | 1.00 |
| Set 2 | BM25 | 0.35 | 0.67 | 0.59 |
| Set 2 | **MITRA** | **0.81** | **0.91** | **0.88** |

### Ablation Study

No formal ablation experiments are included; however, the comparison between Set 1 and Set 2 indirectly validates the necessity of semantic retrieval:
- Under exact keyword conditions, BM25 and MITRA perform comparably.
- Under semantic paraphrasing conditions, BM25 degrades sharply while MITRA remains robust.

### Key Findings

1. **Semantic understanding is the critical differentiator**: On realistic paraphrased queries, MITRA's P@1 is 5.8× that of BM25 (0.75 vs. 0.13), and its MRR is 2.3× higher (0.81 vs. 0.35).
2. **Superior ranking quality**: High MRR indicates that the correct answer is nearly always ranked first; high NDCG confirms strong overall ranking quality, which is crucial for reducing RAG hallucinations.
3. **Effective context isolation**: The system correctly handles out-of-domain queries and avoids cross-analysis confusion.
4. Two-stage retrieval (DPR for initial retrieval + cross-encoder for re-ranking) is necessary, as cross-encoder search over the entire database is too slow.

## Highlights & Insights

1. **Engineering elegance of the two-tier database**: The two-level structure of abstract store → full-text store is a concise yet effective solution, generalizable to any RAG scenario requiring document-level context isolation.
2. **Privacy-first design philosophy**: Although fully local deployment may underperform GPT-4o on individual query quality, it offers greater long-term cost sustainability and zero data leakage risk for collaborations with thousands of members.
3. **Human-in-the-loop validation**: Requiring users to confirm the analysis selection is a pragmatic design decision that prevents the system from silently locking into an incorrect context.
4. **OCR over naive PDF extraction**: For scientific documents with complex layouts (figures, equations, tables), this choice substantially improves knowledge base quality.

## Limitations & Future Work

1. **Limited evaluation scale**: Testing was conducted on a small number of queries, with no large-scale user study or quantitative evaluation of generation quality.
2. **Single document type**: The current system only processes analysis note PDFs and does not cover other formats such as wikis or slides.
3. **No multi-turn dialogue support**: The current system supports single-turn question answering and lacks conversational capability with accumulated context.
4. **Generation quality not quantified**: While retrieval evaluation is thorough, the faithfulness and quality of LLM-generated responses lack systematic assessment.
5. The model scale (7B parameters) is relatively modest, which may limit complex reasoning capability.
6. The document update frequency and the specific workflow for refreshing the knowledge base are not discussed.

## Related Work & Insights

- **chATLAS** (Dal Santo et al., 2025): The ATLAS collaboration's system, which uses the OpenAI API. MITRA provides a localized alternative.
- Traditional **BM25** retrieval remains competitive in exact keyword scenarios but is wholly inadequate for semantic retrieval.
- The **DPR + Cross-encoder** two-stage retrieval pipeline is the standard configuration for RAG systems.
- The two-tier database approach proposed in this paper can be applied to domain-specific knowledge retrieval systems in other fields (e.g., legal, medical).
- Future direction: evolving from a question-answering tool into an active research agent capable of summarizing updates, comparing methodologies, and identifying gaps in the search space.

## Rating

- Novelty: ⭐⭐⭐⭐ All technical components are established solutions in combination (DPR + Cross-encoder + Mistral-7B); the two-tier database represents a meaningful engineering contribution but offers limited technical novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Retrieval evaluation is complete and comparisons are clear; however, the number of queries is small, and generation quality evaluation and user studies are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ System description is clear, motivation is well-articulated, and problem framing is precise.
- Value: ⭐⭐⭐⭐ Offers direct practical value for large scientific collaborations, though the technical contribution and evaluation depth are somewhat insufficient by NeurIPS standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](../../ICML2026/information_retrieval/reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](../../ACL2025/information_retrieval/personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
