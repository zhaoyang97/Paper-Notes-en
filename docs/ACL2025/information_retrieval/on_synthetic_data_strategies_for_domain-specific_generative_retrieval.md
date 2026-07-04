---
title: >-
  [Paper Note] On Synthetic Data Strategies for Domain-Specific Generative Retrieval
description: >-
  [ACL2025][Information Retrieval & RAG][Generative Retrieval] This paper systematically investigates synthetic data strategies for training generative retrieval models on domain-specific corpora, proposing multi-granular query generation, constraint-based queries, and preference learning based on hard negatives, which significantly improves retrieval performance.
tags:
  - "ACL2025"
  - "Information Retrieval & RAG"
  - "Generative Retrieval"
  - "Synthetic Data"
  - "Preference Learning"
  - "Domain Adaptation"
  - "Document Identifiers"
date: 2026-05-08
content_hash: 155d864fe74527b6
---

# On Synthetic Data Strategies for Domain-Specific Generative Retrieval

**Conference**: ACL2025  
**arXiv**: [2502.17957](https://arxiv.org/abs/2502.17957)  
**Code**: Not publicly available  
**Area**: Information Retrieval  
**Keywords**: Generative Retrieval, Synthetic Data, Preference Learning, Domain Adaptation, Document Identifiers  

## TL;DR

This paper systematically investigates synthetic data strategies for training generative retrieval models on domain-specific corpora, proposing multi-granular query generation, constraint-based queries, and preference learning based on hard negatives, which significantly improves retrieval performance.

## Background & Motivation

Generative Retrieval (GR) is an emerging paradigm in the field of information retrieval that utilizes generative models to directly generate document identifiers relevant to a user query, rather than relying on external indices like traditional dense retrieval methods. Although existing work has made progress in training strategies, modeling techniques, and inference methods, the role of **data strategies**—especially on domain-specific corpora—remains severely overlooked.

Generative retrieval models must "internalize" the entire corpus into their parametric memory; thus, the selection and quality of training data are crucial. Existing works mostly follow the DSI-QG paradigm, using docT5query to generate synthetic queries from passages, but this "one-size-fits-all" data strategy is often insufficient when transferring to new domains.

Unlike dense retrieval, generative retrieval models need to possess three core abilities simultaneously:

**Memorization capability**: Storing the corpus content and mapping it to document identifiers.

**Generalization capability**: Inferring beyond the explicit textual cues of user queries.

**Relevance scoring**: Accurately ranking the relevance of document identifiers.

Domain-specific corpora amplify these challenges, as models must adapt to domain nuances while maintaining robust generalization and ranking capabilities.

## Method

### Overall Architecture

This paper introduces a **two-stage training framework** (see Figure 1):

- **First Stage (Supervised Fine-Tuning)**: Learning to map inputs to document identifiers, focusing on enhancing memorization and generalization capabilities.
- **Second Stage (Preference Learning)**: Further enhancing document ranking performance via ranking optimization.

### Document Identifier Design

Semantic identifiers are primarily used, employing a keyword-based approach: an LLM is used to generate a list of keywords describing the content of each document as its identifier. In addition, the generalization capability is validated by extending this to **atomic identifiers** (i.e., unique tokens decoded in a single step).

### First Stage: Supervised Fine-Tuning Data Strategy

Synthetic data consists of two parts:

#### Context2ID (Context-to-Identifier)
Pairs each text chunk from the corpus with its corresponding document identifier to help the model "memorize" document content. The training objective not only optimizes the output sequence (document identifier) but also includes learning to decode the input content, with the full loss function defined as:

$$\mathcal{L}_{\text{sft}}(q,d) = -\sum_i \log P(q_i | q_{<i}; \theta) - \sum_i \log P(d'_i | d'_{<i}, q; \theta)$$

#### Query2ID (Query-to-Identifier)

Utilizes LLMs (rather than docT5query) for synthetic query generation, specifically including three strategies:

**1. Multi-Granular Query Generation**
- **Chunk-level queries**: Takes the entire text chunk as input to generate $m_c$ queries capturing high-level semantics.
- **Sentence-level queries**: Takes a single sentence as input to generate $m_s$ queries focusing on local details.

**2. Constraint-Based Query Generation**
Leveraging the instruction-following capabilities of LLMs, domain-specific metadata constraints (such as author name, political leaning, etc.) are incorporated during query generation to generate more professional domain queries. For each document, $m_i$ constrained queries are generated.

**3. Combination Strategy of Context2ID and Query2ID**
Employs an interleaving approach rather than simple concatenation, upsampling the smaller Context2ID dataset.

### Second Stage: Preference Learning Data Strategy

#### Preference Optimization Objective
**RPO (Regularized Preference Optimization)** is adopted as the ranking optimization method. RPO is an extension of DPO that adds a supervised fine-tuning loss to mitigate the over-optimization problem on negative samples.

#### Synthetic Query Strategy Adjustment
- Requests LLMs to generate queries that are **as difficult as possible**.
- Simultaneously requests the **corresponding answer** to the query to ensure that difficult queries remain answerable.
- Differentiates these queries from the first-stage queries to prevent the model from over-fitting to the same batch of data.

#### Negative Sample Candidate Selection
The key innovation lies in selecting negative samples from the model's own retrieval results:
- Use the first-stage model to retrieve the synthetic queries generated for preference learning.
- Select top-k negative sample candidates that rank higher than the positive sample.
- Skip the query if the positive sample is already ranked first.
- Each negative sample is paired with the positive sample to form a training instance.

## Experiments

### Experimental Setup
- **Datasets**: MultiHop-RAG, AllSides, AGNews (three domain-specific) + Natural Questions (general)
- **Base Models**: Mistral 7B series
- **Query Generation**: Mixtral 8x7B
- **Keyword Generation**: Claude 3 Sonnet

### Main Results

#### Effectiveness of Multi-Granular Queries (MultiHop-RAG)

| Method | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|------|-------|--------|--------|--------|
| Chunk only | 43.64 | 66.65 | 13.98 | 31.14 |
| +Sent | **61.64** | **81.69** | **22.13** | **47.20** |

Sentence-level queries bring an improvement of approximately 18 percentage points in HIT@4.

#### Effectiveness of Constraint-Based Queries

In MultiHop-RAG, HIT@4 increased from 61.64 to 69.98, and in AllSides, HIT@1 increased from 10.19 to 14.20.

#### Effectiveness of Context2ID Data

Removing Context2ID caused HIT@4 on MultiHop-RAG to plunge from 69.98 to 41.33; the interleaved combination method (69.98) significantly outperformed simple concatenation (44.30).

#### Preference Learning Stage

| Strategy | HIT@4 | HIT@10 | MRR@10 |
|------|-------|--------|--------|
| SFT only | 69.98 | 88.34 | 52.29 |
| Random 5 negatives | 58.94 | 82.88 | 43.53 |
| Top-5 negatives | 71.53 | 89.62 | 55.40 |
| Top-10 negatives | **71.88** | **89.80** | 54.94 |

Random negative samples instead harm performance, while high-quality hard negative samples yield robust improvements.

### Ablation Study

#### LLM vs docT5query Query Generation
The synthetic queries generated by Mixtral 8x7B achieved a HIT@4 of 61.64 on MultiHop-RAG, far exceeding the 50.86 from docT5query. Jaccard similarity analysis further validates that LLM queries are closer to the distribution of real queries.

#### Generalization of Atomic Identifiers
The ablation results on atomic identifiers are consistent with those on semantic identifiers, with all three data types making significant contributions, among which sentence-level queries contribute the most.

#### Comparison with Out-of-the-Box Retrievers
Generative retrieval models trained solely on in-domain synthetic data (without retrieval pre-training) can achieve comparable or even superior performance to retrievers such as BM25, BGE-large, and E5-Mistral-7B.

## Highlights & Insights

1. **Multi-granular + constraint-based synthetic data strategy**: Systematically exploits the advantages of LLMs in synthetic query generation, where queries of different granularities and constraints strongly complement each other.
2. **Importance of Context2ID**: By including corpus content memorization as part of the training objective, the memorization capability of generative retrieval is significantly enhanced.
3. **RPO + Hard Negatives**: Demonstrates the critical importance of negative sample quality in preference learning, showing that random negative samples are not only useless but also harmful.
4. **Transferability of Data Strategies**: The proposed strategies are effective across different identifier types (semantic/atomic) and different domains.

## Limitations & Future Work

1. Synthetic queries are mainly based on a single document and do not involve complex queries requiring multi-document reasoning.
2. Scenarios of incremental learning or generalization to unseen documents are not explored.
3. The synthetic data strategy is only validated on generative retrieval, without a systematic comparison of its effect on dense retrieval fine-tuning.
4. The optimal choice for the number of negative samples in preference learning still requires more in-depth investigation.

## Related Work & Insights

- **Generative Retrieval Modeling**: DSI, GENRE, SEAL, MINDER, etc., explore identifier types, ranking losses, and constrained decoding.
- **Synthetic Query Generation**: InPars, GPL, etc., apply synthetic data in dense retrieval, but data strategies for generative retrieval remain understudied.
- **Preference Optimization**: Methods like DPO and RPO are widely applied in LLM alignment, and this paper introduces them into retrieval ranking.

## Rating

⭐⭐⭐⭐ — A systematic and comprehensive study on data strategies, with solid experimental design and thorough ablation. It provides important guidance for the practical deployment of generative retrieval. Although the method is not overly complex, it is effective. The shortcoming lies in not addressing more complex multi-document query scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ICML 2025\] Understanding Synthetic Context Extension via Retrieval Heads](../../ICML2025/information_retrieval/understanding_synthetic_context_extension_via_retrieval_heads.md)
- [\[ACL 2025\] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](../../ACL2026/information_retrieval/why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)

</div>

<!-- RELATED:END -->
