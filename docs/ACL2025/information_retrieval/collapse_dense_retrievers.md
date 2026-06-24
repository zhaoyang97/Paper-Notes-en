---
title: >-
  [Paper Note] Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence
description: >-
  [ACL 2025][Information Retrieval & RAG][Dense Retrievers] This paper presents the first systematic study of the individual and combined effects of multiple heuristic biases (brevity, position, literal, and repetition biases) in dense retrievers. It reveals that when multiple biases are compounded, the probability of a retriever selecting the document containing the answer drops below 10%, and these biases can be exploited to manipulate RAG systems…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Dense Retrievers"
  - "Retrieval Biases"
  - "RAG Robustness"
  - "Adversarial Attacks"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: cfaa5bbd5f0a2454
---

# Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence

**Conference**: ACL 2025  
**arXiv**: [2503.05037](https://arxiv.org/abs/2503.05037)  
**Area**: Information Retrieval / NLP  
**Keywords**: Dense Retrievers, Retrieval Biases, RAG Robustness, Adversarial Attacks, Retrieval-Augmented Generation  

## TL;DR

This paper presents the first systematic study of the individual and combined effects of multiple heuristic biases (brevity, position, literal, and repetition biases) in dense retrievers. It reveals that when multiple biases are compounded, the probability of a retriever selecting the document containing the answer drops below 10%, and these biases can be exploited to manipulate RAG systems, leading to a 34% drop in performance.

## Background & Motivation

**Central Role of Dense Retrievers**: Dense retrieval models (such as Dragon+, Contriever) are widely used in Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) systems. As the initial step, the retrieval quality directly determines the downstream performance.

**Importance of Robustness**: Existing evaluations primarily focus on downstream task performance (such as the BEIR benchmark), lacking an in-depth investigation into the intrinsic behavior of retrievers. Consequently, key questions regarding their adversarial robustness remain unanswered.

**Known but Fragmented Issues**: Prior work has individually identified issues such as position bias, lexical overlap, and common entity preference. However, the individual effects and combined interactions of multiple biases have never been simultaneously investigated within a unified framework.

**Innovative Methodology**: A controlled experimental framework is constructed by repurposing a document-level relation extraction dataset (Re-DocRED). This achieves precise control over document structures and factual distributions, which is unattainable in traditional IR benchmarks.

## Method

### Overall Architecture

Constructing a controlled query-document pair experimental framework based on the Re-DocRED dataset:

1. **Relation-to-Query Mapping**: Converts relation triples (Head Entity, Relation, Tail Entity) in Re-DocRED into retrieval queries via templates. For example, "educated at" $\rightarrow$ "Where was {Head Entity} educated?"

2. **Controlled Document Pair Construction**: Constructs a pair of documents $D_1$ and $D_2$ for each bias type, differing only along the target bias dimension while strictly controlling all other confounding factors.

3. **Statistical Testing**: Employs paired t-tests to compare the difference in retrieval scores, $M(Q,D_1) - M(Q,D_2)$, with 250 queries configured for each bias setting.

### Key Designs

Five bias types are investigated, each defined with mathematical precision:

#### 1. Answer Importance
- $D_1$: Evidence sentence (containing head and tail entities) + neutral sentences
- $D_2$: Head entity sentence (without tail entity) + identical neutral sentences
- Tests whether the retriever truly identifies the existence of the answer.

#### 2. Position Bias
- Places the evidence sentence at different positions in the document (beginning, middle, end).
- The remaining content consists of neutral sentences that do not contain the head or tail entities.
- Tests whether the retriever favors information at the beginning of the document.

#### 3. Literal Bias
- Leverages multiple surface forms of entities (e.g., "NYC" vs "New York City").
- Tests whether query-document scoring favors exact literal matches over semantic equivalence.

#### 4. Brevity Bias
- $D_1$: Evidence sentence only
- $D_2$: Evidence sentence + the rest of the document
- Tests whether the retriever favors shorter documents.

#### 5. Repetition Bias
- $D_1$: Evidence sentence + 2 sentences containing the head entity (repeating the head entity)
- $D_2$: Evidence sentence + 2 neutral sentences without head or tail entities
- Tests whether the retriever inflates scores due to entity repetition.

#### 6. Multi-Bias Combination (Foil vs Evidence)
- **Foil Document** $D_1$: 2$\times$ head entity repetitions + sentence containing head entity (a short document, head entity at the beginning, repeatedly occurring — but containing no answer)
- **Evidence Document** $D_2$: 4 irrelevant sentences + evidence sentence + 4 irrelevant sentences (a long document, evidence in the middle — but containing the answer)

## Key Experimental Results

### Main Results

**Dense Retrieval Models Evaluated**:

| Model | Pooling Method | NQ nDCG@10 | NQ Recall@10 |
|------|----------|------------|--------------|
| Dragon RoBERTa | CLS | 0.55 | 0.75 |
| Dragon+ | CLS | 0.54 | 0.74 |
| COCO-DR Base | CLS | 0.50 | 0.71 |
| Contriever MSMARCO | avg | 0.50 | 0.71 |
| RetroMAE MSMARCO FT | CLS | 0.48 | 0.68 |
| Contriever | avg | 0.25 | 0.41 |

**Catastrophic Results of Multi-Bias Combinations** (Foil vs Evidence, 250 samples):

| Model | Accuracy in Selecting Evidence Doc | t-statistic | p-value |
|------|---------------------|---------|-----|
| Contriever | **0.4%** | -34.58 | <0.01 |
| RetroMAE MSMARCO | **0.4%** | -41.49 | <0.01 |
| Contriever MSMARCO | **0.8%** | -42.25 | <0.01 |
| Dragon RoBERTa | **0.8%** | -36.53 | <0.01 |
| Dragon+ | **1.2%** | -40.94 | <0.01 |
| COCO-DR Base | **2.4%** | -32.92 | <0.01 |
| ColBERT v2 | **7.6%** | -20.96 | <0.01 |
| ReasonIR-8B | **8.0%** | -36.92 | <0.01 |

**The accuracy of all retrieval models in selecting the correct document under the multi-bias combination falls below 10%!**

**Practical Impact on RAG Systems**:

| Document Type | GPT-4o-mini Accuracy | GPT-4o Accuracy |
|----------|------------------|--------------|
| Poisoned Document (Preferred) | **32.0%** | **30.8%** |
| Foil Document | 44.0% | 62.8% |
| No Document | 52.0% | 64.8% |
| Evidence Document | 88.0% | 93.6% |

**Key Finding**: RAG performance when using poisoned documents is worse than providing no documents at all (32.0% vs 52.0%), representing a 34% drop in performance.

### Key Findings

1. **Significant Impact of Single Biases**: Paired t-tests indicate that brevity bias, literal bias, and position bias are the most severe issues, whereas repetition bias has a relatively minor effect.

2. **Neglect of Answer Existence**: The retrievers' sensitivity to the existence of answers is weaker than their sensitivity to bias signals. Contriever even assigns higher scores to documents that do not contain the answer.

3. **Catastrophic Compounding Effects**: When multiple biases are combined, all models exhibit extreme degradation. Even the best-performing model has only an 8% probability of selecting the correct document.

4. **Position Bias Originating from Training**: Research indicates that position bias is already introduced during the contrastive pre-training phase and further worsens during MS MARCO fine-tuning.

5. **Mechanistic Explanation for Brevity Bias**: With mean-pooling and CLS pooling strategies, when compressing document representations, irrelevant content "pollutes" the representation of evidence, resulting in higher scores for shorter documents.

6. **Practical Harm of Literal Bias**: Retrievers fail to recognize the semantic equivalence of different surface forms (e.g., "Gomes" and "Gomez"), severely limiting cross-lingual and cross-cultural retrieval capabilities.

## Highlights & Insights

1. **Unified Analytical Framework**: This work systematically compares the individual and combined effects of multiple retrieval biases within a single framework for the first time, using paired t-tests to ensure statistical rigor.

2. **Methodological Innovation**: The strategy of repurposing relation extraction datasets to construct controlled retrieval experiments is ingenious and generalizable, solving the limitation of traditional IR benchmarks where document content cannot be precisely controlled.

3. **DecompX Visualization**: By using DecompX to decompose BERT representations, the contribution of each query and document token to the final retrieval score is visualized at the token level, intuitively demonstrating the underlying bias mechanisms.

4. **Practical Security Threats**: It demonstrates how attackers can exploit these biases to construct poisoned documents, steering retrievers to favor poisoned documents 100% of the time, thereby misleading downstream RAG systems.

5. **Coverage of Recent Models**: Evaluations include newer models such as ColBERT v2 and ReasonIR-8B, demonstrating that these vulnerabilities persist in modern architectures.

## Limitations & Future Work

1. **Focus on Dense Retrieval**: The analysis primarily targets dense retrieval models, with limited analysis on biases in sparse retrieval (e.g., BM25), hybrid retrieval, and re-ranking models.

2. **Dataset Construction Limitations**: Experiments based on Re-DocRED (derived from Wikipedia) may not fully represent all retrieval scenarios.

3. **Lack of Systematic Mitigation Solutions**: The paper mainly focuses on discovering and quantifying the problem, without proposing specific bias-mitigation or robust-retrieval methods.

4. **Scale Constraints**: Each bias setting is configured with only 250 samples, which, while sufficient for statistical testing, may not cover long-tail cases.

5. **Query Type Limitations**: Queries are generated via templates from relation triples, which may not represent the diversity of real-world user queries.

## Related Work & Insights

- **IR Benchmarks**: BEIR (Thakur et al., 2021), COIR (Li et al., 2024) code retrieval, LitSearch (Ajith et al., 2024) scientific literature retrieval
- **Retrieval Model Analysis**: Coelho et al. (2024) position bias; Ram et al. (2023) reliance on lexical overlap; Sciavolino et al. (2021) common entity preference
- **Adversarial Attacks**: Lin et al. (2024) corpus poisoning; Long et al. (2024) backdoor attacks; Boucher et al. (2023) encoding attacks
- **Neural IR Analysis**: MacAvaney et al. (2022) framework for bias and sensitivity; Modarressi et al. (2023) DecompX representation decomposition

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 8 | First unified analysis of individual and combined effects of multiple biases |
| Technical Depth | 8 | Well-designed controlled experiments with rigorous statistical analysis |
| Experimental Thoroughness | 9 | 6 bias types $\times$ 6 models + validation of downstream RAG impact |
| Writing Quality | 8 | Clear structure, rich and intuitive figures and tables |
| Value | 9 | Direct guidelines for improving retriever robustness and ensuring RAG system security |
| **Overall Score** | **8.4** | An excellent empirical study revealing critical vulnerabilities of dense retrievers |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR](when_should_dense_retrievers_be_updated_in_evolving_corpora_detecting_out-of-dis.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ACL 2025\] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval](helios_harmonizing_early_fusion_late_fusion_and_llm_reasoning_for_multi-granular.md)
- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)

</div>

<!-- RELATED:END -->
