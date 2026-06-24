---
title: >-
  [Paper Note] QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering
description: >-
  [ACL2025][NLP Understanding][query-focused summarization] This paper proposes the QQSUM task and the QQSUM-RAG framework. By leveraging KP-oriented retrieval and clustering alongside a Next-KP-Generation training strategy, it generates Key Point summaries containing diverse opinions and their quantified popularity from product reviews, addressing the limitation of traditional PQA systems that only output single-perspective answers.
tags:
  - "ACL2025"
  - "NLP Understanding"
  - "query-focused summarization"
  - "key point analysis"
  - "product QA"
  - "RAG"
  - "opinion mining"
date: 2026-05-08
content_hash: 1038891b8486f1bc
---

# QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering

**Conference**: ACL 2025  
**arXiv**: [2506.04020](https://arxiv.org/abs/2506.04020)  
**Code**: None  
**Area**: NLP Understanding  
**Keywords**: query-focused summarization, key point analysis, product QA, RAG, opinion mining  

**Conference**: ACL2025  
**arXiv**: [2506.04020](https://arxiv.org/abs/2506.04020)  
**Code**: [antangrocket1312/QQSUMM](https://github.com/antangrocket1312/QQSUMM)  
**Area**: nlp_understanding  
**Keywords**: query-focused summarization, key point analysis, product QA, RAG, opinion mining

## TL;DR

This paper proposes the QQSUM task and the QQSUM-RAG framework. By leveraging KP-oriented retrieval and clustering alongside a Next-KP-Generation training strategy, it generates Key Point summaries containing diverse opinions and their quantified popularity from product reviews, addressing the limitation of traditional PQA systems that only output single-perspective answers.

## Background & Motivation

### Problem Definition

Existing review-based product question answering (PQA) systems suffer from a core limitation: they can only generate answers reflecting a single perspective, failing to capture the diverse opinions present in user reviews. For example, when comparing camera lenses, some users focus on versatility and price, while others focus on image quality and speed. Existing systems cannot present these different perspectives simultaneously.

### Limitations of Prior Work

**Traditional RAG-based PQA**: After retrieving relevant reviews, LLMs are tasked with generating answers. However, LLMs tend to output dominant opinions, making it difficult to present multifaceted perspectives.

**Key Point Analysis (KPA)**: This can summarize reviews into key points and quantify their popularity, but only performs generic summarization and cannot focus on specific user queries.

**Abstractive Summarization**: This generates fluent review summaries but lacks the capability to show and quantify diverse opinions.

### Design Motivation

By combining KPA with query-focused summarization, this paper proposes the **Quantitative Query-Focused Summarization (QQSUM)** task, which generates structured answers for user queries containing multiple Key Points and their popularity counts.

## Method

### Task Formulation

Given a query $q$ and a set of product reviews $R_e$, the goal of QQSUM is to:

1. Retrieve a subset of query-relevant reviews $D$.
2. Generate a KP summary $S = \{kp_1, ..., kp_n\}$, with each KP accompanied by a popularity count $|C_i|$.

### QQSUM-RAG Framework

The framework is based on the RAG paradigm and consists of two phases:

#### Phase 1: KP-Oriented Retrieval

- Uses a shared encoder $E$ to encode queries and reviews, calculating similarity via dot product.
- Only reviews with a similarity $\ge 1$ are retained.
- **Key Innovation**: Performs incremental clustering on the retrieved reviews, grouping semantically similar reviews together. Each group conceptually corresponds to a KP.
- Clustering Algorithm: Iterates through reviews, calculating the average cosine similarity with existing clusters. If it exceeds a threshold $\lambda=1.2$, the review is assigned to that cluster; otherwise, a new cluster is created (a single review can belong to multiple clusters).
- Training Objective: Minimizes the MSE loss between predicted and ground-truth clusters.

#### Phase 2: KP Summary Generation

- **Next-KP-Generation Training**: Inspired by Next-Token Prediction, the LLM is trained to iteratively generate the next KP using the previously generated KPs as context to avoid redundancy.
- The generation loss for each $kp_i$ is calculated using negative log-likelihood (NLL), referencing the most similar gold KP.
- **Perplexity Distillation**: Feeds back supervision signals from the LLM to the retriever, helping the retriever better rank documents.

#### Joint Training Loss

$$\mathcal{L} = (1-d) \cdot (\mathcal{L}_{clus} + \text{gold\_score}) + d \cdot \mathcal{L}_{gen}$$

where $d$ is a damping factor balancing retrieval loss and generation loss.

### Data Annotation: Human-LLM Collaboration Pipeline

An **AmazonKP** dataset is constructed based on the AmazonQ&A dataset through a three-stage annotation process:

1. **Stage 1**: GPT-4o-mini extracts non-overlapping KPs from community gold answers (precision 87.5%, coverage 90%).
2. **Stage 2**: LLM annotates review-KP pairs, followed by MTurk manual verification.
3. **Stage 3**: Human annotators write KP summaries (format: "N comments say that $kp_i$").

## Key Experimental Results

### AmazonKP Dataset Statistics

| Metric | Train | Test |
|---|---|---|
| Number of Product Categories | 17 | 17 |
| Number of Instances per Category | 2 | 148 |
| Total Number of Instances | 34 | 2,516 |
| Number of Reviews per Query | 452.03 | 431.62 |
| Number of KPs per Query | 9.26 | 6.90 |
| Popularity per KP | 6.37 | — |

### KP Text Quality (Automatic Evaluation, Best Configuration)

| Method | ROUGE-1 | BERTScore sF1 | BLEURT sF1 | G-Eval sF1 | BERTScore RD↓ |
|---|---|---|---|---|---|
| **QQSUM-RAG + Mistral** | **0.256** | **0.33** | **0.46** | **0.85** | **0.37** |
| (Retriever+LLM)_co-train + Mistral | 0.209 | 0.32 | 0.44 | 0.81 | 0.43 |
| Frozen Ret. + GPT-4-Turbo | 0.197 | 0.28 | 0.41 | 0.77 | 0.44 |
| Frozen Ret. + PAKPA | 0.179 | 0.31 | 0.44 | 0.80 | 0.46 |
| Frozen Ret. + RKPA-Base | 0.121 | 0.14 | 0.39 | 0.69 | 0.50 |

### KP Quantification Performance

| Method | Precision | Recall | F1 | QuantErr↓ | AlignScore |
|---|---|---|---|---|---|
| **QQSUM-RAG + Mistral** | 0.694 | **0.869** | **0.792** | **4.24** | **0.749** |
| QQSUM-RAG + Vicuna | 0.538 | 0.684 | 0.602 | 7.83 | 0.630 |
| (Ret.+LLM)_co-train + Mistral | 0.567 | 0.249 | 0.346 | 18.10 | 0.653 |
| Frozen Ret. + GPT-4-Turbo | **0.746** | 0.200 | 0.313 | 16.63 | 0.673 |
| Frozen Ret. + PAKPA | 0.762 | 0.520 | 0.619 | 6.68 | 0.749 |

### Human Evaluation (Bradley-Terry Score, 7 Dimensions)

QQSUM-RAG leads significantly across all 7 dimensions, achieving a Coverage score of 28.44 (runner-up: 16.20), a Validity score of 35.23 (runner-up: 22.91), and an overall improvement of up to 4.58x.

## Highlights

1. **Task Innovation**: Establishes the QQSUM task, combining query-focused summarization with opinion quantification, which fills the gap of multi-perspective opinion quantification in PQA.
2. **KP-Oriented Retrieval and Clustering**: Groups retrieved results by semantic similarity, with each cluster representing a KP, naturally enabling multi-view display and popularity estimation.
3. **Next-KP-Generation**: Adapts the idea of Next-Token Prediction to the KP level, preventing the generation of redundant KPs.
4. **Highly Few-Shot Efficient**: Outperforms baselines, including in-context learning with GPT-4-Turbo, with only 34 training samples.
5. **Outstanding Quantifying Performance**: Achieves an F1 score of 0.792 and a QuantErr of only 4.24, marking a 67.12% improvement over the SOTA KPA system PAKPA.

## Limitations

1. **Single Dataset**: Evaluation is limited to AmazonQ&A, lacking generalization validation across different datasets and domains.
2. **Clustering Quality Depends on Thresholds**: Retrieval and clustering thresholds ($\lambda=1.2$) are empirically set and may require tuning for different scenarios.
3. **Sentence-Level Quantification Error**: Review sentences often contain mixed opinions, making it hard to cleanly isolate different aspects into separate clusters.
4. **High Annotation Cost**: Each query requires 2K-3.5K review-KP matching annotations, which limits the scale of the training data.
5. **Limited Model Size**: Experiments are restricted to 7B parameter LLMs (Vicuna-7B, Mistral-7B); the impact of scaling up to larger models remains unexplored.

## Related Work

- **Review-based PQA**: Evaluated from extractive (Yu et al., 2012) to generative (Chen et al., 2019; Gao et al., 2019) methods, but both tend to output single-sided answers and suffer from hallucination and factual inconsistency issues.
- **Key Point Analysis**: Bar-Haim et al. (2020, 2021) introduced KPA for argument/review summarization; Tang et al. (2024a,b) incorporated ABSA to improve KP extraction, but these methods do not support query-focused tasks.
- **Text Summarization**: Extractive (Mihalcea & Tarau, 2004) and abstractive (Bražinskas et al., 2020) approaches lack the ability to quantify diverse opinions. LLM-based summarization (Bhaskar et al., 2023) is fluent but non-quantitative.
- **RAG Frameworks**: Atlas (Izacard et al., 2023) provided a foundation for joint retriever-generator training. This work expands on the framework by introducing KP-oriented retrieval and clustering.

## Rating

| Dimension | Score (1-10) | Explanation |
|---|---|---|
| Novelty | 8 | Novel task definition, successfully integrating KPA with query-focused summarization and quantification for the first time. |
| Technical Depth | 7 | The design of KP-oriented retrieval/clustering, Next-KP-Generation, and joint training is reasonable, though not overly complex. |
| Experimental Thoroughness | 8 | Offers both automatic and human evaluations with multi-dimensional comparisons against various baselines, along with ablation and case studies. |
| Practical Value | 7 | Practical for e-commerce QA scenarios, though the reliance on annotated data and single-dataset evaluation limits immediate production deployment. |
| Writing Quality | 7 | Clear structure, though some formulas and notations are relatively dense. |
| Overall Score | **7.5** | The task definition is highly valuable, and the proposed method is effective, though generalization remains to be verified. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](../../ICCV2025/nlp_understanding/balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[ACL 2025\] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)

</div>

<!-- RELATED:END -->
