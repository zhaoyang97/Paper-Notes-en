---
title: >-
  [Paper Note] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling
description: >-
  [AAAI 2026][Recommender Systems][Synthetic Data] This paper proposes SSRA (Semi-Supervised Relevance-Aware synthetic data pipeline), a two-stage framework that generates domain-adaptive short video data with controllable fine-grained relevance labels (4 levels) to enhance the semantic relevance modeling capability of embedding models. Online A/B testing on Douyin's dual-column feed achieves a 1.45% CTR improvement.
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Synthetic Data"
  - "Fine-Grained Relevance"
  - "Short Video Search"
  - "Semi-Supervised Learning"
  - "Embedding Model"
date: 2026-05-08
content_hash: 39f5a55aacd3ae20
---

# Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling

**Conference**: AAAI 2026
**arXiv**: [2509.16717](https://arxiv.org/abs/2509.16717)  
**Code**: None  
**Area**: Recommender Systems / Search Relevance
**Keywords**: Synthetic Data, Fine-Grained Relevance, Short Video Search, Semi-Supervised Learning, Embedding Model

## TL;DR

This paper proposes SSRA (Semi-Supervised Relevance-Aware synthetic data pipeline), a two-stage framework that generates domain-adaptive short video data with controllable fine-grained relevance labels (4 levels) to enhance the semantic relevance modeling capability of embedding models. Online A/B testing on Douyin's dual-column feed achieves a 1.45% CTR improvement.

## Background & Motivation

### Limitations of Existing Synthetic Data Methods

Embedding models are fundamental components in search and recommendation systems. Recent works leverage LLMs to synthesize diverse training data for improving embedding quality (e.g., Gemini Embedding, Qwen3-Embedding). However, two critical issues remain:

**Domain Gap**: Prompt-based synthesis methods are constrained by LLM generation capabilities, resulting in a distributional gap between synthetic data and real domain-specific data. Experiments on FinMTEB demonstrate that SOTA models on MTEB suffer significant performance degradation in the financial vertical domain.

**Insufficient Relevance Granularity**: The vast majority of synthetic methods employ only **binary relevance** (relevant/irrelevant), whereas practical retrieval tasks require **fine-grained relevance ranking** — creating a misalignment between binary labels and downstream task requirements.

### Unique Challenges in the Short Video Domain

- Short videos are inherently **multimodal**, lacking explicit textual representations
- The distribution of different relevance levels is severely imbalanced (intermediate levels 1/2 are drastically underrepresented)
- Search result ranking is influenced by non-semantic factors such as personalization and popularity

### Core Idea

The paper constructs a **controllable query generation model** $f:(d,s)\mapsto\hat{q}$, which, given a document $d$ and a target relevance label $s\in\{0,1,2,3\}$, generates queries that conform to the domain distribution and match the target relevance level semantically.

## Method

### Overall Architecture

SSRA is a two-stage semi-supervised pipeline:
- **Stage 1**: Enhances query diversity via score-based re-annotation
- **Stage 2**: Enhances alignment between generated queries and target relevance labels via iterative refinement

Two core models are trained collaboratively:
- **Query Model**: Given a document and a target relevance label, generates a query
- **Score Model**: Given a query-document pair, predicts the relevance label

### Key Designs

#### 1. Short Video Relevance Dataset Construction

**Function**: Constructs the first Chinese short video search dataset with 4-level fine-grained relevance annotations.

**4-Level Relevance Definition**:

| Label | Definition |
|-------|------------|
| 3 | Precisely and completely satisfies user intent |
| 2 | Largely satisfies intent but with missing non-critical elements |
| 1 | Partially satisfies; key entities/concepts are related but important aspects are mismatched |
| 0 | Completely fails to satisfy user intent |

**Data Construction Pipeline**:
1. **Query-Item Collection**: Query-item pairs are collected from Douyin search click logs (using two strategies: query-driven retrieval and click sampling)
2. **Document Generation**: The Doubao large model rewrites video OCR+ASR into coherent textual descriptions, combined with titles to form documents
3. **Dual-Annotation Protocol**: Two annotators label independently; disagreements are resolved by an expert arbitrator

**Data Scale**: Training set: 207,439 pairs | Retrieval test set: 10,866 pairs | Classification test set: 3,390 pairs. Intermediate relevance labels (1/2) are severely scarce (~4.5% of the training set).

#### 2. Stage 1: Score-Based Re-Annotation for Diversity Enhancement

**Function**: Addresses the query generation uniformity problem caused by "multiple documents sharing one query" in the original annotated data.

**Core Mechanism**:
1. Train a score model on annotated data to predict 4-level relevance labels
2. Group unlabeled data by document, and use the score model to assign relevance scores to multiple queries associated with each document
3. Merge re-annotated data with deduplicated original data to train the initial query model

**Design Motivation**: Multiple documents sharing a high-frequency query in the original data causes the query model to map different documents to the same query, resulting in poor generation diversity. After re-annotation, each document is associated with multiple queries of different relevance levels, forming a document-to-queries (D2Q) structure.

**Empirical Result**: Duplicate query rate reduced from 6.57% to 5.20% (a relative reduction of 20.85%).

#### 3. Stage 2: Iterative Refinement for Relevance Alignment

**Function**: Improves the alignment between queries generated by the query model and their target relevance labels.

**Core Steps**:
1. **Initial Synthesis**: Use the Stage 1 query model to generate queries conditioned on different relevance labels over the unlabeled document set
2. **Score Model Filtering**: Use the score model to predict the relevance label for each synthetic query-document pair; retain only samples where the predicted label matches the target label
3. **LLM Pairwise Consistency Filtering**: Use an LLM to compare query pairs generated under different relevance labels for the same document; discard samples whose relative ranking is inconsistent with the labels
4. Merge high-quality samples with Stage 1 training data for a second round of query model training

**Design Motivation**: The query model initially generates queries with insufficient alignment to target labels. Dual filtering via the score model and LLM ensures the reliability of relevance labels in the training data.

**Empirical Results (Human Annotation Verification)**:

| Relevance Label | Stage 1 Only | Stage 1+2 |
|-----------------|--------------|-----------|
| Label 1 | 81/200 | 130/200 |
| Label 2 | 80/200 | 131/200 |
| Label 3 | 189/200 | 178/200 |

Stage 2 improves relevance matching consistency by 25.43% (with significant gains for labels 1 and 2).

### Loss & Training

**Embedding Model Training**: Uses InfoNCE loss with label weighting:

$$\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}s_i\cdot\mathcal{L}_{\text{infoNCE}}$$

where $s_i\in\{0,1,2,3\}$ is the relevance label, causing higher-relevance positive pairs to receive greater weight.

**Implementation Details**:
- Query/Score model backbone: Doubao-1.5-Pro-32K
- Embedding model backbone: Qwen3-Embedding (0.6B and 4B scales)
- LoRA fine-tuning with rank=32, batch size=512
- Queries are synthesized for 1 million documents across multiple relevance levels; after filtering, the synthetic data is merged with annotated data for training

## Key Experimental Results

### Main Results

| Method | 0.6B nDCG@10 | 0.6B Avg AP | 4B nDCG@10 | 4B Avg AP |
|--------|--------------|-------------|------------|-----------|
| Base Model | 71.36 | 69.50 | 73.20 | 69.57 |
| SyCL Modified (prompt synthesis) | 71.50 | 67.33 | 73.56 | 67.95 |
| Vanilla SFT | 71.88 | 70.39 | 74.32 | 70.87 |
| **SSRA** | **71.97** | **70.79** | **74.47** | **71.52** |

**Key Comparisons**:
- SSRA vs. Base: 4B model nDCG@10 +1.73%, AP +2.80%
- SyCL Modified (prompt-based) **degrades** classification performance (AP drops 2.17/1.62%), demonstrating the fundamental unreliability of prompt-based synthesis in domain-specific scenarios
- SSRA consistently outperforms Vanilla SFT, validating the value of the two-stage refinement

### Ablation Study

| Method | 0.6B nDCG@10 | 0.6B AP | 4B nDCG@10 | 4B AP |
|--------|--------------|---------|------------|-------|
| w/o Stage 1 & 2 | 71.44 | 70.70 | 74.02 | 70.34 |
| w/o Stage 2 | 71.79 | 70.77 | 74.23 | 71.30 |
| **SSRA (full)** | **71.97** | **74.13** | **74.47** | **74.92** |

**Task-Specific Analysis**:
- **Retrieval tasks** benefit from both Stage 1 (diversity) and Stage 2 (relevance alignment)
- **Classification tasks** primarily benefit from Stage 2 (dependent on relevance precision rather than query diversity)

### Binary vs. Multi-Level Relevance

| Configuration | 0.6B nDCG@10 | 4B nDCG@10 | Note |
|---------------|--------------|------------|------|
| Binary labels (0,1) | 71.66 | 73.73 | Positive/negative only |
| Multi-level labels (0,1,2,3) | 71.78 | 74.23 | 4 levels |

Multi-level relevance labels consistently outperform binary labels on retrieval tasks and on intermediate classification thresholds (AP@≥1, AP@≥2), validating the value of fine-grained relevance.

### Online A/B Testing

| Metric | Gain | Description |
|--------|------|-------------|
| CTR | +1.45% | Click-through rate |
| SRR | +4.9% | Proportion of strongly relevant content |
| IUPR | +0.1054% | Image-text user penetration rate |

Randomized experiment on Douyin's dual-column feed with 190 million users/day, conducted over 10 days.

### Key Findings

1. **Prompt-based synthesis is unsuitable for vertical domains**: SyCL Modified degrades classification performance, revealing a fundamental limitation of LLM prompt synthesis in capturing domain-specific distributions
2. **Augmenting intermediate relevance labels yields significant value**: Labels 1/2 constitute only ~4.5% of the original data; SSRA compensates for these missing levels via synthesis
3. **Each stage serves a distinct role**: Stage 1 → diversity (duplicate rate −20.85%), Stage 2 → precision (consistency +25.43%)
4. **Scale matters**: The 4B model benefits more than the 0.6B model, suggesting that SSRA's synthetic data is better utilized by larger models

## Highlights & Insights

1. **Data Resource Contribution**: The first 4-level relevance-annotated Chinese short video search dataset, filling a benchmark gap in this domain
2. **Semi-Supervised Closed-Loop Design**: Score model annotation → Query model generation → Score model verification → LLM filtering forms a quality-improving closed loop
3. **Novel Perspective**: Relevance diversity — rather than merely query/document diversity — is identified as a critical dimension for embedding model training
4. **Industrial Validation**: Significant gains in online experiments at Douyin scale provide strong practical credibility
5. **Label-Weighted InfoNCE**: A simple yet effective approach for incorporating multi-level relevance into contrastive learning loss

## Limitations & Future Work

1. **Dataset Not Released**: Although the dataset is claimed as a contribution, commercial constraints may prevent full open-sourcing
2. **Dependency on Domain-Annotated Data**: SSRA still requires a sufficient amount of high-quality annotated data to train the score model, entailing non-trivial cold-start costs
3. **Cost of LLM Filtering**: The pairwise consistency check in Stage 2 incurs considerable overhead at large scale
4. **Fixed Number of Label Levels**: Only 4-level relevance is validated; the effectiveness of finer granularity (e.g., continuous scores) remains unexplored
5. **Active learning** could be explored to select the most informative samples for annotation, reducing human labeling costs
6. **Multimodal information is underutilized**: Short videos contain visual content; converting video to text via OCR/ASR alone may result in information loss

## Related Work & Insights

- **SyCL (Esfandiarpoor et al. 2025)**: Generates 4-level relevance documents via prompting, but this paper demonstrates that prompt-based methods perform poorly in vertical domains
- **Gecko (Lee et al. 2024)**: Uses generated queries to retrieve candidate documents and applies LLM scoring to select positives, inspiring the "synthesis + verification" paradigm
- **Qwen3-Embedding (Zhang et al. 2025)**: Leverages Persona Hub to guide synthesis, focusing on persona diversity rather than relevance diversity
- **Hard Negative Mining**: Widely used to enhance model sensitivity to subtle distinctions; this paper provides a more systematic solution
- Insight: The value of **semi-supervised frameworks** in data synthesis is underappreciated — collaborative iterative training of two models is more effective than single-pass prompt strategies

## Rating

- Novelty: ⭐⭐⭐⭐ (Two-stage semi-supervised synthesis pipeline + relevance diversity perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Offline multi-metric evaluation + online A/B test with 190M users + multi-scale models + ablation + human verification)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though some sections are overly verbose)
- Value: ⭐⭐⭐⭐⭐ (Validated in production deployment, addressing real industrial problems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[AAAI 2026\] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information](generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[CVPR 2025\] FineVQ: Fine-Grained User Generated Content Video Quality Assessment](../../CVPR2025/recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)

</div>

<!-- RELATED:END -->
