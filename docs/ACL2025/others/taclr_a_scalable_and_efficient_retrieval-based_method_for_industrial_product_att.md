---
title: >-
  [Paper Note] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification
description: >-
  [ACL 2025][Product Attribute Value Identification] TACLR proposes the first product attribute value identification (PAVI) method based on the retrieval paradigm. By incorporating taxonomy-aware contrastive learning and an adaptive inference mechanism, it completely outperforms classification and generation methods in handling implicit values, OOD (out-of-distribution) values, and normalized outputs, and has been successfully deployed on the Xianyu platform.
tags:
  - "ACL 2025"
  - "Product Attribute Value Identification"
  - "Contrastive Learning"
  - "Taxonomy-Aware Negative Sampling"
  - "Retrieval Paradigm"
  - "Industrial Deployment"
date: 2026-05-08
content_hash: 1ed21d395bd0bd8a
---

# TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification

**Conference**: ACL 2025  
**arXiv**: [2501.03835](https://arxiv.org/abs/2501.03835)  
**Code**: Yes ([https://github.com/SuYindu/TACLR](https://github.com/SuYindu/TACLR))  
**Area**: Other  
**Keywords**: Product Attribute Value Identification, Contrastive Learning, Taxonomy-Aware Negative Sampling, Retrieval Paradigm, Industrial Deployment

## TL;DR

TACLR proposes the first product attribute value identification (PAVI) method based on the retrieval paradigm. By incorporating taxonomy-aware contrastive learning and an adaptive inference mechanism, it completely outperforms classification and generation methods in handling implicit values, OOD (out-of-distribution) values, and normalized outputs, and has been successfully deployed on the Xianyu platform.

## Background & Motivation

Product attribute values are the core structured information of e-commerce platforms, supporting search, recommendation, and business analysis. However, attribute values provided by sellers are often incomplete or even inaccurate—which is particularly severe on second-hand e-commerce platforms (such as Xianyu).

Existing methods face three types of fundamental challenges:

| Paradigm | Implicit Values | OOD Values | Normalized Output | Core Problem |
|------|--------|-------|-----------|---------|
| Extraction (NER/QA) | ✗ | ✓ | ✗ | Cannot reason implicit values; requires post-processing normalization |
| Classification | ✓ | ✗ | ✓ | Cannot identify new values outside the training set |
| Generation (LLM) | ✓ | ✓ | ✗ | Uncontrollable output, high computational cost |
| **Retrieval (TACLR)** | ✓ | ✓ | ✓ | **First to address all three simultaneously** |

For example, if a product description says "iphone12pm", PAVI needs to identify the standardized value "iPhone 12 Pro Max" (unnormalized value); if the brand "Apple" is not directly mentioned in the description but can be inferred from "iPhone 12 Pro Max", it should be captured (implicit value).

## Method

### Overall Architecture

TACLR models PAVI as an information retrieval task:
- **Query**: Product (title + description)
- **Corpus**: All attribute values within the attribute taxonomy
- A shared text encoder is used to encode the products and candidate values into embeddings, and matching values are retrieved based on similarity.

### Key Designs

1. **Encoding Strategy**

    - Product side: Concatenate title and description -> `title: {title} description: {description}`
    - Value side: Prompt enriched with context combining category and attribute -> `A {category} with {attribute} being {value}`
    - Design Motivation: Context-enriched prompts allow the model to better distinguish semantically close values that belong to different attributes.

2. **Taxonomy-Aware Contrastive Learning**

    - Core Idea: Instead of using random in-batch negatives, hard negatives are selected from the **same category and same attribute**.
    - For example: For the phone brand Apple, the negative samples are Huawei and Samsung, rather than T-shirt size L.
    - Learnable null value $v_0^a$: Explicitly learns a null embedding for product-attribute pairs that do not possess a certain attribute.
    - Loss function: InfoNCE contrastive loss with temperature parameter $\tau$.

3. **Adaptive Inference**

    - Problem: Static thresholds are infeasible in large-scale taxonomies (with over 26K+ category-attribute pairs).
    - Solution: Utilizes the similarity score of the null value embedding as a **dynamic threshold**.
    - If $\max_{v \in \mathcal{V}_a} s(i,v) > s(i, v_0^a)$, the top-1 value is retrieved; otherwise, null is outputted.
    - This is equivalent to adding the null value to the candidate set and performing a top-1 selection.

4. **Efficient Inference Pipeline**

    - Offline: Precomputes embeddings for all values and builds an index using Faiss.
    - Online: Requires encoding the product embedding only once, then comparing it against candidate values in the index.

### Loss & Training

$$\mathcal{L}_a = -\log \frac{\exp(s(i, v_a^+)/\tau)}{\exp(s(i, v_a^+)/\tau) + \sum_{v \in \mathcal{V}_a^-} \exp(s(i,v)/\tau)}$$

The total loss is the sum of losses over all attributes: $\mathcal{L}_i = \sum_{a \in \mathcal{A}_c} \mathcal{L}_a$

## Key Experimental Results

### Main Results — Xianyu-PAVI and WDC-PAVE (Table 4)

| Paradigm | Method | Xianyu F1 | WDC F1 | WDC F1 (Excl.) |
|------|------|-----------|--------|----------------|
| Classification | BERT-CLS | 50.5 | 20.5 | 23.4 |
| Generation | Llama3.1 (zero-shot) | 35.7 | 58.6 | 64.6 |
| Generation | Llama3.1 (RAG) | 47.6 | 77.2 | 80.1 |
| Generation | Llama3.1 (fine-tune) | 84.7 | 59.0 | 64.5 |
| Generation | Qwen2.5 (RAG) | 63.2 | 74.2 | 78.3 |
| **Retrieval** | **TACLR** | **86.2** | **72.6** | **80.3** |

### Inference Efficiency (Table 5)

| Method | Latency (ms) | Throughput (samples/sec) |
|------|----------|----------------|
| BERT-CLS | 8.6 | 930 |
| **TACLR** | **12.7** | **630** |
| Qwen2.5 (zero-shot) | 84.0 | 95 |
| Llama3.1 (RAG) | 137.9 | 58 |

### Ablation Study

| Ablation Option | Conclusion |
|--------|------|
| Taxonomy-aware vs. In-batch negative sampling | Taxonomy-aware improves F1 from 53.3% to 86.2% |
| Dynamic threshold vs. Static threshold | Dynamic threshold (86.2%) vs. Optimal static (80.2%) |
| Context-enriched prompt vs. Value only | Steadily improves from 83.2% to 86.2% |
| Normalized vs. Unnormalized/implicit values | TACLR achieves optimal performance on both (87.9% / 82.9%) |

### Key Findings

1. TACLR achieves an F1-score of 86.2% on a large-scale e-commerce dataset (8,803 categories, 6.3 million attribute-value tuples), outperforming fine-tuned LLMs.
2. Inference speed is 5-10 times faster than LLM-based solutions (630 vs. 58-95 samples/sec), being only slightly slower than simple classification.
3. Taxonomy-aware hard negative sampling is the core contributor to the performance gain (53.3% -> 86.2%).
4. Adaptive dynamic threshold solves the problem wherein different attribute pairs require different cutoff values in large-scale taxonomies.
5. It has been successfully deployed in production on the Xianyu platform, processing millions of product listings daily.

## Highlights & Insights

1. **Paradigm Innovation**: First to model PAVI as a retrieval task, achieving optimal performance across three capabilities simultaneously: implicit values, OOD values, and normalization.
2. **Exquisite Engineering for Industrial Deployment**: The design of null value embeddings + adaptive thresholds elegantly resolves the "when to output nothing" problem.
3. **Balancing Efficiency and Effectiveness**: The online inference architecture utilizing single-pass encoding + Faiss index is highly suitable for high-load industrial scenarios.
4. **Strong Scalability**: The shared encoder design enables the model to generalize to newly added categories and values.

## Limitations & Future Work

- On WDC-PAVE, the gap between TACLR and the RAG baseline is marginal (72.6 vs. 77.2), demonstrating limited advantages in small-dataset scenarios.
- It only supports text inputs and does not integrate multimodal information (such as product images).
- It currently only handles top-1 values, and further expansion is needed for multi-valued attribute scenarios.
- The incremental learning capability during taxonomy updates is not discussed.

## Related Work & Insights

- Similar to the CLIP approach (shared encoder + contrastive learning), but introduces taxonomy-aware negative sampling and a null-value mechanism.
- Compared to RAG-based approaches, TACLR does not rely on LLM generative capabilities, leading to faster and more stable inference.
- Brinkmann et al. (2024) explored LLMs for extraction + normalization, but suffered from low efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of the retrieval paradigm in PAVI; the design of null-value embeddings is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering large-scale industrial datasets, public datasets, multi-paradigm comparisons, efficiency analyses, and thoroughly designed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with precise problem definitions.
- **Value**: ⭐⭐⭐⭐⭐ — Deployed on a real-world e-commerce platform, demonstrating extremely high industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Value Residual Learning](value_residual_learning.md)
- [\[ACL 2025\] HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](../../ICML2025/others/efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)

</div>

<!-- RELATED:END -->
