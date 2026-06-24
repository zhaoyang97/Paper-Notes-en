---
title: >-
  [Paper Note] Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings
description: >-
  [ACL 2025][Information Retrieval & RAG][Text Embeddings] This paper systematically studies the performance robustness of prompt-based text embedding models (such as gte-Qwen2, E5-mistral, etc.) under post-processing dimensionality reduction. It discovers that classification/clustering tasks can largely preserve performance with only 0.5% of the original dimensions, and quantitatively explains the differences in embedding redundancy across different task prompts using two metr…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Text Embeddings"
  - "Dimensionality Reduction"
  - "Intrinsic Dimension"
  - "Isotropy"
  - "Prompt-Based Embedding"
  - "MTEB"
date: 2026-05-08
content_hash: 58d93962941e29d7
---

# Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings

**Conference**: ACL 2025  
**arXiv**: [2506.01435](https://arxiv.org/abs/2506.01435)  
**Author**: Hayato Tsukagoshi, Ryohei Sasano (Nagoya University)  
**Code**: Unreleased  
**Area**: Information Retrieval  
**Keywords**: Text Embeddings, Dimensionality Reduction, Intrinsic Dimension, Isotropy, Prompt-Based Embedding, MTEB  

## TL;DR

This paper systematically studies the performance robustness of prompt-based text embedding models (such as gte-Qwen2, E5-mistral, etc.) under post-processing dimensionality reduction. It discovers that classification/clustering tasks can largely preserve performance with only 0.5% of the original dimensions, and quantitatively explains the differences in embedding redundancy across different task prompts using two metrics: Intrinsic Dimension (ID) and isotropy (IsoScore).

## Background & Motivation

### Background
Prompt-based text embedding models (such as the instruction-based E5-mistral, prefix-based Nomic, etc.) generate embeddings by receiving task-specific natural language instructions or prefixes, demonstrating excellent performance on various downstream tasks. However, these models typically output embeddings with **thousands of dimensions** (e.g., E5-mistral outputs 4096 dimensions), leading to high storage and computational costs.

### Limitations of Prior Work
- Methods like Matryoshka Representation Learning (MRL) require **introducing special mechanisms during training** to support post-processing dimensionality reduction, increasing training complexity.
- Existing studies on embedding isotropy (such as SimCSE, WhiteningBERT) mainly focus on STS tasks, without systematically analyzing the differences in embedding geometric properties under different task prompts.
- There is a lack of quantitative analysis on the **redundancy** in prompt-based embeddings, making it unclear why certain tasks are more robust to dimensionality reduction.

### Design Motivation
To validate an intuition: Do high-dimensional prompt-based embeddings contain a large number of redundant dimensions? If so, does the degree of redundancy vary by task? Can geometric metrics (intrinsic dimension, isotropy) be used to quantitatively explain this difference?

## Method

### Experimental Framework Design
The study is divided into two phases: (1) Dimensionality reduction robustness experiments—systematically evaluating the impact of dimensionality reduction on different tasks on the MTEB benchmark; (2) Redundancy analysis—quantifying the geometric properties of embeddings using ID and IsoScore.

### Key Design 1: Naive Dimensionality Reduction Evaluation

The simplest dimensionality reduction method is adopted: **directly retaining the first $d$ dimensions of the embeddings** without any transformation or normalization. By gradually reducing $d$ across four types of tasks (classification, clustering, retrieval, STS), the performance variation curves are observed.

To verify that the observed trend is not an artifact of a specific dimensionality reduction method, comparison is also made with random selection of $d$ dimensions, PCA, UMAP, and Isomap. Experiments show that naive truncation is sufficient to reveal the core patterns:

- **Classification tasks**: Instruction-based models suffer almost no performance loss when reduced to 8 dimensions (0.2% of the original dimensions). gte-Qwen2 achieves a score of 76.34 using only 2 dimensions, surpassing the full 1024-dimensional performance of E5-large (75.69).
- **Clustering tasks**: LLM-based models lose only about 0.8 points in performance when reduced to 128 dimensions ($<4\%$).
- **Retrieval/STS tasks**: Performance decreases rapidly as the dimensionality decreases, with consistent trends across all models.

### Key Design 2: Intrinsic Dimension and Isotropy Analysis

10,000 text segments were randomly sampled from English Wikipedia to generate embeddings for each model under different task prompts, and then the following metrics were calculated:

**Intrinsic Dimension (ID)**: Estimated using the TwoNN method. TwoNN estimates the true dimension of the data manifold by analyzing the ratio of distances from each data point to its two nearest neighbors. On a $d$-dimensional uniformly distributed manifold, this distance ratio follows a Pareto distribution:

$$\mu = \frac{r_2}{r_1} \sim \text{Pareto}(\alpha = d)$$

where $r_1, r_2$ are the distances to the first and second nearest neighbors, respectively. The maximum likelihood estimation yields:

$$\hat{d} = \left[\frac{1}{N}\sum_{i=1}^{N}\ln\frac{r_{2,i}}{r_{1,i}}\right]^{-1}$$

TwoNN is robust to manifold curvature and non-uniform sampling.

**Isotropy (IsoScore)**: Measures the degree of uniform distribution of embeddings in space. By computing the variance-covariance matrix of embeddings, the deviation from the identity matrix is measured after normalization. IsoScore $\in [0, 1]$, where a value close to 1 represents an isotropic distribution, and a value close to 0 represents an anisotropic distribution.

### Theoretical Framework of Key Findings

This paper establishes the following correspondences:
- **Low ID + Low IsoScore** $\rightarrow$ Embeddings concentrate in a low-dimensional subspace $\rightarrow$ High redundancy $\rightarrow$ Robust to dimensionality reduction (classification, clustering).
- **High ID + High IsoScore** $\rightarrow$ Embeddings are uniformly distributed in high-dimensional space $\rightarrow$ Low redundancy $\rightarrow$ Sensitive to dimensionality reduction (retrieval, STS).

This aligns with the nature of the tasks: classification/clustering only need to distinguish finite categories, requiring less information; retrieval/STS need to capture fine-grained semantic similarity, requiring the preservation of more dimensions of information.

## Key Experimental Results

### Table 1: ID and IsoScore under different models and task prompts

| Prompt Type | gte-Qwen2 ID | gte-Qwen2 IsoScore | E5-mistral ID | E5-mistral IsoScore | SFR-2 ID | SFR-2 IsoScore |
|-----------|-------------|-------------------|--------------|--------------------|---------|----|
| Classification | 22.02 | .0052 | 22.26 | .0057 | 37.03 | .0077 |
| Clustering | 10.78 | .0058 | 13.01 | .0060 | 16.29 | .0138 |
| Retrieval (Query) | 31.90 | .0779 | 51.36 | .0761 | 81.38 | .1117 |
| Retrieval (Passage) | 35.94 | .0813 | 36.69 | .0332 | 35.07 | .0555 |
| STS | 38.47 | .0784 | 34.07 | .0439 | 41.69 | .0533 |

Key findings: The ID gap for instruction-based models between classification/clustering and retrieval/STS is **greater than 10**, and the IsoScore gap is about **10-fold**.

### Table 2: Comparison of ID and IsoScore of non-prompt models

| Model | Prompt | ID | IsoScore |
|------|--------|-----|----------|
| E5-small | query: | 41.57 | .4419 |
| E5-small | passage: | 37.60 | .3905 |
| E5-large | query: | 42.44 | .2022 |
| E5-large | passage: | 38.50 | .1977 |
| Unsup-SimCSE | — | 27.01 | .1611 |
| BERT (CLS) | — | 20.78 | .0186 |
| BERT (Mean) | — | 17.56 | .0973 |

Regardless of the prefix used, E5 models maintain high ID and IsoScore, indicating that they generate low-redundancy embeddings to adapt to multi-task demands. Contrastive learning (SimCSE vs BERT) significantly improves isotropy.

### Table 3: Average performance of gte-Qwen2 on various tasks after dimensionality reduction

| Dimensions | Classification (Acc) | Clustering (V-M) | Retrieval (nDCG@10) | STS (Spearman) |
|------|----------|----------|-------------|--------------|
| Full Dim (3584) | ~79 | ~47 | ~62 | ~85 |
| 512 | ~79 | ~46 | ~60.5 | ~83 |
| 128 | ~79 | ~46.2 | ~55 | ~78 |
| 32 | ~78 | ~42 | ~40 | ~65 |
| 8 | ~77 | ~35 | ~25 | ~50 |
| 2 | 76.34 | ~20 | <15 | <40 |

## Key Findings

1. **Feasibility of extreme dimensionality reduction**: Instruction-based models remain usable even when reduced to 2 dimensions for classification tasks—gte-Qwen2 with 2 dimensions (76.34) outperforms the full 1024-dimensional E5-large (75.69).
2. **Task-dependent redundancy**: Classification > Clustering >> Retrieval $\approx$ STS. Dimensionality reduction robustness is positively correlated with redundancy.
3. **Prompt-regulated geometric properties**: The same model generates totally different embedding geometric structures under different task prompts—classification prompts drive low ID/low IsoScore, while retrieval prompts drive high ID/high IsoScore.
4. **Model scale effect**: Larger models (LLM-based) have lower ID and lower IsoScore, but larger gaps between tasks, indicating that large models are better at customizing embeddings for specific tasks.
5. **Contrastive learning improves isotropy**: SimCSE and E5 exhibit higher ID and IsoScore than vanilla BERT, consistent with the known conclusion that contrastive learning enhances uniformity.
6. **Isotropy and ID stability post-reduction**: The ranking of IDs across tasks remains unchanged until dimensions are reduced to 128, and the task differences in IsoScore persist at low dimensions.

## Highlights & Insights

- **Simple yet effective research design**: Without training or special datasets, important patterns are revealed using only naive truncation and two geometric metrics.
- **High practical value**: Directly points out that embedding dimensions can be compressed significantly (50-200x) in classification/clustering scenarios, saving storage and computation resources.
- **First systematic analysis of the regulatory effect of prompts on embedding geometric properties**: Previous studies focused on performance comparison, whereas this work delves into the structural level of the embedding space.
- **Comprehensive coverage**: Involves 7 models, 20+ datasets, 4 types of tasks, and 5 dimensionality reduction methods, ensuring generalizability of the conclusions.

## Limitations & Future Work

- **Underlying causes remain unexplained**: The phenomenon where prompts alter the geometric properties of embeddings is validated, but the internal mechanisms within LLMs are not revealed.
- **Insufficient data-side analysis**: ID and IsoScore are only calculated on English Wikipedia, without considering the impact of variables such as text length, domain, or language.
- **Relatively simple dimensionality reduction methods**: Mainly uses naive truncation. Although it is verified that methods like PCA share consistent trends, better task-adaptive dimensionality reduction strategies are not explored.
- **Lack of end-to-end system evaluation**: The impact of post-reduction embeddings on real RAG or production systems (e.g., latency, throughput) is not evaluated.
- **Exclusion of models trained with MRL**: Matryoshka models specifically optimized for dimensionality reduction are not compared, leaving it unclear whether naive truncation already approaches the upper bound.

## Related Work & Insights

- **Matryoshka (Kusupati et al. 2022)**: Requires special mechanisms during training; this paper proves that instruction-based models can achieve significant dimensionality reduction without extra training.
- **Dinu et al. (2025)**: Studies the impact of temperature parameters on ID; this paper finds that LLM-based models naturally regulate ID through prompts.
- **Ait-Saada & Nadif (2023)**: Points out that improving isotropy is unfavorable for clustering; this paper confirms this from an experimental perspective—clustering embeddings are indeed more anisotropic.
- **Mickus et al. (2024)**: Theoretically demonstrates the trade-off between classification/clustering and isotropy; this paper provides large-scale experimental evidence.
- **SimCSE (Gao et al. 2021)**: Proves that contrastive learning improves isotropy and STS performance; this paper verifies that SimCSE-generated embeddings have higher ID and IsoScore.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to systematically quantify the redundancy and geometric property differences of prompt-based embeddings.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 7 models, 4 task categories, 20+ datasets, 5 dimensionality reduction methods, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich charts/tables, and explicit conclusions.
- Value: ⭐⭐⭐⭐ — Directly guides embedding compression for deployment and inspires understanding of prompt mechanisms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ICLR 2026\] Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement](../../ICLR2026/information_retrieval/let_llms_speak_embedding_languages_generative_text_embeddings_via_iterative_cont.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](../../ACL2026/information_retrieval/why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)

</div>

<!-- RELATED:END -->
