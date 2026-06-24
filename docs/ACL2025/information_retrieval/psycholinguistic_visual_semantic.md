---
title: >-
  [Paper Note] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space
description: >-
  [Information Retrieval & RAG] Proposes the Neighborhood Stability Measure (NSM)—an unsupervised, distribution-free method that estimates word imageability and concreteness by quantifying the stability of neighborhoods in the text embedding space, outperforming existing approaches that rely on multimodal or generative models while using only the text modality.
tags:
  - "Information Retrieval & RAG"
date: 2026-05-08
content_hash: 805cb806c5f1cd1f
---

# Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space

## Paper Information

- **Conference**: ACL 2025
- **arXiv**: [2505.23029](https://arxiv.org/abs/2505.23029)
- **Code**: [https://github.com/Artificial-Memory-Lab/imageability](https://github.com/Artificial-Memory-Lab/imageability)
- **Area**: Information Retrieval / Psycholinguistics
- **Keywords**: Imageability, Concreteness, Text Embedding, Neighborhood Stability, Unsupervised, Psycholinguistics

## TL;DR

Proposes the Neighborhood Stability Measure (NSM)—an unsupervised, distribution-free method that estimates word imageability and concreteness by quantifying the stability of neighborhoods in the text embedding space, outperforming existing approaches that rely on multimodal or generative models while using only the text modality.

## Background & Motivation

- **Background**: Imageability (the capacity of a word to arouse mental imagery) and concreteness (the degree to which a word refers to a perceptible entity) are key psycholinguistic properties linking visual and semantic spaces. Traditionally, ratings are acquired via human surveys, which are highly expensive.
- **Data Scarcity**: The MRC Psycholinguistic Database covers imageability ratings for only 4,848 words; Brysbaert et al. (2014) expanded concreteness ratings to 37,058 words via crowdsourcing, but coverage remains limited.
- **Limitations of Prior Work**: Wu & Smith (2023) estimated imageability using text-to-image models, which is computationally prohibitive (requiring 120 GPU hours to process the entire vocabulary). Hessel et al. (2018) estimated concreteness using image-caption datasets, but this covers only about 2% of the rated vocabulary and suffers from vocabulary mismatch issues.
- **Key Insight**: Text within image-caption datasets inherently contains sufficient signals to estimate these properties—the neighborhood structure (peak sharpness) of concrete/imageable words in the embedding space exhibits systematic differences compared to that of abstract words.
- **Core Idea**: Develop an unsupervised method that utilizes only a single text modality, requires no generative models, is computationally highly efficient, and achieves 100% coverage.

## Method

### Overall Architecture

The NSM method is based on a three-step pipeline: (1) use a text embedding model to convert captions from image-caption datasets into a set of vectors; (2) retrieve the $k$-nearest neighbors for a query word in the embedding space to form its neighborhood; (3) compute the neighborhood stability measure—the proportion of points in the neighborhood whose own nearest neighbors also belong to this neighborhood. The vector set needs to be constructed only once and can be reused indefinitely.

### Datasets and Embedding Models Used

- **Datasets**: MS COCO (1.5M captions), CC3M (3.3M), CC12M (12M)—utilizing only the caption text portion.
- **Embedding Models**: AllMiniLM (384D, 33M parameters), Gte-Base (768D, 137M parameters), Gte-Large (1024D, 434M parameters).
- **Rating Data**: MRC Psycholinguistic Database (4,848 words for imageability) and Brysbaert et al. (37,058 words for concreteness).

### Key Designs

1. **Core Hypothesis (Hypothesis 1)**: In semantic space, the distribution of contexts surrounding concrete/imageable words forms sharper peaks—meaning their embedding neighborhoods are more "stable" (closer to their neighbors and more separable), whereas abstract words have more dispersed neighborhoods that overlap with other regions. t-SNE visualizations provide initial support for this hypothesis.
2. **$\alpha$-Stable Neighborhood Definition**: The $\alpha$-stability of a neighborhood is the proportion of points within the neighborhood whose nearest neighbor also belongs to the same neighborhood. An $\alpha$ closer to 1 indicates a more stable neighborhood and a more concrete/imageable word. This concept is generalized from "natural neighbors" (where two points are mutual nearest neighbors).
3. **Efficient Implementation**: Leveraging approximate nearest neighbor (ANN) search (via the IVF index in the Faiss library), the nearest neighbor mapping for each point is precomputed. This reduces the algorithm complexity from $O(kT)$ to $O(T)$, where $T$ is the cost of a single ANN query.
4. **Dataset Choice Considerations**: The proposed method uses the text portion of image-caption datasets rather than general text corpora, because imageability and concreteness are visual-semantic properties that require a "visual region of the semantic space" for accurate estimation.

## Experiments

### Main Results — Comparison of Spearman Correlation with Baselines

| Method | Imageability ↑ | Concreteness ↑ | Coverage | Requires Visual Modality |
|------|----------|---------|--------|------------|
| Freq (CC12M) | 0.34 | 0.35 | 98.0% | No |
| HML (MS COCO) | 0.49 | 0.45 | ~2.7% | Yes (Image-Caption Pairs) |
| CosineSim | 0.45 | 0.40 | 100% | Yes (Generative Models) |
| AvgClip | 0.56 | 0.45 | 100% | Yes (Generative Models) |
| **NSM-AllMiniLM(CC12M)** | **0.66** | **0.58** | 100% | **No** |
| NSM-Gte-Base(CC12M) | 0.58 | 0.58 | 100% | No |

- NSM achieves the highest correlation coefficients for both imageability and concreteness, entirely without requiring any visual modality.

### Ablation Study and In-depth Analysis

| Dimension of Analysis | Findings |
|---------|------|
| Dataset Scale | Larger text corpora consistently improve performance: CC12M > CC3M > MS COCO |
| Embedding Dimension | Low-dimensional embeddings (384D AllMiniLM) outperform high-dimensional ones (1024D Gte-Large), as the distance concentration effect (curse of dimensionality) in high-dimensional spaces weakens neighborhood structures |
| Hyperparameter $k$ | The neighborhood radius $k$ is tuned within the range of $[64, 4096]$ via the validation set, showing a moderate impact on the results |
| Word Frequency vs. NSM | Word frequency in image-caption datasets is itself a strong baseline (outperforming previously reported results), yet NSM still significantly surpasses it |
| Computational Efficiency | NSM requires constructing the embedding vector set only once for repeated use, whereas AvgClip requires 120 GPU hours of computation |

### Key Findings

1. The distributional structure of the text embedding space itself encodes imageability and concreteness information—without requiring visual modalities.
2. There is a strong correlation between neighborhood stability and psycholinguistic properties, confirming Hypothesis 1.
3. The curse of dimensionality is an important factor affecting NSM performance, with low-dimensional embeddings being more conducive to capturing variations in neighborhood structure.

## Highlights & Insights

- Clear theoretical contribution: Proposes and validates the hypothesis that "neighborhood stability in embedding space reflects psycholinguistic properties."
- Simple and efficient methodology: Inspired by approximate nearest neighbor (ANN) search literature, the algorithm is easy to implement and scalable to large datasets.
- Rigorous experimental design: Computes averages over 10 random seeds, uses an 80/20 validation-test split, and performs comprehensive cross-evaluation across multiple embedding models and datasets.
- Outperforms visual-dependent methods using only a single text modality, challenging the implicit assumption that "visual information is required to estimate visual properties."
- The vector set can be constructed once and reused indefinitely, far superior to AvgClip's computational cost of 120 GPU hours per word.

## Limitations & Future Work

- The method relies on image-caption datasets (such as MS COCO, CC3M, etc.); its effectiveness may decrease when applied to general natural text corpora.
- Only word-level properties in English are evaluated, without validating generalization to phrase or sentence levels, or to other languages.
- It has an implicit dependency on the choice and quality of embedding models; more robust embedding models could potentially further enhance performance.
- The selection of hyperparameter $k$ requires tuning on a validation set and may need readjustment for new domains.
- The theoretical analysis remains empirical, lacking a formal mathematical proof of the relationship between neighborhood stability and psycholinguistic properties.

## Related Work & Insights

- **Imageability Estimation**: Wu & Smith (2023) estimate imageability using text-to-image models combined with CLIP scoring, which is computationally expensive. The MRC Psycholinguistic Database (Coltheart, 1981) provides human ratings for 4,848 words.
- **Concreteness Estimation**: Hessel et al. (2018) measure the clustering of images corresponding to words based on image-caption pairs. Brysbaert et al. (2014) expanded crowdsourced ratings to 37,058 words.
- **Distributional Semantics**: Frassinelli et al. (2017) and Schulte im Walde & Frassinelli (2022) study abstract/concrete differences from word co-occurrence distribution profiles, but are based on traditional lexical features rather than learned embedding representations.
- **Supervised Approaches**: Tater et al. (2024) use supervised classifiers based on visual features; Charbonnier & Wartena (2019) train regression models on word embeddings. In contrast, the proposed method (Ours) is entirely unsupervised and requires no labeled data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The hypothesis is highly novel and thoroughly validated. The cross-domain inspiration drawing from ANN literature to psycholinguistics is highly impressive.
- **Value**: ⭐⭐⭐⭐ — Simple and efficient methodology. Constructing the vector set once allows for infinite reuse, making it considerably more practical than generative approaches.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full cross-evaluation across 3 datasets and 3 embedding models, featuring complete comparisons with a wide variety of baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Extremely smooth narrative logic flowing from intuition to formalization and algorithm design. Exceptional readability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](reidentification_deidentified.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](../../ICCV2025/information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](semantic_outlier_removal_with_embedding_models_and_llms.md)

</div>

<!-- RELATED:END -->
