---
title: >-
  [Paper Note] Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding
description: >-
  [AAAI 2026][Similarity Metrics] This paper proposes two parameter-free, magnitude-aware vector similarity metrics—Overlap Similarity (OS) and Hyperbolic Tangent Similarity (HTS)—that achieve significantly lower MSE than Cosine Similarity and Dot Product on classification tasks (paraphrase detection, natural language inference) across 4 sentence embedding models and 8 NLP benchmarks, without any additional training overhead.
tags:
  - AAAI 2026
  - Similarity Metrics
  - Sentence Embeddings
  - Anisotropy
  - Overlap Similarity
  - Hyperbolic Tangent Similarity
date: 2026-05-08
content_hash: 76e664213954c1a2
---

# Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding

**Conference**: AAAI 2026
**arXiv**: [2509.19323](https://arxiv.org/abs/2509.19323)
**Code**: To be released (MIT License)
**Area**: Information Retrieval
**Keywords**: Similarity Metrics, Sentence Embeddings, Anisotropy, Overlap Similarity, Hyperbolic Tangent Similarity

## TL;DR

This paper proposes two parameter-free, magnitude-aware vector similarity metrics—Overlap Similarity (OS) and Hyperbolic Tangent Similarity (HTS)—that achieve significantly lower MSE than Cosine Similarity and Dot Product on classification tasks (paraphrase detection, natural language inference) across 4 sentence embedding models and 8 NLP benchmarks, without any additional training overhead.

## Background & Motivation

**State of the Field**: Sentence embedding similarity comparison has long been dominated by Cosine Similarity, which is computationally efficient, geometrically intuitive (measuring only directional angle), and insensitive to vector magnitude.

**Limitations of Prior Work**: Embeddings produced by modern pretrained language models (e.g., BERT, sentence-transformers) occupy highly anisotropic spaces, with all vectors concentrated in a narrow cone. In this regime, Cosine Similarity loses discriminative power—semantically unrelated sentences can still receive high similarity scores. Moreover, Cosine completely discards vector magnitude information, which may encode semantic specificity or importance.

**Root Cause**: Dot Product is overly sensitive to magnitude (unbounded, susceptible to non-semantic factors such as sentence length), while Cosine ignores magnitude entirely (failing in anisotropic spaces). A metric that strikes a balance between these two extremes—integrating magnitude information in a controlled manner—is needed.

**Paper Goals**: Design parameter-free, drop-in replacement similarity functions that better capture semantic similarity than Cosine without incurring any additional training cost.

**Starting Point**: Rather than modifying the embedding space (e.g., retraining with contrastive learning as in SimCSE), the paper directly replaces the similarity formula applied to existing embeddings with a more robust alternative.

**Core Idea**: Incorporate vector magnitude into similarity computation in a controlled manner via relational normalization (OS) and nonlinear compression (HTS), thereby overcoming the discriminative bottleneck of Cosine Similarity in anisotropic spaces.

## Method

### Overall Architecture

The approach is purely post-hoc—given a pair of sentence embeddings $(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d$ produced by a pretrained model, it outputs a scalar similarity score. No model training or parameter tuning is involved; only the similarity computation formula is replaced.

### Key Designs

1. **Overlap Similarity (OS)**

   - **Function**: Integrates magnitude information while performing robust normalization.
   - **Mechanism**: $\text{sim}_{OS}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - |\mathbf{x} \cdot \mathbf{y}| + \epsilon}$. The denominator is analogous to the inclusion-exclusion principle in set theory: it computes the total "energy" of the two vectors ($\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$) and subtracts their shared component ($|\mathbf{x} \cdot \mathbf{y}|$), effectively measuring the "union" of the two vectors' energies.
   - **Design Motivation**: In contrast to Cosine, which normalizes by $\|\mathbf{x}\| \cdot \|\mathbf{y}\|$ (depending solely on each vector's independent properties), the normalization factor in OS depends on the relationship between the vectors. This yields more stable and meaningful scores when all vectors already point in similar directions (anisotropy).

2. **Hyperbolic Tangent Similarity (HTS)**

   - **Function**: Achieves magnitude-aware, bounded similarity measurement through nonlinear transformation.
   - **Mechanism**: $\text{sim}_{HTS}(\mathbf{x}, \mathbf{y}) = \tanh\left(2 \cdot \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 + \epsilon}\right)$. The dot product is first normalized by the sum of squared norms, then compressed into $[-1, 1]$ via $\tanh$.
   - **Design Motivation**: (1) The S-shaped curve of $\tanh$ amplifies differences in the intermediate range and compresses extreme values, providing robustness to outliers; (2) The relationship between semantic similarity and vector similarity may be inherently nonlinear, and $\tanh$ may more closely reflect human perception.

3. **Evaluation Protocol**

   - **Function**: Zero-shot application of four metrics on fixed embeddings, evaluated using both MSE and Spearman $\rho$.
   - **Mechanism**: MSE measures absolute error; Spearman $\rho$ measures rank correlation. Statistical significance is assessed via the Wilcoxon signed-rank test, and bootstrapped 95% confidence intervals quantify the magnitude of improvement.
   - **Design Motivation**: A single metric may be misleading; the combination of dual metrics and statistical testing ensures reliability of conclusions.

### Loss & Training

No training is involved—the core method is a purely parameter-free substitution of mathematical formulas. The only preprocessing step is normalizing gold-standard scores to $[0, 1]$. Experiments were conducted on an NVIDIA RTX 4070 GPU (6 GB VRAM) using Python 3.8.19, PyTorch 2.4.1, and Sentence-Transformers 3.2.1. The core evaluation pipeline (embedding generation and similarity computation) is deterministic; each result is a single reproducible run, with bootstrap resampling performed 1,000 times.

## Key Experimental Results

### Main Results (MSE Comparison, all-mpnet-base-v2)

| Dataset | Dot Product | Cosine | OS | HTS |
|---------|------------|--------|----|-----|
| GLUE-STSB | 0.1916 | 0.1916 | 0.1732 | 0.1875 |
| SICK | 0.0316 | 0.0316 | 0.0773 | 0.0490 |
| Quora | 0.2487 | 0.2487 | **0.1773*** | **0.2067*** |
| PAWS | 0.5109 | 0.5109 | **0.4765*** | **0.3347*** |
| SNLI | 0.1872 | 0.1872 | **0.1627*** | **0.1791*** |
| MultiNLI | 0.2295 | 0.2295 | **0.1804*** | **0.2102*** |
| STS16 | 0.0593 | 0.0593 | **0.0414*** | **0.0533*** |

*Asterisks denote significance relative to both Cosine and Dot Product ($p < 0.05$)*

### Ablation Study (Cross-Model Consistency, PAWS MSE)

| Model | Cosine | OS | HTS | OS Relative Gain |
|-------|--------|----|-----|-----------------|
| all-mpnet-base-v2 | 0.5109 | 0.4765 | 0.3347 | 6.7% |
| all-MiniLM-L6-v2 | 0.5179 | 0.4880 | 0.3372 | 5.8% |
| bge-large-en-v1.5 | 0.5243 | 0.4968 | 0.3393 | 5.2% |
| paraphrase-mpnet-base-v2 | 0.5232 | 0.4955 | 0.3387 | 5.3% |

### Key Findings

- **Strong task dependence**: OS and HTS significantly outperform Cosine on classification tasks (Quora, PAWS, SNLI, MultiNLI), but Cosine performs comparably or better on fine-grained semantic regression tasks (SICK, GLUE-STSB).
- **Identical Spearman $\rho$ across all metrics**: All four metrics yield identical Spearman rank correlation coefficients on any given model–dataset pair, indicating that the improvement stems from calibration of absolute scores rather than changes in ranking.
- **Largest gain for HTS on PAWS**: MSE drops from 0.51 to 0.33 (~35% relative improvement), suggesting that the decision-boundary amplification effect of $\tanh$ is particularly effective for paraphrase detection.
- **Dot Product MSE explosion on paraphrase-mpnet-base-v2**: MSE reaches 37–79 due to the large-magnitude embeddings produced by this model, validating the fragility of unnormalized metrics.

## Highlights & Insights

- **Zero-cost drop-in improvement**: Replacing a single mathematical formula—without retraining any model—yields significant performance gains on classification tasks, presenting an extremely low engineering barrier. Any system using sentence-transformers can benefit immediately.
- **Insight from equal Spearman but unequal MSE**: This reveals that when the embedding space is approximately isotropic (uniform norms), all metrics are rank-equivalent but differ in absolute calibration. For applications requiring calibrated scores (e.g., threshold-based decision making), the choice of metric is critical.
- **Relational normalization in OS**: The denominator depends on the inter-vector relationship rather than on each vector's independent properties—a design principle transferable to other settings involving comparison of high-dimensional representations, such as retrieval, clustering, and temperature scaling in contrastive learning.
- **Decision-boundary amplification in HTS**: The S-shaped $\tanh$ curve amplifies differences in the intermediate range, making it particularly effective for tasks like PAWS that require binary discrimination based on degree of semantic difference.

## Limitations & Future Work

- Validation is limited to English BERT-family models; multilingual models and decoder-only large language models (e.g., LLaMA, GPT) are not evaluated. The embedding space geometry of such models may differ substantially.
- Performance degrades on SICK and STS-B, suggesting that magnitude information may introduce noise in fine-grained semantic regression tasks. An adaptive mechanism for automatic metric selection based on task and model characteristics is needed.
- Identical Spearman $\rho$ across all metrics implies that genuine ranking improvements may require more fundamental modifications to the embedding space (e.g., isotropy-inducing post-processing).
- The combination of OS/HTS with contrastive learning training objectives (e.g., SimCSE) is not explored.
- Theoretical analysis is absent: no error bounds or optimality conditions for OS/HTS under specific embedding distribution assumptions are provided.
- The role of $\epsilon$ is limited to numerical stability; its sensitivity is not investigated.
- End-to-end evaluation in practical retrieval systems (e.g., RAG pipelines) is not conducted; assessment is restricted to offline benchmarks.

## Related Work & Insights

- **vs. SimCSE (Gao et al. 2021)**: SimCSE retrains embeddings via contrastive learning to improve isotropy and then applies Cosine; this paper leaves the embedding space unchanged and replaces the metric. The two approaches are complementary—OS/HTS could be applied on top of SimCSE-trained embeddings.
- **vs. WhiteningBERT (Huang et al. 2021)**: Whitening normalizes embedding dimensions via data-dependent statistical transformations; OS/HTS are entirely data-agnostic and substantially simpler.
- **vs. L2 Distance (Tessari et al. 2025)**: L2 distance outperforms Cosine in in-context learning retrieval, also highlighting the importance of magnitude; however, L2 is unbounded, whereas OS/HTS are bounded and thus better suited as similarity metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The proposed metrics are mathematically simple, but the combination of magnitude-awareness, parameter-free design, and drop-in usability is a precise and valuable contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 models × 8 datasets with full statistical significance testing and bootstrap confidence intervals.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and experimental design is rigorous, though theoretical analysis is limited (no convergence or error bounds).
- **Value**: ⭐⭐⭐⭐ A practical improvement to semantic similarity computation; zero-cost deployment is the primary selling point.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[ICLR 2026\] Mapping Semantic & Syntactic Relationships with Geometric Rotation](../../ICLR2026/information_retrieval/mapping_semantic_syntactic_relationships_with_geometric_rotation.md)
- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/information_retrieval/external_knowledge_injection_for_clip-based_class-incremental_learning.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](../../CVPR2026/information_retrieval/beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)

<!-- RELATED:END -->
