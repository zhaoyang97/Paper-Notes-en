---
title: >-
  [Paper Note] Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score
description: >-
  [AAAI 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper proposes Spectrum Projection Score (SPS), a training-free metric that evaluates retrieval summary quality by measuring the alignment bet…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Perplexity"
  - "Spectrum Projection Score"
  - "Summary Compression"
  - "Representation Space Alignment"
date: 2026-05-08
content_hash: a8d97390d70d0238
---

# Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score

**Conference**: AAAI 2026
**arXiv**: [2508.05909](https://arxiv.org/abs/2508.05909)  
**Code**: [https://zhanghao-aaai2026-sps.github.io/AAAI2026-SPS/](https://zhanghao-aaai2026-sps.github.io/AAAI2026-SPS/)  
**Area**: RAG / Information Retrieval
**Keywords**: Retrieval-Augmented Generation, Perplexity, Spectrum Projection Score, Summary Compression, Representation Space Alignment

## TL;DR
This paper proposes Spectrum Projection Score (SPS), a training-free metric that evaluates retrieval summary quality by measuring the alignment between summary token embeddings and the principal subspace of the reader LLM, serving as a replacement for conventional perplexity-based metrics. Combined with the xCompress inference-time controller, SPS achieves substantial improvements over perplexity-based methods across 5 QA datasets (HotpotQA EM +3.6).

## Background & Motivation

**Background**: In RAG, retrieved documents must be compressed or summarized into more compact forms before being fed into the reader LLM. Existing approaches commonly use perplexity (PPL) to assess summary quality, under the assumption that lower perplexity implies higher quality.

**Limitations of Prior Work**: PPL measures token "typicality" rather than semantic alignment — a summary may be fluent yet poorly aligned with the reader's representation space. Empirical analysis reveals that the correlation between PPL and downstream QA performance is near zero or even negative.

**Key Challenge**: A high-quality summary is not one that the reader can easily generate, but one that enables the reader's internal representation space to fully capture the contained information.

**Goal**: Design a training-free summary quality metric that is highly correlated with downstream QA performance, and build an inference-time summary selection/compression framework upon it.

**Key Insight**: Starting from the reader LLM's embedding matrix, SVD decomposition is applied to obtain the model's "principal representation subspace," and the residual of a summary outside this subspace is measured.

**Core Idea**: Summary quality equals the projection residual of its token embeddings onto the reader's principal subspace — the smaller the residual, the better.

## Method

### Overall Architecture
xCompress is an inference-time controller consisting of three steps: (1) generate $K$ candidate summaries via stochastic decoding; (2) score and select the best candidate using SPS; (3) apply adaptive norm filtering to detect whether the initial summary is already of sufficient quality, bypassing additional sampling when appropriate.

### Key Designs

1. **Spectrum Projection Score (SPS)**:

    - **Function**: Measures the alignment between a summary and the reader's representation space.
    - **Mechanism**: SVD is applied to the reader's embedding matrix $W \in \mathbb{R}^{D \times M}$: $W = U\Sigma V^\top$. The top singular components accounting for 95% of the cumulative singular values are retained to form the projection $P = U\Sigma_p V^\top$. Token representations from the penultimate layer of the reader are extracted for the summary, and element-wise max pooling is applied to obtain a "boundary vector" $\mathbf{x}$. SPS is then computed as $\text{SPS}(\mathbf{x}) = \|(I - P)\mathbf{x}\|_2$.
    - **Design Motivation**: Max pooling captures the convex hull boundary of the token distribution (the "most salient" dimensions), while the residual norm quantifies the amount of information lying outside the principal subspace — a lower value indicates the summary falls more firmly within the reader's "comfort zone."

2. **Adaptive Norm-Guided Filtering**:

    - **Function**: Avoids redundant sampling for summaries that are already of sufficient quality.
    - **Mechanism**: The ratio $L2_{\text{mean}}/L1_{\text{max}}$ is computed as a concentration indicator; if this ratio exceeds the top-30% threshold on the training set, the initial summary is used directly.
    - **Design Motivation**: Reduces computational overhead while maintaining accuracy.

### Loss & Training
This is a training-free method. SPS relies solely on a one-time SVD decomposition of the reader's embedding matrix and inference-time extraction of token embeddings.

## Key Experimental Results

### Main Results

| Model + Method | HotpotQA EM/F1 | 2WikiMQA EM/F1 | NQ EM/F1 |
|------------|---------------|---------------|---------|
| LLaMA 3.1 8B + CompAct | 34.0/43.2 | 27.2/31.8 | 35.2/47.5 |
| LLaMA 3.1 8B + **xCompress** | **37.6/47.9** | **29.6/34.2** | **39.4/51.2** |
| Gemma 3 12B + CompAct | 19.2/29.7 | 23.8/29.0 | 27.6/40.9 |
| Gemma 3 12B + **xCompress** | **25.2/35.7** | **25.0/29.3** | **31.6/42.3** |

### Metric Correlation Analysis

| Dataset | PPL-PCC(EM) | SPS-PCC(EM) | PPL-PCC(F1) | SPS-PCC(F1) |
|--------|------------|------------|------------|------------|
| HotpotQA | 0.022 | **0.643** | -0.067 | **0.753** |
| NQ | 0.202 | **0.650** | 0.452 | **0.628** |
| 2Wiki | -0.318 | **0.557** | 0.295 | **0.503** |

### Key Findings
- PPL exhibits near-zero or negative PCC with QA performance, whereas SPS achieves PCC values in the range of 0.4–0.75.
- Max pooling outperforms mean pooling and last-token pooling (HotpotQA EM: 37.6 vs. 36.2 vs. 33.6).
- Penultimate-layer embeddings yield the best results — the final layer is overly specialized for next-token prediction.
- Performance saturates with as few as 5 candidate summaries.

## Highlights & Insights
- **The finding that perplexity is ineffective for RAG is itself significant**: it challenges a widely adopted assumption, and the PCC evidence presented is compelling.
- **The geometric intuition behind SVD principal subspaces**: the reader's "comprehension capacity" is formalized as a linear subspace, and summary quality is equated with "representability" within that space.

## Limitations & Future Work
- The linear subspace assumption may be overly simplistic — the reader's representation space may constitute a nonlinear manifold.
- Validation is limited to QA tasks; effectiveness on other RAG applications such as summarization and translation remains unexplored.
- Generating multiple candidate summaries introduces additional inference latency and computational cost.

## Related Work & Insights
- **vs. CompAct**: xCompress performs post-hoc selection over CompAct summaries without modifying the compression method itself, making it a plug-and-play solution.
- **vs. LongPPL**: LongPPL also attempts to improve upon PPL but remains within the perplexity framework, exhibiting similarly near-zero correlation with QA performance.

## Rating
- Novelty: ⭐⭐⭐⭐ SPS redefines summary quality from a representation space geometry perspective, offering a genuinely novel viewpoint.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 5 datasets, 4 LLMs, with detailed ablation studies and correlation analyses.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and the geometric intuition is well articulated.
- Value: ⭐⭐⭐⭐ Makes an important methodological contribution to summary evaluation in RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](../../ICLR2026/information_retrieval/summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](../../CVPR2026/information_retrieval/beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](../../ACL2026/information_retrieval/beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](../../ACL2026/information_retrieval/beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->
