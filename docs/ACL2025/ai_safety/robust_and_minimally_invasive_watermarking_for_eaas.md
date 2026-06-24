---
title: >-
  [Paper Note] Robust and Minimally Invasive Watermarking for EaaS
description: >-
  [ACL 2025][AI Safety][Embedded Watermarking] Proposed ESpeW (Embedding-Specific Watermark), an embedding-specific watermarking method that injects unique watermarks at different positions of each embedding vector, achieving robust copyright protection for Embeddings as a Service (EaaS). It resists various watermark removal attacks while affecting the embedding quality by less than 1%.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "Embedded Watermarking"
  - "EaaS Copyright Protection"
  - "Model Extraction Attacks"
  - "Robust Watermarking"
  - "Embedding-as-a-Service"
date: 2026-05-08
content_hash: 3b4ca9c0badc57e3
---

# Robust and Minimally Invasive Watermarking for EaaS

**Conference**: ACL 2025  
**arXiv**: [2410.17552](https://arxiv.org/abs/2410.17552)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Embedded Watermarking, EaaS Copyright Protection, Model Extraction Attacks, Robust Watermarking, Embedding-as-a-Service  

## TL;DR

Proposed ESpeW (Embedding-Specific Watermark), an embedding-specific watermarking method that injects unique watermarks at different positions of each embedding vector, achieving robust copyright protection for Embeddings as a Service (EaaS). It resists various watermark removal attacks while affecting the embedding quality by less than 1%.

## Background & Motivation

### 1. Background
As LLMs' capability to generate embeddings grows, an increasing number of organizations provide EaaS (e.g., OpenAI, Mistral, Google). Users obtain high-quality embedding vectors via APIs to build downstream applications. However, EaaS faces severe threats from model extraction attacks, where attackers can replicate embedding models with comparable performance at a low cost simply by accessing the API.

### 2. Limitations of Prior Work
- **EmbMarker** (Peng et al., 2023): Injects watermark embeddings into target embeddings via linear interpolation, but all watermark embeddings share the same components, making them easy to identify and eliminate.
- **WARDEN** (Shetty et al., 2024a): Injects multiple watermarks to enhance strength, but also suffers from the shared component issue.
- **CSE Attack** (Shetty et al., 2024a): Effectively removes the aforementioned watermarks by detecting anomalous sample pairs and eliminating shared principal components.
- Core Problem: The watermarked embeddings of existing methods have a **common direction**, making the watermarks traceable.

### 3. Key Challenge
Watermarks need to be detectable for copyright verification but must not be easily recognized and removed by attackers. Existing methods fail to balance watermark detectability and removal resistance.

### 4. Goal
Design a watermarking method such that watermarked embeddings do not share common components (resisting CSE removal), its distance distribution from target embeddings does not deviate from the original distribution (resisting anomaly detection), and it has a minimal impact on embedding quality.

### 5. Key Insight
Leverage the **high dimensionality and sparsity** of LLM embeddings by replacing only a small proportion of dimensions with the smallest absolute values (the least important positions) in each embedding. Different embeddings have different replaced positions, making the watermark "embedding-specific".

### 6. Core Idea

**Select the proportion $\alpha$ of dimensions with the smallest absolute values in each embedding vector and replace them with the target embedding values. Different embeddings have different replacement positions, ensuring no shared components among watermarked embeddings and making their distributions indistinguishable.**

## Method

### Overall Architecture

ESpeW consists of two phases:
1. **Watermark Injection**: Injecting a personalized watermark before returning embeddings to the user.
2. **Watermark Verification**: Verifying copyright through statistical hypothesis testing.

### Key Designs

#### Watermark Injection

1. **Select trigger word set** $T = \{t_1, t_2, ..., t_n\}$ (medium-frequency words) and target embedding $\boldsymbol{e}_t$.

2. **Construct position mask**: For the embedding $\boldsymbol{e}_o$ of a sentence containing trigger words, select the proportion $\alpha$ of dimensions with the smallest absolute values:

$$\mathcal{I}_\alpha = \text{argsort}(|\boldsymbol{e}_o|)[:\alpha|\boldsymbol{e}_o|]$$

$$\boldsymbol{M}[i] = \begin{cases} 1 & \text{if } i \in \mathcal{I}_\alpha \\ 0 & \text{otherwise} \end{cases}$$

3. **Partial replacement**: Replace only the selected positions with the target embedding values:

$$\boldsymbol{e}_p' = \boldsymbol{e}_o * (1 - \boldsymbol{M}) + \boldsymbol{e}_t * \boldsymbol{M}$$

4. **Normalization**: $\boldsymbol{e}_p = \boldsymbol{e}_p' / \|\boldsymbol{e}_p'\|_2$

**Core Advantages**:
- The replacement positions for each embedding are different (depending on their respective absolute value ranking), so there are no shared components among watermarked embeddings.
- Only the minimum absolute value positions are replaced, minimizing the impact on embedding quality.

#### Watermark Verification

Construct a backdoor dataset $D_b$ (containing trigger words) and a benign dataset $D_n$ (without trigger words), and calculate the difference in their cosine similarity to the target embedding:

$$\Delta\cos = \frac{1}{|C_b|}\sum_{i \in C_b} i - \frac{1}{|C_n|}\sum_{j \in C_n} j$$

$$\Delta l_2 = \frac{1}{|L_b|}\sum_{i \in L_b} i - \frac{1}{|L_n|}\sum_{j \in L_n} j$$

Use the Kolmogorov-Smirnov (KS) test to determine whether the two distributions are significantly different. A $p$-value $< 10^{-4}$ determines the model as a stolen version.

### Loss & Training
- The embedding model uses OpenAI GPT-3 text-embedding-002.
- The stealer uses BERT-Base-Cased + a two-layer MLP.
- The watermark ratio $\alpha$ is the only hyperparameter, with a recommended range of 15%-35%.

## Key Experimental Results

### Main Results: Copyright Verification Under Different CSE Intensities on SST2

| CSE Intensity K | Method | ACC(%) | p-value↓ | Δcos(%)↑ | COPY? |
|-----------|------|--------|---------|---------|-------|
| No CSE | EmbMarker | 93.46 | $<10^{-11}$ | 9.71 | ✓ |
| No CSE | WARDEN | 94.04 | $<10^{-11}$ | 12.18 | ✓ |
| No CSE | **ESpeW** | 93.46 | $<10^{-10}$ | 6.46 | ✓ |
| K=50 | EmbMarker | 90.51 | >0.01 | 12.28 | **✗** |
| K=50 | WARDEN | 89.85 | >0.08 | 6.38 | **✗** |
| K=50 | **ESpeW** | 86.73 | $<10^{-11}$ | 65.11 | **✓** |
| K=100 | EmbMarker | 90.19 | >0.01 | 12.66 | **✗** |
| K=100 | **ESpeW** | 84.66 | $<10^{-11}$ | 64.46 | **✓** |
| K=1000 | EmbMarker | 85.29 | >0.35 | -2.52 | **✗** |
| K=1000 | **ESpeW** | 73.57 | $<10^{-11}$ | 49.38 | **✓** |

**ESpeW is the only method that correctly verifies copyright under all CSE intensities.**

### Impact on Embedding Quality

| Method | Cosine Similarity Change |
|------|-------------|
| EmbMarker | ~92-95% |
| WARDEN | ~90-93% |
| ESpeW (Positions with Minimum Magnitude) | **>99%** |
| ESpeW (Random Positions) | ~98% |

ESpeW affects embedding quality by **<1%**, significantly outperforming all baselines.

### Ablation Study

The impact of $\alpha$ (without CSE):
- $\alpha = 15\%$: The minimum threshold for successful watermark injection.
- $\alpha \leq 35\%$: Watermark embeddings are indistinguishable in PCA visualization.
- $\alpha = 100\%$: Equivalent to complete replacement, degrading to EmbMarker.

### Key Findings
1. **The stronger the CSE intensity, the stronger ESpeW's detection capability**—because the removal operation instead amplifies the watermark signal.
2. **Dropout Attack**: A dropout rate of 0.7-0.8 is required to destroy the watermark, but by then the embeddings themselves are unusable.
3. ESpeW highly overlaps with non-watermarked embeddings in terms of cosine similarity distribution, making anomaly detection methods unable to identify watermarked embeddings.
4. Consistently effective across four datasets: SST2, MIND, AG News, and Enron Spam.

## Highlights & Insights

1. **"Embedding specificity" is the core innovation**—the concept of injecting watermarks at different positions for different embeddings is simple yet highly effective, fundamentally solving the removability issue caused by shared components.
2. **Leveraging high-dimensional sparsity**: LLM embeddings contain a large number of near-zero dimensions that can be safely replaced, a highly practical observation.
3. **Minimalist design**: Only one hyperparameter $\alpha$, requiring no complex optimization processes.
4. **Counter-intuitive conclusion**: Stronger CSE attacks actually enhance ESpeW's detection capability (because destroying the quality of non-watermark parts of the embedding makes the watermark signal stand out even more).

## Limitations & Future Work

1. **Efficiency bottleneck**: Finding the $K$ positions with the smallest absolute values requires a sorting operation, which may become a computational bottleneck in ultra-high-dimensional embeddings and high-concurrency scenarios.
2. The target embedding $\boldsymbol{e}_t$ must be kept confidential—if leaked, attackers could design targeted removal strategies.
3. Only verified on GPT-3 text-embedding-002; the applicability to other embedding models (e.g., E5, BGE) has not been fully explored.
4. Randomly selecting watermark positions can resolve the efficiency issue, but it increases the impact on embedding quality from <1% to ~2%.

## Related Work & Insights

- **EmbMarker** (Peng et al., 2023): The direct improvement target of ESpeW, evolving from global linear interpolation $\rightarrow$ partial replacement.
- **CSE Attack** (Shetty et al., 2024a): The watermark removal method that ESpeW is specifically designed against.
- **Model Extraction Attacks** (Liu et al., 2022): The threat model for studies on EaaS copyright protection.
- **Insights**: The concept of leveraging sparsity in the embedding space for information hiding can potentially be extended to other embedding protection scenarios (such as knowledge base protection in RAG systems).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of embedding-specific watermarking is simple and effective, fully exploiting high-dimensional sparse features.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 datasets, multiple attack intensities, ablation analysis, visualization, and various anti-attack tests.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, intuitive figures (the distribution comparison figures are very convincing), and thorough analysis.
- **Value**: ⭐⭐⭐⭐ — EaaS copyright protection is a practical problem, and the proposed method is simple, practical, and highly robust.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](../../CVPR2026/ai_safety/recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](../../CVPR2026/ai_safety/clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)
- [\[CVPR 2026\] Meta-FC: Meta-Learning with Feature Consistency for Robust and Generalizable Watermarking](../../CVPR2026/ai_safety/meta-fc_meta-learning_with_feature_consistency_for_robust_and_generalizable_wate.md)
- [\[ACL 2025\] Sandcastles in the Storm: Revisiting Watermarking Impossibility](sandcastles_watermarking_impossibility.md)

</div>

<!-- RELATED:END -->
