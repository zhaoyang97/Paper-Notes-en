---
title: >-
  [Paper Note] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service
description: >-
  [AAAI 2026][AI Safety][EaaS Copyright Protection] This paper proposes RegionMarker, a watermarking framework triggered by semantic regions. By defining trigger regions in a low-dimensional space and injecting semantic watermarks, it represents the first EaaS copyright protection method capable of simultaneously defending against CSE attacks, paraphrasing attacks, and dimensional perturbation attacks.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "EaaS Copyright Protection"
  - "Embedding Watermarking"
  - "Semantic Region Triggering"
  - "Model Extraction Attack Defense"
  - "Locality-Sensitive Hashing"
date: 2026-05-08
content_hash: 33ff777b7a042e69
---

# RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service

**Conference**: AAAI 2026  
**arXiv**: [2511.13329](https://arxiv.org/abs/2511.13329)  
**Code**: Not publicly available  
**Area**: AI Security  
**Keywords**: EaaS Copyright Protection, Embedding Watermarking, Semantic Region Triggering, Model Extraction Attack Defense, Locality-Sensitive Hashing

## TL;DR

This paper proposes RegionMarker, a watermarking framework triggered by semantic regions. By defining trigger regions in a low-dimensional space and injecting semantic watermarks, it represents the first EaaS copyright protection method capable of simultaneously defending against CSE attacks, paraphrasing attacks, and dimensional perturbation attacks.

## Background & Motivation

Embedding-as-a-Service (EaaS) is a commercial deployment strategy for Large Language Models, providing paid text embedding services (such as OpenAI's text-embedding-3-large). However, EaaS faces severe threats from model extraction attacks: attackers can query the provider model with a text corpus to obtain embeddings and train functionally similar surrogate models at an extremely low cost, resulting in substantial financial losses.

Existing watermarking defense methods fall into two categories, both of which exhibit obvious limitations:

- **Trigger word methods** (EmbMarker, WARDEN, EspeW): These rely on specific trigger words to inject watermarks, which are easily bypassed by paraphrasing attacks since the trigger words disappear after paraphrasing.
- **Linear transformation methods** (WET): These apply a secret linear transformation to all embeddings. While they can resist paraphrasing attacks, they assume that the dimensions and their order remain unchanged, making them highly vulnerable to dimensional perturbation attacks (dimension shifting, dimension truncation).

The key challenge lies in: No existing method can simultaneously defend against CSE attacks, paraphrasing attacks, and dimensional perturbation attacks. In real-world scenarios, attackers will try various attack methods, and the defense fails if any single attack succeeds.

## Method

### Overall Architecture

The RegionMarker framework consists of three steps: **trigger region definition**, **semantic watermark injection**, and **copyright verification**. The core idea is to use semantic regions (instead of shallow vocabulary) as triggers, leverage multiple semantic regions to resist watermark removal attacks, and use the text embedding itself as a watermark to defend against dimensional perturbation attacks.

### Trigger Region Definition

Since data in high-dimensional space is sparse and unevenly distributed, direct division in high-dimensional space is easily identified by CSE attacks. Therefore, PCA is first used to reduce the dimensionality to a $d$-dimensional compact semantic space to make the data distribution more uniform and the watermark more imperceptible.

After dimensionality reduction, **Locality-Sensitive Hashing (LSH)** is used to uniformly divide the $d$-dimensional space into $2^d$ regions. For each text embedding $\mathbf{v}$, its corresponding region is determined by computing a $d$-bit binary LSH signature via random hyperplane projection:

$$\text{LSH}_i(\mathbf{v}) = \mathbb{1}(\mathbf{n}_i \cdot \mathbf{v} > 0)$$

$$\text{LSH}(\mathbf{v}) = [\text{LSH}_1(\mathbf{v}), \cdots, \text{LSH}_d(\mathbf{v})]$$

where $\mathbf{n}_i$ represents mutually orthogonal hyperplane normal vectors. Once the division is complete, given a watermark region ratio $\alpha$, $R = \alpha \cdot 2^d$ regions are randomly sampled as trigger regions $A = \{a_1, a_2, \ldots, a_R\}$.

**Key Security**: The dimensionality reduction matrix and the trigger regions are only known to the provider, making it impossible for attackers to locate the watermark.

### Semantic Watermark Injection

Each trigger region is assigned a unique watermark embedding $\mathbf{W} = \{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_R\}$, where $\mathbf{w}_r$ is the embedding of a target sample. If the dimensionality-reduced text embedding falls into trigger region $a_r$:

$$\mathbf{e}_p = \text{Norm}((1 - \lambda) \cdot \mathbf{e}_0 + \lambda \cdot \mathbf{w}_r)$$

where $\lambda$ controls the watermark strength (default 0.2). The regions partitioned by LSH are mutually disjoint, meaning each text embedding carries at most one watermark. The benefit of using the target sample embedding as the watermark is: even if the attacker shifts or truncates the dimensions, the watermark embedding undergoes the same changes, preserving the relative relationship.

### Copyright Verification

Construct a verification corpus containing multiple backdoor corpora $D_p^{b_r}$ (texts falling into watermarked regions) and benign corpora $D_p^n$ (texts in non-watermarked regions). Calculate the cosine similarity difference $\Delta_{cos}$ and the L2 distance difference $\Delta_{l2}$ between each group of texts and the watermark embedding, and employ the **Kolmogorov-Smirnov test** to determine the distribution difference:

$$\Delta_{cos} = \max_{1 \leq r \leq R} \Delta_{cos_r}, \quad p\text{-value} = \min_{1 \leq r \leq R} p\text{-value}_r$$

A conservative strategy is adopted: the copyright is deemed infringed if the p-value < 0.05 for any of the watermarked regions.

## Main Results

### Experimental Setup

- **Provider Model**: GPT-3 text-embedding-002; **Stealer Model**: BERT
- **Datasets**: SST-2 (Sentiment Classification), AG News (News Classification), Enron (Spam Detection), MIND (News Recommendation)
- **Attack Types**: CSE Attack, NLLB/gpt-4o-mini Paraphrasing Attack, Dimension Shifting Attack, Dimension Truncation Attack
- **Hyperparameters**: $d=4$, $\alpha=20\%$, $\lambda=0.2$

### Table 1: Overall comparison of different methods on SST-2 dataset

| Method | No Attack | CSE | Paraphrasing(NLLB) | Paraphrasing(GPT-4o-mini) | Dimension Shifting | Dimension Truncation |
|------|--------|-----|-----------|-------------------|---------|---------|
| WARDEN | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| EspeW | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| WET | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **RegionMarker** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

RegionMarker is the only method that successfully protects copyright under all attacks. Task performance (ACC approx. 93%) is on par with baseline methods, and watermark injection does not significantly degrade embedding quality.

### Table 2: Detection performance comparison of different methods on Enron dataset

| Method | Attack Type | p-value | Δcos(%) | Copyright Protection |
|------|---------|---------|---------|---------|
| RegionMarker | No Attack | <10⁻⁵ | 11.91 | ✓ |
| RegionMarker | CSE | <10⁻⁴ | 26.27 | ✓ |
| RegionMarker | Paraphrasing(NLLB) | <10⁻⁴ | 7.12 | ✓ |
| RegionMarker | Dimension Shifting | <0.01 | 2.33 | ✓ |
| RegionMarker | Dimension Truncation | <0.02 | 1.96 | ✓ |
| WARDEN | CSE | >0.05 | 1.47 | ✗ |
| EspeW | Paraphrasing(NLLB) | >0.49 | 0.40 | ✗ |
| WET | Dimension Shifting | >0.08 | -1.23 | ✗ |

On the Enron dataset, RegionMarker also passes all tests completely, whereas other methods have their weaknesses exposed and bypassed.

## Ablation Study & Key Findings

1. **Necessity of PCA Dimensionality Reduction**: Removing PCA degrades the p-value under CSE attack from <0.05 to >0.5 (protection failure), demonstrating that using dimensionality reduction to homogenize data distribution is critical to resisting CSE.
2. **Necessity of Multiple Watermark Embeddings**: When using a single watermark embedding, the p-value under CSE attack degrades from <0.05 to >0.08 (protection failure), as a single watermark is easy to identify and remove.
3. **Watermark Region Ratio $\alpha$**: Increasing $\alpha$ improves detection performance, but a low ratio of 20% is maintained to minimize the impact on embedding quality.
4. **PCA Dimension $d$**: As $d$ increases, detection performance under no attack decreases (fewer samples per region), but robustness in attack scenarios improves (more watermarks make removal harder). $d=4$ is chosen as the balance point.

## Highlights & Insights

- **Comprehensive Defending Capability**: The first EaaS watermark method to simultaneously withstand three major types of attacks (CSE, paraphrasing, dimensional perturbation).
- **Ingenious Design**: Replaces trigger words with semantic regions, naturally resisting paraphrasing attacks (paraphrasing does not change semantic region mapping); uses the text embedding itself as the watermark, naturally resisting dimensional perturbation (watermarks change identically with embeddings).
- **Strong Confidentiality**: A three-layer secret scheme consisting of the dimensionality reduction matrix + random trigger regions + multiple watermark embeddings, making it difficult for attackers to reverse engineer.
- **High Practicality**: Almost no loss in task performance, with simple hyperparameter settings (only three parameters: $d$, $\alpha$, $\lambda$).

## Limitations & Future Work

- Only experimented on text embedding scenarios, not extended to multimodal embeddings (images, audio, etc., in EaaS).
- Although attack types cover existing mainstream methods, potential adaptive attacks are not considered (e.g., targeted attacks when attackers know the region-triggering strategy is used).
- The stealer model is fixed to BERT; the effect of stronger surrogate models (e.g., GPT-level models) on watermark retention has not been evaluated.
- PCA dimensionality reduction and LSH region division depend on the provider's training data distribution; the performance under data distribution shift remains unknown.
- The verification process requires constructing backdoor and benign corpora for statistical tests, which does not support fast single-sample verification.

## Related Work & Insights

- **Model Extraction Attacks**: liu2022stolenencoder discovered that public EaaS APIs are susceptible to imitation attacks.
- **Trigger Word Watermarking**: EmbMarker (Peng et al., 2023) pioneered this direction but is vulnerable to paraphrasing; WARDEN uses multiple watermark embeddings to resist CSE; EspeW enhances imperceptibility in sub-dimensional embeddings.
- **Transformation Watermarking**: WET (Shetty et al., 2024) employs secret linear transformations to resist paraphrasing but cannot resist dimensional perturbation.
- **Copyright Verification**: Statistical verification methods based on the KS test.

## Rating

⭐⭐⭐⭐ — The problem definition is clear, revealing for the first time the dilemma where existing methods fail to defend comprehensively against all three types of attacks. The proposed method design is elegant, utilizing two core innovations—semantic region triggering and text embedding watermarks—to naturally solve both major challenges. The evaluation is thorough, covering 4 datasets and 5 types of attacks. The limitations lie in the lack of adaptive attack analysis and multimodal extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MaxMark: High-Capacity Diffusion-Native Watermarking via Robust and Invertible Latent Embedding](../../CVPR2026/ai_safety/maxmark_high-capacity_diffusion-native_watermarking_via_robust_and_invertible_la.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[CVPR 2026\] ReMoE: Region-Mixture Experts for Adversarially-Robust Vision Transformers](../../CVPR2026/ai_safety/remoe_region-mixture_experts_for_adversarially-robust_vision_transformers.md)

</div>

<!-- RELATED:END -->
