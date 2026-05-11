---
title: >-
  [Paper Note] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service
description: >-
  [AAAI 2026][AI Safety][EaaS copyright protection] This paper proposes RegionMarker, a semantic watermarking framework based on region-triggered mechanisms. It defines trigger regions in a low-dimensional space and inject…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "EaaS copyright protection"
  - "embedding watermarking"
  - "semantic region triggering"
  - "model extraction attack defense"
  - "locality-sensitive hashing"
date: 2026-05-08
content_hash: ac8851e09347416d
---

# RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service

**Conference**: AAAI 2026
**arXiv**: [2511.13329](https://arxiv.org/abs/2511.13329)
**Code**: Not released
**Area**: AI Security
**Keywords**: EaaS copyright protection, embedding watermarking, semantic region triggering, model extraction attack defense, locality-sensitive hashing

## TL;DR

This paper proposes RegionMarker, a semantic watermarking framework based on region-triggered mechanisms. It defines trigger regions in a low-dimensional space and injects semantic watermarks, constituting the first EaaS copyright protection method capable of simultaneously resisting CSE attacks, paraphrasing attacks, and dimension perturbation attacks.

## Background & Motivation

Embedding-as-a-Service (EaaS) is a commercial deployment strategy for large language models that provides text embedding services on a pay-per-use basis (e.g., OpenAI's text-embedding-3-large). However, EaaS faces serious model extraction attack threats: adversaries can query the provider's model using text corpora to obtain embeddings, then train functionally similar surrogate models at minimal cost, resulting in significant economic losses.

Existing watermarking defense methods fall into two categories, each with notable shortcomings:

- **Trigger-word-based methods** (EmbMarker, WARDEN, EspeW): rely on specific trigger words to inject watermarks and are easily bypassed by paraphrasing attacks, since the trigger words disappear after paraphrasing.
- **Linear transformation-based methods** (WET): apply a secret linear transformation to all embeddings; they resist paraphrasing attacks but assume that dimensions and their ordering remain unchanged, making them highly vulnerable to dimension perturbation attacks (dimension shifting, dimension truncation).

The root cause is that no existing method can simultaneously defend against all three attack types: CSE attacks, paraphrasing attacks, and dimension perturbation attacks. In practice, adversaries attempt multiple attack strategies, and succeeding with any single one is sufficient to circumvent the protection.

## Method

### Overall Architecture

The RegionMarker framework consists of three steps: **trigger region definition**, **semantic watermark injection**, and **copyright verification**. The core idea is to use semantic regions (rather than surface-level lexical items) as triggers, to leverage multiple semantic regions to resist watermark removal attacks, and to use the text embeddings themselves as watermarks to defend against dimension perturbation attacks.

### Trigger Region Definition

Because data in high-dimensional spaces is sparse and unevenly distributed, direct partitioning in high-dimensional space is easily identified by CSE attacks. Therefore, PCA is first applied to reduce dimensionality to a $d$-dimensional compact semantic space, making the data distribution more uniform and the watermarks less detectable.

After dimensionality reduction, **locality-sensitive hashing (LSH)** is used to uniformly partition the $d$-dimensional space into $2^d$ regions. For each text embedding $\mathbf{v}$, a $d$-bit binary LSH signature is computed via random hyperplane projection to determine its region assignment:

$$\text{LSH}_i(\mathbf{v}) = \mathbb{1}(\mathbf{n}_i \cdot \mathbf{v} > 0)$$

$$\text{LSH}(\mathbf{v}) = [\text{LSH}_1(\mathbf{v}), \cdots, \text{LSH}_d(\mathbf{v})]$$

where $\mathbf{n}_i$ denotes mutually orthogonal hyperplane normal vectors. After partitioning, a watermark region ratio $\alpha$ is set, and $R = \alpha \cdot 2^d$ regions are randomly sampled as trigger regions $A = \{a_1, a_2, \ldots, a_R\}$.

**Key security property**: The dimensionality reduction matrix and trigger regions are known only to the provider, making it impossible for adversaries to locate the watermarks.

### Semantic Watermark Injection

A unique watermark embedding is assigned to each trigger region: $\mathbf{W} = \{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_R\}$, where $\mathbf{w}_r$ is the embedding of a target sample. If a text embedding reduced by PCA falls into trigger region $a_r$, the watermarked embedding is computed as:

$$\mathbf{e}_p = \text{Norm}((1 - \lambda) \cdot \mathbf{e}_0 + \lambda \cdot \mathbf{w}_r)$$

where $\lambda$ controls the watermark strength (default 0.2). Since LSH-partitioned regions are mutually disjoint, each text embedding carries at most one watermark. Using target sample embeddings as watermarks ensures that even if an adversary shifts or truncates dimensions, the watermark embedding undergoes the same transformation, preserving the relative relationship.

### Copyright Verification

A verification corpus is constructed comprising multiple backdoor corpora $D_p^{b_r}$ (texts falling into watermarked regions) and a benign corpus $D_p^n$ (texts from non-watermarked regions). The cosine similarity difference $\Delta_{cos}$ and L2 distance difference $\Delta_{l2}$ between each group and the watermark embeddings are computed, and the **Kolmogorov-Smirnov test** is applied to assess distributional differences:

$$\Delta_{cos} = \max_{1 \leq r \leq R} \Delta_{cos_r}, \quad p\text{-value} = \min_{1 \leq r \leq R} p\text{-value}_r$$

A conservative strategy is adopted: a p-value < 0.05 for any single watermarked region is sufficient to conclude infringement.

## Key Experimental Results

### Experimental Setup

- **Provider model**: GPT-3 text-embedding-002; **Surrogate model**: BERT
- **Datasets**: SST-2 (sentiment classification), AG News (news classification), Enron (spam detection), MIND (news recommendation)
- **Attack types**: CSE attack, NLLB/gpt-4o-mini paraphrasing attacks, dimension shifting attack, dimension truncation attack
- **Hyperparameters**: $d=4$, $\alpha=20\%$, $\lambda=0.2$

### Main Results

**Table 1: Comprehensive comparison on SST-2 dataset**

| Method | No Attack | CSE | Paraphrase (NLLB) | Paraphrase (GPT-4o-mini) | Dim. Shift | Dim. Truncation |
|--------|-----------|-----|-------------------|--------------------------|------------|-----------------|
| WARDEN | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| EspeW | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| WET | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **RegionMarker** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

RegionMarker is the only method that successfully protects copyright under all attack types. Task performance (ACC ≈ 93%) is on par with baseline methods, indicating that watermark injection does not significantly degrade embedding quality.

**Table 2: Detection performance comparison on Enron dataset**

| Method | Attack Type | p-value | Δcos (%) | Copyright Protection |
|--------|-------------|---------|----------|----------------------|
| RegionMarker | No Attack | <10⁻⁵ | 11.91 | ✓ |
| RegionMarker | CSE | <10⁻⁴ | 26.27 | ✓ |
| RegionMarker | Paraphrase (NLLB) | <10⁻⁴ | 7.12 | ✓ |
| RegionMarker | Dim. Shift | <0.01 | 2.33 | ✓ |
| RegionMarker | Dim. Truncation | <0.02 | 1.96 | ✓ |
| WARDEN | CSE | >0.05 | 1.47 | ✗ |
| EspeW | Paraphrase (NLLB) | >0.49 | 0.40 | ✗ |
| WET | Dim. Shift | >0.08 | -1.23 | ✗ |

RegionMarker passes comprehensively on the Enron dataset as well, while all other methods are defeated by at least one attack type.

## Ablation Study

1. **Necessity of PCA dimensionality reduction**: Removing PCA causes the p-value under CSE attacks to degrade from <0.05 to >0.5 (protection failure), demonstrating that uniform data distribution through dimensionality reduction is critical for resisting CSE attacks.
2. **Necessity of multiple watermark embeddings**: Using a single watermark embedding causes the p-value under CSE attacks to degrade from <0.05 to >0.08 (protection failure), as a single watermark is easily identified and removed.
3. **Watermark region ratio $\alpha$**: Detection performance improves as $\alpha$ increases, but a low ratio of 20% is maintained to minimize impact on embedding quality.
4. **PCA dimension $d$**: Increasing $d$ reduces detection performance in the no-attack setting (fewer samples per region) but improves robustness under attacks (more watermarks are harder to remove); $d=4$ is selected as the balance point.

## Highlights & Insights

- **Comprehensive defense**: The first EaaS watermarking method capable of simultaneously resisting all three mainstream attack types (CSE, paraphrasing, and dimension perturbation).
- **Elegant design**: Replacing trigger words with semantic regions naturally resists paraphrasing attacks (paraphrasing does not alter semantic region membership); using text embeddings themselves as watermarks naturally resists dimension perturbation (the watermark co-varies with the embedding).
- **Strong secrecy**: Three layers of secrets—the dimensionality reduction matrix, random trigger regions, and multiple watermark embeddings—make reverse engineering by adversaries highly difficult.
- **Practical utility**: Task performance is nearly unaffected, and the hyperparameter configuration is simple (only three parameters: $d$, $\alpha$, $\lambda$).

## Limitations & Future Work

- Experiments are conducted exclusively on text embedding scenarios; the framework has not been extended to multimodal embeddings (image, audio, etc. EaaS).
- Although the attack types cover existing mainstream methods, adaptive attacks are not considered (e.g., targeted attacks by adversaries who are aware that a region-triggered strategy is in use).
- The surrogate model is fixed as BERT; the effect of stronger surrogate models (e.g., GPT-level models) on watermark retention has not been evaluated.
- PCA dimensionality reduction and LSH region partitioning depend on the provider's training data distribution; performance under distributional shift remains unknown.
- The verification process requires constructing backdoor and benign corpora for statistical testing and does not support rapid single-sample verification.

## Related Work & Insights

- **Model extraction attacks**: liu2022stolenencoder identifies that public EaaS APIs are susceptible to imitation attacks.
- **Trigger-word watermarking**: EmbMarker (Peng et al., 2023) is the pioneering work but is not robust to paraphrasing; WARDEN uses multiple watermark embeddings to resist CSE; EspeW enhances concealment through sub-dimensional embedding.
- **Transformation-based watermarking**: WET (Shetty et al., 2024) applies a secret linear transformation to resist paraphrasing but is vulnerable to dimension perturbation.
- **Copyright verification**: Statistical verification methods based on the KS test.

## Rating

⭐⭐⭐⭐ — The problem is clearly defined, and the paper is the first to reveal that existing methods cannot comprehensively defend against all three attack types. The method design is elegant, with the two core innovations of semantic region triggering and text-embedding-based watermarks naturally resolving two major challenges. Experiments comprehensively cover 4 datasets and 5 attack types. Limitations include the absence of adaptive attack analysis and multimodal extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[AAAI 2026\] Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness](sim-to-real_an_unsupervised_noise_layer_for_screen-camera_watermarking_robustnes.md)

</div>

<!-- RELATED:END -->
