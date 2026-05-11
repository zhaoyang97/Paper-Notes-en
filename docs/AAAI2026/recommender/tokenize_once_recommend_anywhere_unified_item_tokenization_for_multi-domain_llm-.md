---
title: >-
  [Paper Note] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation
description: >-
  [AAAI 2026][Recommender Systems] This paper proposes UniTok, a unified item tokenization framework that employs a customized Mixture-of-Experts architecture (TokenMoE) combined with shared codebooks to achieve efficient…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
date: 2026-05-08
content_hash: e500045a8a315b9c
---

# Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation

**Conference**: AAAI 2026
**arXiv**: [2511.12922](https://arxiv.org/abs/2511.12922)
**Code**: [github.com/jackfrost168/UniTok](https://github.com/jackfrost168/UniTok)

## TL;DR

This paper proposes UniTok, a unified item tokenization framework that employs a customized Mixture-of-Experts architecture (TokenMoE) combined with shared codebooks to achieve efficient discrete item representations across multiple domains, eliminating the need to train a separate tokenizer per domain, while maintaining cross-domain semantic balance through a mutual information calibration mechanism.

## Background & Motivation

### State of the Field
LLM-based recommendation systems map item spaces to language spaces via item tokenization, enabling LLMs to process items as part of natural language sequences. However, existing item tokenization methods (e.g., TIGER, LC-Rec, LETTER) are designed for **single-domain** settings and require training a separate tokenizer for each domain.

### Core Challenges

**Training Overhead (C1)**: When a recommendation system must cover multiple item domains, repeatedly training domain-specific tokenizers is inefficient and resource-intensive. Across 10 domains, existing methods require 9.63× more trainable parameters than UniTok.

**Semantic Alignment (C2)**: Data distributions and semantic characteristics vary substantially across domains; naively sharing a token space leads to semantic confusion and biased token allocation.

### Motivation
While unified multi-domain learning has become a trend in NLP and CV, item tokenization in recommender systems remains confined to a "one model per domain" paradigm. This paper presents the first attempt to construct a cross-domain unified item tokenization framework.

## Method

### Overall Architecture
UniTok consists of four core components: a **shared autoencoder**, **TokenMoE**, **codebook identifiers**, and a **mutual information calibration mechanism**.

### 1. Shared Autoencoder
- A pretrained content encoder extracts semantic embeddings $\mathbf{X}^k \in \mathbb{R}^{|\mathcal{I}_k| \times d}$ for items across all domains.
- A shared encoder $f_\theta$ projects items from all domains into a unified latent space: $\mathbf{z}_i^k = f_\theta(\mathbf{x}_i^k)$.
- A shared decoder $g_\phi$ reconstructs original embeddings from quantized latent representations.
- The reconstruction loss is optimized: $\mathcal{L}_{\text{Rec}} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i^k} \|\mathbf{x}_i^k - \hat{\mathbf{x}}_i^k\|^2$.

### 2. TokenMoE: A Customized MoE Architecture
**Core innovation**—introducing MoE into the tokenization module rather than the conventional Transformer FFN layers:
- **Domain-specific experts**: Each domain corresponds to one expert network, capturing domain-exclusive semantic patterns.
- **Shared expert**: Always activated, encoding general cross-domain knowledge.
- **Router**: A softmax distribution determines the selection of the top-$N$ domain-specific experts.
- Output computation: $\hat{\mathbf{z}}_i^k = \sum_{k=1}^{K} G_k E_k(\mathbf{z}_i^k) + E_{\text{share}}(\mathbf{z}_i^k)$.
- Expert initialization: Experts are initialized with per-domain mean features, providing a strong inductive bias.

### 3. Codebook Identifiers (Residual Quantization)
- Residual quantization (RQ) within each expert discretizes items into compact token sequences.
- $L$ levels of codebooks are used, each containing $T$ code vectors.
- Each item is ultimately represented as: $\mathbf{z}_i^k \mapsto \mathbf{c}_i^k = (z_1, \dots, z_L, e_1, \dots, e_N)$, where $z_\ell$ denotes codebook indices and $e_n$ denotes the IDs of selected experts.
- The RQ training loss includes both codebook learning and commitment terms.

### 4. Mutual Information (MI) Calibration Mechanism
Addresses cross-domain semantic imbalance:
- Uses HSIC (Hilbert-Schmidt Independence Criterion) as a proxy for mutual information.
- Measures the dependency between input semantic embeddings $\mathbf{X}^k$ and latent embeddings $\mathbf{Z}^k$.
- MI calibration loss: $\mathcal{L}_{\text{MI}} = \text{Var}[\hat{I}^{(k)}] - \beta \mathbb{E}[\hat{I}^{(k)}]$.
- The first term penalizes cross-domain MI variance to alleviate semantic imbalance; the second encourages each domain to retain sufficient domain-specific information.

### Overall Optimization Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Rec}} + \lambda_{\text{RQ}} \mathcal{L}_{\text{RQ}} + \lambda_{\text{MI}} \mathcal{L}_{\text{MI}}$$

## Key Experimental Results

### Experimental Setup
- **Datasets**: 10 real-world datasets (Beauty, Cellphones, Grocery, Instruments, Office, Pet Supplies, Tools, Toys, Games, Yelp).
- **Baselines**: 4 collaborative filtering methods (MF, LightGCN, SASRec, Bert4Rec) + 5 item tokenization methods (P5-TID, P5-SemID, TIGER, LC-Rec, LETTER).
- **Metrics**: Recall@M and NDCG@M ($M \in \{5, 10\}$), full-ranking protocol.
- **Implementation details**: 4-level codebooks with 256 code vectors per level, dimension 32; $\lambda_{\text{RQ}}=1$, $\lambda_{\text{MI}}=0.03$.

### Main Results (Table 1: NDCG@10)

| Method | Beauty | Cellphones | Grocery | Tools | Toys | Yelp |
|--------|--------|------------|---------|-------|------|------|
| LETTER (2nd best) | 0.0364 | 0.0473 | 0.0392 | 0.0298 | 0.0291 | 0.0231 |
| **UniTok** | **0.0478** | **0.0647** | **0.0533** | **0.0439** | **0.0442** | **0.0321** |
| Gain | +25.5% | +36.8% | +36.0% | +43.0% | +51.9% | +39.0% |

- UniTok achieves state-of-the-art performance on all 10 datasets, with a maximum improvement of **51.89%** (Toys).
- Note: UniTok uses a single unified model for all 10 datasets, whereas baselines require separate training per dataset.

### Efficiency Comparison (Table 2: Trainable Parameters)

| Module | Conventional Codebook Methods (10 datasets total) | UniTok |
|--------|---------------------------------------------------|--------|
| Codebook | 0.33M | 0.36M |
| Autoencoder | 87.45M | 8.75M |
| Router | — | 0.01M |
| **Total** | **87.78M** | **9.11M** |

Parameter count is reduced by approximately **9.63×**, primarily owing to the shared autoencoder design.

### Zero-Shot Generalization (Table 4: Unseen Domains)
Direct evaluation on three domains unseen during training (Clothing, Health, Sports):
- UniTok achieves state-of-the-art performance on new domains without retraining.
- NDCG@10 improves by **17.87%** on Health; Recall@10 improves by **12.33%** on Clothing.

### Ablation Study (Table 5)

| Variant | Description | Beauty N@10 |
|---------|-------------|-------------|
| UniTok-1 | Remove TokenMoE + MI | 0.0304 |
| UniTok-2 | Keep MoE, remove shared expert + MI | 0.0436 |
| UniTok-3 | Remove MI calibration | 0.0457 |
| **UniTok** | Full model | **0.0478** |

TokenMoE contributes the most, while MI calibration provides further consistent gains.

## Theoretical Analysis

The paper provides three theoretical guarantees:
1. **Theorem 1**: UniTok's token space has strictly higher entropy, implying greater token space capacity.
2. **Theorem 2**: UniTok's expected quantization error does not exceed that of standard codebook methods, i.e., quantization is more precise.
3. **Theorem 3**: The upper bound on cross-domain performance discrepancy is controlled by MI variance; reducing the variance promotes more stable cross-domain generalization.

## Highlights & Insights

- **Tokenize once, recommend anywhere**: A single unified model handles multiple domains, significantly reducing training and deployment costs.
- **Novel TokenMoE design**: Introducing MoE into the tokenization module rather than the FFN; the combination of domain-specific and shared experts preserves both domain-exclusive characteristics and general cross-domain knowledge.
- **MI calibration mechanism**: Using HSIC as a proxy for mutual information and minimizing cross-domain variance elegantly addresses the multi-domain semantic imbalance problem.
- **Dual validation via theory and experiments**: Three theorems provide theoretical guarantees from the perspectives of entropy, quantization error, and performance consistency.
- **Zero-shot generalization**: Competitive performance on new domains without retraining.

## Limitations & Future Work

1. **Content semantics only**: To maintain generality, collaborative filtering signals (user–item interactions) are deliberately excluded, potentially sacrificing some recommendation accuracy.
2. **Expert count tied to domain count**: The number of domain-specific experts $K$ typically equals the number of domains; adding new domains may require architectural adjustments.
3. **Fixed codebook capacity**: All domains share the same codebook size (256 code vectors, 4 levels), whereas different domains may require different granularities.
4. **Limited evaluation scenarios**: Validation is primarily conducted on Amazon and Yelp datasets; industrial-scale systems have not yet been tested.
5. **HSIC computational overhead**: The MI calibration requires computing kernel matrices, which may introduce additional computational burden at large scale.

## Related Work & Insights

- **Item Tokenization for LLM-Rec**: TIGER (Rajput et al. 2023) generates codebook identifiers via RQ; LC-Rec (Zheng et al. 2024) builds upon this; LETTER (Wang et al. 2024) further improves it. All are designed for single-domain settings.
- **Mixture-of-Experts**: From classical MoE (Jacobs et al. 1991) to Switch Transformers (Fedus et al. 2022) and DeepSeekMoE (Dai et al. 2024), MoE has been widely adopted for model scaling.
- **Multi-domain Recommendation**: ADIN (Jiang et al. 2022) and MDRED (Ning et al. 2023) study cross-domain recommendation but do not address the unification of item tokenization.
- **P5 Series**: P5-TID and P5-SemID (Hua et al. 2023) explore different item ID indexing strategies but lack semantic codebook support.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The paper identifies a neglected yet important problem in recommender systems—cross-domain item tokenization—and addresses it with a well-designed method (TokenMoE + MI calibration), solid theoretical analysis, and extensive experiments demonstrating substantial improvements (up to 51.89%). Points are deducted because the deliberate exclusion of collaborative signals limits the approach to content-based recommendation rather than a complete recommendation scenario, and the scalability of HSIC in large-scale settings remains questionable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)

</div>

<!-- RELATED:END -->
