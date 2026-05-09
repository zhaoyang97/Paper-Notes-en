---
title: >-
  [Paper Note] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization
description: >-
  [AAAI 2026][Recommender Systems][Cross-domain recommendation] This paper proposes GenCDR, a framework that introduces the generative semantic ID paradigm into LLM-driven cross-domain recommendation for the first time, via two core modules: domain-adaptive semantic tokenization and cross-domain autoregressive recommendation. GenCDR effectively addresses the non-transferability of item IDs and insufficient domain-personalized modeling in conventional approaches.
tags:
  - AAAI 2026
  - Recommender Systems
  - Cross-domain recommendation
  - semantic ID
  - generative recommendation
  - large language models
  - domain adaptation
date: 2026-05-08
content_hash: a71ef695f64ee7c4
---

# From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization

**Conference**: AAAI 2026
**arXiv**: [2511.08006](https://arxiv.org/abs/2511.08006)
**Code**: [https://github.com/hupeiyu21/GenCDR](https://github.com/hupeiyu21/GenCDR)
**Area**: Recommender Systems
**Keywords**: Cross-domain recommendation, semantic ID, generative recommendation, large language models, domain adaptation

## TL;DR

This paper proposes GenCDR, a framework that introduces the generative semantic ID paradigm into LLM-driven cross-domain recommendation for the first time, via two core modules: domain-adaptive semantic tokenization and cross-domain autoregressive recommendation. GenCDR effectively addresses the non-transferability of item IDs and insufficient domain-personalized modeling in conventional approaches.

## Background & Motivation

Cross-domain recommendation (CDR) aims to leverage users' interaction behaviors across multiple heterogeneous domains to improve recommendation accuracy and generalization. However, existing methods face two core challenges:

**Item Tokenization Gap**: Conventional methods heavily rely on shared user/item IDs as the bridge for cross-domain knowledge transfer, yet in real-world scenarios (e.g., across different platforms), user and item IDs are often not aligned. Directly using raw IDs leads to vocabulary explosion and fails to capture high-order collaborative knowledge.

**Domain Personalization Gap**: Existing methods struggle to effectively disentangle and model the dynamic interaction between general interests and domain-specific expressions. For example, "Apple Watch" in the technology domain and fresh "Apple" in the lifestyle domain share a semantic concept but exhibit entirely different domain attributes (fitness vs. sweetness/vitamins).

Existing LLM-based CDR methods either use LLMs as feature enhancers or reformulate recommendation as a natural language problem, yet neither fundamentally addresses the two challenges above. The core insight of this paper is: **raw semantic information (e.g., textual descriptions) is naturally transferable across domains, whereas item IDs are not.** This motivates replacing conventional IDs with discrete semantic IDs (SIDs).

## Method

### Overall Architecture

The GenCDR framework consists of three core modules: (a) a **Domain-adaptive Tokenization** module that generates disentangled semantic IDs for items; (b) a **Cross-Domain Autoregressive Recommendation** module that integrates general and domain-specific interests for user preference modeling; and (c) a **Domain-aware Prefix-tree** that ensures efficient and valid generation. The overall pipeline follows a two-stage design: a tokenizer is trained first to generate SIDs, followed by training of the recommendation model.

### Key Designs

#### 1. **Domain-General Semantic Token Generation (RQ-VAE)**

Based on a pretrained Residual Quantization Variational Autoencoder (RQ-VAE), the textual features of all items are encoded into discrete semantic code sequences $\mathbf{c} = (c_0, \dots, c_{M-1})$. An encoder $E$ maps the feature embedding $\mathbf{x}$ to a latent representation $\mathbf{z}$, which is then quantized level by level across $M$ codebooks. The training objective consists of three terms:

- **Reconstruction loss** $\mathcal{L}_{\text{REC}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
- **Quantization loss** $\mathcal{L}_Q$: aligns encoder outputs with codebook vectors via a commitment term
- **Masked Token Modeling loss** $\mathcal{L}_{\text{MTM}}$: predicts masked semantic codes to ensure contextual consistency

The total pretraining loss is $\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{REC}} + \mu\mathcal{L}_Q + \lambda\mathcal{L}_{\text{MTM}}$. After pretraining, the general encoder and codebooks are frozen.

#### 2. **Domain-Specific Semantic Token Adapters (LoRA Adapters)**

Since the general encoder cannot fully capture domain-distinctive features (e.g., visual aesthetics in videos, narrative style in books), a Low-Rank Adaptation (LoRA) module is introduced for each domain $d$:

$$h_{\text{out}} = W_0 h_{\text{in}} + B_d A_d h_{\text{in}}$$

The general encoder weights $W_0$ are frozen, and only domain-specific parameters $\theta_d = \{B_d, A_d\}$ are fine-tuned using a self-supervised reconstruction loss.

#### 3. **Item-level Dynamic Semantic Routing Network (Dynamic Routing)**

To avoid negative transfer caused by static fusion strategies, a per-item gated routing network $R_\phi$ is designed to dynamically balance general and domain-specific representations:

$$\alpha = \sigma(R_\phi(\mathbf{x})), \quad \mathbf{z}_{\text{fused}} = (1-\alpha)\cdot\mathbf{z}_{\text{uni}} + \alpha\cdot\mathbf{z}_{\text{spec}}$$

The router is regularized via a Variational Information Bottleneck (VIB), which constrains the amount of information extracted from the input through a KL divergence term, thereby promoting disentangled representations.

#### 4. **Cross-Domain Autoregressive Recommendation Module**

A symmetric disentanglement of general and specific interests is applied on the user side:

- **General interest modeling**: A mixture of multi-LoRA experts is added on top of a frozen LLM and trained on data from all domains to capture transferable behavioral patterns.
- **Domain-specific interest adaptation**: After freezing general parameters, dedicated LoRA adapters are trained for each domain.
- **User-level dynamic routing**: The general model probability distribution $P_{\text{uni}}$ and the domain-adapted model distribution $P_{\text{spec}}$ are fused as: $P_{\text{final}} = (1-\gamma)\cdot P_{\text{uni}} + \gamma\cdot P_{\text{spec}}$

#### 5. **Domain-aware Prefix-tree Inference**

An offline prefix tree is constructed for each domain, encoding all valid SID sequences. During inference, the LLM is constrained to generate only from the valid subset under the current prefix, preventing invalid ID outputs while significantly reducing computational overhead.

### Loss & Training

The overall training proceeds in multiple stages:
1. RQ-VAE pretraining (general semantic tokens)
2. Domain LoRA fine-tuning + VIB routing (domain-adaptive tokenization)
3. General interest mixture-of-LoRA fine-tuning (autoregressive recommendation)
4. Domain-specific LoRA fine-tuning + user-level routing

## Key Experimental Results

### Main Results

Experiments are conducted on three cross-domain scenario pairs (all real-world data): Sports-Clothing (lifestyle), Phones-Electronics (technology), and Books-Movies (entertainment).

| Scenario | Domain | Metric | GenCDR (Ours) | LLM4CDSR | TriCDR | TIGER |
|----------|--------|--------|----------------|----------|--------|-------|
| Lifestyle | Sports | R@5 | **0.0274** | 0.0263 | 0.0266 | 0.0267 |
| Lifestyle | Clothing | R@5 | **0.0181** | 0.0176 | 0.0174 | 0.0173 |
| Technology | Phones | N@5 | **0.0411** | 0.0401 | 0.0396 | 0.0315 |
| Technology | Electronics | N@5 | **0.0235** | 0.0230 | 0.0231 | 0.0214 |
| Entertainment | Books | R@5 | **0.0192** | 0.0161 | 0.0155 | 0.0172 |
| Entertainment | Movies | R@5 | See full table | - | - | - |

GenCDR significantly outperforms existing state-of-the-art methods across all scenarios and metrics, including three categories of baselines: single-domain (SDSR), generative (GenRec), and cross-domain (CDSR).

### Ablation Study

| Configuration | R@5 (Sports) | N@5 (Phones) | Note |
|---------------|-------------|--------------|------|
| GenCDR (Full) | **0.0274** | **0.0411** | All modules |
| w/o Dynamic Routing | Decrease | Decrease | Validates routing necessity |
| w/o Domain Adapters | Decrease | Decrease | Domain-specific info matters |
| w/o VIB Regularization | Decrease | Decrease | Prevents overfitting |
| w/o Prefix-tree | Validity drops | — | Ensures generation validity |

### Key Findings

- GenCDR achieves superior performance on real-world datasets with high sparsity (>99.9%), validating the effectiveness of the semantic ID paradigm for cross-domain recommendation.
- The dynamic routing mechanism is more robust than static fusion, adaptively determining the mixing ratio of general and domain-specific knowledge for each item/user.
- VIB regularization effectively prevents router overfitting and promotes disentangled representation learning.
- The domain-aware prefix tree significantly improves inference efficiency while guaranteeing generation validity.

## Highlights & Insights

- **First integration of generative semantic IDs into LLM-based cross-domain recommendation**, elegantly resolving the long-standing problem of non-transferable item IDs.
- **Symmetric design philosophy**: item-side and user-side adopt symmetric general-specific disentanglement architectures (LoRA + dynamic routing), reflecting the unified nature of the framework.
- **Clever application of information bottleneck regularization** provides theoretical grounding for disentangling general and domain-specific representations.
- The prefix-tree-constrained generation is a simple yet effective design that addresses the common problem of invalid IDs in generative recommendation.

## Limitations & Future Work

- Although the experiments cover three scenarios, each involves only two domains; scalability to a large number of domains (>5) remains unvalidated.
- The number of LoRA adapters grows linearly with the number of domains, increasing parameter management cost in highly multi-domain settings.
- Cold-start scenarios (new domains or new users) are not explored, which are critical in practical applications.
- The two-stage tokenization-then-recommendation training pipeline is relatively complex; end-to-end joint training may further improve performance.

## Related Work & Insights

- **Relation to TIGER** (single-domain generative recommendation): GenCDR extends it to multiple domains and adds a domain disentanglement mechanism.
- **Distinction from LLM4CDSR**: the latter applies LLMs directly to CDR without semantic IDs or domain adaptation.
- The combination of mixture-of-LoRA experts and routing networks inspires a general paradigm for multi-domain modeling.
- The general-specific disentanglement idea is similarly applicable to multimodal recommendation systems.

## Rating

- Novelty: ⭐⭐⭐⭐ — First introduction of semantic IDs into cross-domain recommendation; the symmetric disentanglement design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three scenarios, six domains, comprehensive baseline comparisons and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Provides a new paradigm for cross-domain recommendation in the LLM era.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)

<!-- RELATED:END -->
