---
title: >-
  [Paper Note] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual expansion] By analyzing the cross-lingual representation similarity across different layers of LLMs, this paper proposes LayerMoE, which dynamically allocates different numbers of experts for new languages on a layer-wise basis (fewer for high-similarity layers, more for low-similarity layers). It outperforms SOTA with 60% fewer expert parameters and further mitigates catastrophic forgetting by introducing routing cl…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual expansion"
  - "MoE"
  - "layer-wise expert allocation"
  - "catastrophic forgetting"
  - "continual learning"
date: 2026-05-08
content_hash: 93e78f326bf134d8
---

# Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts

**Conference**: ACL 2025  
**arXiv**: [2505.22582](https://arxiv.org/abs/2505.22582)  
**Code**: [https://github.com/XZhang00/LayerMoE](https://github.com/XZhang00/LayerMoE)  
**Area**: Multilingual Translation  
**Keywords**: multilingual expansion, MoE, layer-wise expert allocation, catastrophic forgetting, continual learning

## TL;DR
By analyzing the cross-lingual representation similarity across different layers of LLMs, this paper proposes LayerMoE, which dynamically allocates different numbers of experts for new languages on a layer-wise basis (fewer for high-similarity layers, more for low-similarity layers). It outperforms SOTA with 60% fewer expert parameters and further mitigates catastrophic forgetting by introducing routing classifiers in high-similarity layers.

## Background & Motivation

**Background**: Continually expanding new languages is a sustainable approach for building multilingual LLMs. MoE-LPR learns new languages by adding new experts while freezing old ones to prevent forgetting, but uniformly adding experts to each layer leads to severe parameter inflation (e.g., a 1.8B model grows to 3.2x its size after expanding to 3 languages).

**Limitations of Prior Work**: (1) Uniform expert allocation is inefficient and costly in terms of parameters. (2) Even with LPR loss guiding the routing of old languages, the performance on old languages still degrades significantly. (3) Parameter growth is unsustainable in lifelong learning scenarios.

**Key Challenge**: Different layers exhibit different levels of language specificity—some layers are naturally language-agnostic (requiring no new experts), while others are language-specific (requiring experts for adaptation). Uniform allocation ignores this heterogeneity.

**Goal**: (1) How to efficiently allocate the number of experts? (2) How to further prevent the forgetting of old languages?

**Key Insight**: Analyzed the cross-lingual cosine similarity of hidden states across different layers, revealing that intermediate and late layers have high similarity (language-agnostic), while layers 0-4 and 17-21 have low similarity (language-specific). The inverse of similarity is used as the indicator for expert allocation.

**Core Idea**: Automatically allocate the number of experts per layer based on the inverse of cross-lingual representation similarity across layers, allocating fewer experts to high-similarity layers and more to low-similarity layers.

## Method

### Overall Architecture
Step 1: Compute cross-lingual representation similarity (old-to-new / new-to-new) for each layer → Step 2: Allocate the number of experts based on the inverse similarity $N^i = \lceil \frac{(S^i)^{-1}}{\sum (S^i)^{-1}} \times \delta \rceil$ → Stage 1: Freeze old experts and train new experts → Step 3: Add routing classifiers in high-similarity layers → Stage 2: Recover old languages using LPR + classifier loss.

### Key Designs

1. **Layer-wise Similarity Analysis**:

    - Sample 100K tokens for both old and new languages, and calculate the average cosine similarity of post-attention hidden states in each layer.
    - Key Finding: Intermediate and final layers show high similarity (~0.7-0.9) while layers 0-4 and 17-21 show low similarity (~0.3-0.5).
    - High similarity = language-agnostic information → no new experts needed; low similarity = language-specific information → new experts needed.

2. **Routing Classifier**:

    - Add a binary classifier $W_c$ before the routing network in high-similarity layers (where routing easily confuses old and new languages).
    - Train the classifier using cross-entropy loss to predict whether a token belongs to an old language or a new language. If it belongs to an old language, force routing to old experts.
    - Design Motivation: Since hidden states are highly similar in high-similarity layers, simple linear routing networks fail to distinguish between them, needing assistance from an explicit classifier.

## Key Experimental Results

### Main Results (Single Expansion: English → Adding Bengali+Greek)

| Method | New Language Performance | Old Language Retention | Expert Parameters |
|------|:---:|:---:|:---:|
| MoE-LPR (Uniform) | Baseline | Baseline | 100% |
| **LayerMoE** | **Better** | **Better** | **40% (-60%)** |

### Ablation Study

| Configuration | New Language | Old Language | Description |
|------|:---:|:---:|------|
| LayerMoE (Full) | Best | Best | 60% fewer experts |
| w/o Classifier | Good | Degraded | Classifier is key to retaining old languages |
| Uniform Allocation (Original) | Medium | Medium | More parameters but inferior performance |
| Lifelong Expansion (3 language groups) | Best | Best | 33.3% fewer experts |

### Key Findings
- **60% fewer experts perform even better**: Demonstrates significant wastage in uniform allocation.
- **Classifiers effectively prevent old language degradation**: Classifiers in high-similarity layers significantly improve routing accuracy.
- **Equally effective in lifelong expansion scenarios**: The benefits of parameter efficiency accumulate over multiple expansions.

## Highlights & Insights
- **Representation similarity as an expert allocation indicator is elegant and simple**: No complex search is required, as measurable layer-wise similarity directly guides architectural design.
- **The discovery of layer-wise heterogeneity aligns with existing interpretability studies**: Lower layers and certain higher layers capture language-specific features, whereas intermediate layers are language-agnostic.

## Limitations & Future Work
- Experiments are only conducted on Qwen1.5-1.8B; scaling to larger models remains unverified.
- The classifier threshold K needs to be selected manually.
- Similarity calculation requires data samples from each language, which may limit applicability in cold-start scenarios.

## Related Work & Insights
- **vs MoE-LPR**: LayerMoE outperforms it on both old and new languages with 60% fewer parameters.
- **vs Language-Specific/Agnostic Neurons Studies**: This paper successfully applies these findings to MoE architecture design, representing an elegant transition from theory to practice.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of layer-wise similarity-driven expert allocation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers single & lifelong expansion with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive method descriptions.
- Value: ⭐⭐⭐⭐ Highly practical for the efficient expansion of multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](../../ICLR2026/multilingual_mt/multilingual_routing_in_mixture-of-experts.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings](hierarchical_news_clustering.md)
- [\[ACL 2025\] ZIPA: A Family of Efficient Models for Multilingual Phone Recognition](zipa_a_family_of_efficient_models_for_multilingual_phone_recognition.md)

</div>

<!-- RELATED:END -->
