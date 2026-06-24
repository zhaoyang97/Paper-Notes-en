---
title: >-
  [Paper Note] GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion
description: >-
  [ACL 2025][Recommender Systems][Generative Recommendation] This work proposes GRAM, a generative recommendation framework. By utilizing **semantic-to-lexical translation**, it encodes implicit hierarchical taxonomic and collaborative relationships of items into the LLM vocabulary space. Employing **multi-granular late fusion**, it independently encodes different-grained prompts and fuses them at the decoder side, yielding Recall@5 improvements of 11.5–16.0% and NDCG@5 improve…
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "Generative Recommendation"
  - "Multi-Granular Fusion"
  - "Semantic Translation"
  - "LLM Recommendation"
  - "Collaborative Filtering"
date: 2026-05-08
content_hash: 01154b7c3e383954
---

# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion

**Conference**: ACL 2025  
**arXiv**: [2506.01673](https://arxiv.org/abs/2506.01673)  
**Code**: [https://github.com/skleee/GRAM](https://github.com/skleee/GRAM)  
**Area**: Recommender Systems  
**Keywords**: Generative Recommendation, Multi-Granular Fusion, Semantic Translation, LLM Recommendation, Collaborative Filtering  

## TL;DR

This work proposes GRAM, a generative recommendation framework. By utilizing **semantic-to-lexical translation**, it encodes implicit hierarchical taxonomic and collaborative relationships of items into the LLM vocabulary space. Employing **multi-granular late fusion**, it independently encodes different-grained prompts and fuses them at the decoder side, yielding Recall@5 improvements of 11.5–16.0% and NDCG@5 improvements of 5.3–13.6% across four benchmarks.

## Background & Motivation

**Background**: Generative Recommendation models recommendation as a text-to-text generation task, leveraging the pre-trained knowledge of LLMs to directly generate item identifiers. Methods such as P5, TIGER, LC-Rec, and IDGenRec have validated the effectiveness of this paradigm across multiple benchmarks.

**Limitations of Prior Work**: (a) **Absence of implicit item relationships**—LLMs excel at understanding textual semantics, but the hierarchical taxonomic relationships among items (e.g., "skincare $\to$ face cream $\to$ moisturizing cream") and collaborative filtering relationships (co-occurrence patterns) are difficult to express directly in natural language. Existing methods either completely ignore these relationships or introduce out-of-vocabulary (OOV) tokens via quantization schemes like RQ-VAE, which decouples them from linguistic semantics. (b) **Bottleneck of long-text information**—Items contain rich metadata such as titles, brands, descriptions, and categories. However, concatenating all item information from the historical sequence into a single prompt leads to excessively long sequences (averaging 1,440 tokens in Amazon Beauty). The quadratic complexity of Transformers makes this unscalable, while truncation or keyword extraction inevitably suffers from information loss.

**Key Challenge**: Recommendation tasks require two fundamentally different types of knowledge: **collaborative filtering signals** (user-item interaction patterns) and **semantic information** (item text descriptions). Existing LLM frameworks lack a mechanism to uniformly represent and efficiently utilize both.

**Key Insight**: This paper proposes "semantic-to-lexical translation" to encode recommendation signals into LLM-native vocabulary tokens (as opposed to external embeddings). It also leverages "multi-granular late fusion" to independently encode different information sources and aggregate them during decoding, preserving both information integrity and computational efficiency.

**Core Idea**: Encoding item relationships via semantic translation + preserving rich semantics via multi-granular late fusion = superior generative recommendation.

## Method

### Overall Architecture

GRAM is based on the T5 encoder-decoder architecture and consists of two core components working in tandem: **(1) Semantic-to-Lexical Translation**—In the pre-training preprocessing phase, this encodes hierarchical taxonomic relations and collaborative filtering relations of items into textual IDs within the LLM vocabulary space; **(2) Multi-granular Late Fusion**—During inference, independent encoders are used to separately encode coarse-grained user prompts and fine-grained item prompts. Fusion is performed only within the decoder's cross-attention, avoiding the information loss and efficiency issues of early concatenation.

### Key Designs

1. **Semantic-to-Lexical Translation**

    - **Hierarchical Semantic Indexing**: Performs hierarchical $k$-means clustering on item embeddings (recursively splitting until the cluster size is $< c$ or the maximum depth $l$ is reached). Each cluster is named using the most representative vocabulary tokens selected via TF-IDF, generating hierarchical textual IDs like "soap-mild-mango". Items sharing prefixes are semantically related, and autoregressive decoding generates them step-by-step from coarse to fine.
    - **Collaborative Semantic Textualization**: Utilizes a pre-trained SASRec model to obtain item embeddings, identifies the top-$k$ most similar items for each item, and concatenates their hierarchical IDs into an additional attribute: "similar items: soap-essence-argan, shampoo-essence-argan, ...", injecting collaborative filtering signals in textual form.
    - **Design Motivation**: Utilizing the LLM's existing vocabulary instead of OOV tokens preserves pre-trained knowledge, while the hierarchical structure ensures semantically consistent recommendations.

2. **Multi-granular Late Fusion**

    - **Coarse-grained User Prompt**: Concatenates the item IDs from the user's historical sequence (in reverse chronological order to prevent truncation of recent items) into "What would the user purchase after {ID sequence}?", capturing global preferences and sequential dependencies.
    - **Fine-grained Item Prompt**: Constructs independent prompts for each item in the history, containing all attributes such as its ID, collaborative semantics, title, brand, and description to preserve detailed information.
    - **Late Fusion Decoding**: Employs T5 encoders to separately encode each prompt, adds positional embeddings, and concatenates them into a unified key-value matrix to be fused during decoder cross-attention. Information linking (retaining the ID in item prompts) bridges coarse-grained and fine-grained information.
    - **Design Motivation**: Early fusion (concatenating input texts) leads to $\mathcal{O}(n^2)$ complexity and attention dilution. Late fusion allows each information source to be fully encoded and only interact during decoding.

3. **Loss & Training**

    - **Training**: Uses standard sequence-to-sequence cross-entropy loss with teacher forcing, aiming to generate the hierarchical textual ID of the next item.
    - **Inference**: Pre-computes all item prompt encodings during an offline phase; during the online phase, only the user prompt is encoded, followed by constrained beam search (using a prefix tree to ensure valid ID generation) with a beam size of 50.
    - **Design Motivation**: Two-phase inference offlines the computation of fine-grained item encodings, drastically reducing online latency.

## Key Experimental Results

### Main Results (Four Benchmark Datasets × 14 Baselines)

| Method | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 | Sports R@5 | Sports N@5 | Yelp R@5 | Yelp N@5 |
|------|-----------|-----------|---------|---------|------------|------------|---------|---------|
| FDSA (best trad.) | 0.0570 | 0.0412 | 0.0619 | 0.0455 | 0.0283 | 0.0201 | 0.0331 | 0.0218 |
| LC-Rec | 0.0503 | 0.0352 | 0.0543 | 0.0385 | 0.0259 | 0.0175 | 0.0341 | 0.0235 |
| IDGenRec | 0.0463 | 0.0328 | 0.0462 | 0.0323 | 0.0273 | 0.0186 | 0.0310 | 0.0219 |
| **GRAM** | **0.0641** | **0.0451** | **0.0718** | **0.0516** | **0.0375** | **0.0256** | **0.0476** | **0.0326** |
| **Gain** | +12.4% | +9.5% | +16.0% | +13.6% | +11.5% | +5.3% | +12.3% | +8.1% |

### Ablation Study (Beauty / Toys, R@5 / N@5)

| Configuration | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 |
|------|-----------|-----------|---------|---------|
| GRAM (full) | 0.0641 | 0.0451 | 0.0718 | 0.0516 |
| w/o Hierarchical ID | 0.0605 | 0.0438 | 0.0630 | 0.0466 |
| w/o Collaborative Semantics | 0.0567 | 0.0396 | 0.0589 | 0.0406 |
| w/o User Prompt | 0.0634 | 0.0443 | 0.0709 | 0.0510 |
| w/o Item Prompt | 0.0582 | 0.0404 | 0.0574 | 0.0397 |
| w/o Info Linking | 0.0628 | 0.0441 | 0.0702 | 0.0507 |
| w/o Positional Embeddings | 0.0563 | 0.0395 | 0.0665 | 0.0465 |

### ID Type Comparison

| ID Type | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 |
|---------|-----------|-----------|---------|---------|
| Hierarchical ID | **0.0641** | **0.0451** | **0.0718** | **0.0516** |
| RQ-VAE ID | 0.0605 | 0.0432 | 0.0662 | 0.0477 |
| Keyword ID | 0.0605 | 0.0438 | 0.0630 | 0.0466 |
| Category ID | 0.0512 | 0.0367 | 0.0465 | 0.0350 |
| Title ID | 0.0478 | 0.0342 | 0.0564 | 0.0412 |

### Key Findings

- **Collaborative semantics contribute the most**: Removing collaborative filtering attributes drops NDCG@5 by 10.8–27.2%, proving that LLMs lack behavioral co-occurrence information.
- **Item prompts > User prompts**: Removing fine-grained item prompts drops NDCG@5 by up to 30.1%, whereas removing user prompts only drops it by 1.2%, showing that detailed information is much more critical for recommendation.
- **Significant gains on tail items**: For low-frequency (bottom 80%) items, GRAM improves Recall@5 by 42.6% and NDCG@5 by 47.8% compared to the best baseline—indicating that semantic translation effectively mitigates data sparsity.
- **Hierarchical IDs outperform RQ-VAE**: Utilizing native LLM vocabulary rather than OOV tokens better exploits pre-trained knowledge, yielding an 8.2% improvement in NDCG@5.

## Highlights & Insights

- **"Translation" rather than "Embedding"**—Translating recommendation signals into known LLM vocabulary tokens completely bypasses the semantic gap of OOV tokens, offering a more elegant solution than RQ-VAE/codebook.
- **Engineering Wisdom of Late Fusion**—Encoding each item independently and fusing them at the decoder side preserves comprehensive information while avoiding the quadratic complexity of over-long sequences. Furthermore, item encodings can be pre-computed offline.
- **Orthogonal and Complementary Innovations**—Semantic translation addresses "what information to encode," while late fusion addresses "how to utilize it efficiently." Both are indispensable.
- **Huge Improvement on Tail Items** (Recall@5 +42.6%) highlights the potential of generative recommendation in long-tail scenarios.

## Limitations & Future Work

- Hierarchical clustering relies heavily on the quality of text embeddings and is sensitive to the choice of embedding models.
- Validated only on T5-small; performance on larger-scale LLMs remains unknown.
- The collaborative filtering model (SASRec) requires pre-training, increasing pipeline complexity.
- Evaluated entirely offline; lacks validation via online A/B testing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of embedding recommendation signals into the LLM vocabulary space via semantic translation is novel and intuitive. Multi-granular late fusion is an ingenious adaptation of Fusion-in-Decoder (FiD) for the recommendation domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on four datasets against 14 baselines (6 traditional + 8 generative), complete with extensive ablation studies, ID type comparisons, head/tail analyses, and hyperparameter sensitivity analyses.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, visualizations in Figures 1–5 facilitate understanding, and methodological derivations are comprehensive.
- **Value**: ⭐⭐⭐⭐ Both components can be independently reused, providing a substantial step forward for the generative recommendation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](../../AAAI2026/recommender/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](../../ACL2026/recommender/hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](../../ICML2025/recommender/parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)

</div>

<!-- RELATED:END -->
