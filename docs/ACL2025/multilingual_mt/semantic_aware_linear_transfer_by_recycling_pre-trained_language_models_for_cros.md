---
title: >-
  [Paper Note] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer
description: >-
  [ACL 2025][Multilingual & Machine Translation][cross-lingual transfer] This paper proposes SALT (Semantic Aware Linear Transfer). By constructing independent least-squares transformation matrices for each non-shared vocabulary token based on semantically similar shared token pairs, it transfers the rich embedding representations of a target language PLM to the embedding space of an English-centric LLM. It outperforms existing methods across downstream tasks…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "cross-lingual transfer"
  - "embedding initialization"
  - "vocabulary replacement"
  - "PLM recycling"
  - "linear least squares"
date: 2026-05-08
content_hash: a972f79af791c62a
---

# Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer

**Conference**: ACL 2025  
**arXiv**: [2505.10945](https://arxiv.org/abs/2505.10945)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: cross-lingual transfer, embedding initialization, vocabulary replacement, PLM recycling, linear least squares

## TL;DR

This paper proposes SALT (Semantic Aware Linear Transfer). By constructing independent least-squares transformation matrices for each non-shared vocabulary token based on semantically similar shared token pairs, it transfers the rich embedding representations of a target language PLM to the embedding space of an English-centric LLM. It outperforms existing methods across downstream tasks, continual pre-training convergence speed, and cross-lingual understanding.

## Background & Motivation

**Background**: Although current mainstream LLMs (e.g., Gemma, XGLM) include multilingual vocabularies, their training data is predominantly English, resulting in limited utility in target languages. A large number of target-language-irrelevant tokens in their embedding layers wastes parameter space.

**Limitations of Prior Work**: Existing cross-lingual transfer methods (e.g., FOCUS, OFA) initialize new vocabulary embeddings through weighted averages of source model embeddings. However, the source model embeddings themselves are insufficiently learned for target languages, making it difficult to capture deep target-language semantics using only these shallow representations.

**Key Challenge**: Vocabulary replacement requires high-quality embedding initialization. However, the target language representations provided by both the source model and external static word vectors (e.g., fastText) are insufficiently rich, making knowledge alignment with the upper layers of the model difficult.

**Goal**: Introduce target-language specialized PLMs (e.g., small-scale pre-trained models such as Korean GPT, Arabic BERT) as embedding sources. Their pre-training on target languages allows the embeddings to contain richer semantic information.

**Key Insight**: "Recycle" existing target-language PLMs. Although these models are small in parameter scale, they have undergone sufficient target-language pre-training, resulting in embedding layers that are higher in quality than the target-language embeddings of multilingual LLMs.

**Core Idea**: Fit an independent linear regression line (least squares) from the PLM embedding space to the LLM embedding space for each non-shared token. By determining mapping relationships based on the semantically most similar shared token pairs, this token-level personalized transfer preserves the semantic richness of PLM embeddings.

## Method

### Overall Architecture

SALT completes embedding transfer in five steps: (1) extracting auxiliary subword embeddings using fastText; (2) identifying the shared vocabulary $V_{\text{shared}}$ between the source vocabulary $V_s$ and target vocabulary $V_t$, and calculating semantic similarity between non-shared and shared tokens; (3) dynamically selecting the top-k nearest-neighbor shared tokens using Sparsemax; (4) fitting an independent transformation matrix $X_{t_i}$ for each non-shared token using the least-squares method and performing embedding mapping; and (5) performing language-adaptive continual pre-training on target-language corpora to align the embedding layer with the upper-layer weights.

### Key Designs

1. **Semantic-Aware Token-Level Linear Transformation**: The core formulation is $\arg\min_{X \in \mathbb{R}^{h_t \times h_s}} \|E'_{t_i} X_{t_i} - E'_{s_i}\|$, where $E'_{t_i}$ and $E'_{s_i}$ are the stacked embedding matrices in the PLM and LLM, respectively, of the top-k nearest-neighbor shared tokens selected for the non-shared token $v_{t_i}$. The closed-form solution is $X_{t_i} = E'^{+}_{t_i} \cdot E'_{s_i}$ (pseudo-inverse). Each target token has an independent transformation matrix, which is the key distinction from global linear mapping methods.
2. **Sparsemax-based Dynamic k Selection**: For each non-shared token, fastText is first used to compute its cosine similarity set $C_{t_i}$ with all shared tokens. Sparsemax is then applied to $C_{t_i}$ (setting the weights of irrelevant tokens to zero) to dynamically determine the number of nearest neighbors $k$ for each token, preventing the introduction of noisy tokens.
3. **PLM Embeddings as Transfer Source**: Unlike FOCUS/OFA, which rely solely on source LLM embeddings, SALT utilizes the embedding layer of target-language-specific PLMs, which have already learned rich semantic representations during target-language pre-training. Shared tokens directly copy the source LLM embeddings, while non-shared tokens are mapped from the PLM embedding space.

### Engineering Details

- The source models are Gemma-2b (256K vocabulary) and XGLM-1.7b (256K vocabulary).
- Target PLMs cover three architectures (BERT, GPT, T5) to verify the architectural generalisability of the method.
- fastText provides auxiliary similarity computation across vocabularies; rare tokens not in fastText are initialized using a normal distribution from the source embeddings.
- Continual pre-training uses Wikipedia corpora for each language (8M sentences) with a CLM objective.

## Key Experimental Results

### Knowledge Benchmark Alignment (Gemma Source Model, Evaluated Directly After Initialization)

| Method | German Avg | Arabic Avg | Vietnamese Avg |
|---|---|---|---|
| Multivariate | 31.02 | 29.03 | 33.21 |
| FOCUS | 29.82 | 28.17 | 32.76 |
| OFA | 30.70 | 29.47 | 32.52 |
| **SALT (Ours)** | **32.71** | **30.64** | **34.66** |

SALT achieves the best average scores across all languages and source models. German HellaSwag reaches 42.58, which is over 3 points higher than the runner-up, demonstrating its superior ability to capture target-language logical flow.

### Target Language Reading Comprehension (Gemma After Continual Pre-training, EM/F1)

| Method | German MLQA | Arabic MLQA | Vietnamese MLQA | German XQuAD | Vietnamese XQuAD |
|---|---|---|---|---|---|
| Multivariate | 19.70/31.68 | 10.35/25.37 | 23.44/42.16 | 22.23/34.62 | 31.01/51.74 |
| FOCUS | 25.66/37.97 | 12.63/25.63 | 27.17/44.77 | 27.21/39.00 | 35.71/54.65 |
| OFA | 23.22/37.05 | 16.49/33.52 | 25.06/45.14 | 26.52/38.99 | 35.88/55.07 |
| **SALT** | **28.07/41.47** | **18.33/36.36** | **29.70/49.63** | **31.28/44.41** | **36.30/56.90** |

### Cross-Lingual Understanding (MLQA English-Target Mixed Setting, Gemma)

| Method | English $\to$ Target Avg EM/F1 | Target $\to$ English Avg EM/F1 |
|---|---|---|
| FOCUS | 23.11/32.52 | 21.92/34.70 |
| OFA | 24.66/35.99 | 20.05/33.93 |
| **SALT** | **29.22/41.40** | **24.29/38.62** |

SALT outperforms the runner-up OFA by approximately 13% in cross-lingual F1, demonstrating that the shared bilingual space between English and the target language within the PLM embeddings is effectively preserved.

### Key Findings

- Continual pre-training convergence curves (Figure 2) show that SALT converges the fastest and reaches the lowest final loss in all scenarios.
- FOCUS/OFA occasionally underperform simple multivariate Gaussian initialization (Multivariate), suggesting that relying solely on weighted averages of source embeddings can be detrimental.
- SALT also outperforms other methods in preserving English capabilities, despite undergoing continual pre-training only on the target language.
- PLMs of different architectures (BERT/GPT/T5) can all serve as target transfer sources with minor performance differences; T5 even outperforms GPT in certain languages.

## Highlights & Insights

- **PLM Recycling Concept**: Repurpose the embeddings of small-scale PLMs in the LLM era—these "outdated" models possess irreplaceable representation advantages in target languages.
- **Token-Level Personalized Mapping**: Fitting a transformation matrix independently for each token is more fine-grained than a global mapping, while maintaining manageable computational costs (as least squares has a closed-form solution).
- **Significant Cross-Lingual Advantage**: The largest improvements are observed in English-target mixed scenarios, indicating that the bilingual shared space of the PLM embeddings is effectively utilized.
- **Architecture-Agnostic**: Embeddings from BERT, GPT, and T5 can all serve as transfer sources, offering great flexibility.

## Limitations & Future Work

- Dependency on the existence of a target language PLM: For extremely low-resource languages (with no dedicated PLM), SALT cannot be applied directly.
- Validated only on 2B-scale models; not extended to LLMs of 7B+ parameters. The effectiveness and efficiency on larger models remain to be verified.
- The performance after instruction tuning is not evaluated; the study focuses solely on the continual pre-training stage of the base model.
- Only tested on three languages (German, Arabic, Vietnamese); while covering diverse language families, the linguistic coverage is limited.

## Related Work & Insights

### vs FOCUS (Dobler & De Melo, 2023)
FOCUS constructs target embeddings from source model embeddings through weighted averages guided by fastText, meaning it remains fundamentally limited by the source model's degree of learning on the target language. SALT introduces target-language PLM embeddings as a richer information source and replaces the global weighted average with independent linear regressions. In experiments, FOCUS occasionally underperforms random initialization in several scenarios, while SALT consistently performs best.

### vs OFA (Liu et al., 2024)
OFA adopts a strategy similar to FOCUS and introduces SVD-based dimensionality reduction, but its core remains the weighted average of the source model embeddings. SALT fundamentally changes the embedding source (relying on the PLM instead of the source LLM), outperforming OFA by a wide margin of approximately 13% in cross-lingual F1 for cross-lingual understanding.

### vs Vocabulary Expansion Methods (Cui et al., 2023; Zhao et al., 2024)
Vocabulary expansion preserves the source vocabulary and adds target tokens, which is applicable but increases the model size. SALT adopts a vocabulary replacement strategy, completely replacing the source vocabulary with the target vocabulary. This keeps the model size unchanged or even reduces it, making it more suitable for resource-constrained scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ PLM embedding recycling combined with token-level semantic-aware linear mapping offers a novel perspective and an elegant methodology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation using 3 languages $\times$ 2 source models $\times$ 3 PLM architectures $\times$ various tasks, as well as thorough convergence curve and cross-lingual analyses.
- **Writing Quality**: ⭐⭐⭐⭐ The five-step methodology is clearly described, Figure 1 provides an accurate summary, and the derivation of mathematical formulas is complete.
- **Value**: ⭐⭐⭐⭐ Provides an effective pathway for repurposing PLMs in the LLM era, offering high practicality under cross-lingual transfer scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)

</div>

<!-- RELATED:END -->
