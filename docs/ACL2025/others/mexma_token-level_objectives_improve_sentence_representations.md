---
title: >-
  [Paper Note] MEXMA: Token-level Objectives Improve Sentence Representations
description: >-
  [ACL 2025][Cross-lingual sentence encoder] MEXMA is proposed, a cross-lingual sentence encoder training method that combines sentence-level and token-level objectives: using the sentence representation of one language to predict the masked tokens of another language, while allowing gradients from both sentence and token levels to directly update the encoder, outperforming SONAR and LaBSE on bitext mining and multiple downstream tasks.
tags:
  - "ACL 2025"
  - "Cross-lingual sentence encoder"
  - "token-level objectives"
  - "sentence representations"
  - "multilingual alignment"
  - "masked language model"
date: 2026-05-08
content_hash: 65917d054e375ff3
---

# MEXMA: Token-level Objectives Improve Sentence Representations

**Conference**: ACL 2025  
**arXiv**: [2409.12737](https://arxiv.org/abs/2409.12737)  
**Code**: None  
**Area**: Others  
**Keywords**: Cross-lingual sentence encoder, token-level objectives, sentence representations, multilingual alignment, masked language model

## TL;DR

MEXMA is proposed, a cross-lingual sentence encoder training method that combines sentence-level and token-level objectives: using the sentence representation of one language to predict the masked tokens of another language, while allowing gradients from both sentence and token levels to directly update the encoder, outperforming SONAR and LaBSE on bitext mining and multiple downstream tasks.

## Background & Motivation

Cross-lingual sentence encoders (CLSE) aim to create cross-lingually aligned fixed-length sentence representations. Existing methods face a key challenge:

**Pre-training phase uses token-level objectives**: CLSEs are typically based on pre-trained encoders like XLM-RoBERTa and NLLB, which update representation of each token using token-level objectives like masked language modeling (MLM) during pre-training.

**Fine-tuning phase uses only sentence-level objectives**: During CLSE training (e.g., contrastive learning in LaBSE, translation bottleneck in SONAR), the encoder is updated solely via sentence representations, with no token-level objectives.

**Consequences**: This leads to the degradation of token-level information (particularly lexical information), which in turn hurts sentence representation quality.

**Hypothesis**: Retaining token-level objectives alongside sentence-level objectives during CLSE training can better update the encoder and improve sentence representation quality.

**Differences from existing hybrid methods**:
- DAP: Has token-level objectives but does not use them to update sentence representations.
- RetroMAE: Sentence representations are used to guide token unmasking, but the encoder itself does not receive direct gradients from tokens.

## Method

### Overall Architecture

The architecture of MEXMA is symmetric: given a translation pair of two languages, two views are created for each language (a masked version and a clean version), resulting in a total of four encoder instances (sharing parameters). The core operation is **cross-lingual unmasking**: using the clean sentence representation of language A to predict the masked tokens of language B, and vice versa.

### Key Designs

1. **Cross-Unmasking**:

    - Mask the input of language A at a high ratio (40%), making it difficult for the encoder and MLM head to recover the missing tokens without extra context.
    - Provide the clean sentence vector $S_B$ from language B as additional context, forcing the model to utilize the information in $S_B$ to predict the masked tokens in language A.
    - Symmetric operation: run the reverse process simultaneously (predicting masked tokens of language B using $S_A$).
    - Loss function: $\mathcal{L}_{mlm} = CE([S_B, \hat{A}], A) + CE([S_A, \hat{B}], B)$
    - **Key difference**: Gradients flow through both sentence representations and individual token representations back to the encoder.

2. **Alignment Loss**:

    - Cross-unmasking yields implicit alignment, but is insufficient to force sentences with identical semantics close enough in the embedding space.
    - Use MSE loss to force alignment of sentence representations between the two languages: $\mathcal{L}_{alignment} = MSE(S_A, S_B)$
    - This is a non-contrastive alignment method (since the masking operation prevents representation collapse).

3. **KoLeo Loss**:

    - Address the anisotropy of representations.
    - Based on the Kozachenko-Leonenko differential entropy estimator, it encourages sentence representations to be uniformly distributed in the latent space.
    - $\mathcal{L}_{KoLeo} = -\frac{1}{n}\sum_{i=1}^n \log(d_{n,i})$, where $d_{n,i}$ is the distance between $x_i$ and its nearest neighbor in the batch.

### Loss & Training

The total loss is a weighted combination of the three components:
$$\mathcal{L}_{MEXMA} = \alpha \cdot \mathcal{L}_{alignment} + \beta \cdot \mathcal{L}_{mlm} + \gamma \cdot \mathcal{L}_{K}$$

- The encoder is based on XLM-RoBERTa (base: 277M / large: 559M).
- Training data is a subset of the NLLB-200 corpus, covering 81 languages (all data are paired between English and the other 80 languages).
- 15M to 25M sentences per language, with mined data supplemented for low-resource languages.
- Masking ratio of 40% (optimal range: 30%-60%).

## Key Experimental Results

### Main Results

**Bitext Mining**:

| Model | xsim ↓ | xsim++ ↓ | BUCC F1 ↑ |
|------|--------|----------|-----------|
| DAP | — | — | 98.68 |
| SONAR (766M) | 0.09 | 12.08 | 98.25 |
| LaBSE (471M) | 0.92 | 18.65 | 98.75 |
| **MEXMA (559M)** | **0.06** | **9.60** | **98.93** |

xsim++ absolute gain of 2.48% (vs SONAR), indicating significantly enhanced robustness against hard negative samples.

**Classification Tasks**:

| Model | SentEval | MTEB Average |
|------|----------|-----------|
| SONAR | 85.82 | 63.02 |
| LaBSE | 85.63 | 62.77 |
| **MEXMA** | **86.38** | **65.35** |

**Pairwise Classification (Average Precision)**:

| Model | Average AP |
|------|---------|
| SONAR | 69.70 |
| LaBSE | 68.47 |
| **MEXMA** | **71.55** |

### Ablation Study

| Configuration | xsim ↓ | xsim++ ↓ | SentEval ↑ |
|------|--------|----------|-----------|
| Sentence-level gradient only | 0.15 | 11.37 | 85.06 |
| + Token-level gradient | 0.10 (↓0.05) | 9.67 (↓1.7) | 85.98 (↑0.92) |
| + KoLeo (Full MEXMA) | 0.06 (↓0.04) | 9.60 (↓0.07) | 86.38 (↑0.4) |

**Model Scale Ablation**:

| Model | Parameters | xsim++ ↓ | SentEval ↑ |
|------|--------|----------|-----------|
| MEXMA-base | 277M | 13.03 | 85.30 |
| LaBSE | 471M | 18.65 | 85.63 |
| MEXMA | 559M | 9.60 | 86.38 |
| SONAR | 766M | 12.08 | 85.82 |

MEXMA-base (277M) outpaces LaBSE with only 58.8% of LaBSE's parameters, and closely approaches SONAR (with 2.77 times the parameters).

### Key Findings

1. **Token-level gradients are the core contribution**: Moving from sentence-level gradients only to adding token-level gradients reduces xsim++ by 1.7 percentage points, which is the single largest improvement factor.
2. **Small models are also powerful**: MEXMA-base (277M) achieves an xsim++ of 13.03%, significantly outperforming LaBSE (471M) at 18.65%.
3. **Compatible with contrastive learning**: After replacing the MSE alignment loss with contrastive loss, the token-level gradients of MEXMA still yield significant improvements.
4. **Token embedding analysis**: MEXMA's tokens exhibit strong cross-lingual semantic alignment (97.88% mapping to translation) while retaining more lexical information (1.33% same language vs. 0.13% for SONAR).
5. **STS task is an exception**: The only task where LaBSE outperforms MEXMA, indicating that contrastive loss is more suitable for STS.

## Highlights & Insights

- **Simple yet effective design**: The core idea is highly intuitive—training sentence encoders should not only update encoders through sentence representations but also directly through token-level objectives. In implementation, it only requires allowing gradients to flow through tokens.
- **Symmetric cross-lingual unmasking**: Cleverly achieves two goals simultaneously—forcing sentence vectors to encode sufficient information (for unmasking) and maintaining token representation quality (receiving direct gradients).
- **KoLeo for anti-anisotropy**: Idea borrowed from vision models (DINOv2) to address the anisotropy issue in non-contrastive methods.
- **Token nearest neighbor analysis**: By analyzing the nearest neighbor categories of token embeddings (same language / same sentence / translation / others), it intuitively demonstrates the characteristic differences in token representations of different models.

## Limitations & Future Work

1. **Suboptimal performance on STS**: Non-contrastive alignment is less effective than contrastive methods on Semantic Textual Similarity tasks.
2. **Support for only 81 languages**: Narrower coverage compared to SONAR's 200 languages.
3. **Dependence on training data**: Heavily dependent on NLLB mined data; data quality for low-resource languages can be inconsistent.
4. **Generative downstream tasks unexplored**: Mainly evaluated on classification and mining tasks; performance on generative tasks remains unknown.
5. **Adaptability of masking ratio**: Currently uses a fixed 40% mask rate; different language pairs or sentence lengths might have different optimal values.

## Related Work & Insights

- **Relationship with RetroMAE**: RetroMAE first proposed an IR method using sentence representations to guide token unmasking, but the encoder does not receive token-level gradients; MEXMA extends this by allowing gradients to flow bi-directionally.
- **Relationship with SONAR**: SONAR uses a translation bottleneck to realize alignment, but the bottleneck prevents decoder gradients from directly updating tokens; MEXMA has no such restriction.
- **Inspirations**: The co-training paradigm of token-level and sentence-level objectives can be generalized to other tasks requiring hierarchical representations (e.g., document retrieval, passage representation).

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea is simple yet profound—allowing gradients to flow from both token and sentence levels simultaneously.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple benchmarks (xsim/xsim++/BUCC/MTEB/SentEval), complete ablation studies, model scale analysis, and token embedding analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, comprehensive experiments, and analysis diving deep to the token level.
- Value: ⭐⭐⭐⭐⭐ Establishes a new SOTA on multiple key benchmarks, and the method can be combined with contrastive learning, showing strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)

</div>

<!-- RELATED:END -->
