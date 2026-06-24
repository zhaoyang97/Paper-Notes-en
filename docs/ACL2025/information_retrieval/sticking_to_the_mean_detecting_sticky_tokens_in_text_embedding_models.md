---
title: >-
  [Paper Note] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models
description: >-
  [ACL 2025 Main][Information Retrieval & RAG][Text Embeddings] This paper systematically investigates the "sticky token" phenomenon in text embedding models, where repeating certain anomalous tokens in sentences drags their cosine similarity towards a fixed value. It proposes an efficient detection method, STD, and identifies 868 sticky tokens across 40 checkpoints from 14 model families, revealing performance degradation of up to 50% on downstream tasks.
tags:
  - "ACL 2025 Main"
  - "Information Retrieval & RAG"
  - "Text Embeddings"
  - "Sticky Tokens"
  - "Vocabulary Anomalies"
  - "Tokenization Robustness"
  - "Embedding Model Analysis"
date: 2026-05-08
content_hash: 007961fa8ab23035
---

# Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models

**Conference**: ACL 2025 Main  
**arXiv**: [2507.18171](https://arxiv.org/abs/2507.18171)  
**Code**: [GitHub](https://github.com/March-7/StickyToken)  
**Area**: Information Retrieval  
**Keywords**: Text Embeddings, Sticky Tokens, Vocabulary Anomalies, Tokenization Robustness, Embedding Model Analysis

## TL;DR

This paper systematically investigates the "sticky token" phenomenon in text embedding models, where repeating certain anomalous tokens in sentences drags their cosine similarity towards a fixed value. It proposes an efficient detection method, STD, and identifies 868 sticky tokens across 40 checkpoints from 14 model families, revealing performance degradation of up to 50% on downstream tasks.

## Background & Motivation

**Background**: Transformer-based text embedding models are widely used in NLP tasks, including semantic retrieval, text clustering, and sentence similarity computation. These models typically employ mean pooling or the `[CLS]` token to generate sentence-level embedding representations.

**Limitations of Prior Work**: Researchers have discovered that repeating certain "anomalous" tokens within a sentence can hijack the model's internal representations, dragging the cosine similarity between sentence embeddings towards a fixed value and severely disrupting the normal distribution of the embedding space. This phenomenon has not been systematically studied, and detection tools are lacking.

**Key Challenge**: Text embedding models rely on a tokenizer to convert text into token sequences. However, tokenizers contain many special symbols, unused tokens, and multilingual subword fragments in their vocabularies. These tokens are not sufficiently learned during training, leading to anomalous behaviors during inference. The mean pooling mechanism amplifies the influence of these anomalous tokens—even inserting a small number of sticky tokens can dominate the embedding representation of the entire sentence.

**Goal**: (1) Formally define the concept of sticky tokens; (2) propose an efficient detection method; (3) perform a large-scale analysis of the origins and characteristics of sticky tokens; and (4) evaluate their impact on downstream tasks.

**Key Insight**: The authors observe that if a token is repeatedly inserted into different sentences, the cosine similarity between the sentences converges to a fixed value instead of reflecting the true semantic relationship. This "sticky" behavior violates the basic assumption of embedding models—that semantically similar sentences should have higher similarity.

**Core Idea**: A two-stage detection method (STD) consisting of sentence-level filtering and token-level filtering is proposed to efficiently identify sticky tokens in vocabularies and systematically analyze their causes and consequences.

## Method

### Overall Architecture

STD (Sticky Token Detector) employs a two-stage filtering pipeline: first, it filters out anomalous embedding behaviors possibly containing sticky tokens at the sentence level, and then precisely localizes specific sticky tokens at the token level. The inputs are a text embedding model and its vocabulary, and the output is a list of tokens labeled as sticky.

### Key Designs

1. **Formal Definition of Sticky Tokens**:

    - **Function**: Provides a mathematical definition of sticky tokens to establish a theoretical foundation for detection.
    - **Mechanism**: A token $t$ is defined as a sticky token if and only if, after repeating $t$ for $k$ times and inserting it into any sentence $s$, the cosine similarity between different sentence pairs $(s_i, s_j)$ approaches a fixed constant $\mu_t$, i.e., $\text{cos}(\text{emb}(s_i \oplus t^k), \text{emb}(s_j \oplus t^k)) \to \mu_t$, where $\mu_t$ is independent of the original sentence content.
    - **Design Motivation**: Previously, the concept of "anomalous tokens" was vague and lacked a unified definition. Formalization facilitates subsequent quantitative analysis and automated detection.

2. **Sentence-level Filtering**:

    - **Function**: Rapidly filters out most normal tokens, leaving only suspicious tokens for fine-grained detection.
    - **Mechanism**: For each token in the vocabulary, it is inserted multiple times into a set of seed sentences. Then, the standard deviation of cosine similarity among all sentence pairs after insertion is calculated. If the standard deviation is extremely low (below a threshold), it indicates that the token "drags" different sentences to the same position, and the token is flagged as a suspicious sticky token.
    - **Design Motivation**: Inserting normal tokens does not significantly change the relative distances between sentences, meaning the standard deviation should remain at a normal level. Conversely, sticky tokens dominate the embedding, pulling the standard deviation close to zero.

3. **Token-level Filtering**:

    - **Function**: Performs fine-grained validation on the suspicious tokens selected during sentence-level filtering.
    - **Mechanism**: Verifies suspicious tokens using more sentence pairs and different numbers of repetitions. It observes whether the similarity stably converges to a fixed value as the number of repetitions increases. Tokens that continue to exhibit sticky behavior after multiple verification rounds are finalized.
    - **Design Motivation**: Sentence-level filtering may introduce false positives (e.g., certain low-frequency but normal tokens). Token-level filtering reduces false positives through more rigorous validation.

### Loss & Training

This paper is an analytical work and does not involve model training.

## Key Experimental Results

### Main Results

In 14 model families across 40 checkpoints, the application of STD yielded the following results:

| Model Family | Checkpoints | Sticky Tokens Found | Typical Source |
|----------|---------|----------------|---------|
| BERT Series | 8 | 120+ | Unused `[unused]` tokens |
| RoBERTa Series | 5 | 80+ | Special symbols, multilingual fragments |
| Sentence-BERT | 6 | 90+ | Inherited from backbone models |
| E5 Series | 4 | 70+ | Multilingual subword fragments |
| GTE Series | 4 | 60+ | Untrained vocabulary entries |
| Other Models | 13 | 448+ | Mixed sources |
| **Total** | **40** | **868** | — |

### Ablation Study

Evaluation of the impact of sticky tokens on downstream tasks:

| Task | Metric | No Sticky Tokens | Insert 1 | Insert 5 | Performance Drop |
|------|------|-----------|--------|--------|---------|
| Text Clustering | NMI | Normal | -15% | -35% | Up to 50% |
| Semantic Retrieval | MRR@10 | Normal | -10% | -30% | Significant |
| STS Similarity | Spearman | Normal | -8% | -25% | Linear Growth |

### Key Findings

- **Source Analysis of Sticky Tokens**: Sticky tokens mainly originate from three categories: (1) special or unused entries in the vocabulary (e.g., `[unused]` tokens in BERT), (2) subword fragments from multilingual corpora (such as low-frequency characters in Japanese, Arabic, etc.), and (3) control characters and special symbols.
- **Incoherence between Model Scale and Sticky Token Count**: Larger models do not necessarily have fewer sticky tokens, suggesting that the root of the problem lies in the tokenizer design rather than the model capacity.
- **Attention Analysis**: Sticky tokens receive disproportionately high weights in the attention layers, "attracting" the attention of other tokens, and thereby dominating the final mean pooled representation.

## Highlights & Insights

- **The STD detection method is simple and efficient**: It requires only a few seed sentences and standard deviation calculations to perform detection without accessing the internal weights of the model. This black-box detection approach can be directly applied to quality auditing for any embedding model.
- **Revealing systemic flaws in tokenizer design**: The widespread existence of sticky tokens demonstrates that dominant tokenizers (BPE/WordPiece/SentencePiece) have fundamental issues in handling insufficiently trained subwords, posing a threat to the reliability of embedding models.
- **Vulnerability of the attention mechanism**: The phenomenon where sticky tokens dominate attention weights reveals the vulnerability of the mean pooling + self-attention combination when facing anomalous inputs. This insight is transferable to adversarial attacks and model robustness research.

## Limitations & Future Work

- **Lack of Mitigation Strategies**: The paper focuses on detection and analysis but does not propose effective mitigation methods (such as token filtering, attention regularization, etc.).
- **Limited to Text Embedding Models**: It does not discuss whether similar phenomena exist in generative models (such as the GPT series).
- **Choice of Detection Thresholds**: Standard deviation thresholds must be set manually. Different models may require different thresholds, lacking an adaptive mechanism.
- **Future Directions**: Exploring the integration of sticky token detection as regularization during the tokenizer training phase, or dynamically filtering out anomalous tokens during inference.

## Related Work & Insights

- **vs. Token Anomaly Detection**: Previous token anomaly research has mostly focused on adversarial triggers designed to purposefully attack inputs. This paper reveals that these anomalies are inherent flaws within the model's own vocabulary.
- **vs. Embedding Quality Evaluation**: Benchmarks like MTEB focus on downstream task performance but do not detect structural flaws in the embedding space. STD provides a complementary dimension for quality evaluation.
- The core findings of this paper serve as an actual warning for all systems relying on text embeddings (RAG, semantic search, clustering).

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically define and detect sticky tokens; the phenomenon itself is highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The large-scale analysis across 40 checkpoints offers wide coverage, but lack of mitigation experiments is a drawback.
- **Writing Quality**: ⭐⭐⭐⭐ Clear definitions, systematic analysis, and well-structured.
- **Value**: ⭐⭐⭐⭐ Practical guidance significance for embedding model deployment and tokenizer design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)
- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)

</div>

<!-- RELATED:END -->
