---
title: >-
  [Paper Note] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval
description: >-
  [ACL 2025][Information Retrieval & RAG][Low-resource languages] For the low-resource and morphologically rich language Amharic, this paper proposes dense retrieval models based on pre-trained Amharic BERT/RoBERTa and a ColBERT late-interaction model. These achieve substantial improvements in passage retrieval with parameters far fewer than those of multilingual baselines, establishing the first systematic retrieval benchmark for the language.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Low-resource languages"
  - "dense retrieval"
  - "text embedding"
  - "Amharic"
  - "ColBERT"
date: 2026-05-08
content_hash: d28fbed80262c08a
---

# Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval

**Conference**: ACL 2025  
**arXiv**: [2505.19356](https://arxiv.org/abs/2505.19356)  
**Code**: [Yes](https://github.com/kidist-amde/amharic-ir-benchmarks)  
**Area**: NLP / Information Retrieval  
**Keywords**: Low-resource languages, dense retrieval, text embedding, Amharic, ColBERT

## TL;DR

For the low-resource and morphologically rich language Amharic, this paper proposes dense retrieval models based on pre-trained Amharic BERT/RoBERTa and a ColBERT late-interaction model. These achieve substantial improvements in passage retrieval with parameters far fewer than those of multilingual baselines, establishing the first systematic retrieval benchmark for the language.

## Background & Motivation

Neural retrieval methods have achieved great success in high-resource languages like English. However, for morphologically complex, low-resource languages like Amharic, their performance remains suboptimal due to data scarcity and poor tokenization. Amharic is the working language of the Federal Government of Ethiopia, written in Ge'ez script (an abugida). Its templatic morphology allows a root word to derive numerous word forms through affixes and vowel variations. This poses severe challenges for traditional vocabulary-matching retrieval (BM25) and general multilingual embedding models:

- **Over-tokenization**: Multilingual tokenizers over-segment Amharic words into semantically meaningless subwords, disrupting semantic representations.
- **Insufficient cross-lingual transfer**: General multilingual models are not optimized for the morphological features of Amharic, a deficiency that cannot be compensated for even with a larger parameter size.
- **Lack of systematic benchmarks**: Previously, there was no systematic comparative evaluation of sparse and dense retrieval methods specifically for Amharic.

This paper aims to fill this gap by developing Amharic-specific retrieval embedding models and establishing the first retrieval benchmark.

## Method

### Overall Architecture

The authors design two types of schemes: a bi-encoder-based dense retrieval model and a ColBERT-based late-interaction retrieval model. The core idea is to leverage existing pre-trained Amharic language models (BERT and RoBERTa) as backbone networks, fine-tuning them on Amharic query-passage pairs via contrastive learning to obtain high-quality text embeddings.

### Key Designs

1. **Three Amharic Embedding Models of Different Sizes**:

    - **RoBERTa-Base-AM-Embed (110M)**: 12-layer Transformer, hidden dimension 768, based on XLM-RoBERTa, offering the strongest contextual representation capabilities.
    - **RoBERTa-Medium-AM-Embed (42M)**: 8-layer Transformer, hidden dimension 512, targeting latency-sensitive and resource-constrained scenarios.
    - **BERT-Medium-AM-Embed (40M)**: Also 8 layers / 512 dimensions, based on the original BERT architecture.
   
   *Design Motivation*: To provide different tradeoffs between accuracy and efficiency, proving that even compact models can outperform large-scale multilingual models.

2. **Embedding Vector Generation**: Mean pooling is performed on the last hidden state of the encoder to obtain a fixed-length vector, followed by L2 normalization to support cosine similarity calculations. The context length is fixed at 512 tokens.

3. **ColBERT Late-Interaction Model**: This model retains token-level interactions, where queries and passages are encoded separately and their relevance score is computed via MaxSim (Maximum Similarity) pooling. This offers finer-grained matching capability than single-vector matching while maintaining inference efficiency. ColBERT uses RoBERTa-Medium-Amharic as the encoder backbone.

### Loss & Training

- **Multiple Negatives Ranking Loss (MNRL)**: Employs in-batch negative contrastive loss to encourage the model to assign higher similarity scores to positive pairs.
- Training parameters: 4 epochs, AdamW optimizer, lr=5e-5, batch size=128, cosine learning rate decay.
- ColBERT additionally uses 8 negatives sampled from the top-150 retrieval results, lr=1e-5, batch size=32.

## Key Experimental Results

### Main Results: Amharic Embeddings vs. Multilingual Baselines

| Model | Parameters | MRR@10 | NDCG@10 | Recall@10 | Recall@100 |
|------|--------|--------|---------|-----------|------------|
| gte-modernbert-base | 149M | 0.019 | 0.023 | 0.033 | 0.067 |
| multilingual-e5-large | 560M | 0.672 | 0.709 | 0.825 | 0.931 |
| Arctic Embed 2.0 | 568M | 0.659 | 0.701 | 0.831 | 0.942 |
| BERT-Medium-AM-Embed | 40M | 0.682 | 0.720 | 0.843 | 0.954 |
| RoBERTa-Medium-AM-Embed | 42M | 0.735 | 0.771 | 0.884 | 0.971 |
| **RoBERTa-Base-AM-Embed** | **110M** | **0.775** | **0.808** | **0.913** | **0.979** |

### Sparse vs. Dense Retrieval Benchmarks

| Type | Model | MRR@10 | Recall@10 | Recall@100 |
|------|------|--------|-----------|------------|
| Sparse Retrieval | BM25-AM | 0.657 | 0.774 | 0.871 |
| Dense Retrieval (Bi-encoder) | RoBERTa-Base-AM-Embed | 0.775 | 0.913 | 0.979 |
| Dense Retrieval (Late-interaction) | **ColBERT-RoBERTa-Base** | **0.843** | **0.939** | **0.979** |

### Ablation Study: Fine-Tuning Multilingual Models

| Model | MRR@10 | Recall@10 | Recall@100 |
|------|--------|-----------|------------|
| Arctic Embed 2.0 (Zero-shot) | 0.659 | 0.831 | 0.942 |
| Arctic Embed 2.0-AM (Fine-tuning) | 0.827 | 0.942 | 0.985 |

### Key Findings

1. **Language-Specific Models Outperform Multilingual Models**: The 110M parameter RoBERTa-Base-AM-Embed outperforms the 568M parameter Arctic Embed 2.0 by 17.6% under MRR@10. The 42M compact variant still outperforms all multilingual baselines.
2. **Tokenization Quality is the Decisive Factor**: gte-modernbert-base has a subword fertility as high as 13.80 (meaning an average of 13.8 subwords per word), resulting in an MRR of only 0.019; RoBERTa-Base-AM has a fertility of only 1.46, with an MRR of 0.775, demonstrating an extremely strong correlation.
3. **ColBERT Late-Interaction Achieves Best Results**: ColBERT-RoBERTa-Base-Amharic reaches 0.843 MRR@10, representing a 28.3% relative improvement over BM25, demonstrating the value of token-level interaction for morphologically rich languages.
4. **Fine-tuning Multilingual Models is Effective but Inferior to Dedicated Models**: After fine-tuning Arctic Embed, its MRR rises from 0.659 to 0.827, but it is still inferior to ColBERT.

## Highlights & Insights

- **Extremely High Parameter Efficiency**: The 42M RoBERTa-Medium-AM outperforms 568M multilingual models, showing that for low-resource languages, language-tokenization alignment is far more effective than brute-force parameter scaling.
- **Subword Fertility as a Diagnostic Metric**: Using tokenization quality (average subwords per word) as a predictor of retrieval performance provides a simple diagnostic tool for language suitability.
- **Qualitative Error Analysis Reveals Deep Challenges**: The model still struggles with fine-grained semantic comprehension, such as negation and tense shifts—for example, matching the query "Protests did not take place" to a passage stating "Protests took place."

## Limitations & Future Work

- The dataset is built solely on heuristic headline-body pairings in the news domain, lacking human relevance annotations, which introduces noise to evaluations.
- The pre-training corpus contains only 300M tokens, far less than English models (30B+), limiting generalization.
- Morphological analyzers or morpheme segmenters are not integrated, relying solely on standard tokenizers and fine-tuning.
- Only validated in the news domain; performance in other domains, such as medical or legal, remains unknown.
- Parameter-efficient fine-tuning (like LoRA) is not explored; full-parameter fine-tuning might be less flexible in cross-lingual transfer scenarios.

## Related Work & Insights

This work continues an important direction of NLP research for low-resource languages. Previous pre-trained models for African languages, such as AfriBERTa and SERENGETI, provided foundations for cross-lingual transfer but did not systematically evaluate retrieval tasks. Unsupervised contrastive learning frameworks like Contriever perform well in zero-shot retrieval but do not cover the morphological complexity of Amharic. This work suggests that for any morphologically rich low-resource language (e.g., Arabic dialects, Turkish), language-specific tokenizers and embedding models are likely to be much more effective than general multilingual approaches.

## Rating

- **Novelty**: ⭐⭐⭐ — The technical pipeline (bi-encoder + ColBERT + contrastive learning) is not new, but systematic adaptation and benchmarking for Amharic have high value in filling this gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experimental design, covering multiple model sizes, various retrieval paradigms, tokenization analysis, fine-tuning comparisons, and qualitative error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, research-question driven, with rich charts and tables.
- **Value**: ⭐⭐⭐⭐ — Outlines highly actionable practical guidance and provides reproducible benchmarks and open resources (data, code, models) for the low-resource IR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)

</div>

<!-- RELATED:END -->
