---
title: >-
  [Paper Note] TokAlign: Efficient Vocabulary Adaptation via Token Alignment
description: >-
  [ACL 2025][LLM Pretraining][Vocabulary Substitution] This paper proposes TokAlign, which learns a one-to-one mapping matrix between two vocabularies based on token co-occurrence information, efficiently replacing LLM vocabularies to realize cross-lingual knowledge transfer and cross-model token-level distillation.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Vocabulary Substitution"
  - "Token Alignment"
  - "Co-occurrence Matrix"
  - "Cross-lingual Transfer"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: ef1a1527c2aee043
---

# TokAlign: Efficient Vocabulary Adaptation via Token Alignment

**Conference**: ACL 2025  
**arXiv**: [2506.03523](https://arxiv.org/abs/2506.03523)  
**Area**: LLM Pre-training  
**Keywords**: Vocabulary Substitution, Token Alignment, Co-occurrence Matrix, Cross-lingual Transfer, Knowledge Distillation  

## TL;DR

This paper proposes TokAlign, which learns a one-to-one mapping matrix between two vocabularies based on token co-occurrence information, efficiently replacing LLM vocabularies to realize cross-lingual knowledge transfer and cross-model token-level distillation.

## Background & Motivation

### Core Problem

LLM tokenizers are often highly inefficient in new domains or languages—for instance, LLaMA3's tokenizer produces text sequences for Armenian that are 3.95 times longer than for English (given the same number of bytes). This directly leads to:

**Decreased Training and Inference Speed**: Low compression rate = longer token sequences = slower processing.

**Blocked Cross-Model Knowledge Transfer**: Different LLMs use different vocabularies, preventing token-level distillation or ensembling.

**Prohibitive Re-training Costs**: Training an LLM from scratch for a new tokenizer is extremely expensive.

### Limitations of Prior Work

- **Focus**: Relies on overlapping tokens between two vocabularies to initialize new tokens, which performs poorly when vocabulary overlap is minimal (with an initial perplexity up to 2.9e5).
- **WECHSEL**: Requires bilingual dictionaries and external word embeddings, making it inapplicable to resource-poor languages lacking dictionaries.
- **ZeTT**: Trains an auxiliary hypernetwork to generate token parameters, which incurs prohibitive computational overhead (requiring 661 GPU hours for Pythia-2.8B).
- **OFA**: Concocts a complex pipeline involving matrix factorization and multilingual word embeddings.

## Method

### Overall Architecture

TokAlign approaches vocabulary alignment from the perspective of token-token co-occurrence, executing alignment in three steps:

**Step 1: Corpus Tokenization**
- Uses a mixed corpus (40% CulturaX + 30% The Stack + 30% Proof-Pile-2).
- Tokenizes the same corpus separately with the source and target tokenizers.
- Adopts a standard corpus scale of 1B tokens.

**Step 2: Token Representation Learning**
- Learns token representation vectors from the respective token sequences using GloVe.
- Rationale for choosing GloVe over CBOW/FastText: GloVe leverages **global** statistical co-occurrence information.
- Extremely low training overhead: under 2 hours on a 128-core CPU server.

**Step 3: Token Alignment**
- Computes pairwise cosine similarity between source and target vocabularies based on the learned token representations.
- Builds an alignment matrix $M_{s \to t}$ to match each source token with its most similar target token.
- Directly maps tokens shared by both vocabularies without alignment.

### Key Designs

**1. Alignment Quality Evaluation Metrics**

Two new metrics are proposed to evaluate the quality of the alignment matrix:
- **BLEU-1**: Compares the target token ID sequence with the source token ID sequence converted via the alignment matrix (text-matching perspective).
- **BERTScore**: Decodes the converted token sequence back to text and calculates the semantic similarity with the original text.

These two metrics are positively correlated with the initial pre-training loss, serving as prior indicators of alignment quality.

**2. Progressive Adaptation**

Fine-tuning after vocabulary replacement is divided into two phases:
- **First half**: Only the embedding layer and `lm_head` are updated to prevent severe loss fluctuations.
- **Second half**: All parameters are fine-tuned together.

This progressive strategy significantly improves training stability and avoids loss spikes.

**3. Parameter Initialization**

Once the alignment matrix is determined, the parameters (embedding and `lm_head`) of each token in the target vocabulary are initialized by copying from the most similar source token, providing a superior starting point.

## Key Experimental Results

### Main Results

**Cross-lingual Transfer (Pythia $\to$ Qwen2 Vocabulary)**:

| Initialization Method | Initial PPL (Avg) | PPL after + LAT |
|---|---|---|
| Focus | 3.1e5 | 12.4 |
| ZeTT | 3.4e2 | 7.8 |
| **TokAlign** | **1.2e2** | **6.9** |

$\to$ TokAlign's initial PPL is **3 orders of magnitude** lower than Focus and **65%** lower than ZeTT.

**Improved Text Compression Rate**: Following vocabulary substitution, the token sequence length across 13 languages is reduced by **29.2% on average**.

**Impressive Performance on Low-Resource Languages**: Pythia-1B initialized via TokAlign achieves a PPL on 3 low-resource languages that even **outperforms Qwen2 of the same scale**.

**Performance Recovery after Vocabulary Adaptation**:

| Method | GPU Hours | Recovery Steps | Avg @6 tasks |
|------|---------|---------|-------------|
| Focus | 99.70 | >5k | 45.29 |
| ZeTT | 418.94 | ~5k | 45.48 |
| **TokAlign** | **99.70** | **~5k** | **48.42** |

$\to$ Training is **1.92 times faster** than Focus, requiring only 5k steps to recover 97.6% of the original model's performance.

### Token-level Distillation

Unifying the vocabulary enables token-level distillation, which significantly outperforms sentence-level distillation:

| Method | ARC-E | BoolQ | HellaSwag | Avg |
|------|-------|-------|-----------|-----|
| Original Pythia-1B | 56.82 | 60.43 | 37.68 | 49.55 |
| + Sentence-level Distillation (Qwen2-7B) | 52.27 | 67.49 | 39.03 | 49.90 |
| + Token-level Distillation (Qwen2-7B) | **62.33** | **70.18** | **41.58** | **54.02** |

$\to$ Token-level distillation outperforms sentence-level distillation by **4.4%** using only 235M tokens.

$\to$ The performance of Pythia-1B after token-level distillation **rivals that of the original Pythia-7B**.

### Key Findings

1. **Cross-lingual zero-shot ICL**: TokAlign outperforms Focus by 4.4% on average (+2.3% on Japanese, +2.2% on Vietnamese).
2. **Retention of English Capability**: TokAlign retains 54.2% of English performance, substantially exceeding Focus's 40.8%.
3. **Corpus Robustness**: Switching to SlimPajama to train GloVe embeddings yields virtually identical results.
4. **Significant Reduction in Initial Loss**: The first-step loss of Pythia-2.8B drops from 17.8 (Focus) to 9.5 (TokAlign).
5. **Relative Representation Improvement**: Transforming GloVe embeddings into relative representations using 300 shared anchor tokens further improves alignment quality.

## Highlights & Insights

1. **Unsupervised and Resource-free**: Requires no bilingual dictionaries, external word embeddings, or auxiliary hypernetworks; it learns alignments purely from co-occurrence statistics.
2. **Extremely Low Computational Cost**: GloVe training and alignment take only 2 hours on CPU, representing a 3-order-of-magnitude reduction compared to ZeTT's 661 GPU hours.
3. **Enabling Cross-Model Knowledge Transfer**: Unifying vocabularies renders token-level distillation highly effective, allowing smaller models to efficiently absorb knowledge from larger models.
4. **Successful Application of the Distributional Hypothesis**: Inspired by the classical distributional hypothesis (co-occurrence $\to$ semantics), this work validates its effectiveness at the subword token level.
5. **Full Vocabulary Replacement rather than Expansion**: Unlike most approaches that only extend vocabularies, TokAlign supports complete replacement, offering broader applicability.

## Limitations & Future Work

1. **Dependence on GloVe Training Quality**: Co-occurrence statistics for low-frequency tokens might be insufficient, affecting alignment accuracy.
2. **One-to-One Mapping Limitation**: When source and target vocabularies have different token granularities, a strictly one-to-one mapping may lose information.
3. **Validated Only on Pythia**: Main experiments are based on Pythia (which is not a SOTA model), leaving its efficacy on larger or stronger models to be verified.
4. **Corpus Mixture Ratios Require Tuning**: The 40/30/30 ratio for CulturaX/Stack/Proof-Pile-2 is empirical and might need adjustment for different scenarios.
5. **Simplistic Progressive Fine-Tuning Strategy**: The simple two-stage division might not be optimal; a more granular learning rate schedule could offer room for improvement.

## Related Work & Insights

- **Word Representation Learning**: Classical methods such as Word2Vec, GloVe, and FastText; TokAlign borrows GloVe's global co-occurrence matrix factorization idea.
- **Vocabulary Adaptation**: Methods such as Focus, WECHSEL, ZeTT, and OFA, which suffer from limitations like relying on external resources or high computational costs.
- **Knowledge Distillation**: Comparison of token-level vs. sentence-level distillation, discussed previously in the NMT domain, which this work extends to LLMs.
- **Multilingual Models**: Multilingual LLMs such as Qwen2, LLaMA3, and Gemma, where vocabulary design heavily impacts low-resource languages.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Combining GloVe co-occurrence with token-level alignment is highly creative |
| Theoretical Depth | ⭐⭐⭐ | Method is intuitive and reasonable, but theoretical analysis is limited |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Covers multiple scales, languages, and downstream tasks with comprehensive ablation |
| Practical Value | ⭐⭐⭐⭐⭐ | Highly cost-effective with great performance; a truly practical vocabulary substitution scheme |
| Overall Recommendation | ⭐⭐⭐⭐ | A concise and practical vocabulary adaptation method that enables cross-model distillation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion](second_language_arabic_acquisition_of_llms_via_progressive_vocabulary_expansion.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](../../NeurIPS2025/llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)

</div>

<!-- RELATED:END -->
