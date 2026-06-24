---
title: >-
  [Paper Note] Language Models over Canonical Byte-Pair Encodings
description: >-
  [ICML 2025][LLM Pretraining][BPE tokenization] Reveals that autoregressive language models under BPE tokenization allocate unnecessary probability mass to an exponential number of noncanonical token encodings. Two correction schemes, conditioning and construction based on Finite State Automata (FSA), are proposed to consistently improve held-out likelihood across various models and corpora.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "BPE tokenization"
  - "canonical encoding"
  - "language modeling"
  - "probability mass"
  - "FSA"
date: 2026-05-08
content_hash: 26a1fc48109887d8
---

# Language Models over Canonical Byte-Pair Encodings

**Conference**: ICML 2025  
**arXiv**: [2506.07956](https://arxiv.org/abs/2506.07956)  
**Code**: [GitHub](https://github.com/genlm/canonical-icml-2025)  
**Area**: LLM Pretraining  
**Keywords**: BPE tokenization, canonical encoding, language modeling, probability mass, FSA

## TL;DR

Reveals that autoregressive language models under BPE tokenization allocate unnecessary probability mass to an exponential number of noncanonical token encodings. Two correction schemes, conditioning and construction based on Finite State Automata (FSA), are proposed to consistently improve held-out likelihood across various models and corpora.

## Background & Motivation

**Background**: Modern language models (LMs) map character strings to token sequences using a deterministic tokenizer (e.g., BPE), establishing probability distributions over the token space. This paradigm forms the cornerstone of almost all current large-scale language model training.

**Limitations of Prior Work**: BPE tokenizers generate only a single **canonical encoding** for each string. However, the same string can be decomposed into an exponential number of different token sequences (noncanonical encodings). Although these sequences can decode back to the original string, they never appear in any training corpora. Currently, models still assign non-zero probability to these impossible sequences.

**Key Challenge**: The probability allocation to noncanonical encodings causes two distinct issues:
1. **Incorrectness**: These sequences do not exist in the training data, so models should not assign them positive probability.
2. **Wastefulness**: The probability mass allocated to noncanonical encodings is effectively "stolen" from the correct canonical sequences, reducing the model's predicted probability for legitimate outputs.

**Goal**: How to enforce token-level language models to assign positive probability exclusively to canonical BPE encodings, eliminating the aforementioned waste and incorrectness.

**Key Insight**: Formalize the canonicality constraint of the BPE tokenizer as the reachability conditions of a Finite State Automaton (FSA), based on which two correction strategies are designed for inference time and training time.

**Core Idea**: Utilize FSAs to precisely characterize the set of canonical BPE encodings, eliminating probability allocation to noncanonical sequences through conditional inference or parametric constraints.

## Method

### Overall Architecture

The authors propose two complementary schemes to achieve canonicality guarantees:
1. **Canonicality by Conditioning**: Inference-time constraint, requiring no extra training.
2. **Canonicality by Construction**: Parameterized model guarantee, requiring retraining.

### Key Designs

1. **Canonical FSA Construction**:
    - **Function**: Precisely describes the formal language constraints of the canonical BPE encoding set.
    - **Mechanism**: The greedy merging rule of BPE is inherently a deterministic process, which can be captured by the state transitions of an FSA. At each decoding step, the FSA maintains the state of whether the current prefix is still on the canonical path, allowing only tokens that keep the prefix canonical. The complexity of the FSA is linear with respect to the vocabulary size.
    - **Design Motivation**: Formal language theory provides precise and efficient tools to express constraints of deterministic tokenizers, and FSA is the perfectly matched tool in terms of expressivity—both powerful enough and computationally efficient.

2. **Conditioning Method**:
    - **Function**: Directly uses the FSA to mask out illegal tokens during inference.
    - **Mechanism**: In each step of autoregressive decoding, the FSA's current state is queried to obtain the set of legal tokens. The logits of illegal tokens are set to $-\infty$, and the probabilities of legal tokens are then re-normalized. This is equivalent to calculating the conditional probability $p_{\text{canonical}}(t | \mathbf{t}_{<i}) = p(t | \mathbf{t}_{<i}) / Z$, where $Z = \sum_{t' \in \text{legal}} p(t' | \mathbf{t}_{<i})$.
    - **Design Motivation**: Similar to the idea of constrained decoding, allowing any existing model to be corrected without retraining.

3. **Construction Method**:
    - **Function**: Embeds canonicality constraints into the model's output layer, making the probability of noncanonical sequences strictly zero.
    - **Mechanism**: Directly encodes the FSA into the parameterization of the output probability distribution. It masks the output logits to ensure that $p(\text{noncanonical token}) = 0$ holds under any state. The model learns to allocate probability over the canonical subspace during training from scratch or fine-tuning.
    - **Design Motivation**: Provides stronger theoretical guarantees—preventing the waste of probability mass even during the training process.

### Loss & Training

- **Conditioning Method**: Zero training cost, directly applied to existing models.
- **Construction Method**: Trained using the standard cross-entropy loss, but the output layer includes masks constrained by the FSA. During training, the model only computes softmax over the canonical token subset.
- The two methods can be used independently or in combination.

## Key Experimental Results

### Main Results: Improving Perplexity via Canonicality Correction

| Model | Corpus | Original PPL | Conditioning PPL | Gain |
|------|------|----------|-----------|---------|
| GPT-2 Small (124M) | WikiText-103 | Baseline | Decreased | ~0.5-1% |
| GPT-2 Medium (355M) | WikiText-103 | Baseline | Decreased | ~0.3-0.8% |
| GPT-2 Large (774M) | WikiText-103 | Baseline | Decreased | ~0.2-0.5% |
| Customized Small Model | Multi-corpus | Baseline | Construction Best | ~1-3% |

### Noncanonical Probability Mass Analysis

| Vocab Size | Noncanonical Encodings Order | Wasted Probability Mass | Description |
|----------|-----------------|-------------|------|
| Small Vocab (~1K) | Fewer | Smaller | Few merge rules, few noncanonical paths |
| Medium Vocab (~10K) | Significant | Medium | Practically substantial mass waste |
| Large Vocab (~50K+) | Exponential Growth | Maximum | The problem is most severe in large vocabularies |

### Key Findings

- The noncanonical encoding problem is ubiquitous across all tested models and corpora.
- Correcting it consistently improves held-out likelihood, proving that the wasted probability mass is indeed effectively recovered.
- Larger vocabularies lead to more noncanonical encodings, yielding more significant gains from the correction.
- While the conditioning method is training-free, it introduces inference overhead; conversely, the construction method incurs no inference overhead but requires training.
- Improvements are observed across different languages (English, multilingual) and various domains.

## Highlights & Insights

1. **Reveals a long-overlooked systematic issue**: The combination of BPE and autoregressive LMs suffers from a probability leakage. This is not an issue of individual models, but a defect at the paradigm level.
2. **Elegant and efficient FSA formalization**: Converts the intuitive concept of "canonicality" into precise formal language constraints.
3. **Strong complementarity between the two schemes**: The conditioning method is suitable for plug-and-play correction of pre-trained models, while the construction method is suited for newly trained models.
4. **Surprising scale of the problem**: The quantity of noncanonical encodings scales exponentially with string length.

## Limitations & Future Work

- The conditioning method requires maintaining FSA states during inference, which increases computational overhead, especially under large vocabularies.
- The construction method requires training from scratch or large-scale fine-tuning, which is impractical for massive models like GPT-4.
- The paper primarily evaluates the method on small-to-medium scale models; the effectiveness on models with tens of billions of parameters remains unknown.
- The concrete impact on downstream tasks (e.g., QA, summarization) has not been quantified, as evaluation is restricted to perplexity.
- The scale of improvement tends to decrease for larger models, and whether "large models automatically learn to avoid noncanonical encodings" has not been thoroughly discussed.

## Related Work & Insights

- **BPE Dropout / Subword Regularization**: These works deliberately introduce noncanonical encodings for data augmentation, whereas this work argues that these encodings should not be assigned probability—providing a completely different perspective.
- **Constrained Decoding**: The conditioning method is closely related to constrained generation techniques, though the source of constraints differs (canonicality vs. syntax/format).
- **Impact of Tokenizer Choice on LMs**: This work further confirms that tokenizers do not only affect efficiency but also compromise the correctness of probability distributions.
- **Insight**: Do other deterministic preprocessing steps (e.g., text normalization) suffer from similar probability leakages?

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Discovered and formalized a widespread yet overlooked theoretical defect under probability theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified across multiple models and corpora, but lacks scale-up to extremely large models and downstream task evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous and clear theoretical derivations, with a natural flow in describing the FSA construction.
- **Value**: ⭐⭐⭐⭐ Makes significant contributions to tokenization and foundational LM theory, though the practical deployment benefits are relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [\[ICML 2025\] Large Language Models are Demonstration Pre-Selectors for Themselves](large_language_models_are_demonstration_pre-selectors_for_themselves.md)
- [\[ACL 2025\] Byte Latent Transformer: Patches Scale Better Than Tokens](../../ACL2025/llm_pretraining/byte_latent_transformer.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
