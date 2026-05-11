---
title: >-
  [Paper Note] Vocabulary Customization for Efficient Domain-Specific LLM Deployment
description: >-
  [NeurIPS 2025][LLM Pretraining][vocabulary expansion] This paper proposes a BPE tokenizer expansion algorithm that guarantees monotonically non-increasing encoding length…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "vocabulary expansion"
  - "tokenizer adaptation"
  - "domain adaptation"
  - "BPE"
  - "inference acceleration"
date: 2026-05-08
content_hash: d9605a34a7afed33
---

# Vocabulary Customization for Efficient Domain-Specific LLM Deployment

**Conference**: NeurIPS 2025
**arXiv**: [2509.26124](https://arxiv.org/abs/2509.26124)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: vocabulary expansion, tokenizer adaptation, domain adaptation, BPE, inference acceleration

## TL;DR

This paper proposes a BPE tokenizer expansion algorithm that guarantees monotonically non-increasing encoding length, appending domain-frequent tokens to the Llama 3.1 vocabulary (+30K tokens). In an e-commerce setting, the approach shortens input sequences by 20% and improves inference throughput by 20–30%. After 10K steps of continual training, model quality is fully preserved, and in approximately 98% of cases the model actively generates the newly added tokens.

## Background & Motivation

**Background**: When deploying LLMs in specialized domains, general-purpose tokenizers often fail to encode domain-specific terminology efficiently. In e-commerce, for instance, brand names, SKU identifiers, and multilingual product descriptions are frequently split into multiple subword tokens, resulting in high token fertility that directly increases inference latency and cost.

**Limitations of Prior Work**: Existing tokenizer expansion work primarily targets new-language adaptation (e.g., Chinese, Thai), while systematic study of domain adaptation remains scarce. Yamaguchi et al. prepend new merge operations to the head of the merge list, which alters the priority of existing merges and can degrade encoding efficiency on general text. AdaptiVocab replaces existing tokens with n-gram tokens, providing no guarantee of monotonically non-increasing encoding length.

**Key Challenge**: Vocabulary expansion involves an efficiency–compatibility trade-off: adding too many new tokens enlarges the embedding and projection matrices, slowing each forward pass; adding too few limits compression gains. More critically, whether an autoregressive LLM will actively use new tokens during generation is a question that has never been systematically studied.

**Key Insight**: New merge operations are appended to the end of the merge list rather than prepended. By leveraging the sequential execution semantics of BPE, the original tokenization behavior is fully preserved, and the token count for any input can only decrease or remain the same.

**Core Idea**: Append-based vocabulary expansion + joint optimization of encoding efficiency and forward-pass speed + analysis of new-token adoption rate — a unified solution to domain-specific LLM inference efficiency.

## Method

### Overall Architecture

A five-step pipeline: train a BPE tokenizer on domain data → select new tokens and append them to the original tokenizer → initialize new embeddings → continually train the LLM → evaluate efficiency and quality.

### Key Designs

1. **Append-Based Tokenizer Expansion Algorithm**:

    - A BPE tokenizer is trained from scratch on the domain dataset to obtain candidate domain-frequent tokens; tokens absent from the original vocabulary are filtered as new candidates.
    - **Core Innovation**: New merge operations are appended to the end of the merge list. Because BPE executes merges sequentially, appending preserves the priority of all existing merges; new merges are activated only after existing tokenization is complete.
    - **Guaranteed Property**: For any input, the token count after expansion $\leq$ that of the original tokenizer. New merges can only combine existing tokens into one, never increasing the count.
    - Comparison with prepending: prepending alters merge priorities and, as verified experimentally, can increase the token count on general text.

2. **Vocabulary Size–Efficiency Trade-off Analysis**:

    - Conducted without any model training: the number of new tokens is swept from 1K to 80K, measuring encoding efficiency and forward-pass latency.
    - For the 8B model, 30K new tokens is the optimal trade-off: forward pass is only 1% slower, e-commerce tasks see an average 8% sequence reduction and up to 20%.
    - The trade-off is model-size-dependent: in larger models the embedding matrices constitute a smaller fraction of total parameters, allowing more tokens to be added.

3. **Embedding Initialization and Continual Training**:

    - Embedding and projection vectors for new tokens are initialized as the mean of their constituent subtoken vectors.
    - Mixed data (50% general + 50% domain), cosine learning rate schedule ($1\text{e-}5 \to 5\text{e-}7$), 10K steps.
    - 480 H100 GPUs, Megatron-LM framework, training completed in under 24 hours.

### Loss & Training

Standard autoregressive language modeling loss with a cosine learning rate schedule.

## Key Experimental Results

### Main Results

**Inference Throughput** (vLLM, H100, Llama-3.1 8B):

| Input/Output Length | Original RPS | Expanded RPS | Throughput Gain |
|---------------------|-------------|-------------|----------------|
| 300 words | 29.19 | 35.23 | 20.7% |
| 3000 words | 1.95 | 2.52 | 29.2% |

**Model Quality** (14 e-commerce tasks):

| Model | General NLU (En) | MMLU | E-commerce (En) | E-commerce (non-En) |
|-------|-----------------|------|----------------|-------------------|
| 8B LLM | 71.6 | 63.5 | 60.5 | 47.9 |
| +30K vocab | 71.8 | 63.4 | 60.1 | 47.6 |

Quality is fully preserved across both general and domain-specific tasks.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Append vs. Prepend | Append never increases token count on general text; prepend causes token count to increase on general text when >20K tokens are added |
| 30K new tokens | Forward pass 1% slower; encoding shortened by 8% on average |
| New-token adoption rate (>15 words) | Model generates new tokens in ~98% of cases |
| New-token adoption rate (<15 words) | ~95.3%; short sequences occasionally fall back to old tokenization |

### Key Findings

- **New-Token Adoption Rate**: The first empirical demonstration that autoregressive LLMs actively use newly added tokens (98%), resolving a long-standing concern in the community.
- Long sequences benefit more: throughput improves by 29.2% at 3,000 words vs. 20.7% at 300 words, as the quadratic complexity of attention amplifies the effect of token reduction.
- The append strategy guarantees no degradation on general text, whereas the prepend strategy can increase token counts on Wikipedia.
- Vocabulary expansion is orthogonal to quantization and speculative decoding and can be combined with them for multiplicative speedups.

## Highlights & Insights

- **Monotonicity-Guaranteed Append Strategy**: The sequential execution semantics of BPE provide a backward-compatibility guarantee without requiring additional validation, making this approach transferable to any setting where incremental tokenizer updates are needed.
- **First Analysis of New-Token Adoption Rate**: This fills a critical gap in domain tokenizer expansion research. The 98% adoption rate indicates that mean initialization combined with 10K steps of training is sufficient for the model to "accept" the expanded vocabulary.
- **Practical Efficiency–Speed Analysis Framework**: The optimal vocabulary size can be identified by a simple sweep without training any model, making this directly applicable to industrial LLM deployment optimization.

## Limitations & Future Work

- Validation is limited to the e-commerce domain; compression gains in other domains (medical, legal, financial) remain to be verified.
- All experiments are based on Llama 3.1 8B; trade-offs may differ for larger models.
- Mean initialization may be insufficient for semantically complex new tokens; smarter initialization strategies are worth exploring.
- Tokenizer management strategies for multi-domain serving scenarios are not discussed.
- The optimal number of continual training steps beyond 10K lacks ablation analysis.

## Related Work & Insights

- **vs. Yamaguchi et al. (2024)**: The prepend-merge strategy yields faster domain efficiency gains but breaks general-text encoding; the append strategy proposed here is better suited for production environments.
- **vs. AdaptiVocab**: Replacing tokens with n-grams provides no monotonic encoding guarantee and may perform worse on out-of-distribution inputs.
- **vs. Language Adaptation Works**: Language adaptation targets quality improvement, whereas this work focuses on efficiency improvement; the two methodologies are complementary.

## Rating

- Novelty: ⭐⭐⭐ The core idea is intuitive and straightforward; innovation is largely at the engineering level, though the new-token adoption rate analysis is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Encoding efficiency, quality, speed, and adoption rate are all evaluated, though only one domain is studied.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically structured, precisely defined problem, thorough analysis.
- Value: ⭐⭐⭐⭐ Directly applicable to domain-specific LLM deployment; the new-token adoption rate analysis opens a new evaluation dimension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](../../ICCV2025/llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)

</div>

<!-- RELATED:END -->
