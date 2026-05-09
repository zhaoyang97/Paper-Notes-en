---
title: >-
  [Paper Note] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
description: >-
  [NeurIPS 2025][Model Compression][cross-tokenizer distillation] This paper proposes Approximate Likelihood Matching (ALM), a principled cross-tokenizer distillation method based on binarized f-divergence, which for the first time enables effective distillation and pure distillation across fundamentally different tokenizers (e.g., subword → byte-level).
tags:
  - NeurIPS 2025
  - Model Compression
  - cross-tokenizer distillation
  - approximate likelihood matching
  - tokenizer transfer
  - f-divergence
  - LLM distillation
date: 2026-05-08
content_hash: 3d95cc0bb40be63b
---

# Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching

**Conference**: NeurIPS 2025
**arXiv**: [2503.20083](https://arxiv.org/abs/2503.20083)
**Code**: [https://github.com/bminixhofer/tokenkit](https://github.com/bminixhofer/tokenkit)
**Area**: Model Compression / Knowledge Distillation
**Keywords**: cross-tokenizer distillation, approximate likelihood matching, tokenizer transfer, f-divergence, LLM distillation

## TL;DR

This paper proposes Approximate Likelihood Matching (ALM), a principled cross-tokenizer distillation method based on binarized f-divergence, which for the first time enables effective distillation and pure distillation across fundamentally different tokenizers (e.g., subword → byte-level).

## Background & Motivation

Knowledge distillation is an important paradigm for creating efficient language models, yet existing methods require the teacher and student to share the same or similar tokenizers, which severely restricts the set of viable teacher–student pairs. Modern LLMs employ highly diverse tokenizers: different models (GPT, Llama, Gemma, Qwen) use distinct vocabularies and tokenization functions, and a recent trend has emerged toward character- or byte-level tokenization away from subword schemes.

**Key Challenge**: Standard distillation compares token-level probability distributions between teacher and student via KL divergence, which requires both to share the same token space. When tokenizers differ, teacher and student segment the same text into different token sequences, making direct comparison infeasible. Existing cross-tokenizer methods (ULD, MinED, DSKD) incorporate teacher information through heuristics and can only serve as auxiliary objectives alongside a primary objective (e.g., next-token prediction), precluding pure distillation.

**Key Insight**: The problem is formalized as comparing the likelihoods of aligned token chunks, and a binarized f-divergence approximation is adopted to avoid enumerating the infinitely many possible byte-sequence outcomes.

## Method

### Overall Architecture

Given a text $\mathbf{x}$, it is tokenized separately by the teacher and student tokenizers, and next-token probabilities are computed for all tokens. Aligned chunks—token subsequences that encode the same text span in both sequences—are identified, and the discrepancy in likelihoods between aligned chunks is then minimized.

### Key Designs

1. **Chunk-Level Probability Alignment**: The core idea is to find token chunks in the teacher and student sequences that encode the same text, defining the chunk-level probability as $p(\mathbf{x}, i:j) = p(T(\mathbf{x})_{i:j} | T(\mathbf{x})_{:i})$. Computing the f-divergence over all possible chunk outcomes is intractable due to the infinite number of possible byte sequences; instead, a binarized f-divergence is computed, considering only two outcomes—"the chunk occurs" and "the chunk does not occur." This forms an upper bound on the true f-divergence while preserving the key property that it is minimized if and only if $p_S = p_T$.

2. **Outcome Chunk Debiasing**: Subword tokenizers introduce tokenization bias—for example, the chunk "Hello_Wor" implicitly encodes information that the continuation is not "ld." This bias is removed by multiplying the chunk probability by a debiasing probability (the probability that the next token begins with a pre-tokenization boundary byte). A threshold $\gamma$ is applied to filter out chunks with excessively low debiasing probabilities, preventing probabilities from being suppressed to near-zero values.

3. **Hidden-State Distillation**: Since chunk-level probabilities provide only $|A_c| \times 32$ bits of signal (compared to $|T(\mathbf{x})| \times |\mathcal{V}| \times 32$ bits in same-tokenizer distillation), a hidden-state alignment loss is added to enrich the training signal. Hidden states at corresponding positions in the teacher and student are aligned through a learned projection function proj that maps between dimensions.

4. **GradMag Loss Combination**: To address the large discrepancy in gradient magnitudes across multiple objectives, a simple gradient-magnitude normalization scheme is proposed: the gradient of each loss with respect to the last layer is computed, and the weight is set to $1/\|G_W^i\|$. This directly solves the equal-gradient-magnitude condition and is both simpler and comparably effective to GradNorm.

### Loss & Training

The ALM objective is:
$$\mathcal{L}^{ALM}_{S,T}(\mathbf{x}) = \sum_{i,j,k,l \in A_c(\mathbf{x})} f(p_T^{1/\tau} \| p_S^{1/\tau}) + f(1-p_T^{1/\tau} \| 1-p_S^{1/\tau})$$

where $f$ is the generator function of the f-divergence and $\tau$ is a temperature hyperparameter. Either a pure distillation mode (ALM loss only) or a mixed mode (ALM + SFT next-token prediction loss) can be selected, with GradMag used for automatic balancing.

## Key Experimental Results

### Main Results: Tokenizer Transfer (Use Case 1)

| Model | Target | Method | Avg | MMLU | BoolQ | IFEval |
|-------|--------|--------|-----|------|-------|--------|
| Gemma2 2B IT | Original | — | 58.0 | 56.9 | 83.8 | 62.5 |
| Gemma2 2B IT | →Qwen2 | SFT | 51.6 | 49.8 | 77.7 | 54.2 |
| Gemma2 2B IT | →Qwen2 | MinED | 53.0 | 51.8 | 79.6 | 57.1 |
| Gemma2 2B IT | →Qwen2 | ALM | 55.1 | 53.6 | 82.7 | 53.2 |
| Gemma2 2B IT | →Byte | SFT | 46.5 | 43.1 | 67.9 | 51.5 |
| Gemma2 2B IT | →Byte | ALM+SFT | 51.3 | 51.0 | 80.5 | 51.9 |
| Llama3.2 3B IT | →Qwen2 | ALM | 58.6 | 61.6 | 79.0 | 76.3 |

### Ablation Study

| Configuration | Key Effect | Notes |
|---------------|-----------|-------|
| Outcome Chunk Debiasing | Significant performance gain | Removes tokenization bias; threshold $\gamma$ yields further improvement |
| GradMag vs. GradNorm | On par or better | Simpler loss-balancing strategy |
| Pure ALM vs. ALM+SFT | Pure ALM superior for subword transfer | Pure distillation better preserves original model behavior |
| Byte transfer: ALM+SFT | ALM+SFT superior for byte transfer | SFT signal necessary under extreme tokenization change |

### Use Case 2: Large → Small Cross-Tokenizer Distillation

| Method | GSM8K | MATH | Avg |
|--------|-------|------|-----|
| Teacher (OpenMath2-Llama3.1-8B) | 88.9 | 60.2 | 74.6 |
| SFT | 67.2 | 36.2 | 51.7 |
| DSKD | 65.7 | 34.9 | 50.3 |
| ALM+SFT | **70.2** | **36.4** | **53.3** |

### Key Findings

- ALM is the first method to achieve effective subword → byte-level distillation transfer (prior methods were entirely ineffective or inferior to SFT).
- Migrating different models to a shared tokenizer enables token-level ensembling, yielding performance superior to any individual model.
- ALM requires neither the cross-attention computation of DSKD nor the large logit-matrix alignment of MinED, offering computational efficiency.
- ALM closes an additional 34% of the teacher–student gap compared to the best prior method.

## Highlights & Insights

- **Principled Approach**: Unlike prior heuristic methods, ALM provides a mathematically grounded distillation objective—the binarized f-divergence is an upper bound on the true divergence and preserves the optimality condition.
- **Strong Generality**: The same method applies across diverse settings: subword → subword, subword → byte, large → small distillation, and hypernetwork training.
- **Tokenizer Transfer as Self-Distillation**: Reframing tokenizer transfer as cross-tokenizer self-distillation is an elegant and insightful perspective.
- **New Possibilities for Model Ensembling**: Migrating models to a unified tokenizer enables token-level ensembling across different model families.

## Limitations & Future Work

- The binarized f-divergence is a coarse approximation of the true divergence and may lose information.
- Byte-level transfer still lags considerably behind the original model; additional techniques (e.g., hourglass architectures, multi-byte prediction) may be required.
- Outcome Chunk Debiasing addresses only the outcome-side bias; conditioning-side bias remains unresolved.
- Hidden-state alignment requires prior knowledge of layer correspondence between teacher and student.

## Related Work & Insights

This paper elevates cross-tokenizer distillation from heuristic approaches to a principled framework. Compared to DSKD (cross-attention-based token alignment) and MinED (minimum edit-distance matching), ALM is more efficient and requires no precomputation steps. The insight that tokenizer transfer is a form of self-distillation offers a more economical path to byte-level models than training from scratch, and opens new possibilities for composing knowledge across different model families.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First principled solution to distillation across fundamentally different tokenizers; highly original perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three use cases with comprehensive validation, detailed ablations, and sufficient efficiency comparisons
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, clear method descriptions, and intuitive figures
- Value: ⭐⭐⭐⭐⭐ Addresses a core bottleneck in LLM distillation with substantial practical significance

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/model_compression/ctpd_cross_tokenizer_preference_distillation.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [\[NeurIPS 2025\] Distillation Robustifies Unlearning](distillation_robustifies_unlearning.md)

<!-- RELATED:END -->
