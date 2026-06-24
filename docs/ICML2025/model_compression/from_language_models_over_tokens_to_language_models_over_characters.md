---
title: >-
  [Paper Note] From Language Models over Tokens to Language Models over Characters
description: >-
  [ICML 2025][Model Compression][tokenization] An algorithmic framework is proposed to precisely convert token-level language models into character-level language models. By defining "covering" (minimally covering prefix encoding sets) and approximating it using beam search, user-end issues caused by tokenization, such as the prompt boundary problem, are resolved, while improving the compression rate (bits/byte).
tags:
  - "ICML 2025"
  - "Model Compression"
  - "tokenization"
  - "character-level language models"
  - "probability flows"
  - "covering"
  - "beam search"
date: 2026-05-08
content_hash: b5b7326d43c47fdc
---

# From Language Models over Tokens to Language Models over Characters

**Conference**: ICML 2025  
**arXiv**: [2412.03719](https://arxiv.org/abs/2412.03719)  
**Code**: [Yes](https://github.com/genlm/genlm-bytes)  
**Area**: Model Compression  
**Keywords**: tokenization, character-level language models, probability flows, covering, beam search

## TL;DR

An algorithmic framework is proposed to precisely convert token-level language models into character-level language models. By defining "covering" (minimally covering prefix encoding sets) and approximating it using beam search, user-end issues caused by tokenization, such as the prompt boundary problem, are resolved, while improving the compression rate (bits/byte).

## Background & Motivation

**Background**: Mathematically, modern LLMs are probability distributions over token sequences, rather than character strings. Users interact with models via strings, which are converted by a tokenizer in the background.

**Limitations of Prior Work — Prompt Boundary Problem**: Due to the non-bijective nature of tokenizers like BPE, a single space at the end of a prompt can lead to completely different continuation results. For example, GPT-2 continues "the" and "the " completely differently (with probability plummeting from 0.98 to 5e-7). Token healing can only backtrack one token and fails to handle more complex scenarios (e.g., backtracking one token for "Hello, worl" still produces "worlwide").

**Key Challenge**: The conditional probability of token-level models depends on the tokenization method, whereas users actually require character-level conditional probability. Conditioning directly on the canonical tokenization $\tau(\sigma)$ is incorrect—the correct approach is to consider all possible token encodings.

**Key Insight**: Leveraging the "Probability 101" conditional probability formula $p_{\Delta|\Sigma}(\delta|\sigma) = P[Y=\delta | \kappa(Y) \succeq \sigma]$ to correctly condition token-level models on character strings.

**Core Idea**: Define a **covering** (the minimal set of prefix encodings) to transform an infinite summation into a finite one, which is then efficiently approximated via beam search.

## Method

### Overall Architecture (pipeline)

1. Given a character string $\sigma$, enumerate the covering $\mathcal{C}(\sigma)$ of all token sequences.
2. Calculate the prefix probability for each token sequence in the covering.
3. Sum them up to obtain the character-level prefix probability $\overrightarrow{p_\Sigma}(\sigma) = \sum_{\delta \in \mathcal{C}(\sigma)} \overrightarrow{p_\Delta}(\delta)$.
4. Obtain the conditional probability and the character-level distribution through the ratio of prefix probabilities.

### Key Designs

1. **Definition and Properties of Covering**: The covering $\mathcal{C}(\sigma)$ is the set of all token sequences that minimally cover the character string $\sigma$. A token sequence $\delta$ minimally covers $\sigma$ if and only if $\kappa(\delta_{1..M-1}) \prec \sigma \preceq \kappa(\delta_{1..M})$. The core theorem (Proposition 1) proves that: under a strict-prefix monotone decoder $\kappa$, the prefix probability can be exactly calculated using a finite summation over the covering. Design Motivation: Reduce the summation over the infinite set $\mathcal{P}(\sigma)$ to the finite set $\mathcal{C}(\sigma)$.

2. **enum_cover Recursive Enumeration Algorithm**: Recursively construct the covering character-by-character. For each new character, traverse all possible token extensions in the vocabulary, filter candidates that match the target character, and maintain a tuple of (probability, decoded string, token sequence). The time complexity is $O(|\Delta| \cdot |\mathcal{C}(\sigma)|)$, which is exponential in the worst case. Design Motivation: Utilize strict-prefix monotonicity to prune invalid paths during the enumeration process.

3. **Top-K Bucket Beam Search Pruning**: Group the covering members into buckets based on their prefixes after removing the last token, and keep the $K$ buckets with the highest probabilities. The workload per step is $O(K \cdot |\Delta|)$, resulting in a total time of $O(N \cdot K \cdot |\Delta|)$, which is linear with respect to the string length. Design Motivation: The high-probability elements in the covering are usually sparse, allowing beam search to achieve accurate approximations with a small $K$.

4. **Bundled Beam Implementation Optimization**: Pack token sequences in the same bucket into bundles, and use a trie to efficiently filter character matches for the next token. Treat the trie as a local language model to generate the next token character-by-character. Design Motivation: Reduce the constant factor to improve actual execution speed.

5. **Conditional Generation Algorithm (conditional_token_generation)**: Sample a token sequence from the covering proportionally to its prefix probability to serve as the token-level representation of the prompt, and then continue generation directly from the token-level model. Proposition 2 proves that this process is equivalent to the correct conditional distribution $p_{\Delta|\Sigma}(\cdot|\sigma)$. Compared with character-by-character generation, once a covering element is sampled, the highly efficient autoregressive generation of the token-level model can be utilized.

6. **next_character_probability Algorithm**: Reuse the output of enum_cover and handle three cases: (1) exact match -> accumulate the EOS probability and all single-token extension probabilities; (2) over-match -> directly accumulate the probability of the next character of the decoded string. Finally, divide by the total prefix probability $Z$ to obtain the character-level conditional distribution.

### Loss & Training

This work does not involve training—the proposed method runs directly on pre-trained token-level LLMs and is an inference-time conversion algorithm. The core optimization objective is to maximize the approximation quality of beam search (minimizing JSD/byte). The experiments are evaluated on the first 4000 bytes of the wikitext-103-v1 test set. The reference distribution uses a large beam size of $K=128$, and the error bars represent the bootstrapped 95% confidence intervals.

## Key Experimental Results

### Main Results — Approximation Quality vs. Speed

| Model | K=8 JSD/byte | K=64 JSD/byte | K=8 Speed (bytes/s) | K=64 Speed (bytes/s) |
|------|-------------|--------------|------------------|-------------------|
| Llama-3.2-1B | ~1e-4 | ~1e-6 | ~250 | ~30 |
| Meta-Llama-3.1-8B | ~1e-4 | ~1e-6 | ~80 | ~10 |
| DeepSeek-R1-Distill-8B | ~1e-4 | ~1e-6 | ~80 | ~10 |
| phi-4 (14B) | ~1e-4 | ~1e-6 | ~50 | ~8 |

### Improvement in Compression Rate (bits/byte)

| Model | Token-level bits/byte | Character-level bits/byte | Gain |
|------|-------------------|------------------|------|
| Llama-3.2-1B | ~1.05 | ~0.95 | Significant |
| Meta-Llama-3.1-8B | ~0.85 | ~0.78 | Significant |
| phi-4 (14B) | ~0.80 | ~0.74 | Significant |

### Key Findings

- Even with a small beam size of $K=8$, the JSD/byte is already on the order of $10^{-4}$, demonstrating excellent approximation quality.
- The bits/byte of the character-level model is significantly better than the canonical tokenization of the token-level model.
- Among the high-probability covering elements, canonical tokenization dominates (e.g., in the 78 encodings of "Hello, world", the top-1 is the canonical one).
- The method performs consistently across 4 public models of different scales.
- Both BPE and WordPiece satisfy the strict-prefix monotone condition, making the method applicable to current mainstream tokenizers.
- There is a clear Pareto curve between error (JSD/byte) and speed, allowing users to choose the $K$ value according to their requirements.
- The string "Hello, world" has a total of 78 token encodings, but the probability is highly concentrated on the canonical tokenization (with top-1 accounting for >99%).

## Highlights & Insights

- Formalizes the token-to-character conversion problem as a probability theory problem, with a clear and elegant mathematical definition of covering.
- Reveals the profound observation that "$\tau$ is useless at inference time": after training, only the decoder $\kappa$ is needed, rather than the encoder $\tau$.
- The bucket strategy of beam search is clever—grouping by the prefix after removing the last token, which allows different partially matching tokens within the same bucket.
- Provides a principled rather than heuristic solution to the prompt boundary problem.

## Limitations & Future Work

- Inference speed is relatively slow (~250 bytes/s at $K=8$ for a 1B model), making it unsuitable for real-time applications.
- The size of the covering is exponential in the worst case; although high-probability elements are sparse in practice, there is no theoretical guarantee.
- Only 4 models were validated, and evaluation on larger-scale models (>14B) is lacking.
- Engineering schemes for integrating the method into actual serving systems have not been explored.

## Related Work & Insights

- **Token healing** (Lundberg & Ribeiro, 2023): Heuristically backtracks one token; this work provides a complete, principled solution.
- **Gastaldi et al., 2025**: Consistency estimation theory of tokenization; the covering framework in this work is complementary to it.
- **Psycholinguistic Applications**: Computing the contextual surprisal of arbitrary character substrings can predict human reading times.
- Potential value for the field of constrained decoding (such as structured output)—character-level constraints can be directly applied.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to provide a complete mathematical framework for token-to-character model conversion.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models, multiple beam sizes, but lacks evaluation on downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical narrative is extremely rigorous and clear, with vivid examples.
- Value: ⭐⭐⭐⭐ Has a foundational impact on the entire LLM interface layer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](../../CVPR2025/model_compression/dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[ICML 2025\] Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding](rethinking_addressing_in_language_models_via_contexualized_equivariant_positiona.md)

</div>

<!-- RELATED:END -->
