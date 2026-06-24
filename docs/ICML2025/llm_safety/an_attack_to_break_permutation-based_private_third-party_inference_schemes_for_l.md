---
title: >-
  [Paper Note] An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs
description: >-
  [ICML 2025][LLM Safety][private inference] An attack method based on token-by-token vocabulary matching is proposed. By leveraging the non-collision property of the hidden states in decoder-only LLMs, the original input tokens can be almost perfectly reconstructed from three types of permuted hidden states, breaking the security claims of three private inference schemes: PermLLM, STIP, and Centaur.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "private inference"
  - "permutation security"
  - "hidden state reversal"
  - "LLM"
  - "SMPC"
date: 2026-05-08
content_hash: 16c50fc93cb3e94a
---

# An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs

**Conference**: ICML 2025  
**arXiv**: [2505.18332](https://arxiv.org/abs/2505.18332)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: private inference, permutation security, hidden state reversal, LLM, SMPC

## TL;DR

An attack method based on token-by-token vocabulary matching is proposed. By leveraging the non-collision property of the hidden states in decoder-only LLMs, the original input tokens can be almost perfectly reconstructed from three types of permuted hidden states, breaking the security claims of three private inference schemes: PermLLM, STIP, and Centaur.

## Background & Motivation

**Background**: LLM inference increasingly relies on third-party services, making privacy protection a critical demand. Secure Multi-Party Computation (SMPC) is a classic approach, but its processing efficiency for a large number of non-linear operations in Transformers is extremely low (thousands of times slower than plaintext inference), making it difficult to scale to large LLMs.

**Limitations of Prior Work**: To avoid the prohibitive overhead of SMPC, prior works like PermLLM, STIP, and Centaur propose replacing cryptography with statistical obfuscation—permuting hidden states and exposing them to the inference provider in plaintext. The security arguments rely on the extremely large permutation space (e.g., $N!(d!)^N$), making brute-force attacks infeasible.

**Key Challenge**: These works assume that the exponential size of the permutation space equates to security, but they ignore the structural properties of LLM hidden states—the hidden states of different tokens at the same position almost never overlap (the non-collision property), which reduces the attack complexity from $V^N$ to $O(VN)$.

**Goal**: To prove that permutation-based private inference schemes are insecure, and to pinpoint the exact flaws in their theoretical security proofs.

**Key Insight**: Leveraging the causal attention of decoder-only LLMs—where the hidden state at the $n$-th position is a function of only the first $n$ tokens, enabling position-by-position greedy matching.

**Core Idea**: Converting the exponential search problem into a linear search by performing forward passes to traverse the vocabulary token-by-token and matching against the target hidden states using L1 distance.

## Method

### Overall Architecture

The attacker possesses the model weights and a certain layer of hidden states $\mathbf{h} = [h_1, ..., h_N]$. Starting from the first position, the attacker traverses all tokens in the vocabulary to perform forward passes, seeking the token whose L1 distance is closest to $h_1$ as $\hat{x}_1$. Then, $\hat{x}_1$ is fixed, and the process is repeated for the second position. The overall complexity is $O(VN)$.

### Key Designs

1. **Non-deterministic Processing and Proposal Acceleration**:

    - **Function**: Overcoming GPU floating-point non-determinism and accelerating the search.
    - **Mechanism**: Performing fuzzy matching using an L1 distance threshold $\epsilon$ (tuned via ternary search on 50 calibration samples); a distance below the threshold is accepted. A proposal model is used to rank the vocabulary by likelihood, reducing the average search size from $V/2$ to approximately 100 ($1000\times$ acceleration). Coupled with KV cache optimization, a 50-token sequence can be decoded in about 2 minutes.
    - **Design Motivation**: Exact matching is infeasible due to floating-point non-determinism, but the non-collision property of LLM hidden states ensures that fuzzy matching can still uniquely determine the correct token.

2. **Sequence-Dimension Permutation Attack**:

    - **Function**: Reconstructing the sequence even when the row order of hidden states is shuffled.
    - **Mechanism**: Causal attention implicitly tags each position—exactly one element is a function of only a single embedding ("the first"), and exactly one element depends on only two embeddings ("the second"). This allows the matching process to search for the best fit among all remaining rows, excluding already paired rows.
    - **Design Motivation**: It requires cross-position non-collision, meaning hidden states at different positions do not overlap either.

3. **Hidden-Dimension Permutation Attack**:

    - **Function**: Reconstructing the sequence when the dimensions of each hidden vector $h_i$ are independently permuted.
    - **Mechanism**: Utilizing **sorted L1 distance (sorted-L1)**, where the two vectors are sorted individually before computing the L1 distance. Sorting maps any dimensional permutation to the same result.
    - **Design Motivation**: It requires the vectors to remain non-colliding even after being sorted, showing that permutation across hidden dimensions does not provide genuine obfuscation.

### Theoretical Flaws in Distance Correlation Analysis

The security arguments cited by STIP/Centaur suffer from three fundamental flaws: (1) Theorem 1 proves that even a random 1D projection is invertible; (2) Theorem 2 constructs counterexamples with low distance correlation but complete reconstructibility, demonstrating that distance correlation does not measure reconstructibility; (3) The causal, token-to-token dependencies in Transformers are completely ignored.

## Key Experimental Results

### Perfect Decoding Rate under Un-permuted Hidden States (1000 Samples)

| Layer | Gemma-2-2B-IT | Llama-3.1-8B-Instruct |
|-----|--------------|----------------------|
| 1 | 100% | 100% |
| 11 | 100% | 100% |
| 21 | 100% | 99.9% |
| 26 | 100% | 99.7% |

### Perfect Decoding Rate under Decomposed 2D Permutation (Hardest Setting)

| Layer | Gemma-2-2B-IT | Llama-3.1-8B-Instruct |
|-----|--------------|----------------------|
| 1 | 99.9% | 98.4% |
| 11 | 99.5% | 98.9% |
| 21 | 99.1% | 98.0% |
| 26 | 99.0% | 97.6% |

### Noise Defense (Gemma-2-2B-IT, ROUGE-L / LiveBench Performance)

| Defense | Un-permuted | 2D Permuted | Downstream Performance |
|------|--------|--------|---------|
| No Defense | 1.00 | 1.00 | 100% |
| Gaussian $\sigma=0.01$ | 0.93 | 0.07 | 101.4% |
| Gaussian $\sigma=0.1$ | 0.91 | 0.01 | 5.8% |
| 4-bit Quantization | 0.88 | 0.71 | 92.2% |

### Key Findings

- The perfect decoding rate reaches over 97% under all three types of permutation, indicating the extremely strong non-collision property of LLM hidden states.
- The combination of permutation and small noise $\sigma=0.01$ appears secure (ROUGE-L < 0.1, with no loss in downstream performance), but this is necessary but not sufficient.
- 4-bit quantization is still insufficient to defend against the attack (ROUGE-L = 0.71), demonstrating the high robustness of the non-collision property.
- A small number of failure cases stem from special format characters, which lead the proposal model to prioritize incorrect tokens.

## Highlights & Insights

- The insight of converting exponential search into linear search—by exploiting the causal structure of causal attention rather than the algebraic properties of the permutation space.
- A rigorous and systematic three-fold rebuttal to the distance correlation theory.
- The discovery of the non-collision property uncovers critical structural characteristics of LLM representation spaces.
- An honest analysis of noise defense: instead of claiming a perfect defense, it clearly demonstrates the security-utility trade-off.

## Limitations & Future Work

- White-box assumption (requiring model weights), though under the threat model of STIP/Centaur, the inference provider naturally has the weights.
- Unconstrained 2D permutation (where any element can be moved to any position) is not handled.
- Only two model families (Gemma-2, Llama-3.1) are tested.
- Inapplicable to encoder-only or encoder-decoder architectures due to bidirectional attention.

## Related Work & Insights

- **vs Wan et al. (2024)**: Reconstructs inputs using a trained transformer, achieving an F1 score of ~60%. Ours requires no training and is nearly perfect.
- **vs Morris et al. (2023)**: Focuses on logit-based reconstruction (~75% accuracy), whereas ours targets intermediate hidden states.
- **Against PermLLM/STIP/Centaur**: All three schemes are effectively broken.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Overthrows the security claims of three published works at once.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 2 models $\times$ 6 layers $\times$ 3 permutation types + defense analysis + theoretical dissection.
- Writing Quality: ⭐⭐⭐⭐⭐ A perfect logical chain from attack to theoretical critique, experiments, and defense.
- Value: ⭐⭐⭐⭐⭐ Possesses a fundamental impact on the private inference community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Cascade: Token-Sharded Private LLM Inference](cascade_token-sharded_private_llm_inference.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[NeurIPS 2025\] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](../../NeurIPS2025/llm_safety/mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)
- [\[ICLR 2026\] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](../../ICLR2026/llm_safety/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)
- [\[ACL 2025\] A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models](../../ACL2025/llm_safety/a_statistical_and_multi-perspective_revisiting_of_the_membership_inference_attac.md)

</div>

<!-- RELATED:END -->
