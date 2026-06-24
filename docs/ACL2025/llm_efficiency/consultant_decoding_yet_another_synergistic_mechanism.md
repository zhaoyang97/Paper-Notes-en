---
title: >-
  [Paper Note] Consultant Decoding: Yet Another Synergistic Mechanism
description: >-
  [ACL 2025][LLM Efficiency][Consultant Decoding] This paper proposes Consultant Decoding (CD), a novel cooperative decoding mechanism that verifies draft tokens based on the target model's negative log-likelihood (NLL). Compared to the likelihood-ratio verification methods of traditional speculative decoding, CD significantly improves the acceptance rate, reduces the frequency of target model calls, and maintains or even exceeds the generation quality of the target model.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Consultant Decoding"
  - "Speculative Decoding"
  - "NLL Verification"
  - "Inference Acceleration"
  - "LLM Inference"
date: 2026-05-08
content_hash: 3af82d2334c8958b
---

# Consultant Decoding: Yet Another Synergistic Mechanism

**Conference**: ACL 2025  
**arXiv**: [2506.02391](https://arxiv.org/abs/2506.02391)  
**Code**: None  
**Area**: Other  
**Keywords**: Consultant Decoding, Speculative Decoding, NLL Verification, Inference Acceleration, LLM Inference

## TL;DR

This paper proposes Consultant Decoding (CD), a novel cooperative decoding mechanism that verifies draft tokens based on the target model's negative log-likelihood (NLL). Compared to the likelihood-ratio verification methods of traditional speculative decoding, CD significantly improves the acceptance rate, reduces the frequency of target model calls, and maintains or even exceeds the generation quality of the target model.

## Background & Motivation

Speculative Decoding (SD) is currently the prevailing paradigm for accelerating large language model (LLM) inference. Its core mechanism involves using a small draft model to rapidly generate candidate token sequences, which are then verified in parallel by a larger target model. However, existing methods suffer from two fundamental limitations:

**High Rejection Rate**: SD verifies tokens based on the likelihood ratio of importance sampling, $r(x_i) = \min(1, p_i(x_i)/q_i(x_i))$. This metric does not fully reflect the actual quality of the token. For instance, when both the target and draft models assign low predictive probabilities, but the target's is slightly higher, the token is still accepted, which may degrade output quality.

**Frequent LLM Calls**: Due to the high rejection rate, every rejection necessitates a re-call to the large model for resampling, offsetting the efficiency gains from parallel verification.

**Parameter Sensitivity**: While improved approaches like Mentored Decoding (MD) relax the verification criteria, they introduce hard-to-determine threshold and tolerance parameters.

The core insight is that instead of using the probability ratio of two model distributions to verify a token, it is more effective to directly measure how "correct" the target model deems the token to be. If the NLL of a token is close to or lower than the convergence loss of the target model during training, the token is considered reliable.

## Method

### Overall Architecture

CD adheres to the draft-and-verify paradigm: a small model autoregressively generates $\gamma$ candidate tokens, and the large model performs a single forward pass to obtain probability distributions at all positions, subsequently verifying the candidate tokens one by one. The key difference lies in the verification mechanism.

### Key Designs

1. **NLL-based Verification Criterion**: Function -> Directly utilize the target model's NLL to evaluate the correctness of the draft token; Mechanism -> Accept the token if $-\log(p_i(x_i)) \leq \varepsilon$; Design Motivation -> NLL directly reflects the target model's confidence in the token, aligning with the convergence loss during training. The verification formula is:

$$V_{CD}(x_i) = \varepsilon - (-\log(p_i(x_i)))$$

If $V_{CD}(x_i) > 0$, the draft token is accepted; otherwise, it is resampled from the target model's distribution.

2. **EMA-smoothed Verification**: Function -> Use exponential moving average (EMA) to smooth the token-level NLL, incorporating contextual information; Mechanism -> Compute $r_i = \beta \cdot r_{i-1} + (1-\beta) \cdot (-\log(p_i(x_i)))$, and then evaluate if $r_i \leq \varepsilon$; Design Motivation -> Single-token level decisions may lead to false rejections due to occasional high NLL. EMA introduces contextual smoothing to reduce anomalous errors. The parameter $\beta$ controls the trade-off between the context history and the current NLL.

3. **Determination of Universal Threshold ε**: Function -> Utilize the Chinchilla scaling law to estimate the training convergence loss of the target model as the threshold $\varepsilon$; Mechanism -> Apply $L(N,D) = E + A/N^\alpha + B/D^\beta$ using the model parameter counts and training data volume; Design Motivation -> Align the verification threshold with the training loss, ensuring that the accepted tokens remain within the target model's training data distribution. For multiple mainstream LLMs, the estimated convergence loss is approximately 2.0; thus, $\varepsilon = 2.0$ is set as the default.

4. **Consistency Analysis with Top-P Sampling**: Function -> Provide theoretical proof that tokens accepted by CD are equivalent to falling within the high-quality nucleus of the target model's Top-P sampling; Mechanism -> As $\beta \to 0$, the verification condition reduces to $p(x_i) > e^{-\varepsilon}$, which is equivalent to checking if the token is inside the top nucleus size of $1 - e^{-\varepsilon}$ in Top-P sampling; Design Motivation -> Provide theoretical guarantees to demonstrate that CD preserves the generation quality of the target model.

### Loss & Training

CD is a training-free, inference-time method that requires no extra training or loss functions. It involves only two core parameters: the threshold $\varepsilon$ (default 2.0) and the EMA decay weight $\beta$ (default 0.2).

## Key Experimental Results

### Main Results

| Model Pair | Method | GSM8K Acc | Speedup | HumanEval Pass@1 | Speedup | MT-Bench Score | Speedup |
|---------|------|-----------|--------|-------------------|--------|----------------|--------|
| Qwen2.5-0.5B/72B | SD | 99.9% | 2.22× | 100.7% | 2.18× | 100.0% | 1.57× |
| Qwen2.5-0.5B/72B | MD | 96.8% | 2.54× | 97.9% | 2.41× | 95.2% | 1.77× |
| Qwen2.5-0.5B/72B | **CD** | **96.1%** | **3.09×** | **97.2%** | **3.04×** | **95.4%** | **2.06×** |
| Qwen2.5-7B/72B | SD | 99.9% | 2.09× | 100.0% | 2.12× | 100.0% | 1.68× |
| Qwen2.5-7B/72B | MD | 98.7% | 2.82× | 97.2% | 2.91× | 95.8% | 2.63× |
| Qwen2.5-7B/72B | **CD** | **99.1%** | **3.00×** | **100.7%** | **3.13×** | **96.3%** | **2.83×** |

### Ablation Study

**LLM Call Rate Comparison (draft length=20, Generic setting)**

| Model Pair | Method | GSM8K LLM-calls | HumanEval LLM-calls | MT-Bench LLM-calls |
|---------|------|----------------|---------------------|---------------------|
| Qwen2.5-0.5B/72B | SD | 11.7% | 11.8% | 22.0% |
| Qwen2.5-0.5B/72B | MD | 10.2% | 10.2% | 19.0% |
| Qwen2.5-0.5B/72B | **CD** | **9.1%** | **9.1%** | **15.8%** |
| Qwen2.5-7B/72B | SD | 9.0% | 8.5% | 14.1% |
| Qwen2.5-7B/72B | **CD** | **6.5%** | **6.2%** | **8.6%** |

**Generalization to Self-Drafting Architectures (EAGLE-2 + CD)**

| Method | HumanEval Pass@1 | Speedup | MT-Bench Score | Speedup |
|------|-------------------|--------|----------------|--------|
| EAGLE-2-70B | 100.0% | 3.43× | 100.0% | 3.18× |
| EAGLE-2-70B-CD | 100.8% | **3.66×** | 98.2% | **3.51×** |

### Key Findings

1. **Significant Acceleration**: CD consistently outperforms SD and MD across all tasks. The 0.5B/72B model pair achieves a 2.0-3.1× speedup, which is on average 0.8× faster than SD.
2. **Extremely Few LLM Calls**: On HumanEval, the 7B/72B model pair requires only 6.2% of LLM calls to outperform the base target model (103.5%).
3. **Exceeding Target Model Ceiling**: CD allows the draft model to guide superior reasoning paths, enabling actual performance to transcend the ceiling of the target model's standalone inference.
4. **Robust to Draft Length**: When the draft length increases from 6 to 20, the speedup of SD and MD drops by 0.53× and 0.32× respectively, whereas CD's speedup drops by only 0.08×.
5. **Favorable Scalability**: As the draft model size increases, the average acceptance length (AAL) of CD scales significantly faster than that of SD.

## Highlights & Insights

- **Simple and Elegant Core Innovation**: Using the training convergence loss as a natural threshold for token correctness avoids complex distribution alignment calculations.
- **Apt "Consulting" Metaphor**: The large model acts as a consultant offering suggestions, while the final decision matches the draft model's trajectory—striking a delicate balance between independence and synergy.
- **Exceeding Mechanism Revealed by Case Studies**: On GSM8K, while SD can get trapped in erroneous reasoning paths because of strict alignment with the target model's distribution, CD allows following the draft model's reasoning chain, which occasionally yields correct answers.
- **Efficient Cooperation Across Two Orders of Magnitude**: The 0.5B and 72B model pair (a 144× discrepancy in parameter scale) still achieves a 3× speedup.

## Limitations & Future Work

1. **Suboptimality of Fixed Threshold**: $\varepsilon = 2.0$ is a rough estimate. The optimal threshold depends on specific tasks and model combinations. Dynamic threshold adjustment remains an important future direction.
2. **Unnecessary Replacements**: CD occasionally modifies tokens that are semantically equivalent but expressed differently (e.g., "Therefore" → "Since"), requiring a better determination of token equivalency.
3. **Validation Limited to Specific Model Families**: The method is primarily validated on Qwen2.5 and Llama3.1, necessitating experimental support across a wider variety of model families.
4. **Lack of End-to-End System Evaluation**: Throughput and latency performance in production-level serving systems have not been evaluated.

## Related Work & Insights

- **Speculative Decoding Series**: SD → Draft optimizations (EAGLE, Medusa) → relaxed verification (MD, BiLD) → non-distributional verification (this paper).
- **Distinction from BiLD**: BiLD also evaluates tokens using distributional distance but still relies on likelihood ratios, whereas CD completely abandons distribution alignment.
- **Connection to Top-P Sampling**: Provides a theoretical foundation for CD, bridging verification mechanisms and sampling strategies.

## Rating

- **Novelty**: ★★★★☆ — The NLL verification concept is simple yet highly effective. The perspective of aligning with training loss is remarkably novel.
- **Experimental Thoroughness**: ★★★★☆ — Covers multiple tasks and model combinations with comprehensive ablation and scalability analyses.
- **Writing Quality**: ★★★★☆ — Possesses clear motivational explanations, intuitive illustrations, and convincing theoretical analyses.
- **Value**: ★★★★☆ — A plug-and-play inference acceleration approach with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[ACL 2025\] CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)

</div>

<!-- RELATED:END -->
