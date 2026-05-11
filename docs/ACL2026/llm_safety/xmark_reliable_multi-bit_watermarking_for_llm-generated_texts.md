---
title: >-
  [Paper Note] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts
description: >-
  [ACL 2026][LLM Safety][Multi-bit watermarking] This paper proposes XMark, a multi-bit text watermarking method based on the Leave-one-Shard-out (LoSo) strategy and evergreen lists. By taking the intersection of green lis…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multi-bit watermarking"
  - "LLM text detection"
  - "digital watermarking"
  - "text provenance"
  - "logit perturbation"
date: 2026-05-08
content_hash: 1d2df4ba7f9c1bca
---

# XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts

**Conference**: ACL 2026
**arXiv**: [2604.05242](https://arxiv.org/abs/2604.05242)
**Code**: [https://github.com/JiiahaoXU/XMark](https://github.com/JiiahaoXU/XMark)
**Area**: Text Watermarking
**Keywords**: Multi-bit watermarking, LLM text detection, digital watermarking, text provenance, logit perturbation

## TL;DR

This paper proposes XMark, a multi-bit text watermarking method based on the Leave-one-Shard-out (LoSo) strategy and evergreen lists. By taking the intersection of green lists across multiple vocabulary permutations and employing a constrained token-shard mapping matrix, XMark significantly improves decoding accuracy under limited token budgets while preserving text quality.

## Background & Motivation

**Background**: Multi-bit text watermarking embeds extractable binary information—such as user IDs and timestamps—into LLM-generated text, enabling traceability and attribution of malicious use. Existing approaches fall into two categories: distortion-free methods (where watermarked text follows the same distribution as unwatermarked text) and logit-perturbation methods (which embed information by modifying logits).

**Limitations of Prior Work**: (1) Early methods (CycleShift, CTWL, DepthW) require exhaustive enumeration of all candidate messages during decoding, making long messages computationally infeasible. (2) MPAC addresses feasibility through block-wise encoding and decoding, but constrains the green list ratio to $\gamma \leq 0.25$, severely distorting token sampling probabilities and degrading text quality. (3) StealthInk improves text quality but weakens the watermark signal, reducing decoding accuracy. (4) **All methods suffer a sharp drop in decoding accuracy under limited token budgets**, yet short texts are common in practice.

**Key Challenge**: There exists a fundamental trade-off between text quality and decoding accuracy—a larger green list reduces distributional distortion but weakens the watermark signal, while a smaller green list strengthens the signal but severely impairs quality. This tension is especially acute under limited token budgets.

**Goal**: Simultaneously improve watermarked text quality and decoding accuracy under limited token budgets.

**Key Insight**: Invert the green list selection strategy—rather than using the shard corresponding to the encoded message as the green list (as in MPAC), exclude that shard and use all remaining shards as the green list, raising the green list ratio from $\leq 0.25$ to $\geq 0.75$.

**Core Idea**: Use Leave-one-Shard-out to improve text quality; use the intersection of multiple permuted evergreen lists to increase the number of observations per token and compensate for signal strength; use a constrained TMM to prevent count explosion in non-perturbed shards.

## Method

### Overall Architecture

XMark follows the block-wise encoding and decoding paradigm: a $b$-bit message is divided into $r$ blocks of $d$ bits each. During encoding, one message block is embedded per generated token; during decoding, each block's message is recovered from the suspect text. The core innovations lie in the LoSo + evergreen list design of the encoder and the cTMM design of the decoder.

### Key Designs

1. **Leave-one-Shard-out (LoSo) Encoding**:

    - **Function**: Substantially improves text quality by inverting the green list selection strategy.
    - **Mechanism**: MPAC uses the shard corresponding to message value $[\mathbf{m}_i]_{10}$ as the green list ($\gamma = 2^{-d} \leq 0.25$). LoSo excludes that shard and forms the green list from all remaining shards ($\gamma = 1 - 2^{-d} \geq 0.75$). Decoding recovers the message by identifying the shard with the fewest token counts. For example, when $d=2$ and $\mathbf{m}_i=11$, $\mathcal{S}_3$ is excluded and $\mathcal{S}_0, \mathcal{S}_1, \mathcal{S}_2$ are perturbed.
    - **Design Motivation**: Raising the green list ratio from 0.25 to 0.75 means the logit distribution of the majority of the vocabulary remains unchanged, yielding text quality close to that of unwatermarked text.

2. **Evergreen List (Multi-Permutation Intersection)**:

    - **Function**: Increases each token's informational contribution to decoding while maintaining a large green list, compensating for the weaker signal of LoSo.
    - **Mechanism**: $k$ distinct hash keys are used to generate $k$ vocabulary permutations, each yielding a LoSo green list $\mathcal{G}_j$. The intersection of all green lists forms the evergreen list $\mathcal{E} = \bigcap_{j=0}^{k-1} \mathcal{G}_j$. Only the logits of tokens in $\mathcal{E}$ are perturbed. The expected green list ratio is $\mathbb{E}[\gamma] \approx (1-2^{-d})^k$. During decoding, each token can contribute one observation per permutation, yielding up to $kT$ observations from $T$ tokens.
    - **Design Motivation**: A single LoSo yields too weak a signal (bit accuracy below MPAC). The multi-permutation evergreen list maintains a relatively large green list ratio while amplifying the number of observations by a factor of $k$, significantly improving decoding reliability under limited token budgets.

3. **Constrained Token-Shard Mapping Matrix (cTMM)**:

    - **Function**: Prevents count explosion in non-perturbed shards during decoding, improving decoding robustness.
    - **Mechanism**: In the standard TMM, a single token may be mapped to the same non-perturbed shard across all $k$ permutations, inflating that shard's count by a factor of $k$ and obscuring the distinction between perturbed and non-perturbed shards. cTMM constrains each token to contribute at most 1 count to each shard: $\mathbf{A}^t[i,:] - \mathbf{A}^{t-1}[i,:] \in \{0,1\}^{2^d}$.
    - **Design Motivation**: Without this constraint, tokens that belong to no green list are counted $k$ times toward non-perturbed shards, potentially causing their counts to exceed those of perturbed shards and leading to decoding failure.

### Loss & Training

XMark is a training-free, inference-time watermarking method. Encoding is achieved by adding a positive logit bias $\delta$ to tokens in the evergreen list during LLM generation. Default settings use $d=2$ (2 bits per block), and the hyperparameter $k$ controls the quality–accuracy trade-off.

## Key Experimental Results

### Main Results

Text completion task (LLaMA-2-7B, C4 dataset, $b=8$ bits):

| Method | T=150 BA↑ | T=300 BA↑ | Avg. PPL↓ | Notes |
|--------|-----------|-----------|-----------|-------|
| MPAC | 94.00 | 98.25 | 5.08 | Small green list, poor quality |
| StealthInk | 85.00 | 92.50 | 4.13 | Good quality but low accuracy |
| CycleShift | 95.25 | 98.25 | 5.06 | Requires exhaustive enumeration |
| XMark | **98.75** | **100.00** | **4.61** | Best in both quality and accuracy |

Unwatermarked text PPL is 3.97; XMark achieves the closest PPL.

### Ablation Study

| Configuration | T=100 BA↑ | Notes |
|---------------|-----------|-------|
| LoSo ($k=1$) | 74.12 | Signal too weak |
| MPAC | 83.62 | Small green list but strong signal |
| XMark (LoSo+evergreen+cTMM) | ~95+ | Three designs work synergistically |
| XMark with TMM instead of cTMM | Decreases | Count explosion in non-perturbed shards |

### Key Findings

- XMark simultaneously outperforms all baselines in both accuracy and text quality across all token budgets ($T=150$–$300$).
- **The advantage is greatest under limited token budgets**: at $T=150$, XMark BA is 98.75% vs. MPAC's 94.00%, a gap of 4.75%.
- The advantage is even more pronounced on harder tasks such as text summarization—XMark BA 79.81% vs. MPAC 76.94%, with PPL 1.28 lower.
- Hyperparameter $k$ effectively controls the quality–accuracy trade-off: increasing $k$ improves accuracy at a slight cost to PPL.

## Highlights & Insights

- **The "inversion" thinking behind LoSo** is elegant—simply reversing the green list selection raises $\gamma$ from $\leq 0.25$ to $\geq 0.75$, substantially reducing distributional distortion. This idea is analogous to the concept of parity bits in error-correcting codes.
- **The constraint design of cTMM** precisely addresses the decoding bias introduced by the evergreen list—by limiting each token's contribution to at most 1 count per shard, it prevents the count explosion caused by multiple permutations.
- The three designs (LoSo, evergreen list, cTMM) form a tightly coupled whole: LoSo improves quality but loses signal; the evergreen list restores signal but introduces bias; cTMM eliminates that bias.

## Limitations & Future Work

- Validation is limited to LLaMA-2-7B; performance on larger or more recent models remains unknown.
- Analysis of robustness against editing attacks (paraphrase, deletion, etc.) is limited.
- The choice of $k$ requires per-scenario tuning.
- Security analysis of multi-bit watermarking (e.g., whether the watermark can be maliciously extracted or forged) is not thoroughly discussed.

## Related Work & Insights

- **vs. MPAC**: MPAC uses the message-corresponding shard as the green list ($\gamma=2^{-d}$); XMark inverts this to exclude that shard ($\gamma=1-2^{-d}$). Combined with the evergreen list and cTMM, XMark simultaneously surpasses MPAC in both quality and accuracy.
- **vs. StealthInk**: StealthInk improves quality by directly boosting the probability of high-logit tokens, but weakens the signal. XMark addresses the root problem more fundamentally by maintaining a large green list while strengthening the signal through multi-permutation intersection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combined design of LoSo + evergreen list + cTMM is creative, though the technical contribution of each individual component is modest.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task and multi-baseline comparisons are comprehensive, and the analysis across different token budgets is valuable, though model diversity is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, design motivations are clearly articulated, and figures aid understanding.
- **Value**: ⭐⭐⭐⭐ High practical value for limited-token scenarios, though the text watermarking field is highly competitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)

</div>

<!-- RELATED:END -->
