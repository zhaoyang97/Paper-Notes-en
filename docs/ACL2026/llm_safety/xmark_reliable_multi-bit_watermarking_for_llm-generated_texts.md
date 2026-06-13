---
title: >-
  [Paper Note] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts
description: >-
  [ACL 2026][LLM Safety][Multi-bit watermarking] Proposes XMark, a multi-bit text watermarking method based on the Leave-one-Shard-out (LoSo) strategy and evergreen lists. By utilizing the intersection of green lists acros…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multi-bit watermarking"
  - "LLM text detection"
  - "digital watermarking"
  - "text provenance"
  - "logit perturbation"
date: 2026-05-08
content_hash: 61291008fe65082d
---

# XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts

**Conference**: ACL 2026  
**arXiv**: [2604.05242](https://arxiv.org/abs/2604.05242)  
**Code**: [https://github.com/JiiahaoXU/XMark](https://github.com/JiiahaoXU/XMark)  
**Area**: Text Watermarking  
**Keywords**: Multi-bit watermarking, LLM text detection, digital watermarking, text provenance, logit perturbation

## TL;DR

Proposes XMark, a multi-bit text watermarking method based on the Leave-one-Shard-out (LoSo) strategy and evergreen lists. By utilizing the intersection of green lists across multiple vocabulary permutations and a constrained token-shard mapping matrix, it significantly improves decoding accuracy under limited token conditions while maintaining text quality.

## Background & Motivation

**Background**: Multi-bit text watermarking enables the embedding of extractable binary information, such as user IDs or timestamps, into LLM-generated text for traceability and attribution of malicious use. Existing methods are categorized into distortion-free methods (where watermarked text follows the same distribution as unwatermarked text) and logit perturbation methods (embedding information by modifying logits).

**Limitations of Prior Work**: (1) Early methods (CycleShift, CTWL, DepthW) require brute-force enumeration of all candidate messages during decoding, making long messages computationally infeasible; (2) MPAC addresses feasibility through block coding/decoding but restricts the green list ratio to $\gamma \leq 0.25$, leading to severe distortion of token sampling probabilities and significant degradation in text quality; (3) StealthInk improves text quality but weakens the watermark signal, reducing decoding accuracy; (4) **All existing methods experience a sharp decline in decoding accuracy when the number of available tokens is limited**, whereas short texts are common in practical scenarios.

**Key Challenge**: A fundamental trade-off exists between text quality and decoding accuracy—larger green lists reduce distributional distortion but weaken the watermark signal, while smaller green lists enhance the signal but severely impact quality. This contradiction is particularly acute under limited token conditions.

**Goal**: To simultaneously improve watermarked text quality and decoding accuracy under limited token conditions.

**Key Insight**: Reverse the green list selection strategy—instead of designating the shard corresponding to the encoded message as the green list (as in MPAC), exclude that shard and designate all remaining shards as the green list, thereby increasing the green list ratio from $\leq 0.25$ to $\geq 0.75$.

**Core Idea**: Utilize Leave-one-Shard-out to enhance text quality, use the intersection of multiple permuted evergreen lists to increase observations per token to compensate for signal strength, and use a constrained TMM to prevent unperturbed shard count explosion.

## Method

### Overall Architecture

XMark follows the block coding/decoding paradigm: a $b$-bit message is divided into $r$ blocks, each containing $d$ bits. During encoding, each generated token embeds information from one message block; during decoding, message blocks are recovered from the suspect text. The core innovations lie in the LoSo + evergreen list design of the encoder and the cTMM design of the decoder.

### Key Designs

1. **Leave-one-Shard-out (LoSo) Encoding**:
    - **Function**: Significantly improves text quality by reversing the green list selection strategy.
    - **Mechanism**: While MPAC sets the shard corresponding to the message value $[\mathbf{m}_i]_{10}$ as the green list ($\gamma = 2^{-d} \leq 0.25$), LoSo excludes that shard and forms the green list from all other shards ($\gamma = 1 - 2^{-d} \geq 0.75$). Decoding recovers the message by identifying the shard with the minimum token count. For example, if $d=2$ and $\mathbf{m}_i=11$, shard $\mathcal{S}_3$ is excluded, and $\mathcal{S}_0, \mathcal{S}_1, \mathcal{S}_2$ are perturbed.
    - **Design Motivation**: Increasing the green list ratio from 0.25 to 0.75 ensures that the logit distribution of most of the vocabulary remains unchanged, pulling text quality closer to unwatermarked text.

2. **Evergreen List (Multi-permutation Intersection)**:
    - **Function**: Increases the information contribution of each token to the decoding process while maintaining a large green list, compensating for the weak signal of LoSo.
    - **Mechanism**: $k$ different hash keys are used to generate $k$ vocabulary permutations. Each permutation has its own LoSo green list $\mathcal{G}_j$. The intersection of all green lists forms the evergreen list $\mathcal{E} = \bigcap_{j=0}^{k-1} \mathcal{G}_j$. Only logits of tokens in $\mathcal{E}$ are perturbed. The expected green list ratio is $\mathbb{E}[\gamma] \approx (1-2^{-d})^k$. During decoding, each token can contribute one observation across each of the $k$ permutations, resulting in up to $kT$ observations from $T$ tokens.
    - **Design Motivation**: While a single LoSo signal is weak (bit accuracy lower than MPAC), the evergreen list across multiple permutations maintains a high green list ratio while amplifying the number of observations by $k$ times, significantly enhancing decoding reliability under limited token conditions.

3. **Constraint Token-Shard Mapping Matrix (cTMM)**:
    - **Function**: Prevents count explosion of unperturbed shards during decoding, enhancing decoding robustness.
    - **Mechanism**: In a standard TMM, a token might be mapped to the same unperturbed shard across all $k$ permutations, causing that shard's count to be amplified by $k$ and drowning out the distinction between perturbed and unperturbed shards. cTMM constrains each token to contribute at most once to each shard: $\mathbf{A}^t[i,:] - \mathbf{A}^{t-1}[i,:] \in \{0,1\}^{2^d}$.
    - **Design Motivation**: Without this constraint, tokens not belonging to any green list would be counted $k$ times for an unperturbed shard, potentially exceeding the count of perturbed shards and leading to decoding failure.

### Loss & Training

XMark is a training-free inference-time watermarking method. Encoding is achieved by adding a positive bias $\delta$ to the logits of evergreen list tokens during LLM generation. Default settings use $d=2$ (2 bits per block), and hyperparameter $k$ controls the quality-accuracy trade-off.

## Key Experimental Results

### Main Results

Text completion task (LLaMA-2-7B, C4 dataset, $b=8$ bits):

| Method | T=150 BA↑ | T=300 BA↑ | Avg PPL↓ | Notes |
|------|----------|----------|----------|------|
| MPAC | 94.00 | 98.25 | 5.08 | Small green list, poor quality |
| StealthInk | 85.00 | 92.50 | 4.13 | Good quality but low accuracy |
| CycleShift | 95.25 | 98.25 | 5.06 | Requires brute-force |
| XMark | **98.75** | **100.00** | **4.61** | Superior quality and accuracy |

Unwatermarked text PPL is 3.97; XMark's PPL is the closest.

### Ablation Study

| Configuration | T=100 BA↑ | Notes |
|------|----------|------|
| LoSo (k=1) | 74.12 | Signal is too weak |
| MPAC | 83.62 | Small green list but strong signal |
| XMark (LoSo+evergreen+cTMM) | ~95+ | Synergy of the triple design |
| XMark using TMM instead of cTMM | Decrease | Unperturbed shard count explosion |

### Key Findings

- XMark outperforms all baselines in both accuracy and text quality across all token budgets (T=150-300).
- **Greatest advantage under limited token conditions**: At T=150, XMark achieves a BA of 98.75% vs. MPAC's 94.00%, a 4.75% improvement.
- The advantage is even more pronounced in harder tasks like text summarization—XMark BA 79.81% vs. MPAC 76.94%, with 1.28 lower PPL.
- Hyperparameter $k$ effectively controls the quality-accuracy trade-off: as $k$ increases, accuracy improves while PPL slightly increases.

## Highlights & Insights

- The **"reversal" logic of the LoSo strategy** is elegant—simply reversing the green list selection increases $\gamma$ from $\leq 0.25$ to $\geq 0.75$, drastically reducing distributional distortion. This approach is reminiscent of the "parity bit" concept in error correction coding.
- The **constraint design of cTMM** precisely addresses the decoding bias introduced by evergreen lists—ensuring each token contributes at most once per shard prevents count explosion caused by multi-permutation mapping.
- The three designs (LoSo, evergreen list, and cTMM) form a tightly coupled whole—LoSo solves quality but loses signal, the evergreen list restores signal but introduces bias, and cTMM eliminates the bias.

## Limitations & Future Work

- Validated only on LLaMA-2-7B; performance on larger or newer models remains unknown.
- Robustness analysis against editing attacks (paraphrasing, deletion, etc.) is limited.
- Choice of $k$ needs to be tuned for specific scenarios.
- Security analysis of multi-bit watermarks (e.g., susceptibility to extraction or forgery) is not discussed in depth.

## Related Work & Insights

- **vs. MPAC**: MPAC uses the message-corresponding shard as the green list ($\gamma=2^{-d}$), while XMark reverses this to exclude that shard ($\gamma=1-2^{-d}$). Combined with the evergreen list and cTMM, XMark surpasses MPAC in both quality and accuracy.
- **vs. StealthInk**: StealthInk improves quality by directly increasing the probability of high-logit tokens but weakens the signal. XMark provides a more fundamental solution by using multi-permutation intersections to enhance signals while maintaining a large green list.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of LoSo+evergreen list+cTMM is creative, though individual components are technically straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across multiple tasks and baselines are thorough, and analysis under different token budgets is valuable, but model diversity is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, clear design motivation, and helpful illustrations.
- Value: ⭐⭐⭐⭐ High practical value for limited token scenarios, though the watermarking field remains highly competitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability](a_linguistics-aware_llm_watermarking_via_syntactic_predictability.md)
- [\[ACL 2026\] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)
- [\[AAAI 2026\] iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](../../AAAI2026/llm_safety/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
