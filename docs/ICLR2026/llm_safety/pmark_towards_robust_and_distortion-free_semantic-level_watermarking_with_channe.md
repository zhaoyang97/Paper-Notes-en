---
title: >-
  [Paper Note] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints
description: >-
  [ICLR 2026][LLM Safety][LLM watermarking] PMark is a theoretically distortion-free and paraphrase-robust semantic-level watermarking method for LLMs. It employs cascaded binary filtering over candidate sentences using multiple orthogonal pivot vectors, with median-based sampling to guarantee distortion-freeness. Multi-channel design increases watermark evidence density and enhances robustness. Under paraphrase attacks, TP@FP1% reaches 95%+, outperforming prior SWM methods by 14.8%.
tags:
  - ICLR 2026
  - LLM Safety
  - LLM watermarking
  - semantic-level watermarking
  - distortion-free
  - multi-channel constraints
  - robustness theory
date: 2026-05-08
content_hash: dbe209b92473e040
---

# PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints

**Conference**: ICLR 2026
**arXiv**: [2509.21057](https://arxiv.org/abs/2509.21057)
**Code**: Coming soon
**Area**: AI Safety / Watermarking
**Keywords**: LLM watermarking, semantic-level watermarking, distortion-free, multi-channel constraints, robustness theory

## TL;DR
PMark is a theoretically distortion-free and paraphrase-robust semantic-level watermarking method for LLMs. It employs cascaded binary filtering over candidate sentences using multiple orthogonal pivot vectors, with median-based sampling to guarantee distortion-freeness. Multi-channel design increases watermark evidence density and enhances robustness. Under paraphrase attacks, TP@FP1% reaches 95%+, outperforming prior SWM methods by 14.8%.

## Background & Motivation

**Background**: LLM watermarking falls into two categories: token-level (e.g., Green-Red watermarking) and semantic-level (SWM). SWM embeds watermark signals in the sentence semantic space to improve robustness against paraphrase attacks.

**Limitations of Prior Work**:
   - Existing SWM methods (SemStamp/k-SemStamp) rely on rejection sampling, introducing distributional distortion.
   - Single-channel watermarks have sparse evidence density, making them easy to break under paraphrase attacks.
   - No rigorous theoretical framework exists for analyzing watermark properties (distortion-free conditions, robustness bounds).

**Key Challenge**: A fundamental trade-off between distortion-freeness (preserving generation quality) and robustness (resisting paraphrase attacks).

**Goal**: Simultaneously achieve theoretical distortion-free guarantees and strong practical robustness against paraphrase attacks.

**Key Insight**: Multi-channel orthogonal pivot vectors — each sentence embeds multiple independent watermark bits, multiplying evidence density.

**Core Idea**: Distortion-free median sampling + cascaded multi-orthogonal-channel filtering = high-density watermark evidence → robustness.

## Method

### Overall Architecture
During generation: for each sentence to be generated, sample $N$ candidates → apply $b$ orthogonal pivot vectors to successively bisect the candidate set → uniformly sample from the final subset. During detection: re-sample $N$ candidates per sentence to reconstruct the median → apply a soft z-test for statistical hypothesis testing. The offline variant simplifies to a zero-median prior, eliminating the need for re-sampling at detection time.

### Key Designs

1. **Proxy Function Theoretical Framework**:

    - Function: Unifies the theoretical analysis of semantic-level watermarking.
    - Core Theorem: The watermarked distribution is distortion-free if and only if $q(u) = 1/M$ (i.e., the proxy value distribution is uniform), which is difficult to satisfy in practice.
    - Design Motivation: Provides a theoretical tool for analyzing the sources of distortion in existing methods.

2. **Single-Channel Distortion-Free Sampling**:

    - Function: Ensures that single-channel watermark sampling introduces no distributional shift.
    - Mechanism: Given pivot $v$, compute cosine similarities $\langle v, \mathcal{T}(s) \rangle$ for $N$ candidates and find the median to split them into two halves. A key bit selects one half, from which a candidate is sampled uniformly. Since each candidate has selection probability $1/N$, it follows that $P_M^w(s|\pi) = P_M(s|\pi)$.
    - Theoretical Guarantee (Theorem 3): Strictly distortion-free.

3. **Multi-Channel Cascaded Filtering (Online PMark)**:

    - Function: Uses $b$ orthogonal pivot vectors to multiply evidence density by a factor of $b$.
    - Mechanism: $b$ orthogonal pivots are generated via QR decomposition. For $N$ candidates, median-based bisection is applied sequentially at each channel, retaining one half per channel (determined by a key bit): $V^{(0)} \to V^{(1)} \to \cdots \to V^{(b)}$. The final candidate is sampled uniformly from $V^{(b)}$ (containing $N/2^b$ candidates).
    - Robustness Theory (Theorem 7): If attacks corrupt per-channel evidence with probability $\epsilon$, the SNR satisfies $\text{SNR} \geq \frac{(1-2\epsilon)\sqrt{bT}}{2\sqrt{\epsilon(1-\epsilon)}}$, growing with channel count $b$ and sentence count $T$.
    - Design Motivation: Single-channel embeds only 1 bit of evidence per sentence; multi-channel embeds $b$ bits per sentence, multiplying evidence density.

4. **Offline PMark (Simplified Variant)**:

    - Function: An efficient variant that requires no re-sampling at detection time.
    - Mechanism: In high-dimensional space, random vectors are nearly orthogonal and proxy function values concentrate around $[-\epsilon, \epsilon]$, so the median is close to zero. Zero is used directly as the prior median, eliminating re-sampling overhead at detection.
    - Distortion Bound (Theorem 8): $\delta_{TV} \leq \epsilon$, with $\epsilon \leq 0.08$ in practice.

### Loss & Training
- No training required — purely a sampling-based algorithm.
- Generation requires $N$ samples per sentence ($N = 16$–$64$); the Online variant requires re-sampling at detection time to estimate the median.

## Key Experimental Results

### Main Results: TP@FP1% Under Paraphrase Attack

| Method | No Attack | Doc-P (GPT Paraphrase) | Gain |
|--------|-----------|------------------------|------|
| SemStamp (C4/Mistral) | ~99% | 73.5% | — |
| k-SemStamp | 100% | ~80% | — |
| **PMark Online** | **100%** | **97.8%** | **+24.3%** |
| **PMark Offline** | 99.7% | 92.6% | +19.1% |

### Ablation Study: Channel Count $b$ and Sample Count $N$

| N\b | b=1 | b=2 | b=3 | b=4 |
|-----|-----|-----|-----|-----|
| N=8 (Online) | 81.0 | 97.0 | 98.0 | — |
| N=16 | 84.0 | **100.0** | 100.0 | 100.0 |
| N=64 | 99.0 | 100.0 | 100.0 | 100.0 |

### Key Findings
- **Multi-channel is the key**: Detection rate jumps from 81% to 97% when increasing from $b=1$ to $b=2$.
- **Text quality improves rather than degrades**: PMark achieves lower PPL (4.37) than k-SemStamp (~5.0), as distortion-free sampling introduces no distributional shift.
- **Robust to GPT-level paraphrase**: Even under heavy GPT-based paraphrasing (Doc-P), TP@FP1% remains above 95%.

## Highlights & Insights
- **Elegant unification of theory and practice**: The paper rigorously proves distortion-free conditions and a robustness bound where SNR grows as $\sqrt{bT}$ — a rare contribution in the watermarking literature. Theory directly drives method design.
- **Core intuition of multi-channel evidence density**: Analogous to redundancy in error-correcting codes — embedding multiple independent bits per sentence allows overall signal recovery even when some bits are corrupted by attacks.
- **The offline simplification is remarkably clever**: It exploits the near-orthogonality of high-dimensional random vectors to approximate the median as zero, eliminating re-sampling overhead at detection time.

## Limitations & Future Work
- **Sampling overhead**: Each sentence requires $N$ samples ($N = 16$–$64$), which introduces latency for real-time applications.
- **Dependency on semantic encoder**: A fixed encoder (e.g., RoBERTa) is used; encoder quality directly affects watermarking performance.
- **Sentence-level embedding only**: Reliable detection requires texts of at least ~10 sentences; short texts are not well supported.
- **Future directions**: A hybrid scheme combining token-level and semantic-level watermarking — token-level for short texts, PMark for long texts — is a promising direction.

## Related Work & Insights
- **vs. SemStamp/k-SemStamp**: These methods introduce distortion via rejection sampling; PMark achieves strict distortion-freeness via median-based sampling, with a 14.8% robustness improvement.
- **vs. Green-Red token-level watermarking**: Token-level methods are fragile under paraphrasing (each token substitution represents information loss); PMark embeds watermarks at the semantic level, providing robustness to synonym-level paraphrasing.
- **vs. UPV (best token-level method)**: PMark improves paraphrase robustness by 44.6%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the theoretical framework and the multi-channel distortion-free design are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, datasets, and attack types, though experiments at larger LLM scales are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and method descriptions are clear.
- Value: ⭐⭐⭐⭐⭐ Addresses two core challenges in semantic watermarking (distortion and robustness) with high theoretical and practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[ACL 2026\] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](../../ACL2026/llm_safety/compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[AAAI 2026\] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](../../AAAI2026/llm_safety/panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](../../ACL2026/llm_safety/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)

<!-- RELATED:END -->
