---
title: >-
  [Paper Note] Approximately Aligned Decoding
description: >-
  [NeurIPS 2025][LLM Efficiency][constrained decoding] This paper proposes Approximately Aligned Decoding (AprAD), a method for constrained generation in LLMs that leverages the prefix-selection algorithm from speculative decoding. Upon encountering a constraint violation, AprAD neither reverts only one token (as in constrained generation, which causes extreme probability amplification) nor resamples entirely from scratch (as in ASAp, which incurs prohibitive computational cost). Instead, it intelligently selects a rollback position via speculative sampling, achieving a favorable trade-off between output distribution distortion and computational efficiency.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - constrained decoding
  - error avoidance
  - speculative sampling
  - probability amplification
  - lipogram
date: 2026-05-08
content_hash: 9215d663da1e7a1f
---

# Approximately Aligned Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2410.01103](https://arxiv.org/abs/2410.01103)
**Code**: None (detailed implementation description in appendix)
**Area**: LLM Efficiency
**Keywords**: constrained decoding, error avoidance, speculative sampling, probability amplification, lipogram

## TL;DR
This paper proposes Approximately Aligned Decoding (AprAD), a method for constrained generation in LLMs that leverages the prefix-selection algorithm from speculative decoding. Upon encountering a constraint violation, AprAD neither reverts only one token (as in constrained generation, which causes extreme probability amplification) nor resamples entirely from scratch (as in ASAp, which incurs prohibitive computational cost). Instead, it intelligently selects a rollback position via speculative sampling, achieving a favorable trade-off between output distribution distortion and computational efficiency.

## Background & Motivation

### State of the Field

**Background**: LLM-generated text may contain undesirable content—syntactically invalid code, hallucinated PII, profanity, failed tool calls, etc. Existing constraint-satisfaction methods each suffer from critical drawbacks: (1) *Constrained generation*: upon violation, it masks the offending token, but can catastrophically amplify low-probability outputs—if the probability of a valid token is extremely low (e.g., 0.0001), it is still forced to be selected, severely distorting the output distribution; (2) *ASAp (sampling without replacement)*: samples exactly from $P^{\mathcal{B}}$ without distorting the distribution, but on dense error sets it must discover exponentially many erroneous sequences before correctly adjusting probabilities, making its efficiency approach that of rejection sampling; (3) *Posterior estimation methods (FUDGE, SMC Steering, Ctrl-G)*: perform poorly when the constraint violation probability does not depend on the prefix.

### Approach

**Goal**: How to balance two conflicting objectives in constrained LLM generation: (1) keeping the output distribution as close as possible to the ideal error-free distribution $P^{\mathcal{B}}$; and (2) maintaining tractable inference cost? This is particularly challenging in *dense error set* scenarios, where every token has a nonzero error probability and long sequences almost inevitably contain errors.

## Method

### Overall Architecture
The core insight of AprAD is that constrained generation and ASAp can be unified as *rollback strategies upon encountering an error*—constrained generation reuses nearly the entire sequence (rolling back only one token), while ASAp reuses nothing (resampling from scratch). A natural intermediate approach is to reuse a portion of the sequence. The key question then becomes "how much to reuse?"—too much leads to probability amplification, too little wastes computation. The prefix-selection algorithm from speculative decoding provides exactly the right adaptive rollback strategy.

### Key Designs
1. **Speculative sampling as rollback strategy**: When a violating sequence $x$ is sampled from $P^B$, the updated distribution $P^{B \cup \{x\}}$ is computed. Since these two distributions are typically very close, the speculative decoding algorithm (SpecSample) is applied to determine how much of $x$'s prefix to retain, such that the retained prefix is equivalent to sampling from $P^{B \cup \{x\}}$. In high-entropy settings, most of the prefix is typically retained; rollback is more likely at low-probability tokens—a perfectly adaptive behavior.
2. **Trie-based probability caching**: A trie data structure efficiently maintains adjustments to the probability distribution induced by observed erroneous sequences. Each trie node stores both the original LLM probability and the adjusted unnormalized probability; adjustments are back-propagated along the path via the AddBadSample procedure.
3. **Theoretical bound on probability amplification**: AprAD is proven to amplify probabilities by at most a factor of 2 per rollback operation (typically far less in practice), whereas constrained generation offers no such bound. A hyperparameter $h$ allows continuous interpolation between constrained generation ($h=0$) and ASAp ($h \to \infty$).

### Loss & Training
No training is required; AprAD is a purely inference-time method. The core hyperparameter is the acceptance probability ratio in speculative sampling: $r = P^{B\cup\{x\}}(x_i|x_{1...i-1}) / P^B(x_i|x_{1...i-1})$.

## Key Experimental Results

| Task | Metric | AprAD | Constrained Gen. | ASAp | Unconstrained |
|------|--------|-------|-----------------|------|---------------|
| Lipogram (exclude vowels) | Quality score (1–5) | 4.52 | 3.56 | 1.72 | 4.68 |
| Lipogram | Constraint intent (1–3) | 2.84 | 2.32 | 2.36 | 1.00 |
| Lipogram | Generation ratio | 4.20 | 1.00 | 321.00 | 1.00 |
| BigCodeBench-15B | Pass@1 | 0.26 | 0.22 | 0.26 | 0.21 |
| BigCodeBench-15B | Hallucination-free rate | 0.98 | 0.93 | 0.98 | 0.83 |
| BigCodeBench-15B | Generation ratio | 1.08 | 1.02 | 1.56 | 1.00 |

### Ablation Study
- Simulation experiments: AprAD achieves substantially lower KL divergence than constrained generation and a far lower generation ratio than ASAp.
- In the lipogram experiment, ASAp is nearly unable to generate 200 tokens within 2,000 model calls (generation ratio of 321), whereas AprAD requires only a ratio of 4.2.
- Constrained generation produces abundant non-ASCII characters and spelling errors in the lipogram task (Cyrillic letter substitutions), whereas AprAD satisfies the constraint by selecting appropriate words that naturally avoid the target letter.
- On BigCodeBench, AprAD achieves task performance comparable to ASAp while incurring generation overhead close to that of constrained generation.

## Highlights & Insights
- **Elegant algorithmic insight**: Speculative decoding is reframed from "inference acceleration" to "intelligent rollback"—a remarkably creative cross-domain transfer.
- **Theoretical amplification bound**: The proof that each rollback incurs at most a 2× amplification factor, compared to the unbounded amplification of constrained generation, provides a valuable theoretical guarantee.
- **Lipogram visualization**: When generating "a story without the letter E," AprAD produces fluent, well-formed text, while constrained generation outputs garbled text riddled with Cyrillic substitutions.

## Limitations & Future Work
- On denser error sets, AprAD still exhibits some degree of probability amplification.
- The primary performance metric is generation ratio rather than wall-clock time, as timing measurements are unreliable due to differences in test environments and optimizations.
- Integration with search algorithms (e.g., MCTS) is a promising future direction.
- Adaptive tuning strategies for the hyperparameter $h$ remain largely unexplored.

## Related Work & Insights
Compared to constrained generation methods (e.g., llama.cpp grammar, Outlines), AprAD substantially reduces probability amplification. Compared to ASAp, AprAD achieves efficiency improvements of tens to hundreds of times on dense error sets. Compared to posterior estimation methods (FUDGE, Ctrl-G, SMC Steering), AprAD requires neither training a discriminator nor distilling an HMM, and remains effective in settings where constraint violation probability does not depend on the prefix (e.g., lipograms). AprAD fills the methodological gap of achieving both low amplification and high efficiency.

## Related Work & Insights
- A prime example of cross-domain technique transfer: speculative decoding → constrained generation.
- Practically valuable for any LLM application requiring constraint satisfaction (structured output, code generation, safety filtering).
- The generation ratio metric is more appropriate than wall-clock time for evaluating the computational efficiency of sampling methods.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Creative transfer of speculative decoding to constrained generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three levels of evaluation—simulation, lipogram, and code generation—covering both synthetic and real-world scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ A running example threads through the entire paper; visualizations of probability amplification are clear and intuitive.
- **Value**: ⭐⭐⭐⭐ A practical inference-time tool for constraint satisfaction.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[NeurIPS 2025\] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)

<!-- RELATED:END -->
