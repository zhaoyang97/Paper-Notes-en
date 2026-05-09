---
title: >-
  [Paper Note] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching
description: >-
  [ICLR 2026][LLM/NLP][Diffusion LLM] This paper proposes d²Cache, a training-free approximate KV cache framework for diffusion-based LLMs (dLLMs), achieving 4.1× inference speedup while simultaneously improving generation quality via a two-stage strategy: deterministic prior-guided masked token selection followed by attention-aware non-masked token selection.
tags:
  - ICLR 2026
  - LLM/NLP
  - Diffusion LLM
  - KV Cache
  - Inference Acceleration
  - dLLM
  - Attention Pruning
date: 2026-05-08
content_hash: c594ebd71769e37e
---

# d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching

**Conference**: ICLR 2026
**arXiv**: [2509.23094](https://arxiv.org/abs/2509.23094)
**Code**: [https://github.com/Kamichanw/d2Cache](https://github.com/Kamichanw/d2Cache)
**Area**: LLM/NLP
**Keywords**: Diffusion LLM, KV Cache, Inference Acceleration, dLLM, Attention Pruning

## TL;DR
This paper proposes d²Cache, a training-free approximate KV cache framework for diffusion-based LLMs (dLLMs), achieving 4.1× inference speedup while simultaneously improving generation quality via a two-stage strategy: deterministic prior-guided masked token selection followed by attention-aware non-masked token selection.

## Background & Motivation

**Background**: Diffusion-based LLMs (dLLMs, e.g., LLaDA, Dream) generate text through iterative denoising with bidirectional attention, competing with autoregressive models (ARMs) on reasoning and instruction-following tasks.

**Limitations of Prior Work**: dLLMs employ bidirectional attention, where updating any masked token at each step changes the context for all tokens, rendering **standard KV caching entirely inapplicable**—the full KV states of the entire sequence must be recomputed at every step. Existing approximate KV cache methods (dLLM-Cache, Fast-dLLM) operate at a coarse granularity, partitioning the sequence into static/dynamic segments with fixed update windows, resulting in insufficient flexibility or complex hyperparameter tuning.

**Key Challenge**: The bidirectional attention mechanism in dLLMs provides contextual modeling advantages at the cost of the natural KV cache acceleration enjoyed by ARMs. The key challenge is recovering cache-based acceleration without degrading generation quality.

**Goal**: Design a fine-grained, adaptive KV cache strategy that precisely identifies which tokens truly require KV updates at each step.

**Key Insight**: Fine-grained analysis reveals that the KV states of masked tokens undergo three phases (slow change → rapid change → stabilization), requiring updates only during the rapid-change phase; prompt/decoded tokens exhibit highly concentrated attention, necessitating updates only for high-attention tokens.

**Core Idea**: A two-stage fine-grained token selection scheme—Stage 1 selects masked tokens via deterministic priors, Stage 2 selects remaining tokens by attention scores—updating KV states for only a small subset of critical tokens per step while reusing cached states for all others.

## Method

### Overall Architecture
dLLMs begin from a fully masked sequence and generate text through $T$ iterative denoising steps. At each step, d²Cache classifies tokens into three categories (prompt/masked/decoded) and applies a two-stage strategy to identify a small subset of tokens requiring KV updates; the remaining tokens reuse cached KV states from the previous step, substantially reducing per-step computation.

### Key Designs

1. **Three-Phase KV State Analysis (Core Observation)**:

    - Function: PCA visualization of masked token KV trajectories reveals three distinct phases.
    - Core Findings: (1) Early slow change—tokens far from decoded positions exhibit minimal KV variation; (2) Pre-decoding rapid change—tokens about to be decoded undergo sharp KV transitions; (3) Post-decoding stabilization—already-decoded token KV states remain nearly constant.
    - Design Motivation: KV updates are only necessary during the rapid-change phase; caching is safe in all other phases.

2. **Stage 1: Certainty Prior-Guided Selection**:

    - Function: Selects the subset of masked tokens that are about to be decoded and updates their KV states.
    - Mechanism: Defines a position-aware certainty density $D(i) = \sum_{j \notin M} \exp(-|i-j|^2 / 2\sigma^2)$, measuring the density of known tokens surrounding each masked token. Higher density indicates earlier decoding (since dLLMs tend to decode the next token near already-decoded positions). The top-$k$ masked tokens by score $D(i) \cdot s^i$ (certainty prior × prediction confidence) are selected.
    - Design Motivation: Experiments show that 90% of tokens are decoded within 10 steps of the previous decoded position, making certainty density a reliable indicator for predicting imminent decoding.

3. **Stage 2: Attention-Aware Selection**:

    - Function: Selects a high-attention subset of prompt and decoded tokens for KV updates.
    - Mechanism: Applies the Attention Rollout algorithm to recursively aggregate attention matrices across layers, $C^{(l)} = W^{(l)} \cdot C^{(l-1)}$, yielding global influence scores $c_j = \sum_i C_{ij}^{(N)}$. Tokens are ranked by influence, and the smallest set whose cumulative probability exceeds threshold $p$ is selected.
    - Design Motivation: Analysis shows that dLLM attention is similarly concentrated on a small number of salient tokens (analogous to ARMs), and KV updates for low-attention tokens have negligible impact on outputs.

4. **Certainty Prior-Guided Decoding (Byproduct)**:

    - Function: Uses certainty priors rather than raw confidence to determine decoding order.
    - Mechanism: Tokens with higher certainty (closer to already-decoded regions) are decoded first, producing an approximately left-to-right generation order.
    - Design Motivation: Mitigates the problem of premature overconfidence for end-of-sequence tokens in dLLMs.

### Loss & Training
d²Cache is a **completely training-free** inference acceleration framework that does not modify any model parameters. All optimizations are applied online at inference time.

## Key Experimental Results

### Main Results

**LLaDA-8B-Instruct (GSM8K, 4-shot)**:

| Method | Throughput | Latency | Accuracy |
|--------|-----------|---------|----------|
| Vanilla | 2.77 (1.0×) | 110.26s | 77.6 |
| dLLM-Cache | 8.29 (3.0×) | 30.34s | 76.8 |
| Fast-dLLM | 9.64 (3.5×) | 26.15s | 77.0 |
| **d²Cache** | **11.39 (4.1×)** | **22.41s** | **79.2** |

4.1× speedup with a simultaneous accuracy improvement of 1.6%.

### Ablation Study

| Configuration | Speedup | Quality | Notes |
|--------------|---------|---------|-------|
| Full d²Cache | 4.1× | ↑ | Both stages complete |
| Stage 1 only | ~3× | ↑ | No prompt/decoded token selection |
| Stage 2 only | ~2.5× | ≈ | No efficient masked token selection |
| Certainty prior decoding | — | ↑↑ | Replacing confidence-based decoding yields larger quality gain |

### Key Findings
- d²Cache is consistently effective across two dLLMs (LLaDA and Dream), **simultaneously improving both speed and quality**—a highly unusual property for acceleration methods.
- Certainty prior-guided decoding produces higher generation quality than default confidence-based decoding by enforcing a more structured, approximately left-to-right generation order.
- The finding that 90% of tokens are decoded within 10 steps of the previous decoded position reveals that, despite theoretical support for arbitrary decoding order, dLLMs exhibit strong spatial locality in practice.
- The observation that attention is highly concentrated on prompt and decoded tokens can be directly transferred to other dLLM acceleration research.

## Highlights & Insights
- **Dual Win on Speed and Quality**: By guiding decoding order via certainty priors, d²Cache not only accelerates inference but also alleviates premature overconfidence in dLLMs, improving generation quality. The insight that "a good caching strategy equals a good decoding strategy" is particularly profound.
- **Value of Fine-Grained Analysis**: The three-phase KV dynamics analysis provides foundational understanding for the broader dLLM inference optimization community, upon which subsequent work can build additional strategies.
- **Training-Free Plug-and-Play**: No model retraining is required; the method applies directly at inference time. Adapting to new dLLMs requires only running an Attention Rollout analysis.

## Limitations & Future Work
- Attention Rollout itself incurs $O(NL^2)$ computational overhead, which is cheaper than a full forward pass but non-negligible.
- The Gaussian kernel width $\sigma$ in the certainty prior and threshold $p$ are hyperparameters that may require tuning across different tasks.
- Validation is limited to LLaDA and Dream; the dLLM ecosystem is still maturing, and whether the method generalizes to larger future dLLMs remains to be verified.
- Combinations with other inference acceleration techniques such as quantization and sparsity are not explored.

## Related Work & Insights
- **vs. dLLM-Cache**: dLLM-Cache employs a coarse-grained prompt/response segmentation strategy with fixed-frequency updates; d²Cache's fine-grained, token-level adaptive strategy is more flexible and efficient.
- **vs. Fast-dLLM**: Fast-dLLM uses block-level semi-autoregressive decoding with block-wise KV caching; d²Cache's per-token strategy offers greater flexibility.
- **vs. ARM KV Cache Pruning**: d²Cache successfully transfers the attention concentration observation from ARMs to the bidirectional attention setting of dLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ The dLLM KV cache direction is nascent; the fine-grained analysis and two-stage design are well-grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two dLLMs and multiple datasets with ablations, but lacks larger-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis-driven method design with clear logic and excellent visualizations.
- Value: ⭐⭐⭐⭐ dLLM inference acceleration is a current hotspot; 4.1× speedup with quality improvement has strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](../../ACL2026/llm_nlp/please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](../../NeurIPS2025/llm_nlp/adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)

</div>

<!-- RELATED:END -->
