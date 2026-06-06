---
title: >-
  [Paper Note] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models
description: >-
  [ICLR 2026][LLM Alignment][diffusion language model] A2D is proposed, a token-level safety alignment method for diffusion language models (dLLMs) that trains the model to output [EOS] tokens at masked positions containin…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "diffusion language model"
  - "safety alignment"
  - "token-level defense"
  - "jailbreak"
  - "masked diffusion"
date: 2026-05-08
content_hash: 3cf3134b463fc2b7
---

# A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.23286](https://arxiv.org/abs/2509.23286)  
**Code**: Available  
**Area**: LLM Alignment
**Keywords**: diffusion language model, safety alignment, token-level defense, jailbreak, masked diffusion

## TL;DR
A2D is proposed, a token-level safety alignment method for diffusion language models (dLLMs) that trains the model to output [EOS] tokens at masked positions containing harmful content, enabling robust defense across any decoding order and any decoding step. It reduces DIJA template attack success rates from 80%+ to near zero (1.3%/0.0%) while supporting early rejection for a 19.3× speedup.

## Background & Motivation
**Background**: Diffusion language models (e.g., LLaDA, Dream) generate text through iterative unmasking rather than left-to-right generation, supporting arbitrary-order decoding. Existing safety alignment methods are inherited from AR models, relying on response-level rejection and fixed decoding order assumptions.

**Limitations of Prior Work**: The arbitrary-order decoding of dLLMs dramatically expands the attack surface—harmful content can appear at any position. The DIJA attack bypasses early rejection by interleaving adversarial text among [MASK] tokens, achieving success rates exceeding 80%. Per-token KL analysis reveals that safety alignment in dLLMs is "superficial"—effective only in the first few steps, with the safety signal rapidly decaying in subsequent steps.

**Key Challenge**: Response-level rejection in AR models assumes fixed left-to-right decoding, whereas dLLM decoding can proceed in any order at any step—making conventional alignment fundamentally inapplicable to dLLMs.

**Goal**: How can dLLMs reliably refuse harmful content across arbitrary decoding orders and arbitrary decoding steps?

**Key Insight**: Shifting safety alignment from the response level to the token level—training the model to output [EOS] at any masked position when harmful content is present, as a universal suppression signal.

**Core Idea**: Token-level [EOS] alignment combined with random mask training, enabling dLLMs to refuse harmful continuations at any position during any decoding step.

## Method

### Overall Architecture
A2D modifies the standard masked diffusion training objective: (1) for harmful text, the supervision target at all masked positions is replaced from the original token to [EOS]; (2) for safe text (including harmful prompts with safe responses), the normal reconstruction objective is retained. Uniform sampling of the mask ratio exposes the model to both early and late decoding stages.

### Key Designs

1. **Token-Level [EOS] Alignment**:

    - **Function**: Trains dLLMs to predict [EOS] at any masked position within harmful continuations.
    - **Mechanism**: For harmful samples, a random mask is sampled and the target at all masked positions is set to [EOS]. The model learns to identify harmful content from any partial context and emit a termination signal.
    - **Design Motivation**: [EOS] is already a familiar token to the model (used for padding and sequence termination), requiring no new vocabulary. Token-level alignment is naturally compatible with the arbitrary-order decoding of dLLMs.

2. **Uniform Mask Ratio Sampling**:

    - **Function**: Uniformly samples a mask ratio $\lambda \sim U(0,1)$ during training.
    - **Mechanism**: $\lambda = (1-\epsilon)t + \epsilon$, exposing the model to all stages from nearly fully masked (early decoding) to nearly unmasked (late decoding).
    - **Design Motivation**: Addresses the "superficial alignment" problem—per-token KL analysis shows existing dLLMs exhibit safety signals only in the first few steps; uniform sampling ensures alignment across all decoding stages.

3. **Early Rejection Mechanism**:

    - **Function**: Detects the [EOS] probability at the leftmost masked position at the first decoding step; if it exceeds a threshold, decoding is immediately terminated.
    - **Mechanism**: After A2D training, the model assigns high [EOS] probability at masked positions for harmful inputs, which serves as an internal safety signal. Thresholding enables fast rejection without generating any output.
    - **Effect**: Up to 19.3× faster safety termination.

### Loss & Training
Standard masked diffusion cross-entropy loss, with the sole modification being the substitution of [EOS] as the target for harmful samples. Training is conducted on 30K samples from the BeaverTails dataset and applied on top of already aligned instruction-tuned dLLMs.

## Key Experimental Results

### Main Results (Attack Success Rate ↓%)

| Model | Method | Zeroshot | PAIR | ReNeLLM | Prefilling | DIJA | Avg |
|-------|--------|----------|------|---------|------------|------|-----|
| LLaDA | Original | 14.6 | 77.5 | 56.5 | 69.6 | 82.9 | 60.2 |
| | VRPO | 2.5 | 32.3 | 19.2 | 9.0 | 45.0 | 21.6 |
| | **A2D** | **2.1** | **~low** | **~low** | **~low** | **1.3** | **~lowest** |
| Dream | A2D | - | - | - | - | **0.0** | - |

### Capability Retention

| Metric | Original | A2D |
|--------|----------|-----|
| General (MMLU, etc.) | 66.6 | **66.2** |
| Math (GSM8K, etc.) | 41.4 | **40.6** |
| Coding (HumanEval, etc.) | 32.6 | **35.0** |

### Key Findings
- A2D reduces DIJA attack success rates from 82.9% to 1.3% (LLaDA) and 0.0% (Dream), effectively eliminating template-based attacks.
- General capabilities are preserved and slightly improved (coding: 32.6→35.0), demonstrating that token-level alignment does not harm general performance.
- 0% false rejection rate on XSTest—no over-refusal of benign prompts.
- Early rejection achieves up to 19.3× faster safety termination.
- Effective across three decoding strategies (left-to-right / confidence-based / random)—a truly any-order defense.

## Highlights & Insights
- **Uncovering the core safety vulnerability of dLLMs**: KL divergence analysis provides the first systematic evidence that safety alignment in dLLMs is superficial, and more severely so than in AR models.
- **Simplicity of token-level [EOS] alignment**: No new architecture or external classifier is introduced; only the training objective is modified—minimal changes yield maximal defense.
- **Built-in safety monitoring**: [EOS] probability naturally serves as a real-time safety signal, enabling continuous monitoring throughout the decoding process.

## Limitations & Future Work
- Training is conducted solely on BeaverTails, potentially providing insufficient coverage of harmful content types.
- A2D is applied on top of already aligned models (rather than aligning from scratch); the interaction with the original alignment is not fully understood.
- Robustness against adaptive attacks (attacks designed with knowledge of the A2D mechanism) is not thoroughly analyzed.
- The early rejection threshold requires per-model tuning.

## Related Work & Insights
- **vs. AR model safety alignment**: RLHF/DPO for AR models assumes a fixed decoding order and is inapplicable to dLLMs; A2D natively supports arbitrary-order decoding.
- **vs. DIJA attack**: DIJA exploits the masking mechanism of dLLMs to construct template attacks; A2D leverages the same masking mechanism for defense—turning the adversary's own weapon against it.
- **vs. Circuit Breaker / AlphaSteer**: These methods operate in the activation space of AR models, whereas A2D directly modifies the training objective of dLLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic study of dLLM safety alignment; the token-level [EOS] approach is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three dLLMs, five attack types, multi-dimensional capability evaluation, and KL analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from vulnerability analysis to method design to experimental validation is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Paves the way for safe deployment of dLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](../../AAAI2026/llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)

</div>

<!-- RELATED:END -->
