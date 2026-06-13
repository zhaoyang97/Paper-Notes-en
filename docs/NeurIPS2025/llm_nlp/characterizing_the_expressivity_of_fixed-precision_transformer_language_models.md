---
title: >-
  [Paper Note] Characterizing the Expressivity of Fixed-Precision Transformer Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][transformer expressivity] This work precisely characterizes the expressive power of fixed-precision, strictly causal, soft-attention…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "transformer expressivity"
  - "formal language theory"
  - "linear temporal logic"
  - "fixed precision"
  - "length generalization"
date: 2026-05-08
content_hash: 1519bcb312444675
---

# Characterizing the Expressivity of Fixed-Precision Transformer Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.23623](https://arxiv.org/abs/2505.23623)  
**Code**: [GitHub](https://github.com/jiaoda-li/LTL-P-Transformer)  
**Area**: LLM Theory
**Keywords**: transformer expressivity, formal language theory, linear temporal logic, fixed precision, length generalization

## TL;DR
This work precisely characterizes the expressive power of fixed-precision, strictly causal, soft-attention, NoPE Transformers — showing it is exactly equivalent to linear temporal logic restricted to past operators, LTL[P] — and unifies this characterization with partially ordered deterministic finite automata (PODFA) and $\mathcal{R}$-trivial monoids.

## Background & Motivation
**Background**: The theoretical expressivity of Transformers is an important open problem. Existing work analyzes Transformers by relating them to formal languages, logic, and circuit complexity. Yang et al. (2024) show that fixed-precision Transformers with unique hard attention (UHA) are equivalent to full LTL[P,F,S,U] (with four temporal operators). Yang & Chiang (2024) analyze soft-attention variants and establish an upper bound of C-RASP, but a precise characterization remains open.

**Limitations of Prior Work**:
- Many theoretical results assume arbitrary or length-dependent precision, overestimating the capabilities of practical Transformers.
- UHA (unique hard attention) diverges substantially from the softmax attention used in practice.
- A precise expressivity characterization for soft-attention, fixed-precision Transformers is still missing.

**Key Challenge**: Practically deployed Transformers use fixed precision (16/32-bit) and soft attention, yet theoretical results either assume stronger models (arbitrary precision) or simpler attention mechanisms (hard attention), providing limited guidance for understanding the capabilities of real models.

**Goal**: To provide a precise expressivity characterization of fixed-precision, soft-attention, strictly causal, NoPE Transformers.

**Key Insight**: A two-way reduction — proving that such Transformers can be translated into $\text{PFO}^2[<]$ (two-variable past first-order logic), while every LTL[P] formula can be simulated by a Transformer.

**Core Idea**: The expressive power of fixed-precision soft-attention NoPE Transformers is exactly LTL[P] — a logic that can only look into the past and perform bounded counting.

## Method

### Overall Architecture
This is a purely theoretical work. The central contribution is establishing the equivalence chain:
$$\text{Fixed-precision Soft-Attn NoPE Transformer} = \text{PFO}^2[<] = \text{LTL}[\text{P}] = \text{PODFA} = \mathcal{R}\text{-trivial monoids}$$
The proof proceeds in two steps: (1) Transformer $\to$ $\text{PFO}^2[<]$ (upper bound); (2) LTL[P] $\to$ Transformer (lower bound). Combined with the independently proved result $\text{PFO}^2[<] = \text{LTL}[\text{P}]$, this yields a precise characterization.

### Key Designs

1. **Upper Bound: Transformer → $\text{PFO}^2[<]$**

    - **Function**: Proves that any language recognized by a fixed-precision Transformer can be described by a $\text{PFO}^2[<]$ formula.
    - **Mechanism**: Under fixed precision, attention weights can only take finitely many values. Since attention distributions are position-invariant (NoPE), attention patterns depend only on which tokens have appeared in the past and how many times — truncated by finite precision — which is exactly what $\text{PFO}^2[<]$ can express.
    - **Design Motivation**: The "weighted average" computed by soft attention under fixed precision can distinguish only finitely many cases, drastically reducing its power compared to the arbitrary-precision setting.

2. **Lower Bound: LTL[P] → Transformer**

    - **Function**: Proves that the language corresponding to any LTL[P] formula can be recognized by a Transformer.
    - **Mechanism**: LTL[P] formulas are simulated by structural induction. Atomic formulas $\pi_a$ ("current symbol is $a$") are trivially realizable; Boolean operations are realizable at any precision; the past operator P ("does there exist a past position satisfying $\psi$") is realizable via the attention mechanism, which can attend to historical positions satisfying $\psi$.
    - **Design Motivation**: Under the NoPE setting, there is no mechanism to simulate the future operator F, until operator U, or since operator S, so expressivity is strictly weaker than LTL[P,F,S,U].

3. **Equivalence of LTL[P] with PODFA and $\mathcal{R}$-trivial Monoids**

    - **Function**: Connects the logical characterization to automata theory and algebraic theory.
    - **PODFA**: Partially ordered deterministic finite automata — DFAs whose state transition graph forms a partial order.
    - **$\mathcal{R}$-trivial monoids**: Monoids whose syntactic $\mathcal{R}$-classes are all trivial.
    - **Formal Language Perspective**: This corresponds precisely to the class of languages recognized by left-deterministic polynomials.

4. **Extension to Transformer Language Models**

    - **Function**: Proves that Transformer LMs (generative models rather than recognizers) have the same expressive power.
    - **Mechanism**: A language model implicitly defines language recognition via thresholding $p(\text{eos} \mid \mathbf{w})$, which is equivalent to recognition by a classifier.

### Loss & Training
No training methodology is proposed — this is a purely theoretical work with empirical validation.

## Key Experimental Results

### Main Results
Length generalization experiments (trained on short sequences, tested on longer sequences):

| Language | In LTL[P]? | Length Generalization Accuracy |
|---|---|---|
| $(ab)^*$ | ✅ | **100%** |
| $a^*b^*$ | ✅ | **100%** |
| All languages in LTL[P] | ✅ | **100%** |
| Bounded Dyck-1 | ❌ | < 100% |
| $(aa)^*$ (requires counting) | ❌ | **Fails** |
| All languages outside LTL[P] | ❌ | **Consistently fails** |

### Ablation Study

| Configuration | Result | Note |
|---|---|---|
| Varying learning rates / random seeds | Conclusions unchanged | Languages in LTL[P] always succeed; outside always fail |
| Increasing layers / heads | Still fails outside LTL[P] | Architectural changes do not alter theoretical limits |

### Key Findings
- **Strong theory–practice agreement**: Languages in LTL[P] achieve 100% generalization; those outside fail without exception.
- **Fixed precision + NoPE severely limits expressivity**: Many seemingly simple languages (e.g., bounded Dyck, even-length strings) exceed the model's capacity.
- **Soft attention ≠ hard attention**: Soft attention (this work) can only express LTL[P], whereas UHA can express full LTL[P,F,S,U] — a substantial gap.

## Highlights & Insights
- **Exact characterization rather than bounds**: Unlike most expressivity analyses, this work establishes a complete equivalence with no remaining gap.
- **Unification across multiple mathematical frameworks** is particularly elegant: logic (LTL[P]) = automata (PODFA) = algebra ($\mathcal{R}$-trivial monoids) = formal language theory (left-deterministic polynomials) = computation (fixed-precision Transformer).
- **Practical implication**: NoPE Transformers are far less capable than commonly assumed — they cannot even reliably perform bracket matching. This explains why positional encodings are critical for Transformer performance.

## Limitations & Future Work
- **Restricted to NoPE (no positional encoding)**: Practical Transformers employ positional encodings such as RoPE, APE, and ALiBi; characterizing expressivity with positional encodings is an important open problem.
- **Fixed precision as idealization**: The numerical behavior of gradients and activations during training is more complex than what the theoretical model assumes.
- **Language recognition/generation only**: Expressivity on non-linguistic tasks such as reasoning and arithmetic is not addressed.
- **Future directions**: (1) Expressivity characterization with RoPE; (2) Extension to Transformers with scratchpad or chain-of-thought.

## Related Work & Insights
- **vs. Yang et al. (2024, UHA)**: UHA Transformer = LTL[P,F,S,U] (full four operators); this work's soft-attention model = LTL[P] (past operators only), demonstrating that soft attention is strictly weaker than UHA under fixed precision.
- **vs. Yang & Chiang (2024, C-RASP)**: They establish the upper bound C-RASP; this work provides the precise characterization LTL[P].
- **vs. Merrill et al. (Saturated Transformers)**: They analyze saturated attention patterns; this work analyzes fixed-precision soft attention.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Precise expressivity characterization represents one of the hardest problem types in this field; the equivalence proof and multi-framework unification are both significant contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Carefully designed language hierarchies and length generalization experiments yield perfect alignment between theoretical predictions and empirical results.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure (the Figure 1 roadmap is excellent), mathematically rigorous, and accessible to non-specialists.
- **Value**: ⭐⭐⭐⭐⭐ — Provides one of the most precise theoretical foundations for understanding the capability boundaries of Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Characterizing the Expressivity of Local Attention in Transformers](../../ACL2026/llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)
- [\[NeurIPS 2025\] On the Role of Hidden States of Modern Hopfield Network in Transformer](on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)

</div>

<!-- RELATED:END -->
