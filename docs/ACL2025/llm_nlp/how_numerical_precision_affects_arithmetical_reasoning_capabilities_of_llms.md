---
title: >-
  [Paper Note] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs
description: >-
  [ACL 2025 (Findings)][LLM (Other)][numerical precision] Based on circuit complexity theory, this study rigorously proves that low-precision (e.g., int4/int8) Transformers require super-polynomial size to solve iterative addition and integer multiplication, whereas standard-precision (float32) Transformers can efficiently solve three classes of arithmetic tasks with constant depth and polynomial width. The critical impact of precision on arithmetic capability is empirically ve…
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "numerical precision"
  - "arithmetic reasoning"
  - "Transformer expressiveness"
  - "circuit complexity"
  - "quantization"
date: 2026-05-08
content_hash: 30984e55381577ff
---

# How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2410.13857](https://arxiv.org/abs/2410.13857)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: numerical precision, arithmetic reasoning, Transformer expressiveness, circuit complexity, quantization

## TL;DR
Based on circuit complexity theory, this study rigorously proves that low-precision (e.g., int4/int8) Transformers require super-polynomial size to solve iterative addition and integer multiplication, whereas standard-precision (float32) Transformers can efficiently solve three classes of arithmetic tasks with constant depth and polynomial width. The critical impact of precision on arithmetic capability is empirically verified on LLaMA-3.1-8B.

## Background & Motivation

**Background**: Transformer-based LLMs demonstrate outstanding performance across various NLP tasks, but mathematical reasoning remains a bottleneck. Although various strategies (e.g., CoT prompting, inference-time search) show improvements, there is a lack of deep understanding regarding the intrinsic limitations of LLMs' mathematical capabilities.

**Limitations of Prior Work**: (a) Prior theoretical analyses (Feng et al., 2023; Yang et al., 2024) assume that each digit represents an independent token, which is inconsistent with the tokenization of actual LLMs (modern LLMs split numbers into tokens of up to 3 digits); (b) in practice, low-precision quantization (int4, float8) leads to a significant decline in mathematical capabilities, yet a theoretical explanation is lacking.

**Key Challenge**: How does numerical precision affect the arithmetic expressiveness of Transformers? Why do low-precision models fail in arithmetic tasks?

**Goal**: Provide rigorous expressiveness bounds for three fundamental arithmetic tasks (integer addition, iterative addition, and integer multiplication), including impossibility results under low precision and solvability proofs under standard precision.

**Key Insight**: Model the Transformer as a computational circuit and analyze the effect of precision on expressiveness using the separation results of circuit complexity classes $\mathsf{AC}^0$ and $\mathsf{TC}^0$.

**Core Idea**: The expressiveness of low-precision Transformers (constant bit-width) is restricted to $\mathsf{AC}^0$, whereas iterative addition and multiplication exceed $\mathsf{AC}^0$; standard precision ($O(\log n)$ bits) upgrades the Transformer to $\mathsf{TC}^0$, which is sufficient to efficiently solve all arithmetic tasks.

## Method

### Overall Architecture
Three arithmetic tasks of increasing complexity are studied:
- $\text{ADD}_p(n)$: addition of two $n$-digit base-$p$ integers.
- $\text{IterADD}_p(n,k)$: iterative addition of $k$ $n$-digit integers.
- $\text{MUL}_p(n,l)$: multiplication of two $n$-digit integers (with output truncated to $l$ digits).

A tokenization scheme consistent with modern LLMs is adopted: each token contains at most $c$ consecutive digits. The results are generated token-by-token under the autoregressive paradigm.

### Key Designs

1. **Low-Precision Impossibility Results (Theorems 4.2, 4.3)**

    - Function: Prove that bounded-depth Transformers with constant precision ($c$ bits per neuron) require super-polynomial size to solve iterative addition and multiplication.
    - Mechanism: Model constant-precision Transformers as $\mathsf{AC}^0$ circuits (polynomial size, constant depth, unbounded fan-in AND/OR gates). Utilizing the classic result that the Majority function is not in $\mathsf{AC}^0$ (Razborov, 1987), it is proven via reduction that $\text{IterADD}$ and $\text{MUL}$ are also not in $\mathsf{AC}^0$.
    - Design Motivation: Under low precision, a single neuron cannot store intermediate calculation results (such as the accumulation of carry chains), requiring a super-polynomial number of neurons for distributed storage, which leads to model size explosion.
    - Special Case: Simple two-integer addition $\text{ADD}$ can be solved in $\mathsf{AC}^0$ (Theorem 4.1), requiring only $O(n^2)$ width.

2. **Standard-Precision Solvability Proofs (Theorems 5.1-5.3)**

    - Function: Constructively prove that $O(\log n)$-precision Transformers can efficiently solve all three classes of tasks.
    - Mechanism: The expressiveness of logarithmic-precision Transformers corresponds to $\mathsf{TC}^0$ (which includes Majority gates). Concrete construction schemes:
        - $\text{ADD}_p(n)$: constant depth + constant width (independent of $n$).
        - $\text{IterADD}_p(n,k)$: constant depth + constant width (independent of $n$ and $k$).
        - $\text{MUL}_p(n,l)$: constant depth + $O(n^2)$ width.
    - Design Motivation: $O(\log n)$-bit precision allows each neuron to store integer values on the scale of $n$ (since $\log_2(n) \approx 32$ corresponds to context lengths of $\sim$100K), which is sufficient to represent carry chains and partial products.

3. **The Critical Transition from $\mathsf{AC}^0$ to $\mathsf{TC}^0$**

    - Function: Explain the essence of precision improvement.
    - Mechanism: Constant precision $\rightarrow$ $\mathsf{AC}^0$ (without Majority); logarithmic precision $\rightarrow$ $\mathsf{TC}^0$ (with Majority). This is not an incremental improvement but a qualitative change in complexity classes—even if precision is only increased from 8 bits to 32 bits (constant $\rightarrow$ logarithmic), it crosses the $\mathsf{AC}^0/\mathsf{TC}^0$ boundary.
    - Practical Implication: Float32 is not a "luxury" but a "necessity" for mathematical reasoning.

### Empirical Verification Strategy
- **Training small models from scratch**: base-2/base-10, 3-layer/5-layer Transformers, float32 vs. bfloat16.
- **LLaMA-3.1-8B**: original bfloat16 vs. int4 quantization vs. LoRA fine-tuning vs. QLoRA fine-tuning.

## Key Experimental Results

### Main Results (Training from Scratch)

| Task | Precision | Digit Length = Short | Digit Length = Long | Trend |
|------|-----------|-----------|-----------|------|
| Integer Addition (base-10) | float32 | >94% | >94% | Almost no decline |
| Integer Addition (base-10) | bfloat16 | >94% | >94% | Almost no decline |
| Iterative Addition (base-2, 3 numbers) | float32 | ~100% | ~100% | Stable |
| Iterative Addition (base-2, 3 numbers) | bfloat16 | ~100% | Significant decline | Sharp deterioration at lengths 7-10 |
| Integer Multiplication (base-2) | float32 | ~100% | Gradual decline | Remains high for long digits |
| Integer Multiplication (base-2) | bfloat16 | ~100% | Sharp decline | Drops to nearly zero after length 13 |

### LLaMA-3.1-8B Experiments

| Task | Setting | Short Digits | Long Digits | Description |
|------|------|--------|--------|------|
| Iterative Addition (3 numbers) | Original bfloat16 | ~80% | ~60% | Baseline |
| Iterative Addition (3 numbers) | int4 Quantization | ~60% | ~40% | Drops by ~20% |
| Iterative Addition (3 numbers) | LoRA (bf16) | ~95% | ~85% | Large fine-tuning gain |
| Iterative Addition (3 numbers) | QLoRA (int4) | ~90% | ~70% | Low-precision fine-tuning still worse than original bf16 |

### Key Findings
- **Simple addition is insensitive to precision**: float32 and bfloat16 perform almost identically, which is highly consistent with theoretical predictions ($\text{ADD} \in \mathsf{AC}^0$).
- **Precision gaps for complex tasks widen sharply with increasing digits**: in base-2 multiplication, the accuracy of bfloat16 plummets after length 13, whereas float32 maintains a reasonable level.
- **Low-precision fine-tuning cannot compensate for precision deficiencies**: LLaMA fine-tuned with QLoRA (int4) even performs worse than the original bfloat16 model on certain tasks.
- **Deeper networks partially alleviate but cannot cure the issue**: 5-layer models perform better than 3-layer models, but bfloat16 still collapses on long digits.

## Highlights & Insights
- **Precise Alignment Between Theory and Experiment**: The three impossibility/possibility theorems closely reflect the experimental phenomena of the three tasks—addition is feasible under both precisions, while iterative addition and multiplication collapse under low precision. This perfect correspondence between theoretical prediction and empirical validation is highly compelling.
- **Practical Guidance**: Direct answer to "how much quantization hurts mathematical reasoning"—simple computations can be quantized, whereas complex computations involving carry propagation (e.g., multi-number addition, multiplication) are non-quantizable. This provides direct guidance for LLM deployment strategies.
- **Value of the Circuit Complexity Framework**: The $\mathsf{AC}^0$ vs $\mathsf{TC}^0$ separation provides a clean framework for understanding Transformer expressiveness, holding more depth than empirical "quantization loss" analyses.

## Limitations & Future Work
- **Only three basic arithmetic operations covered**: More complex operations such as division, modulo, and floating-point arithmetic are not considered.
- **Only a single factor (precision) is analyzed**: Practical mathematical reasoning is also affected by CoT length, context window, attention patterns, etc.
- **Gap between theoretical and practical definitions of precision**: There is a gap between "constant precision" ($c$ bits) in theory and actual bfloat16 (16-bit) in practice—bfloat16 is not strictly $\mathsf{AC}^0$, but the finite bits indeed restrict the range of intermediate computation.
- **Mixed-precision not discussed**: Practical deployment often adopts different precisions across different layers. Is there a selective quantization scheme where "critical layers maintain high precision"?

## Related Work & Insights
- **vs. Feng et al. (2023)**: They assume a single token per digit (unrealistic), whereas this work adopts the multi-digit tokenization of modern LLMs, making the conclusions more practically valuable.
- **vs. Li et al. (2024)**: They proved that constant-precision Transformers $\subseteq \mathsf{AC}^0$. Building on this, this work further proves the impossibility of specific arithmetic tasks.
- **vs. empirical studies on quantization (Jin et al., 2024; Marchisio et al., 2024)**: While they empirically found that quantization hurts mathematical capabilities, this work provides a theoretical explanation.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical conclusions are novel, but the technical approach (circuit complexity analysis of Transformers) has been paved by prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive verification with training from scratch + large-scale LLaMA + multi-task and multi-precision.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory + rich experiments + clear practical guidance, with a complete 40-page appendix.
- Value: ⭐⭐⭐⭐ Deep theory and useful practical guidance, but the critical bottleneck of mathematical reasoning might not be restricted to precision.
- Value: Pending evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States](enigmatom_improve_llms_theory-of-mind_reasoning_capabilities_with_neural_knowled.md)
- [\[ACL 2025\] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)
- [\[ACL 2025\] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
