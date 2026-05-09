---
title: >-
  [Paper Note] Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models
description: >-
  [AAAI 2026][LLM Safety][Hallucination] This paper employs computational complexity theory to demonstrate that the per-step inference complexity of Transformer-based LLMs is $O(N^2 \cdot d)$. Grounded in the Hartmanis–Stearns Time Hierarchy Theorem, it proves that any computational task exceeding this complexity bound—such as $O(n^3)$ matrix multiplication, $O(n^k)$ token enumeration, or TSP verification—necessarily causes hallucination. Furthermore, LLM agents are shown to be incapable of verifying the correctness of such tasks.
tags:
  - AAAI 2026
  - LLM Safety
  - Hallucination
  - Computational Complexity
  - Time Hierarchy Theorem
  - Transformer
  - Agentic AI
  - Verification Impossibility
date: 2026-05-08
content_hash: 715b035e59cfa0b8
---

# Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models

**Conference**: AAAI 2026
**arXiv**: [2507.07505](https://arxiv.org/abs/2507.07505)
**Area**: LLM Safety
**Keywords**: Hallucination, Computational Complexity, Time Hierarchy Theorem, Transformer, Agentic AI, Verification Impossibility

## TL;DR

This paper employs computational complexity theory to demonstrate that the per-step inference complexity of Transformer-based LLMs is $O(N^2 \cdot d)$. Grounded in the Hartmanis–Stearns Time Hierarchy Theorem, it proves that any computational task exceeding this complexity bound—such as $O(n^3)$ matrix multiplication, $O(n^k)$ token enumeration, or TSP verification—necessarily causes hallucination. Furthermore, LLM agents are shown to be incapable of verifying the correctness of such tasks.

## Background & Motivation

**Background**: LLMs have been widely deployed across diverse domains; however, hallucination—the generation of false, inaccurate, or nonsensical content—remains a central obstacle to reliable deployment. Simultaneously, the rise of Agentic AI has extended LLMs from information provision to the autonomous execution of real-world tasks (e.g., financial transactions, booking services, legal document processing).

**Limitations of Prior Work**: (1) Understanding of hallucination has largely remained empirical (attributed to insufficient data or distribution shift), lacking a fundamental explanation from computational theory; (2) proposals to have one agent verify another agent's output have not been subjected to rigorous theoretical scrutiny; (3) reasoning models (e.g., o3, R1) are widely assumed to overcome hallucination, yet no formal justification exists.

**Key Challenge**: The reasoning capacity of LLMs is bounded by the computational complexity of their architecture, while real-world tasks can be arbitrarily complex. This intrinsic gap implies that certain hallucinations are unavoidable.

**Goal**: To rigorously establish, from a computational complexity perspective, the capability upper bound of LLMs, and to demonstrate that hallucination constitutes a fundamental architectural limitation.

**Key Insight**: LLM inference is treated as a deterministic computational process, and classical results from complexity theory—specifically the Time Hierarchy Theorem—are leveraged to prove the existence of infinitely many tasks that exceed this computational capacity.

**Core Idea**: LLM computation is subject to a hard upper bound of $O(N^2 \cdot d)$; tasks exceeding this complexity will inevitably cause hallucination—not as a flaw, but as a theorem.

## Method

### Overall Architecture

The paper adopts a structure of theoretical analysis supported by illustrative examples. It first establishes the computational complexity upper bound of LLM inference, then demonstrates failure on super-complexity tasks through three progressively demanding examples, and finally provides a formal proof grounded in the Time Hierarchy Theorem.

### Key Designs

1. **Computational Complexity Analysis of LLMs**
   - **Core Argument**: The self-attention mechanism in Transformers operates at $O(N^2 \cdot d)$ complexity over $N$ tokens of dimension $d$; all other operations are linear or lower.
   - **Empirical Validation**: For any 17-token input, Llama-3.2-3B-Instruct consistently executes exactly 109,243,372,873 floating-point operations—regardless of input content.
   - **Core Intuition**: If the computational task expressed in a prompt exceeds $O(N^2 \cdot d)$ in complexity, the LLM cannot complete it correctly.

2. **Three Progressive Impossibility Examples**
   - **Example 1 (Token Enumeration)**: Enumerating all length-$k$ strings over $n$ tokens requires $O(n^k)$ time, which quickly surpasses $O(N^2 \cdot d)$ for moderately large $k$.
   - **Example 2 (Matrix Multiplication)**: Naïve matrix multiplication requires $O(n^3)$; LLMs cannot correctly perform this when the matrix dimension exceeds the vocabulary size.
   - **Example 3 (Agentic AI Verification)**: Verifying an optimal TSP solution requires checking $(n-1)!/2$ routes—a complexity that neither a solving agent nor a verifying agent can handle.

3. **Formal Theorem and Proof**
   - **Theorem 1**: Given a prompt of length $N$ encoding a computational task of complexity $O(n^3)$ or higher, any LLM or LLM agent will necessarily hallucinate.
   - **Proof**: The Hartmanis–Stearns Time Hierarchy Theorem guarantees the existence of decision problems solvable in $O(t_2(n))$ but not in $O(t_1(n))$.
   - **Corollary**: There exist tasks that an LLM agent can execute, yet whose correctness cannot be verified by any LLM agent.

### Discussion on Reasoning Models

The authors argue that reasoning models (o3, R1) cannot overcome this limitation: (1) the complexity of fundamental operations remains unchanged; and (2) the reasoning token budget is far smaller than what computationally demanding tasks require. Research from Apple corroborates this, showing that reasoning models exhibit "reasoning collapse" on exponential-complexity problems such as the Tower of Hanoi.

## Key Experimental Results

### FLOPs Measurement (Llama-3.2-3B-Instruct)

| Input | Token Count | Total FLOPs | Difference |
|-------|-------------|-------------|------------|
| "You are a helpful assistant..." | 17 | 109,243,372,873 | — |
| "Can you find a number that..." | 17 | 109,243,372,873 | **Identical** |

### Task Complexity vs. LLM Capability

| Task | Complexity | Exceeds $O(N^2 \cdot d)$? | LLM Correctness |
|------|-----------|--------------------------|-----------------|
| Simple Q&A | $O(N)$ | No | Probabilistically possible |
| Token enumeration | $O(n^k)$ | Yes | ✗ |
| Large matrix multiplication | $O(n^3)$ | Yes | ✗ |
| TSP optimal solution verification | $O((n-1)!/2)$ | Yes | ✗ |

### Key Findings

- LLM FLOPs are entirely determined by input length and model dimensionality, independent of task content.
- The Time Hierarchy Theorem guarantees the existence of infinitely many task classes that LLMs cannot correctly complete.
- Multi-agent mutual verification strategies fail equally on super-complexity tasks.

## Highlights & Insights

1. **Hallucination elevated from empirical phenomenon to theoretical inevitability**: hallucination is a direct corollary of the computational complexity upper bound.
2. **Impossibility of agent-verifies-agent**: a fundamental challenge to multi-agent cross-checking strategies.
3. **Minsky-style insight**: while individual LLMs are computationally bounded, collaborative multi-LLM systems may exhibit emergent capabilities.
4. **Empirical validation is intuitive and compelling**: inputs of identical length but different content yield exactly the same FLOPs.

## Limitations & Future Work

1. The analysis is coarse-grained: it does not distinguish between deterministic correctness and probabilistic approximate correctness.
2. Multi-step autoregressive generation accumulates computational capacity; Chain-of-Thought reasoning may circumvent single-step limitations.
3. Tool-use scenarios are not rigorously analyzed—when LLMs invoke calculators or code interpreters, their effective computational capacity expands substantially.
4. At only 6 pages, the paper does not engage deeply with related work on circuit complexity class TC0.
5. The definition of "hallucination" is overly broad, conflating computational errors with the generation of factually false statements.

## Related Work & Insights

- This work is complementary to Xu et al. (2024) "Hallucination is Inevitable," which approaches the problem from an information-theoretic perspective.
- **Implication for Agentic AI**: tasks exceeding the complexity bound must be offloaded to dedicated solvers.
- The "thinking tokens" of reasoning models are essentially additional computation steps rather than a change in per-step complexity.

## Rating

⭐⭐⭐

- **Novelty** ⭐⭐⭐: The core argument is not entirely novel, but the systematic treatment adds value.
- **Experimental Thoroughness** ⭐⭐: Limited to FLOPs measurement; lacks systematic empirical validation.
- **Writing Quality** ⭐⭐⭐⭐: The argument is clearly presented within 6 pages, with intuitive examples.
- **Value** ⭐⭐⭐: Contributes to the theoretical understanding of LLM capability boundaries, though the depth of argumentation is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)

</div>

<!-- RELATED:END -->
