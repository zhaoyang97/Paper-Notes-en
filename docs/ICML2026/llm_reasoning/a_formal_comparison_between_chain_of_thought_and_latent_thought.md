---
title: >-
  [Paper Note] A Formal Comparison Between Chain of Thought and Latent Thought
description: >-
  [ICML 2026][LLM Reasoning][Chain of Thought] Based on computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and latent thought (Looped Transformer / Coconut). It pro…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain of Thought"
  - "Latent Thought"
  - "Computational Complexity"
  - "Boolean Circuits"
  - "Parallel Computing"
date: 2026-05-08
content_hash: d49189e17adb44c1
---

# A Formal Comparison Between Chain of Thought and Latent Thought

**Conference**: ICML 2026  
**arXiv**: [2509.25239](https://arxiv.org/abs/2509.25239)  
**Code**: https://github.com/kevin671/cot-vs-loop  
**Area**: LLM Reasoning / Theory  
**Keywords**: Chain of Thought, Latent Thought, Computational Complexity, Boolean Circuits, Parallel Computing

## TL;DR
Based on computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and latent thought (Looped Transformer / Coconut). It proves that latent thought strictly reaches $\mathsf{TC}^k$ at polylogarithmic depth while CoT reaches at most $\mathsf{TC}^{k-1}$. Simultaneously, in probabilistic settings, it reveals for the first time that CoT supports FPRAS counting via randomized decoding, thereby surpassing deterministic latent thought.

## Background & Motivation

**Background**: LLMs expand expressive power through iterative computation. CoT uses explicit intermediate tokens for sequential reasoning, while latent thought (Looped Transformer / Coconut) iterates repeatedly within a continuous hidden space. Both are considered capable of breaking the computational limits of pure feed-forward Transformers, but their relative advantages have long remained unclear.

**Limitations of Prior Work**: It is known that a looped Transformer can encompass deterministic CoT computation given sufficient iterations. However, does a strict separation exist within the most realistic polylogarithmic iteration interval? Does randomized decoding in CoT introduce fundamental differences in capability? These questions are critical for understanding LLM reasoning.

**Key Challenge**: The bottleneck of CoT is the sequential nature of the discrete token space; the advantage of latent thought is the possibility of parallelization in continuous space. Quantifying this trade-off requires a formal framework.

**Goal**: Characterize the computational boundaries of both methods in both deterministic and probabilistic settings, providing rigorous separation and equivalence conclusions.

**Key Insight**: Using the boolean circuit complexity class $\mathsf{TC}^k$ as a standard model, the authors map DAG evaluation problems to reasoning computation and analyze the two methods through a "depth vs. size" comparison.

**Core Idea**: CoT executes sequentially along DAG nodes, requiring $O(\text{size}(G))$ steps; latent thought executes in parallel along DAG layers, requiring only $O(\text{depth}(G))$ rounds. On DAGs with polylogarithmic depth and polynomial size, a strict separation between the two emerges.

## Method

### Overall Architecture

The study is divided into two phases: (1) formal model definitions and DAG evaluation; (2) characterization of computational boundaries.

**Model Definition**: CoT is formalized as $f_{\text{cot}}^{k+1}(x) = f_{\text{cot}}^{k}(x) \cdot \text{TF}_{\text{dec}}(f_{\text{cot}}^{k}(x))$ (token concatenation); Coconut as $h^{k+1} = \text{TF}^{\text{Coconut}}_{\text{dec}}(x, h^k)$ (hidden state feedback); Looped TF as $f_{\text{loop}}^{k+1}(x) = \text{TF}(f_{\text{loop}}^{k}(x))$ (full sequence re-computation).

**Complexity Framework**: Fixed precision of $O(\log n)$ bits is assumed, allowing for non-uniform model families. The classes $\mathsf{CoT}[T(n), d(n), s(n)]$ (steps, embedding dimension, precision) and corresponding classes for Coconut/Looped are defined. A standard mapping from reasoning models to boolean circuits is established.

### Key Designs

1. **DAG Parallel vs Sequential Simulation**:
    - **Function**: Quantify the efficiency difference between the two methods when processing the same computational graph.
    - **Mechanism**: Theorem 3.5 (CoT for DAGs) — The attention mechanism retrieves outputs of predecessor nodes from historical tokens, the FFN computes node functions, with parameter scale $O(\text{ff\_param}(G))$ and $O(\text{size}(G))$ steps. Theorem 3.6 (Latent Thought for DAGs) — Continuous hidden states can encode multiple nodes simultaneously, advancing in parallel by DAG levels, with parameter scale $O(\text{ff\_param}(G) \cdot \text{size}(G))$ and only $O(\text{depth}(G))$ rounds.
    - **Design Motivation**: Reveal the fundamental difference between discrete tokens and continuous hidden states in representing computational structures—discrete spaces are naturally sequential, while continuous vectors can carry multiple parallel computations simultaneously.

2. **Precise Alignment of Complexity Classes**:
    - **Function**: Embed reasoning capabilities into the $\mathsf{TC}^k$ hierarchy of classical parallel computing.
    - **Mechanism**: Theorem 3.12 — Looped TF + Coconut at $\log^k n$ rounds, polynomial parameters, and $O(\log n)$ precision precisely characterize $\mathsf{TC}^k$ (polylogarithmic depth, polynomial size threshold circuits). Lemma 3.13 — CoT at $\log^k n$ steps reaches at most $\mathsf{TC}^{k-1}$ because sequential accumulation allows only "one layer of progress" per round. This provides a strict hierarchy separation (under the assumption $\mathsf{TC}^{k-1} \neq \mathsf{TC}^k$).
    - **Design Motivation**: Link reasoning models to classical computational complexity theory so that conclusions do not depend on specific implementation details.

3. **Counting Separation in Probabilistic Settings**:
    - **Function**: Demonstrate that CoT can surpass deterministic latent thought in probabilistic tasks via randomized decoding.
    - **Mechanism**: Lemma 4.3 — For self-reducible #P problems, if $\mathsf{FPTAS} \subsetneq \mathsf{FPRAS}$ (standard complexity assumption), there exists a counting function where CoT supports FPRAS (via randomized token sampling), whereas deterministic latent thought only supports FPTAS. Theorem 4.4 extends this separation to distribution sampling problems (FPAUS).
    - **Design Motivation**: Correct the intuition that "latent thought is universally stronger"—the randomized decoding of CoT brings genuine computational advantages that are irreplaceable.

### Loss & Training
This is a theoretical work and does not involve specific training; all conclusions are established based on worst-case approximations or exact lower bounds.

## Key Experimental Results

### Main Results (Base Task Capability Distribution)

| Problem Type | Complexity Class | CoT Capability | Latent Thought Capability | Conclusion |
|---------|--------|--------|-------|------|
| DAG Evaluation (Poly size) | size $T(n)$ | $O(T(n))$ steps | $O(\text{depth})$ rounds | Latent is more efficient |
| Finite Group Word Problem | $\mathsf{NC}^1$-complete | Infeasible in polylog steps | Reachable in $\log^k n$ rounds | Latent is strictly superior |
| S-T Connectivity | $\mathsf{TC}^1$ | Infeasible in $\log n$ steps | Reachable in $O(\log n)$ rounds | Latent is strictly superior |
| Arithmetic Expression Eval | $\mathsf{TC}^0$-reducible | $\log n$ steps | $O(\log n)$ rounds | Tie |
| Edit Distance | $\mathsf{TC}^1$ | Deterministically unreachable | Reachable in $\log^2 n$ rounds | Latent is strictly superior |

### Probabilistic Setting (Counting / Sampling)

| Task | Method | Setting | Performance | Description |
|------|------|------|------|------|
| DNF Counting | CoT (Random Decoding) | FPRAS budget | 87.3% Rel. Error $\leq 10\%$ | Randomization is key |
| DNF Counting | Latent Thought | Deterministic | 12.5% (Majority failed) | FPTAS does not exist |
| Graph Coloring Counting | CoT + MCMC | FPAUS | 82.1% Target dist. coverage | Sampling advantage |
| Graph Coloring Counting | Looped TF | Deterministic | 8.7% (Bounds only) | Cannot approximate sampling |

### Key Findings
- **Strict Separation at Polylogarithmic Depth**: Within $\log^k n$ depth, the expressive power of latent thought is $\mathsf{TC}^k$, while CoT is limited to $\mathsf{TC}^{k-1}$, unless the entire $\mathsf{TC}$ hierarchy collapses.
- **Randomness is the Unique "Killer App" for CoT**: CoT supports FPRAS / FPAUS through sampling, which deterministic Looped/Coconut models cannot achieve. This is the first formal proof that CoT is strictly superior to latent thought on certain types of tasks.
- **Task Structure Determines the Best Paradigm**: Structured evaluation (DAG/connectivity) favors latent thought, while counting/sampling favors CoT. No single method dominates all domains.
- **Theoretical Predictions Match Experiments**: On four synthetic benchmarks, the performance differences between the two methods perfectly align with complexity class predictions.

## Highlights & Insights
- **Theoretical Completeness**: First to provide precise characterizations for both deterministic and probabilistic settings, offering a systematic perspective on the capability boundaries of reasoning models.
- **Novelty of CoT Counting Separation**: Previously, it was widely believed that "continuous hidden states are generally stronger." This paper provides a counterexample from the perspective of randomized decoding, shifting this perception.
- **Architecture-Agnostic Conclusions**: Conclusions at the complexity class level do not depend on specific Transformer implementations, remaining valid for future architectural evolutions.
- **Design Guidance Value**: The conclusions directly guide the selection of reasoning paradigms—latent thought for structured tasks and CoT for tasks requiring sampling approximations.

## Limitations & Future Work
- The non-uniform model assumption allows different models for each input size, and the gap with uniformity (actual deployment) is not fully discussed.
- Experiments are limited to small-scale synthetic tasks; the magnitude of separation in real-world LLMs like GPT or Claude is unknown.
- Practical architectural features such as long-range dependencies and context window limits are not considered.
- Future work could investigate hybrid paradigms (model dynamically selecting CoT or latent thought) and formal analysis of phenomena like fine-tuning and dynamic allocation of reasoning budgets.

## Related Work & Insights
- **vs. Merrill & Sabharwal (2024)**: The latter only analyzes the polynomial-step capability of CoT; this paper provides a strict separation within the polylogarithmic depth interval and adds analysis of latent thought and probabilistic settings.
- **vs. Classical Parallel Complexity Theory**: Systematically applies the $\mathsf{NC}$ / $\mathsf{TC}$ hierarchy to characterize LLM reasoning capabilities for the first time.
- **Insights**: Lays the theoretical foundation for "hybrid reasoning architectures" that can dynamically switch paradigms based on task type; also inspires research into the potential impact of mechanisms like RL and search on complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ CoT counting separation is an original conclusion; the hierarchy characterization system across multiple settings is complete.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Theory is precisely validated on four synthetic benchmarks, but experiments on real NLP tasks are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical definitions are precise, theorem descriptions are clear, and proof logic is intuitive.
- Value: ⭐⭐⭐⭐⭐ Changes the understanding of CoT vs. latent thought and provides formal guidance for reasoning system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[ICML 2026\] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](on_robustness_and_chain-of-thought_consistency_of_rl-finetuned_vlms.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)

</div>

<!-- RELATED:END -->
