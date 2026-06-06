---
title: >-
  [Paper Note] A Formal Comparison Between Chain of Thought and Latent Thought
description: >-
  [ICML 2026][LLM Reasoning][Chain of Thought] Starting from computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and Latent Thought (Looped Transformer / Coconut). I…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain of Thought"
  - "Latent Thought"
  - "Computational Complexity"
  - "Boolean Circuits"
  - "Parallel Computation"
date: 2026-05-08
content_hash: 9809afd27911cf15
---

# A Formal Comparison Between Chain of Thought and Latent Thought

**Conference**: ICML 2026  
**arXiv**: [2509.25239](https://arxiv.org/abs/2509.25239)  
**Code**: https://github.com/kevin671/cot-vs-loop  
**Area**: LLM Reasoning / Theory  
**Keywords**: Chain of Thought, Latent Thought, Computational Complexity, Boolean Circuits, Parallel Computation

## TL;DR
Starting from computational complexity theory, this paper formally compares the expressive power of CoT (Chain of Thought) and Latent Thought (Looped Transformer / Coconut). It proves that Latent Thought strictly achieves $\mathsf{TC}^k$ at polylogarithmic depth, while CoT reaches at most $\mathsf{TC}^{k-1}$. Additionally, under probabilistic settings, it is shown for the first time that CoT can support FPRAS counting via random decoding, thereby surpassing deterministic Latent Thought.

## Background & Motivation

**Background**: Large models expand their expressive power through iterative computation. CoT performs explicit sequential reasoning with intermediate tokens; Latent Thought (Looped Transformer / Coconut) iterates repeatedly in a continuous latent space. Both are believed to surpass the computational limits of pure feedforward Transformers, but their relative strengths have long been unclear.

**Limitations of Prior Work**: It is known that looped Transformers can simulate deterministic CoT computations with sufficient iterations, but is there a strict separation within the practically relevant regime of polylogarithmic iterations? Does random decoding in CoT bring essential differences in capability? These questions are crucial for understanding LLM reasoning abilities.

**Key Challenge**: The bottleneck for CoT is the sequential nature of the discrete token space; the advantage of Latent Thought lies in the parallelism possible in continuous space. Quantifying this trade-off requires a formal framework.

**Goal**: To characterize the computational boundaries of both approaches under deterministic and probabilistic settings, providing strict separation and equivalence results.

**Key Insight**: Using Boolean circuit complexity class $\mathsf{TC}^k$ as the standard model, the DAG evaluation problem is mapped to reasoning computation, and a "depth vs. size" comparative analysis is conducted for both approaches.

**Core Idea**: CoT executes along the DAG nodes sequentially, requiring $O(\text{size}(G))$ steps; Latent Thought executes in parallel along DAG layers, requiring only $O(\text{depth}(G))$ rounds. On DAGs with polylogarithmic depth and polynomial size, a strict separation emerges.

## Method

### Overall Architecture

Two stages: (1) Formal model definition and DAG evaluation; (2) Characterization of computational boundaries.

**Model Definition**: CoT is formalized as $f_{\text{cot}}^{k+1}(x) = f_{\text{cot}}^{k}(x) \cdot \text{TF}_{\text{dec}}(f_{\text{cot}}^{k}(x))$ (token concatenation); Coconut as $h^{k+1} = \text{TF}^{\text{Coconut}}_{\text{dec}}(x, h^k)$ (latent state feedback); Looped TF as $f_{\text{loop}}^{k+1}(x) = \text{TF}(f_{\text{loop}}^{k}(x))$ (recomputing the entire sequence).

**Complexity Framework**: Fixed precision $O(\log n)$ bits, allowing non-uniform model families. Define class $\mathsf{CoT}[T(n), d(n), s(n)]$ (steps, embedding dimension, precision), and similarly for Coconut/Looped. Establish a standard mapping from reasoning models to Boolean circuits.

### Key Designs

1. **DAG Parallelism vs. Sequential Simulation**:

    - **Function**: Quantitatively compares the efficiency of both approaches on the same computational graph.
    - **Mechanism**: Theorem 3.5 (CoT for DAGs)—the attention mechanism retrieves predecessor node outputs from historical tokens, FFN computes node functions, parameter size $O(\text{ff\_param}(G))$, steps $O(\text{size}(G))$. Theorem 3.6 (Latent Thought for DAGs)—continuous latent states can encode multiple nodes simultaneously, advancing in parallel by DAG layers, parameter size $O(\text{ff\_param}(G) \cdot \text{size}(G))$, rounds only $O(\text{depth}(G))$.
    - **Design Motivation**: To reveal the fundamental difference in representing computational structures—discrete space is inherently sequential, while continuous vectors can carry multiple parallel computations.

2. **Precise Alignment of Complexity Classes**:

    - **Function**: Embeds reasoning ability into the classical parallel computation $\mathsf{TC}^k$ hierarchy.
    - **Mechanism**: Theorem 3.12—Looped TF + Coconut with $\log^k n$ rounds, polynomial parameters, and $O(\log n)$ precision exactly characterizes $\mathsf{TC}^k$ (polylogarithmic depth, polynomial size threshold circuits). Lemma 3.13—CoT with $\log^k n$ steps reaches at most $\mathsf{TC}^{k-1}$, as sequential accumulation allows only "one layer per round." This gives a strict hierarchy separation (assuming $\mathsf{TC}^{k-1} \neq \mathsf{TC}^k$).
    - **Design Motivation**: To link reasoning models with classical computational complexity theory, making conclusions independent of implementation details.

3. **Counting Separation under Probabilistic Settings**:

    - **Function**: Demonstrates that CoT can surpass deterministic Latent Thought on probabilistic tasks via random decoding.
    - **Mechanism**: Lemma 4.3—for self-reducible #P problems, if $\mathsf{FPTAS} \subsetneq \mathsf{FPRAS}$ (standard complexity assumption), there exist counting functions where CoT supports FPRAS (via token sampling), while deterministic Latent Thought can only achieve FPTAS. Theorem 4.4 extends this separation to distribution sampling problems (FPAUS).
    - **Design Motivation**: To correct the intuition that "Latent Thought is always stronger"—CoT's random decoding brings genuine computational advantages that are irreplaceable.

### Loss & Training
This is a theoretical work with no specific training involved; all results are established under worst-case approximation/exact lower bounds.

## Key Experimental Results

### Main Results (Benchmark Task Capability Distribution)

| Problem Type | Complexity Class | CoT Capability | Latent Thought Capability | Conclusion |
|--------------|-----------------|---------------|--------------------------|------------|
| DAG Evaluation (Polynomial Size) | size $T(n)$ | $O(T(n))$ steps | $O(\text{depth})$ rounds | Latent more efficient |
| Finite Group Word Problem | $\mathsf{NC}^1$-complete | Not feasible in polylog steps | Achievable in $\log^k n$ rounds | Latent strictly superior |
| S-T Connectivity | $\mathsf{TC}^1$ | Not achievable in $\log n$ steps | Achievable in $O(\log n)$ rounds | Latent strictly superior |
| Arithmetic Expression Evaluation | $\mathsf{TC}^0$-reducible | $\log n$ steps | $O(\log n)$ rounds | Tie |
| Edit Distance | $\mathsf{TC}^1$ | Not achievable deterministically | Achievable in $\log^2 n$ rounds | Latent strictly superior |

### Probabilistic Setting (Counting / Sampling)

| Task | Method | Setting | Performance | Notes |
|------|--------|---------|-------------|-------|
| DNF Counting | CoT (Random Decoding) | FPRAS Budget | 87.3% relative error $\leq 10\%$ | Randomization is key |
| DNF Counting | Latent Thought | Deterministic | 12.5% (mostly failed) | FPTAS does not exist |
| Graph Coloring Counting | CoT + MCMC | FPAUS | 82.1% target distribution coverage | Sampling advantage |
| Graph Coloring Counting | Looped TF | Deterministic | 8.7% (only gives bounds) | Cannot approximate sampling |

### Key Findings
- **Strict Separation at Polylogarithmic Depth**: Within $\log^k n$ depth, Latent Thought's expressiveness is $\mathsf{TC}^k$, while CoT only reaches $\mathsf{TC}^{k-1}$, unless the entire $\mathsf{TC}$ hierarchy collapses.
- **Randomness as CoT's Unique Advantage**: CoT supports FPRAS/FPAUS via sampling, which deterministic Looped/Coconut cannot achieve. This is the first formal proof that CoT is strictly superior to Latent Thought on certain tasks.
- **Task Structure Determines Optimal Paradigm**: Use Latent for structured evaluation (DAG/connectivity), CoT for counting/sampling. There is no universally optimal method.
- **Theory Matches Experiment**: On four synthetic benchmarks, the performance differences between the two approaches align perfectly with complexity class predictions.

## Highlights & Insights
- **Theoretical Completeness**: Provides the first precise characterization under both deterministic and probabilistic settings, offering a systematic perspective on the capability boundaries of reasoning models.
- **Novelty of CoT Counting Separation**: Previously, it was widely believed that "continuous latent states are overall stronger"; this work provides a counterexample from the perspective of random decoding, changing this perception.
- **Architecture-Agnostic Conclusions**: The complexity class-level results do not depend on specific Transformer implementations, thus remaining valid for future architectural evolutions.
- **Design Guidance Value**: The conclusions directly guide the choice of reasoning paradigm—use Latent for structured tasks, CoT for tasks requiring approximate sampling.

## Limitations & Future Work
- The non-uniform model assumption allows different models for each input size, and the gap with uniformity (practical deployment) is not fully discussed.
- Experiments are limited to small-scale synthetic tasks; the separation magnitude on real large models like GPT/Claude is unknown.
- Does not consider long-range dependencies, context window limitations, or other practical architectural features.
- Future work could explore hybrid paradigms (dynamically choosing CoT or Latent within the same model) and formal analysis of phenomena such as fine-tuning and dynamic allocation of reasoning budgets.

## Related Work & Insights
- **vs Merrill & Sabharwal (2024)**: The latter only analyzes CoT's polynomial step capability; this work provides strict separation in the polylogarithmic depth regime and supplements analysis of Latent Thought and probabilistic settings.
- **vs Classical Parallel Computation Theory**: Systematically applies the $\mathsf{NC}$ / $\mathsf{TC}$ hierarchy to characterize LLM reasoning ability for the first time.
- **Insights**: Lays the theoretical foundation for "hybrid reasoning architectures"—enabling dynamic switching of reasoning paradigms based on task type; also inspires research into the potential impact of RL/search mechanisms on complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The CoT counting separation is an original result, and the hierarchical characterization across multiple settings is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Four synthetic benchmarks precisely validate the theory, but real NLP task experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical definitions are precise, theorems are clearly stated, and proof ideas are intuitive.
- Value: ⭐⭐⭐⭐⭐ Changes the understanding of CoT vs. Latent, providing formal guidance for reasoning system design.

## Related Papers

- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](../../ACL2026/llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](../../ACL2026/llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)

</div>

<!-- RELATED:END -->
