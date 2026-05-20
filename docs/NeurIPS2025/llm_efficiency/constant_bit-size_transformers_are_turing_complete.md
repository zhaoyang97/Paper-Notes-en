---
title: >-
  [Paper Note] Constant Bit-Size Transformers Are Turing Complete
description: >-
  [NeurIPS 2025][LLM Efficiency][Turing completeness] This paper provides the first proof that a Transformer with constant bit-size precision and a fixed number of parameters — permitting only context window growth — is Tu…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Turing completeness"
  - "transformer theory"
  - "computational complexity"
  - "Post machines"
  - "context window"
date: 2026-05-08
content_hash: fc516acff5246a70
---

# Constant Bit-Size Transformers Are Turing Complete

**Conference**: NeurIPS 2025
**arXiv**: [2506.12027](https://arxiv.org/abs/2506.12027)  
**Code**: None  
**Area**: LLM Efficiency
**Keywords**: Turing completeness, transformer theory, computational complexity, Post machines, context window

## TL;DR
This paper provides the first proof that a Transformer with constant bit-size precision and a fixed number of parameters — permitting only context window growth — is Turing complete. It establishes the exact complexity equivalence WINDOW[s(n)] = SPACE[s(n)], demonstrating that expanding the context window, rather than model size, suffices for universal computation.

## Background & Motivation
**Background**: CoT-based Transformers have been shown to be Turing complete; however, all prior proofs require the model's bit-size to grow with input length — either precision must scale as $O(\log t(n))$ bits, or the embedding dimension must scale as $O(\log t(n))$.

**Limitations of Prior Work**: Bit-size growing with input length implies that processing longer inputs requires larger models, posing a principled obstacle for AGI systems pursuing lifelong learning, where input length grows unboundedly as the system interacts with an ever-expanding environment.

**Key Challenge**: Is scaling model precision or parameters a necessary condition for handling longer inputs?

**Goal**: To prove that model bit-size need not grow, and that expanding the context window alone suffices to simulate arbitrary Turing machines.

**Key Insight**: The behavior of a Post Machine — a queue-based Turing-equivalent model — exhibits a natural structural correspondence with Transformer autoregressive decoding, wherein the context window can be viewed as a queue.

**Core Idea**: Turing completeness is achieved by constructing a constant-precision single-layer Transformer that simulates a Post Machine (queue automaton) rather than a conventional Turing machine.

## Method

### Overall Architecture
The proof proceeds in two directions: (1) WINDOW[s(n)] ⊆ SPACE[s(n)]: a Turing machine (TM) can simulate a Transformer with context window s(n) using O(s(n)) space (by direct construction, maintaining a buffer of size s(n)); (2) SPACE[s(n)] ⊆ WINDOW[s(n)]: simulation proceeds stepwise as TM → Post Machine → Transformer.

### Key Designs

1. **Post Machine as an Intermediate Bridge**:

    - **Function**: The TM is first converted into an equivalent Post Machine (PM), which is a finite automaton equipped with a queue.
    - **Mechanism**: The PM's queue stores the TM tape in a cyclic manner — a right head move deletes from the front and appends to the rear; a left head move cyclically rotates the queue. A "delayed append" technique avoids the need to pre-read the next queue element.
    - **Design Motivation**: The PM's operations (reading from the queue front and writing to the rear) naturally correspond to Transformer autoregressive decoding (old tokens sliding out of the window, new tokens appended), making the simulation construction clean.

2. **Constant-Precision Single-Layer Transformer Construction**:

    - **Function**: A single-layer, single-head, hardmax attention Transformer is constructed, using a context window of size s(n) as the PM's queue.
    - **Mechanism**: The vocabulary is $\mathcal{V} = \Sigma \times Q$, encoding both tape symbol and state in each token; relative positional encodings distinguish only three position types (current position, oldest window position, intermediate positions); hardmax attention precisely retrieves the oldest token in the window (queue front); the FFN implements the PM transition function $\delta$.
    - **Design Motivation**: Hardmax attention combined with positional encoding reduces attention to a precise "retrieve oldest token" lookup, while the FFN serves as a finite-state lookup table for transitions — the entire construction requires only constant precision and constant parameters.

3. **Fixed Queue Size**:

    - **Function**: The PM is adapted so that queue size always equals s(n), ensuring a fixed context window size.
    - **Mechanism**: The input is padded on the right with # symbols to length s(n), with both heads advancing one position per step.
    - **Design Motivation**: Ensures that the Transformer's context window size precisely matches the PM's queue size.

### Complexity Analysis
- Simulating a TM with s(n) space requires a context window of size s(n).
- Each TM step requires O(s(n)) CoT tokens (since one PM step simulating one TM step traverses the entire queue).
- Total CoT length: $O(t(n) \cdot s(n))$.

## Key Experimental Results

### Comparison of Theoretical Results

| Work | Precision | Embedding Dim. | Window Size | CoT per TM Step |
|------|-----------|----------------|-------------|-----------------|
| Pérez et al. (2021) | $O(\log t(n))$ | $O(1)$ | $n+t(n)$ | 1 |
| Li et al. (2024) | $O(1)$ | $O(\log t(n))$ | $O(t(n)\log t(n))$ | $O(\log t(n))$ |
| **Ours** | **$O(1)$** | **$O(1)$** | **$s(n)$** | **$s(n)$** |

### Key Corollaries

| Corollary | Description |
|-----------|-------------|
| WINDOW[poly(n)] = PSPACE | A polynomial window suffices to solve all PSPACE problems |
| Window vs. Time | The window need only grow with space complexity, not time complexity |
| Universal Simulation | A single fixed Transformer loaded with a universal TM description can compute any computable function |

### Key Findings
- **Window Equals Space Complexity**: The central result — the computational role of the context window is equivalent to the space resource of a TM.
- **Large Gap Between Space and Time**: Many PSPACE-complete problems (SAT, Sokoban) require only polynomial space but potentially exponential time, implying that window requirements are far smaller than those of prior constructions.
- **Uniformity of Relative Positional Encoding**: When simulating inputs of different lengths, only the relative positional encoding need be adjusted according to an explicit formula; all other parameters remain fixed.

## Highlights & Insights
- **Elegant Choice of Post Machine**: Prior constructions directly simulating TM tapes require log-precision to encode tape positions; the PM's queue structure encodes positional information implicitly through window ordering, eliminating the need for additional precision.
- **A New Interpretation of Attention as Queue Operations**: The attention mechanism is not merely a statistical aggregator; it can also function as a discrete computational operation over a queue structure — offering a new perspective on understanding the reasoning capabilities of Transformers.
- **Theoretical Justification for "Expanding the Context Window"**: The engineering community has already been moving toward longer context windows (100K+ tokens); this paper provides a computational-theoretic justification for this direction.

## Limitations & Future Work
- **Low Simulation Efficiency**: Each TM step requires O(s(n)) CoT tokens, making the total simulation time $O(t(n) \cdot s(n))$ — significantly slower than directly running the TM.
- **Purely Existential Proof**: The constructed Transformer parameters are hand-crafted and do not involve learning — there is no guarantee that a trained Transformer will learn analogous computational patterns.
- **Hardmax Assumption**: Hardmax is used in place of softmax; while a standard theoretical simplification, this deviates from practice.
- **Finite-Precision Floating Point Not Addressed**: Practical Transformers use fp16/bf16, which does not fully align with the theoretical notion of "constant precision."

## Related Work & Insights
- **vs. Pérez et al. (2021)**: Their construction requires $O(\log t(n))$ precision; the present work replaces this with constant precision at the cost of more CoT steps per TM step.
- **vs. Li et al. (2024)**: They achieve constant precision but require $O(\log t(n))$-dimensional embeddings, so the actual bit-size continues to grow.
- **vs. Yang et al. (2025) PENCIL**: PENCIL also relates window size to space complexity, but still requires growing bit-size; the present work achieves truly constant bit-size.
- **vs. Schuurmans et al. (2024)**: They also employ a Post Machine approach, but analyze the more restricted Lag system and require $O(s(n)^3)$ CoT per step, compared to $O(s(n))$ in the present work.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First proof of Turing completeness under constant bit-size; the WINDOW = SPACE equivalence is highly elegant.
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical work; no experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Proof structure is clear; motivation and remarks effectively bridge theoretical and practical significance.
- **Value**: ⭐⭐⭐⭐⭐ A milestone contribution to the computational theory of Transformers with direct implications for architectural design philosophy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[NeurIPS 2025\] From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)
- [\[NeurIPS 2025\] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](../../ICLR2026/llm_efficiency/fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)

</div>

<!-- RELATED:END -->
