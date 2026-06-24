---
title: >-
  [Paper Note] Positional Attention: Expressivity and Learnability of Algorithmic Computation
description: >-
  [ICML2025][others (Transformer Theory][Positional Attention] Proposes **Positional Transformer**—a Transformer variant where attention weights are determined solely by positional encodings and are independent of input data. The authors prove that it maintains equivalent expressivity to the MPC parallel computation model (costing only $O(\log n)$ additional depth), and demonstrates significantly better out-of-distribution (OOD) generalization capability on algorithmic tasks.
tags:
  - "ICML2025"
  - "others (Transformer Theory"
  - "Algorithmic Reasoning)"
  - "Positional Attention"
  - "Transformer Expressivity"
  - "Algorithmic Execution"
  - "Parallel Computation Models"
  - "Generalization Bounds"
date: 2026-05-08
content_hash: eca5bf7ab58df00e
---

# Positional Attention: Expressivity and Learnability of Algorithmic Computation

**Conference**: ICML2025  
**arXiv**: [2410.01686](https://arxiv.org/abs/2410.01686)  
**Code**: To be confirmed  
**Area**: others (Transformer Theory / Algorithmic Reasoning)  
**Keywords**: Positional Attention, Transformer Expressivity, Algorithmic Execution, Parallel Computation Models, Generalization Bounds

## TL;DR

Proposes **Positional Transformer**—a Transformer variant where attention weights are determined solely by positional encodings and are independent of input data. The authors prove that it maintains equivalent expressivity to the MPC parallel computation model (costing only $O(\log n)$ additional depth), and demonstrates significantly better out-of-distribution (OOD) generalization capability on algorithmic tasks.

## Background & Motivation

Transformers have demonstrated powerful capabilities in executing algorithmic tasks (sorting, prefix sums, minimum value, etc.), and recent studies have established their equivalence with parallel computation models such as MPC.

A key observation is that in many parallel algorithms, the **communication patterns between processors depend solely on the processor indices (positions) and are independent of the processed data**. For instance, when using binary tree reduction to find the minimum value, the communication topology is a fixed binary tree structure.

Based on this, the authors raise a natural question: if Transformer attention weights are restricted to depend only on positional encodings, does the model still retain sufficient expressivity? Can this constraint bring about generalization advantages?

## Method

### Positional Transformer Architecture

The attention matrix of the $\ell$-th layer in a standard Transformer is computed from the input:

$$A^{(\ell,h)}(X) = \mathrm{softmax}\left((X W_Q^{(\ell,h)}) \cdot (X W_K^{(\ell,h)})^\top\right)$$

**Positional Transformer** decouples the attention weights, making them depend solely on positional encodings $P$:

$$A^{(\ell,h)} = \mathrm{softmax}\left((P W_Q^{(\ell,h)}) \cdot (P W_K^{(\ell,h)})^\top\right)$$

where $P \in \mathbb{R}^{n \times d_P}$ is a fixed positional encoding matrix that remains **invariant across layers**. The Value matrix is still computed from the output of the previous layer, meaning that the **information aggregation pattern** is fixed while the **aggregated content** is dynamically updated. The structure of each layer is:

$$F^{(\ell)}(X) = \Phi^{(\ell)}\left(\left(\bigoplus_{h=1}^{H} A^{(\ell,h)} X W_V^{(\ell,h)}\right) W_O^{(\ell)} \oplus X\right)$$

### Expressivity Results (Theorem 5.1)

**Core Theorem**: For an $R$-round MPC ($N$ machines, local memory $s$), there exists a Positional Transformer with $n = N+1$ nodes, $2R\lceil \log N \rceil$ layers, and $2s$ attention heads that can approximate this MPC instance with arbitrary accuracy.

The proof consists of two steps:

1. Introduce a proxy model **PCOC** (Parallel Computation with Oracle Communication), which permits communication only over a static network, utilizing a Beneš network to simulate the dynamic communication of MPC.
2. Prove that the Positional Transformer can simulate PCOC: the attention mechanism simulates communication between nodes, and the MLPs implement local computation through the universal approximation theorem.

**Corollary (Remark 5.1)**: Standard Transformers ($N$ nodes, $L$ layers) can be simulated by Positional Transformers ($O(N^2)$ nodes, $O(L \log N)$ layers). The extra logarithmic depth cost is an inherent penalty of the static communication restriction.

### Generalization Bounds (Theorem 6.1)

For a norm-bounded class $\mathcal{F}$ of $L$-layer, $H$-head Positional Transformers, with probability $\geq 1-\delta$ over $m$ samples, for all $f \in \mathcal{F}$:

$$|\mathsf{risk}(f) - \widehat{\mathsf{risk}}(f)| \leq \tilde{O}\left((H L_\sigma B_2)^{cL} B_{2,1} \sqrt{\frac{\log(Hdmn)}{m}} + \sqrt{\frac{\log(1/\delta)}{m}}\right)$$

**Key Advantage**: Compared to the generalization bounds of standard Transformers, this bound **does not contain** the $B_{QK}^{O(L)}$ term. Because standard Transformers have input-dependent attention weights, their generalization bounds exponentially depend on the spectral norm of the Query-Key matrices. Positional Transformer eliminates this factor.

**Trade-off**: In certain tasks, the Positional Transformer may require more layers (e.g., $O(\log n)$), which increases sample complexity through the $(H L_\sigma B_2)^{cL}$ factor. Whether this trade-off is beneficial depends on the specific task.

## Key Experimental Results

### Five Algorithmic Tasks

Cumulative Sum, Cumulative Min, Cumulative Median, Sorting, and Cumulative Max Subarray Sum.

### In-Distribution Performance (Figure 2)

- Input length $n=8$, training samples from 5K to 50K.
- **For both architectures, loss monotonically decreases as the number of samples increases across all tasks**, validating learnability.
- Positional Transformer achieves comparable in-distribution performance to the standard Transformer.

### k-hop Induction Head Task (Figure 3)

- Standard Transformer requires only 3 layers to approach near-zero loss.
- Positional Transformer requires more layers to match, **validating the theoretical prediction of an additional logarithmic depth cost**.
- This task inherently requires data-dependent dynamic communication, which does not align with the motivation of positional attention.

### Out-of-Distribution Generalization (Figure 4 — Core Results)

Training range $[-2, 2]$, with testing extended to $[-2c, 2c]$ ($c > 1$):

| Task | Standard Transformer OOD | Positional Transformer OOD |
|------|---------------------|---------------------------|
| Cumulative Sum | Loss rises sharply as $c$ increases | **Loss remains stable** |
| Cumulative Min | Significant degradation | **Almost unchanged** |
| Cumulative Median | Degradation | **Stable** |
| Sorting | Degradation | **Stable** |
| Max Subarray Sum | Degradation | **Stable** |

Positional Transformer **significantly outperforms standard Transformer in OOD generalization** across all five algorithmic tasks.

### k-hop Induction Head OOD (Figure 5)

- **Both architectures perform poorly.**
- This validates the hypothesis that this task fundamentally requires dynamic communication.

### Mixed-Type Input Experiments (Figure 6)

The input contains textual class labels and numerical values to test conditional reasoning + pattern matching + aggregation computation:

- Training numerical range is $[0, 5]$, testing extended to $[0, 5c]$.
- Positional Transformer **outperforms the standard Transformer in all three aggregation operations: min, sum, and multi-task(min+max)**.
- It even outperforms the OOD performance of a fine-tuned GPT-2.

## Highlights & Insights

- Empirical support for the **algorithm alignment hypothesis**: when the communication topology of the target algorithm is independent of the input data, the inductive bias of positional attention naturally aligns with the task, leading to better generalization.
- **A closed loop of theory and experiment**: expressivity equivalence $\rightarrow$ learnability guarantees $\rightarrow$ OOD benefits, argued progressively through three layers.
- **A simple yet profound architectural modification**: merely modifying the Q/K inputs from $X$ to $P$ eliminates the exponential reliance term in the generalization bound.
- Elegantly bridges the static communication constraints and the dynamic communication of MPC via the PCOC proxy model.

## Limitations & Future Work

- **Fixed positional encodings** restrict generalization to longer inputs (length generalization remains an open question).
- The OOD theoretical analysis is not fine-grained enough; current OOD generalization bounds are insufficient to characterize the differences between the two architectures.
- Positional Transformer performs poorly on tasks that require **dynamic communication** (such as induction heads).
- Simulating a standard Transformer requires a quadratic overhead of $O(N^2)$ nodes (more efficient strategies might exist).
- The experiments only cover numerical and simple text algorithmic tasks, leaving more complex reasoning (e.g., graph algorithms, program execution) unexplored.

## Related Work & Insights

- **Sanford et al., 2024**: Proved that standard Transformers can simulate MPC (this work builds directly on top of it).
- **Edelman et al., 2022**: Generalization bounds for standard Transformers (this work improves the dependency on the QK norm).
- **Engelmayer et al., 2023**: Supervised training using intermediate labels from parallel algorithms.
- **Kazemnejad et al., 2023**: Investigated the impact of different positional encodings on tasks.
- Insight: The concept of positional attention can be generalized to the Graph Neural Network (GNN) domain—when the message-passing topology is fixed, graph attention weights can depend purely on structural positions.

## Rating

- Novelty: ⭐⭐⭐⭐ (The concept of positional attention is simple yet insightful, and the PCOC proxy model is ingenious)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ID/OOD coverage across 5 algorithmic tasks + induction head + mixed inputs)
- Writing Quality: ⭐⭐⭐⭐⭐ (A trinity of theory, experiment, and intuition; Figure 1 is extremely clear)
- Value: ⭐⭐⭐⭐ (Provides an important theoretical framework for understanding the roles of position and content in attention mechanisms)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](../../ICLR2026/learning_theory/online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/learning_theory/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICLR 2026\] Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets](../../ICLR2026/learning_theory/algorithmic_guarantees_for_distilling_supervised_and_offline_rl_datasets.md)
- [\[ICLR 2026\] Poly-attention: a general scheme for higher-order self-attention](../../ICLR2026/learning_theory/poly-attention_a_general_scheme_for_higher-order_self-attention.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](../../ICLR2026/learning_theory/on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)

</div>

<!-- RELATED:END -->
