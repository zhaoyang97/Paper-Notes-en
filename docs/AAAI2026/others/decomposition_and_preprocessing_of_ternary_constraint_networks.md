---
title: >-
  [Paper Note] Decomposition and Preprocessing of Ternary Constraint Networks
description: >-
  [AAAI 2026][Constraint Programming] This paper proposes a complete theoretical framework for formally decomposing arbitrary discrete constraint networks into ternary constraint networks (TCNs)…
tags:
  - "AAAI 2026"
  - "Constraint Programming"
  - "Ternary Constraint Networks"
  - "GPU Solving"
  - "Preprocessing"
  - "MiniZinc"
date: 2026-05-08
content_hash: 38aeb2520aae93e4
---

# Decomposition and Preprocessing of Ternary Constraint Networks

**Conference**: AAAI 2026
**arXiv**: [2511.11872](https://arxiv.org/abs/2511.11872)  
**Code**: [Turbo solver](https://github.com/ptal/turbo)  
**Area**: Other
**Keywords**: Constraint Programming, Ternary Constraint Networks, GPU Solving, Preprocessing, MiniZinc

## TL;DR
This paper proposes a complete theoretical framework for formally decomposing arbitrary discrete constraint networks into ternary constraint networks (TCNs), and reduces the variable/constraint blowup introduced by decomposition from a median of 8x/6x to 4.8x/4.3x via seven preprocessing techniques (propagation, algebraic simplification, common subexpression elimination, etc.), providing a regularized data layout for efficient constraint solving on GPU hardware.

## Background & Motivation

Constraint Programming (CP) is a general-purpose exact solving method based on constraint propagation and backtracking search. Modern solvers (Choco, OR-Tools) implement propagators via interpreter-style traversal of constraint ASTs (view-based propagation), handling diverse constraint forms through subtype or parametric polymorphism.

**Core Challenges**:
- **Irregular data layout**: Varying arities and operators across constraints cause severe thread divergence during GPU execution
- **Complex propagator implementation**: It is infeasible to implement distinct propagator functions for every possible constraint form
- **GPU parallelism bottleneck**: Constraint solving inherently requires sequential backtracking search, which conflicts with the SIMT execution model given irregular constraint structures

**Proposed Approach**: Uniformly decompose all constraints into ternary constraints of the form $x = y \odot z$, where $\odot \in \{+, *, /, mod, min, max, =, \leq\}$, yielding a regular data layout suitable for GPU execution.

## Method

### Overall Architecture
Given an arbitrary constraint network $\langle d, C \rangle$, processing proceeds in two stages:
1. **TCN Decomposition** (Section 3): A recursive function $tcn(d, C) = \langle d', C' \rangle$ rewrites each constraint into ternary form
2. **Preprocessing** (Section 4): Seven optimization techniques reduce the size of the decomposed network

### Key Designs

#### Module 1: TCN Decomposition Function $tc(d, t)$

The core recursive function $tc(d, t) = (d', T, x)$ rewrites a term or constraint $t$ into a set of ternary constraints $T$:

- **Base cases**: Variables are returned directly; constants create shared `__CONSTANT_k` variables
- **Unary constraints**: Negation $-t \to 0-t$; absolute value $abs(t) \to max(t, 0-t)$; set membership $t \in S$ is converted to an interval disjunction
- **Binary operations**: $t_1 \odot t_2$ introduces an auxiliary variable $x = y \odot z$; subtraction exploits relational semantics $x = y-z \Leftrightarrow y = x+z$
- **Logical connectives**: $\land \to min$, $\lor \to max$, $\Leftrightarrow \to =$; non-Boolean variables are mapped to $[0,1]$ via a `booleanize` function

The decomposition guarantees solution-set equivalence (Proposition 3.2) and uniqueness (Proposition 3.4).

#### Module 2: Seven Preprocessing Techniques

Preprocessing computes the greatest fixpoint over the triple $\langle d, C, E \rangle$:

| Technique | Purpose | Key Mechanism |
|-----------|---------|---------------|
| Constraint Propagation | Reduce variable domains | Compute greatest fixpoint of propagators |
| Algebraic Simplification (AS) | Eliminate constraints directly expressible via domains | Pattern matching: $x=y+0 \to merge(E,x,y)$, $x=y*1 \to merge(E,x,y)$, 25+ rules |
| Identical Common Subexpression Elimination (ICSE) | Merge constraints with identical right-hand sides | Hash map $y \odot z \to x$, $O(\|C\|)$ complexity |
| Equivalence Class Domain Merging | Synchronize domains of equivalent variables | $d_E(x) = \bigcap_{y \in [x]_E} d(y)$ |
| Implied Constraint Elimination | Remove constraints implied by current domains | Check $\gamma(d) \subseteq rel(c)$ |
| Variable Renaming | Assign a unique representative to each equivalence class | Union-Find data structure |
| Useless Variable Elimination | Remove variables not appearing in any constraint scope | Empty-domain variables retained to detect unsatisfiability |

#### Module 3: Equivalence Partitioning and Fixpoint Computation

Variable equivalence partitioning $E$ is implemented via Union-Find, with equivalence classes merged through $merge(E, x, y)$. Preprocessing iterates until neither $d$ nor $E$ changes. Implied constraint checking and useless variable elimination are extracted outside the fixpoint loop (guaranteed by monotonicity).

### Loss & Training
This is a theoretical/systems paper with no neural network loss functions. The optimization objective is to minimize the size (number of variables and constraints) of the decomposed constraint network while preserving solution-set equivalence.

## Experiments

### Main Results: Decomposition Size Analysis (89 instances, MiniZinc 2024 Challenge)

| Stage | Var Growth (avg) | Var Growth (med) | Var Growth (max) | Con Growth (avg) | Con Growth (med) | Con Growth (max) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| FlatZinc | 9.42x | 1.86x | 111.62x | 24.94x | 2.95x | 486.87x |
| TCN Decomposition | 53.61x | 7.97x | 1133.22x | 76.68x | 6.21x | 1837.19x |
| **After Preprocessing** | **22.06x** | **4.76x** | 344.62x | **36.39x** | **4.33x** | 746.09x |

### Ablation Study: Preprocessing Effectiveness

| Metric | Variable Reduction (TCN→Preprocessed) | Constraint Reduction (TCN→Preprocessed) |
|--------|:---:|:---:|
| Mean reduction | 53.61x → 22.06x (2.4×) | 76.68x → 36.39x (2.1×) |
| Median reduction | 7.97x → 4.76x (1.7×) | 6.21x → 4.33x (1.4×) |
| Preprocessing time (avg) | 24.22s (std=96.91s) | — |
| Preprocessing time (med) | 0.91s | Only 11 instances >10s |

### Key Findings
1. **FlatZinc stage already introduces the primary blowup**: For 63/89 instances, FlatZinc decomposition remains within one order of magnitude compared to Choco
2. **Preprocessing substantially reduces network size**: On average, it reduces the variable growth introduced by TCN decomposition by 2.5× and constraint growth by 1.5×
3. **Operator distribution is highly concentrated**: After preprocessing, instances predominantly use only three operators—addition, maximum (encoding disjunction), and reified equality—effectively eliminating thread divergence concerns
4. **12 hard-to-handle instances**: 12 instances still exhibit over 100× blowup after decomposition, representing extreme cases

## Highlights & Insights
- **Theoretical completeness**: Solution-set equivalence and uniqueness of TCN decomposition are formally proved, making the framework self-contained and directly implementable
- **Practical effectiveness**: Seven preprocessing techniques reduce blowup to an acceptable level (median 4.8× variables / 4.5× constraints)
- **Educational value**: TCN formalization is applicable to CP pedagogy and has been validated in teaching settings
- **GPU-friendly**: The regularized ternary layout eliminates thread divergence caused by operator diversity

## Limitations & Future Work
- Global constraints are not natively supported (relying on the MiniZinc compiler for pre-decomposition), limiting solving efficiency
- 12 instances still exceed 100× blowup after decomposition, requiring stronger preprocessing techniques
- Preprocessing time exhibits extremely high variance (std=96.91s vs. median=0.91s), with prohibitive cost on individual instances
- No end-to-end GPU solver performance improvement is demonstrated

## Related Work & Insights
- **TCNs in constraint logic programming**: CLP(FD), clp(fd), and cc(FD) employed ternary forms early on but without systematic formalization
- **View-based propagation**: The AST interpreter approach in Choco/OR-Tools avoids decomposition but is ill-suited for GPU execution
- **Preprocessing techniques**: Savile Row and Andrea Rendl's compilation optimizations
- **GPU constraint solving**: The Turbo solver, proposed at AAAI 2022, introduced GPU-based concurrent constraint programming

## Rating
⭐⭐⭐ (Theoretically rigorous with high formalization value, but lacks end-to-end GPU performance validation; practical impact remains to be observed)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Formal Abductive Latent Explanations for Prototype-Based Networks](formal_abductive_latent_explanations_for_prototype-based_networks.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/others/singular_bayesian_neural_networks.md)

</div>

<!-- RELATED:END -->
