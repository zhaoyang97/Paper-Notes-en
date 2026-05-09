---
title: >-
  [Paper Note] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks
description: >-
  [NeurIPS 2025][ReLU networks] This paper systematically identifies six mutually non-equivalent definitions of "linear regions" in ReLU networks, proves that counting linear regions is #P-hard under all six definitions (even for single-hidden-layer networks), and establishes strong inapproximability results together with polynomial-space upper bounds for deeper networks.
tags:
  - NeurIPS 2025
  - ReLU networks
  - linear regions
  - computational complexity
  - "#P-hard"
  - NP-hard
date: 2026-05-08
content_hash: bca6e754a86df116
---

# The Computational Complexity of Counting Linear Regions in ReLU Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2505.16716](https://arxiv.org/abs/2505.16716)
**Code**: Unavailable
**Area**: Theoretical Deep Learning / Computational Complexity
**Keywords**: ReLU networks, linear regions, computational complexity, #P-hard, NP-hard

## TL;DR

This paper systematically identifies six mutually non-equivalent definitions of "linear regions" in ReLU networks, proves that counting linear regions is #P-hard under all six definitions (even for single-hidden-layer networks), and establishes strong inapproximability results together with polynomial-space upper bounds for deeper networks.

## Background & Motivation

Functions computed by ReLU networks are continuous piecewise linear (CPWL)—the input space can be partitioned into finitely many "linear regions," on each of which the function is affine. Since the seminal works of Pascanu et al. (2014) and Montúfar et al. (2014), the number of linear regions has been regarded as a standard measure of the expressive power of ReLU networks. A large body of subsequent work has focused on deriving upper and lower bounds on this count and developing enumeration algorithms.

Yet a most fundamental computational question has been overlooked: **given a network, how much time and space is actually required to compute the number of its linear regions?** More surprisingly, even before addressing counting, the seemingly simple question of "what is a linear region?" remains ambiguous—at least six distinct, mutually non-equivalent definitions appear in the literature. These inconsistencies have already caused confusion and even errors in prior work.

The paper is motivated by a systematic treatment of these questions from the perspective of computational complexity: first unifying and distinguishing the various definitions, then establishing rigorous complexity classifications for each.

## Method

### Overall Architecture

The contributions are organized at three levels: (1) **definitional clarification**—identifying and comparing six non-equivalent definitions; (2) **lower bound proofs**—establishing #P-hard and NP-hard results; (3) **upper bound algorithms**—showing that counting is achievable within polynomial space under certain definitions.

### Key Designs

1. **Systematic taxonomy of six linear region definitions**:

   - **Definition 1 (Activation Region)**: The set $S_a$ of inputs sharing the same activation pattern $a \in \{0,1\}^{s(N)}$ (may be lower-dimensional or non-convex).
   - **Definition 2 (Proper Activation Region)**: A full-dimensional activation region.
   - **Definition 3 (Convex Region)**: A full-dimensional polyhedron in a polyhedral complex compatible with $f_N$ (using the minimum such count).
   - **Definition 4 (Open Connected Region)**: A maximal open connected subset on which $f_N$ is affine.
   - **Definition 5 (Closed Connected Region)**: A maximal closed connected subset on which $f_N$ is affine.
   - **Definition 6 (Affine Region)**: A maximal subset on which $f_N$ is affine (connectivity not required).

   Containment chain: $R_6 \leq R_5 \leq R_4 \leq R_3 \leq R_2 \leq R_1$, with each inequality being strict in general. **Design Motivation**: Rather than advocating a single "best" definition, the paper eliminates ambiguity by precisely characterizing the distinctions and relationships among all six, and annotates 20 representative prior works with the definition each actually employs.

2. **Complexity analysis for shallow networks (one hidden layer)**:

   - **Positive result**: Deciding whether a network has exactly one linear region (1-region-decision) lies in P for all six definitions. The key mechanism is checking whether the hyperplane associated with each neuron is "cancelled"—i.e., whether the gradient of $f_N$ is continuous across that hyperplane.
   - **Negative result**: Exact counting is #P-complete for Definitions 1–4 and #P-hard for Definitions 5–6. The proofs proceed via reduction from counting cells in a hyperplane arrangement (known to be #P-complete), by constructing a neural network whose linear regions correspond exactly to the cells of the arrangement. **Design Motivation**: This establishes that exact counting is fundamentally intractable even for the simplest one-layer networks (unless #P = FP).

3. **Complexity analysis for deep networks (multiple hidden layers)**:

   - **NP-hardness of the decision version** (Theorem 5.2): For any fixed constants $K, L \geq 2$, deciding whether an $L$-layer network has more than $K$ linear regions is NP-hard. **Key Lemma** (Lemma 5.3): Reduction from SAT—given a formula $\phi$, a two-layer network $N_\phi$ is constructed such that: (a) if $\phi$ is unsatisfiable, $N_\phi$ computes the constant zero function (1 region); (b) if $\phi$ is satisfiable, $N_\phi$ has at least 2 regions (a bounded non-zero region near each satisfying assignment). This improves upon Wang (2022), which required $\Theta(\log n)$ depth.
   - **Inapproximability** (Theorem 5.7): For $L \geq 3$, no polynomial-time algorithm can approximate the number of linear regions to within a ratio greater than $(2^n+1)^{-1}$ unless NP = RP. Intuitively, an unsatisfiable formula yields 1 region while a satisfiable one yields at least $2^n + 1$; any effective approximation would thus resolve SAT.
   - **Inapproximability for two-layer networks** (Theorem 5.8): For two-layer networks, the approximation ratio decays exponentially as $2^{-O(n^{1-\epsilon})}$.

4. **Polynomial-space upper bounds** (positive results): For Definitions 1, 2, and 6, linear region counting lies in FPSPACE. Algorithm sketch:

   - Definitions 1–2: Enumerate all $2^{s(N)}$ possible activation patterns and verify each individually.
   - Definition 6: Enumerate all combinations of affine function coefficients whose encoding length is within a polynomial bound, and check whether each is realized by some proper activation region.

   **Design Motivation**: Although polynomial-time algorithms are not expected to exist, this establishes that the problem does not require exponential space.

### Loss & Training

This is a purely theoretical paper and involves no training. All results are worst-case complexity analyses.

## Key Experimental Results

### Main Results (Summary of Theoretical Results)

| Problem | Network Depth | Def. 1–2 | Def. 3 | Def. 4–5 | Def. 6 |
|---|---|---|---|---|---|
| 1-region-decision | 1 layer | P | P | P | P |
| Exact counting | 1 layer | #P-complete | #P-complete | #P-hard | #P-hard |
| K-region-decision | ≥2 layers | ? | NP-hard | NP-hard | NP-hard |
| Approximate counting | ≥3 layers | ? | NP-hard | NP-hard | NP-hard |
| Polynomial space | Any | ✓ | ? | ? | ✓ |

### Ablation Study (Region Count Comparison Across Definitions)

| Network function example | $R_1$ | $R_2$ | $R_3$ | $R_4$ | $R_5$ | $R_6$ |
|---|---|---|---|---|---|---|
| $f(x,y)=\max(-y,\min(0,-x))$ | 7 | 5 | 4 | 3 | 3 | 3 |
| Example function in Figure 2 | — | — | — | 9 | 8 | 7 |

### Key Findings

- Exact counting of linear regions is already #P-hard for networks with a single hidden layer, implying that no efficient exact counting algorithm is likely to exist.
- Two hidden layers suffice to render the decision problem "does the network have more than one region?" NP-hard.
- For three or more layers, not only is exact counting intractable, but approximation within an exponential factor is also infeasible.
- There is a subtle yet important distinction between open and closed connected regions: a closed connected region can "extend" along a lower-dimensional slice of another closed connected region.

## Highlights & Insights

- **Clarifying the literature**: The systematic classification of six non-equivalent definitions and their relationships identifies and corrects prior confusion—for instance, the algorithm of Wang (2022) is shown to apply to Definition 4 rather than Definition 5 as claimed. This serves as an important reference for future research.
- **Elegance of the reduction**: The SAT-to-two-layer-network reduction is particularly elegant: the constructed network produces non-zero values in bounded regions near satisfying assignments and is identically zero elsewhere, establishing a precise correspondence between satisfiability and region count.
- **Research implications**: Although the results are worst-case, they suggest an interesting direction: networks trained via gradient descent may possess fewer linear regions, potentially admitting faster counting algorithms.

## Limitations & Future Work

- All hardness results are worst-case; networks obtained from practical training may exhibit special structure that makes the problem tractable.
- Not all results apply uniformly to all six definitions (e.g., the inapproximability for Definitions 1–2 and the question of whether counting lies in #P for Definitions 5–6 remain open).
- The paper is purely theoretical and does not connect the complexity results to the empirical performance of practical network analysis tools.
- The fixed-parameter tractability (FPT) status under various parameterizations and definitions remains unresolved.

## Related Work & Insights

- **Comparison with Wang (2022)**: Wang's NP-hardness result requires $\Theta(\log n)$ depth; the present paper improves this to any fixed constant $L \geq 2$. Furthermore, Wang's claimed EXPTIME upper bound applies to Definition 4 rather than Definition 5.
- **Comparison with practical algorithms of Serra et al.**: Mixed-integer programming methods can enumerate linear regions for concrete networks but carry no theoretical complexity guarantees.
- **Implications**: The computational hardness motivates the study of alternative measures—perhaps the "effective" number of regions in trained networks or more tractable notions of expressive power deserve greater attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic classification of six definitions with a complete complexity landscape; multiple technical contributions including the SAT-to-two-layer-network reduction, inapproximability results, and polynomial-space upper bounds.
- **Experimental Thoroughness**: ⭐⭐⭐ — Purely theoretical work; definitional distinctions are verified through precise mathematical examples, but no empirical experiments are conducted.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematically rigorous, clearly organized; the definition table is an excellent reference tool.
- **Value**: ⭐⭐⭐⭐ — Significant theoretical contribution; clarifies confusion in the literature and establishes fundamental complexity boundaries for the problem, with lasting impact on theoretical deep learning research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)

<!-- RELATED:END -->
