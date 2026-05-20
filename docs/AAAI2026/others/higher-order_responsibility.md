---
title: >-
  [Paper Note] Higher-Order Responsibility
description: >-
  [AAAI 2026][higher-order responsibility] This paper studies higher-order responsibility in sequential decision-making mechanisms and establishes two core theorems: (1) any mechanism with $n$ agents is necessarily $n$-gap…
tags:
  - "AAAI 2026"
  - "higher-order responsibility"
  - "responsibility gap"
  - "sequential decision-making"
  - "computational complexity"
  - "polynomial hierarchy"
date: 2026-05-08
content_hash: 6915c3be11b59f56
---

# Higher-Order Responsibility

**Conference**: AAAI 2026
**arXiv**: [2506.01003](https://arxiv.org/abs/2506.01003)  
**Code**: None  
**Area**: AI Ethics / Computational Social Choice / Formal Methods
**Keywords**: higher-order responsibility, responsibility gap, sequential decision-making, computational complexity, polynomial hierarchy

## TL;DR

This paper studies higher-order responsibility in sequential decision-making mechanisms and establishes two core theorems: (1) any mechanism with $n$ agents is necessarily $n$-gap-free (i.e., a responsible agent can always be found at some order); (2) determining whether a mechanism is $d$-gap-free is $\Pi_{2d+1}$-complete.

## Background & Motivation

As autonomous agents — such as self-driving vehicles, military robots, and medical assistants — increasingly participate in decisions affecting human life, clearly attributing responsibility in multi-party human-machine collaborative environments has become critical. This matters both for establishing accountability frameworks and for maintaining societal trust.

**Definition of counterfactual responsibility**: Traditionally, individual responsibility is grounded in Frankfurt's Principle of Alternate Possibilities — "a person is morally responsible for what they have done only if they could have done otherwise." In the AI literature, "could have done otherwise" is interpreted as possessing a strategy that guarantees avoidance of harmful outcomes regardless of what other agents do.

**The responsibility gap problem**: This definition gives rise to a *responsibility gap* in collective decision-making — harmful outcomes occur, yet no single agent bears counterfactual responsibility. Consider two factories each accumulating 20 kg of pollutants, where only 15 kg suffices to kill fish. If they act simultaneously, neither factory has a strategy guaranteeing fish survival (since the other may have already discharged), creating a responsibility gap.

**Limitations of existing solutions**: One approach is collective responsibility, but this dilutes individual accountability and generates blame-shifting cycles. Another recently proposed approach is *higher-order responsibility* — rather than asking who directly caused the outcome, it asks who should be held responsible for the fact that no one is responsible.

**Key Insight**: The authors formalize the concept of higher-order responsibility and investigate two key questions: when can higher-order responsibility close the gap, and what is the computational complexity of determining whether it does?

## Method

### Overall Architecture

The paper adopts a purely theoretical approach. It first defines a formal model of sequential decision-making mechanisms, then recursively defines $d$-th order responsibility and $d$-th order gaps, and finally analyzes the complexity of the gap-closure problem.

### Key Designs

1. **Formalization of Decision Mechanisms (Definition 1)**:

    - Function: Defines a sequential decision mechanism $(n, \mathbf{v}, \gamma)$
    - Core definition: $n$ is the number of agents; $\mathbf{v} = \{\mathbf{v}_i\}_{1 \leq i \leq n}$ is the set of Boolean variables each agent controls (representing available actions); $\gamma$ is an obligatory constraint (a Boolean formula specifying which action combinations are acceptable)
    - Key assumption: Agents act in order $1, 2, \ldots, n$, and later agents can observe the choices of earlier ones
    - Connection to game theory: This is essentially a perfect-information sequential game, but the focus is on responsibility attribution rather than equilibria

2. **Formalization of Counterfactual Responsibility (Equation 6)**:

    - Function: Precisely characterizes counterfactual responsibility using quantified Boolean formulas
    - Core formula: Agent $i$'s first-order responsibility is $\mathsf{R}_i = \neg\gamma \wedge \exists\mathbf{v}_i \forall\mathbf{v}_{i+1} \ldots \forall\mathbf{v}_n \gamma$
    - Intuition: The obligatory constraint is violated ($\neg\gamma$), yet agent $i$ possesses some action ($\exists\mathbf{v}_i$) such that no matter what subsequent agents do ($\forall\mathbf{v}_{i+1} \ldots$), the constraint would be satisfied ($\gamma$)
    - Key technical treatment: Variables with identical names inside and outside quantifiers denote different objects (hypothetical alternative actions vs. actual actions) — unconventional mathematically but renders the higher-order definition more compact

3. **Recursive Definition of Higher-Order Responsibility (Equation 7)**:

    - Function: Recursively defines $d$-th order responsibility
    - Core Idea: $d$-th order responsibility = counterfactual responsibility for a $(d{-}1)$-th order gap; a $d$-th order gap = the set of action profiles in which the obligatory constraint is violated and no agent bears responsibility at any order from 1 to $d{-}1$
    - Formula: $\mathsf{R}_i^d = (\neg\gamma \wedge \bigwedge_{j \leq n} \neg\mathsf{R}_j \wedge \ldots \wedge \bigwedge_{j \leq n} \neg\mathsf{R}_j^{d-1}) \wedge \exists\mathbf{v}_i \forall\mathbf{v}_{i+1} \ldots \forall\mathbf{v}_n \neg(\ldots)$
    - Design Motivation: This recursive definition extends responsibility attribution from "who directly caused the outcome" to "who should be held responsible for the absence of accountability," progressively tightening the responsibility net layer by layer

4. **Gap Closure and the Definition of $d$-gap-free**:

    - Function: Defines mechanisms that admit no gap up to order $d$
    - Core definition (Definition 3): A mechanism is $d$-gap-free if and only if for every action profile violating the obligatory constraint, there exist some $d' \leq d$ and some agent $i$ such that $i$ bears $d'$-th order responsibility for that profile
    - Denoted $\mathsf{GF}^d$; clearly $\mathsf{GF}^{d_1} \subseteq \mathsf{GF}^{d_2}$ for $d_1 \leq d_2$

### Main Theorems

**Theorem 2 (Gap-Closure Guarantee)**: If the Boolean formula $\gamma$ is satisfiable, then $(n, \mathbf{v}, \gamma) \in \mathsf{GF}^n$.

- Implication: In a sequential decision mechanism with $n$ agents, provided the obligatory constraint can be satisfied, responsibility of order at most $n$ suffices to close all gaps — some agent can always be held accountable
- Proof technique: Backward induction via Lemma 1, starting from a "good" action profile satisfying the obligatory constraint and progressively comparing deviations toward the "bad" profile

**Theorem 3 (Complexity Characterization)**: $\mathsf{GF}^d$ is $\Pi_{2d+1}$-complete.

- Implication: Determining whether a mechanism is $d$-gap-free lies exactly at the $\Pi_{2d+1}$ level of the polynomial hierarchy — as the order increases, the problem's complexity climbs layer by layer
- Upper bound (Lemma 2): Proved by induction on the number of quantifier alternations in the formula $\mathsf{R}_i^d$
- Lower bound (Lemmas 8–9): A clever Devil vs. Moralist game is constructed, reducing the problem of deciding quantified Boolean formulas (QBF) to the gap-decision problem; the notion of *degree of immorality* is introduced to track the number of "violations" in an action profile

## Key Experimental Results

This is a purely theoretical paper with no experiments. All core results are established by mathematical proof.

### Comparison of Theoretical Results

| Result | Ours | Prev. SOTA (Shi, 2024) | Gain |
|--------|------|------------------------|------|
| Order $d$ at which gap is guaranteed empty | $d \geq n$ (number of agents) | $d \geq 2^n - 1$ (number of leaf nodes) | Exponential improvement |
| Complexity of gap-decision | $\Pi_{2d+1}$-complete | Polynomial time (but parameterized by leaf nodes) | First exact characterization parameterized by number of agents |

### Comparison with Collective Responsibility

| Approach | Responsibility Attribution | Dilution | Example (three factories, 20 kg each) |
|----------|--------------------------|----------|---------------------------------------|
| Collective responsibility | Minimal group (e.g., B and C) | Yes, blame-shifting cycle | B and C jointly responsible |
| Second-order responsibility | Individual (B alone) | No | B solely bears second-order responsibility |

### Key Findings

- Sequential decision-making produces smaller responsibility gaps than simultaneous decision-making (additional information makes later-acting agents more likely to be held accountable)
- Higher-order responsibility preserves individual accountability and avoids the dilution effect of collective responsibility
- The precise complexity characterization shows that verifying responsibility of increasing order becomes exponentially harder

## Highlights & Insights

- The paper elegantly transforms the philosophical question of "who should be responsible for the absence of responsibility" into a computational complexity problem, demonstrating the formalization potential of AI ethics
- The Devil vs. Moralist game construction is particularly elegant: by introducing auxiliary variables $q_{2i}$ and the concept of degree of immorality, QBF is naturally embedded into the responsibility-decision problem
- The improvement of the gap-closure order bound from $2^n - 1$ to $n$ is exponential, demonstrating that the special structure of sequential decision mechanisms can be fully exploited
- The acknowledgments mention that "an AI reviewer identified a non-trivial gap in the proof of Lemma 8" — an interesting case study in AI-assisted peer review

## Limitations & Future Work

- Only simple sequential mechanisms in which each agent acts exactly once are considered; in more general extensive-form games (where agents may act multiple times), the results weaken considerably
- Probabilistic behavior and incomplete information settings are not addressed, whereas real-world decisions routinely involve uncertainty
- $\Pi_{2d+1}$-completeness implies that even for small $d$, the problem may be practically intractable
- Real-world responsibility attribution often involves legal, social, and other non-formal factors, limiting the direct applicability of purely formal methods
- Future directions: higher-order responsibility under incomplete information, probabilistic obligatory constraints, and approximation algorithms for gap decision

## Related Work & Insights

- Frankfurt's Principle of Alternate Possibilities serves as the starting point, though this paper extends it via multi-order generalization to collective decision-making settings
- Unlike the responsibility gap literature on "discursive dilemmas" in Braham & van Hees (2018), this paper focuses on sequential mechanisms rather than voting mechanisms
- Compared with higher-order responsibility in extensive-form games studied by Shi (2024), this paper operates under a more restricted setting yet achieves exponentially stronger results
- Insight: This formalized hierarchical accountability framework can be applied to AI safety — when an autonomous system fails and no one is directly responsible, one can ask "who should be responsible for failing to establish adequate safety mechanisms"

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The computational complexity analysis of higher-order responsibility is entirely novel; the proof techniques (Devil-Moralist game, degree of immorality) are highly original
- Experimental Thoroughness: ⭐⭐⭐ — A purely theoretical work; theorems are rigorously and completely proved, but no case studies or simulation validation are provided
- Writing Quality: ⭐⭐⭐⭐⭐ — The series of factory pollution examples, progressing from simple to complex, makes abstract concepts highly intuitive
- Value: ⭐⭐⭐⭐ — Provides a rigorous computational complexity foundation for responsibility attribution in AI ethics, though bridging work is needed for practical application

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STEM Faculty Perspectives on Generative AI in Higher Education](stem_faculty_perspectives_on_generative_ai_in_higher_education.md)
- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](../../CVPR2026/others/order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[NeurIPS 2025\] Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics](../../NeurIPS2025/others/uncertainty_quantification_for_reduced-order_surrogate_models_applied_to_cloud_m.md)

</div>

<!-- RELATED:END -->
