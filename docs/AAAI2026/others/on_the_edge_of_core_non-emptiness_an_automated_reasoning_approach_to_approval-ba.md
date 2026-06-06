---
title: >-
  [Paper Note] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting
description: >-
  [AAAI 2026][Core Stability] This paper proposes an automated reasoning framework based on Mixed Integer Linear Programming (MILP) to investigate the major open problem of whether core stability always exists in approval-…
tags:
  - "AAAI 2026"
  - "Core Stability"
  - "Multi-Winner Voting"
  - "Mixed Integer Linear Programming"
  - "Automated Reasoning"
  - "Proportional Representation"
date: 2026-05-08
content_hash: 77640815d2f09cf9
---

# On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting

**Conference**: AAAI 2026
**arXiv**: [2512.16895](https://arxiv.org/abs/2512.16895)  
**Code**: [GitHub](https://github.com/emanueltewolde/Core-MILP)  
**Area**: Computational Social Choice / Automated Reasoning
**Keywords**: Core Stability, Multi-Winner Voting, Mixed Integer Linear Programming, Automated Reasoning, Proportional Representation

## TL;DR

This paper proposes an automated reasoning framework based on Mixed Integer Linear Programming (MILP) to investigate the major open problem of whether core stability always exists in approval-based multi-winner voting. The framework establishes new existence results, uncovers previously unknown relationships between core stability and other axioms (e.g., Lindahl pricability), and refutes an existing conjecture.

## Background & Motivation

In multi-winner voting, the task is to select a committee from a set of candidates. Core stability is a natural and important notion of group fairness: a committee is core stable if and only if no "coalitional deviation" exists—that is, no group of voters can jointly support an alternative committee that is strictly preferred by all of them, using their proportionally allocated seats.

Core stability sits at the top of the representativeness axiom hierarchy proposed by Aziz et al. (2017) and implies many subsequently introduced axioms. Its applicability extends beyond political elections: fairness in federated learning (Chaudhury et al.) and multi-system decision-making in AI alignment (Conitzer et al.) can both be characterized via core stability.

Nevertheless, a **major open problem** remains: does a core stable committee **always** exist in approval-based elections? Prior results were limited to committee size $k \leq 3$ (Cheng et al. 2019) or $k \leq 8$ (Peters 2025, but only for the PAV rule). This paper adopts a **rule-agnostic** approach, searching over all possible voter preferences for the cases where core stability is closest to being violated.

The methodological contribution lies in a key advantage over the SAT-based methods prevalent in computational social choice: the MILP approach can prove results for a given number of candidates **independently of the number of voters**, yielding substantial computational gains.

## Method

### Overall Architecture

The technical approach proceeds as follows:

1. Linearize the core stability problem using vote distributions, eliminating dependence on the number of voters $n$
2. Reformulate the question of core emptiness as a nested optimization problem (max-min-max), then reduce it to a single-level MILP
3. Solve the MILP for different parameters $(m, k)$ using a solver (Gurobi)
4. Identify patterns from experimental results and prove matching upper bounds via duality theory
5. Modify the MILP to explore relationships between core stability and other axioms (e.g., pricability)

### Key Designs

1. **Linearization via Vote Distribution Space (Xia 2024)**: Core stability can be expressed in terms of vote distributions $\mathbf{x} \in \Delta(2^C)$ rather than concrete voter sets. Key lemma: committee $W$ is core stable if and only if for all deviations $W' \in \mathcal{M}_{\leq k}$:

$$\boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k} < 0$$

where $\boldsymbol{\delta}_{W,W'} \in \{0,1\}^{2^C}$ encodes which ballots $A$ strictly prefer $W'$ over $W$. This linearization eliminates dependence on $n$.

2. **MILP Formulation**: Determining whether the core can be empty for given $m, k$ is equivalent to solving:

$$\max_{\mathbf{x} \in \Delta(2^C)} \min_{W \in \mathcal{M}_k} \max_{W' \in \mathcal{M}_{\leq k}} \boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k}$$

This nested optimization is converted into a single-level MILP with binary variables $\mathbf{y}[W, W']$: when $\mathbf{y}[W, W'] = 1$, the deviation $W'$ maximizes the inner max problem. If the MILP optimal value $\mu^*$ is negative, the core is non-empty for all vote distributions; if non-negative, there exists a vote distribution that empties the core.

3. **Dual Analysis (DLP)**: Fixing the binary variables $\mathbf{y}$ yields a linear program whose dual produces a more compact dual linear program DLP. The dual variables $\mathbf{q}$ define a probability distribution over committees and provide an upper bound for the MILP. Core Theorem (Theorem 3): MILP $\leq v$ if and only if for all deviation functions $D$, DLP $\leq v$.

   This yields a probabilistic reformulation of core non-emptiness (Corollary 7): the core is non-empty for all distributions if and only if for every deviation function $D$, there exists a distribution $\mathbf{q}$ over committees such that for all ballots $A$, the probability of preferring a deviation is strictly less than the average deviation size divided by $k$.

4. **Droop Core and Hare Quota**: Replacing the Hare quota ($\frac{1}{k}$) in the core definition with the stricter Droop quota ($\frac{1}{k+1}$) yields the Droop core. Theorem 2 proves that the MILP lower bound is $\frac{-1}{k(k+1)}$, exactly corresponding to the Droop threshold—implying that the Droop quota is the **smallest quota that can guarantee core non-emptiness**.

### Loss & Training

This paper involves no machine learning training; the central formulation is a mathematical optimization problem. The core objective is the MILP:

$$\max_{\mathbf{x}, \mu, \mathbf{y}} \mu \quad \text{s.t.} \quad \forall W, W': \mu \leq \boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k} + 3(1 - \mathbf{y}[W,W'])$$

Global optimality is guaranteed using the Gurobi solver.

## Key Experimental Results

### Main Results

MILP optimal values for different parameters $(m, k)$:

| $k \backslash m$ | 4 | 5 | 6 | 7 | Formula |
|---|---|---|---|---|---|
| 1 | -0.5000 | -0.5000 | -0.5000 | -0.5000 | $-\frac{1}{2}$ |
| 2 | -0.1667 | -0.1667 | -0.1667 | -0.1667 | $-\frac{1}{6}$ |
| 3 | -0.0833 | -0.0833 | -0.0833 | -0.0833 | $-\frac{1}{12}$ |
| 4 | — | -0.0500 | -0.0500 | -0.0500 | $-\frac{1}{20}$ |
| 5 | — | — | -0.0333 | -0.0333 | $-\frac{1}{30}$ |
| 6 | — | — | — | -0.0238 | $-\frac{1}{42}$ |

**Key finding**: All optimal values are negative and follow the formula $\frac{-1}{k(k+1)}$, independent of the number of candidates $m$.

### Ablation Study (Upper Bounds for Different Deviation Function Classes)

| Deviation Function Type | DLP Upper Bound | DrDLP Upper Bound | Notes |
|------------|---------|-----------|------|
| All unit deviations | $-\frac{1}{k(k+1)}$ | $\leq 0$ | Theorem 4: lower bound matches exactly |
| At most one non-unit deviation $|D(W^*)| = t$ | $-\frac{1}{k(k+2-t)}$ | $\leq 0$ | Non-unit deviations only push value further from 0 |
| Large committees $m = k+1$ | $-\frac{1}{k(k+1)}$ | $\leq 0$ | Theorem 5: holds for all deviation functions |

### Key Findings

1. **Core is always non-empty for $m \leq 7$**: Improves upon the prior experimental result of $m + n \leq 14$
2. **Droop quota is the "optimal quota"**: No quota smaller than Droop can guarantee core non-emptiness (Corollary 6)
3. **Core non-emptiness for large committees**: For any $m$ and $k = m-1$, the core (and even the Droop core) is always non-empty (Corollary 9)
4. **Refutation of the Lindahl pricability conjecture**: Munagala et al. conjectured that core stability is equivalent to Lindahl pricability; this paper finds a minimal counterexample ($m=4, k=2$) showing that the Droop core does not imply Lindahl pricability (Theorem 6)
5. **Confirmation of minimality**: The framework not only finds counterexamples but also confirms their minimality within the parameter space

## Highlights & Insights

1. **Methodological innovation—MILP over SAT**: Computational social choice has long relied on SAT solvers, which struggle as the number of voters grows. By linearizing via vote distributions, this paper converts the problem into a MILP that completely eliminates dependence on $n$. The same instance $(m=7, k=3)$ is solved within 2.5 hours under MILP, whereas the method of Peters (2025) fails to converge after 37 hours.
2. **Elegant application of duality theory**: LP duality elevates the pattern observed in computational experiments ($\frac{-1}{k(k+1)}$) to a rigorous mathematical proof, exemplifying the elegant paradigm of "computational discovery → theoretical proof."
3. **Probabilistic core non-emptiness condition (Corollary 7)**: The deterministic core non-emptiness problem is restated in probabilistic language—for each deviation function, there exists a distribution over committees such that the probability of preferring a deviation is sufficiently small—paving the way for future probabilistic proof techniques.
4. **Conjecture refutation and minimal counterexample**: The paper not only refutes an existing conjecture but also precisely locates the minimal parameter values at which counterexamples exist, providing a human-readable proof.
5. **Game-theoretic perspective**: The core non-emptiness problem is interpreted as an adversarial team game, establishing a connection with the least core concept in cooperative game theory.

## Limitations & Future Work

1. **Limited computational scalability**: The MILP contains $2^m + 1$ continuous variables and $\binom{m}{k} \cdot \sum_{l=1}^{k} \binom{m}{l}$ binary variables, growing super-polynomially when $k \approx m/2$. The instance $m=8, k=4$ does not converge within 72 hours.
2. **General case remains unproven**: Although both experiments and partial theory point to a MILP optimal value of $\frac{-1}{k(k+1)}$ (suggesting the core is always non-empty), a proof for the general case remains open.
3. **Droop core non-emptiness not fully resolved**: While all experimental DrMILP values converge to 0, a general proof is absent.
4. **Compatibility with other axioms unknown**: Whether core stability is compatible with EJR+, committee monotonicity, and related axioms remains open.
5. **Restricted to approval voting**: The framework is not extended to related settings such as approval-disapproval voting.

## Related Work & Insights

- **Cheng et al. (2019)**: Prove core non-emptiness for $k \leq 3$ and establish the existence of stable lotteries. Corollary 7 of this paper has a deep connection to their results.
- **Peters (2025)**: Proves that PAV yields core stable committees for $k \leq 8$ but provides a PAV counterexample for $k = 9$. The present method is rule-agnostic.
- **Xia (2024)**: The "linear" property of the core—that it depends only on vote distributions rather than specific voters—constitutes the theoretical foundation for the MILP in this paper.
- **Munagala et al. (2022, 2024)**: Introduce Lindahl pricability and conjecture its equivalence to core stability; refuted by this paper.
- **Implications**: The MILP approach can be generalized to other social choice problems (e.g., fair division, matching markets), particularly those with linear structure that currently rely on SAT solvers.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A breakthrough methodological innovation; the MILP-over-SAT paradigm has broad impact in computational social choice.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — A sophisticated combination of mixed integer programming, LP duality, and game theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Constrained by computational resources but thorough within feasible range; theoretical proofs compensate for experimental limitations.
- **Value**: ⭐⭐⭐ — Primarily a theoretical contribution, with potential implications for democratic electoral system design and AI fairness.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear mathematical writing, with running examples throughout; the theory–experiment–theory progression is elegant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)

</div>

<!-- RELATED:END -->
