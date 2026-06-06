---
title: >-
  [Paper Note] Certified Branch-and-Bound MaxSAT Solving (Extended Version)
description: >-
  [AAAI 2026][MaxSAT] This paper introduces VeriPB-based certification for Branch-and-Bound MaxSAT solvers, covering two core techniques: look-ahead bounding methods and multi-valued decision diagram (MDD) encodings. Exper…
tags:
  - "AAAI 2026"
  - "MaxSAT"
  - "Proof Logging"
  - "Branch-and-Bound"
  - "VeriPB"
  - "Multi-valued Decision Diagrams"
date: 2026-05-08
content_hash: f2f79f2def51350d
---

# Certified Branch-and-Bound MaxSAT Solving (Extended Version)

**Conference**: AAAI 2026
**arXiv**: [2511.10273](https://arxiv.org/abs/2511.10273)  
**Code**: [Zenodo](https://zenodo.org/) (Vandesande et al., 2025)  
**Area**: Other
**Keywords**: MaxSAT, Proof Logging, Branch-and-Bound, VeriPB, Multi-valued Decision Diagrams

## TL;DR

This paper introduces VeriPB-based certification for Branch-and-Bound MaxSAT solvers, covering two core techniques: look-ahead bounding methods and multi-valued decision diagram (MDD) encodings. Experiments on the MaxCDCL solver demonstrate a median proof logging overhead of only 19%, filling the last remaining gap in certified MaxSAT solving paradigms.

## Background & Motivation

### Problem Definition

Combinatorial optimization solvers are widely deployed in safety-critical domains (spacecraft control, organ transplant matching, traffic infrastructure verification, etc.), yet solvers may produce **infeasible solutions or incorrectly claim optimality** due to source-code bugs. A systematic approach to guarantee the correctness of solver outputs is therefore needed.

### Why Proof Logging

**Formal verification** of the solver itself is prohibitively expensive and degrades performance.

**Certifying Algorithms**: the solver outputs not only an answer but also a correctness proof, which is verified by an independent proof checker.

In the SAT community, proof logging has already become a mandatory requirement in competitions.

### State of Certification for MaxSAT

The main solving paradigms for MaxSAT (the optimization variant of SAT):

**Solution-Improving Search**: certification exists ✅

**Core-Guided Search**: certification exists ✅

**Implicit Hitting Set**: addressed in a concurrent paper ✅

**Branch-and-Bound**: **the gap filled by this paper** ❌→✅

### Challenges in MaxCDCL

MaxCDCL (the current state-of-the-art Branch-and-Bound solver, winner of the 2023/2024 MaxSAT Evaluations) combines CDCL search with sophisticated bounding functions. The main challenges for certification are:

**Look-ahead methods** produce conditional unsatisfiable cores (local cores), whose correctness must be proved.

**MDD encodings** translate pseudo-Boolean constraints into CNF; due to the reducedness of BDD/MDDs, a single node may represent multiple equivalent constraints—the proof must demonstrate that these constraints are indeed equivalent.

## Method

### Overall Architecture

Within the MaxCDCL solver, the VeriPB proof system (a proof system based on 0-1 integer linear inequalities) is used to record each reasoning step. VeriPB maintains a **proof configuration** $\mathcal{F}$ and performs derivations using the cutting planes proof system: literal axioms, linear combinations, division, saturation, and Reverse Unit Propagation (RUP).

### Key Designs

#### 1. Certifying the Look-Ahead Bounding Method

**Weighted Local Core**: a triple $q = \langle w, R, K \rangle$, where $R \subseteq \alpha$ (a subset of the current assignment) and $K$ contains negations of objective literals, satisfying $F \wedge R \wedge K \models \bot$.

**$\mathcal{O}$-compatible core set**: a core set $\mathcal{Q}$ is compatible if the total weight assigned to each objective literal by cores does not exceed its weight in the objective function. This guarantees the following two types of derivation:

- **Soft conflict** (Proposition 3.1): if $w_\mathcal{O}(\mathcal{Q}) \geq v^*$ (sum of core weights ≥ current optimum), the current node cannot improve the solution, and backtracking is justified.
- **Hardening** (Proposition 3.2): if the residual weight of an objective literal plus the total core weight $\geq v^*$, that literal can be propagated as cost-free.

**Proof method**:

- **Proposition 4**: for each core $q$, the clause $C_q = \bigvee_{\ell \in R \cup K} \bar{\ell}$ can be proved via RUP (since the core is discovered by propagation to contradiction).
- **Theorem 5**: starting from core clauses and the solution-improvement constraint, the reason clause $C_\mathcal{Q}$ can be derived in at most $3|\mathcal{O}| + 2|\mathcal{Q}| + 1$ cutting planes steps.

Example derivation: for objective $\mathcal{O} = 3x_1 + 5x_2 + 5x_3 + 6x_4$, cores $q_1 = \langle 3, \{y_1\}, \{\bar{x_1}, \bar{x_2}\}\rangle$ and $q_2 = \langle 5, \{y_2\}, \{\bar{x_3}, \bar{x_4}\}\rangle$:
1. Derive core clauses via RUP.
2. Multiply core clauses by their respective weights and sum.
3. Derive an upper bound on objective literals from the solution-improvement constraint.
4. Add, simplify, and apply division to obtain the final clause.

#### 2. Certifying BDD/MDD Encodings

BDDs are used to encode the solution-improvement constraint $\mathcal{O} \leq v^* - 1$ as CNF. The core challenge is that BDD reducedness implies a single node $\eta = \text{bdd}(k, l, u)$ simultaneously represents two equivalent constraints: $\sum_{i \geq k} v_i b_i \leq l$ (tight) and $\sum_{i \geq k} v_i b_i \leq u$ (loose).

**Definition constraints**: for each node $\eta$, it is necessary to prove:
$$v_\eta \Rightarrow \sum_{i \geq k} v_i b_i \leq l \quad \text{and} \quad v_\eta \Leftarrow \sum_{i \geq k} v_i b_i \leq u$$

**Proposition 10** (key lemma): given the definition constraints of child nodes, the definition constraints of the parent node can be derived via the following steps:
1. Introduce two reification variables $v_\eta$ and $v_\eta'$.
2. Perform case analysis on variable $b_k$.
3. Apply proof by contradiction (using child node definition constraints).
4. Unify the two variables via short cutting planes derivations.

**Proposition 11**: when the BDD skips variables (due to reducedness causing multi-constraint representation), the definition constraints are unified in $6(k-k')+3$ steps.

**MDD generalization**: MDDs branch on a set of variables under at-most-one constraints ($|I|+1$ children instead of 2). The proof of Proposition 10 requires only extending the 2-case analysis to $|I|+1$ cases.

#### 3. Key Rules of the VeriPB Proof System

- **Cutting Planes**: literal axioms, linear combinations, division, saturation.
- **RUP** (Reverse Unit Propagation): if $\mathcal{F} \wedge \neg C$ propagates to contradiction, $C$ may be added.
- **Redundance-based Strengthening**: used for reification and proof by contradiction.
- **Objective Improvement**: adds the solution-improvement constraint upon finding a new solution.

### Loss & Training

No machine learning training is involved. The algorithmic strategy follows the standard MaxCDCL CDCL + look-ahead loop.

## Key Experimental Results

### Main Results

| Metric | Value | Notes |
|--------|-------|-------|
| Benchmark instances | 701 (excluding those unsolvable with proof logging) | MaxSAT Evaluation 2024 |
| Instances unsolvable with proof logging | 6 | Very few instances exceed time limit due to overhead |
| Median proof logging overhead | **19%** | Overhead is manageable for most instances |
| 90th percentile overhead | 4.61× | 10% of instances incur larger overhead |
| Proof checking success rate | 485/695 (69.8%) | 202 timeouts, 8 out-of-memory |
| Median proof checking time | 42.94× solving time | Checking time is the bottleneck |

### Ablation Study

| Configuration | Impact | Notes |
|---------------|--------|-------|
| Number of nodes in MDD definition constraints | Linear in number of objective literals | Significant overhead for instances with many objective literals |
| BDD vs. MDD | MDD provides more compact encoding | But certification cost is also higher |
| RUP vs. explicit derivation | RUP is more concise but checking is slower | Hinted RUP may improve checking |

### Key Findings

1. **Proof logging overhead is acceptable for most instances** (median 19%), but for 10% of instances (typically those with many objective literals) overhead can exceed 4.6×.
2. **Proof checking is the current bottleneck**: the median checking time is 43× the solving time, limiting practical deployment.
3. The derivation of MDD node definition constraints (linear in the number of objective literals) is the primary cause of high overhead on certain instances.
4. This work ensures that **all major MaxSAT solving paradigms** (solution-improving, core-guided, implicit hitting set, branch-and-bound) are now certified under VeriPB.

## Highlights & Insights

1. **Filling the last gap**: Branch-and-Bound was the only remaining uncertified MaxSAT solving paradigm; this work completes the full certification landscape for MaxSAT.
2. **Elegant handling of MDD equivalence proofs**: by constructing definition constraint pairs and propagating them linearly along the BDD, the NP-hard equivalence checking problem is solved efficiently under the specific structural properties of BDDs.
3. **Software engineering value**: proof logging not only guarantees correctness but also serves as a powerful methodology for testing and debugging.
4. **Step count bound of Theorem 5**: $O(|\mathcal{O}| + |\mathcal{Q}|)$, demonstrating that certification incurs only polynomial overhead.

## Limitations & Future Work

1. **Proof checking is too slow** (43× solving time): faster proof checkers (e.g., PBOxide) or proof trimmers are needed.
2. Proof logging overhead for instances with many objective literals is excessive—more compact representations of definition constraints should be explored.
3. Certification of recent techniques such as literal unlocking has not been implemented.
4. Experiments are limited to MaxSAT Evaluation 2024 instances; larger-scale industrial instances have not been tested.
5. Memory usage is not reported in detail.

## Related Work & Insights

- **VeriPB proof system** (Bogaerts et al., 2023) provides a unified proof framework for pseudo-Boolean constraints.
- **Vandesande et al. (2022)** established the methodological foundations for VeriPB-based certification of MaxSAT.
- **Berg et al. (2023, 2024)** certified wlog reasoning in core-guided search and solution-improving search, respectively.
- **Ihalainen et al. (2026)** (concurrent paper) certifies implicit hitting set search.
- WMaxCDCL (Coll et al., 2025) is the current state-of-the-art Branch-and-Bound solver.

## Rating

- Novelty: ⭐⭐⭐⭐ (Certification of MDD encodings is an entirely new contribution; look-ahead certification is also a first.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on the complete MaxSAT Evaluation dataset with quantified overhead.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Formally rigorous, with clear algorithmic pseudocode and helpful examples.)
- Value: ⭐⭐⭐⭐⭐ (Completes the last piece of the MaxSAT certification puzzle; highly significant for safety-critical applications.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[CVPR 2026\] ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](../../CVPR2026/others/enhancing_outofdistribution_detection_with_extende.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](../../ICML2026/others/rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)

</div>

<!-- RELATED:END -->
