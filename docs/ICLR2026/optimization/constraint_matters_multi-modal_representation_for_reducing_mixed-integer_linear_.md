---
title: >-
  [Paper Note] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming
description: >-
  [ICLR2026][Optimization][MILP] This paper proposes a constraint-reduction framework for simplifying MILP models. It defines a fixed-constraint strength $\rho$ and uses information gain $\Delta H = -\log\rho$ to identify critical tight constraints (CTCs). A multi-modal GNN combining an instance-level bipartite graph with an abstract-level type graph is designed to predict CTCs. On four large-scale benchmarks, the method achieves an average improvement of 51.06% in solution quality ($\text{gap}_\text{abs}$) and an average speedup of 17.47% in convergence (PDI).
tags:
  - ICLR2026
  - Optimization
  - MILP
  - constraint reduction
  - tight constraint
  - multi-modal representation
  - GNN
date: 2026-05-08
content_hash: 4458218ec5fc5fa3
---

# Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming

**Conference**: ICLR2026
**arXiv**: [2508.18742](https://arxiv.org/abs/2508.18742)
**Code**: [https://github.com/Liwow/Constraint_Matters](https://github.com/Liwow/Constraint_Matters)
**Area**: Optimization / Theory
**Keywords**: MILP, constraint reduction, tight constraint, multi-modal representation, GNN

## TL;DR

This paper proposes a constraint-reduction framework for simplifying MILP models. It defines a fixed-constraint strength $\rho$ and uses information gain $\Delta H = -\log\rho$ to identify critical tight constraints (CTCs). A multi-modal GNN combining an instance-level bipartite graph with an abstract-level type graph is designed to predict CTCs. On four large-scale benchmarks, the method achieves an average improvement of 51.06% in solution quality ($\text{gap}_\text{abs}$) and an average speedup of 17.47% in convergence (PDI).

## Background & Motivation

**Background**: Large-scale MILP problems are prevalent in industrial settings such as production scheduling, supply chain management, and energy systems. Exact solvers like Gurobi and SCIP incur prohibitive computational costs on large instances, making ML-based model reduction (predicting and fixing subsets of variables) the dominant acceleration strategy.

**Limitations of Prior Work**:

- Variable reduction methods (Predict-and-Search, ConPaS, etc.) can produce infeasible solutions when predictions are incorrect.
- Constraint reduction (converting inequality constraints to equality) has received virtually no systematic study.
- Naively fixing all tight constraints yields highly unstable results: on the CA_easy task, the best fixing requires only 1.85s vs. 465.74s for the worst case (vs. 378.23s for the original).

**Key Challenge**: Different tight constraints contribute vastly different amounts to solver acceleration — random fixing may actually slow down solving, necessitating a principled selection strategy.

**Key Insight**: From the perspective of duality theory, constraints and variables are naturally coupled; fixing a tight constraint is equivalent to reducing the feasible region. The key question becomes: which constraints are most worth fixing, and how can this be predicted efficiently?

## Method

### Overall Architecture

A three-stage pipeline: (1) **Multi-modal representation** — inter-level message passing between an instance bipartite graph and an abstract type bipartite graph for feature extraction; (2) **CTC identification** — 10 prototype constraint types ranked by information gain, with high-gain subsets selected as training labels; (3) **Model reduction** — GNN predicts CTCs → fixes top-$k_c$ constraints + trust-region tolerance $\Delta_c$ → feeds into the solver.

### Key Designs

1. **Critical Tight Constraint Identification (TCP Module)**: Defines fixed-constraint strength $\rho(C_i) = |S_{\hat{C}_i}| / |S_{C_i}|$, measuring the reduction ratio of the local feasible region upon fixing all tight constraints of type $C_i$. A larger information gain $\Delta H_{C_i} = -\log\rho$ indicates a greater contribution to solver acceleration and thus higher fixing priority. Specifically, 10 prototype types are selected from the 17 basic constraint types in MIPLIB, and the strength of each type is estimated: Set Packing exhibits weak strength $\rho = n/(n+1) \to 1$ (low information gain), while Knapsack/Bin Packing exhibit strong strength $\rho = O(1/A\sqrt{n})$ (high information gain), and the latter are prioritized.

2. **Multi-Modal GNN Representation**: The lower level is a standard instance bipartite GNN (variable–constraint nodes with the coefficient matrix as edge weights); the upper level is an abstract type bipartite graph (nodes represent variable/constraint types, initialized with PLM-encoded text descriptions). After independent intra-level message passing, the two levels are fused via Cross-Attention — abstract type nodes query the corresponding instance node features — and a gating mechanism $\alpha = \text{Gate}(\bar{h} \| \hat{h})$ dynamically balances the two levels before propagating information back to the instance nodes.

3. **Trust-Region Tolerance**: After predicting the top-$k_c$ constraints, auxiliary variables $z_i \in \{0, 1\}$ are introduced to allow at most $\Delta_c$ constraints to violate the equality requirement, preventing infeasibility due to prediction errors. This guarantees that the reduced feasible region remains a non-empty subset of the original.

### Loss & Training

Focal Loss (rather than cross-entropy) is employed to mitigate selective learning bias: neural networks tend to make high-confidence predictions on easy (typically non-critical) constraints, and Focal Loss downweights such easy samples via $(1 - \hat{y})^\gamma$. The total loss is $\mathcal{L} = \lambda \mathcal{L}^{\text{sol}}_{\text{Focal}} + (1-\lambda) \mathcal{L}^{\text{con}}_{\text{Focal}}$, supervising variable prediction and constraint prediction respectively.

## Key Experimental Results

### Main Results (800s time limit, 100 test instances)

| Method | CA $\text{gap}_\text{abs}$↓ | CA PDI↓ | MIS $\text{gap}_\text{abs}$↓ | MIS PDI↓ | MVC $\text{gap}_\text{abs}$↓ | MVC PDI↓ | WA $\text{gap}_\text{abs}$↓ | WA PDI↓ |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| Gurobi | 477.51 | 76.29 | 2.70 | 33.09 | 8.38 | 68.02 | 0.20 | 4.76 |
| PS+Gurobi | 210.19 | 73.78 | 0.04 | 0.69 | 0.28 | 7.39 | 0.07 | 4.97 |
| ConPaS+Gurobi | 197.36 | 73.75 | 0.02 | 0.71 | 0.10 | 7.64 | 0.10 | 4.97 |
| **Ours+Gurobi** | **104.72** | **73.02** | **0.00** | **0.47** | **0.00** | **6.26** | **0.06** | **4.40** |
| Gain (%) | 77.06% | 4.22% | 100% | 98.58% | 100% | 90.80% | 70.00% | 7.56% |
| SCIP | 4005.99 | 190.18 | 4.16 | 103.91 | 12.42 | 140.89 | 2.60 | 37.06 |
| PS+SCIP | 3575.18 | 179.48 | 0.66 | 6.49 | 2.98 | 21.03 | 1.27 | 34.85 |
| ConPaS+SCIP | 3545.92 | 146.88 | 0.43 | 5.52 | 2.56 | 24.05 | 1.20 | 29.91 |
| **Ours+SCIP** | **3401.63** | **109.41** | **0.34** | **4.65** | **1.48** | **18.34** | **0.83** | **21.52** |
| Gain (%) | 15.08% | 42.47% | 91.83% | 95.52% | 88.08% | 86.98% | 68.08% | 41.93% |

On Gurobi, $\text{gap}_\text{abs}$ for MIS and MVC drops to 0 (optimal solutions found exactly); on SCIP, the maximum PDI improvement reaches 42.47% (CA dataset).

### Generalization and Real-World Scenarios

| Method | CA (large-scale) $\text{gap}_\text{abs}$↓ | CA PDI↓ | MVC (large-scale) $\text{gap}_\text{abs}$↓ | MVC PDI↓ | MMCN (real-world) $\text{gap}_\text{abs}$↓ | MMCN PDI↓ |
|------|---:|---:|---:|---:|---:|---:|
| SCIP | 7009.58 | 198.78 | 22.07 | 228.21 | 13819.90 | 427.63 |
| PS | 6541.87 | 169.86 | 4.07 | 39.63 | 8084.36 | 329.60 |
| ConPaS | 6287.22 | 156.95 | 2.20 | 35.39 | 6064.85 | 330.05 |
| **Ours** | **5955.86** | **131.44** | **0.55** | **24.54** | **3547.08** | **246.89** |

On the real-world logistics network benchmark MMCN, $\text{gap}_\text{abs}$ is reduced from 13819.90 (SCIP) to 3547.08 (↓74.3%), validating practical applicability.

### Sensitivity Analysis: Motivation for Constraint Selection

| Setting | CA_easy Time (s) | WA Time (s) |
|------|---:|---:|
| Original solving | 378.23 | >3600 |
| Best constraint fixing | 1.85 | 50.73 |
| Worst constraint fixing | 465.74 | >3600 |

The gap between best and worst fixing can reach 250×, directly demonstrating that *which* constraints are fixed is of critical importance.

## Highlights & Insights

- **A new dimension in constraint reduction**: This is among the first systematic studies of constraint reduction in ML4Optimization, complementing variable reduction — ablation experiments confirm that combining both yields the best performance.
- **Information-theoretic principled design**: $\rho$ and $\Delta H$ provide theoretically grounded measures of constraint importance, as opposed to heuristic judgment.
- **Novel "multi-modal" framing**: The abstract MILP model (constraint type semantics) is treated as a distinct modality and fused with the instance coefficient matrix modality — a perspective generalizable to other structured optimization problems.
- **Fixing only 5% of constraints achieves significant acceleration**: Low prediction overhead with high returns.

## Limitations & Future Work

- The local decoupling assumption (each constraint type independently affects the feasible region) does not fully hold when constraints are highly coupled.
- CTC annotation relies on oracle optimal solutions, making data acquisition costly.
- Evaluation is limited to binary integer variables; extension to general integer encodings has not been explored.
- The method cannot be directly applied to standard benchmarks such as MIPLIB, which lack explicit mathematical constraint descriptions.

## Related Work & Insights

- **vs. Variable Reduction (PS/ConPaS)**: This work opens a complementary direction via constraint reduction from a dual perspective; the two approaches are mutually beneficial.
- **vs. Bertsimas & Stellato (2022)**: The latter predicts all tight constraints simultaneously, posing a high-dimensional prediction challenge; this paper selects only a critical subset.
- **vs. Gasse et al. (2019) GNN representation**: This work augments the representation with an abstract type graph and PLM text embeddings, enabling stronger cross-instance generalization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — innovative combination of constraint reduction, information theory, and multi-modal representation
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets + real-world scenario + generalization + ablation
- Writing Quality: ⭐⭐⭐⭐ — motivation-driven experimental narrative with clear theoretical analysis
- Value: ⭐⭐⭐⭐⭐ — opens a new direction of constraint reduction for ML4Optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)
- [\[ICCV 2025\] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer](../../ICCV2025/optimization/addressing_representation_collapse_in_vector_quantized_models_with_one_linear_la.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[ICLR 2026\] Rethinking Consistent Multi-Label Classification Under Inexact Supervision](rethinking_consistent_multi-label_classification_under_inexact_supervision.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)

</div>

<!-- RELATED:END -->
