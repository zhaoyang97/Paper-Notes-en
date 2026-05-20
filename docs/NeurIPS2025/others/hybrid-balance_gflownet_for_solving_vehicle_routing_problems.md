---
title: >-
  [Paper Note] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems
description: >-
  [NeurIPS 2025][GFlowNet] This paper proposes the Hybrid-Balance GFlowNet (HBG) framework, which for the first time introduces Detailed Balance (DB) in the VRP setting and unifies it with Trajectory Balance (TB)…
tags:
  - "NeurIPS 2025"
  - "GFlowNet"
  - "Vehicle Routing Problem"
  - "Trajectory Balance"
  - "Detailed Balance"
  - "Hybrid Optimization"
date: 2026-05-08
content_hash: 9add2c97dfb443f9
---

# Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems

**Conference**: NeurIPS 2025
**arXiv**: [2510.04792](https://arxiv.org/abs/2510.04792)  
**Code**: [GitHub](https://github.com/ZHANG-NI/HBG)  
**Area**: Others
**Keywords**: GFlowNet, Vehicle Routing Problem, Trajectory Balance, Detailed Balance, Hybrid Optimization

## TL;DR

This paper proposes the Hybrid-Balance GFlowNet (HBG) framework, which for the first time introduces Detailed Balance (DB) in the VRP setting and unifies it with Trajectory Balance (TB), along with a depot-guided inference strategy. HBG consistently and significantly improves two existing GFlowNet-based solvers (AGFN and GFACS) on CVRP and TSP benchmarks.

## Background & Motivation

### State of the Field

**Background**: The Vehicle Routing Problem (VRP) is a core optimization problem in logistics and transportation. Traditional heuristic methods rely on hand-crafted rules and exhibit limited adaptability.

### Limitations of Prior Work

**Limitations of Prior Work**: GFlowNets offer a promising alternative by learning distributions over the solution space to generate diverse, high-quality routes.

### Root Cause

**Key Challenge**: Existing GFlowNet-based VRP solvers (AGFN, GFACS) rely solely on Trajectory Balance (TB) for global optimization, neglecting local optimization signals.

### Starting Point

**Key Insight**: TB's limitation: when an entire route has high cost, even high-quality local transitions (e.g., W→X→Y) receive weak training signals due to poor global reward.

### Supplementary Note

**Supplementary Note**: Detailed Balance (DB) provides step-level local signals, but using DB alone cannot satisfy the global perspective required for VRP.

### Supplementary Note

**Supplementary Note**: Core insight: **global and local optimization are complementary and require a unified framework**.

## Method

### Overall Architecture

HBG augments the TB training of existing GFlowNet solvers (AGFN or GFACS) with an additional DB loss for local optimization. During training, state transition information is recorded at each step; upon completion of a full trajectory, both the TB loss (global) and DB loss (local) are computed and jointly optimized. At inference time, a depot-guided strategy is adopted: sampling is applied only at depot nodes, while greedy selection is used for customer nodes.

### Key Designs

1. **VRP-Specific Detailed Balance (DB)**:
    - Function: Computes a local optimization objective for each state transition in VRP.
    - Mechanism: DB loss $\ell_{DB}(s_t^i, s_{t+1}^i; \theta) = \left(\log \frac{P_f \cdot F(s_t^i) \cdot \exp(\tilde{\mathcal{E}}(s_{t+1}^i))}{P_b \cdot F(s_{t+1}^i) \cdot \exp(\tilde{\mathcal{E}}(s_t^i))}\right)^2$
    - Design Motivation: Enables the model to identify and reinforce high-quality local decisions even from imperfect global trajectories.

2. **Backward Probability Derivation**:
    - Function: Derives closed-form backward probabilities for both TB and DB in CVRP.
    - Mechanism: A complete trajectory consists of $a$ multi-node sub-trajectories and $j$ single-node sub-trajectories; the backward destruction order count is $B(\mathcal{A}_a, \mathcal{J}_j) = (a+j)! \cdot 2^a$.
    - Design Motivation: Ensures theoretical consistency between DB and TB backward probabilities.

3. **Depot-Guided Inference**:
    - Function: Applies sampling only at depot nodes while using greedy selection for customer nodes.
    - Mechanism: Analysis of the backward policy reveals that only depot nodes have flexibility in terms of multiple predecessor choices within the trajectory structure, whereas successors of customer nodes are deterministic.
    - Design Motivation: Restricts stochastic exploration to depot nodes where flexibility exists, reducing ineffective exploration.

### Loss & Training

The total loss is the sum of TB and DB losses:

$$\ell_{HB}(\mathcal{T}; \theta) = \sum_{i=1}^{h} (\ell_{TB}(\tau_i; \theta) + \ell_{DB}(\tau_i; \theta))$$

- The energy term in DB, $\tilde{\mathcal{E}}(s_t^i) = R(s_t^i) - \frac{1}{h}\sum_{k=1}^{h} R(s_t^k)$, measures relative advantage.
- Models are trained on 100-node instances and evaluated on instances with 200/500/1000 nodes.

## Key Experimental Results

### Main Results

CVRP benchmark results (Gap% relative to LKH):

| Method | CVRP200 Gap↓ | CVRP500 Gap↓ | CVRP1000 Gap↓ |
|--------|-------------|-------------|--------------|
| AGFN | 11.48 | 12.21 | 11.15 |
| **HBG-AGFN** | **9.95** | **10.44** | **9.34** |
| GFACS | 23.11 | 23.83 | 23.82 |
| **HBG-GFACS** | **16.48** | **13.53** | **10.61** |
| GFACS+LS | 2.10 | 3.03 | 3.00 |
| **HBG-GFACS+LS** | **1.96** | **2.81** | **2.75** |

### Ablation Study

- **HB module**: Adding HB alone reduces AGFN's Gap by 0.64–1.68 percentage points.
- **Depot-guided inference**: Further reduces Gap by 0.89–1.09 percentage points on top of HB (AGFN).
- **DB vs. TB vs. HB**: Training with DB alone yields the worst results (Gap 18.85–22.72%); TB alone is reasonable (11.15–11.48%); HB achieves the best performance.
- **Larger gains on GFACS**: Gap reduction reaches 55.46% on CVRP1000, as GFACS has greater room for improvement under TB-only training.

### Key Findings

- Improvements grow more pronounced as problem scale increases, indicating strong scalability of HBG.
- Inference overhead is minimal (only 0.01–0.04 seconds), as DB-related parameters are only temporarily loaded.
- Gains are also observed on real-world CVRPLib instances.
- Poor performance of DB alone validates the necessity of a global perspective for VRP.

## Highlights & Insights

- DB is rigorously defined and implemented in the VRP setting for the first time, filling a theoretical gap in GFlowNet-VRP research.
- The backward probability derivation is elegant; the factor of 2 for multi-node sub-trajectories reflects the bidirectional flexibility of depot nodes.
- Depot-guided inference cleverly exploits the structural properties of CVRP, concentrating exploration at meaningful decision points.
- The framework is broadly applicable—it is compatible with both AGFN (constructive) and GFACS (improvement-based) solver paradigms.

## Limitations & Future Work

- Depot-guided inference is only applicable to VRP variants with depot nodes and cannot be directly extended to depot-free problems such as TSP.
- The local reward definition in DB (Euclidean distance between adjacent nodes) is relatively simple; more sophisticated local evaluation criteria merit exploration.
- Comparisons with more advanced learned solvers (e.g., LLM-based methods) are not conducted.
- Training is performed solely on synthetic data; performance on real-world logistics datasets requires further validation.

## Related Work & Insights

- HBG is complementary to learning-based VRP solvers such as DeepACO and NeuOpt, providing improvements internal to the GFlowNet optimization process.
- The TB/DB hybrid idea is generalizable to other sequential decision-making applications of GFlowNets (e.g., molecular generation, causal discovery).
- Balancing local and global optimization is a universal challenge in combinatorial optimization; HBG offers an elegant unified perspective.

## Rating

- ⭐⭐⭐⭐ — Theoretically rigorous with consistent and significant improvements, though the problem scope is relatively narrow and generalizability remains partially unverified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](../../AAAI2026/others/or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](../../ICLR2026/others/evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](../../ICCV2025/others/revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)
- [\[ICCV 2025\] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding](../../ICCV2025/others/hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](../../AAAI2026/others/certified_branch-and-bound_maxsat_solving_extended_version.md)

</div>

<!-- RELATED:END -->
