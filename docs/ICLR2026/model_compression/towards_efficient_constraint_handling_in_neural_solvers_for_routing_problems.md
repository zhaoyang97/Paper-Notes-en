---
title: >-
  [Paper Note] Towards Efficient Constraint Handling in Neural Solvers for Routing Problems
description: >-
  [ICLR 2026][Model Compression][Neural Combinatorial Optimization] This paper proposes the Construct-and-Refine (CaR) framework, which achieves efficient feasibility repair through joint training of a construction module…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Neural Combinatorial Optimization"
  - "Vehicle Routing Problem"
  - "Constraint Handling"
  - "Construct-and-Refine Hybrid"
  - "Feasibility Repair"
date: 2026-05-08
content_hash: 5db12a5ba6f3ce2f
---

# Towards Efficient Constraint Handling in Neural Solvers for Routing Problems

**Conference**: ICLR 2026
**arXiv**: [2602.16012](https://arxiv.org/abs/2602.16012)  
**Code**: [https://github.com/jieyibi/CaR-constraint](https://github.com/jieyibi/CaR-constraint)  
**Area**: Model Compression
**Keywords**: Neural Combinatorial Optimization, Vehicle Routing Problem, Constraint Handling, Construct-and-Refine Hybrid, Feasibility Repair

## TL;DR

This paper proposes the Construct-and-Refine (CaR) framework, which achieves efficient feasibility repair through joint training of a construction module and a lightweight refinement module. CaR provides the first general and efficient neural constraint-handling solution for hard-constrained routing problems, substantially outperforming classical and neural SOTA solvers on TSPTW and CVRPBLTW.

## Background & Motivation

- **VRP Constraint Challenges**: Real-world vehicle routing problems involve complex constraints (time windows, capacities, backhauls, etc.), yet existing neural solvers primarily focus on simpler variants.
- **Limitations of Existing Constraint-Handling Approaches**:
  1. **Feasibility Masking**: Excludes invalid actions in the MDP — effective for simple VRPs, but mask computation itself is NP-hard in complex settings (e.g., TSPTW), or overly restricts the search space (e.g., >60% of nodes filtered in CVRPBLTW).
  2. **Implicit Feasibility Awareness**: Guides feasibility implicitly via reward penalties or feature augmentation — limited effectiveness with high computational overhead.
- **Limitations of Hybrid Solvers**: Existing construct-and-refine methods (RL4CO, NCS) are designed for optimization objectives rather than feasibility, requiring thousands of refinement steps (e.g., 5,000 steps) and may yield 100% infeasible solutions on hard-constrained VRPs.
- **Core Problem**: Can a simple, general, and efficient neural constraint-handling framework be designed?

## Method

### Overall Architecture

CaR unifies a construction module and a refinement module through end-to-end joint training:
1. The construction module generates diverse, high-quality initial solutions.
2. The lightweight refinement module rapidly repairs infeasibilities (only 10 steps vs. the conventional 5,000).
3. Refinement outcomes are fed back to guide the construction policy.

### Key Design 1: Joint Training for Feasibility Repair

**Construction Policy Loss**: Policy gradient via REINFORCE:

$$\mathcal{L}_{\text{RL}}^{\text{C}} = \frac{1}{S} \sum_{i=1}^{S} \left[\left(\mathcal{R}(\tau_i) - \frac{1}{S}\sum_{j=1}^{S}\mathcal{R}(\tau_j)\right) \log \pi_{\theta}^{\text{C}}(\tau_i)\right]$$

**Auxiliary Diversity Loss**: Encourages policy exploration:

$$\mathcal{L}_{\text{DIV}} = -\sum_{t=1}^{|\tau|} \pi_{\theta}^{\text{C}}(e_t \mid \tau_{<t}) \log \pi_{\theta}^{\text{C}}(e_t \mid \tau_{<t})$$

**Supervised Learning Loss**: Uses improved solutions as pseudo-labels when they outperform the original:

$$\mathcal{L}_{\text{SL}} = -\mathbb{I} \cdot \sum_{t=1}^{|\tau^*|} \log \pi_{\theta}^{\text{C}}(e_t^* \mid \tau_{<t}^*)$$

**Refinement Policy**: Short-horizon CMDP ($T_R = 10$ steps), optimized via REINFORCE to improve solution quality at each step.

**Joint Loss**:

$$\mathcal{L}(\theta) = \mathcal{L}(\theta^{\text{C}}) + \omega \mathcal{L}(\theta^{\text{R}})$$

### Key Design 2: Cross-Paradigm Representation Learning

**Shared Encoder**: Both construction and refinement share a 6-layer Transformer encoder.
- Input features $f_i^{\text{n}} = \{x_i, y_i, q_i, l_i, u_i, \ell\}$ (coordinates, demand, time windows, duration).
- Refinement additionally encodes solution structure via circular positional encoding.
- An MLP fuses node-level attention scores $a^{\text{n}}$ and solution-level scores $a^{\text{s}}$.

**Separate Decoders**: Construction and refinement maintain independent decoders (ablations confirm that a unified decoder degrades performance).

### Key Design 3: Flexible Constraint Handling

CaR integrates three complementary constraint-handling strategies:
1. **Flexible Masking**: Applied when computable; optionally relaxed (flexible masking).
2. **Implicit Awareness via Shared Representation**: Cross-paradigm representation learning enhances constraint awareness.
3. **Explicit Feasibility Repair**: The lightweight refinement process directly corrects infeasible solutions.

### Constraint Violation Cost

$$C(\tau) = \sum_{e_{ij} \in \tau} \left[C_L(e_{ij}) + \sum_{\eta=1}^{m} C_V^{\eta}(e_{ij})\right]$$

This combines the objective cost $C_L$ (tour length) and constraint violation penalties $C_V$ (e.g., time window excess).

## Key Experimental Results

### Main Results — TSPTW (NP-hard mask computation)

| Method | n=50 Gap↓ | n=50 Infsb%↓ | n=50 Time | n=100 Gap↓ | n=100 Infsb%↓ |
|--------|-----------|-------------|-----------|-----------|--------------|
| LKH-3 (100 trials) | 0.004% | 11.88% | 7m | 0.103% | 31.05% |
| PIP (neural SOTA) | — | — | — | — | — |
| NCS | — | 100% | — | — | 100% |
| **CaR** | **best** | **best** | **6–8× faster** | **best** | **best** |

- CaR achieves a 6–8× speedup while improving both feasibility and solution quality.
- CaR finds feasible solutions that LKH-3 fails to identify.

### Main Results — CVRPBLTW (multiple constraints, masking over-restrictive)

CaR demonstrates dominant performance under the four-constraint combination of capacity, backhaul, duration, and time windows, outperforming both classical and neural SOTA solvers.

### Ablation Study

| Component | Effect |
|-----------|--------|
| Joint training vs. separate training | Joint training significantly improves feasibility and solution quality |
| Shared encoder vs. independent encoders | Shared encoder yields the largest gains on complex constraints (CVRPBLTW) |
| Diversity loss $\mathcal{L}_{\text{DIV}}$ | Compensates for diversity loss incurred by removing multi-start strategies |
| Supervised loss $\mathcal{L}_{\text{SL}}$ | Leverages refinement feedback to improve construction quality |
| 10 vs. 5,000 refinement steps | 10 steps suffice for effective repair, reducing runtime from hours to minutes/seconds |

### Generality Verification

- **Construction backbones**: POMO and PIP are both compatible with CaR.
- **Refinement backbones**: NeuOpt (k-opt) and N2S (remove-and-reinsert) are both supported.
- **Problem variants**: Applicable to both simple VRP (CVRP) and complex VRP (TSPTW, CVRPBLTW).

## Highlights & Insights

- **First General Framework**: CaR is the first unified neural constraint-handling framework applicable to both simple and complex VRPs.
- **Remarkable Efficiency**: Refinement steps are compressed from 5,000 to 10, reducing runtime from hours to minutes.
- **First Explicit Feasibility Repair in Neural Solvers**: Introduces learnable feasibility repair to the neural solver literature, filling a critical gap.
- **Cross-Paradigm Synergy**: The shared encoder enables knowledge transfer between construction and refinement, with pronounced advantages in complex constraint settings.

## Limitations & Future Work

- Joint training increases implementation complexity, requiring simultaneous management of two MDPs (construction and refinement).
- The shared encoder provides limited benefit on simple VRPs (marginal gains on CVRP).
- The choice of refinement operators (k-opt, remove-and-reinsert) requires adaptation for different VRP variants.
- Validation is currently limited to $n \leq 100$; scalability to larger instances ($n > 1000$) remains unexplored.
- Training relies on synthetically generated instances drawn from uniform distributions; generalization to real-world distributions has not been fully verified.

## Related Work & Insights

- **Construction Solvers**: AM, POMO, PIP — incrementally build solutions and handle simple constraints via masking.
- **Improvement Solvers**: NeuOpt, N2S — iteratively refine existing solutions, requiring a large number of steps.
- **Hybrid Solvers**: LCP, RL4CO, NCS — combine both paradigms but primarily target optimality on simple VRPs.
- **Constraint Handling**: Lagrangian relaxation (Tang et al.), constraint features (MUSLA), approximate masking (PIP).
- **CaR**: The first unified framework integrating flexible masking, implicit awareness, and explicit repair.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★☆ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★★ |
| Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver](../../NeurIPS2025/model_compression/mtl-kd_multi-task_learning_via_knowledge_distillation_for_generalizable_neural_v.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
