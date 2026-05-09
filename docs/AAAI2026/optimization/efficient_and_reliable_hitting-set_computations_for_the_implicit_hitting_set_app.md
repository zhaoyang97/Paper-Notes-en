---
title: >-
  [Paper Note] Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach
description: >-
  [AAAI2026][Optimization][implicit hitting set] To address numerical instability arising from the reliance on commercial IP solvers in the hitting-set component of the implicit hitting set (IHS) framework, this paper proposes alternative approaches based on pseudo-Boolean (PB) reasoning and stochastic local search (SLS), as well as hybrid strategies. The work realizes the first certifiable IHS computation and demonstrates effective trade-offs between efficiency and reliability across 1,786 benchmark instances.
tags:
  - AAAI2026
  - Optimization
  - implicit hitting set
  - pseudo-Boolean optimization
  - certifiable computation
  - stochastic local search
  - IP solver
date: 2026-05-08
content_hash: 17b2aa555825d3b5
---

# Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach

**Conference**: AAAI2026
**arXiv**: [2508.07015](https://arxiv.org/abs/2508.07015)
**Code**: [https://bitbucket.org/coreo-group/pbhs](https://bitbucket.org/coreo-group/pbhs)
**Area**: Combinatorial Optimization / Formal Verification
**Keywords**: [implicit hitting set, pseudo-Boolean optimization, certifiable computation, stochastic local search, IP solver]

## TL;DR

To address numerical instability arising from the reliance on commercial IP solvers in the hitting-set component of the implicit hitting set (IHS) framework, this paper proposes alternative approaches based on pseudo-Boolean (PB) reasoning and stochastic local search (SLS), as well as hybrid strategies. The work realizes the first certifiable IHS computation and demonstrates effective trade-offs between efficiency and reliability across 1,786 benchmark instances.

## Background & Motivation

**Background** The implicit hitting set (IHS) method is a general-purpose framework for solving NP-hard combinatorial optimization problems, with successful practical applications in MaxSAT, constraint optimization, and pseudo-Boolean optimization (PBO). IHS iteratively alternates between a decision oracle that extracts sources of inconsistency (cores) and an optimizer that computes hitting sets over these cores.

**Limitations of Prior Work** Hitting-set optimizers are almost exclusively implemented via integer programming (IP) solvers. However, commercial IP solvers use floating-point arithmetic by default, which can produce incorrect solutions due to numerical instability. Experiments confirm that Gurobi incorrectly claimed optimality nine times across four instances, and exact SCIP produced four erroneous results.

**Key Challenge** There is a fundamental trade-off between efficiency and reliability: floating-point IP solvers are fastest but unreliable, while exact IP solvers are reliable but reduce the number of solved instances by more than 15% and double memory consumption. More critically, existing IHS frameworks provide no independently verifiable correctness certificates for their computed results.

**Goal** The paper aims to provide certifiable correctness guarantees for IHS computations while maintaining acceptable performance.

**Key Insight** The paper exploits recent advances in conflict-driven pseudo-Boolean solvers to explore PB-reasoning-based alternatives for hitting-set computation, and investigates hybrid combinations with IP solving and stochastic local search.

**Core Idea** Through a carefully designed combination of PB reasoning, SLS, and IP solving, the paper realizes the first certifiable IHS computation framework, achieving verifiable correctness guarantees with controlled efficiency loss.

## Method

### Overall Architecture

The PBO-IHS algorithm iterates between a core set $\mathcal{K}$ and an objective function $O$: in each round, it calls Solve-HS to compute an optimal (or sub-optimal) solution $\gamma$ over the current cores as a lower bound, then calls Extract-Cores to extract new cores from the original formula $F$ under the assumption $\gamma$. The algorithm terminates when the upper bound equals the lower bound. This paper focuses on improving the Solve-HS component, proposing a unified algorithmic framework (Algorithm 2) that flexibly combines IP, PB optimization, and SLS.

### Key Designs

1. **Instantiating Solve-HS with Multiple PB Optimization Algorithms**

   - **Function**: Replace the IP solver with conflict-driven pseudo-Boolean optimization to compute optimal solutions over the core set.
   - **Mechanism**: Three PB optimization variants are proposed—Solution Improvement Search (SIS, maintaining upper bound $ub_c$ and solving $\mathcal{K} \wedge (O < ub_c)$ each round), Core Guided search (CG, maintaining a reconstructed objective $O^R$ and driving search via cores), and Core Boosted search (CB, a hybrid of CG and SIS). To support incremental invocations, SIS introduces a reified constraint $r_{ub_c} \Rightarrow (\gamma(O) < ub_c)$, avoiding solver reconstruction on each call.
   - **Design Motivation**: PB solvers are based on exact arithmetic reasoning, eliminating floating-point numerical issues, and natively support generation of correctness certificates in VeriPB format.

2. **Hybrid Strategies and SLS Integration**

   - **Function**: Combine stochastic local search (SLS) and proof-generating PB solvers with IP solvers to minimize the number of invocations of the proof generator.
   - **Mechanism**: Three hybrid strategies are proposed—OptLB (invokes the PB solver for verification only when the lower bound equals the upper bound), AllLB (verifies on every lower-bound update), and ForceLB (distinguishes between calls that require optimality and those that do not). SLS is implemented via an adapted NuPBO supporting incremental calls, with a diversification mechanism (two-phase search: repair of the previous solution followed by random restarts) that ensures solution diversity via $\text{argmax}_i \cos(z_t, z_i)$.
   - **Design Motivation**: Non-optimal invocations do not require proofs; the OptLB strategy invokes the more expensive PB solver only at critical moments, minimizing the overhead of certification.

### Loss & Training

This paper concerns combinatorial optimization solver design and does not involve a traditional loss function. The core optimization objective is the pseudo-Boolean optimization problem $(F, O)$: minimize $O = \sum_i w_i \ell_i + lb$ subject to all PB constraints $\sum_i a_i \ell_i \geq A$ in formula $F$. Proof generation follows the VeriPB format, using cutting-plane operations and strengthening rules, with support for introducing reified variables.

## Key Experimental Results

### Main Results

| Configuration | Instances Solved | Avg. Memory (GB) |
|---|---|---|
| Gurobi (floating-point IP) | 951 | 0.66 |
| SCIP (floating-point IP) | 913 | 1.16 |
| Exact SCIP (exact IP) | 762 | 2.10 |
| Core Guided (PB) | 687 | 0.63 |
| SIS + CB (PB) | 716 | 2.62 |
| OptLB hybrid (Gurobi+SIS+CB) | 746 | 1.46 |

### Ablation Study

| Configuration | Instance Change | Memory Change (GB) | Exclusive Instances |
|---|---|---|---|
| Gurobi + SLS | +17 | −0.02 | 31 |
| Exact SCIP + SLS | +44 | −0.48 | 58 |
| SIS+CB + SLS | +19 | −0.53 | 53 |
| OptLB + CG + SLS | +11 | −0.02 | 21 |

### Key Findings

- Gurobi incorrectly claimed optimality nine times across four instances; Exact SCIP also produced four erroneous results, confirming the real-world risk of floating-point numerical instability.
- The median runtime overhead of proof logging is only 10.7%, with a 90th-percentile overhead of 100%.
- SLS integration yields the largest gain for Exact SCIP (+44 instances) while reducing memory by nearly 20%.
- The OptLB hybrid strategy achieves the best balance between efficiency and certifiability: among the 746 instances solved, 741 have generated proofs.

## Highlights & Insights

- This work realizes the first fully certifiable computation within the IHS paradigm, filling a gap in the application of VeriPB to this domain.
- The layered certification strategy (OptLB) minimizes certification overhead by invoking the proof generator only when an optimal solution is confirmed.
- The complementarity between SLS and exact solvers is found to be particularly strong: a large number of instances are solved exclusively by one approach but not the other.
- A general PB proof-logging API is developed and shared across two solvers via the same proof interface, with broad anticipated reuse.

## Limitations & Future Work

- PB solvers used in isolation still lag noticeably behind commercial IP solvers (687–716 vs. 951 instances).
- The median proof-checking time is 588% of the solving time, making checker efficiency a bottleneck.
- Evaluation is limited to the PBO domain; extension to other IHS instantiations such as MaxSAT has not yet been explored.
- The SLS diversification strategy remains susceptible to local optima; further improvements to SLS heuristics present an opportunity for additional gains.

## Related Work & Insights

This paper sits at the intersection of several research threads: IHS methods for MaxSAT, the VeriPB proof format, and NuPBO-based local search. The key insight for the formal verification community is that hybrid strategies can achieve certifiability with negligible efficiency loss, and this "trust but verify" paradigm deserves broader adoption in solver design. The work also has direct implications for the certified track of PBO competitions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First certifiable IHS computation; novel hybrid strategy design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 1,786 benchmark instances with systematic comparison across multiple configurations.
- **Writing Quality**: ⭐⭐⭐⭐ Technical details are clear; algorithmic descriptions are well-structured.
- **Value**: ⭐⭐⭐⭐ Significant engineering and theoretical value for trustworthy AI and verifiable computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](../../NeurIPS2025/optimization/set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)

</div>

<!-- RELATED:END -->
