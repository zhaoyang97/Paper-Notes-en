---
title: >-
  [Paper Note] SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs
description: >-
  [ICML2026][LLM Pretraining][Fault Masking] SPARe cyclically stacks $r$ layers of data shards across groups in the data-parallel dimension. After node failures…
tags:
  - "ICML2026"
  - "LLM Pretraining"
  - "Fault Masking"
  - "Data Parallelism"
  - "Redundant Computation"
  - "Adaptive Reordering"
  - "Checkpointing"
date: 2026-05-08
content_hash: 5d527d9d09defd81
---

# SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs

**Conference**: ICML2026  
**arXiv**: [2603.00357](https://arxiv.org/abs/2603.00357)  
**Code**: https://github.com/padsysl/SPARe  
**Area**: LLM Pretraining / Fault-Tolerant Systems / Distributed Parallelism  
**Keywords**: Fault Masking, Data Parallelism, Redundant Computation, Adaptive Reordering, Checkpointing

## TL;DR
SPARe cyclically stacks $r$ layers of data shards across groups in the data-parallel dimension. After node failures, it employs Hopcroft-Karp + min-cost max-flow for adaptive reordering of the "all-reduce stack count." In a 600k GPU restart-dominant scenario, it achieves availability equivalent to $r\times$ traditional replication with only $2\sim 3\times$ computational overhead, reducing time-to-train by $40\sim 50\%$ compared to Rep+CKPT.

## Background & Motivation

**Background**: Current frontier LLM pretraining clusters have reached the $10^5$ GPU scale (e.g., Llama-3 16k H100, future 600k H100). Primary fault-tolerance methods include: checkpointing (GEMINI, Just-in-Time, Universal CKPT), partial recovery (communicator shrink), and replication ($r$ redundant compute copies per group).

**Limitations of Prior Work**: As cluster scales expand, MTBF decreases according to $\mathcal{O}(1/\#\mathrm{GPU})$, while the cost of NCCL_init and collective communication for each global restart increases linearly with $\#\mathrm{GPU}$. Llama-3 reported one failure every 3 hours for 16k GPUs; projected to 96k, this becomes 30 min, and at 600k, it is 5 min. Since a global restart on 600k GPUs takes 60 min, systems enter a **restart-dominant regime** where downtime dominates. Checkpointing only reduces rework waste but cannot decrease the number of restarts; traditional replication maintains availability but incurs an unbearable $20\times$ computational cost when $r=20$.

**Key Challenge**: The hard trade-off between availability gains (requiring high redundancy $r$) and computational overhead (increasing linearly with $r$).

**Goal**: To implement a failure-masking scheme at the data-parallel layer with "redundant but near-constant overhead," independent of model architecture and inner parallelism topologies (TP/PP/EP).

**Key Insight**: It is not necessary to complete all computations for every group before triggering an all-reduce—as long as at least one surviving group has computed each of the $N$ shard types, the gradients can be aggregated. Therefore, "redundancy" is placed at the shard level rather than the group level, allowing for early all-reduce by "skipping layers in the stack."

**Core Idea**: $N$ data shards $\{D_0,\dots,D_{N-1}\}$ are stacked in $r$ layers across $N$ model-parallel groups using cyclic rotation (each group holds $r$ different shards). During training, all-reduce is triggered once the minimum stack count $c(k)=\lceil N/(N-k)\rceil$ required to "collect all types" is reached. Upon node failure, HK + MCMF algorithms are used for **adaptive reordering** of the stack sequence to ensure this minimum stack count remains achievable.

## Method

### Overall Architecture
SPARe is built entirely on synchronous Data Parallelism: $N$ model-parallel groups, each with $M$ GPUs holding one model replica, forming $M$ DP communicators with a world size of $N$. Modifications are restricted to the **set of shards assigned to each group** and the **all-reduce triggering timing**:

- **Static Layout**: Each group $g_i$ sequentially holds a shard stack $[D_i, D_{i+1}, \dots, D_{i+r-1}]$ (mod $N$). This ensures any two shard types overlap at most once across all groups (independence condition, derived in Thm. 4.1).
- **Dynamic Scheduling**: Each training step maintains an `all-reduce stack` count $S$ (initial value 1), representing the current layer to be computed before all-reduce. When a failure occurs, the ReCtlr decides whether to reorder and update $S$. Surviving groups then perform "patch compute" for missing types, shrink the communicator, all-reduce, and update parameters.

### Key Designs

1. **Stack & Early all-reduce Triggering**:
    - **Function**: Transitions data redundancy from "each group computes $r$ copies" to "each group maintains $r$ copies but only computes the first $S$," reducing $r\times$ overhead to $\approx S\times$.
    - **Mechanism**: At the $k$-th failure, the minimum stack lower bound for $N-k$ surviving groups to cover $N$ shard types is $c(k)=\lceil N/(N-k)\rceil$. SPARe sets $S$ based on this and triggers all-reduce immediately after the $S$-th layer. If the ring all-reduce does not hang, the update proceeds; otherwise, it enters the ReCtlr failure handling process. This is transparent to model layers: reordering only changes "who provides type $i$," while gradients remain $\bar{\mathbf g}=\frac{1}{N}\sum_i \mathbf g_i$, leaving optimizer states and updates unchanged.
    - **Design Motivation**: Traditional replication is expensive because redundancy is coupled with mandatory execution. Decoupling these allows for high redundancy $r$ with minimal overhead in common cases where $k$ is small.

2. **ReCtlr: HK-Fixed / HK-Free / MCMF Three-Stage Reordering**:
    - **Function**: Determines if the current $S$ can still cover all types after a failure; if not, it finds a new minimum $S^\star$ and reorders the stack with minimal movement.
    - **Mechanism**: The mapping from "$N$ shard types → first $S$ layers of surviving groups" is modeled as a bipartite graph. **HK-Fixed** (Hopcroft-Karp) first checks if a perfect matching exists in the current layout. If so, it returns with zero movement. Otherwise, Phase 1 runs **HK-Free** (allowing arbitrary stack permutations within groups) while incrementing $S$ from $S_0$ to find the minimum feasible $S^\star$. If $S$ exceeds $r$, a wipe-out is declared, triggering a global restart. Phase 2 uses **MCMF** (min-cost max-flow) on the surviving groups × stack slots to find a reordering that satisfies $S^\star$ with minimal displacement cost. These algorithms are lightweight for $N\sim 10^{2\sim 3}$ (approx. 0.1 s).
    - **Design Motivation**: HK-Fixed hits directly in 90%+ of steps, avoiding unnecessary reordering. The combination of HK-Free + MCMF guarantees theoretical feasibility while suppressing data movement.

3. **SPARe+CKPT Joint Optimization**:
    - **Function**: Couples SPARe with Young & Daly style checkpointing to solve for the optimal $r^\star$ and checkpoint interval $T_c^\star$ to minimize time-to-train.
    - **Mechanism**: The normalized time-to-train is defined as $J(r)=\bar S(N,r)/A^\star(\mu(N,r)\, m)$, where $\bar S(N,r)\approx \frac{1}{\lfloor\mu\rfloor}\sum_{k=0}^{\lfloor\mu\rfloor-1}(c(k)+\rho_k)$ is the average compute overhead and $\mu(N,r)\approx \frac{\Gamma(1/r)}{r}N^{1-1/r}$ is the average failures masked before wipe-out. $A^\star$ adopts the maximum availability from Saxena et al. (2024). Using approximations $\mu(N,r^\star)\approx N/2$ and $\bar S\approx 2$ yields $r^\star\approx \lfloor\log_2 N + 0.833\rfloor$. $T_c^\star=T_s+\sqrt{T_s^2+2T_s(T_f+T_r)}$ remains a closed-form solution.
    - **Design Motivation**: Since SPARe cannot mask failures indefinitely, CKPT is required. Solving the "redundancy vs overhead" and "checkpoint frequency vs rework" trade-offs simultaneously provides engineers with a direct lookup for $r^\star$.

### Loss & Training
The optimizer and model architecture remain unchanged. ReCtlr is inserted at each all-reduce step: see pseudocode in Alg. 1 (Main training loop) and Alg. 2 (Three-stage ReCtlr). Fault detection assumes standard NCCL all-reduce hang/drop methods. Shrink and ReCtlr cost 0.1 s each; CKPT intervals follow $T_c^\star$.

## Key Experimental Results

### Main Results
Based on FedDES (SimGrid) discrete event simulation, modeling a 600k H100 cluster, 10T parameter model, $T_r=60$ min, MTBF $m=5$ min (Weibull $k=0.78$), $T_s=60$ s, and 10,000 training steps. Baselines: Rep+CKPT, CKPT-only.

| $N$ | Rep+CKPT Opt. $\text{TTT}/T_0$ | Rep Availability | SPARe+CKPT Opt. $\text{TTT}/T_0$ | SPARe $r^\star$ | SPARe Availability | Relative TTT Gain |
|------|------|------|------|------|------|------|
| 200 | 6.07 | 61.74% | **2.92** | 9 | 87.00% | **51.9%** |
| 600 | 4.27 | 79.89% | **2.49** | 8 | 93.90% | **41.7%** |
| 1000 | 3.88 | 84.41% | **2.34** | 9 | 96.54% | **39.6%** |

CKPT-only is crushed by the baseline in this restart-dominant setting as it barely makes progress.

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| $\mu(N,r)$ Formula vs Monte-Carlo | 1.13% Absolute Error | Average maskable failures closed-form aligns with MC. |
| $\bar S(N,r)$ Lower Bound vs MC | 0.60% Absolute Error | Average compute overhead lower bound is tight. |
| $\bar S(N,r)$ vs DES Simulation | ≤4% Absolute Error | Full estimation including patch compute is accurate. |
| $r=20$, $N=600$ | $\mu\approx 426$, $\bar S\approx 2.8\times$ | Compared to $20\times$ for traditional Replication. |
| Rep+CKPT Optimal $r$ | $r=3$ | Aligns with Ferreira et al. (2011); higher $r$ is hindered by $r\times$ cost. |
| SPARe $r^\star$ Theory | $\lfloor\log_2 N+0.833\rfloor=8,10,10$ | Sim $r^\star=9,8,9$; variance due to Weibull $k<1$. |

### Key Findings
- **Theoretical closed-forms are accurate**: The formulas for maskable failures $\mu$, compute overhead $\bar S$, and optimal redundancy $r^\star$ match DES simulations within 5%. The paper provides engineering guidance rather than just hyperparameter tuning.
- **High $r$ performs better than theory**: Simulations show higher availability than $A^\star(\mu m)$ at large $r$, because the active GPU count decreases as failures are masked, naturally lowering the real-time failure rate—a self-reinforcing favorable effect.
- **Low $r$ (especially $r=2$) underperforms theory**: Due to Weibull $k=0.78<1$, early failures are burstier, resulting in small $\mu$ and premature wipe-outs. This can be corrected using dynamic checkpointing (Bougeret 2011 / Benoit 2022) without modifying SPARe.
- **Gain slowly decays as $N$ increases**: From 51.9% → 41.7% → 39.6%, as Rep+CKPT with $r=3$ becomes more sufficient at large $N$. However, absolute TTT remains $\sim 2.3\times T_0$, leaving room for improvement.

## Highlights & Insights
- **Decoupling redundancy from execution** is the true key insight. Traditional replication couples these; SPARe uses cyclic stacking + early all-reduce to allow redundancy up to $r=20$ while keeping overhead at $2.8\times$. This mirrors erasure coding in storage systems but is novel in gradient computation with online HK + MCMF maintenance.
- **Closed-form engineering metrics**: Formulas like $\mu\approx\Gamma(1/r)N^{1-1/r}/r$ and $r^\star\approx\log_2 N+0.833$ are highly valuable for HPC system engineers to evaluate fault-tolerance strategies without requiring large-scale GPU resources.
- **Abstraction at the DP layer**: Algorithms 1 and 2 only manipulate shard placement and all-reduce stacks. They are orthogonal to TP/PP/EP topologies and can be layered on top of other internal schemes like Bamboo or FT-HSDP.
- **Clever algorithm selection**: Bipartite matching is a natural fit for "shard type ↔ surviving group" constraints, and its polynomial time complexity is negligible at the $N\sim 10^3$ scale.

## Limitations & Future Work
- **Simulation only**: Although FedDES + SimGrid is standard in HPC, real-world NCCL behavior, Weibull parameters, and memory pressure on 600k GPUs may vary.
- **Weibull $k<1$ performance at low $r$**: The paper acknowledges issues with early burst failures and suggests dynamic CKPT as a remedy but lacks joint experimental results.
- **Shard physical isolation assumption**: The closed-form independence assumes shard placement is decoupled from physical failure domains (racks/zones). This is recommended but not explicitly modeled as correlated rack-level failures.
- **Storage/Bandwidth modeling**: Storing $r$ shards per group increases HBM usage and preload bandwidth, which is not yet quantized as non-compute overhead.
- **Future directions**: (i) Integration with universal checkpointing (Lian 2025) for elastic-$N$ training; (ii) Incorporating cost-aware MCMF into ReCtlr to include reorder IO costs; (iii) Extending static $r^\star$ to an adaptive scheme that adjusts during wall-clock time.

## Related Work & Insights
- **vs Rep+CKPT (Ferreira 2011 / Benoit 2019)**: Both use redundancy to mask failure, but SPARe replaces "compute redundancy" with "data shard redundancy + adaptive early stopping," reducing overhead from $r\times$ to $2\sim 3\times$ and increasing optimal $r$ from $\approx 3$ to $\log_2 N+0.833$.
- **vs Bamboo / ReCycle / FT-HSDP**: These methods perform "passive rerouting" in pipeline/hybrid layers, whereas SPARe performs "active redundancy" in the DP layer; they are complementary.
- **vs GEMINI / DataStates-LLM / Universal CKPT**: These reduce the cost of a single rollback; SPARe directly reduces the number of rollbacks.
- **vs TrainMover (hot spare migration)**: TrainMover migrates upon failure, while SPARe uses survivors to compute pre-stacked shards, simplifying deployment without needing a standby pool.
- **Theoretical roots**: $\mu(N,r)$ aligns with Ferreira (2011); $T_c^\star$ follows Saxena (2024); HK and MCMF are textbook algorithms that perfectly fit the problem constraints.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "early all-reduce + adaptive reorder" perspective is a distinctive design in the LLM fault-tolerance landscape.
- **Experimental Thoroughness**: ⭐⭐⭐ Entirely simulation-based; however, the SimGrid foundation is solid, covering multiple $N$ scales and verifying theoretical expectations.
- **Writing Quality**: ⭐⭐⭐⭐ High-quality integration of closed-forms, pseudocode, and diagrams with a clear narrative arc.
- **Value**: ⭐⭐⭐⭐ Addresses a genuine problem for 600k GPU clusters; specialized formulas are immediately useful for system engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)
- [\[ICML 2026\] FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)

</div>

<!-- RELATED:END -->
