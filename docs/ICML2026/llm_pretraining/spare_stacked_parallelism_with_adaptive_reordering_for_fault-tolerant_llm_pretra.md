---
title: >-
  [Paper Note] SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs
description: >-
  [ICML 2026][Pretraining][checkpointing] SPARe cyclically stacks $r$ layers of data shards across groups in the data parallelism dimension. Upon node failure, it employs Hopcroft-Karp and min-cost max-flow algorithms for adaptive reordering of the "all-reduce stack number." In restart-dominant scenarios with 600k GPUs, it achieves availability comparable to $
tags:
  - ICML 2026
  - Pretraining
  - checkpointing
date: 2026-05-08
content_hash: 188c967ffda15c68
---
# SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs

**Conference**: ICML2026  
**arXiv**: [2603.00357](https://arxiv.org/abs/2603.00357)  
**Code**: https://github.com/padsysl/SPARe  
**Area**: LLM Pretraining / Fault-Tolerant Systems / Distributed Parallelism  
**Keywords**: Failure Masking, Data Parallelism, Redundant Computation, Adaptive Reordering, Checkpointing

## TL;DR
SPARe cyclically stacks $r$ layers of data shards across groups in the data parallelism dimension. Upon node failure, it employs Hopcroft-Karp and min-cost max-flow algorithms for adaptive reordering of the "all-reduce stack number." In restart-dominant scenarios with 600k GPUs, it achieves availability comparable to $r\times$ traditional replication with only $2\sim 3\times$ computational overhead, reducing time-to-train by $40\sim 50\%$ compared to Rep+CKPT.

## Background & Motivation

**Background**: Current frontier LLM pretraining clusters have entered the $10^5$ GPU scale (e.g., Llama-3 with 16k H100, future 600k H100). Prevailing fault-tolerance methods include: checkpointing (GEMINI, Just-in-Time, Universal CKPT), partial recovery (communicator shrink), and replication (redundant computation with $r$ copies per group).

**Limitations of Prior Work**: As cluster size expands, MTBF decreases according to $\mathcal{O}(1/\#\mathrm{GPU})$, while the cost of global restarts (NCCL_init, collective communication) grows linearly with $\#\mathrm{GPU}$. Llama-3 reported one failure every 3 hours on 16k GPUs; extrapolated to 96k, this becomes 30 min, and 5 min for 600k. Given that a global restart on 600k GPUs takes 60 min, the system enters a **restart-dominant regime** where downtime dominates. Checkpointing only reduces rework waste but cannot cut the frequency of restarts. Traditional replication maintains availability but incurs an unbearable $20\times$ computation cost when $r=20$.

**Key Challenge**: The hard trade-off between availability gains (requiring high redundancy $r$) and computational overhead (growing linearly with $r$).

**Goal**: To implement a "redundancy with near-constant overhead" failure-masking scheme at the data parallelism layer, independent of model architecture and inner parallelism topologies (TP/PP/EP).

**Key Insight**: The crucial observation is that **not all computations in every group must be completed before all-reduce**. As long as at least one surviving group processes each of the $N$ shard types, gradients can be aggregated. Thus, "redundancy" is placed at the shard level rather than the group level, allowing for early all-reduce with "fewer stack layers."

**Core Idea**: $N$ data shards $\{D_0,\dots,D_{N-1}\}$ are stacked in $r$ layers across $N$ model-parallel groups using cyclic rotation (each group holds $r$ different shards). During training, all-reduce is triggered once a minimum stack number $c(k)=\lceil N/(N-k)\rceil$ is reached to "collect all types." After node failures, HK + MCMF algorithms are used for **adaptive reordering** of the stack sequence to ensure the minimum stack number remains achievable.

## Method

### Overall Architecture
SPARe is built entirely on synchronous Data Parallelism: $N$ model-parallel groups, each with $M$ GPUs holding a model replica, forming $M$ DP communicators with a world size of $N$. It does not alter the model structure or inner TP/PP/EP topologies; it only rearranges **which shards each group processes** and **when all-reduce is triggered**. The layout is static: each group $g_i$ holds a shard stack $[D_i, D_{i+1}, \dots, D_{i+r-1}]$ (index mod $N$) via cyclic rotation, ensuring any two shard types overlap at most once across all groups (independence condition, see Thm. 4.1). Scheduling is dynamic: each training step maintains an all-reduce stack number $S$ (initially 1). Failure triggers the ReCtlr controller to decide on shard reordering or increasing $S$. Surviving groups then perform "patch compute" for missing shard types, shrink the communicator, all-reduce, and update parameters. The following diagram illustrates the main pipeline:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 420, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    CFG["SPARe+CKPT Joint Optimization<br/>Closed-form determination of r*≈⌊log₂N+0.833⌋ and checkpoint period"]
    subgraph S1["Stacking + Early All-reduce"]
        direction TB
        L["Cyclic stacking of r layers: each group holds shard stack [D_i … D_{i+r−1}]"]
        T["Calculate only top S layers per step to trigger all-reduce (initial S=1)"]
        L --> T
    end
    CFG --> L
    T --> Q{"Node failure during all-reduce?"}
    Q -->|No| U["Aggregate gradients ḡ → Update parameters → Next step"]
    subgraph S2["ReCtlr Three-stage Adaptive Reordering"]
        direction TB
        P0["Phase 0 HK-Fixed: Can current layout cover all types within S layers?"]
        P1["Phase 1 HK-Free: Increment S to find minimum feasible S*"]
        P2["Phase 2 MCMF: Min-cost reordering"]
        P0 -->|No| P1
        P1 -->|Found S*| P2
    end
    Q -->|Yes| P0
    P0 -->|Yes, zero move| PA
    P1 -->|Not covered at S=r (wipe-out)| R["Global restart: Reset stack, S←1"]
    P2 --> PA["Patch compute: Survivors calculate missing shard types"]
    PA --> SH["Shrink communicator → all-reduce → update → commit new S and layout"]
    SH --> U
    R --> L
```

### Key Designs

**1. Stacking + Early All-reduce: Decoupling Redundancy from Completion**

Traditional replication incurs a full computation cost for each redundant copy because it couples "maintaining $r$ redundancy" with "completing all $r$ copies before aggregation." The key insight of SPARe is that not every group needs to compute all $r$ shards; as long as each of the $N$ shard types is computed by at least one survivor, gradients can be aggregated. Consequently, the cost is reduced from $r\times$ to approximately $S\times$. Specifically, at the $k$-th failure, $N-k$ groups must cover $N$ types, with a lower bound for the stack number $c(k)=\lceil N/(N-k)\rceil$. SPARe sets $S$ targeting this bound. The process is transparent to the model layer: reordering only changes "who provides type $i$," while the aggregated gradient remains $\bar{\mathbf g}=\frac{1}{N}\sum_i \mathbf g_i$. This decoupling allows for large $r$ while keeping overhead near constant ($S\approx 2$) under normal conditions.

**2. ReCtlr: HK-Fixed / HK-Free / MCMF Three-stage Adaptive Reordering**

The challenge after failure is determining if the current shard layout can still cover all types within $S$ layers, and if not, finding the minimum $S$ and reordering with minimal data movement. ReCtlr models the relationship between "$N$ shard types" and the "top $S$ layers of surviving groups" as a bipartite graph. **HK-Fixed** (Hopcroft-Karp) first checks for a perfect matching in the current layout; this succeeds in 90%+ of steps, avoiding unnecessary reordering. If it fails, **HK-Free** (Phase 1) searches for the smallest feasible $S^\star$ by allowing arbitrary permutations within group stacks. If no solution exists even at $S=r$, a wipe-out is declared, triggering a global restart. Finally, **MCMF** (Phase 2) solves for a reordering scheme on "surviving groups × stack slots" that satisfies $S^\star$ with minimal move cost. These algorithms run in polynomial time ($< 0.1$ s for $N\sim 10^3$), ensuring reordering is not a bottleneck.

**3. SPARe+CKPT Joint Optimization: Solving for Optimal $r^\star$ and Checkpoint Period**

Since SPARe cannot mask infinite failures, it requires checkpointing. The paper solves the trade-offs (redundancy vs. cost, checkpoint frequency vs. rework) simultaneously. It defines a normalized time-to-train $J(r)=\bar S(N,r)/A^\star(\mu(N,r)\, m)$, where the average computation overhead $\bar S(N,r)\approx \frac{1}{\lfloor\mu\rfloor}\sum_{k=0}^{\lfloor\mu\rfloor-1}(c(k)+\rho_k)$ and the average failures masked before wipe-out $\mu(N,r)\approx \frac{\Gamma(1/r)}{r}N^{1-1/r}$. By setting $\mu(N,r^\star)\approx N/2$ and $\bar S\approx 2$, the optimal redundancy is derived as:

$$r^\star\approx \big\lfloor\log_2 N + 0.833\big\rfloor,$$

The checkpoint period $T_c^\star=T_s+\sqrt{T_s^2+2T_s(T_f+T_r)}$ follows the Young & Daly style. This allows engineers to calculate optimal parameters based on cluster size $N$ without extensive GPU experimentation.

### Loss & Training
The optimizer and model structure remain unchanged. ReCtlr is inserted at each all-reduce step (Alg. 1 and Alg. 2). Failure detection uses standard NCCL all-reduce timeout methods. Communicator shrinking and ReCtlr cost $\sim 0.1$ s each. Checkpointing intervals follow the calculated $T_c^\star$.

## Key Experimental Results

### Main Results
Based on FedDES (SimGrid) discrete event simulation, simulating a 600k H100 cluster, 10T parameter model, $T_r=60$ min, MTBF $m=5$ min (Weibull $k=0.78$), $T_s=60$ s, and 10,000 training steps.

| $N$ | Rep+CKPT Optimal $\text{TTT}/T_0$ | Rep Availability | SPARe+CKPT Optimal $\text{TTT}/T_0$ | SPARe $r^\star$ | SPARe Availability | TTT Relative Gain |
|------|------|------|------|------|------|------|
| 200 | 6.07 | 61.74% | **2.92** | 9 | 87.00% | **51.9%** |
| 600 | 4.27 | 79.89% | **2.49** | 8 | 93.90% | **41.7%** |
| 1000 | 3.88 | 84.41% | **2.34** | 9 | 96.54% | **39.6%** |

CKPT-only is non-competitive in this restart-dominant setting as it fails to progress significantly.

### Ablation Study / Theory vs. Simulation

| Configuration | Key Metric | Description |
|------|---------|------|
| $\mu(N,r)$ Formula vs. Monte-Carlo | 1.13% Absolute Error | Closed-form masked failures match MC results |
| $\bar S(N,r)$ Lower Bound vs. MC | 0.60% Absolute Error | Lower bound for computation overhead is tight |
| $\bar S(N,r)$ vs. DES Simulation | ≤4% Absolute Error | Full estimation including patch compute is accurate |
| $r=20$, $N=600$ | $\mu\approx 426$, $\bar S\approx 2.8\times$ | Rep requires $20\times$ for the same redundancy |
| Rep+CKPT Optimal $r$ | $r=3$ | Consistent with Ferreira et al. (2011); higher $r$ is too costly |
| SPARe $r^\star$ Theory | $\lfloor\log_2 N+0.833\rfloor=8,10,10$ | Simulation $r^\star=9,8,9$ (difference due to Weibull $k<1$) |

### Key Findings
- **Theoretical closed-forms are accurate**: Formulas for $\mu$, $\bar S$, and $r^\star$ align with DES simulation within 5%, providing engineering guidance rather than just hyperparameter tuning.
- **High $r$ performs better than theory**: Simulation availability at high $r$ exceeds $A^\star(\mu m)$ because active GPU count decreases as redundancies are masked, naturally lowering the real-time failure rate—a self-reinforcing favorable effect.
- **Low $r$ (especially $r=2$) underperforms**: Due to Weibull $k=0.78<1$, early failures are burstier, leading to premature wipe-out. This requires dynamic checkpointing for correction.
- **Gains decay slowly with $N$**: Reduction from 51.9% to 39.6% occurs because Rep+CKPT becomes more efficient with $r=3$ at large $N$, though absolute TTT remains $\sim 2.3\times T_0$.

## Highlights & Insights
- **Decoupling redundancy from completion** is the core insight. By allowing early all-reduce with cyclic stacking, redundancy can be increased to $r=20$ while overhead only rises to $2.8\times$. This mirrors erasure coding in storage but is adapted for gradient computation with online HK + MCMF maintenance.
- **Closed-form engineering metrics**: Formulas like $\mu\approx\Gamma(1/r)N^{1-1/r}/r$ allow engineers to evaluate fault-tolerance strategies via lookup tables without physical GPU runs.
- **DP-layer abstraction**: SPARe operates only on shard placement and all-reduce stacks, making it orthogonal and complementary to inner-layer parallelisms like TP/PP/EP (e.g., Bamboo, ReCycle).
- **Clever algorithm selection**: HK + MCMF naturally maps to shard-type and surviving-group constraints; their polynomial complexity makes them effectively "free" at the scale of $N\sim 10^3$.

## Limitations & Future Work
- **Simulation-only validation**: While FedDES + SimGrid is standard in HPC, real-world NCCL behavior, Weibull parameters, and memory pressure on 600k GPUs may vary.
- **Performance at low $r$**: Performance under Weibull $k<1$ distribution requires dynamic checkpointing integration.
- **Shard physical isolation assumption**: The theory assumes shard placement is decoupled from physical failure domains (racks/zones), which was not explicitly modeled in simulation.
- **Storage/Bandwidth overhead**: The storage and preload bandwidth for $r$ shards per group increases linearly with $r$; these non-computational costs were not quantified.
- **Future Directions**: (i) Coupling with elastic-$N$ training; (ii) Cost-aware MCMF including IO costs of reordering; (iii) Adaptive $r^\star$ that adjusts over total wall-clock time.

## Related Work & Insights
- **vs. Rep+CKPT (Ferreira 2011 / Benoit 2019)**: Both mask failures with redundancy, but SPARe replaces "full computation redundancy" with "shard redundancy + early stopping," reducing cost from $r\times$ to $2\sim 3\times$.
- **vs. Bamboo / ReCycle / FT-HSDP**: These methods perform "passive rerouting" in inner layers; SPARe performs "proactive redundancy" in the DP layer.
- **vs. GEMINI / Universal CKPT**: These reduce the cost of single rollbacks; SPARe reduces the frequency of rollbacks.
- **vs. TrainMover**: TrainMover migrates on failure; SPARe uses pre-stacked shards in surviving groups, requiring no standby pool.

## Rating
- Novelty: ⭐⭐⭐⭐ Early all-reduce combined with adaptive reordering is a distinct and impactful design for LLM training fault tolerance.
- Experimental Thoroughness: ⭐⭐⭐ Simulation-only, but the use of SimGrid is rigorous and the theory-experiment loop is well-validated.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation, pseudocode, and diagrams; the narrative from restart-dominant regimes to the proposed solution is cohesive.
- Value: ⭐⭐⭐⭐ Addresses a critical problem for 600k GPU clusters; closed-form formulas and open-source code are highly useful for system engineers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[ICLR 2026\] ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning](../../ICLR2026/llm_pretraining/adept_continual_pretraining_via_adaptive_expansion_and_dynamic_decoupled_tuning.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICML 2026\] FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)
- [\[ICLR 2026\] Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining](../../ICLR2026/llm_pretraining/beyond_urls_metadata_diversity_and_position_for_efficient_llm_pretraining.md)

</div>

<!-- RELATED:END -->
