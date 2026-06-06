---
title: >-
  [Paper Note] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs
description: >-
  [NeurIPS 2025][LLM Pretraining][distributed training] This paper proposes Arnold, a scheduling system that aligns the communication patterns of LLM training (DP/PP groups) with the physical network topology of data cente…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "distributed training"
  - "GPU scheduling"
  - "network topology"
  - "communication alignment"
  - "hybrid parallelism"
  - "MIP"
date: 2026-05-08
content_hash: b0b1e7f81c25fcc8
---

# Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs

**Conference**: NeurIPS 2025
**arXiv**: [2509.15940](https://arxiv.org/abs/2509.15940)  
**Code**: To be confirmed  
**Area**: LLM Pre-Training
**Keywords**: distributed training, GPU scheduling, network topology, communication alignment, hybrid parallelism, MIP

## TL;DR
This paper proposes Arnold, a scheduling system that aligns the communication patterns of LLM training (DP/PP groups) with the physical network topology of data centers. In simulation, Arnold reduces the maximum communication group span by 1.67×, and achieves a 10.6% end-to-end throughput improvement in production-scale training on 9,600+ GPUs.

## Background & Motivation

**Background**: LLM pre-training requires thousands of GPUs operating in concert, with network communication accounting for 30%–50% of total training time. Modern data centers employ multi-tier fat-tree topologies (leaf→spine→core), where available bandwidth decreases at higher tiers.

**Limitations of Prior Work**: (a) Existing GPU schedulers (e.g., bin-packing) optimize only for GPU locality—packing GPUs as close together as possible—without any awareness of the communication structure of LLM training. This causes DP/PP communication groups to span multiple spine switches, degrading bandwidth. (b) DP and PP are orthogonal parallelism strategies, and each GPU participates in both simultaneously, making it impossible to perfectly align both communication patterns at once; a trade-off is necessary.

**Key Challenge**: LLM training communication is sparse yet high-throughput—99% of GPU pairs have no direct traffic, with data exchange occurring only within specific communication groups. However, schedulers are unaware of this sparse structure, and random or greedy GPU assignments cause communication groups to span multiple pods, severely degrading bandwidth (collective operations by up to 17%, P2P by up to 70%).

**Goal**: Design a topology-aware scheduling algorithm for LLM pre-training jobs (LPJs) that maps communication groups onto the physical topology to minimize cross-pod communication.

**Key Insight**: The paper characterizes the performance degradation patterns of different communication operations under cross-pod placement, and formulates the scheduling problem as a Mixed Integer Program (MIP) that minimizes the weighted maximum span of communication groups.

**Core Idea**: Align the communication matrix of LLM hybrid parallelism with the data center topology structure via MIP-based optimization to minimize the physical span of communication groups.

## Method

### Overall Architecture
Arnold takes as input the GPU count and parallelism configuration (DP/TP/PP) of an LPJ, constructs a communication matrix, and invokes a MIP solver to find an optimal GPU-to-minipod assignment. A resource reservation strategy is also included to hold nodes for incoming LPJs.

### Key Designs

1. **Communication Matrix Modeling**:

    - Function: Abstracts the communication pattern of an LPJ into a two-dimensional matrix.
    - Mechanism: Rows represent DP groups and columns represent PP groups. Each matrix node $v_{ij}$ carries a vector $[v_w, v_d, v_p]$ encoding weight size and DP/PP communication volumes. Communication group size is computed as $DP = \#GPUs / TP / PP$.
    - Design Motivation: Simplifies the complex hybrid-parallel communication structure into an optimizable matrix representation.

2. **MIP Scheduling Algorithm**:

    - Function: Minimizes the maximum physical span of communication groups across minipods.
    - Mechanism: The objective function is $\text{MIN}[\alpha \cdot \text{max DP spread} + \beta \cdot \text{max PP spread}]$, where $\alpha + \beta = 1$ controls the DP/PP trade-off. Exploiting the homogeneous synchronous nature of communication groups, the problem is reduced to a bin-packing variant MIP and solved efficiently with an off-the-shelf solver (SCIP).
    - Design Motivation: Traditional bin-packing ignores communication structure; this paper treats communication groups as the unit of scheduling and jointly optimizes cross-pod span.

3. **Communication Profiling + Automatic Trade-off**:

    - Function: Determines optimal values of $\alpha$ and $\beta$.
    - Mechanism: Communication performance is profiled in advance for different model and GPU types and stored in a database. At scheduling time, the compute-to-communication ratio $r_1$ and the DP-to-PP communication volume ratio $r_2$ are used to retrieve the most similar historical job, from which $\alpha$ and $\beta$ are derived.
    - Design Motivation: Different models have different communication bottlenecks (PP-dominated for dense models; both significant for MoE models), requiring dynamic adjustment of the trade-off.

### Resource Management Strategy
A reservation mechanism is adopted: after an LPJ is planned, its nodes are reserved. Newly arriving jobs are preferentially scheduled outside the reserved zone. If the predicted completion time of a new job is earlier than the LPJ's arrival time, it may temporarily use the reserved zone to improve utilization. A ML-based predictor is used to estimate job completion time (JCT).

## Key Experimental Results

### Simulation: Communication Group Span

| Algorithm | Span Reduction |
|-----------|---------------|
| Best-fit | baseline |
| GPU-packing | ≈ baseline |
| Topo-aware | marginally better |
| Arnold | up to 1.67× |

### Production Cluster Experiments

| Scale | Baseline | Gain |
|-------|----------|------|
| 208 GPUs | vs. MegaScale | +5.7% throughput |
| 9,600+ GPUs | vs. MegaScale | +10.6% throughput |

### Communication Performance Degradation (Cross-minipod)

| Communication Type | Degradation |
|-------------------|-------------|
| Collective (AllGather/ReduceScatter) | up to 17% |
| P2P (Send-Recv) | up to 70% |

### Key Findings
- PP-aligned placement consistently outperforms DP-aligned and unaligned schemes, as PP communication (P2P) is more sensitive to cross-pod latency.
- The larger the model, the higher the communication overhead, and the greater the improvement from Arnold.
- Cross-minipod placement is insensitive to intra-minipod topology (performance difference < 0.3%), validating the assumption that inter-minipod optimization is the primary target.

## Highlights & Insights
- **Industrial-scale system with rigorous academic abstraction**: Validated in ByteDance's 9,600+ GPU production environment while formally modeling the problem as a tractable MIP, combining practical impact with theoretical elegance.
- **Detailed characterization of communication degradation**: Quantitative degradation curves for different communication operations under cross-pod placement provide a rigorous basis for scheduling optimization.
- **Discovery of an unexpected cascade effect**: Topology alignment not only improves communication efficiency but also indirectly benefits compute kernel performance through reduced GPU stream resource contention.

## Limitations & Future Work
- **Dependency on prior profiling**: Communication characteristics must be profiled and stored for each model/GPU type; new models require additional measurement, increasing deployment lead time.
- **Three-tier topology assumption**: The algorithm is designed for leaf-spine-core three-tier topologies; other topologies (e.g., Dragonfly, Torus, rail-only) require adaptation.
- **Static scheduling**: Placement is fixed after assignment and is not re-optimized in response to node failures or load changes during training.
- **Inter-job interference not modeled**: Although the appendix discusses inter-job interference in shared clusters, the scheduling algorithm does not explicitly model network congestion.
- **MIP solver scalability ceiling**: For future clusters exceeding 100,000 GPUs, MIP solve time may become a bottleneck.
- **Future directions**: (1) Dynamic rescheduling—re-optimizing placement based on actual communication load during training; (2) Integration with elastic training—automatic re-alignment upon node addition or removal; (3) Incorporating network congestion and multi-tenant interference into the objective function.

## Related Work & Insights
- **vs. MegaScale**: MegaScale applies full-stack optimization (communication, computation, fault tolerance), while Arnold focuses on the scheduling layer and is orthogonal to MegaScale—yielding an additional 10.6% improvement on top of MegaScale.
- **vs. traditional GPU schedulers**: Gandiva, MLaaS, and similar systems optimize GPU locality via bin-packing but are unaware of the LLM communication matrix structure.
- **vs. Topo-aware**: Existing topology-aware schedulers use graph partitioning but consider only data-parallel communication, ignoring the DP–PP trade-off.
- **Insight**: As LLM model scale continues to grow, communication optimization will become the primary lever for training efficiency; scheduling-level "zero-cost" optimizations are particularly attractive.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic alignment of LLM hybrid-parallel communication patterns with network topology; MIP formulation is clean and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three-tier validation (simulation + 208 GPUs + 9,600+ GPUs), covering communication micro-benchmarks, end-to-end throughput, and breakdown analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, figures and tables are comprehensive, and industrial experience is well integrated with academic analysis.
- Value: ⭐⭐⭐⭐⭐ A 10.6% production-level gain at thousand-GPU scale translates to enormous compute savings, and the approach is composable with other optimizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](../../ICCV2025/llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->
