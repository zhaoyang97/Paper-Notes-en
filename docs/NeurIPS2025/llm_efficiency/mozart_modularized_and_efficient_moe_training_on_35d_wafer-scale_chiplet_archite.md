---
title: >-
  [Paper Note] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures
description: >-
  [NeurIPS 2025][LLM Efficiency][MoE] Mozart is an algorithm-hardware co-design framework that achieves over 1.9× training speedup on three MoE-LLMs via expert clustering and allocation, fine-grained streaming scheduling…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "MoE"
  - "chiplet architecture"
  - "expert parallelism"
  - "wafer-scale"
  - "algorithm-hardware co-design"
date: 2026-05-08
content_hash: 499694b29315e687
---

# Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures

**Conference**: NeurIPS 2025
**arXiv**: [2603.07006](https://arxiv.org/abs/2603.07006)  
**Code**: None  
**Area**: LLM Efficiency / Hardware-Algorithm Co-design
**Keywords**: MoE, chiplet architecture, expert parallelism, wafer-scale, algorithm-hardware co-design

## TL;DR
Mozart is an algorithm-hardware co-design framework that achieves over 1.9× training speedup on three MoE-LLMs via expert clustering and allocation, fine-grained streaming scheduling, and a 3.5D chiplet architecture (NoP-Tree + hierarchical memory).

## Background & Motivation
**Background**: MoE architectures enable efficient scaling through sparse activation (e.g., Mixtral-8x7B, DeepSeek-MoE), yet their sparsity introduces significant challenges for hardware deployment, including poor memory locality, high communication overhead, and uneven utilization of compute resources.

**Limitations of Prior Work**: (1) Existing chiplet solutions are mostly sub-wafer designs and do not support wafer-scale integration; (2) they adopt coarse-grained static workload partitioning that assumes dense and uniform computation, which is ill-suited to the dynamic sparsity of MoE; (3) all-to-all communication in expert parallelism remains a critical bottleneck.

**Key Challenge**: There is a lack of alignment between the logical modularity of MoE and the physical modularity of hardware — frequently co-activated experts may be mapped to distant compute units.

**Goal**: Design a chiplet architecture and scheduling algorithm that matches the modular nature of MoE, reducing communication overhead and improving resource utilization.

**Key Insight**: Drawing an analogy to the modular organization of the human brain — specialized modules handle distinct tasks while neighboring regions coordinate with low latency. Expert activation priors (activation frequency + co-activation patterns) are analyzed to guide expert-to-chiplet mapping.

**Core Idea**: Leveraging expert co-activation priors, frequently co-activated experts are clustered onto the same chiplet group, complemented by a 3.5D NoP-Tree topology and streaming scheduling to enable efficient MoE training.

## Method

### Overall Architecture
Two-level optimization: (1) Algorithm level — analyze routing policies to obtain expert activation priors, followed by expert clustering, allocation, and fine-grained scheduling; (2) Architecture level — a 3.5D chiplet system comprising 3D logic-on-memory stacking, a 2D NoP-Tree interconnect, and two-level storage.

### Key Designs

1. **Expert Clustering & Allocation**:

    - Function: A two-stage approach — first clusters frequently co-activated experts (based on co-activation matrix $\mathcal{C}$), then assigns clusters to chiplet groups to balance load.
    - Mechanism: Clustering employs farthest point sampling to maximize inter-group distance; allocation uses binary integer programming to minimize inter-group load imbalance.
    - Design Motivation: Co-activated experts on the same chiplet require only one copy of a token to be sent (rather than $k$ copies), directly reducing all-to-all communication volume $\mathcal{C}_\mathcal{T}$.

2. **Fine-grained Streaming**:

    - Function: Achieves communication-computation overlap through token- and expert-level pipelining.
    - Mechanism: DRAM→SRAM loading of expert weights is interleaved with token computation, avoiding the need to load all expert weights at once.
    - Design Motivation: Since MoE activates only $k$ experts per forward pass, the majority of expert weights are idle; streaming loading reduces peak memory demand.

3. **3.5D Chiplet Architecture**:

    - Function: Designs a NoP-Tree topology consisting of attention chiplets (central dispatchers) and expert chiplets (leaves).
    - Mechanism: 3D integration (compute die + SRAM die via hybrid bonding) provides low-latency local activation caching; a 2D NoP-Tree enables in-network MoE aggregation (switch nodes perform message aggregation).
    - Design Motivation: Attention and MoE exhibit fundamentally different memory access patterns — attention is compute-intensive while MoE is communication-intensive; the heterogeneous chiplet design matches this distinction.

## Key Experimental Results

### Comparison with Baselines

| MoE Model | Speedup |
|-----------|---------|
| Mixtral-8x7B | >1.9× |
| DeepSeek-MoE | >1.9× |
| Third model | >1.9× |

### Key Findings
- Expert parameters account for 90%+ of total MoE-LLM parameters, yet activation patterns are highly non-uniform.
- Co-activation clustering reduces all-to-all communication volume by 30–40%.
- Streaming scheduling achieves 80%+ communication-computation overlap.

## Highlights & Insights
- **Brain-inspired design philosophy**: The paper maps neuroscience-inspired modularity theory onto hardware design; the alignment between logical modularity (MoE experts) and physical modularity (chiplets) is a novel contribution.
- **Prior-driven optimization**: Routing statistics collected from a pretrained model on instruction-tuning data are used to guide post-training deployment — this "analyze-then-optimize" strategy has strong practical utility.

## Limitations & Future Work
- **Focus on post-training only**: Routing patterns during pretraining may be unstable, and the derived priors may not generalize.
- **Simulation-based validation**: The hardware design has not been evaluated on physical chiplets.
- **Scalability of binary integer programming**: Solving may become slow with larger expert pools (e.g., 256 experts in DeepSeek-V3).

## Related Work & Insights
- **vs. FRED (Rashidi et al. 2024)**: FRED targets wafer-scale LLM training but employs coarse-grained static partitioning; Mozart introduces MoE-aware fine-grained scheduling.
- **vs. Cambricon-LLM**: Cambricon-LLM targets inference only and does not consider wafer-scale integration; Mozart addresses training and supports wafer-scale deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ Algorithm-hardware co-design with MoE-aware chiplet optimization
- Experimental Thoroughness: ⭐⭐⭐ Three models evaluated, but simulation only
- Writing Quality: ⭐⭐⭐⭐ Clear figures and well-motivated design decisions
- Value: ⭐⭐⭐⭐ Significant practical guidance for MoE hardware deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)
- [\[NeurIPS 2025\] Scale-invariant Attention](scale-invariant_attention.md)
- [\[NeurIPS 2025\] A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](a_unified_framework_for_establishing_the_universal_approxima.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[NeurIPS 2025\] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)

</div>

<!-- RELATED:END -->
