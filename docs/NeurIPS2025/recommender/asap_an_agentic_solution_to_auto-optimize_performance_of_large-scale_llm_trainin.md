---
title: >-
  [Paper Note] ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training
description: >-
  [NeurIPS 2025][Recommender Systems][Multi-agent systems] ASAP is a multi-agent system (Coordinator + Analyzer + Proposal) that automatically diagnoses bottleneck types (compute/memory/communication) in large-scale LLM distributed training and proposes sharding configurations. Across 3 experimental scenarios, it matches human expert solutions and achieves up to 2.58× throughput improvement.
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "Multi-agent systems"
  - "distributed training optimization"
  - "sharding configuration"
  - "bottleneck analysis"
  - "Roofline model"
date: 2026-05-08
content_hash: 3d8074eb8cd134fc
---

# ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training

**Conference**: NeurIPS 2025
**arXiv**: [2511.03844](https://arxiv.org/abs/2511.03844)  
**Code**: Not released (Google internal ADK)  
**Area**: Recommender Systems
**Keywords**: Multi-agent systems, distributed training optimization, sharding configuration, bottleneck analysis, Roofline model

## TL;DR
ASAP is a multi-agent system (Coordinator + Analyzer + Proposal) that automatically diagnoses bottleneck types (compute/memory/communication) in large-scale LLM distributed training and proposes sharding configurations. Across 3 experimental scenarios, it matches human expert solutions and achieves up to 2.58× throughput improvement.

## Background & Motivation
**Background**: Large-scale LLM training requires parallelization across hundreds or thousands of TPUs/GPUs (data parallelism, model parallelism, sequence parallelism, pipeline parallelism, etc.), where sharding configurations have a substantial impact on performance — suboptimal configurations can cause 50%+ throughput degradation.

**Limitations of Prior Work**: (a) Manual trial-and-error demands deep expertise in distributed systems; (b) black-box optimization methods (e.g., Bayesian optimization) converge slowly in large configuration spaces and provide no interpretable rationale; (c) different bottleneck types (compute-bound/memory-bound/communication-bound) require different optimization strategies, making general-purpose methods inefficient.

**Key Challenge**: Diagnosis requires understanding performance profiling reports (Xprof profiles), and decision-making requires knowledge of parallelization principles and hardware topology — expertise that has traditionally been confined to human specialists.

**Goal**: Replace human experts with LLM agents for the full pipeline of performance diagnosis → proposal generation → configuration tuning.

**Key Insight**: Combining LLM reasoning capabilities with specialized tools (Xprof profiling, Roofline analysis) and knowledge bases (historical optimization records, JAX parallelization documentation) into a multi-agent system.

**Core Idea**: Coordinator orchestrates → Analyzer diagnoses bottleneck type → Proposal (RAG + knowledge base) generates multiple candidate solutions = LLM agents serving as distributed training experts.

## Method

### Overall Architecture
Three agents collaborate: Coordinator (accepts experiment ID, orchestrates workflow) → Analyzer (invokes Xprof to analyze KPIs + operation-level performance + Roofline, diagnosing compute/memory/communication bottlenecks) → Proposal (retrieves historical configurations and knowledge-base documents via RAG, generates 3 candidate solutions with rationale and expected impact).

### Key Designs

1. **Analyzer Agent (Performance Diagnosis)**:

    - Reads KPIs: accelerator busy rate, training step time, goodput
    - Operation-level analysis: FLOPs utilization and HBM bandwidth utilization for the top-5 most time-consuming operations
    - Roofline analysis: classifies each operation as compute-bound, HBM-bound, or communication-bound
    - Correlates KPI anomalies with operation-level bottlenecks → generates a diagnostic report

2. **Proposal Agent (RAG-based Solution Generation)**:

    - Historical database: past configurations, Xprof profiles, impact summaries
    - Knowledge base: JAX parallelization documentation
    - Generates 3 candidate sharding configurations + rationale + expected trade-offs
    - Confidence scores (85–90%)

3. **Sharding Configuration Space**:

    - `ici_mesh`: four parallelism dimensions — {model, data, replica, sequence}
    - Hardware: TPU v5p-512 (8×8×8), TPU v6e-16 (4×4), TPU v5e-256 (16×16)

### Loss & Training
No training involved — purely LLM inference with tool invocation.

## Key Experimental Results

### Main Results (3 Scenarios)

| Scenario | Hardware | Bottleneck Type | Agent Solution | Effect | Matches Expert? |
|----------|----------|----------------|----------------|--------|----------------|
| 1 | TPU v5p-512 | Compute-bound | model=8, data=16, seq=4 | **−28% step time** | ✓ Exact match |
| 2 | TPU v6e-16 | HBM-bound | data=4, model=4 | **1.43× throughput** | ✓ Exact match |
| 3 | TPU v5e-256 | Communication-bound | data=8, model=2, seq=16 | **2.58× throughput** | ✓ Exact match |

### Ablation Study

| Configuration | Key Finding | Remark |
|---------------|-------------|--------|
| Scenario 1 in historical database | Agent retrieves and improves upon it | Retrieval augmentation |
| Scenarios 2/3 not in historical database | Agent still diagnoses and proposes correctly | Principled reasoning & generalization |
| Contribution of Roofline analysis | Differentiates compute/memory/communication | Critical for diagnostic accuracy |
| 3 proposals vs. 1 proposal | Best solution most often ranked first | Diversity provides fallback options |
| Agent confidence | 85–90% | Accurately reflects uncertainty |

### Key Findings
- All 3 scenarios exactly match human expert solutions — demonstrating that LLM reasoning combined with tools can substitute for distributed systems expertise.
- Scenarios 2 and 3 are absent from the historical database — indicating that the agent reasons rather than memorizes, and generalizes to novel scenarios.
- The largest gain of 2.58× throughput stems from correctly identifying the communication bottleneck and reducing model parallelism degree.

## Highlights & Insights
- **LLM Reasoning + Specialized Tools = Domain Expert**: ASAP demonstrates that LLMs do not merely retrieve historical configurations but reason from parallelization principles — enabling them to handle novel situations not present in the training database.
- **Interpretable Decision-Making**: Each proposal includes detailed rationale and performance references, offering greater engineering value than black-box optimization methods.
- **Full Coverage of Three Bottleneck Types**: Different scenarios demand fundamentally different strategies (compute-bound → increase parallelism degree; communication-bound → reduce parallelism degree), and the agent correctly distinguishes among all three.

## Limitations & Future Work
- Only global sharding is supported; per-layer sharding is not considered.
- No closed-loop feedback — solutions are proposed but not automatically verified and iterated.
- Compiler-level optimizations (operator fusion, memory layout selection) are absent.
- The system depends on accurate Xprof data; robustness to noisy profiling data is untested.
- Holistic trade-off optimization between computation and communication remains insufficient.

## Related Work & Insights
- **vs. Alpa (Zheng et al., 2022)**: Alpa employs integer linear programming to search for optimal parallelization strategies; ASAP uses LLM reasoning with a knowledge base, offering greater flexibility and interpretability.
- **vs. FlexFlow**: An automatic parallelization framework with a limited search space; ASAP handles a substantially larger configuration space.
- **vs. SOLAR (Google, 2024)**: A system recommendation framework that does not perform bottleneck diagnosis; ASAP covers the full pipeline from diagnosis to recommendation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multi-agent systems with performance profiling tools is well-motivated and effective.
- Experimental Thoroughness: ⭐⭐⭐ Only 3 scenarios are evaluated; all match expert solutions, but the sample size is small.
- Writing Quality: ⭐⭐⭐⭐ The diagnostic process is described in detail with a clear reasoning chain.
- Value: ⭐⭐⭐⭐ The approach has practical engineering value for automated tuning of large-scale LLM training.

- **Implications for MLOps**: ASAP's multi-agent diagnose-and-propose paradigm is generalizable to hyperparameter search, data pipeline optimization, and similar tasks.
- **Intersection with the MLSys Community**: Introducing LLM agents into systems optimization represents a promising new research direction.
- **Commercial Value**: Configuration optimization for large-scale training clusters directly impacts computational costs on the order of millions of dollars.
- **Cross-Hardware Generalization**: Success across all 3 TPU topologies indicates that the reasoning capability is not tied to specific hardware.
- **Relevance to the Open-Source Community**: The diagnostic methodology (Roofline + operation-level analysis) is transferable to PyTorch/DeepSpeed environments.
- **Comparison with Auto-Tuning**: Traditional auto-tuning methods do not explain *why* a configuration works; ASAP provides an interpretable decision chain.
- **Extensible Architecture**: The three-agent design can be extended with an Evaluator Agent to enable closed-loop iterative optimization.
- **Value of Agent Confidence Scores**: Confidence estimates of 85–90% help users determine whether additional validation is warranted.
- **Maintainability of the Knowledge Base**: The RAG-based design allows the knowledge base to be updated by appending new records, without retraining the agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Balancing Performance and Costs in Best Arm Identification](balancing_performance_and_costs_in_best_arm_identification.md)
- [\[NeurIPS 2025\] Estimating Hitting Times Locally At Scale](estimating_hitting_times_locally_at_scale.md)
- [\[ICLR 2026\] Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](../../ICLR2026/recommender/adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)
- [\[ICLR 2026\] Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](../../ICLR2026/recommender/rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)

</div>

<!-- RELATED:END -->
