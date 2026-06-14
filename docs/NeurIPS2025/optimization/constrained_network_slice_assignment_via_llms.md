---
title: >-
  [Paper Note] Constrained Network Slice Assignment via Large Language Models
description: >-
  [NeurIPS 2025][Optimization][Network slicing] This paper investigates the use of LLMs (Claude series) for solving constrained optimization problems in 5G network slice resource allocation. Two approaches are proposed: ze…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Network slicing"
  - "5G resource allocation"
  - "LLM"
  - "integer programming"
  - "constrained optimization"
date: 2026-05-08
content_hash: 196479b6aa855ade
---

# Constrained Network Slice Assignment via Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2512.00040](https://arxiv.org/abs/2512.00040)  
**Code**: None  
**Area**: Optimization
**Keywords**: Network slicing, 5G resource allocation, LLM, integer programming, constrained optimization

## TL;DR
This paper investigates the use of LLMs (Claude series) for solving constrained optimization problems in 5G network slice resource allocation. Two approaches are proposed: zero-shot LLM direct assignment and LLM-guided integer programming. Empirical findings show that LLMs alone can produce reasonable initial allocations but may violate hard constraints, whereas combining LLMs with an ILP solver achieves 100% completeness and balanced utilization.

## Background & Motivation

**Background**: 5G network slicing creates isolated virtual networks over shared infrastructure to support diverse service requirements, including eMBB (high bandwidth), URLLC (low latency), and mMTC (massive IoT connectivity).

**Limitations of Prior Work**: Traditional approaches rely on large-scale optimization or heuristic algorithms that are computationally intensive and require expert-level parameter tuning. ILP faces combinatorial explosion as problem size grows.

**Core Idea**: Leverage the semantic comprehension capabilities of LLMs to interpret service request characteristics and generate initial assignments, followed by ILP-based exact solving to guarantee constraint satisfaction.

## Method

### Overall Architecture
The system encompasses three slice types (eMBB for high bandwidth, URLLC for low latency, mMTC for massive connectivity) and a set of user service requests, each characterized by resource demands and latency requirements. The problem is formulated as a constrained assignment problem, where the decision variable $x_{i,m}$ indicates whether request $i$ is assigned to slice $m$.

### Key Designs

1. **Zero-Shot LLM Assignment**:

    - **Function**: Directly generates slice assignment solutions.
    - **Mechanism**: Claude models are provided with slice attributes (capacity, latency) and a list of requests (demand, latency requirements), and are prompted to output assignments in CSV format. The "@" symbol is used as a delimiter to avoid comma ambiguity, and request ordering is shuffled to prevent positional bias.
    - **Design Motivation**: The semantic understanding of LLMs can naturally map "high-bandwidth video streams → eMBB slice" and "low-latency control signals → URLLC slice."
    - **Limitation**: At temperature 0.8, outputs exhibit variability and may violate hard capacity or latency constraints.

2. **LLM-Guided ILP Hybrid Method**:

    - **Function**: Precisely solves constrained assignments while leveraging semantic information from LLMs.
    - **Mechanism**: Claude first evaluates the pairwise compatibility of all requests to produce a binary similarity matrix. An ILP is then formulated to maximize the total similarity score of co-assigned request pairs within the same slice. The auxiliary variable $z_{i,j,m}$ links the joint assignment of request pairs, ensuring all hard constraints (capacity, latency, full assignment) are satisfied.
    - **Design Motivation**: LLMs excel at understanding service semantics (determining which requests "belong" in the same slice), while ILP excels at exact constraint enforcement.

3. **Evaluation Metrics**:

    - **Completeness**: Whether all requests are assigned.
    - **Homogeneity**: Consistency of request types within each slice.
    - **Bandwidth/Density Utilization**: Degree of balanced resource usage across slices.

### Loss & Training
No training is involved. LLMs operate via zero-shot inference, and the ILP is solved using the GLPK solver.

## Key Experimental Results

### Main Results

| Method | Completeness | Homogeneity | Constraint Satisfaction | Notes |
|---|---|---|---|---|
| Claude-3-haiku | 100% | 0.35±0.27 | Occasional violations | Poor homogeneity |
| Claude-3-sonnet | 99.5% | 1.00±0.00 | Occasional violations | Perfect homogeneity |
| ILP+LLM | 100% | High | 100% | Most balanced solution |

### Ablation Study

| Model Variant | Bandwidth Utilization | Density Utilization | Notes |
|---|---|---|---|
| claude-3-haiku | Severely imbalanced | Slice C overloaded | Weaker semantic understanding |
| claude-3-sonnet | 90–100% balanced | Balanced | Best standalone LLM performance |
| claude-3.5-sonnet | Moderately balanced | Moderate | Intermediate performance |

### Key Findings
- Claude-3-sonnet achieves a perfect homogeneity score of 1.0, demonstrating that its semantic understanding can reliably distinguish between different service request types.
- Constraint violations in standalone LLM usage primarily arise from slice capacity overflow, as LLMs cannot accurately track cumulative resource consumption.
- The ILP+LLM hybrid combines the semantic clustering capability of LLMs with the constraint guarantees of ILP, yielding the optimal solution.

## Highlights & Insights
- The LLM–solver hybrid architecture represents a promising paradigm: LLMs provide semantic understanding to reduce the search space, while solvers guarantee constraint satisfaction. This pattern is generalizable to other constrained optimization problems.
- The paper provides clear empirical analysis of LLM capability boundaries in structured decision-making tasks: strong in semantic understanding, weak in tracking numerical constraints.
- The substantial performance variation across Claude variants highlights the significant impact of model selection on task quality.

## Limitations & Future Work
- The study relies on synthetic datasets and has not been validated on real 5G network deployments.
- Only static, one-shot allocation is addressed; dynamic and time-varying scenarios are not considered.
- The problem scale is relatively small (~20 requests), and scalability remains to be verified.
- LLM similarity judgments use binary values; continuous-valued similarity scores may yield greater precision.
- The potential of other LLMs (e.g., GPT-4) as semantic engines has not been explored.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The LLM+ILP hybrid approach is relatively novel in the networking domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic data with limited scale.
- **Writing Quality**: ⭐⭐⭐⭐ Clear but lacking in depth.
- **Value**: ⭐⭐⭐⭐ An exploratory work demonstrating the potential of the hybrid approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Bayes](large_language_bayes.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->
