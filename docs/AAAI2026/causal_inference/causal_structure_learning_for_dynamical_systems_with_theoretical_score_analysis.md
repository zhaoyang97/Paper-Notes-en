---
title: >-
  [Paper Note] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis
description: >-
  [AAAI 2026][Causal Inference][Causal Discovery] This paper proposes CaDyT, which combines Gaussian process-based continuous-time dynamics modeling (via Adams-Bashforth integrators for exact inference) with the Minimum Description Length (MDL) principle for structure search. The method simultaneously addresses irregular sampling and causal structure identification, substantially outperforming all baselines on double-mass spring, diamond graph, and Rössler oscillator benchmarks (AUPRC 0.79 vs. runner-up 0.39).
tags:
  - AAAI 2026
  - Causal Inference
  - Causal Discovery
  - Gaussian Processes
  - MDL Principle
  - Dynamical Systems
  - Irregular Sampling
date: 2026-05-08
content_hash: ba12dcd513698137
---

# CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis

**Conference**: AAAI 2026
**arXiv**: [2512.14361](https://arxiv.org/abs/2512.14361)
**Code**: Provided in appendix
**Area**: Causal Inference / Dynamical Systems
**Keywords**: Causal Discovery, Gaussian Processes, MDL Principle, Dynamical Systems, Irregular Sampling

## TL;DR
This paper proposes CaDyT, which combines Gaussian process-based continuous-time dynamics modeling (via Adams-Bashforth integrators for exact inference) with the Minimum Description Length (MDL) principle for structure search. The method simultaneously addresses irregular sampling and causal structure identification, substantially outperforming all baselines on double-mass spring, diamond graph, and Rössler oscillator benchmarks (AUPRC 0.79 vs. runner-up 0.39).

## Background & Motivation

### State of the Field

**State of the Field**: Real-world dynamical systems evolve in continuous time, and causal discovery requires inferring dependencies among variables from observed time series. Existing methods either assume discrete time (DBN-based methods suffer from discretization errors) or cannot handle irregular sampling.

**Limitations of Prior Work**: (1) DyNoTears and VARLiNGAM assume uniform sampling intervals; (2) PC-MCI+ handles irregular data but cannot distinguish causal direction; (3) there are no theoretical scoring guarantees for structural search.

**Root Cause**: Continuous-time modeling and causal structure identification have previously been studied independently and have not been unified.

**Paper Goals**: Address irregular sampling and causal structure learning simultaneously within a unified framework.

**Starting Point**: GP exact inference + Adams-Bashforth multi-step integrators (for continuous-time modeling) + MDL scoring (theoretical guarantee for structure search).

**Core Idea**: Use GP+AB integrators for continuous-time modeling, and MDL scores for causal structure search, which can be theoretically shown to be a BIC-type regularized log-likelihood.

## Method

### Overall Architecture
For each variable $X_i$, the ODE component $\dot{X}_i = F_i(Pa_i)$ is modeled via GP. A greedy search over the causal graph is performed using the MDL score: forward edge addition followed by backward pruning. The three-stage search consists of: (1) edge scoring; (2) forward addition (no-hypercompression test); (3) backward pruning of redundant edges.

### Key Designs

1. **GP + Adams-Bashforth Integrators**: Order $s \in \{1,2,3\}$, with a custom kernel $K(\bar{X}^{(n)}, \bar{X}^{(m)}) = \mathbf{b}_n^\top k(\bar{X}^{[n:n+s]}, \bar{X}^{[m:m+s]}) \mathbf{b}_m$, enabling exact continuous-time inference. Higher-order integrators are more robust for cyclic systems.

2. **MDL Scoring Function**: Total description length $L(\mathcal{T}, M) = L(M) + \sum_i L(\nu_i)$, where model complexity is encoded via SVD of the kernel matrix and noise is encoded via maximum likelihood variance. **Theorem 2** proves this score is a valid regularized log-likelihood, asymptotically analogous to BIC.

3. **Three-Stage Greedy Search**: The forward stage computes gain $\Gamma_{ij}$ (reduction in description length from adding an edge); the backward stage removes redundant edges. Complexity is $O(N^3 D^3 \log D)$.

### Loss & Training
No training is required — GP hyperparameters are optimized via marginal likelihood, and structure search is performed via MDL greedy algorithm.

## Key Experimental Results

### Main Results

| Method | Double-Mass Spring AUPRC↑ | Bilinear AUPRC↑ | Rössler AUPRC↑ |
|--------|--------------------------|-----------------|----------------|
| DyNoTears | 0.22 | 0.59 | 0.34 |
| PC-MCI+ | 0.24 | 0.30 | 0.22 |
| VARLiNGAM | 0.39 | 0.44 | 0.30 |
| **CaDyT** | **0.79** | **0.79** | **0.55** |

### Ablation Study
- **Independent data sanity check**: CaDyT (AB3) discovers 0 spurious edges across 10 independent ODE graphs; VARLiNGAM, PC-MCI+, and DyNoTears exhibit 30%, 60%, and 100% false positive rates, respectively.
- **Integration order**: AB3 significantly outperforms AB1 on cyclic graphs; AB2/AB3 are more robust to irregular sampling.
- Polynomial kernels outperform RBF in certain scenarios (diamond graph AUPRC 0.925 vs. 0.866).

### Key Findings
- CaDyT substantially outperforms all baselines across all benchmarks (AUPRC gap of 0.21–0.55).
- Zero false positives on independent data, compared to 30–100% FPR for baselines.
- Irregular sampling only marginally affects performance; CaDyT maintains its largest advantage.
- The MDL score is theoretically proven to be a valid BIC-type upper bound.

## Highlights & Insights
- **The GP+MDL combination provides both theoretical guarantees and practical effectiveness** — the BIC-type upper bound of MDL ensures consistency of model selection.
- **The zero false positive result** in the sanity check is highly compelling — the high FPR of baseline methods exposes their fundamental limitations.

## Limitations & Future Work
- The $O(N^3)$ GP inference complexity limits applicability to large-scale data.
- Validation is restricted to physical dynamical systems; applicability to biological or social systems remains untested.
- Kernel type selection (RBF vs. polynomial) is required; automatic kernel selection could improve the method.

## Related Work & Insights
- **vs. DyNoTears**: A continuous optimization method that cannot handle irregular sampling and exhibits high false positive rates.
- **vs. PCMCI+**: Based on conditional independence testing, unable to identify causal direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified framework of GP+AB integrators+MDL offers strong theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple physical systems + sanity checks + integration order ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with complete theorem proofs.
- Value: ⭐⭐⭐⭐ Significant advancement for continuous-time causal discovery.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ICLR 2026\] AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](../../ICLR2026/causal_inference/agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)

<!-- RELATED:END -->
