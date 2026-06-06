---
title: >-
  [Paper Note] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models
description: >-
  [ICML 2026][Time Series][Continuous-Time Dynamic Graphs] CTDG-SSM is the first to simultaneously capture multi-hop long-range spatial (LRS) and long-range temporal (LRT) dependencies in dynamic graphs via **topology-awar…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Continuous-Time Dynamic Graphs"
  - "State Space Models"
  - "Long-Range Dependencies"
  - "Spatio-Temporal Representation Learning"
date: 2026-05-08
content_hash: 123ed49e204d8074
---

# Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models

**Conference**: ICML 2026  
**arXiv**: [2606.04672](https://arxiv.org/abs/2606.04672)  
**Code**: TBD  
**Area**: Time Series / Graph Learning / Dynamic Graphs  
**Keywords**: Continuous-Time Dynamic Graphs, State Space Models, Long-Range Dependencies, Spatio-Temporal Representation Learning

## TL;DR
CTDG-SSM is the first to simultaneously capture multi-hop long-range spatial (LRS) and long-range temporal (LRT) dependencies in dynamic graphs via **topology-aware HiPPO projection** and state space models. It outperforms SOTA on tasks like link prediction and node classification with only 1/10 of the parameters of competing methods.

## Background & Motivation

**Background**: Continuous-Time Dynamic Graphs (CTDGs) provide a powerful framework for modeling evolving relational data. Existing methods generally fall into two categories: event-driven models (TGAT, TGN), which are computationally efficient but struggle to retain history over long time scales (weak LRT capability), and sequence model variants (DyGFormer, DyGmamba), which capture LRT but restrict attention to 1-hop neighborhoods during preprocessing, losing multi-hop global spatial structural information (weak LRS capability).

**Limitations of Prior Work**: No existing method can simultaneously maintain LRS and LRT, which is critical in real-world applications such as financial fraud detection (where money laundering often spreads through long transaction chains rather than isolated local interactions).

**Key Challenge**: The "spatial-temporal trade-off" dilemma—either breaking the graph structure to capture LRT or limiting the temporal receptive field to utilize the graph structure.

**Goal**: To develop a unified spatio-temporal state space framework that can both compress historical event information into a compact memory (LRT) and aggregate multi-hop neighborhood information through graph polynomial filters (LRS).

**Key Insight**: Extend the classical HiPPO (High-order Polynomial Projection Operator) to graph data. The key observation is that by projecting classical HiPPO coefficients onto the inverse of a Laplacian matrix polynomial, one can obtain a state space model that encodes both temporal and topological dynamics simultaneously.

**Core Idea**: Topology-aware High-order Polynomial Projection (CTT-HiPPO) replaces simple sequence memory mechanisms. Memory coefficients are constrained by both temporal evolution and graph structure, with efficient computation achieved through Zero-Order Hold discretization.

## Method

### Overall Architecture
1. Input: Continuous-time event stream $\{(u, v, t_i)\}$.
2. Subgraph Sampling: Construct batch-level Laplacian matrices $L_B[k]$ by sampling $N_u$ nearest neighbors for each node per batch.
3. Node Feature Encoding: Static node embeddings + dynamic neighborhood features + edge attributes + time encoding are projected into a $d$-dimensional latent space via a 2-layer encoder.
4. CTDG-SSM Layers: Multiple stacked layers, each containing RMSNorm → CTDG-SSM recurrence → GeLU → Residual connection.
5. Output: Latent states from the final layer + static embedding aggregation are processed by an MLP decoder to generate link prediction scores or node classification probabilities.

### Key Designs

1. **Topology-aware High-order Polynomial Projection (CTT-HiPPO)**:
    - **Function**: Designed as a memory compression mechanism for CTDGs that jointly encodes time and graph topology.
    - **Mechanism**: Over a time window $[0, \tau]$, the $i$-th dimension of node features is modeled as $X_{:, i}(t) = p(L_\tau) H_\tau^{(i)} g(t) + r_i(t)$, where $g(t)$ represents orthogonal polynomial bases and $p(L_\tau)$ is a Laplacian matrix polynomial (graph filter). First-order optimality conditions yield $H_\tau^{(i)} = p(L_\tau)^{-1} H_\tau^{(i), \text{HiPPO}}$, where classical HiPPO coefficients are projected through the inverse polynomial filter.
    - **Design Motivation**: Inherits the optimal compression characteristics of HiPPO for time series while injecting graph topological constraints through $p(L_\tau)$. A $K$-th order polynomial filter aggregates $K$-hop neighborhoods automatically, and as the graph structure evolves, the Laplacian matrix changes over time, allowing the filter to adapt dynamically.

2. **Continuous-Time State Space Model (CTDG-SSM)**:
    - **Function**: Models the temporal evolution of the CTT-HiPPO coefficient matrix $H_s$.
    - **Mechanism**: The evolution of memory coefficients is shown to satisfy the differential equation $\frac{d H_s}{d s} = -\frac{H_s A^\top}{M(s)} - p(L_s)^{-1} \frac{d p(L_s)}{d s} H_s + \frac{p(L_s)^{-1} X(s) B^\top}{M(s)}$. The first term represents temporal memory decay, the second compensates for graph topology changes, and the third integrates new observations. Discretization via Zero-Order Hold (ZOH) results in the recurrence $H[k+1] = \bar{A}_{L[k]} H[k] \bar{A} + \bar{B}(L[k], X[k])$.
    - **Design Motivation**: Unifies graph topology changes and temporal evolution into a single differential equation. It degenerates into a classical SSM when $p(L_\tau) = I$ (no graph) and into a piecewise-constant SSM when the graph structure is fixed.

3. **Efficient Discrete Implementation + Robustness Guarantees**:
    - **Function**: Enables scalable and efficient training and inference while providing theoretical guarantees for stability under graph perturbations and node permutation equivariance.
    - **Mechanism**: Batch-level subgraph sampling avoids dense graph Laplacian computations, requiring only $N_B \times N_B$ operations. Residual connections and RMSNorm draw from modern architectures like Mamba. It is proven that the relative error of CTT-HiPPO coefficients is bounded linearly by $\epsilon$ when the Laplacian matrix is perturbed by $\|\Delta L\|_2 \leq \epsilon$, ensuring node permutation equivariance.
    - **Design Motivation**: Avoids the overhead of whole-graph operations, controlling computational complexity through neighborhood sampling while retaining multi-hop information. Theoretical guarantees ensure stability against graph noise.

## Key Experimental Results

### Main Results (Dynamic Link Prediction, AUC ROC)

| Dataset | JODIE | TGN | DyGmamba | **CTDG-SSM** |
|--------|-------|-----|----------|-------------|
| LastFM | 70.89 | 76.64 | 93.31 | **93.79** |
| Enron | 87.77 | 88.72 | 93.34 | **94.98** |
| MOOC | 84.50 | 91.91 | 89.58 | **99.00** |
| Reddit | 98.29 | 98.61 | 99.27 | **99.48** |
| **Avg. Rank** | 7.93 | 4.57 | 2.00 | **1.86** |

CTDG-SSM leads significantly on LRT benchmarks (LastFM, MOOC, Enron), with a 9.4% relative improvement over DyGmamba on MOOC.

### Ablation Study (Sequence Classification, Long-Range Dependency Test)

| Variant | n=3 | n=9 | n=15 | n=20 | Average |
|------|-----|-----|------|------|------|
| TU-SSM (No Topology) | 47.0 | 50.7 | 52.3 | 54.5 | 51.1% |
| CTDG-SSM (FO, 1st order) | 100.0 | 97.1 | 97.4 | 97.1 | 97.9% |
| **CTDG-SSM (SO, 2nd order)** | **100.0** | **98.1** | **97.8** | **98.6** | **98.6%** |

### Efficiency Comparison

| Metric | CTDG-SSM | DyGmamba | DyGFormer |
|------|----------|----------|-----------|
| Parameters (Relative) | 1× | ~10× | ~8× |
| LastFM Training Time / epoch | 4.45 min | 28.45 min | 47.00 min |
| GPU Memory | 1.15 GB | 4.17 GB | 7.57 GB |

### Key Findings
- Removing the topological term (TU-SSM) causes performance to collapse from 98% to 51%, confirming the critical role of structured memory updates.
- Second-order polynomials show significant improvement over first-order for long sequences (n=20: 97.13% → 98.60%).
- The model achieves SOTA performance with 1/10 the parameters, 6.4× faster training speed, and 3.6× less GPU memory than competing methods.

## Highlights & Insights
- **Theoretical Depth**: Derives a topology-aware variant from classical HiPPO, skillfully injecting graph filtering into temporal memory; the derivation is more principled than heuristic designs.
- **Unified Spatio-Temporal Framework**: Instead of a "time-then-space" pipeline, it naturally couples both within the memory dynamics through a differential equation.
- **Parameter Efficiency**: Reaches SOTA using only polynomial filter coefficients and state transition matrices, with 1/10 the parameters of competitors, offering practical value for model compression and edge deployment.
- **Transferable Design**: The CTT-HiPPO concept (projection via inverse filters) can be generalized to other tasks requiring joint spatio-temporal modeling.

## Limitations & Future Work
- The subgraph sampling strategy is fixed ($N_u$ nearest neighbors); future work could explore learnable sampling.
- The Zero-Order Hold assumption may be imprecise for high-frequency graph changes; higher-order discretization could be attempted.
- Lack of interpretability—missing visualization analysis of "what the learned filters represent."
- Does not cover scenarios with extreme variances in event time intervals (sparse sensor data + bursts).

## Related Work & Insights
- **vs. DyGmamba**: The latter only captures LRT with limited structural awareness; CTDG-SSM captures both LRS and LRT via topological terms.
- **vs. GraphSSM**: Designed for discrete graphs assuming fixed structures; CTDG-SSM adapts to continuous event streams and dynamic topologies.
- **vs. Transformer Variants (DyGFormer)**: These use attention but are limited to 1-hop; CTDG-SSM implicitly aggregates multiple hops via graph filters with lower computational cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify LRT and LRS from a state space perspective with rigorous theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully validated across three task types with deep ablation and comprehensive efficiency comparisons; room remains for interpretability and extreme scenario testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, mathematical rigor, and a complete closed loop from motivation to theory to experiments.
- Value: ⭐⭐⭐⭐⭐ Solves important CTDG modeling problems with parameter efficiency valuable for real-world deployment and strong theoretical transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[ICML 2026\] HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)
- [\[AAAI 2026\] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](../../AAAI2026/time_series/loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)
- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](../../NeurIPS2025/time_series/walrus_wavelets_for_long-range_representation_using_ssms.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](nested_spatio-temporal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
