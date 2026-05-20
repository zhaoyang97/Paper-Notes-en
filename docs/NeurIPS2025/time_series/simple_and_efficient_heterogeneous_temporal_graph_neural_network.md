---
title: >-
  [Paper Note] Simple and Efficient Heterogeneous Temporal Graph Neural Network
description: >-
  [NeurIPS 2025][Time Series][Heterogeneous temporal graph] This paper proposes SE-HTGNN, which integrates temporal modeling into spatial learning via a dynamic attention mechanism and initializes attention coefficients us…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Heterogeneous temporal graph"
  - "graph neural network"
  - "dynamic attention"
  - "LLM augmentation"
  - "spatiotemporal modeling"
date: 2026-05-08
content_hash: 01f5cfe92c175fd3
---

# Simple and Efficient Heterogeneous Temporal Graph Neural Network

**Conference**: NeurIPS 2025
**arXiv**: [2510.18467](https://arxiv.org/abs/2510.18467)  
**Code**: Not available  
**Area**: Time Series
**Keywords**: Heterogeneous temporal graph, graph neural network, dynamic attention, LLM augmentation, spatiotemporal modeling

## TL;DR

This paper proposes SE-HTGNN, which integrates temporal modeling into spatial learning via a dynamic attention mechanism and initializes attention coefficients using LLM-generated priors, achieving up to 10× speedup over prior methods while maintaining state-of-the-art predictive accuracy on heterogeneous temporal graph tasks.

## Background & Motivation

Heterogeneous temporal graphs (HTGs) are prevalent in e-commerce networks, epidemic networks, and transportation networks, where each temporal snapshot is a heterogeneous graph containing multiple node types and relation types. Existing heterogeneous dynamic graph neural networks (HDGNNs) suffer from two core problems:

**Excessive model complexity**: Existing methods represent incremental improvements over prior frameworks, continuously stacking attention layers and assigning independent parameters to each snapshot, causing the parameter count to grow linearly with the temporal window length.

**Insufficient information interaction due to decoupled spatiotemporal learning**: Existing frameworks adopt a two-stage sequential strategy—first performing spatial learning on each snapshot, then applying temporal modeling to the spatial representations. This decoupling leads to:
   - The temporal module receiving already "compressed" spatial information, making it difficult to capture global spatiotemporal dependencies
   - Spatial attention being time-agnostic, with attention coefficients computed independently per snapshot, producing **attention discontinuity**—the inability to leverage historical attention information for capturing consistent long-term patterns

## Method

### Overall Architecture

SE-HTGNN consists of three modules: (1) a dynamic attention-based graph learning module that integrates temporal modeling into spatial learning; (2) an LLM-augmented prompt module that initializes attention coefficients with LLM-generated prior knowledge; and (3) a linear projection module that maps spatiotemporal representations to future prediction steps.

### Key Designs

1. **Simplified neighbor aggregation**: Node-level attention (e.g., GAT) is discarded in favor of non-parametric GCN-style aggregation. The key observation is that same-type neighbors exhibit low variance in HTGs, rendering fine-grained node-level attention unnecessary. The aggregation formula is:

    $\mathbf{H}^{t}_{v,r} = \sigma(\mathbf{A}_{r}^{t} \mathbf{H}^{t}_{\mathcal{N}_{r}^{t}(v)})$

   where $\mathbf{A}_{r}^{t}$ is the normalized adjacency matrix for relation $r$ at time $t$. This substantially reduces parameter count and alleviates optimization difficulties.

2. **Dynamic attention fusion**: This is the paper's core contribution. Unlike conventional methods that compute attention independently for each snapshot, dynamic attention employs a GRU to generate attention coefficients sequentially, storing historical attention information in the GRU hidden state to guide attention computation in subsequent snapshots:

    $\mathbf{e}^{t}_{v,r} = \text{GRU}_{r}(\mathbf{H}_{v,r}^{t}, \mathbf{e}^{t-1}_{v,r})$

    $\alpha_{r}^{t} = \frac{\exp(\overline{\mathbf{e}}_{v,r}^{t})}{\sum_{r' \in \mathcal{R}(v)} \exp(\overline{\mathbf{e}}_{v,r'}^{t})}$

   Relation-level GRUs independently capture the evolution of different relation types. The final representation is obtained via attention-weighted fusion: $\mathbf{H}_{v}^{t} = \sum_{r} \alpha_{r}^{t} \cdot \mathbf{H}^{t}_{v,r}$. Temporal information is thus embedded within spatial learning, eliminating the need for a separate temporal modeling module.

3. **LLM-augmented attention initialization**: The GRU's initial hidden state $\mathbf{e}^{0}_{v,r}$ is critical for model convergence. The paper uses an LLM (LLaMA3-8B) to generate semantic representations for each node type, and then computes initial attention coefficients based on the similarity between source and target node type representations:

    $\beta_{r} = \mathbf{Q}_{u} \mathbf{K}_{v}^{\top}, \quad \mathbf{e}_{v,r}^{0} = \frac{\exp(\beta_{r})}{\sum_{r'} \exp(\beta_{r'})}$

   Since attention operates at the relation level, the number of prompts processed by the LLM depends on the number of node types rather than the total number of nodes, making the approach computationally efficient.

### Loss & Training

- **Link prediction**: Binary cross-entropy loss with positive/negative sample contrast
- **Node classification**: Cross-entropy loss with MLP projection to class dimensions
- **Node regression**: MAE loss
- LLM inference can be completed in the preprocessing stage, incurring no additional memory overhead during training

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | SE-HTGNN | CasMLN (Prev. SOTA) | DHGAS | Gain |
|---|---|---|---|---|---|
| OGBN-MAG (link prediction) | AUC% | **93.13** | 90.85 | OOM | +2.11% |
| OGBN-MAG (link prediction) | AP% | **92.71** | 89.47 | OOM | +3.62% |
| Aminer (link prediction) | AUC% | **91.08** | 88.53 | 88.13 | +2.89% |
| YELP (node classification) | Macro-F1% | **44.24** | 42.21 | 41.99 | +4.81% |
| COVID-19 30-day (node regression) | MAE↓ | **497** | 544 | 536 | +7.27% |
| COVID-19 90-day (node regression) | MAE↓ | **1001** | 1084 | 1692 | +6.97% |

### Ablation Study

| Configuration | OGBN-MAG AUC% | Aminer AUC% | YELP F1% | COVID MAE↓ |
|---|---|---|---|---|
| SE-HTGNN (full) | **93.13** | **91.08** | **44.24** | **497** |
| w/o LLM (random init.) | 90.87 | 87.91 | 41.05 | 542 |
| w/o LLM (zero init.) | 91.78 | 89.98 | 43.31 | 524 |
| w/o dynamic attention (projection attention) | 86.83 | 85.42 | 38.19 | 574 |
| w/o dynamic attention (gated attention) | 87.94 | 87.42 | 38.96 | 574 |
| w/o neighbor aggregation (no aggregation) | 83.91 | 62.47 | 35.27 | 672 |

### Key Findings

1. **Dynamic attention is the most critical component**: Its removal causes a sharp performance drop (AUC decreases by 6%+), confirming the importance of integrating temporal information into attention computation.
2. **LLM initialization is effective but not indispensable**: Zero initialization also achieves competitive results, but LLM-provided priors accelerate convergence.
3. **Simplified aggregation outperforms complex alternatives**: Non-parametric GCN aggregation surpasses GAT, indicating that fine-grained attention among same-type neighbors is unnecessary.
4. **Significant efficiency advantage**: SE-HTGNN achieves up to 10× speedup over SOTA baselines without incurring out-of-memory errors.

## Highlights & Insights

- **Paradigm innovation**: Unifying temporal modeling within spatial learning—replacing the conventional two-stage decoupled framework with GRU-driven dynamic attention—represents a concise and elegant design.
- **LLM as a prior knowledge injector**: Rather than using the LLM for direct prediction, the approach extracts its semantic understanding to initialize attention, yielding high returns at low cost.
- **Attention continuity**: The paper addresses the attention discontinuity across snapshots, enabling attention coefficients to evolve smoothly over time.

## Limitations & Future Work

1. The sequential nature of GRU limits training parallelism; more efficient temporal fusion mechanisms warrant exploration.
2. Experiments do not validate performance on very large-scale HTGs (millions of nodes).
3. The LLM augmentation module relies on predefined node type descriptions, requiring manual prompt engineering when transferring to new domains.

## Related Work & Insights

This paper bridges static heterogeneous GNNs (HAN, SeHGNN) and dynamic graph networks (DyHATR, HTGNN), demonstrating that a simplified architecture with unified spatiotemporal modeling outperforms complex stacked designs. The idea of using LLMs as graph learning priors is generalizable to other graph tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paradigm of unifying spatiotemporal modeling via dynamic attention is notably innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four task types with comprehensive ablation and variant analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ Provides an efficient and powerful new baseline for heterogeneous temporal graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)
- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)

</div>

<!-- RELATED:END -->
