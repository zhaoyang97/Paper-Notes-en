---
title: >-
  [Paper Note] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting
description: >-
  [ICML 2025][Time Series][Irregular Multivariate Time Series] HyperIMTS is proposed to represent the observations and their dependencies in irregular multivariate time series (IMTS) using a hypergraph structure. By leveraging three message passing mechanisms (node-to-hyperedge, hyperedge-to-hyperedge, and hyperedge-to-node), it achieves irregularity-aware temporal and variable dependency learning. It achieves SOTA performance on 5 IMTS datasets with superior computational effi…
tags:
  - "ICML 2025"
  - "Time Series"
  - "Irregular Multivariate Time Series"
  - "Hypergraph Neural Networks"
  - "Time-Aware Message Passing"
  - "Variable Dependency Modeling"
  - "Forecasting"
date: 2026-05-08
content_hash: 886514b07553ba72
---

# HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting

**Conference**: ICML 2025  
**arXiv**: [2505.17431](https://arxiv.org/abs/2505.17431)  
**Code**: [https://github.com/qianlima-lab/PyOmniTS](https://github.com/qianlima-lab/PyOmniTS)  
**Area**: Time Series Forecasting  
**Keywords**: Irregular Multivariate Time Series, Hypergraph Neural Networks, Time-Aware Message Passing, Variable Dependency Modeling, Forecasting

## TL;DR
HyperIMTS is proposed to represent the observations and their dependencies in irregular multivariate time series (IMTS) using a hypergraph structure. By leveraging three message passing mechanisms (node-to-hyperedge, hyperedge-to-hyperedge, and hyperedge-to-node), it achieves irregularity-aware temporal and variable dependency learning. It achieves SOTA performance on 5 IMTS datasets with superior computational efficiency compared to padding methods.

## Background & Motivation
Multivariate time series (MTS) are widely present in fields such as healthcare, meteorology, and biomechanics. In reality, due to sensor failures, inconsistent sampling frequencies, and human factors, time series are often **irregular** (IMTS): the sampling time intervals for each variable are unequal, and the observation time points across different variables are misaligned.

Existing methods for handling IMTS fall into two major categories, each with its own limitations:

**Padding methods**: Pad irregular sequences into regular matrices, including time-aligned padding and patch-aligned padding. The downside is that they significantly increase the volume of data (e.g., the original average of 304.8 observation points in the MIMIC-IV dataset inflates to 92,000 after padding) and may destroy the original sampling patterns.

**Non-padding methods (set/bipartite graph)**: Treat observations as set elements or represent them using bipartite graphs. The limitation of sets is their inability to capture dependencies between observations. For bipartite graphs, they struggle to propagate messages when variables lack shared timestamps (e.g., if variables V2 and V3 share no joint time-alignment points, they are completely disconnected in the bipartite graph).

**Key Challenge**: How to **fully capture the temporal and variable dependencies among all observations** without **any padding** (thus maintaining high efficiency)?

The **Key Insight** of this work is to leverage **hypergraphs**: hyperedges can connect an arbitrary number of nodes. By treating each observation value as a node, connecting all observations of the same variable with a variable hyperedge, and connecting all observations at the same time point with a temporal hyperedge, information can be indirectly exchanged via hyperedge-to-hyperedge message passing even if variables do not share identical timestamps.

## Method

### Overall Architecture
The pipeline of HyperIMTS consists of: (1) transforming IMTS samples into hypergraph representations, where observation values are treated as nodes, and timestamps and variables are treated as two types of hyperedges; (2) updating embeddings via a three-level message passing mechanism: node-to-hyperedge (aggregating observation information to temporal/variable hyperedges) $\rightarrow$ hyperedge-to-hyperedge (inter-variable interaction) $\rightarrow$ hyperedge-to-node (propagating aggregated information back to observation nodes); (3) mapping final node embeddings through a linear layer to output the forecasted values.

### Key Designs

1. **Efficient Hypergraph Representation**:

    - Function: Represents IMTS samples as a hypergraph $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$ without any padding.
    - Mechanism: Each observation $(t_j, z_j, u_j)$ corresponds to a node $v_j$. Two classes of hyperedges are defined: temporal hyperedges $\mathcal{E}_{\text{time}} = \{e_t | t=1,...,T\}$ (connecting all observations at the same timestamp to the same hyperedge) and variable hyperedges $\mathcal{E}_{\text{var}} = \{e_u | u=1,...,U\}$ (connecting all observations of the same variable to the same hyperedge). The topology is represented by two incidence matrices $\mathbf{H}^T \in \mathbb{R}^{M \times T}$ and $\mathbf{H}^U \in \mathbb{R}^{M \times U}$.
    - Node Initialization: $\mathbf{V} = \text{ReLU}(\text{FF}_{\text{obs}}(\mathcal{Z}_i))$, temporal hyperedges using sinusoidal encoding $\mathbf{E}_{\text{time}} = \sin(\text{FF}_{\text{time}}(T_i))$, and variable hyperedges using learnable parameters $\mathbf{E}_{\text{var}} = \text{ReLU}(\mathbf{W}_{\text{var}})$.
    - Design Motivation: Compared to padding methods, the hypergraph only processes actually existing observations ($M$ nodes) instead of a dense $T \times U$ matrix; compared to bipartite graphs, hyperedges can connect an arbitrary number of nodes and support direct information exchange between hyperedges.

2. **Node-to-Hyperedge Message Passing**:

    - Function: Aggregating observation node information to temporal and variable hyperedges.
    - Mechanism: Implemented using multi-head attention. Taking temporal hyperedges as an example, the query is $\mathbf{q}^h = \text{FF}_q(\mathbf{E}_{\text{time}})$, and the key-value pair is generated by concatenating node embeddings with variable hyperedge embeddings $\mathbf{k}^h = \text{FF}_k(\mathbf{V} || \mathbf{E}_{\text{var}})$:
    $$\mathbf{O} = ||_{h=1}^{H} \text{Softmax}\left(\frac{\mathbf{q}^h {\mathbf{k}^h}^\intercal}{\sqrt{d/H}}\right) \mathbf{v}^h$$
    $$\mathbf{E}_{\text{time}}' = \mathbf{O} + \text{ReLU}(\text{FF}_O(\mathbf{O}))$$
    The update for variable hyperedges is similar, but temporal information is concatenated into the key-value representation to distinguish temporal coordinates.
    - Design Motivation: Concatenating the collinear hyperedge from the alternative dimension as part of the key-value representations allows the temporal hyperedge to distinguish observations from different variables during aggregation, and vice versa.

3. **Irregularity-Aware Inter-Variable Message Passing**:

    - Function: Transmitting information between variable hyperedges to model inter-variable dependencies.
    - Mechanism: Computes and adaptively fuses two types of variable similarities. For variables $u_a$ and $u_b$:
        - Series-level similarity: $\mathbf{S}_{\text{var}} = \text{FF}_q(e_{u_a}) \cdot \text{FF}_k(e_{u_b})^\intercal$
        - Time-aware similarity (using only aligned observations): $\mathbf{S}_{\text{obs}} = [v_1^{u_a},...,v_{T_{\text{shared}}}^{u_a}] \cdot [v_1^{u_b},...,v_{T_{\text{shared}}}^{u_b}]^\intercal$
        - Adaptive fusion: $\mathbf{S}_{\text{IMTS}} = \alpha \mathbf{S}_{\text{obs}} + (1-\alpha) \mathbf{S}_{\text{var}}$
        - Where $\alpha = T_{\text{shared}} / T_{\text{total}}$ (when $\mathbf{S}_{\text{var}} > \delta$ and $\mathbf{S}_{\text{obs}} \neq 0$), else $\alpha = 0$.
        - Final propagation via attention: $\mathbf{E}_{\text{var}}'' = \text{Softmax}(\mathbf{A}_{\text{var}} / \sqrt{d}) \text{FF}_v(\mathbf{E}_{\text{var}}')$
    - Design Motivation: When two variables share more time-aligned observations, the fine-grained similarity based on aligned observations is more reliable. When completely unaligned ($\mathbf{S}_{\text{obs}}=0$), it falls back to the series-level similarity to prevent disconnection between variables. $\delta$ is a learnable threshold (initialized to 0.5) that controls whether to use the time-aware similarity.

4. **Hyperedge-to-Node Message Passing**:

    - Function: Transmitting aggregated hyperedge information back to the nodes to update observation embeddings.
    - Mechanism: First performs self-attention on nodes, then concatenates with temporal and variable hyperedge information for updates:
    $$\mathbf{V}' = \text{SelfAtten}(\mathbf{V})$$
    $$\mathbf{V}'' = \text{ReLU}(\mathbf{V} + \text{FF}_{\text{node}}(\mathbf{V}' || \mathbf{E}_{\text{time}}' || \mathbf{E}_{\text{var}}'))$$
    Key Design: Use $\mathbf{E}_{\text{var}}'$ (which has not undergone inter-variable message passing) in the first $L-1$ layers to learn temporal and intra-variable dependencies, and only use $\mathbf{E}_{\text{var}}''$ in the final layer to incorporate inter-variable dependencies.
    - Design Motivation: Allows the model to fully capture temporal patterns first and only introduce inter-variable correlations in the final stage, preventing early propagation of cross-variable noise.

### Loss & Training
- Output mapping: $\hat{\mathcal{Z}_i} = \text{FF}_{\text{out}}(\mathbf{V}'' || \mathbf{E}_{\text{time}}' || \mathbf{E}_{\text{var}}'')$
- Training Loss: Standard MSE, computed only on the observed values at the predicted positions.
- Learning rate strategy: Constant for the first 3 epochs, and exponentially decayed as $\mathcal{L}_n = \mathcal{L}_0 \times 0.8^{n-3}$ thereafter.
- Max 300 epochs, early stopping patience=10, with mean and standard deviation computed over 5 random seeds.

## Key Experimental Results

### Main Results

| Dataset | Metric (MSE) | HyperIMTS | Prev. SOTA (GraFITi) | Gain |
|--------|-----------|-----------|-------------------|------|
| MIMIC-III | MSE | **0.4259** | 0.4534 | -6.1% |
| MIMIC-IV | MSE | **0.2174** | 0.2454 | -11.4% |
| PhysioNet'12 | MSE | **0.2996** | 0.3060 | -2.1% |
| Human Activity | MSE | **0.0421** | 0.0435 | -3.2% |
| USHCN | MSE | 0.1738 | 0.2026 | -14.2% |

Ours ranks first in 4 out of 5 datasets. On USHCN, GRU-D (0.1639) is better but exhibits high variance. A broad baseline of 27 models (16 regular TS methods + 11 irregular TS methods) was compared.

### Ablation Study

| Configuration | MIMIC-III MSE | MIMIC-IV MSE | Description |
|------|-------------|-------------|------|
| Complete | 0.4259 | 0.2174 | Complete model |
| w/o VE (w/o variable hyperedges) | 0.9556 | 0.6293 | Suffers the worst degradation, demonstrating the indispensability of variable modeling |
| w/o IAVD (w/o irregularity-aware variable dependency) | 0.4466 | 0.2358 | Confirms the necessity of hyperedge-to-hyperedge message passing |
| rp IAVD (using only series-level similarity) | 0.4317 | 0.2189 | Time-aware similarity provides an additional contribution |
| w/o TE (w/o temporal hyperedges) | 0.4954 | 0.2652 | Temporal information is crucial |
| rp TE (non-learnable time encoding) | 0.4403 | 0.2333 | Learnable time encoding outperforms fixed encoding |

### Key Findings
- **Variable hyperedges are the most critical component**: Removing them degrades the MSE by approximately $2\times$ (MIMIC-III: 0.4259 $\rightarrow$ 0.9556), demonstrating that capturing inter-variable dependencies is vital to IMTS.
- **Irregularity-aware variable similarity outperforms series-level similarity**: Comparing rp IAVD with Complete indicates that leveraging fine-grained similarity at aligned timestamps is indeed valuable.
- **Clear efficiency advantage**: On MIMIC-III, HyperIMTS is faster in training than all padding methods (e.g., Warpformer, GNeuralFlow) and also has lower memory consumption.
- **Some regular TS methods remain competitive on IMTS**: Approaches like Crossformer and PatchTST show that TS benchmarking for IMTS requires comparisons with a broader range of baselines.
- **Pre-trained models (MOIRAI, PrimeNet) perform poorly**: The distribution shift between regular and irregular data likely requires dedicated pre-training strategies.

## Highlights & Insights
- Hypergraph representation is a natural choice for handling IMTS — a single hyperedge connects an arbitrary number of nodes, perfectly matching the varying number of observations for different variables at the same timestamp.
- The design of "irregularity-aware" variable similarity is elegant: it adaptively balances time-aware similarity and series-level similarity based on alignment degree, where $\alpha = T_{\text{shared}} / T_{\text{total}}$ is highly intuitive.
- The "progressive" design of introducing inter-variable dependency only in the final layer effectively avoids premature interference from cross-variable noise.
- A unified benchmark is constructed: a fair comparison across 27 baselines and 5 datasets, alongside the open-sourced PyOmniTS pipeline, providing standardized tools for IMTS research.
- No padding is required, leading to highly efficient data processing: on the MIMIC-IV dataset, the original 304.8 observations contrast starkly with the 92,000 observations after padding.

## Limitations & Future Work
- **No support for multimodal data**: Clinical IMTS datasets can contain text notes or images, which the current model cannot utilize.
- **Attention computational cost**: Although more efficient than padding methods, the $O(M^2)$ complexity of self-attention can still become a bottleneck when observations are extremely dense.
- The variance of results on the USHCN dataset is relatively high (0.0078), suggesting room for improvement in training stability.
- Hyperedge-to-hyperedge message passing is limited to the variable dimension and does not explicitly model interactions between temporal hyperedges, potentially missing certain temporal patterns.
- The number of variable hyperedges $U$ is assumed to be fixed, leaving scenarios with dynamically changing variables (e.g., IoT sensors going online/offline) unaddressed.

## Related Work & Insights
- **vs GraFITi (bipartite graph)**: GraFITi represents IMTS with bipartite graphs but fails to propagate messages between variables without shared timestamps. HyperIMTS resolves this via hyperedge-to-hyperedge message passing, reducing MSE by up to 11.4%.
- **vs tPatchGNN (patch-aligned padding)**: tPatchGNN still requires padding, showing much higher memory footprints on MIMIC-III than HyperIMTS while yielding inferior performance.
- **vs Warpformer (canonical padding)**: Warpformer employs full-time alignment padding, which is slow to train, memory-intensive, and less accurate than HyperIMTS.
- **vs standard time series models (PatchTST, Crossformer)**: These models do not explicitly handle irregularity but remain somewhat competitive after padding, highlighting the necessity of including broader baselines in IMTS evaluations.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply hypergraphs to IMTS forecasting with irregularity-aware message passing, though hypergraphs are already widely utilized in other domains.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 27 baselines, 5 datasets, detailed ablation studies, efficiency analyses, and variable-length analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations (particularly Figures 1 and 3), with a coherent presentation of problem motivations and methodologies.
- Value: ⭐⭐⭐⭐ Provides a robust IMTS benchmarking framework and data-processing pipeline with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[ICML 2025\] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICLR 2026\] ASTGI: Adaptive Spatio-Temporal Graph Interactions for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/astgi_adaptive_spatio-temporal_graph_interactions_for_irregular_multivariate_tim.md)

</div>

<!-- RELATED:END -->
