---
title: >-
  [Paper Note] Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction
description: >-
  [AAAI 2026][Autonomous Driving][Dynamic Graph Learning] GLFormer is proposed, a lightweight attention-free Transformer framework for dynamic graph link prediction. It replaces self-attention with an adaptive token mixer based on interaction order and time intervals, combined with a hierarchical aggregation mechanism to expand the temporal receptive field. It achieves comparable or superior performance to Transformer baselines across six benchmarks while significantly reducing…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Dynamic Graph Learning"
  - "Link Prediction"
  - "Attention Mechanism Alternatives"
  - "Adaptive Token Mixing"
  - "Hierarchical Aggregation"
date: 2026-05-08
content_hash: 7763f79f502519b0
---

# Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction

**Conference**: AAAI 2026  
**arXiv**: [2511.12442](https://arxiv.org/abs/2511.12442)  
**Code**: None  
**Area**: Autonomous Driving / Graph Learning  
**Keywords**: Dynamic Graph Learning, Link Prediction, Attention Mechanism Alternatives, Adaptive Token Mixing, Hierarchical Aggregation

## TL;DR
GLFormer is proposed, a lightweight attention-free Transformer framework for dynamic graph link prediction. It replaces self-attention with an adaptive token mixer based on interaction order and time intervals, combined with a hierarchical aggregation mechanism to expand the temporal receptive field. It achieves comparable or superior performance to Transformer baselines across six benchmarks while significantly reducing computational complexity.

## Background & Motivation

Dynamic graph learning is crucial in fields such as transportation systems, social networks, and recommendation systems. One of the core tasks is dynamic link prediction—predicting whether an interaction will occur between two nodes at a future point in time.

Current methods generally employ the Transformer architecture to capture long-range temporal dependencies in interaction sequences. A typical pipeline involves first extracting structural information via time-aware random walks or memory networks, and then using a Transformer to learn temporal dependencies in the historical interaction sequences. However, the **computational complexity of the self-attention mechanism is quadratic with respect to the sequence length**, making it difficult to scale on high-frequency or large-scale graphs. In addition, the attention mechanism indiscriminately aggregates all pairwise interactions, which may amplify noise and compromise generalization performance.

Key observation: Inspired by works such as MetaFormer in the field of computer vision, the success of Transformers might be attributed more to their architectural design (residual connections, FFNs, etc.) rather than self-attention itself. The authors validated this hypothesis through controlled experiments—**replacing self-attention with pooling or MLP in five Transformer baselines often yields comparable performance across four datasets**.

This leads to the core question: **Can a simpler, attention-free architecture be designed for dynamic graphs that maintains representation capacity while significantly reducing computational overhead?**

## Method

### Overall Architecture
The pipeline of GLFormer:
1. Embedding Layer: Utilizes existing dynamic graph methods (TGN/TGAT/DyGFormer, etc.) to obtain initial neighbor embeddings.
2. Adaptive Token Mixer: Replaces self-attention, performing local aggregation based on interaction order and time intervals.
3. Channel Mixer: A standard FFN to learn dependencies between channels.
4. Hierarchical Aggregation: Multi-layer stacking to expand the temporal receptive field.
5. Link Prediction: An MLP decoder to predict link probability based on the temporal representation of node pairs.

### Key Designs

1. **Adaptive Token Aggregation**:

    - **Function**: For each neighbor $u_i$, weightedly aggregates information from its $M$ nearest neighbors.
    - **Core Idea**: The aggregation weight $\alpha_p^i = \beta \mathbf{w}_p + (1 - \beta) \theta_p^i$ fuses two factors:
        - **Order Weight $\mathbf{w}_p$**: A learnable parameter capturing the importance of interaction sequence.
        - **Temporal Weight $\theta_p^i$**: Calculated by applying softmax over time intervals, $\theta_p^i = \frac{\exp(-(t_i - t_{i-p}))}{\sum_q \exp(-(t_i - t_{i-q}))}$, assigning larger weights to closer temporal distances.
        - The learnable parameter $\beta$ controls the fusion ratio of the two.
    - **Design Motivation**: In dynamic graphs, the nearest neighbors provide the most relevant interaction patterns; local aggregation is more effective and efficient than global attention.

2. **Hierarchical Aggregation**:

    - **Function**: Inspired by dilated causal convolutions, gradually expands the temporal span of aggregation as the number of layers increases.
    - **Core Idea**: Defines the hierarchical offset set $\mathcal{R}_l = \{p \in \mathbb{Z} \mid s^{l-1} \leq p \leq s^l\}$. As $l$ increases, the aggregation range covers more distant historical interactions.
    - Aggregation at the $l$-th layer: $\mathbf{H}_{i,:}^{(l)} = \sum_{p \in \mathcal{R}_l} (\alpha_p^i)^{(l)} \mathbf{H}_{\text{TA}, i-p}^{(l-1)}$
    - Positions exceeding sequence boundaries are handled using causal masking.
    - **Design Motivation**: Captures long-range temporal dependencies through stacking while maintaining the low complexity of local aggregation, analogous to convolutional kernels at different scales.

3. **Complexity Advantage**:

    - **Function**: Reduces the complexity of each layer from $O(N^2)$ in self-attention to $O(NK_l)$.
    - Total complexity is $O(\sum_{l=1}^L NK_l)$, which is far below quadratic when $K_l \ll N$.
    - Requires only $O(K_l)$ kernel parameters per layer, rendering it highly parameter-efficient.

### Loss & Training
- Binary cross-entropy loss, with a 1:1 negative sampling strategy.
- Positive samples are actual interactions $(u_i, v_j, t)$, and negative samples are randomly sampled from non-interacting nodes.
- Predicted probability $\hat{y} = \sigma(\mathbf{K}_2(\text{ReLU}(\mathbf{K}_1([\mathbf{Z}_{u_i}; \mathbf{Z}_{v_j}]))))$.

## Key Experimental Results

### Main Results

**AP (Average Precision) Metric, GLFormer Average Rank**:

| Backbone | Vanilla (Rank) | Pooling (Rank) | MLP (Rank) | GLFormer (Rank) |
|----------|--------------|--------------|-----------|----------------|
| TGN | 3.17 | 2.83 | 2.17 | **1.67** |
| TCL | 3.33 | 3.50 | 2.00 | **1.17** |
| TGAT | 3.00 | 3.50 | 2.17 | **1.17** |
| CAWN | 2.83 | 3.00 | **1.33** | 2.50 |
| DyGFormer | 2.17 | 3.17 | 3.33 | **1.00** |

**Specific Values (DyGFormer Backbone)**:

| Dataset | Vanilla AP | GLFormer AP | Gain |
|--------|-----------|-------------|------|
| Wikipedia | 99.03 | **99.03** | +0.00 |
| Reddit | 99.22 | **99.24** | +0.02 |
| MOOC | 87.52 | **87.87** | +0.35 |
| LastFM | 93.00 | **93.34** | +0.34 |
| SocialEvo | 94.73 | **94.76** | +0.03 |
| Enron | 92.47 | **92.62** | +0.15 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Token Mixer Type | | |
| Self-Attention (Vanilla) | Rank 2-3 | Vanilla Transformer |
| Average Pooling | Rank 3-3.5 | Simple average, globally worst |
| MLP | Rank 1.3-2.3 | Non-linear transformation, relatively strong |
| **GLFormer** | Rank **1-1.67** | Optimal or suboptimal |
| Hierarchical Aggregation | | |
| Single-layer fixed window | Lower performance | Limited receptive field |
| Multi-layer hierarchical | Gain | Capture long-range dependencies |
| AUC-ROC Metric | | |
| GLFormer on TGN | Rank **1.67** | Consistently outperforms vanilla attention |
| GLFormer on DyGFormer | Rank **1.00** | Universally optimal across 6 datasets |

### Key Findings
- GLFormer ranks first or second on most backbones and datasets, demonstrating that **attention-free architectures can indeed match or even surpass Transformer baselines**.
- It performs best on the DyGFormer backbone (ranking 1.00 in both AP and AUC-ROC), indicating a more pronounced advantage in long-sequence modeling scenarios.
- The MLP variant is stronger on the CAWN backbone (rank 1.33 vs. GLFormer 2.50), showing that different backbones exhibit different preferences for token mixers.
- The fusion of temporal and historical order weights is adaptively balanced via a learnable $\beta$; ablation studies show both are indispensable.
- Significant computational efficiency: Inference speed is faster than all baselines utilizing self-attention.

## Highlights & Insights
- **Counter-intuitive Finding**: Self-attention is not irreplaceable in dynamic graph learning; a simple local aggregation combined with hierarchical stacking can match its performance.
- **Time-aware Design**: Simultaneously modeling interaction order (learned sequence weights) and time intervals (physical time decay), which is more suitable for temporal scenarios than general attention.
- **Inspiration from Dilated Causal Convolution**: Introduces the concept of hierarchical receptive field expansion from WaveNet into dynamic graphs.
- **Plug-and-Play**: Can directly replace the attention modules of existing methods (TGN, TGAT, DyGFormer, etc.).
- Clear complexity analysis, with highly consistent theoretical and experimental results.

## Limitations & Future Work
- Performance on the CAWN backbone is inferior to the MLP variant, showing that adaptability varies by backbone.
- The improvement is marginal on some datasets (e.g., only +0.00 on Wikipedia), leaving limited room for improvement in high-performance regimes.
- Hyperparameters of hierarchical aggregation (base $s$, number of layers $L$ ) require tuning for different datasets.
- Scalability has not been validated on ultra-large-scale graphs (at the million-node level).
- Edge features are not considered; currently, only node features and timestamps are utilized.

## Related Work & Insights
- The "attention is not all you need" concept from MetaFormer/PoolFormer holds true in graph learning as well.
- The paradigm of local aggregation combined with hierarchical expansion can be generalized to fields like time-series forecasting.
- The design of the temporal decay weight $\theta_p^i$ is simple and effective, and can be borrowed for other temporal modeling tasks.
- The two-stage "embedding + aggregation" paradigm of dynamic graph learning methods provides a flexible experimental framework for structural exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ — Important problem, counter-intuitive hypothesis validated by experiments, but the technical approach is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on 6 datasets and 5 backbones using both AP and AUC-ROC metrics, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Rigorously argued motivation, beautifully designed preliminary experiments.
- Value: ⭐⭐⭐⭐ — Provides an efficient alternative for dynamic graph learning with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[ICLR 2026\] SceneStreamer: Continuous Scenario Generation as Next Token Group Prediction](../../ICLR2026/autonomous_driving/scenestreamer_continuous_scenario_generation_as_next_token_group_prediction.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)

</div>

<!-- RELATED:END -->
