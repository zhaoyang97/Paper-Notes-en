---
title: >-
  [Paper Note] Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction
description: >-
  [AAAI 2026][Autonomous Driving][dynamic graph learning] This paper proposes GLFormer, a lightweight attention-free Transformer framework for dynamic graph link prediction. It replaces self-attention with an adaptive token mixer conditioned on interaction order and temporal intervals, and employs a hierarchical aggregation mechanism to enlarge the temporal receptive field. GLFormer achieves performance on par with or superior to Transformer baselines across 6 benchmarks while substantially reducing computational complexity.
tags:
  - AAAI 2026
  - Autonomous Driving
  - dynamic graph learning
  - link prediction
  - attention mechanism replacement
  - adaptive token mixing
  - hierarchical aggregation
date: 2026-05-08
content_hash: 064da8f2ab58a5c1
---

# Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction

**Conference**: AAAI 2026
**arXiv**: [2511.12442](https://arxiv.org/abs/2511.12442)
**Code**: N/A
**Area**: Autonomous Driving / Graph Learning
**Keywords**: dynamic graph learning, link prediction, attention mechanism replacement, adaptive token mixing, hierarchical aggregation

## TL;DR
This paper proposes GLFormer, a lightweight attention-free Transformer framework for dynamic graph link prediction. It replaces self-attention with an adaptive token mixer conditioned on interaction order and temporal intervals, and employs a hierarchical aggregation mechanism to enlarge the temporal receptive field. GLFormer achieves performance on par with or superior to Transformer baselines across 6 benchmarks while substantially reducing computational complexity.

## Background & Motivation

Dynamic graph learning is critical in domains such as traffic systems, social networks, and recommender systems. A core task is dynamic link prediction—forecasting whether two nodes will interact at a future time step.

Existing methods commonly adopt Transformer architectures to capture long-range temporal dependencies in interaction sequences. The typical pipeline extracts structural information via time-aware random walks or memory networks, then applies Transformers to model temporal dependencies in historical interaction sequences. However, the **quadratic complexity of self-attention with respect to sequence length** makes such approaches difficult to scale on high-frequency or large-scale graphs. Furthermore, attention mechanisms aggregate all pairwise interactions indiscriminately, potentially amplifying noise and reducing generalization.

A key observation, inspired by MetaFormer and related work in computer vision, is that the success of Transformers may be attributed more to their overall architectural design (residual connections, FFN, etc.) than to self-attention per se. The authors verify this hypothesis through controlled experiments—**replacing self-attention with pooling or MLP across five Transformer baselines on four datasets often yields comparable performance**.

This motivates the central question: **Can a simpler, attention-free architecture for dynamic graphs maintain representational capacity while significantly reducing computational overhead?**

## Method

### Overall Architecture
The GLFormer pipeline consists of:
1. **Embedding layer**: Obtains initial neighbor embeddings using existing dynamic graph methods (TGN/TGAT/DyGFormer, etc.)
2. **Adaptive token mixer**: Replaces self-attention; performs local aggregation conditioned on interaction order and temporal intervals
3. **Channel mixer**: Standard FFN for learning inter-channel dependencies
4. **Hierarchical aggregation**: Multi-layer stacking to expand the temporal receptive field
5. **Link prediction**: An MLP decoder predicts link probability from the temporal representations of node pairs

### Key Designs

1. **Adaptive Token Aggregation Module**:

    - *Function*: For each neighbor $u_i$, aggregates information from its most recent $M$ neighbors via weighted summation.
    - *Mechanism*: The aggregation weight $\alpha_p^i = \beta \mathbf{w}_p + (1 - \beta) \theta_p^i$ fuses two components:
        - **Order weight $\mathbf{w}_p$**: A learnable parameter capturing the importance of interaction order.
        - **Temporal weight $\theta_p^i$**: Computed by applying softmax over temporal gaps, $\theta_p^i = \frac{\exp(-(t_i - t_{i-p}))}{\sum_q \exp(-(t_i - t_{i-q}))}$, assigning higher weight to more recent interactions.
        - A learnable scalar $\beta$ controls the balance between the two components.
    - *Design Motivation*: In dynamic graphs, recent neighbors provide the most relevant interaction patterns; local aggregation is both more effective and more efficient than global attention.

2. **Hierarchical Aggregation Mechanism**:

    - *Function*: Inspired by dilated causal convolutions, progressively expands the temporal aggregation span as layers increase.
    - *Mechanism*: Defines a layer-wise offset set $\mathcal{R}_l = \{p \in \mathbb{Z} \mid s^{l-1} \leq p \leq s^l\}$; as $l$ grows, the aggregation range covers more distant historical interactions.
    - Layer $l$ aggregation: $\mathbf{H}_{i,:}^{(l)} = \sum_{p \in \mathcal{R}_l} (\alpha_p^i)^{(l)} \mathbf{H}_{\text{TA}, i-p}^{(l-1)}$
    - Positions beyond the sequence boundary are handled via causal masking.
    - *Design Motivation*: Captures long-range temporal dependencies through stacking while preserving the low complexity of local aggregation, analogous to convolution kernels at multiple scales.

3. **Complexity Advantage**:

    - *Function*: Reduces per-layer complexity from $O(N^2)$ for self-attention to $O(NK_l)$.
    - Total complexity $O(\sum_{l=1}^L NK_l)$, far below quadratic when $K_l \ll N$.
    - Each layer requires only $O(K_l)$ kernel parameters, yielding high parameter efficiency.

### Loss & Training
- Binary cross-entropy loss with a 1:1 negative sampling strategy.
- Positive samples are observed interactions $(u_i, v_j, t)$; negative samples are randomly drawn from non-interacting node pairs.
- Predicted probability: $\hat{y} = \sigma(\mathbf{K}_2(\text{ReLU}(\mathbf{K}_1([\mathbf{Z}_{u_i}; \mathbf{Z}_{v_j}]))))$

## Key Experimental Results

### Main Results

**AP (Average Precision), average rank of GLFormer**:

| Backbone | Vanilla (Rank) | Pooling (Rank) | MLP (Rank) | GLFormer (Rank) |
|----------|----------------|----------------|------------|-----------------|
| TGN | 3.17 | 2.83 | 2.17 | **1.67** |
| TCL | 3.33 | 3.50 | 2.00 | **1.17** |
| TGAT | 3.00 | 3.50 | 2.17 | **1.17** |
| CAWN | 2.83 | 3.00 | **1.33** | 2.50 |
| DyGFormer | 2.17 | 3.17 | 3.33 | **1.00** |

**Detailed results (DyGFormer backbone)**:

| Dataset | Vanilla AP | GLFormer AP | Diff. |
|---------|------------|-------------|-------|
| Wikipedia | 99.03 | **99.03** | +0.00 |
| Reddit | 99.22 | **99.24** | +0.02 |
| MOOC | 87.52 | **87.87** | +0.35 |
| LastFM | 93.00 | **93.34** | +0.34 |
| SocialEvo | 94.73 | **94.76** | +0.03 |
| Enron | 92.47 | **92.62** | +0.15 |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| **Token mixer type** | | |
| Self-Attention (Vanilla) | Rank 2–3 | Original Transformer |
| Average Pooling | Rank 3–3.5 | Simple average; typically worst |
| MLP | Rank 1.3–2.3 | Nonlinear transform; relatively strong |
| **GLFormer** | Rank **1–1.67** | Best or second best |
| **Hierarchical aggregation** | | |
| Single-layer fixed window | Lower performance | Limited receptive field |
| Multi-layer hierarchical | Improved performance | Captures long-range dependencies |
| **AUC-ROC metric** | | |
| GLFormer on TGN | Rank **1.67** | Consistently outperforms vanilla attention |
| GLFormer on DyGFormer | Rank **1.00** | Best across all 6 datasets |

### Key Findings
- GLFormer ranks first or second on most backbone–dataset combinations, confirming that **attention-free architectures can match or surpass Transformer baselines**.
- The advantage is most pronounced with the DyGFormer backbone (AP and AUC-ROC both rank 1.00), suggesting greater benefit in long-sequence modeling scenarios.
- On the CAWN backbone, the MLP variant outperforms GLFormer (rank 1.33 vs. 2.50), indicating that different backbones may favor different token mixers.
- The learnable $\beta$ adaptively balances temporal and order weights; ablations show that both components are necessary.
- Computational efficiency is significant: inference speed surpasses all self-attention-based baselines.

## Highlights & Insights
- **Counter-intuitive finding**: Self-attention is not indispensable for dynamic graph learning; simple local aggregation combined with hierarchical stacking suffices.
- **Temporally-aware design**: The method jointly models interaction order (learned positional weights) and temporal intervals (physical time decay), making it better suited to temporal settings than generic attention.
- **Inspiration from dilated causal convolutions**: The hierarchical receptive field expansion idea from WaveNet is successfully transferred to dynamic graphs.
- **Plug-and-play**: GLFormer can directly replace the attention module in existing methods (TGN/TGAT/DyGFormer, etc.).
- Complexity analysis is rigorous and consistent with empirical results.

## Limitations & Future Work
- Performance is inferior to the MLP variant on the CAWN backbone; adaptability varies across backbones.
- Improvements on some datasets are marginal (e.g., +0.00 on Wikipedia), indicating limited headroom in high-performance regimes.
- Hyperparameters of hierarchical aggregation (base $s$, number of layers $L$) require dataset-specific tuning.
- Scalability on very large graphs (millions of nodes) has not been validated.
- Edge features are not utilized; the current formulation relies solely on node features and timestamps.

## Related Work & Insights
- The "attention is not all you need" insight from MetaFormer/PoolFormer transfers to graph learning as well.
- The local aggregation + hierarchical expansion paradigm is potentially generalizable to time-series forecasting and other domains.
- The temporal decay weight $\theta_p^i$ is simple yet effective and can be adapted to other temporal modeling tasks.
- The two-stage "embedding + aggregation" paradigm in dynamic graph learning provides a flexible experimental framework for architectural exploration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Addresses an important problem and validates a counter-intuitive hypothesis empirically, though the technical contribution is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 datasets, 5 backbones, dual metrics (AP + AUC-ROC), and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is rigorously argued; the preliminary experiment design is elegant.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient alternative for dynamic graph learning with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)

<!-- RELATED:END -->
