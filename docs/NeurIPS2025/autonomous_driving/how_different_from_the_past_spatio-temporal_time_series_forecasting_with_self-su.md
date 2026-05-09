---
title: >-
  [Paper Note] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning
description: >-
  [NeurIPS 2025][Autonomous Driving][spatio-temporal forecasting] This paper proposes ST-SSDL, a framework that captures dynamic deviations between current inputs and historical patterns via self-supervised deviation learning (SSDL). It discretizes the latent space using learnable prototypes and enforces relative distance consistency through a contrastive loss and a deviation loss, achieving state-of-the-art performance on six spatio-temporal benchmarks.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - spatio-temporal forecasting
  - self-supervised learning
  - deviation modeling
  - prototype learning
  - contrastive learning
date: 2026-05-08
content_hash: 43ba6fd156806abc
---

# How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.04908](https://arxiv.org/abs/2510.04908)
**Code**: [GitHub](https://github.com/Jimmy-7664/ST-SSDL)
**Area**: Autonomous Driving
**Keywords**: spatio-temporal forecasting, self-supervised learning, deviation modeling, prototype learning, contrastive learning

## TL;DR

This paper proposes ST-SSDL, a framework that captures dynamic deviations between current inputs and historical patterns via self-supervised deviation learning (SSDL). It discretizes the latent space using learnable prototypes and enforces relative distance consistency through a contrastive loss and a deviation loss, achieving state-of-the-art performance on six spatio-temporal benchmarks.

## Background & Motivation

Spatio-temporal forecasting (e.g., traffic flow, energy demand) is a core task in urban computing. Although existing methods have achieved improvements in accuracy, they generally overlook a critical signal: **the dynamic deviation between current observations and historical patterns**. In real-world traffic systems, policy interventions, special events, or external incidents frequently cause significant deviations between the current time series and historical states, and these deviations carry important predictive signals for future behavior.

Existing limitations:
1. Some methods (e.g., those using fixed temporal offsets) can only capture static historical context and fail to model dynamic deviations.
2. Simple threshold-based approaches treat deviations as binary events, whereas actual deviations change continuously.
3. Quantifying the mapping from physical-space deviations to latent-space deviations in high-dimensional representation spaces remains a core challenge.

Core observation: sensors in city centers typically exhibit high variance (large deviations), whereas rural roads are more stable (small deviations), and the degree of deviation changes dynamically with spatio-temporal context.

## Method

### Overall Architecture

ST-SSDL consists of three core components:
1. **Historical anchor construction**: historical averages are used as self-supervised anchors.
2. **Self-supervised spatial discretization**: learnable prototypes discretize the continuous latent space.
3. **Self-supervised deviation quantification**: a deviation loss enforces relative distance consistency between physical space and latent space.

The overall architecture adopts an encoder–decoder structure and models spatio-temporal dependencies via GCRU (Graph Convolution Recurrent Unit).

### Key Designs

#### 1. History as Self-Supervised Anchors

Training sequences are segmented by week (exploiting the periodicity of spatio-temporal data), and a history-weighted average $\bar{X}^w = \frac{1}{S}\sum_{s=1}^{S}X_s^w$ is computed for each time step. For the current input $X^c$, the timestamp-aligned historical anchor $X^a$ is retrieved. Both are passed through a shared encoder to obtain latent representations $H^c, H^a \in \mathbb{R}^{N \times h}$.

#### 2. Self-Supervised Spatial Discretization (Prototype Learning + Contrastive Loss)

$M=20$ learnable prototypes $\mathbf{P}_1, \ldots, \mathbf{P}_M \in \mathbb{R}^{M \times d}$ ($d=64$) are introduced, and discretization is achieved via query-prototype attention:

$$\alpha_i = \frac{\exp(Q \cdot \mathbf{P}_i^\top / \sqrt{d})}{\sum_{j=1}^{M} \exp(Q \cdot \mathbf{P}_j^\top / \sqrt{d})}$$

After sorting the attention scores, the most relevant prototype is designated as the positive sample $\mathcal{P}^c$ and the second most relevant as the negative sample $\mathcal{N}^c$. A triplet contrastive loss is applied:

$$\mathcal{L}_{Con} = \max(\|\widetilde{\nabla}(Q^c) - \mathcal{P}^c\|_2^2 - \|\widetilde{\nabla}(Q^c) - \mathcal{N}^c\|_2^2 + \delta, 0)$$

where $\widetilde{\nabla}$ denotes a stop-gradient operation, preventing the model from collapsing all representations to the same prototype.

#### 3. Self-Supervised Deviation Quantification (Deviation Loss)

The core idea is **relative distance consistency**: current–history pairs that are close (or far apart) in physical space should remain close (or far apart) in latent space, i.e., $D_1 > D_2 \Rightarrow \widetilde{D}_1 > \widetilde{D}_2$.

Using the nearest prototype of each representation as a proxy, the deviation loss is computed as:

$$\mathcal{L}_{Dev} = \|\widetilde{\nabla}(\|Q^c - Q^a\|_1) - \|\mathcal{P}^c - \mathcal{P}^a\|_1\|_1$$

The stop-gradient causes $\|Q^c - Q^a\|_1$ to approximate the physical-space distance $D$, while $\|\mathcal{P}^c - \mathcal{P}^a\|_1$ represents the latent-space distance $\widetilde{D}$.

#### 4. GCRU Encoder–Decoder

**Encoder**: GRU units with Chebyshev graph convolution, $Z \star_{\mathcal{G}} \Theta = \sum_{k=0}^{K} \tilde{\mathcal{A}}^k Z W_k$.

**Decoder**: generates an adaptive adjacency matrix from the encoder output and prototype-augmented representations, $\tilde{\mathcal{A}} = \text{Softmax}(\text{ReLU}(H' \cdot H'^\top))$, where $H' = W[H^c | V^c | H^a | V^a] + b$.

### Loss & Training

The joint training objective is:

$$\mathcal{L} = \mathcal{L}_{MAE} + \lambda_{Con} \cdot \mathcal{L}_{Con} + \lambda_{Dev} \cdot \mathcal{L}_{Dev}$$

- $\mathcal{L}_{MAE}$: MAE loss between predictions and ground truth.
- $\lambda_{Con}, \lambda_{Dev}$: hyperparameters controlling the contribution of each loss term.
- Optimizer: Adam with an initial learning rate of 0.001.
- Architecture: 1-layer encoder + 1-layer decoder; hidden dimensions of 128/64/32 depending on the dataset.
- Input/prediction window: both set to 1 hour (12 time steps).

## Key Experimental Results

### Main Results

Comparison against 13 baselines on 6 traffic datasets:

| Dataset | Metric | MegaCRN | STDN | **ST-SSDL** |
|--------|------|---------|------|-------------|
| METRLA (60min) | MAE | 3.48 | 3.57 | **3.37** |
| METRLA (60min) | RMSE | 7.31 | 7.80 | **7.17** |
| PEMSBAY (15min) | MAE | 1.26 | 1.36 | **1.26** |
| PEMSBAY (15min) | RMSE | 2.71 | 2.96 | **2.65** |
| PEMSD7(M) (15min) | MAE | 2.05 | 2.17 | **2.02** |
| PEMSD7(M) (15min) | RMSE | 3.88 | 4.17 | **3.83** |

ST-SSDL achieves the best or tied-best results across all 6 datasets and all prediction horizons (15/30/60 min).

### Ablation Study

| Variant | METRLA MAE | PEMSBAY MAE |
|------|------------|-------------|
| Full ST-SSDL | **3.37** | **1.86** |
| w/o $\mathcal{L}_{Con}$ | 3.42 | 1.89 |
| w/o $\mathcal{L}_{Dev}$ | 3.44 | 1.90 |
| w/o prototype module | 3.48 | 1.91 |
| w/o historical anchors | 3.46 | 1.90 |

### Key Findings

1. Both the contrastive loss and the deviation loss contribute independently; their joint use yields the best performance.
2. A prototype count of $M=20$ provides the optimal balance; performance degrades with either more or fewer prototypes.
3. Visualization confirms that high-deviation inputs are mapped to positions farther from the anchor in prototype space, verifying that the model successfully quantifies dynamic deviations.
4. Complexity analysis shows that the SSDL module adds only $\mathcal{O}(NMd)$ overhead, which does not constitute a bottleneck.

## Highlights & Insights

1. **First proposal of deviation modeling for spatio-temporal data**: the paper pioneering identifies and addresses the overlooked problem of quantifying current-vs.-history deviation.
2. **Elegant relative distance consistency**: rather than requiring precise quantification of absolute deviations, the method preserves the relative ordering of deviations, making it robust and practical.
3. **Fully self-supervised design**: no additional labels are required; historical averages naturally provide anchors, conferring strong generalizability.
4. **Plug-and-play**: SSDL can be viewed as a general latent-space regularization technique that is theoretically applicable to other spatio-temporal models.

## Limitations & Future Work

1. Historical anchors are constructed by simple periodic averaging at weekly granularity, limiting the model's capacity to handle non-periodic or abrupt-change patterns.
2. The prototype count $M$ is fixed and cannot adapt to the complexity of different datasets.
3. Validation is limited to traffic datasets; generalization to other spatio-temporal domains such as climate and energy remains to be verified.
4. The encoder–decoder backbone (GCRU) is relatively conventional; integrating Transformer or Mamba backbones may yield further improvements.

## Related Work & Insights

- **Spatio-temporal forecasting**: STGCN, DCRNN, Graph WaveNet, and related methods focus on graph-based modeling; ST-SSDL builds upon these by incorporating deviation awareness.
- **Self-supervised learning**: draws inspiration from visual contrastive learning (SimCLR, MoCo) and is the first to incorporate deviation quantification into a self-supervised framework.
- **Prototype learning**: combines VQ-VAE-style prototype discretization with contrastive learning for structured representation of spatio-temporal deviations.

## Rating

- Novelty: ⭐⭐⭐⭐ — The deviation modeling perspective is novel and the relative distance consistency design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 6 datasets, comprehensive ablations, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and figures are intuitive.
- Value: ⭐⭐⭐⭐ — The proposed SSDL module has broad plug-and-play potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DBLoss: Decomposition-based Loss Function for Time Series Forecasting](dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[NeurIPS 2025\] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)
- [\[AAAI 2026\] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](../../AAAI2026/autonomous_driving/dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)

</div>

<!-- RELATED:END -->
