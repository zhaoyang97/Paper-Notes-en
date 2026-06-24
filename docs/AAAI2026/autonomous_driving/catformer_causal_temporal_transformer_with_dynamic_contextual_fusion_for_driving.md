---
title: >-
  [Paper Note] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction
description: >-
  [AAAI2026][Autonomous Driving][driving intention prediction] This paper proposes CaTFormer, which explicitly models the causal interaction between driver behavior and environmental context using a Causal Temporal Transformer, achieving state-of-the-art performance with a 98.6% F1 score on the Brain4Cars dataset.
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "driving intention prediction"
  - "causal inference"
  - "Transformer"
  - "dual-stream fusion"
  - "counterfactual reasoning"
date: 2026-05-08
content_hash: 9b2dd0cf028a8cb1
---

# CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction

**Conference**: AAAI2026  
**arXiv**: [2507.13425](https://arxiv.org/abs/2507.13425)  
**Code**: [srwang0506/CaTFormer](https://github.com/srwang0506/CaTFormer)  
**Area**: Autonomous Driving  
**Keywords**: driving intention prediction, causal inference, Transformer, dual-stream fusion, counterfactual reasoning

## TL;DR

This paper proposes CaTFormer, which explicitly models the causal interaction between driver behavior and environmental context using a Causal Temporal Transformer, achieving state-of-the-art performance with a 98.6% F1 score on the Brain4Cars dataset.

## Background & Motivation

Driving intention prediction is critical for autonomous driving safety. By predicting the driver's lane change or turning intention several seconds in advance, the system can actively issue warnings or take evasive actions. Current methods suffer from two major limitations:

1. **Coarse Fusion**: Most existing methods rely on simple concatenation or linear aggregation of inside-cabin (driver state) and outside-cabin (traffic scene) features, without explicitly modeling the causal dependencies between them. Changes in driver state directly affect vehicle behavior, but this causal relationship is ignored.
2. **Insufficient Temporal Modeling**: Early 3D CNN-LSTM architectures struggle to capture long-range, discontinuous temporal dependencies. Even with the introduction of Transformers, they primarily rely on the architecture's implicit attention mechanism, lacking explicit reasoning about causal relationships.

In addition, existing causal inference works (such as domain generalization based on causal models) cover only limited scenarios and fail to simultaneously address temporal alignment, spurious correlation elimination, and multi-view adaptive fusion in a unified framework.

## Core Problem

How to **explicitly model** the **causal temporal dependencies** between driver behavior and environmental context in a dual-stream Transformer architecture, while eliminating spurious correlations to achieve robust driving intention prediction?

## Method

CaTFormer employs a dual-stream architecture to process synchronized inside- and outside-cabin video frame sequences, consisting of three core modules:

### 1. Reciprocal Delayed Fusion (RDF)

- **Core Idea**: Models temporal priority across frames through a time-delay mechanism. At time step $t$, the attention mechanism only accesses information from the previous frame $t-1$, explicitly establishing temporal causal constraints between the cabin-internal and cabin-external feature streams.
- **Bidirectional Dependent Attention (BDA)**: The cabin-internal and cabin-external features of the current frame bidirectionally attend to their single-frame delayed counterparts, using $H=8$ attention heads to capture diverse associations, which are then aggregated through concatenation and linear mapping.
- **Channel Gating**: Applies two-layer channel gating (FC → ReLU → FC → Sigmoid → Hadamard product) to the input from BDA, adaptively enhancing informative channels and suppressing noisy channels.
- **Regularization**: RMSNorm is used to stabilize numerical values + Dropout is applied to prevent overfitting.

### 2. Counterfactual Residual Encoding (CRE)

This is the core innovation of this paper, utilizing counterfactual reasoning to eliminate spurious correlations:

- **Direct Causal Effect Calculation**: At each time step, two attention distributions are computed:
    - **Observational Attention** $A^{\text{obs}}$: Computes normal cross-stream attention using actual outside-cabin features.
    - **Counterfactual Attention** $A^{\text{cf}}$: Replaces all outside-cabin features with their temporal mean (neutral baseline) to compute the attention.
    - The difference between the two, $\Delta = A^{\text{obs}} - A^{\text{cf}}$, quantifies the direct causal contribution of the environmental context to the cabin-internal representation.
- **Orthogonal Projection**: Projects the causal residual onto the orthogonal complement space of the global baseline, ensuring that the causal patterns reflect true intention-related dependencies rather than dataset biases.
- **Dynamic Residual Gating**: A learnable gating coefficient $g_T$ adaptively adjusts the residual contribution according to the predictive value; critical maneuvers are amplified, while routine patterns are suppressed.
- **Adaptive Intention Encoding**: A coarse-grained intention distribution is extracted from the outside-cabin summary via a softmax classifier, which is then embedded as a continuous intention token $z_{\text{intent}}$ to serve as a global semantic anchor, providing top-down guidance for downstream fusion.

### 3. Feature Synthesis Network (FSN)

- Residual non-linear transformations (two-layer FC-ReLU-FC) are applied individually to the inside-cabin, outside-cabin, and interaction branches, combined with the velocity feature $s$.
- The contribution of each branch is adaptively controlled through learnable confidence weights $w_i$: $\ell_{\text{joint}} = \sum_{i} w_i (W_i r_i)$.
- The weights are normalized via softmax to achieve scene-adaptive dynamic fusion.

### Loss & Training

A unified loss function combines the average cross-entropy loss of four stream-level heads (in, out, ctx, joint) and the intention prediction loss:

$$\mathcal{L} = \frac{1}{4}\sum_{i \in \mathcal{H}} \text{CE}(\ell_i, y) + \alpha \cdot \text{CE}(\ell_{\text{intent}}, y)$$

where $\alpha = 0.1$. The Adam optimizer is used with an initial learning rate of $10^{-3}$, training for 160 epochs with a batch size of 16.

## Key Experimental Results

### Dataset

- **Brain4Cars**: Consists of outside-cabin (480×720) and inside-cabin (1088×1920) videos, containing a total of 594 active events covering highway and urban scenes. There are five action classes: driving straight, left turn, right turn, left lane change, and right lane change.

### Main Results (Brain4Cars)

| Method | Modality | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| DCNN | Camera + Speed | 91.8 | 92.5 | 92.1 |
| IDIPN | Camera only | 94.2 | 94.9 | 94.5 |
| FedPRM | Camera + GPS + Speed | 99.0 | 92.0 | 95.2 |
| **CaTFormer (Camera only)** | Camera only | 96.7 | 98.5 | **97.6** |
| **CaTFormer (+ Speed)** | Camera + Speed | 98.7 | 98.5 | **98.6** |

Achieving a 97.6% F1 score using only camera input already outperforms all multimodal methods; incorporating speed reaches 98.6%, which is a 3.4% gain over the best performing FedPRM.

### Early Warning Capability (Truncation Experiment)

| Observation Window | CaTFormer | IDIPN | TIFN |
|----------|-----------|-------|------|
| [-5, 0] | 98.6 | 94.5 | 87.9 |
| [-5, -1] | 97.4 | 84.1 | 80.9 |
| [-5, -2] | 90.1 | 74.2 | 71.0 |
| [-5, -3] | 78.4 | 62.0 | 55.0 |
| [-5, -4] | 63.7 | 55.4 | 44.6 |

The proposed method significantly outperforms baseline approaches across all advance prediction windows, maintaining a 90.1% F1 score even when predicting 2 seconds in advance.

### Ablation Study

| Configuration | F1 [-5,0] | F1 [-5,-2] |
|------|-----------|------------|
| Base (Dual-Stream Transformer) | 95.8 | 85.4 |
| Base+RDF | 97.1 | 87.1 |
| Base+CRE | 97.0 | 86.9 |
| Base+FSN | 96.6 | 86.3 |
| CaTFormer (R+C+F) | **98.6** | **90.1** |

Each of the three modules contributes a gain of +1.0% to 1.3%. When combined, the synergistic gain reaches +2.8% (full window) and +4.7% (2 seconds in advance).

## Highlights & Insights

1. **Counterfactual Causal Inference**: The CRE module identifies authentic causal effects by comparing the differences between observational and counterfactual attention, which are further refined using orthogonal projection and dynamic gating, presenting a highly elegant methodology.
2. **Exceptional Early Warning Capability**: Maintaining a 90.1% F1 score 2 seconds in advance is a substantial improvement over competing methods (74.2%), which is of great significance for practical safety warning systems.
3. **High Performance with Fewer Modalities**: Outperforming multimodal methods that rely on GPS + maps + speed using only camera input demonstrates that causal modeling is more effective than simply stacking sensors.
4. **Interpretable Attention Visualizations**: Temporal attention heatmaps and decision-boundary saliency maps demonstrate a complete reasoning path from dynamic event understanding to static decision attribution.

## Limitations & Future Work

1. **Evaluation on a Single Dataset**: The evaluation is restricted to the Brain4Cars dataset (594 events), which has a relatively small scale and lacks validation on larger-scale datasets and under more diverse scenarios (e.g., adverse weather, nighttime).
2. **Coarse-grained Action Categories**: Only five classes of actions are covered (straight, turns, lane changes), excluding fine-grained intentions such as acceleration/deceleration, stopping, or emergency evasion.
3. **Computational Overhead of Optical Flow**: The external stream relies on offline pre-computation of Farneback optical flow, increasing preprocessing time. End-to-end learnable motion features could be considered as alternatives.
4. **Lack of Real-time Discussion**: Inference latency is not reported, and the real-time feasibility of the 14.53M parameters coupled with optical flow pre-computation remains to be verified.
5. **Selection of Counterfactual Baseline**: Whether using the temporal mean as the neutral baseline in CRE is optimal remains an open question, and more sophisticated counterfactual construction strategies could be explored.

## Related Work & Insights

| Dimension | CaTFormer | TIFN | CEMFormer | IDIPN |
|------|------|-----------|-------|---|
| Fusion Method | Causal Delayed Bidirectional Attention | STU State Update | Cross-View Transformer | Interaction Decoupling |
| Causal Inference | Counterfactual Residual Encoding | None | None | None |
| Best F1 | **98.6** | 87.9 | 87.1 | 94.5 |
| Intention Encoding | Adaptive Intention Token | None | None | None |
| Multi-View Fusion | FSN Dynamic Weighting | Semantic Segmentation Attention | Unified Cross-View | Decoupled Prediction |

The core differentiator of CaTFormer lies in its **explicit causal modeling**: RDF establishes temporal causal constraints, CRE eliminates spurious correlations, and FSN dynamically fuses representations. Together they form a complete causal inference chain, rather than relying on implicit attention-based correlations.

## Insights & Connections

- The idea of **counterfactual attention subtraction** can be generalized to other vision tasks requiring the elimination of spurious correlations (e.g., VQA, video understanding).
- The **delayed fusion mechanism** provides a lightweight yet effective solution for multimodal temporal alignment, which is more concise than complex temporal alignment networks.
- The paradigm combining causal inference with Transformers can be extended to related tasks such as pedestrian intention prediction and traffic flow forecasting.
- The orthogonal projection de-biasing approach in CRE aligns closely with deconfounding methods in causal representation learning (e.g., backdoor adjustment).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of counterfactual residual encoding and reciprocal delayed fusion is relatively novel.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive ablation studies, but validated only on a single small dataset.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical formulations, and rich visualizations.
- Value: ⭐⭐⭐⭐ — The causal inference methodology is highly generalizable, and the early warning performance is outstanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction](../../ICLR2026/autonomous_driving/unisplat_unified_spatio-temporal_fusion_via_3d_latent_scaffolds_for_dynamic_driv.md)
- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](../../CVPR2025/autonomous_driving/socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](../../CVPR2026/autonomous_driving/efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)
- [\[ICLR 2026\] Detecting Temporal Misalignment Attacks in Multimodal Fusion for Autonomous Driving](../../ICLR2026/autonomous_driving/detecting_temporal_misalignment_attacks_in_multimodal_fusion_for_autonomous_driv.md)

</div>

<!-- RELATED:END -->
