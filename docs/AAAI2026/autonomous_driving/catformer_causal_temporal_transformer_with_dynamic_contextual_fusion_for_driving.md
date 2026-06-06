---
title: >-
  [Paper Note] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction
description: >-
  [AAAI2026][Autonomous Driving][driving intention prediction] CaTFormer is proposed to explicitly model causal interactions between driver behavior and environmental context via a causal temporal Transformer…
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "driving intention prediction"
  - "causal inference"
  - "Transformer"
  - "dual-stream fusion"
  - "counterfactual reasoning"
date: 2026-05-08
content_hash: e31dee3584745b16
---

# CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction

**Conference**: AAAI2026
**arXiv**: [2507.13425](https://arxiv.org/abs/2507.13425)  
**Code**: [srwang0506/CaTFormer](https://github.com/srwang0506/CaTFormer)  
**Area**: Autonomous Driving
**Keywords**: driving intention prediction, causal inference, Transformer, dual-stream fusion, counterfactual reasoning

## TL;DR

CaTFormer is proposed to explicitly model causal interactions between driver behavior and environmental context via a causal temporal Transformer, achieving state-of-the-art performance of 98.6% F1 on the Brain4Cars dataset.

## Background & Motivation

Driving intention prediction is critical for autonomous driving safety: by anticipating lane-change or turning intentions several seconds in advance, the system can proactively issue warnings or take evasive action. Existing methods suffer from two primary limitations:

1. **Coarse fusion strategies**: Most methods simply concatenate or linearly aggregate in-vehicle (driver state) and out-of-vehicle (traffic scene) features, without explicitly modeling the causal dependencies between them. Although driver state changes directly influence vehicle behavior, this causal relationship is largely ignored.
2. **Insufficient temporal modeling**: Early 3D CNN-LSTM architectures struggle to capture long-range and non-contiguous temporal dependencies. Even Transformer-based methods primarily rely on implicit attention mechanisms, lacking explicit causal reasoning.

Furthermore, existing causal inference works (e.g., causal model-based domain generalization) address limited scenarios and fail to simultaneously handle temporal alignment, spurious correlation elimination, and multi-view adaptive fusion within a unified framework.

## Core Problem

How to **explicitly model** the **causal temporal dependencies** between driver behavior and environmental context within a dual-stream Transformer architecture, while eliminating spurious correlations to achieve robust driving intention prediction?

## Method

CaTFormer employs a dual-stream architecture to process synchronized in-vehicle and out-of-vehicle video frame sequences, comprising three core modules:

### 1. Reciprocal Delayed Fusion (RDF)

- **Core Idea**: Models inter-frame temporal priority via a time-delay mechanism. At time step $t$, the attention mechanism only accesses information from the previous frame $t-1$, explicitly establishing causal temporal constraints between the two feature streams.
- **Bidirectional Dependency Attention (BDA)**: In-vehicle and out-of-vehicle features of the current frame bidirectionally attend to their single-frame-delayed counterparts, using $H=8$ attention heads to capture diverse associations, aggregated via concatenation and linear projection.
- **Channel Gating**: Two-layer channel gating (FC → ReLU → FC → Sigmoid → Hadamard product) is applied to BDA outputs to adaptively enhance informative channels and suppress noisy ones.
- **Regularization**: RMSNorm for numerical stability and Dropout to prevent overfitting.

### 2. Counterfactual Residual Encoding (CRE)

This is the central innovation of the paper, leveraging counterfactual reasoning to eliminate spurious correlations:

- **Direct Causal Effect Computation**: At each time step, two attention distributions are computed:
    - **Observed attention** $A^{\text{obs}}$: standard cross-stream attention using actual out-of-vehicle features
    - **Counterfactual attention** $A^{\text{cf}}$: attention computed by replacing all out-of-vehicle features with their temporal mean (a neutral baseline)
    - The difference $\Delta = A^{\text{obs}} - A^{\text{cf}}$ quantifies the direct causal contribution of environmental context to in-vehicle representations
- **Orthogonal Projection**: The causal residual is projected onto the orthogonal complement of the global baseline, ensuring that the causal patterns reflect genuine intention-relevant dependencies rather than dataset biases.
- **Dynamic Residual Gating**: A learnable gating coefficient $g_T$ adaptively adjusts residual contributions based on predictive value, amplifying critical maneuver cues and suppressing routine patterns.
- **Adaptive Intention Encoding**: A coarse-grained intention distribution is extracted from the out-of-vehicle summary via softmax classification, then embedded as a continuous intention token $z_{\text{intent}}$ to serve as a global semantic anchor providing top-down guidance for downstream fusion.

### 3. Feature Synthesis Network (FSN)

- Residual nonlinear transformations (two-layer FC-ReLU-FC) are applied to each of three branches—in-vehicle, out-of-vehicle, and interaction—combined with velocity feature $s$.
- Learnable confidence weights $w_i$ adaptively control each branch's contribution: $\ell_{\text{joint}} = \sum_{i} w_i (W_i r_i)$.
- Weights are softmax-normalized to achieve scene-adaptive dynamic fusion.

### Loss & Training

A unified loss function combines the average cross-entropy loss over four stream-level heads (in, out, ctx, joint) with an intention prediction loss:

$$\mathcal{L} = \frac{1}{4}\sum_{i \in \mathcal{H}} \text{CE}(\ell_i, y) + \alpha \cdot \text{CE}(\ell_{\text{intent}}, y)$$

where $\alpha = 0.1$. Adam optimizer with initial learning rate $10^{-3}$, trained for 160 epochs with batch size 16.

## Key Experimental Results

### Dataset

- **Brain4Cars**: Contains out-of-vehicle (480×720) and in-vehicle (1088×1920) videos with 594 valid events covering highway and urban scenarios. Five action categories: going straight, left turn, right turn, left lane change, and right lane change.

### Main Results (Brain4Cars)

| Method | Modality | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| DCNN | Camera+Speed | 91.8 | 92.5 | 92.1 |
| IDIPN | Camera only | 94.2 | 94.9 | 94.5 |
| FedPRM | Camera+GPS+Speed | 99.0 | 92.0 | 95.2 |
| **CaTFormer (camera only)** | Camera only | 96.7 | 98.5 | **97.6** |
| **CaTFormer (+speed)** | Camera+Speed | 98.7 | 98.5 | **98.6** |

Using only camera input, CaTFormer achieves 97.6% F1, surpassing all multimodal methods; adding speed yields 98.6%, outperforming the best prior method FedPRM by 3.4%.

### Early Warning Capability (Truncation Experiments)

| Observation Window | CaTFormer | IDIPN | TIFN |
|--------------------|-----------|-------|------|
| [-5, 0] | 98.6 | 94.5 | 87.9 |
| [-5, -1] | 97.4 | 84.1 | 80.9 |
| [-5, -2] | 90.1 | 74.2 | 71.0 |
| [-5, -3] | 78.4 | 62.0 | 55.0 |
| [-5, -4] | 63.7 | 55.4 | 44.6 |

CaTFormer significantly outperforms all baselines across every early prediction window, maintaining 90.1% F1 when predicting 2 seconds in advance.

### Ablation Study

| Configuration | F1 [-5,0] | F1 [-5,-2] |
|---------------|-----------|------------|
| Base (dual-stream Transformer) | 95.8 | 85.4 |
| Base+RDF | 97.1 | 87.1 |
| Base+CRE | 97.0 | 86.9 |
| Base+FSN | 96.6 | 86.3 |
| CaTFormer (R+C+F) | **98.6** | **90.1** |

Each module individually contributes +1–1.3% improvement; combining all three yields a synergistic gain of +2.8% (full window) and +4.7% (2-second advance prediction).

## Highlights & Insights

1. **Counterfactual causal reasoning**: The CRE module identifies genuine causal effects by contrasting observed and counterfactual attention differences, further purified via orthogonal projection and dynamic gating — an elegant methodological contribution.
2. **Strong early warning capability**: 90.1% F1 at 2 seconds in advance, far exceeding baselines (74.2%), which is highly significant for practical safety warning systems.
3. **High performance with fewer modalities**: Camera-only CaTFormer surpasses multimodal methods relying on GPS, maps, and speed, demonstrating that causal modeling is more effective than naively stacking sensors.
4. **Interpretable attention visualization**: Temporal attention heatmaps and decision margin saliency maps illustrate a complete reasoning pathway from dynamic event understanding to static decision attribution.

## Limitations & Future Work

1. **Single-dataset evaluation**: Evaluated only on Brain4Cars (594 events); the limited scale and absence of validation on larger datasets or more challenging scenarios (adverse weather, nighttime) remain concerns.
2. **Coarse-grained action categories**: Only five categories (straight, turn, lane change) are covered; fine-grained intentions such as acceleration/deceleration, stopping, and emergency avoidance are not addressed.
3. **Optical flow computation overhead**: The external stream relies on pre-computed Farneback optical flow, introducing additional preprocessing cost; end-to-end learnable motion features could be explored as alternatives.
4. **Real-time performance not discussed**: Inference latency is not reported; the real-time feasibility of 14.53M parameters combined with optical flow preprocessing remains to be verified.
5. **Counterfactual baseline selection**: Whether the temporal mean is the optimal neutral baseline for CRE is an open question; more sophisticated counterfactual construction strategies warrant investigation.

## Related Work & Insights

| Dimension | CaTFormer | TIFN | CEMFormer | IDIPN |
|-----------|-----------|------|-----------|-------|
| Fusion Strategy | Causal delayed bidirectional attention | STU state update | Cross-view Transformer | Interaction decoupling |
| Causal Reasoning | Counterfactual residual encoding | None | None | None |
| Best F1 | **98.6** | 87.9 | 87.1 | 94.5 |
| Intention Encoding | Adaptive intention token | None | None | None |
| Multi-view Fusion | FSN dynamic weighting | Semantic segmentation attention | Unified cross-view | Decoupled prediction |

The key distinguishing factor of CaTFormer is **explicit causal modeling**: RDF establishes temporal causal constraints, CRE eliminates spurious correlations, and FSN performs adaptive fusion — forming a complete causal reasoning chain rather than relying on implicit attention correlations.

- The **counterfactual attention subtraction** paradigm is generalizable to other vision tasks requiring spurious correlation elimination (e.g., VQA, video understanding).
- The **delayed fusion mechanism** provides a lightweight yet effective solution for multimodal temporal alignment, simpler than complex temporal alignment networks.
- The combination of causal inference and Transformers can be extended to pedestrian intention prediction, traffic flow forecasting, and related tasks.
- The orthogonal projection debiasing in CRE shares conceptual parallels with deconfounding methods in causal representation learning (e.g., backdoor adjustment).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of counterfactual residual encoding and reciprocal delayed fusion is relatively novel
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive ablation study but limited to a single small dataset
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous formulations, and rich visualizations
- Value: ⭐⭐⭐⭐ — The causal reasoning framework has broad applicability; early warning performance is excellent

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](../../CVPR2026/autonomous_driving/efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults](multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)
- [\[CVPR 2026\] CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](../../CVPR2026/autonomous_driving/causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)

</div>

<!-- RELATED:END -->
