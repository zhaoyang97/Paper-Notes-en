---
title: >-
  [Paper Note] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction
description: >-
  [ICCV 2025][Time Series][Vehicle-to-Everything Collaboration] This paper proposes V2XPnP, a V2X spatio-temporal fusion framework built upon a unified Transformer architecture…
tags:
  - "ICCV 2025"
  - "Time Series"
  - "Vehicle-to-Everything Collaboration"
  - "Spatio-Temporal Fusion"
  - "End-to-End Perception and Prediction"
  - "V2X Dataset"
  - "Transformer"
date: 2026-05-08
content_hash: d10d051a6ba138c3
---

# V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction

**Conference**: ICCV 2025
**arXiv**: [2412.01812](https://arxiv.org/abs/2412.01812)  
**Code**: [mobility-lab.seas.ucla.edu/v2xpnp](https://mobility-lab.seas.ucla.edu/v2xpnp)  
**Area**: Temporal Prediction / Autonomous Driving
**Keywords**: Vehicle-to-Everything Collaboration, Spatio-Temporal Fusion, End-to-End Perception and Prediction, V2X Dataset, Transformer

## TL;DR

This paper proposes V2XPnP, a V2X spatio-temporal fusion framework built upon a unified Transformer architecture, which achieves multi-agent end-to-end perception and prediction under a one-step communication strategy. The work also introduces the first large-scale real-world sequential dataset supporting all V2X collaboration modes, achieving state-of-the-art performance on both perception and prediction tasks.

## Background & Motivation

### State of the Field

Autonomous driving systems require accurate perception of surrounding road users and prediction of their future trajectories. Single-vehicle systems are constrained by limited sensing range and occlusion. Vehicle-to-Everything (V2X) technology addresses these limitations through multi-agent information sharing.

### Limitations of Prior Work

**Focus on single-frame collaborative perception**: Existing V2X work primarily performs per-frame collaborative detection, fusing information from agents at different spatial locations while neglecting temporal cues across frames.

**Absence of temporal tasks**: Short-horizon temporal cues (0.5s) are used only to mitigate asynchrony issues, while long-horizon temporal tasks such as motion prediction remain largely unexplored.

**Scarcity of real-world sequential datasets**: Existing V2X datasets are mostly non-sequential and support only a single collaboration mode, lacking sequential datasets that cover all collaboration modes (V2V, V2I, I2I, VC, IC).

**Absence of end-to-end PnP frameworks**: Decoupled pipelines for perception and prediction suffer from error propagation.

### Core Problem

Three key questions in multi-agent multi-frame collaboration: (1) What information to transmit? (2) When to transmit? (3) How to fuse temporal and spatial dimensions across multiple agents?

## Method

### Overall Architecture

V2XPnP consists of six modules: V2X metadata sharing, LiDAR feature extraction (PointPillar), multi-frame temporal fusion, compression and sharing, multi-agent spatial fusion, and map feature extraction, followed by detection and prediction heads. The framework adopts intermediate feature fusion with a one-step communication strategy, in which each agent first fuses historical BEV features locally and then transmits the compressed single-frame feature to the ego agent.

### Key Designs

#### 1. **One-step Communication**

- **Function**: Each agent shares the fused result of all historical data in a single communication round, rather than transmitting frames iteratively over multiple steps.
- **Mechanism**: Each agent first locally fuses the historical BEV feature sequence $\mathbf{F}_i^{seq} \in \mathbb{R}^{T \times H \times W \times C}$ into a single frame $\mathbf{F}'_i \in \mathbb{R}^{H \times W \times C}$, keeping the transmission volume comparable to single-frame collaborative perception.
- **Design Motivation**: Multi-step communication introduces cumulative latency, data loss, and the risk that neighboring agents may fall outside the communication range in historical frames. One-step communication preserves full spatio-temporal information while reducing transmission from $5 \times 0.269$ Mb to $0.269$ Mb, with a latency of approximately 10–20 ms.

#### 2. **Spatio-Temporal Fusion Transformer**

- **Function**: Achieves unified spatio-temporal fusion through three attention modules.
- **Mechanism**:

**Temporal Attention**: Fuses features at the same spatial location across frames, with learnable timestamp embeddings:
$$\mathbf{F}_i^{tem} = \text{MHSA}(Q: \text{MLP}(\mathbf{F}_i^{seq'}), K: \text{MLP}(\mathbf{F}_i^{seq'}), V: \text{MLP}(\mathbf{F}_i^{seq'}))$$

**Self Spatial Attention**: Employs multi-scale window attention (local/intermediate/global windows) to capture BEV spatial interactions at different scales within a single agent.

**Multi-Agent Spatial Attention**: Employs a heterogeneous design with independent learnable weights for different interaction pairs (V-I, V-V, I-V, I-I):
$$\mathbf{F}_{i,m}^{sp} = \sum_j \text{Softmax}(\mathbf{Q}_i^m \cdot \mathbf{W}_{att}^{(e_{i,j})} \cdot \mathbf{K}_j^n) \cdot \mathbf{V}_j^n$$

- **Design Motivation**: Temporal and spatial information require separate modeling to preserve their respective structural properties. Heterogeneous attention weights account for deployment differences between vehicle and infrastructure sensors.

#### 3. **Map Feature Injection**

- **Function**: Encodes vectorized polylines from HD maps and injects them into BEV features.
- **Mechanism**: Surrounding map polylines for each BEV grid cell are encoded via MLP and fused through BEV-map self-attention:

$$\mathbf{F} = \text{MHSA}(Q: [\mathbf{F}_{bm}, \mathbf{P}_m], K: [\mathbf{F}_{bm}, \mathbf{P}_m], V: \mathbf{F}_{bm})$$

- **Design Motivation**: Map information provides road structure constraints for trajectory prediction, guiding predicted trajectories to follow road directions.

### Loss & Training

- **Perception loss**: Smooth L1 regression loss (location, size, orientation) + Focal Loss classification loss.
- **Prediction loss**: L2 loss (predicted trajectory points vs. ground-truth trajectory).
- **Total loss**: Weighted sum of the three terms.
- Train/val/test split: 76/6/14 scenes.
- Communication range: 50 m; evaluation range: $x \in [-70, 70]$ m, $y \in [-40, 40]$ m.
- History length: 2s (2 Hz); prediction horizon: 3s (2 Hz).

## Key Experimental Results

### Main Results

| Collab. Mode | Method | End-to-End | Map | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|---|---|---|---|---|---|---|---|---|
| VC | No Fusion | ✓ | | 43.9 | 1.87 | 3.24 | 33.8 | 24.3 |
| VC | Late Fusion | | ✓ | 58.1 | 1.59 | 2.82 | 32.4 | 33.0 |
| VC | V2X-ViT* | ✓ | ✓ | 69.6 | 1.39 | 2.56 | 35.2 | 44.7 |
| VC | **V2XPnP** | **✓** | **✓** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |
| V2V | No Fusion | ✓ | | 40.8 | 1.99 | 3.38 | 34.0 | 19.8 |
| V2V | V2X-ViT* | ✓ | ✓ | 64.6 | 1.68 | 3.13 | 39.8 | 36.7 |
| V2V | **V2XPnP** | **✓** | **✓** | **70.5** | 1.78 | 3.28 | 39.9 | **40.6** |
| IC | V2X-ViT* | ✓ | ✓ | 69.3 | 1.27 | 2.39 | 35.4 | 43.3 |
| IC | **V2XPnP** | **✓** | **✓** | **71.0** | **1.18** | **2.16** | **34.0** | **46.0** |

V2XPnP achieves the best EPA across all collaboration modes (VC +3.5, IC +2.7, V2V +3.9, I2I +1.2).

### Ablation Study

| Temporal Fusion | Spatial Fusion | Map Fusion | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|---|---|---|---|---|---|---|---|
| | | | 43.9 | - | - | - | - |
| ✓ | | | 57.2 | 1.52 | 2.76 | 35.5 | 33.8 |
| ✓ | ✓ | | 71.3 | 1.48 | 2.70 | 36.2 | 44.4 |
| **✓** | **✓** | **✓** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |

Communication strategy comparison:

| Strategy | AP@0.5↑ | ADE↓ | FDE↓ | MR↓ | EPA↑ |
|---|---|---|---|---|---|
| Multi-step | 68.2 | 1.56 | 2.84 | 31.8 | 43.0 |
| **One-step** | **71.6** | **1.35** | **2.36** | **31.7** | **48.2** |

### Key Findings

- **Temporal fusion is the critical foundation**: Adding temporal fusion improves AP from 43.9 to 57.2 (+13.3), even surpassing decoupled Late Fusion (55.3–61.3).
- **One-step communication consistently outperforms multi-step**: AP +3.4, EPA +5.2, with transmission volume equivalent to single-frame collaborative perception.
- **Map fusion primarily improves prediction performance**: AP barely changes (71.3→71.6), while ADE drops from 1.48 to 1.35 and EPA improves from 44.4 to 48.2.
- **End-to-end outperforms decoupled pipelines**: FaF* (end-to-end without fusion) achieves better detection than the decoupled no-fusion model and comparable performance to Late Fusion.
- **Necessity of heterogeneous attention**: Unified weights degrade performance due to deployment differences between vehicle and infrastructure sensors.
- **V2XPnP maintains strong performance at 128× compression ratio**, demonstrating superior robustness compared to V2X-ViT*.

## Highlights & Insights

1. **Systematic analysis of V2X spatio-temporal fusion**: The first work to comprehensively investigate the design space of "what to transmit, when to transmit, and how to fuse" in V2X settings.
2. **Superiority of one-step communication**: Counterintuitively, transmitting the fused result in a single step outperforms multi-step per-frame transmission, as it avoids cumulative errors and communication instability.
3. **Introduction of the EPA metric**: Jointly evaluates perception and prediction performance, preventing inflated prediction scores caused by a weak detection module that coincidentally captures simple trajectories.
4. **First real-world sequential dataset covering all collaboration modes**: 96 scenes, 40K frames, 4 agents, supporting V2V/V2I/I2I/VC/IC.

## Limitations & Future Work

1. **LiDAR-only input**: Incorporating camera data may further improve performance.
2. **Fixed communication range of 50 m**: Longer-range communication and scenarios with more agents remain unexplored.
3. **Limited dataset scale**: 96 scenes may be insufficient to cover all complex traffic scenarios.
4. **Prediction horizon of only 3s**: Longer-horizon prediction is more valuable for safe driving but considerably more challenging.
5. **Communication failures not considered**: Robustness to real-world packet loss and latency jitter requires dedicated design.
6. **Relatively outdated PointPillar backbone**: More advanced backbones such as VoxelNet or CenterPoint may yield further improvements.

## Related Work & Insights

- FaF and PnPNet are classic end-to-end perception-prediction frameworks; V2XPnP extends them to multi-agent settings.
- V2X-ViT is the strongest existing intermediate-fusion V2X model; V2XPnP comprehensively surpasses it through spatio-temporal fusion.
- V2X-Seq is the only existing V2X sequential dataset, but is limited to V2I and has restricted download access.
- CoBEVFlow and FFNet leverage short-horizon history (0.5s) to address asynchrony; V2XPnP extends the temporal horizon to 2s and supports prediction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The end-to-end spatio-temporal fusion framework design for V2X scenarios demonstrates systematic innovation, though individual components rely on relatively mature techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 baseline models, 4 collaboration modes, and extensive ablation and robustness evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; the "what, when, how" analytical framework is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — The dataset fills a critical gap in real-world V2X sequential data; the framework and benchmark provide substantial value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](../../AAAI2026/time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](../../NeurIPS2025/time_series/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)

</div>

<!-- RELATED:END -->
