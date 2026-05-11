---
title: >-
  [Paper Note] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction
description: >-
  [AAAI 2026][Human Understanding][Multi-person motion prediction] This paper proposes ST-MoE, the first framework to combine Mixture of Experts (MoE) with bidirectional spatiotemporal Mamba for multi-person motion predict…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "Multi-person motion prediction"
  - "Mixture of Experts"
  - "Mamba"
  - "spatiotemporal modeling"
  - "efficient inference"
date: 2026-05-08
content_hash: af8f2388a7441489
---

# Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction

**Conference**: AAAI 2026
**arXiv**: [2512.21707](https://arxiv.org/abs/2512.21707)
**Code**: [https://github.com/alanyz106/ST-MoE](https://github.com/alanyz106/ST-MoE)
**Area**: Human Understanding
**Keywords**: Multi-person motion prediction, Mixture of Experts, Mamba, spatiotemporal modeling, efficient inference

## TL;DR

This paper proposes ST-MoE, the first framework to combine Mixture of Experts (MoE) with bidirectional spatiotemporal Mamba for multi-person motion prediction. Four heterogeneous spatiotemporal experts flexibly capture complex spatiotemporal dependencies, achieving state-of-the-art accuracy while reducing parameter count by 41.38% and accelerating training by 3.6×.

## Background & Motivation

Multi-Person Motion Prediction (MPMP) aims to forecast future joint positions of multiple individuals from historical motion sequences, with important applications in human-robot interaction, autonomous driving, and surveillance systems.

Existing methods suffer from two core limitations:

**Inflexible spatiotemporal representations**:
   - MRT employs fixed-pattern spatiotemporal positional encodings, lacking flexibility
   - TBIFormer introduces trajectory-aware relative positional encodings to enhance spatial awareness, but body-part connectivity operations increase sequence length
   - IAFormer uses self-attention to explore spatiotemporal features in interaction information, yielding strong performance but poor efficiency

**High computational cost**:
   - Transformer-based methods incur large computational overhead due to the quadratic complexity of self-attention
   - Computational cost grows rapidly as the number of persons increases

Core motivation: **Can a novel paradigm be designed that is both flexible and efficient, comprehensively capturing spatiotemporal dependencies in human motion?**

The authors' insight is threefold: (1) the dynamic activation mechanism of MoE enables flexible sub-network selection; (2) Mamba's linear complexity can replace quadratic-complexity attention; (3) combining both simultaneously addresses flexibility and efficiency.

## Method

### Overall Architecture

Input motion sequences → DCT + Multi-Pose Encoder → gated router assigns features to 4 spatiotemporal experts → expert outputs are aggregated via weighted summation → Multi-Pose Decoder + iDCT → predicted future motion.

### Key Designs

#### 1. Problem Formulation and Input Processing

The historical motion sequence of the $i$-th person is defined as $\textbf{P}_{1:t}^i \in \mathbb{R}^{D \times t}$, and the goal is to predict future motion $\textbf{P}_{t+1:T}^i \in \mathbb{R}^{D \times (T-t)}$, where $D = J \times 3$ denotes 3D coordinates of $J$ joints.

**Input padding**: The last observed frame is replicated $T-t$ times and concatenated to the observation sequence, forming $\textbf{P}_{\text{input}}^i \in \mathbb{R}^{D \times T}$.

**Multi-Pose Encoder**: A 3-layer GCN encoder from IAFormer is adopted. A DCT transform is applied beforehand to improve representation compactness:

$$\textbf{F}_{\text{input}}^i = \text{ME}(\text{DCT}(\textbf{P}_{\text{input}}^i))$$

#### 2. Mixture of Spatiotemporal Mamba Experts (MoSTME)

This is the core contribution. The encoded features $\textbf{F}_{\text{input}} \in \mathbb{R}^{B \times D \times T}$ are fed simultaneously into the expert pool and the router.

**Gated routing mechanism**:

$$\textbf{E}_{\text{output}} = \sum_{e=1}^{N} \textbf{f}_e(\textbf{F}_{\text{input}}) \textbf{p}_e$$

$$\textbf{p}_e = \text{softmax}(\text{TopK}(g(\textbf{F}_{\text{input}}), k))_e$$

where $g(\cdot)$ is an MLP-based gating function. TopK retains the original values of the top-$k$ entries and sets the rest to $-\infty$; after softmax, inactive experts receive near-zero weights, achieving sparse activation. Experiments confirm that activating all experts ($k=4$) yields the best performance.

**Four heterogeneous experts** — each composed of a different combination of bidirectional spatial Mamba and bidirectional temporal Mamba:

| Expert Type | Processing Order | Captured Pattern |
|-------------|-----------------|-----------------|
| Spatial-Temporal (ST) | Spatial → Temporal | Spatial-then-temporal dependencies |
| Temporal-Temporal (TT) | Temporal → Temporal | Strong temporal dependencies |
| Temporal-Spatial (TS) | Temporal → Spatial | Temporal-then-spatial dependencies |
| Spatial-Spatial (SS) | Spatial → Spatial | Strong spatial dependencies |

Taking the ST expert as an example:

$$\textbf{F}'' = \text{rearrange}(\text{Bi-SMamba}(\textbf{F}_{\text{input}}))$$

$$\textbf{F}_{\text{output}_1} = \text{rearrange}(\text{Bi-TMamba}(\textbf{F}''))$$

**Key design**: All experts share the same set of bidirectional temporal Mamba and bidirectional spatial Mamba parameters; only the combination order differs, substantially reducing parameter count.

#### 3. Bidirectional Spatiotemporal Mamba

The unidirectional modeling of vanilla Mamba limits global dependency capture. A bidirectional scanning mechanism is introduced:

$$\textbf{f}_o^s = \text{SMamba}(\overrightarrow{\textbf{f}_s}) + \text{SMamba}(\overleftarrow{\textbf{f}_s}) + \overrightarrow{\textbf{f}_s}$$

$$\textbf{f}_o^t = \text{TMamba}(\overrightarrow{\textbf{f}_t}) + \text{TMamba}(\overleftarrow{\textbf{f}_t}) + \overrightarrow{\textbf{f}_t}$$

Feature representations are then enhanced via LayerNorm + FFN + residual connection:

$$\textbf{F}_o^\star = \text{LN}(\text{LN}(\textbf{f}_o^\star) + \text{FFN}(\text{LN}(\textbf{f}_o^\star)))$$

**Spatial Mamba** scans along the pose dimension $D$; **Temporal Mamba** scans along the temporal dimension $T$. Both follow the standard Selective SSM architecture with discretization and input-dependent $A$, $B$, $C$ matrices.

### Loss & Training

**Spatial loss** $L_s$: constrains joint positions in both the observed and predicted intervals

$$L_s = \frac{\lambda}{J \cdot M \cdot t}\sum_{m,j,i=1}^{t}\|\hat{\textbf{P}}_{i,j}^m - \textbf{P}_{i,j}^m\|^2 + \frac{1}{J \cdot M \cdot (T-t)}\sum_{m,j,i=t+1}^{T}\|\hat{\textbf{P}}_{i,j}^m - \textbf{P}_{i,j}^m\|^2$$

**Temporal consistency loss** $L_t$: mitigates temporal jitter in predicted motion

$$L_t = \text{MSE}(\text{Conv}(\textbf{P}_{\text{pred}}), \text{Conv}(\textbf{P}_{\text{gt}}))$$

**Total loss**: $L = \alpha L_s + \beta L_t$, with $\alpha=1, \beta=1, \lambda=0.1$

**Training configuration**: batch size 96, Adam optimizer, initial learning rate 0.01, exponential decay ($0.1^{1/50}$/epoch), single RTX 3090 GPU.

## Key Experimental Results

### Main Results

**CMU-Mocap (UMPM) dataset — JPE (mm)**:

| Method | 0.2s | 0.6s | 1.0s | Avg |
|--------|------|------|------|-----|
| MRT | 36 | 115 | 193 | 114 |
| TBIFormer | 30 | 109 | 182 | 107 |
| JRFormer | 32 | 104 | 161 | 99 |
| IAFormer | 32 | 96 | 159 | 96 |
| **ST-MoE (Ours)** | **31** | **95** | **158** | **95** |

**CHI3D dataset — JPE (mm)**:

| Method | 0.2s | 0.4s | 0.6s | 0.8s | 1.0s | Avg |
|--------|------|------|------|------|------|-----|
| TBIFormer | 45 | 95 | 145 | 192 | 233 | 142 |
| IAFormer | 39 | 83 | 129 | 176 | 218 | 129 |
| **ST-MoE (Ours)** | 44 | **79** | **123** | **161** | **200** | **121** |

Average JPE is reduced by 8 mm relative to IAFormer and by 21 mm relative to TBIFormer.

**Efficiency comparison**: 41.38% fewer parameters and 3.6× training speedup (vs. IAFormer).

### Ablation Study

**Effectiveness of heterogeneous experts (CMU-Mocap UMPM)**:

| Configuration | Avg JPE (↓) | Avg APE (↓) | Note |
|---------------|------------|------------|------|
| Baseline (Encoder/Decoder only) | 111.1 | 73.3 | No experts |
| +ST expert ×4 | 104.5 | 70.7 | Spatial-temporal only |
| +TT expert ×4 | 98.1 | 66.4 | Temporal-temporal only |
| +TS expert ×4 | 100.1 | 68.7 | Temporal-spatial only |
| +SS expert ×4 | 98.3 | 68.2 | Spatial-spatial only |
| **+All (one of each)** | **95.0** | **65.4** | Heterogeneous combination is optimal |

**Effectiveness of bidirectional scanning**:

| Scanning Strategy | Avg JPE | Avg APE |
|-------------------|---------|---------|
| Forward only | 99.3 | 67.5 |
| Backward only | 98.9 | 67.0 |
| **Bidirectional** | **95.0** | **65.4** |

Bidirectional scanning reduces JPE by 4.3 mm and APE by 3.9 mm compared to unidirectional scanning.

### Key Findings

1. **Activating all experts is optimal**: Experiments show that activating all 4 experts yields the best performance, with JPE/APE consistently decreasing as more experts are activated.
2. **Single MoE layer is best**: Stacking additional MoE layers leads to overfitting.
3. **Heterogeneous outperforms homogeneous**: The combination of four distinct expert types significantly surpasses using four copies of any single expert type.
4. **t-SNE visualization** confirms that the four experts learn distinct feature distributions, forming clearly separated clusters.
5. **Adaptive gating weight visualization**: TT/ST experts tend to capture approximately static motion patterns, while SS/TS experts tend to capture spatially dynamic patterns.

## Highlights & Insights

1. **Elegant combination of MoE and Mamba**: MoE provides flexible expert selection, while Mamba's linear complexity replaces the quadratic complexity of attention — two orthogonal improvements that reinforce each other.
2. **The parameter-sharing design is ingenious** — all four experts share the same Mamba parameters and differ only in combination order, achieving diverse functionality with minimal parameters.
3. **Qualitative analysis is convincing**: t-SNE and gating weight visualizations intuitively demonstrate the mechanism by which different experts capture distinct motion patterns (static vs. dynamic, spatial vs. temporal).
4. The framework is general and can be extended to other sequence prediction tasks requiring spatiotemporal modeling.

## Limitations & Future Work

1. **Deterministic prediction only**: The current method outputs a single deterministic trajectory; future work should extend to stochastic multi-person motion prediction.
2. **Scene limitations**: Experiments are conducted primarily in laboratory settings with few persons (2–10); performance in dense crowd scenarios remains unvalidated.
3. **Single MoE layer limitation**: The authors find that multiple MoE layers cause overfitting, suggesting a need for improved regularization strategies.
4. **Fixed number of four experts**: The design space of expert types warrants further exploration (e.g., introducing cross-person interaction experts).

## Related Work & Insights

- **IAFormer** is the primary baseline, using attention to learn spatiotemporal interaction information with strong performance but poor efficiency.
- **Mamba** proposes a selective scanning mechanism for long-range dependency modeling with linear inference complexity.
- **MoE-Mamba** alternately stacks MoE and Mamba layers; the proposed approach — embedding Mamba inside experts — is more lightweight.
- The idea of applying heterogeneous experts to motion prediction can inspire other spatiotemporal modeling tasks such as traffic flow prediction and action recognition.

## Rating

- Novelty: ⭐⭐⭐⭐ — The heterogeneous expert design combining MoE and Mamba is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets, extensive ablations, and visualization analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and well-designed figures
- Value: ⭐⭐⭐⭐ — A benchmark work for the efficiency–accuracy trade-off

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[AAAI 2026\] Improving Sparse IMU-based Motion Capture with Motion Label Smoothing](improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)
- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](../../ICCV2025/human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)
- [\[AAAI 2026\] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)

</div>

<!-- RELATED:END -->
