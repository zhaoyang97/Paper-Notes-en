---
title: >-
  [Paper Note] Availability-aware Sensor Fusion via Unified Canonical Space
description: >-
  [NeurIPS 2025][Autonomous Driving][Multi-sensor fusion] This paper proposes ASF (Availability-aware Sensor Fusion), which maps Camera/LiDAR/4D Radar features into a shared space via Unified Canonical Projection (UCP), applies cross-sensor along-patch cross-attention (CASAP, complexity $O(N_qN_s)$ vs. $O(N_qN_sN_p)$) to automatically adapt to available sensors, and employs a Sensor Combination Loss (SCL) covering all 7 sensor subsets. ASF achieves AP_3D of 73.6% on K-Radar (su…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Multi-sensor fusion"
  - "sensor degradation robustness"
  - "unified canonical space"
  - "4D Radar"
  - "CASAP"
date: 2026-05-08
content_hash: 96c5d8d037e3ce3f
---

# Availability-aware Sensor Fusion via Unified Canonical Space

**Conference**: NeurIPS 2025
**arXiv**: [2503.07029](https://arxiv.org/abs/2503.07029)  
**Code**: [https://github.com/kaist-avelab/k-radar](https://github.com/kaist-avelab/k-radar)  
**Area**: Autonomous Driving / Sensor Fusion
**Keywords**: Multi-sensor fusion, sensor degradation robustness, unified canonical space, 4D Radar, CASAP

## TL;DR
This paper proposes ASF (Availability-aware Sensor Fusion), which maps Camera/LiDAR/4D Radar features into a shared space via Unified Canonical Projection (UCP), applies cross-sensor along-patch cross-attention (CASAP, complexity $O(N_qN_s)$ vs. $O(N_qN_sN_p)$) to automatically adapt to available sensors, and employs a Sensor Combination Loss (SCL) covering all 7 sensor subsets. ASF achieves AP_3D of 73.6% on K-Radar (surpassing SOTA by 20.1%), with only a 1.7% performance drop under sensor failure.

## Background & Motivation
**Background**: Multi-sensor fusion (Camera+LiDAR+Radar) has become mainstream in autonomous driving. Existing fusion methods fall into two categories: (a) Deep Coupling Fusion (DCF), which directly concatenates sensor features—simple and efficient but assumes all sensors are always available; and (b) Sensor-level Cross-attention Fusion (SCF), which applies cross-attention over per-sensor patches and can handle missing sensors, but incurs computational cost of $O(N_qN_sN_p)$.

**Limitations of Prior Work**: (a) DCF suffers severe performance degradation upon sensor failure and requires separate models for different sensor combinations; (b) SCF lacks a unified feature representation, resulting in inconsistent sensor features in the latent space, with computational cost exploding with the number of patches (e.g., CMT requires 8 A100 GPUs for training).

**Key Challenge**: Features extracted from heterogeneous sensors (2D RGB / 3D point clouds / 4D Radar tensors) are inherently inconsistent—the feature distributions for the same object differ substantially across sensors, making direct fusion both ineffective and brittle.

**Goal**: To design a method that aligns sensor features in a **unified space**, while **automatically adapting** to sensor availability at minimal computational cost.

**Key Insight**: Inspired by Mobileye's "True Redundancy" concept—sensors should operate independently yet complement each other through a canonical representation.

**Core Idea**: UCP for unified feature space + CASAP performing attention only across sensors (not across patches) + SCL covering all sensor combinations during training, enabling robust and efficient availability-aware fusion.

## Method

### Overall Architecture
A three-stage pipeline: (1) sensor-specific encoders—BEVDepth (camera), SECOND (LiDAR), and RTNH (4D Radar)—extract BEV feature maps $\mathbf{FM}^s \in \mathbb{R}^{C_s \times H \times W}$ of identical spatial dimensions; (2) ASF network—UCP projects features into a unified space, followed by CASAP cross-sensor attention; (3) SSD detection head for object detection from the fused feature map.

### Key Designs

1. **Unified Canonical Projection (UCP)**:

    - Function: Projects each sensor's BEV feature into a unified space of dimension $C_u$.
    - Mechanism: Each sensor's BEV FM is divided into an equal number of patches $\mathbf{F}_{p,i}^s \in \mathbb{R}^{C_s \times P_H \times P_W}$, which are then projected to $\mathbf{F}_{u,i}^s \in \mathbb{R}^{C_u}$ via sensor-specific MLP+GeLU+LayerNorm.
    - Key insight: Since patches are spatially aligned (same-position patches from C/L/R correspond to the same region), **no positional encoding is needed**, eliminating the expensive position embeddings used in SCF.
    - Design Motivation: t-SNE visualization confirms that after UCP, features from different sensors align well with the fused representation.

2. **Cross-sensor Along-Patch Attention (CASAP)**:

    - Function: For each patch location, performs cross-attention only among $N_s$ (at most 3) keys/values across sensors.
    - Core formula: $\mathbf{Q}'_{ref,i} = \text{CrossAttn}(Q=\mathbf{Q}_{ref}, K\&V \in \{\mathbf{F}_{u,i}^{S_C}, \mathbf{F}_{u,i}^{S_L}, \mathbf{F}_{u,i}^{S_R}\})$
    - Complexity: $O(N_qN_s)$, where $N_s \leq 3$ is constant—far lower than SCF's $O(N_qN_sN_p)$.
    - Automatic availability awareness: $\mathbf{Q}_{ref}$ learns during training to assign higher attention weights to available/reliable sensors. In adverse weather, camera attention automatically decreases while LiDAR/Radar attention increases.
    - Post-normalization (PN): An additional MLP+LN projection is applied to CASAP outputs to ensure consistency across different sensor combinations.

3. **Sensor Combination Loss (SCL)**:

    - Function: Computes detection losses over all 7 sensor combinations (C/L/R/CL/CR/LR/CLR) during training.
    - Mechanism: $\mathcal{L}_{SCL} = \sum_{s \in \mathcal{S}} (\mathcal{L}_{cls}^s + \mathcal{L}_{reg}^s)$, where $\mathcal{S}$ denotes all possible sensor subsets.
    - Design Motivation: Exposes the model to all sensor-missing scenarios, enabling robust performance under any available subset.

### Loss & Training
Training requires only a single RTX 3090 GPU (1.5–1.6 GB VRAM), compared to 8 A100 GPUs for CMT. AdamW optimizer, 25 epochs.

## Key Experimental Results

### Main Results
3D object detection on K-Radar (IoU=0.5):

| Method | Sensors | AP_BEV↑ | AP_3D↑ | Heavy Snow AP_3D↑ |
|--------|---------|---------|--------|-------------------|
| RTNH | R only | 36.0 | 14.1 | 6.36 |
| RTNH | L only | 66.3 | 37.8 | 24.6 |
| 3D-LRF | L+R | 73.6 | 45.2 | 36.9 |
| L4DR | L+R | 77.5 | 53.5 | 37.0 |
| **ASF** | L+R | **87.0** | **72.9** | **66.7** |
| **ASF** | C+L+R | **87.2** | **73.6** | **66.4** |

### Ablation Study

| Configuration | AP_3D (IoU=0.5)↑ |
|---------------|-------------------|
| ASF C+L+R (full) | 73.6 |
| Camera failure C*+L+R | 71.9 (only −1.7%) |
| L+R | 72.9 |
| R only | 40.0 |
| L only | 55.0 |
| C only | 15.2 |

### Key Findings
- ASF surpasses all SOTA methods by 20.1% AP_3D—a remarkably large improvement margin.
- **True redundancy achieved**: C+L+R and L+R perform almost identically (73.6 vs. 72.9), demonstrating that the model learns to ignore the camera when it is unnecessary.
- In adverse weather, camera attention automatically drops to ~5%, with LiDAR/Radar compensating—verified through attention weight visualization (SAM).
- Computationally highly efficient: trainable on a single RTX 3090 at 20.5 Hz inference, compared to 5.0 Hz for DPFT.
- L+R alone achieves AP_3D of 66.7 in heavy snow, far exceeding L4DR's 37.0, owing to the full exploitation of 4D Radar's advantage in adverse weather.

## Highlights & Insights
- **Elegant simplification in CASAP**: The design decision to avoid cross-patch attention is critical—it reduces complexity from $O(N_qN_sN_p)$ to $O(N_qN_s)$ while improving performance. Spatially aligned patches inherently establish correspondence, eliminating the need for positional encodings.
- **Necessity of the SCL training strategy**: Computing losses over all 7 sensor combinations endows the model with natural robustness to any sensor subset—a more thorough approach than sensor dropout.
- **Realization of True Redundancy**: t-SNE visualizations clearly illustrate the three-stage feature alignment process (UCP→CASAP→PN), from scattered to clustered to unified distributions.
- **Minimal resource requirement**: Trainable on a single RTX 3090, representing an orders-of-magnitude reduction from CMT's 8×A100 requirement.

## Limitations & Future Work
- Validation is limited to the K-Radar dataset, which is relatively small in scale—evaluation on large-scale benchmarks such as nuScenes is needed.
- 4D Radar data formats vary significantly across sensor manufacturers, and generalizability remains to be verified.
- UCP alignment quality may be sensitive to initialization—stronger alignment methods such as contrastive learning could be explored.
- The current work addresses only detection tasks; applicability to dense prediction tasks such as semantic segmentation remains to be investigated.

## Related Work & Insights
- **vs. 3D-LRF / L4DR (DCF methods)**: These methods assume all sensors are available and require retraining for different sensor combinations. ASF handles all 7 combinations with a single set of weights.
- **vs. CMT / DPFT (SCF methods)**: These methods are computationally expensive and rely on positional encodings. ASF eliminates positional encodings through spatially aligned patches, reducing complexity by orders of magnitude.
- **True Redundancy concept**: Originally from an industrial paper by Mobileye; ASF represents its first systematic realization in an academic framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The UCP+CASAP unified fusion framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full combination testing, adverse weather evaluation, efficiency analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-organized; t-SNE visualizations are highly convincing.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for sensor robustness in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)
- [\[CVPR 2025\] CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](../../CVPR2025/autonomous_driving/cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](../../CVPR2025/autonomous_driving/rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)

</div>

<!-- RELATED:END -->
