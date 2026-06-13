---
title: >-
  [Paper Note] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery
description: >-
  [AAAI 2026][3D Vision][Human Mesh Recovery] This paper proposes PressTrack-HMR, the first top-down pipeline for multi-person global human mesh recovery using only pressure signals. It introduces a novel UoE similarity me…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Pressure Sensing"
  - "Multi-Person Tracking"
  - "Privacy Preservation"
  - "Tactile Interaction"
date: 2026-05-08
content_hash: 13ebe2d1955dbb10
---

# PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery

**Conference**: AAAI 2026
**arXiv**: [2511.09147](https://arxiv.org/abs/2511.09147)  
**Code**: [github.com/Jiayue-Yuan/PressTrack-HMR](https://github.com/Jiayue-Yuan/PressTrack-HMR)  
**Area**: 3D Vision
**Keywords**: Human Mesh Recovery, Pressure Sensing, Multi-Person Tracking, Privacy Preservation, Tactile Interaction

## TL;DR

This paper proposes PressTrack-HMR, the first top-down pipeline for multi-person global human mesh recovery using only pressure signals. It introduces a novel UoE similarity metric for pressure footprint tracking (93.6% MOTA) and presents MIP, the first multi-person interaction pressure dataset.

## Background & Motivation

Multi-person global human mesh recovery (HMR) is essential for understanding crowd dynamics and interactions. However, **existing RGB-based HMR methods face three critical bottlenecks in real-world scenarios**:

**Occlusion**: Inevitable mutual occlusions among pedestrians in crowded environments limit the completeness of single-viewpoint information, while deploying multi-camera systems is costly and complex.

**Illumination**: Insufficient lighting degrades visual data quality and availability.

**Privacy**: Growing privacy protection demands make camera-based solutions increasingly undesirable in sensitive environments such as homes and hospitals.

The authors observe that **tactile interactions between humans and the ground provide rich pressure information** that naturally circumvents the above issues. Pressure data is robust to occlusion and illumination, and inherently offers privacy protection advantages.

However, extending pressure-based sensing from single-person to multi-person scenarios introduces **two core challenges**:

**Intra-frame separation**: When multiple individuals walk on a tactile mat simultaneously, their pressure signals become interleaved (as shown in Figure 1a), requiring separation of signals from different individuals.

**Inter-frame association**: Pressure signals belonging to the same individual must be linked across consecutive time steps to obtain temporally continuous single-person data for more accurate HMR.

The authors further analyze the unique dynamics of pressure footprints, which distinguish them from visual tracking targets:
- **Abrupt size changes**: Alternation between single-foot and double-foot contact causes sudden bounding box size variations, reducing IoU reliability.
- **Discontinuous jumping motion**: Bounding box motion is non-smooth and jump-like, invalidating smooth motion predictors such as Kalman filtering.

These characteristics render existing visual tracking methods (ByteTrack, BoT-SORT) inapplicable without modification.

## Method

### Overall Architecture

PressTrack-HMR adopts a **top-down** pipeline comprising two main modules:
1. **PressTrack Module**: Extracts per-individual temporal pressure signals from raw pressure data using a tracking-by-detection strategy.
2. **HMR Module**: Reconstructs global human meshes from temporally coherent single-person pressure image sequences.

### Key Designs

#### 1. **Intra-frame Footprint Detection**

**Function**: Detects per-individual footprint bounding boxes from raw pressure data.

A pretrained YOLOv11 model is fine-tuned for pressure footprint detection. The automated label generation pipeline (Figure 4) proceeds as follows:
- OpenCV thresholding extracts initial discrete pressure regions.
- Foot joint positions (left/right toe and ankle joints) are derived from 3D joint coordinates in RGB data and projected onto the 2D pressure mat coordinate system.
- Each pressure region is assigned to the individual whose foot joint is closest:

$$\text{ID}(r_j) = \arg\min_{i \in \{1,\ldots,N\}} \min_{f \in F_i} \|c_j - f\|_2$$

- All regions sharing the same ID are merged into a minimum enclosing rectangle as the training label.

**Design Motivation**: Joint information from RGB data is leveraged to automatically generate training labels, avoiding the prohibitive cost of manual annotation.

#### 2. **Inter-frame Footprint Association — UoE Metric**

**Function**: Associates pressure footprints belonging to the same individual across frames. This constitutes the primary technical contribution of the paper.

The paper proposes Union over Enclosure (UoE) as a replacement for conventional IoU:

$$\text{UoE} = \frac{|A \cup B|}{|C|}$$

where $A$ and $B$ are detection bounding boxes from the current and previous frames, $C$ is the minimum enclosing rectangle of $A$ and $B$, and $|\cdot|$ denotes area.

**Why not IoU?** Pressure footprints undergo abrupt size changes when transitioning between single-foot and double-foot contact, making IoU unreliable. UoE exploits a key observation: **pressure footprints of different individuals do not overlap simultaneously**, making normalization by the enclosing area more stable than normalization by intersection area.

**Why not motion prediction?** The jumping, discontinuous nature of pressure footprint motion (due to alternating footsteps) renders smooth motion prediction models such as Kalman filtering inappropriate. A UoE cost matrix is therefore computed directly between adjacent frames, and the Hungarian algorithm is applied for optimal matching.

Trajectory management strategy:
- Low-confidence unmatched detections are discarded; high-confidence ones initialize new trajectories (new individual entering).
- Unmatched trajectories are temporarily marked as "lost" and retained for several frames (to handle brief double-foot jumps, etc.).
- Trajectories with prolonged non-matching are considered to have left the scene.

#### 3. **Human Mesh Recovery Module**

**Function**: Recovers SMPL parameters from temporal single-person pressure image sequences.

The architecture consists of three components (Figure 5):

- **Image Encoder**: A ResNet encodes single-person pressure images ($128 \times 128$); the resulting features are concatenated with bounding box center coordinates $c_{\text{bbox}}$ and mat corner coordinates $c_{\text{mat}}$ (spatial priors) before dimensionality reduction.
- **Temporal Encoder**: A 2-layer Transformer encoder captures temporal dependencies via positional encoding.
- **SMPL Regressor**: An N-to-1 mapping (via feature averaging) with iterative error feedback (IEF) regresses SMPL parameters $(\boldsymbol{\theta}, \boldsymbol{\beta}, T)$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{SMPL}} + \mathcal{L}_{3D}$$

$$\mathcal{L}_{\text{SMPL}} = \lambda_\beta |\hat{\beta} - \beta| + \lambda_\theta |\hat{\theta} - \theta| + \lambda_T |\hat{T} - T|$$

$$\mathcal{L}_{3D} = \lambda_{3D} \|J_{3D} - \hat{J}_{3D}\|_2$$

Two dataset splitting strategies are employed:
- **Unseen Sequence Split**: 80%/20% split by sequence, evaluating generalization to unseen sequences of known individuals.
- **Unseen Subject Split**: Data from 2 volunteers is reserved exclusively for testing, evaluating generalization to entirely new individuals.

## Key Experimental Results

### Main Results

#### Footprint Tracking Performance

| Method | MOTA↑ | MOTP↑ | FN↓ | FP↓ | ID sw↓ | IDF1↑ |
|------|-------|-------|-----|-----|--------|-------|
| ByteTrack | 66.1% | 79.1% | 35207 | 22769 | 14453 | 4.8% |
| BoT-SORT | 82.5% | 87.7% | 17119 | 9020 | 10494 | 6.2% |
| **PressTrack** | **93.6%** | **94.8%** | **7317** | **6764** | **437** | **63.1%** |

PressTrack achieves only 437 ID switches (across 86 trajectories averaging 2,660 frames each, approximately once every 523 frames), substantially outperforming all baselines.

#### End-to-End HMR Performance

| Method | Data Split | MPJPE↓ | PA-MPJPE↓ | PVE↓ | WA-MPJPE100↓ | GMPJPE↓ |
|------|---------|--------|-----------|------|-------------|---------|
| GT Dets. | Unseen Seq | 81.8 | 46.1 | 115.7 | 90.9 | 99.4 |
| GT Dets. | Unseen Subj | 92.2 | 43.1 | 132.9 | 100.8 | 112.8 |
| Tracked Dets. | Unseen Seq | 89.2 | 48.8 | 134.4 | 112.6 | 118.3 |
| Tracked Dets. | Unseen Subj | 96.8 | 44.3 | 145.3 | 115.0 | 125.0 |

Using tracked versus ground-truth detections incurs only ~7.4 mm additional MPJPE, indicating that tracking errors have a manageable impact on HMR. The model generalizes well to unseen individuals (only 7.6 mm additional MPJPE).

### Ablation Study

| Sequence Length | GT Dets. MPJPE | Tracked Dets. MPJPE | Note |
|---------|---------------|---------------------|------|
| 1 | 94.2 | 96.2 | Single frame only |
| 4 | 88.7 | 92.3 | Temporal context helps |
| 8 | 86.6 | 90.7 | Continued improvement |
| 12 | 83.8 | 89.9 | |
| 16 | **81.8** | **89.2** | Optimal length |
| 32 | 82.1 | 89.7 | Performance degrades |

The optimal sequence length is 16 frames: shorter sequences provide insufficient temporal context, while longer sequences suffer from weakened temporal correlation and accumulated tracking errors.

### Key Findings

1. **UoE substantially outperforms visual tracking methods**: ID switching is reduced by 96.9%/95.8% compared to ByteTrack and BoT-SORT, respectively.
2. **Temporal information is critical**: Increasing sequence length from 1 to 16 reduces MPJPE by 12.4 mm.
3. **Tracking errors remain manageable**: The additional error introduced by tracking in the end-to-end pipeline is limited (7–8 mm MPJPE).
4. **Strong generalization**: Performance degradation on unseen subjects is minimal (7.6 mm MPJPE).

## Highlights & Insights

1. **Filling an important gap**: This is the first work to achieve multi-person global HMR solely from pressure signals, establishing a new paradigm for privacy-preserving motion analysis.
2. **Rigorous problem analysis**: The characterization of pressure footprint dynamics (abrupt size changes, jumping motion) directly motivates the UoE metric design, rather than naively adapting visual tracking methods.
3. **Dataset contribution**: The MIP dataset addresses the absence of multi-person interaction pressure data, with a resolution (1 cm × 1 cm) superior to all existing datasets.
4. **End-to-end feasibility demonstrated**: The complete pipeline from raw pressure data to multi-person 3D meshes and global trajectories is thoroughly validated.

## Limitations & Future Work

1. **Gap between GT and tracked detections**: A noticeable performance gap exists between the end-to-end pipeline and the isolated HMR module. Future work could analyze the effects of different tracking error types (ID switching vs. localization jitter) on HMR and optimize accordingly.
2. **Limited pressure mat coverage**: The 1.2 × 2.4 m² mat size constrains applicable scenarios.
3. **Reliance on multi-view RGB for annotation**: Dataset annotation depends on EasyMocap processing of multi-view video, resulting in high data collection costs.
4. **Complex interaction scenarios unexplored**: Large-scale crowds (>3 persons) and contact-based interactions (e.g., handshakes, hugs) remain to be investigated.

## Related Work & Insights

- **Comparison with visual MOT**: The paper provides a detailed analysis of the fundamental differences between pressure footprint tracking and visual tracking; the design of UoE embodies a principled, domain-specific methodology.
- **Broad application potential of pressure sensing**: Beyond walking scenarios, pressure sensing has applications in bed monitoring (PID-HMR), sports analysis (VP-MoCap), and related domains.
- **Trend toward privacy-first sensing**: This work represents a significant advancement in privacy-preserving sensing paradigms.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering work; multi-person pressure HMR, the UoE metric, and the MIP dataset are all proposed for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Detailed evaluation of both tracking and HMR, with ablation and generalization studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, dynamics analysis is thorough, and the pipeline description is complete.
- **Value**: ⭐⭐⭐⭐ — A practical solution for multi-person motion analysis in privacy-sensitive settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](../../CVPR2026/3d_vision/fall_risk_gait_analysis_hmr.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](../../CVPR2026/3d_vision/coherent_humanscene_reconstruction_from_multiperso.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/3d_vision/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
