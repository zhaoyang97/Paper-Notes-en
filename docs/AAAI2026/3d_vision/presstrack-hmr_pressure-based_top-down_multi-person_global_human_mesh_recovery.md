---
title: >-
  [Paper Note] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery
description: >-
  [AAAI 2026][3D Vision][Human Mesh Recovery] This paper proposes PressTrack-HMR, the first top-down pipeline for multi-person global human mesh recovery based solely on pressure signals. It achieves pressure footprint tracking via an innovative UoE similarity metric (93.6% MOTA) and builds MIP, the first multi-person interactive pressure dataset.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Pressure Sensing"
  - "Multi-Person Tracking"
  - "Privacy Protection"
  - "Tactile Interaction"
date: 2026-05-08
content_hash: 1180918d7832e655
---

# PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery

**Conference**: AAAI 2026  
**arXiv**: [2511.09147](https://arxiv.org/abs/2511.09147)  
**Code**: [github.com/Jiayue-Yuan/PressTrack-HMR](https://github.com/Jiayue-Yuan/PressTrack-HMR)  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Pressure Sensing, Multi-Person Tracking, Privacy Protection, Tactile Interaction

## TL;DR

This paper proposes PressTrack-HMR, the first top-down pipeline for multi-person global human mesh recovery based solely on pressure signals. It achieves pressure footprint tracking via an innovative UoE similarity metric (93.6% MOTA) and builds MIP, the first multi-person interactive pressure dataset.

## Background & Motivation

Multi-person global human mesh recovery (HMR) is crucial for understanding crowd dynamics and interactions. However, **existing RGB-based HMR methods face three major bottlenecks in practical scenarios**:

**Occlusion**: Inevitable mutual occlusion between pedestrians in crowded environments limits the completeness of single-view information, while deploying multiple cameras is costly and complex.

**Illumination**: Insufficient lighting diminishes the quality and usability of visual data.

**Privacy**: Growing demands for privacy protection render camera-based solutions increasingly unpopular in sensitive environments such as homes and hospitals.

The authors observe that **tactile interactions between humans and the ground provide rich pressure information**, which inherently bypasses the aforementioned issues. Pressure data is robust against occlusions and illumination variations, while naturally offering privacy-preserving advantages.

However, scaling pressure-based sensing from single-person to multi-person scenarios poses **two core challenges**:

**Intra-frame individual pressure signal separation**: When multiple people walk on the tactile mat simultaneously, their pressure signals interlace and mix (as shown in Figure 1a), requiring the separation of signals for different individuals.

**Inter-frame individual pressure signal association**: Individual pressure signals belonging to the same person must be associated across consecutive time frames to obtain continuous single-person temporal sequences for more accurate HMR.

Furthermore, the authors analyze the unique dynamic characteristics of pressure footprints (distinguishing them from visual tracking targets):
- **Sudden scale changes**: Alternating between single-foot and double-foot contact leads to sudden changes in the bounding box size, reducing the reliability of IoU.
- **Jump-like discontinuous motion**: The bounding box moves discretely, which invalidates smooth motion prediction models such as Kalman filtering.

These characteristics make existing visual tracking methods (e.g., ByteTrack, BoT-SORT) inapplicable directly.

## Method

### Overall Architecture

PressTrack-HMR adopts a **top-down** pipeline that consists of two main modules:
1. **PressTrack Module**: Extracts temporal pressure signals for each individual from raw pressure data using a tracking-by-detection paradigm.
2. **HMR Module**: Reconstructs global human meshes from the temporal single-person pressure maps.

### Key Designs

#### 1. **Intra-frame Footprint Detection**

**Function**: Detects the footprint bounding boxes of each individual from raw pressure data.

A pre-trained YOLOv11 model is fine-tuned for pressure footprint detection. The automatic label generation pipeline (Figure 4) works as follows:
- OpenCV thresholding is used to extract initial discrete pressure areas.
- Foot joint coordinates (left/right toes and ankles) are extracted from 3D joint data of RGB modality and projected onto the 2D coordinate system of the pressure mat.
- Each pressure area is assigned to the individual corresponding to the nearest foot joint:

$$\text{ID}(r_j) = \arg\min_{i \in \{1,\ldots,N\}} \min_{f \in F_i} \|c_j - f\|_2$$

- All areas belonging to the same ID are merged into a minimum bounding rectangle as the training annotation.

**Design Motivation**: Utilizing joint data from the RGB modality to automatically generate training annotations, bypassing the immense cost of manual labeling.

#### 2. **Inter-frame Footprint Association — UoE Metric**

**Function**: Associates pressure footprints of the same individual across frames. This is the core technical innovation of the paper.

The Union over Enclosure (UoE) is proposed to replace the traditional IoU as the similarity metric:

$$\text{UoE} = \frac{|A \cup B|}{|C|}$$

where $A$ and $B$ are the bounding boxes from the current and previous frames, $C$ represents the minimum enclosing rectangle of $A$ and $B$, and $|\cdot|$ denotes the area.

**Why not use IoU?** Pressure footprints exhibit sudden size mutations (e.g., sharp bounding box changes during transitions between single/double support phases), making IoU unreliable. UoE is based on a key observation: **the pressure footprints of different individuals do not overlap simultaneously**. Therefore, normalizing by the enclosing area is more stable than normalizing by the intersection area.

**Why not use motion prediction?** The movement of pressure footprints is jump-like and discontinuous (due to step alternation), making smooth motion prediction models like Kalman filters inapplicable. Thus, the UoE affinity matrix is calculated directly between adjacent frames, and the Hungarian algorithm is used to find current optimal matches.

Tracklet management strategy:
- Unmatched detections with low confidence are discarded; high-confidence detections initialize new tracklets (new subjects entering).
- Unmatched tracklets are temporarily marked as "lost" and preserved for a few frames to handle situations like brief jumps.
- Tracklets left unmatched for an extended duration are considered to have exited the scene.

#### 3. **Human Mesh Recovery Module**

**Function**: Recovers SMPL parameters from a temporal sequence of single-person pressure maps.

The architecture comprises three parts (Figure 5):

- **Image Encoder**: Uses a ResNet to encode the single-person pressure map ($128 \times 128$), which is concatenated with the bounding box center coordinates $c_{\text{bbox}}$ and mat corner coordinates $c_{\text{mat}}$ (acting as spatial priors) followed by dimension reduction.
- **Temporal Encoder**: A 2-layer Transformer encoder that captures temporal dependencies via positional encodings.
- **SMPL Regressor**: Performs an N-to-1 mapping (using average features) and employs an iterative error feedback (IEF) strategy to regress the SMPL parameters $(\boldsymbol{\theta}, \boldsymbol{\beta}, T)$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{SMPL}} + \mathcal{L}_{3D}$$

$$\mathcal{L}_{\text{SMPL}} = \lambda_\beta |\hat{\beta} - \beta| + \lambda_\theta |\hat{\theta} - \theta| + \lambda_T |\hat{T} - T|$$

$$\mathcal{L}_{3D} = \lambda_{3D} \|J_{3D} - \hat{J}_{3D}\|_2$$

Two dataset partitioning strategies are utilized for training:
- **Unseen Sequence Split**: An 80%/20% sequence-wise split, testing customization capabilities on unseen sequences of known subjects.
- **Unseen Subject Split**: Data from two volunteers are reserved exclusively for testing to evaluate the generalization capacity to entirely unseen individuals.

## Key Experimental Results

### Main Results

#### Footprint Tracking Performance

| Method | MOTA↑ | MOTP↑ | FN↓ | FP↓ | ID sw↓ | IDF1↑ |
|------|-------|-------|-----|-----|--------|-------|
| ByteTrack | 66.1% | 79.1% | 35207 | 22769 | 14453 | 4.8% |
| BoT-SORT | 82.5% | 87.7% | 17119 | 9020 | 10494 | 6.2% |
| **PressTrack** | **93.6%** | **94.8%** | **7317** | **6764** | **437** | **63.1%** |

PressTrack exhibits only 437 ID switches (over 86 tracklets, with an average length of 2660 frames, averaging roughly one switch every 523 frames), vastly outperforming the baselines.

#### End-to-End HMR Performance

| Method | Data Split | MPJPE↓ | PA-MPJPE↓ | PVE↓ | WA-MPJPE100↓ | GMPJPE↓ |
|------|---------|--------|-----------|------|-------------|---------|
| GT Dets. | Unseen Seq | 81.8 | 46.1 | 115.7 | 90.9 | 99.4 |
| GT Dets. | Unseen Subj | 92.2 | 43.1 | 132.9 | 100.8 | 112.8 |
| Tracked Dets. | Unseen Seq | 89.2 | 48.8 | 134.4 | 112.6 | 118.3 |
| Tracked Dets. | Unseen Subj | 96.8 | 44.3 | 145.3 | 115.0 | 125.0 |

Using tracked bounding boxes compared to ground-truth detection boxes incurs an increase of only around 7.4 mm in MPJPE, demonstrating that the impact of tracking error on HMR is well within acceptable limits. It generalizes robustly to unseen subjects (with an increase of only 7.6 mm).

### Ablation Study

| Sequence Length | GT Dets. MPJPE | Tracked Dets. MPJPE | Description |
|---------|---------------|---------------------|------|
| 1 | 94.2 | 96.2 | Single-frame only |
| 4 | 88.7 | 92.3 | Temporal aid |
| 8 | 86.6 | 90.7 | Consistent improvement |
| 12 | 83.8 | 89.9 | |
| 16 | **81.8** | **89.2** | Optimal length |
| 32 | 82.1 | 89.7 | Performance degrades if too long |

The optimal sequence length is 16 frames: shorter sequences suffer from insufficient temporal awareness, whereas excessively long sequences experience degradation due to decaying temporal correlation and accumulated tracking errors.

### Key Findings

1. **UoE significantly outperforms visual tracking methods**: Compared to ByteTrack and BoT-SORT, ID switching is reduced by 96.9%/95.8%.
2. **Temporal information is critical**: MPJPE decreases by 12.4 mm as the sequence length increases from 1 to 16.
3. **Tracking error is manageable**: The tracking process introduces only a modest extra error (7-8 mm MPJPE) to the end-to-end pipeline.
4. **Strong generalization ability**: The performance drop on unseen subjects is remarkably minimal (7.6 mm MPJPE).

## Highlights & Insights

1. **Filling an important gap**: This is the first work to achieve multi-person global HMR solely from pressure signals, pioneering a new paradigm for privacy-preserving motion analysis.
2. **In-depth problem analysis**: The detailed analysis of pressure footprint dynamics (e.g., scale mutation and discrete locomotion) directly guides the design of the UoE metric, rather than blindly applying visual tracking approaches.
3. **Dataset contribution**: The MIP dataset fills the gap for multi-person interactive pressure signals, offering spatial resolution ($1\text{cm} \times 1\text{cm}$) that outperforms existing tactile datasets.
4. **End-to-end feasibility validation**: The viability of the entire pipeline—from raw pressure data to multi-person 3D meshes and global trajectories—is thoroughly demonstrated.

## Limitations & Future Work

1. **Gap between GT and tracked bounding boxes**: A noticeable performance gap remains between the end-to-end pipeline and the isolated HMR module. Future work could analyze how different tracking errors (ID switching vs. localization jitter) affect HMR and devise targeted optimization strategies.
2. **Limited physical coverage of the pressure mat**: The dimensions of $1.2\text{m} \times 2.4\text{m}$ restrict real-world applicability.
3. **Reliance on multi-view RGB for annotation**: The dataset labels rely on EasyMocap processing multi-view videos, resulting in relatively high acquisition overhead.
4. **Unexplored complex interaction scenarios**: Highly interactive contexts such as crowded groups ($>3$ people) and direct physical contact (e.g., handshaking, hugging) remain unaddressed.

## Related Work & Insights

- **Comparison with Visual MOT**: The paper provides a thorough analysis of the intrinsic differences between pressure footprint tracking and visual tracking. The design of UoE reflects a problem-specific, tailored methodology.
- **Wide Application Prospects of Pressure Sensing**: Beyond gait scenarios, pressure sensing holds immense promise for bedridden patient monitoring (PID-HMR) and athletic motion analysis (VP-MoCap).
- **Trends in Privacy-Preserving Computing**: This work marks a significant step forward in building "privacy-first" sensory systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering work; multi-person pressure HMR, the UoE metric, and the MIP dataset are all proposed for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Both tracking and HMR are comprehensively evaluated, complete with ablation and generalization studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, solid dynamic analysis, and a well-described technical pipeline.
- **Value**: ⭐⭐⭐⭐ — Highly practical solution for multi-person motion analysis under privacy-constrained conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](../../ECCV2024/3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[CVPR 2026\] MetricHMSR: Metric Human Mesh and Scene Recovery from Monocular Images](../../CVPR2026/3d_vision/metrichmsr_metric_human_mesh_and_scene_recovery_from_monocular_images.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](../../CVPR2025/3d_vision/prompthmr_promptable_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
