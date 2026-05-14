---
title: >-
  [Paper Note] MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network
description: >-
  [AAAI 2026][Human Understanding][Glass surface detection] Grounded in the physical observation that objects in reflection/transmission layers move at different velocities than those in non-glass regions…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "Glass surface detection"
  - "video segmentation"
  - "optical flow motion cues"
  - "cross-modal fusion"
  - "temporal attention"
date: 2026-05-08
content_hash: 265a5535ca590e36
---

# MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network

**Conference**: AAAI 2026
**arXiv**: [2601.13715](https://arxiv.org/abs/2601.13715)
**Code**: [github](https://github.com/YT3DVision/MVGDNet)
**Area**: Human Understanding
**Keywords**: Glass surface detection, video segmentation, optical flow motion cues, cross-modal fusion, temporal attention

## TL;DR

Grounded in the physical observation that objects in reflection/transmission layers move at different velocities than those in non-glass regions, this paper proposes MVGD-Net, which leverages optical flow motion cues to guide glass surface detection in videos. The framework comprises four core modules: Cross-scale Multimodal Fusion Module (CMFM), History-Guided Attention Module (HGAM), Temporal Cross-Attention Module (TCAM), and Temporal-Spatial Decoder (TSD). A large-scale dataset, MVGD-D, containing 312 videos and 19,268 frames is also introduced.

## Background & Motivation

Glass surfaces are ubiquitous in daily environments (windows, glass walls, glass doors), and their transparent, colorless nature poses significant challenges for computer vision systems, particularly for robotic/UAV navigation, depth estimation, and 3D reconstruction.

**Limitations of Prior Work**:

**Single-image methods**: Prior works have explored contrastive contextual features, boundary cues, reflection phenomena, ghosting effects, semantic relations, and visual blur priors, but cannot exploit temporal information available in videos.

**Multimodal methods**: RGB-D, polarization, RGB-thermal, and RGB-NIR approaches require additional sensors and still operate on single frames.

**The first video-based method, VGSD-Net** (AAAI 2024): Uses reflection information to assist detection, but reflection extraction is unreliable in complex scenes due to the absence of reflection GT supervision, leading to under- or over-detection.

**Key physical observation (core insight)**: Objects in reflection/transmission layers are farther from the glass surface; thus, under camera motion, they move **more slowly** than objects in non-glass regions at the same spatial plane. This **motion inconsistency** effectively reveals the presence of glass surfaces, even in indoor scenes with weak reflections (transmitted-layer objects at greater depth also exhibit inconsistent motion). This finding aligns with neuroscience research showing that humans rely on dynamic perceptual cues to identify glass regions.

**Optical flow as motion cue carrier**: Optical flow maps estimated via RAFT effectively encode motion inconsistency information, indicating the potential locations of glass surfaces.

## Method

### Overall Architecture

MVGD-Net takes three consecutive frames ($I_{N-2}$, $I_{N-1}$, $I_N$) as input. The processing pipeline is:

1. RAFT estimates inter-frame optical flows $f_{N-1}$, $f_N$
2. Swin Transformer backbone extracts multi-scale RGB features
3. A preliminary glass mask $P_{N-1}$ is generated to filter motion inconsistencies in non-glass regions
4. A second Swin Transformer extracts optical flow features
5. CMFM fuses RGB and optical flow features → spatial features
6. TCAM + HGAM aggregate inter-frame temporal information → temporal features
7. TSD fuses spatial and temporal features → output glass region mask

### Key Designs

#### 1. **Cross-scale Multimodal Fusion Module (CMFM)**: Deep fusion of RGB and optical flow

CMFM is designed to fuse motion cues from optical flow with RGB features at multiple scales. It adopts a U-shaped recurrent structure, completing the fusion of all 8 feature maps via 7 cross-scale cross-attention blocks.

**Left→Right (downsampling compression)**: Feature maps are progressively downsampled to extract more effective spatial representations. The attention mechanism is:
$$Att_{i+1,i}^{top} = \text{SoftMax}(X_{i+1}^Q \otimes X_i^K)$$
$$Y_i = Att_{i+1,i}^{top} \otimes X_i^V$$

**Right→Left (upsampling enhancement)**: Features are progressively upsampled to gradually enhance salient representations.

**Final fusion**: Feature pairs at the same scale are fused via element-wise multiplication:
$$S_i = \begin{cases} F_7 & i=1 \\ F_{i-1} \odot F_{8-i} & i=2,3,4 \end{cases}$$

Input features are first refined by a CBAM attention module, then dimensionality-reduced to $C_1=128$ via $1\times1$ convolution, balancing quality and efficiency.

#### 2. **History-Guided Attention Module (HGAM)**: Leveraging historical frames to enhance current predictions

**Core Idea**: Predictions for frame $N$ can be enhanced using information from the preceding two frames, as glass surface locations are approximately consistent across adjacent frames.

The current frame feature $G_i^N$ serves as the Query, while Keys and Values from the previous two frames are fused via element-wise multiplication and concatenated:
$$\tilde{K}_i^N = [W_K(G_i^{N-2}) \odot W_K(G_i^{N-1}), W_K(G_i^N)]$$

Temporal output features are then generated via self-attention:
$$T_i^N = \text{SelfAttn}(Q_i^N, \tilde{K}_i^N, \tilde{V}_i^N)$$

HGAM specifically designs multiplicative interaction of historical frames to capture stable glass region patterns across frames.

#### 3. **Temporal Cross-Attention Module (TCAM)**: Inter-frame dependency modeling

TCAM captures inter-frame dependencies using standard cross-attention, divided into two groups:
- $T_i^{N-1} = \text{TCAM}(G_i^{N-1}, G_i^{N-2})$: short-term temporal dependencies and motion trends
- $T_i^{N-2} = \text{TCAM}(G_i^{N-2}, G_i^N)$: long-range temporal consistency

#### 4. **Temporal-Spatial Decoder (TSD)**: Balanced fusion of temporal and spatial features

Temporal feature channels are configured as $\{2^{i-1}C_1\}_{i=1}^4$ (non-uniform), while spatial features are uniformly set to $C_1$. TSD addresses this channel inconsistency.

**Mutual weight enhancement**:
$$F_i^t = \text{SA}(\text{CA}(T_i) \odot \text{Sigmoid}(M(S_i)) + \text{CA}(T_i))$$

**Simple gating balance**: Inspired by NAFNet, concatenated features are evenly split along the channel dimension into two halves, and element-wise multiplication produces the gated output:
$$F_i^g = F_{concat}^{[:C/2]} \odot F_{concat}^{[C/2:C]}$$

### Loss & Training

Total loss = preliminary mask loss + three-frame prediction loss:
$$\mathcal{L} = \alpha \mathcal{L}_P + \mathcal{L}_M$$

where $\alpha = 1/8$ is used for balancing; each loss term is a combination of BCE and IoU loss. The preliminary mask $P_{N-1}$ is used to filter motion inconsistency cues from non-glass regions in optical flow maps.

Training is conducted on an NVIDIA RTX 4090; images are resized to $384\times384$. No data augmentation is applied to preserve temporal consistency, and optical flow maps are uniformly generated using RAFT.

## Key Experimental Results

### Main Results

| Method | Type | VGSD-D IoU↑ | VGSD-D MAE↓ | MVGD-D IoU↑ | MVGD-D MAE↓ | MVGD-D ACC↑ |
|------|------|------------|------------|-------------|-------------|-------------|
| MINet | SOD | 71.84 | 0.162 | 71.29 | 0.152 | 0.885 |
| SAM2 | SS | 78.60 | 0.131 | 78.18 | 0.121 | 0.841 |
| GhostingNet | GSD | 80.40 | 0.100 | 80.01 | 0.104 | 0.915 |
| VGSDNet | VGSD | 80.72 | 0.099 | 77.27 | 0.126 | 0.904 |
| MG-VMD | VMD | 76.56 | 0.125 | 73.69 | 0.134 | 0.887 |
| **Ours** | **VGSD** | **86.57** | **0.064** | **82.62** | **0.090** | **0.930** |

On VGSD-D, compared to the second-best VGSDNet: IoU +7.20%, MAE −35.35%, BER −36.45%.
On MVGD-D, compared to the second-best GhostingNet: IoU +3.26%, MAE −13.46%, BER −7.45%.

### Ablation Study

| Model | Config | IoU↑ | F_β↑ | MAE↓ | BER↓ | ACC↑ |
|------|------|------|------|------|------|------|
| A | BS + BD (backbone only) | 74.31 | 80.87 | 0.140 | 0.135 | 0.905 |
| B | A + RAFT + BF (with optical flow) | 75.59 | 82.12 | 0.136 | 0.131 | 0.908 |
| C | BS + CMFM + BT + TSD (w/o TAM) | 79.80 | 86.33 | 0.109 | 0.107 | 0.915 |
| D | BS + BF + TAM + TSD (w/o CMFM) | 78.74 | 85.24 | 0.117 | 0.112 | 0.915 |
| E | BS + CMFM + TAM + BD (w/o TSD) | 80.08 | 86.58 | 0.104 | 0.101 | 0.922 |
| F | w/o preliminary mask P | 80.36 | 86.93 | 0.107 | 0.098 | 0.922 |
| **G** | **Full model** | **82.62** | **89.14** | **0.090** | **0.087** | **0.930** |

- B vs. A: motion cues are effective (IoU +1.28%)
- D vs. G: CMFM contributes substantially (IoU +3.88%)
- F vs. G: preliminary mask filtering is critical (IoU −2.26%)

### Key Findings

1. **Motion inconsistency is an effective glass detection cue**: Performance improves upon introducing optical flow, but dedicated modules are required to fully exploit it.
2. **CMFM is the most critical module**: Cross-scale multimodal fusion substantially outperforms naive feature concatenation.
3. **Preliminary mask filtering is indispensable**: Motion inconsistencies in non-glass regions (e.g., railings) introduce erroneous signals.
4. **Semantic models such as SAM2 are limited**: They are misled by semantic information from the transmission layer behind the glass.
5. **Effective even in indoor scenes with weak reflections**: Motion inconsistency arising from depth differences in the transmission layer remains exploitable.

## Highlights & Insights

- **Physically-intuitive method design**: Deriving motion inconsistency cues from the depth difference of reflection/transmission layers is more principled than purely data-driven approaches.
- **Complete modular design**: Each module has a clearly defined role—CMFM handles spatial fusion, HGAM/TCAM handle temporal aggregation, and TSD handles balanced decoding.
- **Dataset contribution**: The proposed MVGD-D surpasses the existing VGSD-D in scene diversity, glass location distribution, and color contrast, making it a valuable community resource.
- **Successful application of NAFNet's simple gating**: Channel splitting combined with element-wise multiplication elegantly resolves the spatial-temporal feature imbalance.

## Limitations & Future Work

1. **Only three input frames**: Long-term temporal dependencies cannot be captured, leading to inter-frame detection inconsistencies (e.g., missed detection in frame 4 of a region consistently detected in prior frames).
2. **Relatively slow inference speed**: 190.9 ms/frame, compared to GhostingNet (32.9 ms) and VGSDNet (72.4 ms).
3. **False detections at open doors/windows**: As with existing image-based methods, glass-like regions enclosed by door or window frames may be falsely detected as glass.
4. Limited robustness to abrupt camera motion—optical flow estimation may be inaccurate under large displacements.
5. The dataset covers only static scenes with dynamic camera motion, without addressing scenarios involving moving objects.

## Related Work & Insights

- **VGSDNet** (AAAI 2024): The first video glass detection method and the primary baseline for this work.
- **GhostingNet** (TPAMI 2024): A ghosting-cue-based glass detection method sharing the Swin backbone with this work.
- **Warren et al.** (CVPR 2024): Uses motion inconsistency to detect mirrors; this paper extends the idea to glass surfaces and handles the special case of open doors and windows.
- **RAFT**: The foundational tool for optical flow estimation.
- Insight: **Physical priors (motion inconsistency) + multimodal fusion (RGB + optical flow) + temporal reasoning** constitute an effective paradigm for transparent object detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The motion inconsistency perspective is distinctive, and the module design is well-justified.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comparisons against 11 methods, thorough ablations, and comprehensive dataset analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical motivation is articulated clearly, with persuasive illustrations.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to robotic and autonomous driving scenarios, though inference speed requires optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](../../CVPR2026/human_understanding/unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[AAAI 2026\] AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](ahan_asymmetric_hierarchical_attention_network_for_identical.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](../../ICCV2025/human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[AAAI 2026\] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)

</div>

<!-- RELATED:END -->
