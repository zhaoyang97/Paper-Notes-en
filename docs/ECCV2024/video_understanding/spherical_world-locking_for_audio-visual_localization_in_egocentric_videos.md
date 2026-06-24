---
title: >-
  [Paper Note] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos
description: >-
  [ECCV 2024][Video Understanding][Egocentric Video] This paper proposes the Spherical World-Locking (SWL) framework, which implicitly transforms multimodal perception streams into a world-locked spherical coordinate system to eliminate the challenges posed by self-motion, thereby achieving more precise audio-visual localization in egocentric videos.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Egocentric Video"
  - "Audio-Visual Localization"
  - "Spherical World-Locking"
  - "Multisensory Fusion"
  - "Transformer"
date: 2026-05-08
content_hash: 8d86fe83f2607c8d
---

# Spherical World-Locking for Audio-Visual Localization in Egocentric Videos

**Conference**: ECCV 2024  
**arXiv**: [2408.05364](https://arxiv.org/abs/2408.05364)  
**Code**: [Available](https://hs-yn.github.io/SWL/)  
**Area**: Audio & Speech  
**Keywords**: Egocentric Video, Audio-Visual Localization, Spherical World-Locking, Multisensory Fusion, Transformer

## TL;DR

This paper proposes the Spherical World-Locking (SWL) framework, which implicitly transforms multimodal perception streams into a world-locked spherical coordinate system to eliminate the challenges posed by self-motion, thereby achieving more precise audio-visual localization in egocentric videos.

## Background & Motivation

Egocentric videos provide comprehensive contextual information from a personal perspective, but their most prominent characteristic—**self-motion**—poses significant challenges:

**Coordinate Frame Shifts**: Head motion causes continuous changes in the target's relative position within the head-locked coordinate system.

**Field-of-View (FoV) Limitations**: A limited field-of-view combined with frequent motion leads to targets of interest frequently entering and exiting the frame.

**Motion Drift**: Accumulated motion makes temporal alignment challenging.

However, self-motion is also a crucial cue for scene understanding; humans can effectively stabilize perception and utilize head motion to enhance attention. Most existing works treat self-motion purely as a challenge, overlooking its potential as an behavioral proxy. This paper proposes relocating audio-visual streams to a global reference frame via **Spherical World-Locking (SWL)**, which not only eliminates the interference of self-motion but also leverages motion information to enhance scene understanding.

## Method

### Overall Architecture

The framework is built around two core concepts:

| Concept | Head-Locked | Spherical World-Locking (SWL) |
|------|------------------------|---------------------|
| Coordinate Frame | Head-centric 2D plane | World-centric 3D sphere |
| Self-motion Processing | Model must learn to compensate from raw data | Implicitly compensated via IMU measurements |
| Spatial Synchronization | Difficult spatial alignment across different modalities | Natural multimodal spatial alignment |
| Computational Overhead | No extra overhead | Negligible extra overhead |

Two SWL implementations are proposed:
- **Explicit SWL**: Maps videos to a 360° panorama.
- **Implicit SWL** (recommended): Retains original inputs and decouples semantic and positional information via spherical positional embeddings.

### Key Designs

**Implicit Spherical World-Locking**: Instead of directly transforming the input data, spherical positional coordinates are assigned to each patch/token, maintaining the decoupling of semantic and positional embeddings.

**Multi-Classification Tokens (Multi-CLS)**: Multiple classification tokens are deployed, each parameterized as a point on the sphere $c_i = \mathbf{W}_c p_i + \mathbf{b}_c$ to capture semantic information around that location. A sparse $5 \times 10$ grid is used during training.

**MuST Encoder** (Multisensory Spherical World-Locked Transformer) incorporates two key innovations:

1. **Spherical Spatial Similarity**: Computes pairwise spatial relations between multimodal embeddings based on rotation quaternions:
$$\mathbf{P}_{ij}^l = \text{Linear}(\text{GELU}(\text{Linear}([1+p_i \cdot p_j, p_i \times p_j])))$$
The rotation quaternions only need to be computed once and can be reused across all layers.

2. **Modality-Aware Operations (M-ops)**: Applies independent LayerNorm and q/k/v projections to each modality (M-LN + M-Attn) to facilitate cross-modal interaction while preserving modality-specific characteristics.

**MuST Decoder**: Supports three flexible decoding strategies:
- Sparse Decoding: Uses an MLP to predict for each CLS Token.
- Dense Decoding: Generates a dense output map using a lightweight deconvolutional network.
- Horizon Decoding: Employs only the tokens near the equator, leveraging the gravity-alignment property of SWL.

### Loss & Training

- Trained using Binary Cross-Entropy (BCE) loss.
- Adam optimizer with a learning rate of 1e-4 and no learning rate scheduler.
- End-to-end training for 10 epochs.
- The model architecture is based on DeiT-S (a small ViT variant), with slightly fewer parameters than ResNet-50.

## Key Experimental Results

### Main Results

**EasyCom Dataset: Audio-Visual Active Speaker Localization**:

| Method | mAP↑ |
|------|------|
| DOA (Signal Processing Method) | 52.62 |
| TalkNet | 69.13 |
| AVLN | 85.11 |
| MAVASL_C+E | 86.32 |
| **MuST** | **89.88** |
| Oracle (Upper Bound) | 91.03 |

MuST outperforms the previous state-of-the-art by 3.6 percentage points, approaching the Oracle upper bound (which uses near-field microphones) with only a 1.2%p gap.

### Ablation Study

**Ablation of Encoder Components (EasyCom)**:

| Variant | mAP↑ |
|------|------|
| MuST w/o pose (No pose information) | 87.76 |
| MuST w/o rotation (No rotation information) | 88.83 |
| MuST w/o M-ops (No modality-aware operations) | 88.53 |
| MuST M-LN only | 89.67 |
| MuST M-LN+M-Attn (Full version) | 89.88 |

**Analysis of Modality Contributions**:

| Input Modality Combination | mAP↑ |
|-------------|------|
| Pose only | 47.95 |
| Pose + Monaural Audio | 68.57 |
| Pose + Visual | 68.78 |
| Pose + Multichannel Audio + Visual | 89.88 |

### Key Findings

1. **Positional and Rotational Information of SWL Provide Significant Contributions**: Leading to a +2.1%p performance improvement.
2. **Multichannel Microphone Array is the Largest Driver of Performance**: From monaural to multichannel, the mAP leaps from 73.47 to 89.88.
3. **Modality-Aware Operations are Crucial**: Performance drops below the visual-free counterpart when they are removed.
4. **Superior Performance in Spherical Audio Localization**: On the RLR-CHAT dataset, the Mean Absolute Error (MAE) drops from 44.90° (second best) to 12.67°.
5. **Efficiency of Sparse Decoding**: Using only the equatorial tokens (a 5x reduction in quantities) leads to a performance drop of only about 1°.

## Highlights & Insights

- **Paradigm Shift**: Transforms self-motion from "noise requiring learned compensation" to a "directly exploitable valuable signal," achieved via IMU with near-zero overhead.
- **Unified Multimodal Representation**: Audio, visual, and behavioral information are naturally aligned on the world-locked sphere, eliminating the need for complex coordinate transformations.
- **Clever Application of Rotation Quaternions**: Spaces between two points on a sphere are encoded using rotation quaternions, which is both mathematically consistent and computationally efficient.
- **Flexible Decoder Design**: Multi-CLS combined with various decoding strategies (sparse, dense, horizon) flexibly adapts to different task requirements.

## Limitations & Future Work

1. Only rotation is considered, ignoring translation, which limits applicability in translation-heavy mobile scenarios.
2. Short temporal windows (<1 second) are utilized, leaving long-term dependencies unmodeled.
3. Reliance on IMU data restricts deployment on devices without IMU sensors.
4. Explicit SWL suffers from distortion issues during panoramic projection.
5. Extending SWL to other egocentric video tasks (such as action recognition and object interaction) warrants exploration.

## Related Work & Insights

- Unlike traditional 360° video spherical representations, SWL handles the implicit representation of planar videos on a sphere.
- The Multi-CLS Token design can inspire other vision tasks requiring spatialized predictions.
- Modality-aware operations (M-LN, M-Attn) offer effective engineering practices for multimodal Transformer configurations.

## Rating

| Dimension | Rating (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.5 | The concept of spherical world-locking is novel, with an excellent idea of transforming self-motion into useful signals. |
| Technical Depth | 4.5 | The designs of quaternion spherical encoding and modality-aware operations are rigorous and elegant. |
| Experimental Thoroughness | 4.5 | Three different tasks/datasets along with rich ablations fully validate the effectiveness and generalizability of the proposed method. |
| Practicality | 4 | Relies on IMU data, though modern AR/VR devices are commonly equipped with them. |
| Overall | 4.5 | A paradigm-shifting work in the field of egocentric video understanding. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[CVPR 2026\] Ego-Grounding for Personalized Question-Answering in Egocentric Videos](../../CVPR2026/video_understanding/ego-grounding_for_personalized_question-answering_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
