---
title: >-
  [Paper Note] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition
description: >-
  [CVPR 2026][Human Understanding][Micro gesture recognition] This paper introduces OMG-Bench, the first large-scale public benchmark for skeleton-based online micro hand gesture recognition (40 classes, 13,948 instances), and proposes the HMATr framework, which unifies detection and classification end-to-end via a hierarchical memory bank and position-aware queries, achieving a 7.6% improvement in detection rate over the previous state of the art.
tags:
  - CVPR 2026
  - Human Understanding
  - Micro gesture recognition
  - online gesture recognition
  - skeleton data
  - hierarchical memory
  - VR/AR interaction
date: 2026-05-08
content_hash: 04f5e7180c8201a5
---

# OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition

**Conference**: CVPR 2026
**arXiv**: [2512.16727](https://arxiv.org/abs/2512.16727)
**Code**: [Project Page](https://omg-bench.github.io/)
**Area**: Human Understanding / Gesture Recognition
**Keywords**: Micro gesture recognition, online gesture recognition, skeleton data, hierarchical memory, VR/AR interaction

## TL;DR

This paper introduces OMG-Bench, the first large-scale public benchmark for skeleton-based online micro hand gesture recognition (40 classes, 13,948 instances), and proposes the HMATr framework, which unifies detection and classification end-to-end via a hierarchical memory bank and position-aware queries, achieving a 7.6% improvement in detection rate over the previous state of the art.

## Background & Motivation

**State of the Field**: With the maturation of hand pose estimation on head-mounted displays such as Meta Quest and PICO, skeleton-based gesture recognition has become increasingly important for VR/AR interaction. Existing research primarily focuses on macro gestures (large-amplitude movements), employing sliding-window classification or two-stage detection-classification pipelines.

**Limitations of Prior Work**: (1) **Dataset issues**: Existing datasets are small in scale (e.g., SHREC'22 contains only 288 sequences and 1,152 instances), rely on outdated monocular hand pose estimators yielding poor skeleton quality, and exhibit insufficient temporal dynamics (with conspicuous pauses between gestures that do not reflect real continuous interaction). (2) **Absence of micro gesture resources**: Prolonged use of macro gestures causes arm muscle fatigue; micro gestures are better suited for extended interaction, yet no public skeleton-based micro gesture dataset exists. (3) **Algorithmic limitations**: Two-stage methods cannot be optimized end-to-end; CTC-based methods tend to produce blank predictions for weak signals; sliding windows are sensitive to hyperparameters, with non-overlapping windows risking gesture truncation and overlapping windows incurring redundant computation.

**Root Cause**: Three fundamental challenges of micro gestures—subtle inter-class differences (e.g., thumb tip touching index fingertip vs. middle fingertip), rapid dynamics (average duration of only 0.57 seconds), and large variation in temporal length—render existing methods inadequate.

**Paper Goals**: (1) Construct a high-quality, large-scale micro gesture benchmark dataset; (2) design an end-to-end framework that unifies detection and classification, addressing window truncation and insufficient cross-window context.

**Starting Point**: On the data side, multi-view self-supervised hand pose estimation combined with a semi-automatic annotation pipeline ensures quality and scale. On the algorithm side, borrowing the query mechanism from object detection, learnable queries are used to unify detection and classification.

**Core Idea**: Establish the first online micro gesture benchmark and propose a Hierarchical Memory-Augmented Transformer that maintains cross-window contextual continuity via frame-level and window-level memory banks, with position-aware queries implicitly encoding spatiotemporal gesture information.

## Method

### Overall Architecture

HMATr is an end-to-end streaming recognition framework consisting of three components: (1) a lightweight ST-GCN skeleton feature extractor that extracts frame-level features from non-overlapping sliding windows of skeleton sequences; (2) a hierarchical memory bank containing frame-level memory (storing fine-grained spatiotemporal details) and window-level memory (storing high-level semantic abstractions) to supplement the current window with historical context; and (3) position-aware learnable queries that implicitly capture gesture location and semantics through interaction with the memory banks and current frame features, jointly performing detection and classification.

Input skeleton stream → Non-overlapping sliding window segmentation → ST-GCN frame-level feature extraction → Interaction with frame-level/window-level memory → Position-aware query decoding → Detection head + classification head → Output gesture category and temporal location.

### Key Designs

1. **Hierarchical Memory Bank**:

    - **Function**: Provides cross-window historical context for non-overlapping sliding windows, compensating for information loss caused by window truncation.
    - **Mechanism**: Two fixed-length FIFO queues are maintained. Frame-level memory $\mathcal{M}_f \in \mathbb{R}^{B \times L_f \times C}$ stores frame-level skeleton features from recent historical windows, preserving fine-grained spatiotemporal details (e.g., finger motion trajectories). Window-level memory $\mathcal{M}_w \in \mathbb{R}^{B \times L_w \cdot N \times C}$ stores high-level semantic abstractions of historical query features. New features are appended and the oldest are discarded at each window update.
    - **Design Motivation**: A single-granularity memory cannot simultaneously satisfy the need for fine-grained action details and high-level semantic understanding. Frame-level memory provides raw evidence of "what happened," while window-level memory provides an abstract representation of "what it means." The two work in concert to accommodate the variability in micro gesture duration.

2. **Position-aware Queries**:

    - **Function**: Serve as the unified carrier for detection and classification, implicitly encoding the temporal location and semantic information of gestures.
    - **Mechanism**: Learnable queries are initialized with historical priors from the window-level memory—$\mathcal{M}_w^t$ is averaged along the temporal dimension to obtain a global memory query $Q_m^t$, which is added to the current window queries. The queries then interact with the frame-level memory (to obtain fine-grained motion information) and with the current window frame features (to obtain current observations). Detection and classification heads respectively predict gesture location and category.
    - **Design Motivation**: Traditional two-stage methods decouple detection and classification, precluding end-to-end optimization and making detection accuracy a bottleneck. Because gesture onset timing and category are strongly correlated—localization requires category cues to distinguish intentional gestures from unintentional motion, while classification depends on precise temporal localization to capture key action segments—joint encoding via unified queries is more principled.

3. **Query-CTC Loss**:

    - **Function**: Handles brief and weak micro gesture signals, reducing the tendency of the network to predict blank tokens.
    - **Mechanism**: In addition to the standard bipartite matching loss (classification cross-entropy + positional L1/IoU loss), a query-based CTC loss is incorporated to enforce temporal alignment between predicted gestures and ground truth.
    - **Design Motivation**: Pure CTC methods are prone to producing all-blank predictions for the weak signals of micro gestures. By combining the CTC loss with the query mechanism, the semantic information encoded in the queries assists temporal alignment, improving recognition of rapid successive micro gestures.

### Loss & Training

The total loss is $\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{pos} \mathcal{L}_{pos} + \lambda_{q-CTC} \mathcal{L}_{q-CTC}$, where $\lambda_{cls}=2$, $\lambda_{pos}=5$, $\lambda_{q-CTC}=0.2$. Classification loss uses cross-entropy; positional loss combines L1 and IoU; unmatched queries are assigned to the background class. The Adam optimizer is used with batch size 64, learning rate 0.001, and weight decay 0.0004.

## Key Experimental Results

### Main Results

| Method | Type | DR↑ | FP↓ | JI↑ | NLD↑ | Inference (ms)↓ | Avg. Latency (frames)↓ |
|--------|------|-----|-----|-----|------|-----------------|------------------------|
| **HMATr (Ours)** | End-to-end | **89.2%** | **0.22** | **0.71** | **0.77** | **1.61** | **7.67** |
| HiOD | Sliding-window offline | 81.2% | 0.29 | 0.66 | 0.70 | 72.44 | 8.66 |
| Bound.Reg. | Boundary supervision | 81.6% | 0.37 | 0.59 | 0.61 | 22.64 | 8.82 |
| AG-MAE | Boundary supervision | 80.7% | 0.28 | 0.65 | 0.72 | 2.36 | 8.25 |
| BlockGCN | Sliding-window offline | 78.3% | 0.32 | 0.66 | 0.65 | 7.83 | 7.89 |

Generalization on SHREC'22:

| Method | DR↑ | FP↓ | JI↑ |
|--------|-----|-----|-----|
| HMATr | **0.85** | **0.08** | **0.79** |
| AG-MAE | 0.82 | 0.13 | 0.74 |
| DSTA | 0.77 | 0.11 | 0.71 |

### Ablation Study

| Configuration | DR↑ | FP↓ | JI↑ | NLD↑ |
|---------------|-----|-----|-----|------|
| Full HMATr | **89.2%** | **0.22** | **0.71** | **0.77** |
| w/o frame-level memory | 85.7% | 0.26 | 0.67 | 0.73 |
| w/o window-level memory | 86.1% | 0.25 | 0.68 | 0.74 |
| w/o hierarchical memory | 83.4% | 0.30 | 0.63 | 0.70 |
| w/o query-CTC loss | 87.3% | 0.24 | 0.69 | 0.75 |

### Key Findings
- HMATr outperforms all baselines across all four metrics, achieving a 7.6% improvement in detection rate over Bound.Reg., while also attaining the fastest inference speed (1.61 ms) and lowest latency (7.67 frames).
- The hierarchical memory contributes substantially: removing both memory levels reduces DR by 5.8%, demonstrating that cross-window context is critical for micro gesture recognition.
- Frame-level and window-level memory each contribute independently, validating the necessity of multi-granularity design.
- Non-overlapping windows combined with the memory mechanism eliminate redundant computation from overlapping windows while preserving contextual continuity.

## Dataset Highlights
- **OMG-Bench scale**: 40 micro gesture classes, 18 subjects, 1,272 sequences, and 13,948 instances—far exceeding the previously largest dataset SHREC'22 (16 classes, 288 sequences, 1,152 instances).
- **Data quality**: Multi-view self-supervised hand pose estimation achieves a mean joint position error of only 2.78 mm, significantly outperforming commercial monocular solutions.
- **Challenge statistics**: Average gesture duration is only 0.57 seconds; consecutive same-class gestures account for 27.60% of instances (vs. 0.09% in SHREC'22); normalized joint displacement is only 8.95 (vs. 128.73 in SHREC'22), faithfully reflecting the "micro" nature of the gestures.

## Highlights & Insights
- **Multi-view self-supervised + semi-automatic annotation pipeline**: Addresses the pain points of poor skeleton quality and annotation difficulty in micro gesture dataset construction; the pipeline is broadly applicable.
- **Memory mechanism as a replacement for overlapping windows**: Eliminates redundant computation while preserving cross-window context, achieving a better balance between efficiency and performance.
- **Unified detection-classification**: Exploits the strong correlation between gesture temporal location and category, yielding a more elegant and effective solution than two-stage approaches.

## Limitations & Future Work
- The current framework supports only single-hand micro gestures; bimanual interaction and hand-object interaction scenarios are not covered.
- The 40 gesture classes are primarily thumb-to-finger contact gestures; palm gestures and spatial trajectory gestures are not included.
- Memory bank lengths ($L_f=16$, $L_w=3$) are fixed and may require adaptive adjustment for interaction scenarios with different tempos.
- Although the dataset substantially surpasses prior work in scale, the diversity of 18 subjects remains limited.

## Related Work & Insights
- **vs. SHREC'22**: OMG-Bench comprehensively surpasses SHREC'22 in scale (12×), quality (multi-view), and challenge level (micro gestures).
- **vs. STMG**: The first skeleton-based micro gesture method, using only 7 classes involving the thumb and index finger. The present work employs full-hand 21-joint skeletons with 40 classes, posing a substantially greater challenge.
- **vs. OO-dMVMT**: A boundary supervision method; HMATr exceeds it by 8.1% in DR, attributed to the unified query mechanism and hierarchical memory.
- The dataset construction pipeline is transferable to other fine-grained action recognition scenarios, such as surgical operation recognition and sign language recognition.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Both the dataset and method are innovative, filling an important gap in micro gesture recognition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comparisons against multiple state-of-the-art methods, ablation studies, cross-dataset generalization, and efficiency analysis are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed dataset statistics and well-motivated method design.
- **Value**: ⭐⭐⭐⭐ High benchmark contribution value; the dataset will advance research on micro gesture interaction in VR/AR.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] Miburi: Towards Expressive Interactive Gesture Synthesis](miburi_towards_expressive_interactive_gesture_synthesis.md)
- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)

<!-- RELATED:END -->
