---
title: >-
  [Paper Note] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals
description: >-
  [NeurIPS 2025][Human Understanding][EMG signals] This paper proposes the CPEP framework, which employs contrastive learning to align low-quality EMG signal representations with high-quality hand pose representations, endowing the EMG encoder with pose-awareness. CPEP is the first to achieve zero-shot recognition of unseen gestures from EMG signals, yielding a 21% improvement on in-distribution gesture classification and a 72% improvement on unseen gesture classification.
tags:
  - NeurIPS 2025
  - Human Understanding
  - EMG signals
  - EMG
  - gesture recognition
  - contrastive learning
  - zero-shot classification
  - cross-modal alignment
date: 2026-05-08
content_hash: 491466e959f5eddc
---

# CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals

**Conference**: NeurIPS 2025
**arXiv**: [2509.04699](https://arxiv.org/abs/2509.04699)
**Code**: None
**Area**: Human Understanding / Gesture Recognition
**Keywords**: EMG signals, EMG, gesture recognition, contrastive learning, zero-shot classification, cross-modal alignment

## TL;DR
This paper proposes the CPEP framework, which employs contrastive learning to align low-quality EMG signal representations with high-quality hand pose representations, endowing the EMG encoder with pose-awareness. CPEP is the first to achieve zero-shot recognition of unseen gestures from EMG signals, yielding a 21% improvement on in-distribution gesture classification and a 72% improvement on unseen gesture classification.

## Background & Motivation

**Background**: Vision-based gesture recognition has matured considerably, yet remains constrained by power consumption and privacy concerns in wearable device deployments. Surface EMG (sEMG) signals are low-power and easy to integrate, making them well-suited for real-time gesture prediction on wearable platforms.

**Limitations of Prior Work**: (a) EMG signals exhibit low signal-to-noise ratios and high variability, limiting the effectiveness of conventional self-supervised pre-training; (b) supervised methods (e.g., emg2pose for pose regression) generalize poorly and cannot recognize unseen gestures or adapt to new users; (c) large-scale EMG data collection is costly and difficult.

**Key Challenge**: EMG is a "weak modality" from which high-quality representations are difficult to learn in isolation, whereas hand pose is a "strong modality" that encodes rich structural and semantic information. The core challenge is how to leverage the strong modality's prior knowledge to improve representations of the weak modality.

**Goal**: To enable the EMG encoder to learn pose-aware representations, supporting zero-shot gesture classification via embedding-space retrieval against pose references.

**Key Insight**: Inspired by CLIP-style cross-modal contrastive pre-training, with design adaptations specific to the EMG–pose setting—pre-training unimodal encoders to reduce paired data requirements, and freezing the strong modality encoder while training only the weak modality encoder.

**Core Idea**: Contrastive learning is used to pull EMG representations toward paired pose representations, enabling zero-shot gesture recognition without task-specific fine-tuning.

## Method

### Overall Architecture
CPEP consists of three stages: (1) MAE-based self-supervised pre-training of both EMG and pose encoders; (2) contrastive pre-training with the pose encoder frozen, aligning the [CLS] representations of both modalities via InfoNCE; (3) downstream evaluation via linear probing or zero-shot nearest-neighbor retrieval.

### Key Designs

1. **Unimodal Encoder Pre-training (MAE)**:

    - Function: Pre-trains separate Transformer encoders for EMG and pose modalities independently.
    - Mechanism: Standard MAE with temporal patching, mask ratio $r=50\%$; only unmasked tokens are encoded, and the decoder reconstructs the full sequence. $\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \|\psi(\phi(\{\mathbf{z}_j\}_{j\notin\mathcal{M}}))_i - \mathbf{z}_i\|_2^2$
    - Design Motivation: Learning robust unimodal representations prior to contrastive alignment reduces the paired data requirements during the contrastive stage.

2. **Contrastive Pose-EMG Pre-training (CPEP)**:

    - Function: Freezes the pose encoder $\mathcal{E}_p$ while training the EMG encoder together with a projection head $h$.
    - Mechanism: EMG embeddings are computed as $u_i = h(\mathcal{E}_x(x_i))_{[\text{CLS}]}$ and pose embeddings as $v_i = (\mathcal{E}_p(p_i))_{[\text{CLS}]}$. After $\ell_2$ normalization, symmetric InfoNCE is applied: $\mathcal{L}_{\text{CPEP}} = \frac{1}{2N}\sum_{i} [-\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})} - \log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ji})}]$, where $s_{ij} = \tilde{u}_i^\top\tilde{v}_j / \tau$.
    - Design Motivation: Freezing the pose encoder is critical—jointly updating both encoders degrades pose representation quality and leads to training divergence, as confirmed by ablation experiments.

3. **Zero-Shot Classification Protocol**:

    - Function: $k$-nearest neighbor voting in the embedding space.
    - Mechanism: Pose embeddings are pre-computed offline; for each EMG query, the top-$k$ ($k=10$) nearest pose embeddings are retrieved and a majority vote determines the predicted label: $\hat{y}_j = \text{mode}\{y(p) | p \in \mathcal{R}_j\}$.
    - Design Motivation: Zero-shot performance validates that the EMG representations have internalized the structural information of hand poses.

### Loss & Training

Three-stage training pipeline: EMG/Pose-MAE pre-training for 100 epochs each → CPEP contrastive pre-training for 100 epochs (batch size 256, learnable temperature $\tau$ initialized at 0.02) → linear probing. Training is conducted on 4× V100 GPUs and takes approximately 4.5 hours per model.

## Key Experimental Results

### Main Results (Gesture Classification Accuracy)

| Method | LP In-Dist. | LP Unseen | ZS In-Dist. | ZS Unseen |
|--------|-------------|-----------|-------------|-----------|
| emg2pose (baseline) | 0.647 | 0.312 | - | - |
| EMG-MAE | ~0.55 | ~0.30 | - | - |
| PoseT (supervised) | ~0.60 | ~0.35 | - | - |
| **CPEP** | **0.782** | **0.536** | **0.757** | **0.481** |
| Pose-MAE (upper bound) | ~0.85 | ~0.65 | - | - |

### Ablation Study

| Configuration | LP In-Dist. | ZS In-Dist. | LP Unseen | ZS Unseen |
|---------------|-------------|-------------|-----------|-----------|
| EMG encoder Frozen | 0.372 | 0.344 | 0.326 | 0.298 |
| EMG encoder RandInit | 0.748 | 0.701 | 0.479 | 0.454 |
| AvgPool | 0.761 | 0.711 | 0.518 | 0.454 |
| **CPEP (full)** | **0.782** | **0.757** | **0.536** | **0.481** |

### Key Findings
- MAE pre-training initialization is critical: random initialization leads to slower convergence and lower accuracy, and jointly training both encoders fails to converge.
- [CLS] token outperforms AvgPool, indicating that global context is more effective for gesture recognition.
- Freezing the EMG encoder yields drastically worse performance (0.372 vs. 0.782), confirming that fine-tuning the EMG encoder is necessary.
- Longer EMG patches degrade performance, underscoring the need for fine-grained temporal modeling.

## Highlights & Insights
- **First zero-shot gesture recognition framework for EMG**: The zero-shot results surpass the baseline's linear probing performance (0.481 vs. 0.312 on unseen gestures), demonstrating that contrastive pre-training yields representations with genuine generalization capability.
- The **strong-modality-anchors-weak-modality** paradigm is transferable to analogous settings such as IMU–video alignment and EEG–behavior alignment.

## Limitations & Future Work
- Validation is limited to a single dataset (emg2pose); generalizability to other EMG acquisition devices and protocols remains untested.
- The gesture vocabulary is small (4+4 classes), whereas practical applications require recognition of dozens to hundreds of gestures.
- No comparison is made against advanced contrastive learning methods such as SigLIP or CLAP.
- As a workshop paper, the experimental scale is limited and statistical significance is not reported.
- Online adaptation and few-shot fine-tuning scenarios are not explored.
- Robustness to inter-subject EMG signal variability is insufficiently analyzed.

## Related Work & Insights
- **vs. emg2pose**: Supervised pose regression offers limited generalization; CPEP's contrastive alignment produces structured embeddings that support zero-shot retrieval.
- **vs. CLIP**: CPEP adopts the cross-modal contrastive learning paradigm but introduces key adaptations—pre-training encoders to reduce data requirements and freezing the strong modality encoder.
- **vs. NeuroPose/Vemg2pose**: These baselines also employ Transformer architectures but are trained with supervised regression objectives, yielding embeddings of insufficient quality for retrieval-based classification.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of CLIP-style contrastive pre-training to EMG–pose alignment for zero-shot gesture recognition.
- Experimental Thoroughness: ⭐⭐⭐ Workshop paper; single dataset; limited gesture vocabulary.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method description is concise.
- Value: ⭐⭐⭐⭐ Opens a new direction for zero-shot EMG-based gesture recognition.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[NeurIPS 2025\] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)

<!-- RELATED:END -->
