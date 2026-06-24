---
title: >-
  [Paper Note] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals
description: >-
  [AAAI 2026][Human Understanding][Whole-body motion tracking] This paper proposes KineST, a kinematics-guided state space model that reconstructs whole-body motion from sparse HMD signals via a kinematic tree bidirectional scanning strategy and hybrid spatiotemporal representation learning, surpassing state-of-the-art methods in both accuracy and temporal consistency.
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "Whole-body motion tracking"
  - "state space model"
  - "kinematics prior"
  - "AR/VR"
  - "sparse signals"
date: 2026-05-08
content_hash: 592eef5c74514c44
---

# KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals

**Conference**: AAAI 2026
**arXiv**: [2512.16791](https://arxiv.org/abs/2512.16791)  
**Code**: [Project Page](https://kaka-1314.github.io/KineST/)  
**Area**: Video Understanding
**Keywords**: Whole-body motion tracking, state space model, kinematics prior, AR/VR, sparse signals

## TL;DR

This paper proposes KineST, a kinematics-guided state space model that reconstructs whole-body motion from sparse HMD signals via a kinematic tree bidirectional scanning strategy and hybrid spatiotemporal representation learning, surpassing state-of-the-art methods in both accuracy and temporal consistency.

## Background & Motivation

### State of the Field
Whole-body motion tracking plays a critical role in AR/VR applications. However, head-mounted devices (HMDs) provide only three sparse tracking signals from the head and hands, making the inference of full-body motion across 22 joints a highly challenging problem.

### Limitations of Prior Work
Existing methods face a fundamental tension between **accuracy and smoothness**:
- **High-computation approaches** (AvatarJLM: 63M parameters; SAGE: 137M): achieve strong performance by stacking Transformer blocks or employing large generative models, but incur prohibitive deployment costs unsuitable for real-time AR/VR applications.
- **Lightweight approaches** involve trade-offs:
    - RPM: introduces prediction-consistency anchors to improve smoothness, but **at the cost of pose accuracy**.
    - Separate spatiotemporal modules (HMD-Poser, MMD): improve joint interaction modeling but shift part of the modeling capacity to single-frame spatial features, **degrading motion smoothness**.
- The **State Space Duality (SSD)** framework offers efficiency advantages in temporal modeling but performs poorly when applied directly to motion tracking due to **unidirectional scanning and the absence of pose priors**.

### Core Idea

**Embedding kinematics priors into the SSD scanning strategy**: (1) redefining the scan order as bidirectional traversal along the human kinematic tree; (2) tightly coupling spatiotemporal context rather than modeling them separately; (3) defining a geometric angular velocity loss on the Lie group SO(3) to improve motion continuity.

## Method

### Overall Architecture

KineST consists of two types of core modules stacked alternately:

1. **Temporal Flow Module (TFM)**: $N$ modules for learning inter-frame dynamics.
2. **Spatiotemporal Kinematic Flow Module (SKFM)**: $M$ modules for kinematics-guided spatiotemporal modeling.

Input: sparse IMU signals $X \in \mathbb{R}^{L \times C}$ ($C = 3 \times (3+6+3+6) = 54$, each tracked part containing 3D position + 6D rotation + linear velocity + angular velocity).

Output: whole-body pose $Y \in \mathbb{R}^{L \times V}$ ($V = 22 \times 6$, 6D rotations for 22 SMPL joints).

Processing pipeline: linear embedding → $N$ TFMs → $M$ SKFMs → linear regression head.

### Key Designs

#### 1. **Temporal Flow Module (TFM)**

- **Bi-SSD block**: parallel forward and backward branches for bidirectional temporal modeling.
- Forward branch:
    - Generates state vectors $X$, $B$, $C$ via LN + Linear + Conv + SiLU.
    - Generates state transition matrix $A$ via Linear + LN.
    - Adaptive gating vector $f_1$ is element-wise multiplied with SSM output.
- Backward branch: reverses input along the time axis, applies identical operations, then reverses back.
- **Local Motion Aggregator (LMA)**: convolution-based local dependency modeling.
- **Global Motion Aggregator (GMA)**: lightweight Transformer-based global motion periodicity modeling.
- Output: $T_1 = \text{GMA}(\text{LMA}(F_f^t + F_b^t))$

#### 2. **Kinematic Tree Scanning Strategy (KTSS)**

- **Core innovation**: redefines the SSD scan order from standard index ordering to bidirectional traversal along the human kinematic tree.
- **Two variants**:
    - **Five-branch Kinematic Scan (FKS)**: strictly scans along five kinematic branches (head → left arm → right arm → left leg → right leg), better capturing local kinematic dependencies but at the cost of whole-body coherence.
    - **Unified Kinematic Scan (UKS)**: places the root joint at the center to effectively couple upper- and lower-body motion: `[21,19,17,14,15,12,20,18,16,13,9,6,3,0,1,4,7,10,2,5,8,11]`
- **UKS is adopted** as the final design: balancing local kinematic dependencies and global motion coherence.
- **Design Motivation**: the sequential nature of SSD allows each joint feature to be inferred from the preceding joint state; a scan order defined by the kinematic tree enables features to propagate along parent–child joint hierarchies.

#### 3. **Spatiotemporal Mixing Mechanism (STMM)**

- **Mechanism**: merges the spatial joint dimension and temporal frame dimension into a unified axis to achieve tight spatiotemporal coupling.
- **Procedure** (Algorithm 1):
  1. Temporal features $T_N$ are projected to joint space $S_l \in \mathbb{R}^{L \times H}$ and reshaped to $S_l' \in \mathbb{R}^{L \times J \times D}$.
  2. Reordered into forward and backward joint sequences $S_f, S_b$ according to KTSS.
  3. **Key step**: sequence and joint dimensions are merged: $S_f' \in \mathbb{R}^{(LJ_f) \times D}$.
  4. The mixed tensor is processed by Bi-SSD.
  5. Bidirectional features are summed, followed by linear projection + LMA + GMA.
- **Design Motivation**: modeling space or time independently degrades the other dimension; tight coupling ensures a balance between accuracy and smoothness.

### Loss & Training

#### Geometric Angular Velocity Loss

Angular velocity is computed in the tangent space (Lie algebra $\mathfrak{so}(3)$) of the Lie group SO(3):

$$V_t = R_{t-1}^{-1} R_t$$

$$\theta_V = \arccos\left(\frac{\text{Tr}(V)-1}{2}\right)$$

$$\log V = \theta_V \cdot \frac{1}{2\sin\theta_V}\begin{bmatrix}V_{32}-V_{23}\\ V_{13}-V_{31}\\ V_{21}-V_{12}\end{bmatrix}$$

$$\mathcal{L}_{\text{angvel}}^{\text{geo}} = \sum_{t=1}^{T-1}\|\log(V_t) - \log(\hat{V}_t)\|_1$$

- **Distinction from prior work**: prior methods approximate angular velocity via first-order finite differences in Euclidean space, ignoring the nonlinear manifold structure of rotations.
- **Design Motivation**: rotations lie on the SO(3) Lie group; angular velocity must be computed in the tangent space to be physically meaningful.

Total loss: $\mathcal{L} = \alpha \mathcal{L}_{\text{rot}} + \beta \mathcal{L}_{\text{ori}} + \delta \mathcal{L}_{\text{angvel}}^{\text{geo}}$ ($\alpha=1, \beta=0.02, \delta=1$)

Training configuration: NVIDIA 4090, batch\_size=256, Adam, lr=3e-4 (decayed to 3e-5), sequence length $L=96$.

## Key Experimental Results

### Main Results (Protocol 1, AMASS Dataset)

| Method | MPJRE↓ | MPJPE↓ | MPJVE↓ | Hand PE↓ | Upper PE↓ | Lower PE↓ | Jitter↓ | Params |
|--------|--------|--------|--------|----------|-----------|-----------|---------|--------|
| AvatarPoser | 3.08 | 4.18 | 27.70 | 2.12 | 1.81 | 7.59 | 14.49 | 4M |
| AvatarJLM | 2.90 | 3.35 | 20.79 | 1.24 | 1.72 | 6.20 | 8.39 | 63M |
| SAGE | 2.53 | 3.28 | 20.62 | 1.18 | 1.39 | 6.01 | 6.55 | 137M |
| HMD-Poser | 2.32 | 3.15 | 18.15 | 1.35 | 1.34 | 5.76 | 6.21 | 17M |
| MMD | 2.31 | 3.22 | 17.88 | 0.94 | 1.29 | 6.01 | 7.39 | 14M |
| **KineST** | **2.25** | **2.86** | **15.26** | **1.04** | **1.24** | **5.20** | **5.97** | **11M** |

With only **11M parameters** (1/12 of SAGE), KineST achieves state-of-the-art performance across all metrics: MPJRE −2.6%, MPJPE −11.2%, MPJVE −14.7%.

### Protocol 2 & 3

| Method | Protocol 2 MPJRE↓ | MPJPE↓ | MPJVE↓ | Protocol 3 MPJRE↓ | MPJPE↓ | MPJVE↓ |
|--------|-------------------|--------|--------|-------------------|--------|--------|
| AvatarJLM | 4.30 | 4.93 | 26.17 | 7.01 | 9.72 | 27.59 |
| AGRoL | 4.30 | 6.17 | 24.40 | — | — | — |
| **KineST** | **4.28** | **5.17** | **24.08** | **6.91** | **9.68** | **25.16** |

KineST also achieves state-of-the-art results on real-device captured data (Protocol 3), validating its practical applicability.

### Ablation Study

| Scanning Strategy | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|------------------|--------|--------|--------|---------|
| Index-order (SMPL) | 2.32 | 3.11 | 17.81 | 8.27 |
| FKS (five-branch) | 2.28 | 3.00 | 16.25 | 7.01 |
| **UKS (unified)** | **2.25** | **2.86** | **15.26** | **5.97** |

| SKFM Modeling Mechanism | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|------------------------|--------|--------|--------|---------|
| Temporal only | 2.27 | 2.97 | 16.84 | 7.83 |
| Spatial only (holistic) | 2.41 | 3.10 | 16.77 | 7.72 |
| Spatial only (token-wise) | 2.23 | 2.93 | 17.85 | 9.31 |
| **STMM (mixed)** | **2.25** | **2.86** | **15.26** | **5.97** |

| Loss Function | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|--------------|--------|--------|--------|---------|
| Baseline | 2.25 | 2.87 | 16.10 | 6.75 |
| + $L_{angvel}^{diff}$ (finite difference) | 2.29 | 3.03 | 15.91 | 6.44 |
| + **$L_{angvel}^{geo}$ (geometric)** | **2.25** | **2.86** | **15.26** | **5.97** |

### Key Findings

1. **UKS uniformly outperforms FKS and index-order scanning**: Jitter drops from 8.27 to 5.97, demonstrating that unified whole-body scanning is superior to branch-wise decomposition.
2. **STMM is key to accuracy and smoothness**: token-wise spatial modeling achieves the lowest MPJRE but the highest Jitter (9.31); STMM achieves a balanced optimum across all metrics.
3. **Geometric angular velocity loss outperforms the finite-difference variant**: the latter reduces MPJVE but significantly increases rotation and positional errors, whereas the geometric formulation improves smoothness without sacrificing accuracy.
4. **Every component of the Flow Module is indispensable**: removing Bi-SSD causes Jitter to surge from 5.97 to 13.57.

## Highlights & Insights

1. **Elegant integration of kinematics priors**: redefining the SSD scan order as kinematic tree traversal is conceptually simple yet yields substantial gains (28% Jitter reduction).
2. **Spatiotemporal mixing over separation**: tight coupling via dimension merging resolves the accuracy/smoothness trade-off inherent in disjoint modeling.
3. **Geometric consistency**: defining the angular velocity loss on the SO(3) Lie group is mathematically rigorous and physically meaningful.
4. **Extreme efficiency**: 11M parameters surpassing the 137M SAGE model, with inference over 96 frames taking only 12.9 ms, making it suitable for AR/VR deployment.

## Limitations & Future Work

1. Performance degrades on complex motions (e.g., gymnastics, acrobatics), necessitating richer motion priors.
2. The current design infers from only 3 HMD tracking points without exploiting potential additional sensors (e.g., waist IMU).
3. The fixed sequence length of 96 frames requires re-tuning for longer or shorter sequences.
4. The SMPL model covers only the first 22 joints, leaving fine-grained joints such as fingers unaddressed.

## Related Work & Insights

- KineST shares the SSM paradigm with MMD, but whereas MMD models space and time separately, KineST achieves tight coupling via STMM.
- The kinematic tree scanning strategy is generalizable to other skeleton-related tasks such as action recognition and motion generation.
- The Lie group formulation of the geometric angular velocity loss is applicable to any problem involving rotation prediction.
- The LMA + GMA local/global aggregator design constitutes a general-purpose tool for sequence modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Both KTSS and STMM are innovative designs, and the geometric angular velocity loss rests on rigorous mathematical foundations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three protocols, real-device data, and comprehensive ablations covering scanning strategies, modeling mechanisms, loss functions, individual components, and sequence lengths.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-presented algorithm pseudocode, and rich visualizations.
- **Value**: ⭐⭐⭐⭐⭐ — Directly targets real-time AR/VR deployment requirements; the lightweight and efficient solution offers high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](../../ICCV2025/human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)
- [\[AAAI 2026\] Improving Sparse IMU-based Motion Capture with Motion Label Smoothing](improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
