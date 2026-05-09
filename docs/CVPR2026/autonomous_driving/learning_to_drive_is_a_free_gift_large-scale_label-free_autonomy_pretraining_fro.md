---
title: >-
  [Paper Note] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
description: >-
  [CVPR 2026][Autonomous Driving][Autonomous driving pretraining] This paper proposes LFG (Learning to drive is a Free Gift), a fully label-free, teacher-guided pretraining framework for autonomous driving. LFG learns a unified pseudo-4D representation of geometry, semantics, and motion from large-scale unposed YouTube driving videos. On the NAVSIM benchmark, using only a monocular front-facing camera, LFG surpasses multi-camera + LiDAR BEV methods (PDMS 85.2), and demonstrates strong data efficiency (81.4 PDMS with only 10% labels).
tags:
  - CVPR 2026
  - Autonomous Driving
  - Autonomous driving pretraining
  - label-free learning
  - video foundation model
  - 4D scene understanding
  - planning
date: 2026-05-08
content_hash: 49b4de8b3d147082
---

# Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos

**Conference**: CVPR 2026
**arXiv**: [2602.22091](https://arxiv.org/abs/2602.22091)
**Code**: [Project Page](https://lfg-ai.github.io/)
**Area**: Autonomous Driving
**Keywords**: Autonomous driving pretraining, label-free learning, video foundation model, 4D scene understanding, planning

## TL;DR

This paper proposes LFG (Learning to drive is a Free Gift), a fully label-free, teacher-guided pretraining framework for autonomous driving. LFG learns a unified pseudo-4D representation of geometry, semantics, and motion from large-scale unposed YouTube driving videos. On the NAVSIM benchmark, using only a monocular front-facing camera, LFG surpasses multi-camera + LiDAR BEV methods (PDMS 85.2), and demonstrates strong data efficiency (81.4 PDMS with only 10% labels).

## Background & Motivation

The internet contains an enormous volume of egocentric driving videos (e.g., from YouTube), yet these data carry no annotations whatsoever — no 3D labels, no camera poses, no LiDAR, and no semantic segmentation labels. While existing autonomous driving methods benefit from scaling up, they remain **heavily dependent on annotated data** (expert trajectories, LiDAR scans, odometry, semantic annotations).

A natural question arises: **Can we pretrain powerful autonomous driving representations from large-scale unannotated driving videos, in the same spirit as GPT trained on unlabeled text corpora?**

Existing self-supervised methods (e.g., PPGeo) primarily rely on inter-frame consistency losses (e.g., photometric consistency), implicitly assuming static scenes, and thus fail to capture **dynamic objects** — which are precisely the most safety-critical elements in driving. Meanwhile, large world models (e.g., UniPAD, ViDAR) still require a certain degree of supervised labels.

The core insight of LFG is: **safe reactive driving requires not only understanding the 3D structure of the current scene, but also predicting the short-term evolution of geometry, motion, and semantics.** Accordingly, LFG adopts a feedforward "current + future" joint prediction framework, leveraging multiple specialized teacher models (π³, SegFormer, SAM2, CoTracker3) to provide pseudo-supervision, and learns a unified 4D driving representation without any ground-truth annotations.

## Method

### Overall Architecture

LFG is built on top of π³ (a feedforward 3D reconstruction model with ~1B parameters). The core architecture consists of:

1. **Pretrained π³ encoder**: Takes $N$ frames of unposed RGB images as input and outputs latent scene tokens $\mathbf{Z}_{1:N}$
2. **Causal autoregressive Transformer**: Predicts $M$ future tokens $\mathbf{Z}_{N+1:N+M}$ from the observed tokens
3. **Shared decoder**: Decodes all $N+M$ frame tokens into five output modalities

The total model parameter count is approximately 1.45B, running at 5Hz on an RTX 5090.

### Prediction Outputs

For each frame $t \in \{1, ..., N+M\}$, the model predicts:

- **Point map** $P_t: \mathcal{I}(I_t) \to \mathbb{R}^3$ — 3D world coordinates per pixel
- **Camera pose** $T_t \in \mathbb{R}^{4 \times 4}$ — ego-vehicle motion trajectory
- **Semantic segmentation** $S_t \in \mathbb{R}^{7 \times H \times W}$ — 7 classes (road, vehicle, pedestrian, building, vegetation, sky, background)
- **Confidence map** $C_t \in [0,1]^{H \times W}$ — per-pixel reliability of 3D predictions
- **Motion mask** $M_t \in [0,1]^{H \times W}$ — regions of independent motion

### Key Designs

1. **π³ Teacher Distillation (Geometry + Pose)**: The π³ teacher model observes all $N+M$ frames, while the LFG student sees only the first $N$ frames. The teacher provides point maps, confidence maps, and camera poses for all frames as pseudo-supervision. The student is forced to predict both current and future 3D geometric structure from partial observations:

    $\mathcal{L}_{\text{point}} = \alpha \|\mathbf{P} - \widehat{\mathbf{P}}\|_1$

   The pose loss uses pairwise relative rotation (SO(3) geodesic distance) and translation (Huber loss). The motivation for this distillation mechanism is to enable LFG to **extrapolate future scene structure from current observations**.

2. **SegFormer Semantic Distillation**: A pretrained SegFormer (Cityscapes) serves as the semantic teacher, generating pseudo semantic labels for each frame. The teacher has access to all frames (including ground-truth RGB of future frames), whereas LFG sees only the first $N$ frames yet must predict semantic segmentation for all $N+M$ frames. A weighted BCE loss is used to handle class imbalance. The motivation is to equip LFG with scene semantic understanding.

3. **Motion Mask Generation Pipeline**: A fully automated, annotation-free pipeline:

    - Grounded SAM2 detects person and vehicle instances in the first frame
    - CoTracker3 tracks 2D trajectories $\mathbf{u}_t^{(i)}$ of these instances across frames
    - The π³ teacher's point maps are used to back-project 2D trajectories into 3D space
    - The 3D temporal displacement of each instance is computed: $d_t^{(i)} = \|\bar{\mathbf{p}}_{t+1}^{(i)} - \bar{\mathbf{p}}_t^{(i)}\|_2$
    - Instances whose displacement exceeds threshold $\tau_{\text{motion}}$ for at least $K_{\min}$ frames are labeled as dynamic
    - Instance-level motion indicators are converted into per-pixel motion masks $\mathbf{M}_t$

   Motivation: decoupling dynamic objects from the static environment, which is the fundamental limitation of prior inter-frame consistency approaches.

4. **Causal Autoregressive Transformer**: 4 layers, 8 attention heads, dropout 0.1. It receives scene tokens from the encoder and uses causal attention (attending only to past and current tokens, not future ones) to predict latent representations of future frames. Future prediction is formulated as a next-token prediction problem.

### Loss & Training

The total loss comprises current-frame loss plus a weighted future-frame loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{current}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}$$

Each component contains four sub-losses: $\lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{pose}}\mathcal{L}_{\text{pose}} + \lambda_{\text{point}}\mathcal{L}_{\text{point}} + \lambda_{\text{motion}}\mathcal{L}_{\text{motion}}$

- Future frame weight $\omega = 10.0$ (emphasizing extrapolation capability)
- AdamW optimizer, lr=1e-4, cosine annealing
- BF16 mixed-precision training, 32× A100 GPUs, 40k iterations
- Three-stage training: ① geometry + pose autoregression → ② add semantic head → ③ add motion head
- Training data: subset of the OpenDV YouTube driving dataset (~2M samples), multi-framerate training (2/5/10 Hz)

## Key Experimental Results

### Main Results — NAVSIM Planning Benchmark

| Method | Input | NC | DAC | TTC | EP | PDMS |
|--------|-------|----|-----|-----|----|----|
| UniAD | 6Cam | 97.8 | 91.9 | 92.9 | 78.8 | 83.4 |
| TransFuser | 3Cam+LiDAR | 97.7 | 92.8 | 92.0 | 79.2 | 84.0 |
| Hydra-MDP | 3Cam+LiDAR | 96.9 | 94.0 | 94.0 | 78.7 | 84.7 |
| DiffusionDrive | 3Cam+LiDAR | 96.8 | 95.4 | 94.7 | 82.0 | 88.1 |
| **LFG (ours)** | **1Cam** | **98.2** | 93.7 | 94.4 | 79.1 | **85.2** |

### Data Efficiency Comparison (PDMS↑)

| Method | Input | 1% Data | 10% Data | 100% Data |
|--------|-------|---------|----------|-----------|
| PPGeo | 1Cam | 61.5 | 65.6 | 74.6 |
| π³ | 1Cam | 56.2 | 77.5 | 82.8 |
| DINOv3 | 1Cam | 60.0 | 75.8 | 81.4 |
| **LFG** | **1Cam** | **66.3** | **81.4** | **85.2** |

### Ablation Study

| Configuration | 1% Data | 10% Data | 100% Data | Notes |
|---------------|---------|----------|-----------|-------|
| Full LFG | 66.3 | 81.4 | 85.2 | Baseline |
| +2× pretraining data | 76.6 | 82.3 | 84.8 | Large gain under low-label regime |
| +Longer prediction horizon | 80.5 | 84.4 | 84.8 | 14+ gain with 1% data |
| −Semantic/motion heads | 64.8 | 77.1 | 84.6 | Semantic and motion supervision matters |
| −Autoregressive head | 66.3 | 77.7 | 84.2 | Future prediction capability is critical |

### Key Findings

- **Monocular front camera > multi-camera + LiDAR**: LFG, using only a single front-facing camera, surpasses UniAD (6 cameras) and Hydra-MDP (3 cameras + LiDAR) on NAVSIM — demonstrating that large-scale video pretraining can compensate for sensor configuration disadvantages.
- **Exceptional data efficiency**: 10% labels suffice to reach 81.4 PDMS, matching DINOv3 trained on 100% of data.
- **Segmentation surpasses teacher**: On KITTI-360, LFG's segmentation quality (PA 0.947, mIoU 0.768) exceeds that of its SegFormer teacher (PA 0.926, mIoU 0.677) — indicating mutual benefit from multi-task joint learning.
- **Self-correcting motion prediction**: LFG correctly identifies stationary parked vehicles, whereas the pseudo-GT motion labels erroneously mark them as dynamic — the student model outperforms its teacher labels.

## Highlights & Insights

- **"Free lunch" paradigm**: The paper demonstrates that YouTube driving videos can serve as a powerful, cost-free pretraining data source for autonomous driving, requiring no annotations, poses, or LiDAR.
- **Multi-teacher collaborative distillation** is an elegant paradigm: complementary strengths of different expert models are combined to generate pseudo-supervision, circumventing the limitations of any single teacher.
- **Next-token prediction applied to driving**: Geometric and motion future prediction is formulated as token sequence prediction, unifying the paradigm with NLP.
- Experiments confirm that **short-term future prediction** is a key feature for driving planning — removing the autoregressive head drops 10%-data performance from 81.4 to 77.7.

## Limitations & Future Work

- Only **short-term future** frames (3–6) are predicted; long-horizon reasoning is limited and could be extended to multi-scale temporal horizons.
- Only a **single front-facing camera** is used — while this reflects the nature of YouTube videos, multi-view training (e.g., with emerging PhysicalAI datasets) may further improve performance.
- The quality of teacher models (e.g., π³) sets an upper bound on student model performance.
- The motion mask generation pipeline relies on multiple large models (SAM2 + CoTracker3 + π³), incurring high computational cost.
- At 1.45B parameters and 5Hz inference, real-time deployment remains a challenge.

## Related Work & Insights

- π³ (feedforward 3D reconstruction) provides the backbone for LFG; LFG's innovation lies in adding **temporal modeling** and **multi-modal pseudo-supervision**.
- Compared to ViDAR (a pioneer in using future point cloud prediction as a pretraining task), LFG requires no LiDAR data whatsoever.
- PPGeo demonstrated the utility of geometric priors for driving, but its inter-frame consistency assumption (static scenes) is a fundamental limitation.
- The large-scale unannotated pretraining paradigm of DINOv3/GPT is validated in the vision-driving domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Fully label-free YouTube video pretraining surpassing multi-sensor methods — a highly impactful paradigm shift.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task evaluation (segmentation/depth/trajectory/planning), data efficiency analysis, and ablations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, polished figures, and a coherent overall narrative.
- **Value**: ⭐⭐⭐⭐⭐ Opens an "internet-scale free data" avenue for autonomous driving pretraining with broad potential impact.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SimScale: Learning to Drive via Real-World Simulation at Scale](simscale_learning_to_drive_via_real-world_simulation_at_scale.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)

<!-- RELATED:END -->
