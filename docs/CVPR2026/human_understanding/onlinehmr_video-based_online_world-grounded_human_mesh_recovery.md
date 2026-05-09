---
title: >-
  [Paper Note] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery
description: >-
  [CVPR 2026][Human Understanding][Human Mesh Recovery] This paper proposes OnlineHMR, the first online world-grounded human mesh recovery framework that simultaneously satisfies four criteria: system causality, faithfulness, temporal consistency, and efficiency. It achieves streaming camera-space HMR via sliding-window causal learning with KV-cache inference, and performs online global localization through human-centric incremental SLAM combined with EMA trajectory correction.
tags:
  - CVPR 2026
  - Human Understanding
  - Human Mesh Recovery
  - Online Inference
  - SLAM
  - World Coordinates
  - Causal Inference
  - KV Cache
date: 2026-05-08
content_hash: 31d645928e2e2837
---

# OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery

**Conference**: CVPR 2026
**arXiv**: [2603.17355](https://arxiv.org/abs/2603.17355)
**Code**: [GitHub](https://tsukasane.github.io/Video-OnlineHMR/)
**Institution**: Carnegie Mellon University, University of Pennsylvania
**Area**: 3D Vision
**Keywords**: Human Mesh Recovery, Online Inference, SLAM, World Coordinates, Causal Inference, KV Cache

## TL;DR

This paper proposes OnlineHMR, the first online world-grounded human mesh recovery framework that simultaneously satisfies four criteria: system causality, faithfulness, temporal consistency, and efficiency. It achieves streaming camera-space HMR via sliding-window causal learning with KV-cache inference, and performs online global localization through human-centric incremental SLAM combined with EMA trajectory correction.

## Background & Motivation

**Background**: Human mesh recovery (HMR) reconstructs 3D human pose and shape from monocular video. Recent years have seen an extension from camera-space estimation to world-space global human trajectory and motion recovery, with methods such as WHAM, TRAM, and GVHMR achieving notable progress.

**Limitations of Prior Work**: (a) Most methods are **offline** — TCMR/GLoT take 16-frame sequences to estimate the middle frame, and TRAM relies on globally optimized SLAM; (b) WHAM claims to be online, but its global trajectory module actually depends on offline DPVO/DROID-SLAM, which uses future frames to refine past camera poses; (c) Human3R supports online inference but suffers from poor local motion quality and severe jitter, as 4D scene reconstruction provides far fewer human-specific training samples than scene data; (d) Real-time interactive applications such as AR/VR, telepresence, and perception-action loops are consequently excluded.

**Key Challenge**: Under strict causal constraints (no future frames, no global optimization), how can one simultaneously guarantee global trajectory accuracy and local motion quality?

**Goal**: The paper decouples the problem into two expert branches — camera-space HMR (precise local motion) and incremental SLAM (global localization) — each made causal independently and bridged through physical constraints. Four online processing criteria are proposed: (1) system-level causality; (2) faithful geometric/pose reconstruction; (3) temporal consistency; (4) constant-time-complexity inference efficiency.

## Method

### Overall Architecture

**Input**: Streaming monocular RGB video, processed frame by frame. **Output**: Per-frame SMPL human mesh in world coordinates, $\mathbf{M}_i^w \in \mathbb{R}^{6890 \times 3}$.

The framework consists of **two parallel branches**:

- **Branch 1: Camera-space Online HMR** — Initialized from HMR2.0 (ViT backbone); sliding-window causal attention fuses temporal information; KV cache enables streaming inference; outputs per-frame SMPL parameters in camera space (pose $\boldsymbol{\theta}_i \in \mathbb{R}^{23 \times 3}$, shape $\boldsymbol{\beta}_i \in \mathbb{R}^{10}$, root rotation $\mathbf{R}_i^{\text{root}}$, root translation $\mathbf{t}_i^{\text{root}}$), producing camera-space mesh $\mathbf{M}_i^c$.
- **Branch 2: Human-centric Incremental SLAM** — SAM2 segments the human body → dilation + Gaussian blur generates a soft mask → dynamic human regions are masked → MASt3R-SLAM incrementally estimates camera pose $\{\mathbf{q}_i^c, \mathbf{t}_i^c\}$ → EMA smoothing correction → MoGe-V2 metric depth recovers scale factor $s$.

**Coordinate Transformation**: The world-space mesh is obtained via rigid-body transformation:

$$\mathbf{M}_i^w = \mathbf{R}(\mathbf{q}_i^c) \cdot \mathbf{M}_i^c + s \cdot \mathbf{t}_i^c$$

where $\mathbf{R}(\mathbf{q}_i^c)$ is the rotation matrix corresponding to quaternion $\mathbf{q}_i^c \in \mathbb{R}^4$, and $s$ is the metric scale factor.

### Key Design 1: Sliding-Window Causal Learning

**Function**: Leverage short-term temporal information under causal constraints to achieve smooth per-frame HMR.

**Window Partitioning**: The input sequence is divided into overlapping windows of size $N$ with stride 1. Each window contains frames $i-N+1$ through $i$.

**Intra-window Information Fusion**:

- All frames are processed by the ViT backbone to extract patch-level spatial features.
- The last frame in the window (current frame $i$) performs **self-attention** on itself.
- The current frame also serves as query in **cross-attention** over the preceding $N-1$ frames, aggregating temporal context.
- The fused features are fed into the SMPL head to regress single-frame human parameters.
- Per-frame outputs from each window are concatenated to form the complete sequence.

**Design Motivation**: Methods such as TCMR/MPS-Net/GLoT take 16-frame sequences and estimate the middle frame, requiring $N/2$ future frames — violating the causal constraint. The proposed design uses only the current and past frames, naturally satisfying causality. During training, windows can be computed in parallel (non-online); during inference, KV cache converts this to an online mode.

### Key Design 2: KV-Cache Streaming Inference

**Function**: Achieve per-frame inference with constant time complexity.

**Mechanism**: The key and value features of the previous $N-1$ frames ($\mathbf{k}_{i-1}...\mathbf{k}_{i-N+1}$, $\mathbf{v}_{i-1}...\mathbf{v}_{i-N+1}$) are cached. For the current frame, only the following computations are required:

$$\mathbf{A}_{\text{self}} = \text{Softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_i^\top}{\sqrt{d}}\right)\mathbf{v}_i$$

$$\mathbf{A}_{\text{cross}} = \text{Softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_{\text{prev}}^\top}{\sqrt{d}}\right)\mathbf{v}_{\text{prev}}$$

where $d$ is the feature dimension, and $\mathbf{k}_{\text{prev}}$, $\mathbf{v}_{\text{prev}}$ are concatenated from the cache. After processing, the new frame's key/value is added to the cache and the oldest frame is evicted.

**Core Idea**: Training is non-online (windows are parallelized to fully utilize GPU throughput); inference is online (KV cache ensures causality and constant compute). This resolves the training–inference consistency problem.

### Key Design 3: Velocity/Acceleration Regularization

**Function**: Suppress cross-window joint motion jitter and maintain temporal coherence.

**Velocity Regularization** (penalizes inter-frame joint displacement):

$$\mathcal{L}_v = \lambda_5 \frac{\sum_{i,t} c_{i,t} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t-1}\|_2^2}{\sum_{i,t} c_{i,t} + \epsilon}$$

**Acceleration Regularization** (penalizes changes in acceleration):

$$\mathcal{L}_a = \lambda_6 \frac{\sum_{j,i} c_{j,i} \|\mathbf{p}_{j,i+1} - 2\mathbf{p}_{j,i} + \mathbf{p}_{j,i-1}\|_2^2}{\sum_{j,i} c_{j,i} + \epsilon}$$

where $\mathbf{p}$ denotes joint positions relative to the pelvis, $j$ is the joint index, and $c$ is the per-joint confidence provided by ground truth (used as weights to mitigate the influence of occluded joints). The unit-stride sliding window design allows consecutive frame outputs to be concatenated into a complete sequence, enabling velocity/acceleration losses to be computed across window boundaries.

### Key Design 4: Human-centric Incremental SLAM + EMA Correction

**Challenge**: In human-centric videos, the human body occupies a large image region, and dynamic textures and deformations violate the static-scene assumption of SLAM.

#### (a) Human Soft Mask

SAM2 segments the human region $C_i^h$; dilation followed by Gaussian blurring produces a soft confidence mask:

$$C_i^{\text{soft}} = \frac{G_\sigma * (C_i^h \oplus S_k^{(n)})}{\max_p (G_\sigma * (C_i^h \oplus S_k^{(n)}))}$$

where $S_k^{(n)}$ is the dilation kernel. Compared to hard masks, soft masks prevent sharp human-body boundaries from being incorrectly encoded as features by SLAM. Feature extraction and matching are performed only in static background regions.

#### (b) EMA Trajectory Smoothing

A history buffer of size $B$ is maintained with exponentially decaying weights $w_m = (1-\alpha)^{B-1-m}$ (normalized so that $\sum w_m = 1$):

$$\bar{\mathbf{t}}_i = \sum_{m=0}^{B-1} w_m \mathbf{t}_{i-m}, \quad \Delta\mathbf{t}_i = \mathbf{t}_i - \bar{\mathbf{t}}_i$$

Velocity-adaptive clamping: threshold $\tau = \lambda_{\text{clamp}} \bar{v}$; if $\|\Delta\mathbf{t}_i\| > \tau$, the update is scaled. The final smoothed translation is:

$$\mathbf{t}_i' = \bar{\mathbf{t}}_i + \alpha \Delta\mathbf{t}_i$$

Rotation smoothing uses quaternion LERP as an approximation to SLERP (hemisphere flipping is applied first to ensure positive inner product):

$$\mathbf{q}_i' = \text{normalize}((1-\alpha)\mathbf{q}_{i-1}' + \alpha \mathbf{q}_i)$$

**Core Idea**: Rather than directly smoothing human motion (which would constrain extreme actions such as gymnastic flips), the method smooths the SLAM camera trajectory to indirectly regularize human motion — a more general approach that is not limited by motion priors.

#### (c) Metric Scale Recovery

MoGe-V2 estimates per-frame metric depth, which is compared against the SLAM depth map to compute the scale factor $s$. Human-region pixels are excluded from this computation, as the human body is blurred in the SLAM depth map and may exhibit a dolly-zoom effect — where the camera physically approaches the subject while zooming out, keeping the subject's image size constant while actual depth increases.

### Loss & Training

Per-frame HMR loss:

$$\mathcal{L}_f = \lambda_1 \mathcal{L}_{2D} + \lambda_2 \mathcal{L}_{3D} + \lambda_3 \mathcal{L}_{\text{SMPL}} + \lambda_4 \mathcal{L}_V$$

Total loss:

$$\mathcal{L} = \mathcal{L}_f + \mathcal{L}_v + \mathcal{L}_a$$

where $\mathcal{L}_{2D}$ (2D keypoint reprojection), $\mathcal{L}_{3D}$ (3D keypoints), $\mathcal{L}_{\text{SMPL}}$ (SMPL parameters), and $\mathcal{L}_V$ (3D vertices) provide standard per-frame supervision; $\mathcal{L}_v$ and $\mathcal{L}_a$ are cross-window velocity/acceleration regularization terms. Training data: BEDLAM + 3DPW + H3.6M; convergence is reached in approximately 52K iterations on a single H100 GPU.

### Frequency-Domain Jitter Metric

A motion naturalness metric based on STFT spectral analysis is proposed. For a motion sequence $\mathbf{y}(i) \in \mathbb{R}^{F \times 3J}$, the spectrogram is computed as:

$$\mathbf{S}(i,f) = \left|\sum_{k=0}^{L-1} \mathbf{y}(k) w(k-i) e^{-j2\pi fk/N_w}\right|$$

where $w(\cdot)$ is the Hann window function. Natural human motion is predominantly below 10 Hz; high-frequency components reflect the degree of jitter. Compared to conventional Accel/Jitter metrics, this frequency-domain metric better aligns with human perceptual sensitivity to motion jitter.

## Key Experimental Results

### Main Results: Camera-Space HMR (EMDB-1 Dataset, unit: mm)

| Method | Type | PA-MPJPE $\downarrow$ | MPJPE $\downarrow$ | PVE $\downarrow$ | Accel $\downarrow$ |
|------|------|:------:|:------:|:------:|:------:|
| HMR2.0 | Per-frame | 60.7 | 98.3 | 120.8 | 19.9 |
| ReFit | Per-frame | 58.6 | 88.0 | 104.5 | 20.7 |
| TRAM | Offline | 45.7 | 74.4 | 86.6 | 4.9 |
| GVHMR | Offline | 44.5 | 74.2 | 85.9 | — |
| PHMR | Offline | 40.1 | 68.1 | 79.2 | — |
| TRACE | Online | 71.5 | 110.0 | 129.6 | 25.5 |
| Human3R | Online | 48.5 | 73.9 | 86.0 | — |
| **OnlineHMR** | **Online** | **46.0** | **74.0** | **86.1** | **9.0** |

### Main Results: World-Space Global Trajectory (EMDB-2 Dataset)

| Method | Type | PA-MPJPE $\downarrow$ | WA-MPJPE $\downarrow$ | W-MPJPE $\downarrow$ | RTE(%) $\downarrow$ | ERVE $\downarrow$ |
|------|------|:------:|:------:|:------:|:------:|:------:|
| WHAM+DPVO | Offline | 38.2 | 135.6 | 354.8 | 6.0 | 14.7 |
| TRAM | Offline | 38.1 | 76.4 | 222.4 | 1.4 | 10.3 |
| PHMR | Offline | — | 71.0 | 216.5 | 1.3 | — |
| TRACE | Online | 58.0 | 529.0 | 1702.3 | 17.7 | 370.7 |
| Human3R | Online | — | 112.2 | 267.9 | 2.2 | — |
| **OnlineHMR** | **Online** | **40.1** | **93.5** | **310.4** | **2.2** | **12.4** |

### Efficiency Comparison

| Method | Online | FPS $\uparrow$ | Avg. Latency (s) $\downarrow$ | WA-MPJPE $\downarrow$ |
|------|:------:|:------:|:------:|:------:|
| SLAHMR | ✗ | 0.1 | 2435 | 326.9 |
| TRAM | ✗ | 2.1 | 115.95 | 76.4 |
| WHAM+DPVO | ✗ | 9.3 | 26.18 | 135.6 |
| Human3R | ✓ | 4.8 | 0.21 | 112.2 |
| **OnlineHMR** | **✓** | **3.3** | **0.30** | **93.5** |

### Ablation Study

**Velocity Regularization Ablation (Accel / Jitter metrics)**:

| Setting | 3DPW Accel $\downarrow$ | 3DPW Jitter $\downarrow$ | EMDB-1 Accel $\downarrow$ | EMDB-1 Jitter $\downarrow$ |
|------|:------:|:------:|:------:|:------:|
| w/o velocity regularization | 8.9 | 32.3 | 15.7 | 70.1 |
| **w/ velocity regularization** | **6.4** | **19.5** | **9.0** | **33.7** |

**SLAM Masking Strategy Ablation (ATE metric, lower is better)**:

| SLAM Method | No Mask | Hard Mask | Soft Mask |
|------|:------:|:------:|:------:|
| DROID-SLAM | 2.52 | 1.55 | **1.07** |
| MASt3R-SLAM | 1.22 | 0.96 | **0.83** |

### Key Findings

- **Minimal cost of causal inference**: PA-MPJPE on EMDB-1 is only 0.3 mm higher than offline TRAM (45.7), while Accel remains well controlled.
- **Consistently outperforms online methods**: Camera-space accuracy significantly surpasses TRACE (PA-MPJPE 46.0 vs. 71.5); world-space WA-MPJPE is 18.7 lower than Human3R (93.5 vs. 112.2).
- **Velocity regularization is effective**: Jitter decreases from 32.3/70.1 to 19.5/33.7 on 3DPW/EMDB-1, approximately halved.
- **Soft mask outperforms hard mask and no mask**: MASt3R-SLAM ATE improves from 1.22 (no mask) → 0.96 (hard mask) → 0.83 (soft mask).
- **Online method achieves very low latency**: Average latency of 0.30 s vs. 115.95 s for offline TRAM (~400× speedup).
- **Reason for higher W-MPJPE**: Metric scale recovery is insufficiently precise — good WA-MPJPE but elevated W-MPJPE indicates that the global trajectory shape is correct, but scale drifts in later frames of the incremental estimation.

## Highlights & Insights

- **KV-cache design with non-online training and online inference**: Training exploits window parallelism for GPU efficiency; inference uses KV cache to guarantee causality and constant time. This training–inference decoupling pattern is transferable to any video understanding task requiring online deployment.
- **Indirect regularization via camera smoothing**: Rather than imposing motion priors directly on human motion (which would constrain extreme actions), the method smooths the SLAM camera trajectory so that the global transformation is naturally smooth — an elegant solution that avoids the poor generalization of motion priors.
- **Decoupling local motion and global localization into two expert branches**: This avoids the local motion accuracy degradation seen in end-to-end methods such as Human3R, where training data imbalance (4D scene data >> human data) limits precision.
- **Formal four-criterion definition**: The requirements for online HMR are systematically formalized along four dimensions — causality, faithfulness, consistency, and efficiency — providing a clear evaluation framework for future work.
- **Frequency-domain jitter metric**: STFT-based spectral analysis better reflects human perceptual sensitivity to motion jitter compared to conventional Accel/Jitter metrics.

## Limitations & Future Work

1. **World-space accuracy still lags behind offline methods**: WA-MPJPE of 93.5 vs. TRAM 76.4 / PHMR 71.0; metric scale recovery remains the bottleneck.
2. **Scale drift**: Elevated W-MPJPE indicates that scale is insufficiently stable in later frames of incremental estimation, potentially accumulating error over long sequences.
3. **Manual hyperparameter tuning for EMA**: Parameters $\alpha$, $B$, and $\lambda_{\text{clamp}}$ are sensitive to motion type; extreme motions may be over-smoothed.
4. **Assumes continuous viewpoint**: Cannot handle abrupt shot cuts or multi-camera input.
5. **Dependency on external models**: SAM2 (segmentation) + MoGe-V2 (depth) + MASt3R-SLAM (trajectory) increase overall system complexity.
6. **Multi-person scenarios not systematically evaluated**: Although multi-person visualizations are shown, no quantitative evaluation is provided.
7. **Community acceptance of the frequency-domain metric remains to be validated**.

## Related Work & Insights

- **vs. TRAM**: Both adopt a two-branch architecture, but TRAM uses globally optimized SLAM and is offline. OnlineHMR replaces this with incremental SLAM + EMA, trading a modest accuracy drop (WA-MPJPE +17.1) for ~400× latency reduction.
- **vs. Human3R**: End-to-end online reconstruction based on implicit constraints from CUT3R. Local motion is jittery and inaccurate. After decoupling, OnlineHMR achieves higher local accuracy (PA-MPJPE 46.0 vs. 48.5) and better global performance (WA-MPJPE 93.5 vs. 112.2).
- **vs. WHAM**: Camera-space estimation is online, but global localization is offline (DPVO uses future frames to correct past camera poses). OnlineHMR is the first system to achieve fully online processing end-to-end.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First global HMR system to rigorously satisfy all four online criteria; KV-cache online design and indirect smoothing strategy are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Standard benchmarks, in-the-wild videos, efficiency analysis, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear (four criteria); system design is well-structured; motivation-to-solution mapping is explicit.
- **Value**: ⭐⭐⭐⭐ — Direct engineering value for real-time applications such as AR/VR and robotic perception-action loops.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/human_understanding/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/human_understanding/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/human_understanding/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[CVPR 2026\] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)
- [\[CVPR 2026\] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement](phase-net_physics-grounded_harmonic_attention_system_for_efficient_remote_photop.md)

<!-- RELATED:END -->
