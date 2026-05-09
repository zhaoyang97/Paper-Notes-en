---
title: >-
  [Paper Note] A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions
description: >-
  [ICCV 2025][Segmentation][physical motion restoration] This paper proposes a plug-and-play physical motion restoration approach that repairs artifact frames in video-based motion capture via a Mask-conditioned Motion Correction Module (MCM), and achieves physics-based simulation of high-difficulty in-the-wild motions via a Physics-based Motion Transfer Module (PTM) built on pretraining and test-time adaptation, substantially improving the physical plausibility of recovered motions.
tags:
  - ICCV 2025
  - Segmentation
  - physical motion restoration
  - motion imitation
  - test-time adaptation
  - motion correction
  - high-difficulty motions
date: 2026-05-08
content_hash: c06427a43b3468d4
---

# A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions

**Conference**: ICCV 2025
**arXiv**: [2412.17377](https://arxiv.org/abs/2412.17377)
**Code**: [Project Page](https://physicalmotionrestoration.github.io/)
**Area**: Image Segmentation
**Keywords**: physical motion restoration, motion imitation, test-time adaptation, motion correction, high-difficulty motions

## TL;DR

This paper proposes a plug-and-play physical motion restoration approach that repairs artifact frames in video-based motion capture via a Mask-conditioned Motion Correction Module (MCM), and achieves physics-based simulation of high-difficulty in-the-wild motions via a Physics-based Motion Transfer Module (PTM) built on pretraining and test-time adaptation, substantially improving the physical plausibility of recovered motions.

## Background & Motivation

Extracting physically plausible 3D human motion from monocular video is a critical task. Current video motion capture methods (e.g., GVHMR, TRAM) can rapidly recover 3D motions but lack dynamics modeling, leading to severe physical implausibilities:
- **Floating**: the body hovers above the ground
- **Foot sliding**: feet slide while in contact with the ground
- **Self-penetration**: body parts interpenetrate each other
- **Ground penetration**: the body passes through the ground

Existing physics-based motion imitation methods (e.g., PHC+, UHC) can improve the physical quality of everyday motions (walking, running, jumping), but remain ineffective for **high-difficulty motions** (gymnastics, martial arts, dance, etc.) due to two challenges:

**Challenge 1 — Artifact Reference Motions**: Under fast movements and extreme poses, video motion capture algorithms produce artifact frames (temporally incoherent poses), which—even if brief—cause physics simulation to fail.

**Challenge 2 — Intrinsic Imitation Complexity**: The long-tail distribution of high-difficulty motions, complex force control requirements, and catastrophic forgetting make it difficult for a single controller to generalize across diverse high-difficulty actions.

## Method

### Overall Architecture

Given the video motion capture result (reference motion) and the original video, the overall pipeline proceeds as follows:
1. **MCM (Mask-conditioned Correction Module)**: detects and corrects artifact motion frames
2. **PTM (Physics-based Motion Transfer Module)**: performs physics-based motion restoration on the corrected motion
    - If PTM fails initially, test-time adaptation (TTA) is triggered to update network parameters until success or a threshold is reached

The entire approach is designed as a plug-and-play module that can be directly integrated after any video motion capture method.

### Key Designs

#### 1. Mask-conditioned Motion Correction Module (MCM)

**Problem**: Blurry frames in high-difficulty motions cause motion capture algorithms to fail at localizing body parts, producing temporally incoherent artifact motions.

**Key Insight**: Segmentation methods inherently distinguish foreground from background and can define the approximate body region even in blurry frames. Artifact motions are temporally short and surrounded by rich motion context, making segmentation-guided interpolative replacement feasible.

**Mismatch Detection**:
- Project 3D positions to 2D coordinates → compute OKS similarity against 2D keypoints extracted by object detection
- Frames below the threshold are flagged as artifact motions
- Detection can also leverage the overlap between projected SMPL mesh and the human segmentation mask

**Motion Correction**:
- Human segmentation masks are obtained using SAM
- A pretrained ViT serves as the mask feature extractor, capturing rich human pose information
- Mask features and motion context are used as conditions for a diffusion model performing motion in-betweening
- 10% of training data is set to unconditional generation to enhance generalization
- Implemented using a GMD-based UNet architecture

#### 2. Physics-based Motion Transfer Module (PTM)

**Pretraining Stage**:
- An imitation controller is trained on four datasets: AMASS, Human3.6M, AIST++, and Motion-X (kungfu subset)
- Policy $\pi_{\text{PTM}}$ is optimized using PPO
- A PD controller computes joint torques: $\tau^i = k_p^i(a_t^i - x_t^i) - k_d^i q_t^i$
- An AMP discriminator provides style rewards
- Total reward = reconstruction reward + AMP style reward + energy penalty

**RL-based Test-Time Adaptation (TTA)**:

Core innovation — leveraging the trial-and-error nature of RL to perform limited-step parameter updates at test time, processing each motion sequence individually. The adaptation incorporates the following designs:

**Relative Reward**: Captured reference motions contain jitter and accumulated root-node errors; constructing a full reconstruction reward is harmful. Absolute root position is therefore ignored, and global orientation and translation are maintained through explicit rotation guidance and implicit velocity guidance:

$$r_t^g = e^{w_p\|rela(\hat{p}_t) - rela(p_t)\|} + e^{w_r\|\hat{\theta}_t \ominus \theta_t\|} + e^{w_v\|\hat{v}_t - v_t\|} + e^{w_\omega\|\hat{\omega}_t - \omega_t\|}$$

**Relative Early Termination**: Conventional strict termination conditions are easily triggered when facing low-quality reference motions. A termination condition based on the average relative distance across joints is designed to be more permissive for high-difficulty motions:

$$\mathcal{F}_t = \left(\frac{1}{J}\sum_{i=1}^J\|rela(\hat{p}_t^i) - rela(p_t^i)\| > d_{term}\right) \vee \mathcal{F}_t^h \vee \mathcal{F}_t^c$$

**Residual Force**: High-difficulty motions often involve aerial flips and jumps that rely on trampolines or mats; an auxiliary external force is introduced to compensate for missing environmental conditions in simulation.

### Loss & Training

- **MCM**: Denoising training based on a diffusion model; random motion segments are selected as generation targets
- **PTM pretraining**: PPO + AMP, strict reconstruction reward + early termination; approximately 2–3 days on a single A100
- **TTA inference**: everyday motions require <500 steps or no adaptation; high-difficulty motions require 2,000–4,000 steps

## Key Experimental Results

### Main Results

**Comparison on Public Datasets (Table 1 — selected key results)**:

| Dataset | Method | WA-MJE↓ | W-MJE↓ | MPJPE↓ | OKS↑ | GP↓ | Float↓ | FS↓ |
|--------|------|---------|--------|--------|------|------|--------|------|
| AIST++ | TRAM | 106.2 | 159.5 | 91.8 | 0.945 | 20.6 | 490.0 | 2.35 |
| AIST++ | TRAM+PhysPT | 136.8 | 218.3 | 93.6 | 0.903 | 4.08 | 22.7 | 2.07 |
| AIST++ | **TRAM+Ours** | **106.2** | **157.7** | 94.0 | **0.953** | **0.50** | **1.97** | **0.59** |
| AIST++ | GVHMR | 124.4 | 197.3 | 93.5 | 0.965 | 12.4 | 71.2 | 2.23 |
| AIST++ | **GVHMR+Ours** | **123.4** | **193.8** | 94.0 | 0.963 | **0.50** | **1.98** | **0.59** |
| Kungfu | TRAM | 113.4 | 209.7 | 84.6 | 0.925 | 4.32 | 40.9 | 2.57 |
| Kungfu | **TRAM+Ours** | **113.3** | **193.7** | **79.5** | **0.931** | **0.24** | **5.71** | **0.26** |
| EMDB | GVHMR | 109.1 | 274.9 | 252.2 | 0.954 | 82.3 | 510.3 | 0.69 |
| EMDB | **GVHMR+Ours** | **91.2** | **261.6** | 249.1 | 0.948 | **0.25** | **3.63** | **0.17** |

**In-the-Wild High-Difficulty Test Set (Table 2 — 206 videos)**:

| Method | OKS↑ | MPS↑ | GP↓ | Float↓ | FS↓ |
|------|------|------|------|--------|------|
| TRAM | 0.828 | 0.667 | 19.99 | 107.4 | 12.26 |
| TRAM+PhysPT | 0.730 | 0.645 | 7.88 | 39.4 | 6.01 |
| **TRAM+Ours** | **0.845** | **0.687** | **0.60** | **17.0** | **0.78** |
| GVHMR | 0.837 | 0.704 | 10.0 | 138.0 | 3.01 |
| **GVHMR+Ours** | **0.854** | **0.710** | **0.33** | **14.9** | **0.72** |

### Ablation Study

**PTM Physical Transfer Capability (Table 3 — Kungfu dataset)**:

| Method | SR↑ | MPJPEg↓ | MPJPE↓ | PA-MPJPE↓ |
|------|---------|---------|--------|-----------|
| UHC | 42.91% | 86.23 | 48.91 | 39.73 |
| PHC+ | 76.41% | 84.86 | 47.98 | 39.43 |
| **PTM (Ours)** | **98.16%** | **82.13** | **33.45** | **26.12** |

**Component Ablation of TTA (Table 4 — in-the-wild high-difficulty dataset)**:

| Early-Term | Res-F | TTA | Rela-Rwd | OKS↑ | MPS↑ | SR↑ |
|-----------|-------|-----|----------|------|------|-----|
| ✗ | ✗ | ✗ | ✗ | 0.811 | 0.673 | 37% |
| ✓ | ✗ | ✗ | ✗ | 0.784 | 0.652 | 52% |
| ✓ | ✓ | ✗ | ✗ | 0.823 | 0.673 | 61% |
| ✓ | ✓ | ✓ | ✗ | 0.850 | 0.706 | 85% |
| ✓ | ✓ | ✓ | ✓ | **0.853** | **0.710** | **87%** |

**MCM Configuration Ablation (Table 5)**:

| In-between | Mask Cond. | Kpts Cond. | Mask Det. | Kpts Det. | OKS↑ | SR↑ |
|-----------|---------|---------|---------|---------|------|-----|
| ✗ | - | - | - | - | 0.802 | 78% |
| ✓ | ✗ | ✓ | - | ✓ | 0.834 | 83% |
| ✓ | ✓ | ✗ | ✓ | ✗ | 0.845 | 87% |
| ✓ | ✓ | ✗ | ✓ | ✓ | **0.853** | **87%** |

### Key Findings

1. **Substantial improvement in physical plausibility**: ground penetration drops from 82.3 to 0.25 (EMDB-GVHMR), foot sliding from 12.26 to 0.78 (Wild-TRAM), and self-penetration decreases by more than 50%
2. **Original motion patterns are preserved**: joint errors in world/camera coordinates remain largely unchanged, and 2D similarity even improves
3. **TTA is the largest contributor**: success rate improves from 61% (without TTA) to 85% (with TTA enabled)
4. **Residual force is critical for aerial motions**: compensates for the absence of environmental support such as trampolines and mats
5. **Masks outperform keypoints as conditioning signals**: segmentation algorithms only distinguish foreground from background, whereas keypoint detection is prone to failure on blurry frames
6. **PhysPT degrades certain metrics**: its simplification of dynamics equations and lack of understanding of high-difficulty motion distributions cause world-coordinate errors to increase

## Highlights & Insights

1. **Plug-and-play design**: integrates into any video motion capture method without additional training, greatly lowering the barrier to adoption
2. **Elegance of the pretrain + adapt paradigm**: pretraining accumulates motion priors, while TTA adapts to specific motions, naturally addressing long-tail distribution and catastrophic forgetting
3. **Relative reward and termination design**: recognizing that video motion capture outputs are inherently unreliable, the method abandons absolute positional constraints in favor of relative ones—a highly practical design choice
4. **SAM segmentation + ViT features**: leverages the generalization capability of large foundation models to compensate for the failure of keypoint detection in high-difficulty scenarios
5. **Collection of 206 high-difficulty in-the-wild videos as a benchmark**: covering rhythmic gymnastics, taekwondo, yoga, and more, filling an evaluation gap in the field
6. **Thorough problem analysis**: clearly distinguishes "artifact motions" and "complex motion imitation" as two independent problems, addressed respectively by MCM and PTM

## Limitations & Future Work

1. **Single-person motions only**: cannot restore multi-person motions involving close interaction
2. TTA inference is slow: high-difficulty motions require 2,000–4,000 adaptation steps
3. The use of residual forces is not fully physically justified, as it essentially introduces external force assistance
4. MCM relies on the accuracy of SAM and object detection
5. Object interaction scenarios are not considered (human–object interaction data was excluded)
6. The relative reward ignores absolute root position, which may cause drift in long sequences

## Related Work & Insights

- **PHC/PHC+**: motion imitation controllers achieving 97%+ success rate on AMASS, but failing on high-difficulty noisy motions
- **PhysPT**: a pretrained physics-aware Transformer that lacks understanding of high-difficulty motions due to the absence of corresponding training data
- **SimPoE**: integrates kinematic and physical dynamics from images, but is sensitive to control parameters
- **AMP**: adversarial motion prior providing style rewards
- **GVHMR/TRAM**: state-of-the-art video motion capture methods; this work serves as a downstream post-processing stage for them

The pretrain + TTA paradigm proposed in this paper may inspire robot control tasks that require generalization to out-of-distribution samples.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of MCM (mask-guided correction) and PTM (pretraining + TTA) is original; the relative reward design demonstrates genuine insight
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, covering motions from everyday to high-difficulty
- **Experiments**: ⭐⭐⭐⭐⭐ — Three public datasets + 206 high-difficulty in-the-wild videos, comprehensive ablations, and thorough comparison with state-of-the-art
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear, with a clean one-to-one correspondence between two challenges and two modules
- **Overall**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)

</div>

<!-- RELATED:END -->
