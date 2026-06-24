---
title: >-
  [Paper Note] Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation
description: >-
  [ECCV 2024][Video Understanding][Event Camera] This paper proposes a self-supervised method that integrates a non-linear motion prior (parametric trajectory function) into the contrast maximization framework for dense continuous-time motion estimation with event cameras, improving the zero-shot performance of synthetic-data pre-trained models by 29% on the real-world EVIMO2 dataset.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Event Camera"
  - "contrast maximization"
  - "motion estimation"
  - "self-supervised learning"
  - "optical flow"
date: 2026-05-08
content_hash: 308902d5b32ca3c1
---

# Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation

**Conference**: ECCV 2024  
**arXiv**: [2407.10802](https://arxiv.org/abs/2407.10802)  
**Code**: [GitHub](https://github.com/tub-rip/MotionPriorCMax)  
**Area**: Video Understanding  
**Keywords**: Event Camera, contrast maximization, motion estimation, self-supervised learning, optical flow

## TL;DR

This paper proposes a self-supervised method that integrates a non-linear motion prior (parametric trajectory function) into the contrast maximization framework for dense continuous-time motion estimation with event cameras, improving the zero-shot performance of synthetic-data pre-trained models by 29% on the real-world EVIMO2 dataset.

## Background & Motivation

**Background**: Optical flow estimation and point tracking are core tasks in low-level vision. Recently, frame-based methods have made significant progress relying on large-scale synthetic datasets. As a novel vision sensor, event cameras feature high speed, high dynamic range, and low power consumption, making them naturally suited for motion perception tasks.

**Limitations of Prior Work**: The event camera domain faces two main constraints: (a) a lack of large-scale real labeled data, and immature event simulators leading to a significant sim-to-real gap for models trained on synthetic data; (b) existing contrast-loss-based self-supervised methods are limited to low-degree-of-freedom (e.g., feature tracking, ego-motion estimation) or short-term ($\le 0.1$s) linear optical flow problems, failing to handle long-term non-linear complex motion.

**Key Challenge**: Long-term dense motion estimation requires handling non-linear trajectories, but the contrast maximization framework previously only supported linear motion models. Additionally, the association between events and trajectories is a high-dimensional assignment problem, which is difficult to solve efficiently and differentiably.

**Goal**: Long-term ($\ge 0.3$s) dense non-linear event motion estimation, while reducing the domain adaptation gap.

**Key Insight**: A two-stage strategy: supervised pre-training on synthetic data, followed by fine-tuning on real data using the proposed self-supervised contrast loss.

**Core Idea**: Introduce a continuous-time parametric trajectory prior (e.g., Bézier curves) into the contrast maximization framework, and efficiently and differentiably associate events with dense non-linear trajectories via KNN soft assignment and coarse-grained spatio-temporal displacement fields.

## Method

### Overall Architecture

The overall pipeline is divided into a prediction module and a loss module. Events within the time interval $[t_s, t_e]$ are first voxelized and fed into a neural network (Bflow architecture or UNet), which predicts continuous-time trajectory coefficients for each pixel location. The predicted trajectories and the original events are fed together into the self-supervised loss module: a coarse-grained spatio-temporal displacement field is first interpolated, events are then warped according to the look-up-table displacements, and finally, an IWE (Image of Warped Events) is constructed, using the gradient magnitude of the IWE as the training loss.

### Key Designs

1. **Trajectory Motion Prior**:

    - **Function**: Predict a continuous-time trajectory $\mathbf{q}_n(t)$ represented by a weighted combination of parametric basis functions for each pixel.
    - **Mechanism**: Represent the trajectory as a weighted sum of basis functions $\mathbf{q}_n(t) = \sum_{j=1}^{N_c} g_j(t) \mathbf{p}_{n,j}$, where $g_j(t)$ is a shared temporal basis function, and $\mathbf{p}_{n,j}$ are pixel-level control points. The paper explores polynomial bases $g_j(t)=t^j$, Bézier curve bases $g_j(t)=\binom{N_c}{j}(1-t)^{N_c-j}t^j$, and learnable bases.
    - **Design Motivation**: The non-linear trajectory prior strikes a balance between motion versatility and regularization. Compared to linear optical flow models, it can handle long-term complex motion, while the smoothness of trajectories naturally provides temporal regularization to prevent event collapse.

2. **KNN Soft Assignment**:

    - **Function**: Efficiently and differentiably compute the displacement $\Delta\mathbf{x}_k$ for each event.
    - **Mechanism**: First relax the problem into interpolation on a coarse-grained spatio-temporal displacement field of shape $[N_\text{bins}, h/4, w/4]$. For each temporal bin center, KNN search is executed using the KeOps symbolic matrix framework to find the $N_\text{traj}$ nearest neighbor trajectories, computing the displacement as the average displacement of neighbor trajectories: $\Delta\mathbf{x}_k = \frac{1}{N_\text{traj}} \sum_{n=1}^{N_\text{traj}} [\mathbf{q}_n(t_\text{ref}) - \mathbf{q}_n(t_k)]$.
    - **Design Motivation**: Directly computing associations for all event-trajectory pairs is computationally infeasible (the number of events can reach tens of millions). Through coarse-grained lookup tables and KNN interpolation, the problem is dimensionally reduced from 3D to 2D, which saves memory while maintaining differentiability.

3. **Random Reference Time + Regularization**:

    - **Function**: Randomly sample reference times $t_\text{ref} \sim \mathcal{U}(0,1)$ during training and impose spatial smoothness regularization.
    - **Mechanism**: The training loss is $\mathcal{L} = 1/G + \lambda R$, where $G$ is the IWE gradient magnitude and $R$ is the L1 norm of the spatial gradient of the displacement field. A different reference time is randomly selected at each iteration.
    - **Design Motivation**: (a) Random reference times require the IWE to be sharp at any moment, enhancing the regularization effect, with memory overhead growing only linearly (vs. previous methods using a fixed set of multiple reference times); (b) the spatial smoothness term encourages consistency of neighboring trajectories, further suppressing event collapse.

### Loss & Training

- **Pre-training**: Supervised pre-training (L1 loss) on the MultiFlow synthetic dataset for 50 epochs with a batch size of 10.
- **Self-supervised fine-tuning**: Fine-tuned on EVIMO2 using contrast loss for 15 epochs, with batch size 6, learning rate $10^{-4}$, using the AdamW optimizer.
- Spatial smoothness weight $\lambda=0.003$, number of KNN neighbors $N_\text{traj}=32$.
- For the optical flow task, a UNet architecture was additionally trained on DSEC, demonstrating the versatility of the method.

## Key Experimental Results

### Main Results

**EVIMO2 Continuous Optical Flow Dataset**:

| Method | TEPE ↓ | TAE ↓ | %Out ↓ |
|------|--------|-------|--------|
| Paredes et al. (Self-supervised) | 21.69 | 51.91 | 0.634 |
| E-RAFT (Linear) | 19.38 | 74.52 | 0.656 |
| BFlow (Zero-shot/OOD) | 8.63 | 19.94 | 0.363 |
| **Ours (Self-supervised FT)** | **6.14** | **16.98** | **0.254** |
| BFlow (In-domain Supervised) | 3.38 | 11.68 | 0.166 |

**DSEC Optical Flow Benchmark (Comparison of self-supervised methods)**:

| Method | Inference Time (ms) | EPE ↓ | AE ↓ | %Out ↓ |
|------|-------------|-------|------|--------|
| E-RAFT (Supervised) | 46.3 | 0.788 | 10.56 | 2.684 |
| Ours (Self-supervised) | ~9 | Best | AE Gain 19% | %Out Gain 14% |

### Ablation Study

**Impact of Different Motion Priors (EVIMO2)**:

| Motion Prior | TEPE ↓ | TAE ↓ | %Out ↓ |
|---------|--------|-------|--------|
| Polynomial Basis (SSL) | 6.78 | 19.76 | 0.272 |
| Learnable Basis (SSL) | 7.46 | 19.78 | 0.284 |
| **Bézier Curve (SSL)** | **6.14** | **16.98** | **0.254** |
| Polynomial Basis (OOD) | 9.36 | 20.88 | 0.363 |
| Bézier Curve (OOD) | 8.63 | 19.94 | 0.363 |

### Key Findings

- Self-supervised fine-tuning improves zero-shot performance by approximately **29%** (TEPE decreased from 8.63 to 6.14), effectively bridging the sim-to-real gap.
- The Bézier curve performs the best among the three motion priors ($N_c=10$).
- For long-term prediction (>200ms), the improvement from self-supervised fine-tuning is particularly significant.
- The inference speed on DSEC is **5 times** faster than previous self-supervised methods.

## Highlights & Insights

- **Integrating motion priors into the CM framework** is an elegant design: non-linear trajectories adapt to complex motion, while their smoothness naturally provides regularization, subtly solving the event collapse issue.
- The KNN scheme implemented using the KeOps symbolic matrix framework resolves the scalability issues of large-scale event-trajectory association, representing a practical engineering contribution.
- The construction of the EVIMO2 continuous optical flow dataset is itself a contribution, providing a new evaluation benchmark for long-term motion estimation with event cameras.

## Limitations & Future Work

- A significant gap still exists between self-supervised methods and in-domain supervised methods (TEPE 6.14 vs. 3.38), indicating that the information provided by the self-supervised loss signal is limited.
- Handling of occlusions and independently moving objects may not be robust enough.
- KeOps KNN computation may still become a bottleneck at very large resolutions.
- Farther advanced trajectory representations (such as B-splines and neural implicit representations) have not yet been explored.

## Related Work & Insights

- **vs. Bflow [27]**: Bflow relies on supervised training using synthetic data and suffers from a sim-to-real gap on real data. This paper significantly narrows this gap through self-supervised fine-tuning.
- **vs. Shiba et al. [57]**: Limited to linear motion models and fixed reference times, whereas this paper extends to non-linear trajectories and random reference times.
- **vs. Paredes et al. [47]**: Uses piecewise linear optical flow stitching to approximate non-linear motion, yielding far inferior performance compared to explicit non-linear trajectory modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of introducing motion priors into the CM framework is novel, and the KNN association scheme is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two tasks (trajectory estimation, optical flow) and multiple datasets, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, complete formula derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides an effective tool for mitigating the sim-to-real gap in event camera motion estimation, offering practical value to the event camera community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation](../../CVPR2026/video_understanding/from_contrast_to_consistency_rethinking_event-based_continuous-time_optical_flow.md)
- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](../../ICCV2025/video_understanding/simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ECCV 2024\] EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere](egoposer_robust_real-time_egocentric_pose_estimation_from_sparse_and_intermitten.md)

</div>

<!-- RELATED:END -->
