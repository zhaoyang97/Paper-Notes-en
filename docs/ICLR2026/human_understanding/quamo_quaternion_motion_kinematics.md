---
title: >-
  [Paper Note] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture
description: >-
  [ICLR 2026][Human Understanding][Quaternion kinematics] QuaMo proposes a 3D human kinematics capture method based on quaternion differential equations (QDE). By solving kinematic equations under the unit quaternion spher…
tags:
  - "ICLR 2026"
  - "Human Understanding"
  - "Quaternion kinematics"
  - "3D human motion capture"
  - "state space model"
  - "PD controller"
  - "acceleration augmentation"
date: 2026-05-08
content_hash: 2888cfa811138e35
---

# QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture

**Conference**: ICLR 2026
**arXiv**: [2601.19580](https://arxiv.org/abs/2601.19580)  
**Code**: Available (mentioned in the paper; specific link to be released)  
**Area**: Human Understanding / 3D Vision
**Keywords**: Quaternion kinematics, 3D human motion capture, state space model, PD controller, acceleration augmentation

## TL;DR

QuaMo proposes a 3D human kinematics capture method based on quaternion differential equations (QDE). By solving kinematic equations under the unit quaternion sphere constraint $\mathcal{S}^3$ and introducing a second-order acceleration-augmented meta-PD controller, the method achieves discontinuity-free, low-jitter online real-time human motion estimation, surpassing state-of-the-art methods on Human3.6M and several other benchmarks.

## Background & Motivation

**Background**: Monocular 3D human motion capture remains highly challenging in computer vision. While conventional 3D pose estimation methods (e.g., PoseFormer, HMR2.0) achieve high accuracy on distance metrics, they disregard temporal consistency across consecutive frames, resulting in jitter and unnatural artifacts. Recent kinematic methods enforce temporal consistency by incorporating physical models of velocity and acceleration.

**Limitations of Prior Work**: Existing kinematic methods (e.g., SimPoE, HuMoR, DnD) predominantly represent joint rotations using Euler angles. Although intuitive, Euler angles suffer from two fundamental issues: (1) singularities (gimbal lock) and (2) discontinuities (wraparound at 0 and $2\pi$), causing joints to erroneously reverse direction near angular boundaries. This makes motion reconstruction highly unstable—particularly in online settings where retrospective optimization is infeasible.

**Key Challenge**: Quaternions are naturally free of discontinuities and can represent all 3D rotations; however, their derivatives cannot be approximated via simple finite differences due to the rotation constraint, requiring special operations based on the Hamilton product. Furthermore, existing PD controllers respond insufficiently to rapid motion changes.

**Goal**: (1) Replace Euler angles with quaternions as the joint rotation representation; (2) strictly solve the QDE under the unit quaternion sphere constraint $\mathcal{S}^3$; (3) design an adaptive acceleration augmentation mechanism to handle fast motion changes.

**Key Insight**: Quaternions have been widely adopted for attitude control in aerospace and robotics, yet their systematic application to human kinematics remains largely unexplored. This work imports quaternion differential equations and constrained integration methods from aerospace into human motion capture.

**Core Idea**: Quaternions combined with the Hamilton product enable exact solutions to the rotational differential equation (eliminating Euler angle discontinuities); a second-order finite difference of reference poses adaptively augments the PD control signal to improve tracking of fast motions.

## Method

### Overall Architecture

**Input**: Per-frame 3D pose estimates (noisy) from TRACE or HMR2.0. The state-space model comprises two parallel streams: (1) **Angular velocity ODE stream** — a meta-PD controller with acceleration augmentation and a bias term estimates the angular acceleration $\dot{\omega}_t$, which is then integrated via Euler integration to obtain $\omega_{t+\Delta t}$; (2) **Quaternion QDE stream** — the Hamilton product performs exact integration on $\mathcal{S}^3$ to yield the next-frame pose $q_{t+\Delta t}$. The output pose drives the SMPL model to generate the human mesh.

### Key Designs

1. **Quaternion Differential Equation (QDE) and Constrained Integration**

    - **Function**: Accurately solves rotational updates on the unit quaternion sphere, eliminating Euler angle discontinuities.
    - **Mechanism**: Given angular velocity $\omega \in \mathbb{R}^3$, the quaternion velocity is defined as $\dot{q} = \frac{1}{2}\Omega(\omega)q$, where $\Omega(\omega)$ is a $4\times4$ skew-symmetric matrix. Assuming $\omega$ is constant over $\Delta t$, the exact solution is $q_{t+\Delta t} = \exp\!\left(\frac{\Delta t}{2}\Omega(\omega_{t+\Delta t})\right)q_t = q_\omega \otimes q_t$, where $\otimes$ denotes the Hamilton product. This guarantees that $q_{t+\Delta t}$ always lies on $\mathcal{S}^3$, requiring no post-hoc renormalization.
    - **Design Motivation**: Conventional Euler integration $q_{t+\Delta t} \approx q_t + \dot{q}_t \Delta t$ violates the unit quaternion constraint and accumulates errors. Matrix-exponential integration guarantees an exact solution respecting the Lie group constraint.

2. **Meta-PD Controller with Second-Order Acceleration Augmentation**

    - **Function**: Adaptively modulates the control signal strength based on the rate of change of the reference pose.
    - **Mechanism**: The angular acceleration is computed as $\dot{\omega}_t = \kappa_P \operatorname{vec}(\hat{q}_t \otimes q_t^*) - \kappa_D \omega_t + b_t + \kappa_A\!\left(\operatorname{vec}(\hat{q}_t \otimes \hat{q}_{t-\Delta t}^*) - \operatorname{vec}(\hat{q}_{t-\Delta t} \otimes \hat{q}_{t-2\Delta t}^*)\right)$. The first two terms constitute classical PD control (proportional tracking error; derivative jitter suppression), and $b_t$ is a data-driven bias. The key innovation is the acceleration augmentation term (last term), which computes the second-order quaternion finite difference of the reference pose: it increases the control force during rapid motion and naturally diminishes as the target is approached.
    - **Design Motivation**: Standard PD control lags behind sudden motion changes. The acceleration augmentation term leverages the "acceleration" of the reference signal to anticipate motion trends, accelerating tracking while reducing overshoot. All gains $\kappa_P, \kappa_D, \kappa_A$ and the bias $b_t$ are predicted by a ControlNet from the current state.

3. **InitNet Initialization and Global Translation**

    - **Function**: Provides a reasonable initial state for the kinematic system and tracks global displacement.
    - **Mechanism**: InitNet predicts the initial state $q_0, \omega_0$ and a learnable shape parameter $\beta_{fix}$ from the first two reference frames $\hat{q}_{0:1}$ and the initial shape parameter $\beta_0$. Global translation is estimated with a PD controller and Euler integration: $r_{t+\Delta t} = r_t + \left(v_t + (\kappa_P(\hat{r}_t - r_t) - \kappa_D v_t)\Delta t\right)\Delta t$.
    - **Design Motivation**: An online system has no historical state at the first frame, necessitating a dedicated initialization network. The shape parameter is fixed across the sequence but remains fine-tunable.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{local} + \mathcal{L}_{global} + \lambda \mathcal{L}_{beta}$. $\mathcal{L}_{local}$ is an L1 reconstruction loss over per-frame 3D keypoints and root translation. $\mathcal{L}_{global}$ is an L1 consistency loss on second-order finite differences (acceleration), enforcing global motion smoothness. $\mathcal{L}_{beta}$ regularizes the shape parameter. Training strategy: the first 5 epochs perform per-frame updates (global loss disabled, low learning rate); thereafter, the global loss is enabled and training proceeds on 100-frame subsequences. Total: 35 epochs, batch size 64.

## Key Experimental Results

### Main Results

| Method | MPJPE ↓ | P-MPJPE ↓ | Accel ↓ | G-MPJPE ↓ | FS ↓ | Online |
|--------|---------|-----------|---------|-----------|------|--------|
| HMR2.0 | 46.7 | 30.7 | 9.1 | 97.2 | 11.5 | ✓ |
| TRACE | 56.1 | 39.4 | 18.9 | 143.0 | 80.3 | ✓ |
| DnD | - | - | - | - | - | ✗ |
| PhysPT | 52.7 | 36.7 | 2.5 | 335.7 | - | ✗ |
| **QuaMo** | **Best** | **Best** | **Low** | **Best** | **Lowest** | ✓ |

(QuaMo achieves the best local and global metrics among online methods on Human3.6M.)

### Ablation Study

| Configuration | MPJPE | Accel | Notes |
|---------------|-------|-------|-------|
| Full QuaMo | Best | Best | Quaternion + constrained integration + acceleration augmentation |
| Euler angles instead of quaternions | ↑ | ↑ | Errors from discontinuities |
| Euler integration instead of constrained integration | ↑ | — | Violates $\mathcal{S}^3$ constraint |
| Without acceleration augmentation | ↑ | ↑ | Degraded tracking of fast motions |
| Without global consistency loss | — | ↑ | Reduced motion smoothness |

### Key Findings

- Quaternion representations outperform Euler angles on all metrics, with particularly significant differences in global motion metrics and foot sliding (FS)—confirming the detrimental effect of Euler angle discontinuities on kinematic estimation.
- The ablation of constrained integration (matrix exponential) vs. Euler integration clearly demonstrates that strictly satisfying the $\mathcal{S}^3$ constraint yields accuracy improvements.
- Acceleration augmentation contributes most on fast-motion sequences and has smaller impact on slow-motion sequences—consistent with the intended adaptive behavior.
- QuaMo maintains advantages on the more diverse Fit3D, SportsPose, and AIST datasets, demonstrating strong generalization.

## Highlights & Insights

- **Quaternion kinematics** is applied systematically to human motion capture for the first time. Although quaternions have been mature in aerospace and robotics, they have been largely overlooked in the human motion domain. This work fills that gap and demonstrates their superiority.
- **The adaptive nature of acceleration augmentation** is elegant: no auxiliary network is needed to judge whether a motion is "fast." The second-order finite difference is inherently adaptive—it increases for fast motions and diminishes as the target is approached.
- **The ablation of constrained vs. Euler integration** delivers a clear lesson: integration on a Lie group must respect the manifold constraint; approximate methods accumulate errors over long sequences.

## Limitations & Future Work

- As an online method, the system relies on single-step inputs and does not exploit future frame information, placing an upper bound on accuracy below that of offline methods.
- The reference poses $\hat{q}$ are noisy estimates from TRACE or HMR2.0; QuaMo is susceptible to degradation when input estimation quality is poor.
- Occlusion and multi-person scenarios are not addressed.
- Global translation still uses simple Euler integration rather than a quaternion-based approach, leaving room for potential improvement.

## Related Work & Insights

- **vs. DnD (Li et al., 2022)**: DnD also employs a PD controller but requires full-sequence attention and future frame information, making it not truly online. QuaMo is a purely online method and additionally introduces quaternions and acceleration augmentation.
- **vs. OSDCap (Le et al., 2024)**: OSDCap uses a learnable Kalman filter that re-introduces noisy inputs, potentially undermining temporal consistency. QuaMo does not reintroduce noise.
- **vs. PhysPT (Zhang et al., 2024)**: PhysPT uses a Transformer autoencoder processing the full sequence, making it an offline method. QuaMo runs online yet remains competitive.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study of quaternion kinematics for human motion; the acceleration augmentation design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, multiple input sources, detailed ablations, and results reported over 5 random seeds.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear; method descriptions are thorough.
- **Value**: ⭐⭐⭐⭐ Practically significant for online motion capture; the quaternion framework is transferable to other kinematic problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](../../AAAI2026/human_understanding/kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)
- [\[CVPR 2026\] HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](../../CVPR2026/human_understanding/hum4d_markerless_motion_capture.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[ICCV 2025\] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning](../../ICCV2025/human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)

</div>

<!-- RELATED:END -->
