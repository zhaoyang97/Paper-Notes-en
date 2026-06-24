---
title: >-
  [Paper Note] EBS-EKF: Accurate and High Frequency Event-based Star Tracking
description: >-
  [CVPR 2025][Event Camera] This paper proposes EBS-EKF, which models the circuit behavior of event cameras under low-light conditions to obtain intensity-dependent centroid offset correction, combined with a 3D Extended Kalman Filter for star tracking, achieving an order of magnitude higher accuracy than existing methods on real night-sky data.
tags:
  - "CVPR 2025"
  - "Event Camera"
  - "Star Tracking"
  - "Extended Kalman Filter"
  - "Low-Light Signal Modeling"
  - "Pose Estimation"
date: 2026-05-08
content_hash: e2f671ab05fde00e
---

# EBS-EKF: Accurate and High Frequency Event-based Star Tracking

**Conference**: CVPR 2025  
**arXiv**: [2503.20101](https://arxiv.org/abs/2503.20101)  
**Code**: Yes (open-source code and datasets)  
**Area**: Video Understanding / Event Camera  
**Keywords**: Event Camera, Star Tracking, Extended Kalman Filter, Low-Light Signal Modeling, Pose Estimation

## TL;DR
This paper proposes EBS-EKF, which models the circuit behavior of event cameras under low-light conditions to obtain intensity-dependent centroid offset correction, combined with a 3D Extended Kalman Filter for star tracking, achieving an order of magnitude higher accuracy than existing methods on real night-sky data.

## Background & Motivation

**Background**: Star trackers are standard sensors for spacecraft attitude determination. Traditional methods use Active Pixel Sensor (APS) cameras, achieving arcsecond-level accuracy. Recently, Event-Based Sensors (EBS) have emerged as a promising technology for star tracking due to their microsecond-level latency and low power consumption.

**Limitations of Prior Work**: APS trackers are limited by exposure time and frame processing overhead, typically updating at 2-10 Hz, which limits performance during rapid attitude maneuvers. Existing EBS star tracking methods (ICP, Hough, 2D-KF) have been evaluated solely on simulated data and use simplified signal models. The ICP method of Chin et al. is prone to noisy attitude estimation and drift; the 2D-KF method of Ng et al. only models 2D translation, ignoring the non-linear effects of roll.

**Key Challenge**: Existing EBS methods do not consider the circuit characteristics of event cameras under low-light conditions—dim stars cause intensity-dependent offsets in event centroids relative to the true centroids, which is ignored in traditional Gaussian signal models. Furthermore, 2D state estimation cannot properly handle the 3-degree-of-freedom (3D) rotational dynamics in real data.

**Goal**: (1) Model EBS low-light circuit behavior and propose intensity-dependent centroid correction; (2) Design a 3D EKF for high-frequency, high-accuracy attitude estimation; (3) Quantitatively evaluate on real night-sky data.

**Key Insight**: The authors discovered that EBS pixel bandwidth is proportional to photocurrent under low-light conditions, causing the event distribution of dim stars to lag behind their true positions. By analyzing circuit differential equations, this offset can be predicted and corrected.

**Core Idea**: Derive an intensity-dependent centroid offset correction function using a low-light EBS circuit model, combined with a 3D Extended Kalman Filter to achieve high-accuracy star tracking with millisecond-level updates.

## Method

### Overall Architecture
The input is the positive event stream of the event camera, and the output is the 3D attitude (quaternion + angular velocity) of the camera. The system consists of three steps: (1) Initialization: integrate events over a short time window, use DBSCAN clustering to find star clusters, and obtain the initial attitude via Astrometry.net catalog matching; (2) Event-by-event update: for each positive event received, the EKF predicts the current state, checks if the event is near a known star, and if so, applies the intensity-offset correction before executing the EKF update; (3) Output: 1 KHz attitude estimation.

### Key Designs

1. **Low-Light Event Likelihood Model**:

    - **Function**: Model the spatial distribution of event generation in event cameras under low-light conditions
    - **Mechanism**: Traditional models assume event likelihood $E(t) = d\log(I(t)/I_0 + 1)/dt$, but under low-light conditions, EBS circuit bandwidth is proportional to the photocurrent, turning the pixel voltage into a first-order low-pass filter. The authors derive the low-light event likelihood $E_{LL}(t) = 2\pi \cdot f_c(\tilde{I}(t)) \cdot [\tilde{I}(t) - V(t)]$, where $f_c \approx b + a\tilde{I}(t)$. Key finding: the centroid of positive events for dim stars lags behind the true position, and the offset $z(m_s)$ is a function of the star magnitude. Finally, this is approximated using a Gaussian distribution with an added offset correction term.
    - **Design Motivation**: This explains why existing methods fail to estimate centroids accurately—they neglect low-light circuit effects. The correction term improves centroid accuracy from ~3 pixels to ~0.4 pixels.

2. **3D Extended Kalman Filter (EKF)**:

    - **Function**: Recursively estimate the 3D rotation and angular velocity of the camera from asynchronous event streams
    - **Mechanism**: The state $\mathbb{S} = [\mathbf{q}, \omega]$ contains the quaternion and the 3D angular velocity. A constant-velocity prior $q_{t+1} = \exp(\Delta t \omega) \cdot q_t$ is used as the propagation model. The measurement model maps the 3D coordinates of stars to 2D pixels through quaternion rotation + pinhole projection, deriving the complete Jacobian matrix $\mathbf{F}$, process noise $\mathbf{Q}$, and measurement matrix $\mathbf{H}$. The boxplus operator is utilized to handle the composite manifold of $SO(3) \times \mathbb{R}^3$.
    - **Design Motivation**: 2D-KF only estimates translation and velocity while ignoring roll, leading to frequent drift on real 3D rotation data. The 3D EKF directly models the rotation group, eliminating the need for frequent absolute measurement resets.

3. **Intensity-Dependent Centroid Offset Correction**:

    - **Function**: Eliminate systematic bias in centroid estimation for bright and dim stars
    - **Mechanism**: The offset curve $z(m_s)$ is numerically solved from the low-light likelihood model and used as a look-up table function to correct event positions during the EKF update: $\mathcal{L}(e_i | \mathbf{x_0}, \mathbf{v}) \sim \mathcal{N}(\mathbf{x}_i - [\mathbf{x}_0 - \bar{\mathbf{v}} \cdot z(m_s)], \sigma_s^2 \mathbf{I})$. The correction parameters are calibrated using night-sky data ($I_0=1, a=20$ Hz/intensity, $b=2$ Hz) and validated for generalization on laboratory LCD data.
    - **Design Motivation**: The offset can reach 1-2 pixels (~30-60 arcseconds), and is particularly severe for bright stars (due to the "bow-shock" effect); failing to correct this leads to attitude jumps.

### Loss & Training
This method is a non-learning-based approach, centered around Bayesian estimation: $p(\mathbb{S}_i | \mathcal{E}_i) \propto \mathcal{L}(e_i | \mathbb{S}_i) \cdot p(\mathbb{S}_i | \mathbb{S}_{i-1})$, recursively solved via the EKF.

## Key Experimental Results

### Main Results: Real Night-Sky Tracking Accuracy (arcseconds, lower is better)

| Method | Vel. Sweep 1 (across/about) | Multipose 1 (across/about) | Tilt Ladder (across/about) |
|------|------|------|------|
| ICP | 3300/395 | 1600/2200 | 2300/6200 |
| Hough | 8600/87000 | 11000/57000 | 11000/65000 |
| 2D-KF | 229/1300 | 337/7300 | 485/9200 |
| EBS-EKF (Ours) | 25.8/60.3 | 57.4/52.8 | 49.1/64.5 |

### Ablation Study: Effect of Offset Correction

| Configuration | Typical Accuracy Change | Description |
|------|---------|------|
| EBS-EKF w/ offset | 25.8/60.3 (Vel.1) | Full model |
| EBS-EKF w/o offset | 27.0/83.1 (Vel.1) | After removing offset correction, the about direction error increases by ~20 arcseconds |
| Vel. Sweep 7 (with Vega) | 170.1/139.0 vs 172.1/322.0 | Offset correction reduces the about error by 180+ arcseconds in scenes containing the bright star Vega. |

### Key Findings
- The proposed method typically achieves accuracy within 100 arcseconds on the real night sky, whereas existing methods exhibit errors on the scale of thousands of arcseconds or even degrees.
- 2D-KF is the closest competitor but periodically drifts due to insufficient 2D modeling, requiring frequent absolute measurement resets.
- Offset correction yields significant improvements when bright stars are in the field of view (improving about error by ~180 arcseconds when Vega appears in Vel. Sweep 7).
- On a high-speed trajectory of 7.5°/s (exceeding the 3°/s limit of APS trackers), the proposed method still correctly reconstructs the trajectory with a total error of 80.4 arcseconds compared to 774.3 arcseconds for 2D-KF.

## Highlights & Insights
- **EBS Circuit Physics Modeling**: Deriving event likelihood from bottom-up circuit behavior, discovering the intensity-dependent offset, and proposing a correction is an excellent demonstration of "physics-to-algorithm" design, offering greater interpretability than purely data-driven methods.
- **First Real-Data Benchmark**: All previous EBS star tracking works were evaluated using LCD screen emulation. This paper, for the first time, quantitatively evaluates performance on the real night sky and provides a synchronized dataset, serving as a milestone for this field.
- **Transferable Ideas**: The intensity-dependent centroid offset correction can be extended to other low-light EBS application scenarios (e.g., dim point target detection in night-time autonomous driving).

## Limitations & Future Work
- **Ground-Based Data Limitations**: Data captured on Earth contains atmospheric refraction variations (especially near the horizon), which do not exist in space. The authors obtain indirect evaluation by comparing with APS trackers.
- **Global Parameter for Offset Correction**: The current offset curve is globally applied regardless of star velocity. Theoretically, star velocity also affects the offset, though the differences in experiments were minor.
- **Initialization Depends on Astrometry.net**: The initial attitude must be obtained via static integration + catalog matching, which may restrict performance in rapid-start scenarios.
- **Processing Only Positive Events**: Negative events for dim stars have high latency and are directly discarded. Future work could explore utilizing negative events to further enhance performance.

## Related Work & Insights
- **vs Ng et al. (2D-KF)**: 2D-KF models with 2D translation + velocity, ignoring roll and 3D non-linear effects, and requires frequent absolute measurements; our 3D EKF directly models the rotation group, requiring initialization only once.
- **vs Chin et al. (ICP)**: ICP integrates events into frames and registers them, which is highly sensitive to noise and prone to drift; our method performs asynchronous event-by-event updates, offering higher update rates.
- **vs APS Star Trackers**: APS achieves 5 arcseconds across / 55 arcseconds about but updates at only 2 Hz and is limited to 3°/s. Our EBS method achieves comparable accuracy at a 1 KHz update rate and tolerates high-speed rotation up to 7.5°/s.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Deriving the centroid correction model from EBS circuit physics, combined with a 3D EKF; the idea is clear and highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 real night-sky trajectories + LCD laboratory ground truth, comprehensively compared against 3 baseline methods with thorough ablation analysis.
- Reproducibility: ⭐⭐⭐⭐⭐ The code and datasets are fully open-source, and experimental details are comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic from physics modeling and algorithmic derivation to experimental evaluation, with comprehensive supplementary materials.
- Value: ⭐⭐⭐⭐ Significant advancement for the EBS star tracking field by providing the first real-data benchmark; however, the application scenario is limited to the aerospace field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)
- [\[CVPR 2025\] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2025\] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [\[CVPR 2025\] Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling](radio_frequency_ray_tracing_with_neural_object_representation_for_enhanced_rf_mo.md)

</div>

<!-- RELATED:END -->
