---
title: >-
  [Paper Note] LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR Motion Capture] This work proposes LiveHPS++, a robust single-LiDAR-based human motion capture method. By utilizing three components—a trajectory-guided body tracker, a noise-insensitive velocity predictor, and a kinematic-aware pose optimizer—it implicitly and explicitly models the dynamics and kinematics of human motion to achieve accurate and coherent global motion capture in complex noisy environments.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR Motion Capture"
  - "Noise Robustness"
  - "Motion Coherence"
  - "SMPL"
  - "Kinematic Optimization"
date: 2026-05-08
content_hash: 6e32e7e8e58de467
---

# LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment

**Conference**: ECCV 2024  
**arXiv**: [2407.09833](https://arxiv.org/abs/2407.09833)  
**Code**: [Yes (Project Page)](https://4dvlab.github.io/project_page/LiveHPS2.html)  
**Area**: Autonomous Driving  
**Keywords**: LiDAR Motion Capture, Noise Robustness, Motion Coherence, SMPL, Kinematic Optimization

## TL;DR

This work proposes LiveHPS++, a robust single-LiDAR-based human motion capture method. By utilizing three components—a trajectory-guided body tracker, a noise-insensitive velocity predictor, and a kinematic-aware pose optimizer—it implicitly and explicitly models the dynamics and kinematics of human motion to achieve accurate and coherent global motion capture in complex noisy environments.

## Background & Motivation

**Background**: Accurate capture of human motion in large-scale dynamic environments is crucial for downstream applications such as digital cinema, AR/VR, and robotics. Due to its long-range depth sensing capability and robustness to lighting conditions, LiDAR has become an ideal sensor for large outdoor motion capture. Recently, methods such as LiDARCap and MOVIN have demonstrated the feasibility of LiDAR-based motion capture.

**Limitations of Prior Work**:

**Applicable only to clean data**: Methods like LiDARCap and PointHPS rely on cleanly segmented human point cloud inputs. Their performance drops sharply in real-world scenes—when humans are in close contact or interacting with objects, the segmentation results of upstream perception algorithms often contain a large amount of noise.

**Equal treatment of noise features**: Although LiveHPS considers noise interference, its network treats features of real human points and noise points equally, which causes severe noise to significantly degrade estimation accuracy.

**Lack of global motion coherence**: Existing methods only consider interactions between joints while ignoring global kinematic information, resulting in jitter and incoherence in the predicted global poses and trajectories.

**Insufficient synthetic data**: Existing synthetic datasets only simulate random noise and random translations, failing to accurately reflect the complex noise patterns of real-world scenarios such as human-object interactions.

**Key Challenge**: How to maintain the accuracy and temporal coherence of motion capture results in real-world deployment under severe noise interference (from object occlusions, crowd proximity, and segmentation errors)?

**Key Insight**: Model human motion characteristics from both dynamic and kinematic dimensions: recover the global motion dynamics lost after normalization through trajectory information, and explicitly eliminate noise issues while strengthening temporal coherence through velocity prediction and kinematic optimization. Meanwhile, the NoiseMotion synthetic dataset is constructed to simulate complex human-object interaction noise scenarios.

## Method

### Overall Architecture

LiveHPS++ takes sequential noisy point clouds as input and outputs sequential SMPL parameters (pose $\theta$, shape $\beta$, translation $\mathbf{T}$). The pipeline consists of three core modules in series: (1) Trajectory-guided Body Tracker $\rightarrow$ predicts joint positions and translations; (2) Noise-insensitive Velocity Predictor $\rightarrow$ regresses the velocity of each joint to eliminate noise impact; (3) Kinematic-aware Pose Optimizer $\rightarrow$ utilizes velocity information to optimize pose accuracy and coherence. Finally, human body parameters are regressed via a SMPL solver.

The input point cloud is downsampled to a fixed number of $N_{input}=256$ points using farthest point sampling (FPS), and normalized by subtracting the mean position $\mathbf{Loc}(t)$. The SMPL model defines $N_J=24$ joints and $N_V=6890$ mesh vertices.

### Key Designs

1. **Trajectory-guided Body Tracker (TBT)**

   **Problem**: Under noisy data, traditional frame-wise normalization (subtracting the mean position of each frame) leads to severe fluctuations in point cloud distribution between adjacent frames, disrupting the spatial continuity of the trajectory. Meanwhile, sequence-wise normalization (subtracting the first frame's mean) preserves the physical trajectory but sacrifices accuracy.

   **Solution**: Design a specialized encoder to capture trajectory embedding, implicitly modeling the dynamic characteristics of human motion. A **vertex-trajectory-guided adaptive distillation mechanism** is adopted:
    - Guidance network: Takes GT vertex sampling $\mathbf{V}_{pc}^{GT}(t)$ and GT trajectory $\mathbf{Traj}^{GT}(t)$ as inputs.
    - Learning network: Takes input point cloud $\mathbf{PC}(t)$ and calculated trajectory $\mathbf{Traj}(t) = \mathbf{Loc}(t) - \mathbf{Loc}(1)$ as inputs.
    - Approximates the feature distribution of the guidance network with the learning network using a KL divergence distillation loss $\mathcal{L}_{distillation}$.
    - Adds a translation prediction branch.

   Loss function: $\mathcal{L}_{TBT} = \lambda_1 \mathcal{L}_{distillation} + \lambda_2 \mathcal{L}_{mse}(\mathbf{J}_{pc}) + \lambda_3 \mathcal{L}_{mse}(\hat{\mathbf{T}}_{pc})$, where $\lambda_1=10^3, \lambda_2=1, \lambda_3=1$. The guidance network is not needed during inference.

2. **Noise-insensitive Velocity Predictor (NVP)**

   **Problem**: TBT performs regression based on the parent-child joint structure of the skeleton. When a parent joint shifts due to noise, the error propagates and accumulates along the skeletal chain.

   **Mechanism**: Utilize a **cross-attention mechanism** to let each joint search for truly valuable point features in the original point cloud (rather than noise-polluted features), predicting the velocity of each global joint $\mathbf{K}(n)$ and translation velocity $\mathbf{K}_{ts}$ (with time window $L=32$):

    $$\mathcal{L}_{mse}(\mathbf{K}(n)) = \sum_n \|\mathbf{K}(n) - \mathbf{K}^{GT}(n)\|_2^2$$

   Where the velocity GT is defined as the joint position difference between adjacent frames: $\mathbf{K}^{GT}(n,t) = \mathbf{J}^{GT}(n,t+1) - \mathbf{J}^{GT}(n,t)$.

   **Design Motivation**: Velocity information reflects kinematic characteristics. The cross-attention mechanism allows the network to learn to distinguish between real human points and noise points, thereby eliminating the impact of noise at the feature level.

3. **Kinematic-aware Pose Optimizer (KPO)**

   **Mechanism**: Utilize predicted velocities and joint positions to generate candidate joints:

    $$\mathbf{J}_{cds}(n_i, t_i, t_j) = \mathbf{J}_{pc}(n_i, t_i) + \Delta t \sum_{t=t_i}^{t_j} \mathbf{K}(n_i, t)$$

   Each frame can generate $(L-1)$ candidate joints. A cross-attention architecture is used to establish connections between the candidate joints and the original inputs. This integrates short-term and long-term kinematic information for joint correction, outputting coherent and accurate global joints $\mathbf{J}_c(t)$ and translation $\mathbf{T}_c(t)$.

   **Design Motivation**: Short-term optimization enhances the coherence between adjacent frames but may introduce long-sequence jitter, while long-term optimization maintains global coherence but with large cumulative errors. KPO considers both to balance accuracy and coherence.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{TBT} + \mathcal{L}_{NVP} + \mathcal{L}_{KPO} + \mathcal{L}_{smpl}$

SMPL solver loss: $\mathcal{L}_{smpl} = \lambda_6 \mathcal{L}_{mse}(\mathbf{J}_{smpl}) + \lambda_7 \mathcal{L}_{mse}(\mathbf{V}_{smpl}) + \lambda_8 \mathcal{L}_{mse}(\theta) + \lambda_9 \mathcal{L}_{mse}(\beta)$, where $\lambda_6 = 100/N_J, \lambda_7=100/N_V, \lambda_8=1/5, \lambda_9=1$.

Training configuration: PyTorch 1.10.0 + CUDA 11.4, 200 epochs, batch size 64, sequence length 32, learning rate $10^{-3}$, 4×NVIDIA A40 GPUs. Training data includes FreeMotion, Sloper4D, and the self-constructed NoiseMotion dataset.

**NoiseMotion Dataset**: Constructed based on SURREAL (1,021,802 human motions) and ShapeNet (51,300 3D object models) to simulate dynamic/static noise from human-object interactions, which is far more diverse than FreeMotion-OBJ (which has fewer than 10 dynamic objects).

## Key Experimental Results

### Main Results

**Comparison with SOTA methods on multiple datasets (Table 1):**

| Method | NoiseMotion J/V Err(PST)↓ | NoiseMotion Jitter↓ | FreeMotion-OBJ J/V Err(PST)↓ | FreeMotion-OBJ Jitter↓|
|---|---|---|---|---|
| LiDARCap | 400.66/402.58 | 765.89 | 181.82/189.32 | 62.47 |
| LIP | 192.79/198.66 | 451.74 | 158.38/170.90 | 60.19 |
| LiveHPS | 74.70/83.84 | 68.65 | 146.78/158.00 | 117.79 |
| LiveHPS* | 561.49/611.40 | 884.24 | 133.82/146.12 | 100.82 |
| **Ours** | **58.53/64.51** | **59.35** | **128.60/136.94** | **30.96** |

**FreeMotion and Sloper4D datasets:**

| Method | FreeMotion J/V Err(PST)↓ | FreeMotion Jitter↓ | Sloper4D J/V Err(PST)↓ | Sloper4D Jitter↓ |
|---|---|---|---|---|
| LiveHPS | 130.41/141.08 | 85.38 | 88.35/95.85 | 73.56 |
| LiveHPS* | 119.22/128.55 | 86.07 | 77.73/85.83 | 97.41 |
| **Ours** | **112.13/120.39** | **33.16** | **76.98/81.67** | **59.97** |

On the most challenging FreeMotion-OBJ dataset, global vertex error is reduced by **6.28%** and jitter is reduced by **69.29%**. On NoiseMotion, global vertex error is reduced by **23.05%** and jitter is reduced by **13.54%**.

### Ablation Study

**Module ablation on FreeMotion-OBJ (Table 2):**

| Configuration | J/V Err(PS)↓ | Jitter↓ | Description |
|---|---|---|---|
| w/o TBT | 71.68/90.08 | 33.89 | Remove trajectory module $\rightarrow$ accuracy drops significantly |
| w/o NVP & KPO | 68.37/84.92 | 71.82 | Remove velocity + optimization $\rightarrow$ coherence collapses |
| Frame-wise normalization | 68.57/86.15 | 43.53 | Traditional frame-wise normalization |
| Sequence-wise normalization | 85.42/106.91 | 31.32 | Sequence-wise normalization $\rightarrow$ poor accuracy but coherent |
| Short-term optimizer | 64.04/79.27 | 55.50 | Short-term optimization only $\rightarrow$ long-sequence jitter |
| Long-term optimizer | 68.06/83.62 | 42.51 | Long-term optimization only $\rightarrow$ accumulated error |
| **Full LiveHPS++** | **58.11/72.55** | **30.96** | All modules combined achieve optimal results |

### Key Findings

- **TBT and KPO are complementary**: TBT mainly contributes to accuracy (w/o TBT accuracy drops significantly), while KPO mainly contributes to coherence (w/o NVP&KPO, jitter surges from 30.96 to 71.82).
- **Trajectory-guided normalization balances accuracy and coherence**: Frame-wise normalization yields high accuracy but lacks coherence, while sequence-wise normalization is coherent but inaccurate. TBT achieves the advantages of both.
- **Time window of 32 is optimal**: As the window increases from 1 to 32, jitter continuously decreases while accuracy slightly improves.
- **More NoiseMotion data, better performance**: Gradually increasing the ratio of synthetic data (0% $\rightarrow$ 100%) consistently improves performance, proving the value of the synthetic noisy data.
- **LiveHPS shows unstable generalization after training on NoiseMotion** (performance on FreeMotion/Sloper4D deteriorates), whereas LiveHPS++ achieves SOTA on all datasets.

## Highlights & Insights

1. **Dual modeling of dynamics and kinematics**: TBT implicitly models dynamic features (via trajectory embedding), while KPO explicitly models kinematic features (via velocity prediction and candidate joint generation). The two complement each other.
2. **Cross-attention denoising**: Refers to letting joints "actively search" for valuable point features in NVP, rather than passively receiving all points (including noise). The network learns to distinguish between human body points and noise points.
3. **Deep insight into normalization strategies**: Unveils the accuracy-coherence trade-off of frame-wise/sequence-wise normalization. Trajectory embedding serves as an elegant solution to resolve this conflict.
4. **NoiseMotion Dataset**: Utilizes SURREAL + ShapeNet to simulate real-world human-object interaction noise at scale, filling the gap in existing synthetic datasets.

## Limitations & Future Work

- Only targets single-person scenarios; multi-person motion capture remains unresolved.
- Relies on upstream perception algorithms to provide (noisy) human point cloud segmentation results; an end-to-end framework might be more robust.
- The object noise distribution in NoiseMotion may still exhibit a gap compared to real-world scenarios (synthetic-to-real domain gap).
- The time window is fixed at 32 frames; an adaptive window might offer more flexibility across different motion scenarios.
- Real-time performance metrics (inference speed) are not discussed.

## Related Work & Insights

- **LiveHPS**: The direct predecessor of this work. It proposed scene-level human pose estimation and a vertex-guided distillation mechanism but did not sufficiently address noise and coherence.
- **LiDARCap / LIP**: Early LiDAR motion capture methods using graph convolutions and sparse IMUs respectively, but only applicable in clean environments.
- **PointHPS**: A cascaded network that estimates poses from dense point clouds, which is unsuitable for sparse outdoor scenes.
- **Insight**: In noisy environments, kinematic information (velocity, acceleration) is a more robust feature than static pose. This concept can be extended to other sequential estimation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual dynamic-kinematic modeling framework is cleverly designed; trajectory-guided normalization and candidate joint generation are highly insightful designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation on 4 datasets (including the self-built NoiseMotion) with 5 metrics, rich quantitative/qualitative comparisons, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation of the problem, well-explained design rationales for each module, and in-depth ablation analysis.
- **Value**: ⭐⭐⭐⭐ — Significantly advances the practicality of LiDAR-based motion capture in real-world noisy environments, with a 69% reduction in jitter on FreeMotion-OBJ presenting strong application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults](../../AAAI2026/autonomous_driving/multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [\[ECCV 2024\] Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation](train_till_you_drop_towards_stable_and_robust_source-free_unsupervised_3d_domain.md)
- [\[CVPR 2026\] Bezier Degradation Modeling for LiDAR-based Human Motion Capture](../../CVPR2026/autonomous_driving/bezier_degradation_modeling_for_lidar-based_human_motion_capture.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)

</div>

<!-- RELATED:END -->
