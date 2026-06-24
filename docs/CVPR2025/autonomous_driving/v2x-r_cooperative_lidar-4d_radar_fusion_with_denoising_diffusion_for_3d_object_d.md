---
title: >-
  [Paper Note] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection
description: >-
  [CVPR 2025][Autonomous Driving][V2X Cooperative Perception] This paper constructs V2X-R, the first V2X simulation dataset containing three modalities (LiDAR, camera, and 4D radar). It proposes a cooperative LiDAR-4D radar fusion pipeline and a Multi-modal Denoising Diffusion (MDD) module. By leveraging weather-robust 4D radar features to guide a diffusion model in denoising noisy LiDAR features, the approach improves detection performance by up to 5.73%/6.70% in foggy/snowy c…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "V2X Cooperative Perception"
  - "4D Millimeter-Wave Radar"
  - "LiDAR Fusion"
  - "Denoising Diffusion"
  - "Adverse Weather"
date: 2026-05-08
content_hash: 151634596f58d2f1
---

# V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2411.08402](https://arxiv.org/abs/2411.08402)  
**Code**: [https://github.com/ylwhxht/V2X-R](https://github.com/ylwhxht/V2X-R)  
**Area**: Autonomous Driving  
**Keywords**: V2X Cooperative Perception, 4D Millimeter-Wave Radar, LiDAR Fusion, Denoising Diffusion, Adverse Weather  

## TL;DR

This paper constructs V2X-R, the first V2X simulation dataset containing three modalities (LiDAR, camera, and 4D radar). It proposes a cooperative LiDAR-4D radar fusion pipeline and a Multi-modal Denoising Diffusion (MDD) module. By leveraging weather-robust 4D radar features to guide a diffusion model in denoising noisy LiDAR features, the approach improves detection performance by up to 5.73%/6.70% in foggy/snowy conditions with almost no impact on normal weather performance.

## Background & Motivation

**Background**: V2X (Vehicle-to-Everything) perception effectively expands the sensing range and resolves occlusion issues through information sharing among multiple agents. Current research mainly focuses on single-modality LiDAR and dual-modality LiDAR-camera fusion. LiDAR provides precise 3D geometric information, and cameras provide fine-grained semantics; their fusion has achieved favorable results.

**Limitations of Prior Work**: Both LiDAR and cameras are extremely sensitive to adverse weather—fog, snow, and rain severely degrade LiDAR point cloud quality (increasing scattering noise and shortening detection range) and camera image quality (reducing visibility). Although 4D millimeter-wave radar possesses all-weather sensing capabilities and Doppler velocity information, existing V2X datasets (such as OPV2V and V2X-Sim) do not contain 4D radar data, leaving cooperative LiDAR-4D radar fusion completely unexplored.

**Key Challenge**: While multi-agent communication solves the problem of shortened single-agent detection range in adverse weather (by sharing data from other agents), it also amplifies noise—noisy LiDAR features from multiple agents accumulate, resulting in denser noise. Traditional point-level denoising is time-consuming and incompatible with feature-level fusion strategies, while direct feature-level denoising is difficult to fit due to the complex and highly variable distribution of weather noise.

**Goal**: (1) Construct the first V2X dataset containing 4D radar; (2) Propose a cooperative LiDAR-4D radar fusion pipeline; (3) Design an MDD module to denoise LiDAR features using the weather robustness of 4D radar.

**Key Insight**: Millimeter-wave signals from 4D radar can easily penetrate particles in fog and snow, offering inherent resistance to adverse weather. Using 4D radar features as "clean conditions" to guide the diffusion model in denoising LiDAR features—first using a diffusion process to reparameterize the complex weather noise distribution into a Gaussian distribution (which is easier to fit), and then utilizing 4D radar features as conditions for multi-step denoising.

**Core Idea**: Transforming the complex distribution of weather noise into an easily fittable Gaussian distribution through Gaussian reparameterization, and using weather-robust 4D radar BEV features as conditions to guide a U-Net denoiser in multi-step denoising of noisy LiDAR BEV features.

## Method

### Overall Architecture

The V2X-R system includes three types of agents (ego vehicle E, cooperative vehicle C, and roadside unit I), each collecting LiDAR and 4D radar point clouds. The fusion pipeline consists of four stages: (1) **Per-agent encoding**: Shared encoders extract BEV features $\mathcal{F}_j^i$ for each modality of each agent. (2) **Agent fusion**: Multi-agent features of the same modality are fused into multi-agent features $\mathcal{F}_\mathcal{A}^i$ via an agent-fusion module (e.g., self-attention). (3) **Modality fusion**: The MDD module first denoises the noisy LiDAR features to obtain $\tilde{\mathcal{F}_\mathcal{A}^L}$, which are then fused with 4D radar features. (4) **Detection head**: Predicts 3D bounding boxes.

### Key Designs

1. **V2X-R Dataset**:
    - **Function**: Provide the first V2X cooperative perception dataset containing LiDAR, camera, and 4D radar.
    - **Mechanism**: Constructed based on the CARLA and OpenCDA simulation platforms. Each agent is equipped with a 64-line LiDAR (120m range), 4 RGB cameras, and 1 4D radar (150m range, 30° vertical FOV). The dataset contains 12,079 scenes, 37,727 frames of LiDAR/4D radar point clouds, 150,908 images, and 170,859 3D vehicle bounding box annotations. LiDAR degradation in foggy and snowy weather is simulated using physical reflection and geometric optics methods. Analysis shows that after multi-agent cooperation, the instance coverage and point count of 4D radar are significantly improved, especially at medium-to-long distances.
    - **Design Motivation**: The lack of V2X datasets containing 4D radar is the fundamental bottleneck in this research field.

2. **Multi-modal Denoising Diffusion (MDD) Module**:
    - **Function**: Denoise noisy LiDAR BEV features under the guidance of weather-robust 4D radar features.
    - **Mechanism**: Consists of two steps. **Diffusion process** (reparameterization): Apply a $\mathcal{T}$-step Gaussian diffusion $\mathcal{F}_\mathcal{T} = \sqrt{\bar{\alpha}_\mathcal{T}} \mathcal{F}_{init} + \sqrt{1-\bar{\alpha}_\mathcal{T}} \epsilon$ to the agent-fused LiDAR feature $\mathcal{F}_{init}$, transforming the original complex weather noise distribution $\delta_{raw}$ into a near-Gaussian distribution $\delta_{gau} \sim \mathcal{N}(\sqrt{\bar{\alpha}_\mathcal{T}} \delta_{raw}, \sqrt{1-\bar{\alpha}_\mathcal{T}})$. **Denoising process**: In each denoising step, the current feature is concatenated with the 4D radar feature $\mathcal{F}_\mathcal{A}^R$ along the channel dimension and fed into a U-Net denoiser $\mathcal{F}_{t-1} = U_\theta([\mathcal{F}_t, \mathcal{F}_\mathcal{A}^R], t)$. After $\mathcal{T}$ steps, it outputs the denoised clean LiDAR feature $\tilde{\mathcal{F}_\mathcal{A}^L} = \mathcal{F}_0$.
    - **Design Motivation**: LiDAR noise distributions under adverse weather are complex and highly variable (varying by weather, distance, and density), making direct fitting extremely difficult. The reparameterization in the diffusion process "smooths" the complex distribution into a near-Gaussian distribution, significantly lowering the learning difficulty for the denoising model. The 4D radar features as conditions provide a "clean reference" to help the denoiser distinguish noise.

3. **Adaptive Loss Weight Scheduling**:
    - **Function**: Balance the training progress of denoising learning and detection learning.
    - **Mechanism**: The MDD loss is defined as $\mathcal{L}_{MDD} = \mathcal{L}_{MSE}(\tilde{\mathcal{F}_\mathcal{A}^L}, \mathcal{F}_l^L) \cdot \gamma(e, \psi)$, where $\mathcal{F}_l^L$ is the ground-truth feature extracted from clean LiDAR point clouds (shielded from weather noise). The weight $\gamma(e, \psi) = (1 - \tanh(\frac{e}{\tau} - \varphi)) \cdot \psi$ decreases non-linearly with the training epoch $e$. Early-stage models focus on denoising learning, gradually transitioning to the detection task in later stages.
    - **Design Motivation**: If the denoising loss weight is fixed too high, it hinders the training of the detection head; if fixed too low, the denoising effect is insufficient. A non-linear decreasing strategy allows the learning progress of both tasks to transition naturally.

### Loss & Training

The total loss is $\mathcal{L}_{all} = \beta_{cls} \mathcal{L}_{cls} + \beta_{loc} \mathcal{L}_{loc} + \mathcal{L}_{MDD}$. Adam optimizer is used with $lr=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$. The training/validation/testing split is 8,084/829/3,166 frames. The detection range is $x \in [0,140]m$, $y \in [-40,40]m$. The broadcasting range is 70m. Two fusion implementations are provided: SA2MA (Single-Agent to Multi-Agent multimodal extension) and SM2MM (Single-Modality to Multi-Modality multi-agent extension).

## Key Experimental Results

### Main Results

**Cooperative LiDAR Baselines** (Testing, IoU=0.3/0.5/0.7):

| Method | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|------|---------|---------|---------|
| AttFuse (LiDAR only) | 91.21 | 89.51 | 80.01 |
| AdaFusion (LiDAR only) | 92.72 | 91.64 | 84.81 |
| AttFuse (LiDAR+4D Radar) | 91.50 | 90.04 | 82.44 |
| AdaFusion (LiDAR+4D Radar) | **93.44** | **92.43** | **86.09** |

**Effect of the MDD Module under Adverse Weather** (AttFuse, IoU=0.5):

| Weather | w/o MDD | w/ MDD | Gain |
|------|--------|--------|------|
| Normal | 89.51 | ~89.4 | Almost unchanged |
| Foggy | ~83.8 | ~89.5 | **+5.73%** |
| Snowy | ~82.8 | ~89.5 | **+6.70%** |

### Ablation Study

**Comparison of Different Fusion Implementations** (Testing, IoU=0.7):

| Strategy | Source Method | mAP |
|------|---------|-----|
| SA2MA | InterFusion | 69.63 |
| SA2MA | L4DR | 82.26 |
| SM2MM | AttFuse | 82.44 |
| SM2MM | AdaFusion | 86.09 |

### Key Findings

- LiDAR-4D radar fusion consistently outperforms LiDAR-only by 1-2% in normal weather, demonstrating that the Doppler and extra geometric information of 4D radar are indeed valuable.
- The MDD module brings a significant improvement of 5-7% in adverse weather, while incurring almost zero loss in normal weather. This is achieved through reparameterization: under normal weather, $\delta_{raw} \approx 0$, and the diffusion-denoising process approximates an identity transform.
- The SM2MM strategy generally outperforms SA2MA, indicating that performing multi-agent fusion within each modality before cross-modality fusion is more effective than the reverse.
- Multi-agent cooperation significantly improves the instance coverage of 4D radar (especially at medium-to-long distances), compensating for the low resolution of single-agent 4D radar.
- MDD effectively denoises even with a small number of diffusion steps (the proposed MDD is inherently a lightweight feature-space diffusion).

## Highlights & Insights

- Introduces the 4D radar modality to V2X cooperative perception for the first time, filling the gap in existing datasets.
- The design of the MDD module is highly elegant—resolving the core challenge of 'difficult-to-fit weather noise distributions' through reparameterization, converting uncertain complex noise into more deterministic Gaussian noise.
- The property of 'almost zero loss in normal weather' allows MDD to serve as a plug-and-play module without requiring model switching based on weather conditions.
- The value of 4D radar in multi-agent scenarios is systematically validated—multi-agent cooperation compensates for the inherent limitation of low single-point resolution in 4D radar.

## Limitations & Future Work

- V2X-R is a simulation dataset; real-world sensor noise and communication delays may introduce additional challenges.
- Currently, only vehicles are detected; extension to more categories such as pedestrians and cyclists is needed.
- 4D radar only provides front-view data (limited by sensor FOV); 360° omnidirectional perception requires additional radar layouts.
- The physical accuracy of weather simulation may differ from real-world weather conditions.
- Future directions: (1) Validate MDD on real-world 4D radar datasets; (2) Introduce adaptive fusion under communication bandwidth constraints; (3) Extend to more adverse conditions (e.g., heavy rain, dust storms).

## Related Work & Insights

- The key difference from datasets like OPV2V/V2X-Sim lies in the addition of the 4D radar modality and adverse weather simulations.
- The diffusion-denoising concept of MDD can be generalized to other sensor fusion scenarios that handle high-uncertainty noise distributions (e.g., underwater sonar, medical ultrasound).
- The paradigm of using 4D radar as a 'weather anchor' to guide the denoising of other modalities provides broad inspiration for robust multi-sensor perception.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First V2X 4D radar dataset + new paradigm of diffusion-denoising fusion
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive benchmark covering multiple fusion strategies, weather conditions, and evaluation metrics
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, reasonable organization of dataset, method, and experiments
- **Value**: ⭐⭐⭐⭐⭐ — Pioneers the research direction of V2X 4D radar fusion, with the dataset and benchmark benefiting the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2025\] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)

</div>

<!-- RELATED:END -->
