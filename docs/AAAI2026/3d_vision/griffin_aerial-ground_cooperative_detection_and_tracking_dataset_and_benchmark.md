---
title: >-
  [Paper Note] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark
description: >-
  [AAAI 2026][3D Vision][Aerial-Ground Cooperative Perception] Introduces Griffin, an aerial-ground cooperative (AGC) 3D perception dataset and benchmark framework. It comprises 250+ dynamic scenes (37K+ frames), achieving realistic UAV dynamics, variable cruising altitudes (20–60m), and occlusion-aware annotations through CARLA-AirSim co-simulation, alongside a systematic robustness evaluation protocol.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Aerial-Ground Cooperative Perception"
  - "UAV-Vehicle Collaboration"
  - "3D Object Detection"
  - "Multi-Object Tracking"
  - "Collaborative Perception Dataset"
date: 2026-05-08
content_hash: a72f5526fd2e430a
---

# Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark

**Conference**: AAAI 2026  
**arXiv**: [2503.06983](https://arxiv.org/abs/2503.06983)  
**Code**: [https://github.com/wang-jh18-SVM/Griffin](https://github.com/wang-jh18-SVM/Griffin)  
**Area**: 3D Vision / Collaborative Perception  
**Keywords**: Aerial-Ground Cooperative Perception, UAV-Vehicle Collaboration, 3D Object Detection, Multi-Object Tracking, Collaborative Perception Dataset

## TL;DR

Introduces Griffin, an aerial-ground cooperative (AGC) 3D perception dataset and benchmark framework. It comprises 250+ dynamic scenes (37K+ frames), achieving realistic UAV dynamics, variable cruising altitudes (20–60m), and occlusion-aware annotations through CARLA-AirSim co-simulation, alongside a systematic robustness evaluation protocol.

## Background & Motivation

### Background

Collaborative perception has emerged as a crucial direction to overcome the limitations of single-vehicle systems (occlusions, limited field of view). Principal paradigms include:
- **V2V (Vehicle-to-Vehicle)**: OPV2V, V2V4Real, etc.
- **V2I (Vehicle-to-Infrastructure)**: DAIR-V2X, V2X-Seq, etc.
- **UAV Collaboration**: CoPerception-UAVs, UAV3D, etc.

### Key Challenge

V2V/V2I systems require substantial infrastructure investment and high penetration rates of connected vehicles, imposing significant economical barriers. **Aerial-Ground Cooperation (AGC)**—pairing UAVs with ground vehicles—offers a more flexible and cost-effective alternative that can be deployed on demand to provide occlusion-free bird's-eye views (BEVs). However, AGC perception research is constrained by **a lack of high-quality public datasets and benchmarks**.

### Limitations of Prior Work

| Problem | Affected Datasets |
|------|-------------|
| Idealized communication and localization (no noise) | UAV3D, AeroCollab3D, Air-Co-Pred, AirV2X |
| Simplified UAV models (fixed orientation/altitude) | V2U-COO, UAV3D, Air-Co-Pred |
| Lack of occlusion-aware annotations | CoPerception-UAVs, UAV3D, AeroCollab3D, AirV2X |
| No tracking IDs | AGC-Drive |
| Only 2D annotations | CoPeD |

**Key Gap**: No single AGC dataset simultaneously features occlusion-aware annotations, realistic noise simulation, multi-altitude support, and tracking IDs.

### Key Insight

Constructs Griffin—the first AGC perception dataset featuring simultaneous support for occlusion-aware 3D annotations, realistic UAV dynamics, multi-altitude settings, and communication latency/packet loss/localization error simulations, along with a unified detection and tracking benchmark framework.

## Method

### Overall Architecture

Griffin consists of three components:
1. **Dataset**: CARLA-AirSim co-simulation $\rightarrow$ Multi-sensor data acquisition $\rightarrow$ Occlusion-aware annotation
2. **Benchmark Framework**: Standardized implementation of four fusion paradigms (early / intermediate BEV-level / intermediate instance-level / late)
3. **Evaluation Protocol**: Precision + communication efficiency + robustness (latency, packet loss, localization error)

### Key Designs

#### 1. Data Acquisition and Scene Diversity

**Function**: Generates synchronized multi-agent data through CARLA-AirSim co-simulation.

**Sensor Configuration**:
- **Ground Vehicle**: 4 wide-field-of-view RGB cameras ($108.8^\circ$, 1920×1080) + 80-line LiDAR (10Hz, vertical FOV $-25^\circ$ to $15^\circ$)
- **Aerial UAV**: 5 downward-facing cameras (SWaP-constrained, no LiDAR)

**Scene Diversity**:
- 4 CARLA maps (2 urban + 2 suburban)
- Weather: Sunny/Rainy/Foggy $\times$ Noon/Sunset/Night $\times$ Wind speed 0–9 m/s
- Altitude: Griffin-Random (20–60m), Griffin-25m/40m/55m (each $\pm 2\text{m}$)
- 255 scene snippets of approximately 15 seconds each, totaling 37.7K frames, 339.3K images, and 914.8K 3D annotations

**UAV Dynamics Realism**: Leverages AirSim's physics engine to simulate flight dynamics, resulting in pitch/roll angle distributions clustered around zero rather than a sharp peak—reflecting real-world UAV continuous fine-tuning and attitude adjustments under wind.

**Design Motivation**:
- CARLA provides rich environments and traffic flows, while AirSim delivers realistic UAV physical models
- The LiDAR-free UAV configuration aligns with real-world constraints (e.g., BYD-DJI solution, where small UAV payloads are <1kg)
- Variable altitudes and weather conditions evaluate the generalization ability of the methods

#### 2. Occlusion-Aware Annotation

**Function**: Quantifies the visibility rate of each object relative to each agent and filters out invisible targets.

**Mechanism**:
1. Acquire RGB and instance segmentation images (fully aligned)
2. Sample points within each 3D bounding box and project them onto the segmentation images
3. Check the consistency of semantic classes and instance IDs for projected pixels against the ground truth
4. Compute the visibility rate for each agent
5. **Collaborative Perception Ground Truth**: Conserve targets visible to at least one agent

**Design Motivation**:
- Many datasets filter annotations solely by distance (yellow boxes), neglecting heavily occluded objects (red boxes)
- Omitting occlusion filtering introduces annotation noise from invisible targets, deteriorating model training quality
- Experimental verification: Training without occlusion filtering drops Early Fusion AP from 0.607 to 0.586

#### 3. Benchmark Framework

**Function**: Implements four fusion paradigms on a unified backbone (BEVFormer + ResNet-50).

**Four Fusion Strategies**:

| Fusion Type | Representative Method | Bandwidth (BPS) | Key Characteristics |
|---------|---------|-----------|------|
| Early Fusion | Raw Image Transmission | $3.11 \times 10^8$ | Performance upper bound, extremely high bandwidth |
| BEV-level Intermediate | V2X-ViT, Where2comm | $3.3 \text{–} 8.0 \times 10^5$ | Scene-level BEV features, compressed before transmission |
| Instance-level Intermediate | UniV2X, CoopTrack | $0.56 \text{–} 1.17 \times 10^5$ | Sparse object queries, lower bandwidth |
| Late Fusion | Output Transmission | $1.56 \times 10^3$ | Extremely low bandwidth, limited performance |

**Evaluation Protocol**:
- **Precision**: NuScenes AP and AMOTA
- **Communication Efficiency**: Bytes Per Second (BPS)
- **Robustness**: Communication latency (0–400ms), packet loss rate (0–50%), localization error (translation 0–2.5m, rotation $0^\circ \text{–} 5^\circ$)

### Loss & Training

- AdamW optimizer, learning rate $2 \times 10^{-4}$, batch size 8
- Distributed training on 4$\times$ NVIDIA 3090 GPUs
- Input images downsampled from 1920×1080 to 960×540
- Targets merged into 3 categories (cars, pedestrians, cyclists)
- Perception range: $102.4\text{m} \times 102.4\text{m}$ area centered around the ego vehicle

## Key Experimental Results

### Main Results

#### Performance of Methods Across Datasets of Different Altitudes

| Method | Griffin-25m AP/AMOTA | Griffin-55m AP/AMOTA | Bandwidth (BPS) |
|------|---------------------|---------------------|------------|
| No Fusion | 0.375 / 0.365 | 0.335 / 0.359 | 0 |
| Early Fusion | **0.607 / 0.670** | 0.483 / 0.522 | $3.11 \times 10^8$ |
| V2X-ViT | 0.465 / 0.508 | 0.350 / 0.379 | $8.00 \times 10^5$ |
| Where2comm | 0.396 / 0.406 | 0.317 / 0.353 | $3.30 \times 10^5$ |
| **CoopTrack** | 0.479 / 0.488 | 0.364 / 0.402 | **$1.17 \times 10^5$** |
| UniV2X | 0.419 / 0.456 | 0.323 / 0.349 | $5.58 \times 10^4$ |
| Late Fusion | 0.378 / 0.377 | 0.306 / 0.332 | $1.56 \times 10^3$ |

#### Griffin-Random (20–60m Mixed Altitudes)

| Method | AP | vs No Fusion |
|------|-----|-------------|
| No Fusion | 0.459 | — |
| Early Fusion | 0.583 | +0.124 |
| V2X-ViT | 0.400 | **-0.059** |
| Where2comm | 0.406 | -0.053 |
| CoopTrack | 0.468 | +0.009 |
| UniV2X | 0.402 | -0.057 |

### Ablation Study

#### Impact of Occlusion-Aware Annotations (Griffin-25m)

| Model | Annotation Method | AP | AMOTA |
|------|---------|-----|-------|
| Early Fusion | Occlusion-Aware (baseline) | 0.607 | 0.670 |
| Early Fusion | No Filtering | 0.586 (↓) | 0.636 (↓) |
| Vehicle Side | Occlusion-Aware | 0.477 | 0.457 |
| Vehicle Side | No Filtering | 0.412 (↓) | 0.433 (↓) |

#### Communication Robustness

| Latency (ms) | Early Fusion AP Drop | Intermediate Fusion Performance |
|---------|------------------|------------|
| 100 | ~10% | Outperforms No Fusion |
| 200 | ~20% | Detection marginally better, tracking remains good |
| 400 | >30% | Tracking still maintains superior performance |

#### Localization Robustness

| Translation Error std (m) | V2X-ViT | UniV2X |
|--------------|---------|--------|
| 0.5 | Normal | Normal |
| 1.5 | Lower than No Fusion | **Still outperforms No Fusion** |
| 2.5 | Severely degraded | **Still maintains advantage** |

### Key Findings

1. **Altitude variation has a significant impact on collaborative perception**: The collaborative gain peaks at 25m and degrades as altitude increases. Under mixed altitudes (20–60m), most intermediate fusion methods perform even **worse than the single-vehicle (No Fusion) baseline**.
2. **Instance-level fusion is more robust than BEV-level**: CoopTrack is the only intermediate fusion method to maintain positive gains on Griffin-Random, because instance-level methods decouple geometric transformations and semantic features, making them more robust to perspective mismatch.
3. **Where2comm and UniV2X underperform in AGC scenarios**: Because targets are sparse from the UAV's BEV perspective, spatial confidence maps or sparse queries based on positive sample detection are insufficiently trained.
4. **Packet loss has a smaller impact than latency**: Packet loss only leads to missing information (reducing gains) without introducing erroneous data, whereas latency causes spatial misalignment.
5. **UniV2X is the most robust to localization errors**: Selective fusion and instance-level filtering down-weight unreliable signals.
6. **Occlusion-aware annotation is crucial**: Omitting filtering degrades the performance of both collaborative and single-vehicle models.
7. **Tracking is more robust to latency than detection**: Temporal information helps alleviate inter-frame alignment issues.

## Highlights & Insights

1. **The finding that 'altitude variation can render collaborative perception ineffective' is a profound and unique insight of AGC**: This issue is non-existent in V2V/V2I, but is critical for AGC systems.
2. **The occlusion-aware annotation method is elegant and effective**: It leverages the simulator's instance segmentation GT to quantify visibility rates, avoiding massive manual annotation costs.
3. **Aggressive robustness evaluation bounds** (2.5m translation / $5^\circ$ rotation / 400ms latency / 50% packet loss): Going far beyond standard evaluations, these settings reveal the actual failure boundaries of current methods.
4. **CARLA-AirSim co-simulation framework**: Ingeniously combines the respective advantages of both simulators (CARLA's environment and traffic flow, AirSim's UAV physics).
5. **In-depth comparison between BEV vs. instance-level fusion**: Provides clear guidelines for selecting fusion strategies in AGC contexts.

## Limitations & Future Work

1. **Domain gap between simulation and real-world data**: Albeit efforts to mimic reality (e.g., LiDAR-free UAV, noise injection), the sim-to-real gap persists.
2. **Only the car category is evaluated**: Results for pedestrians and cyclists are omitted from the main text.
3. **Fixed backbone (ResNet-50 BEVFormer)**: A stronger single-vehicle detector might alter the relative ranking of fusion methods.
4. **Disparity of weather influences on different methods is not evaluated**: Though the dataset covers various weather conditions, performance is not analyzed grouping by weather.
5. **Height-adaptive and scale-aware fusion mechanisms** should be developed to address the core challenges.
6. More advanced Late Fusion strategies can be explored to potentially achieve better trade-offs under extremely low bandwidth.

## Related Work & Insights

- **OPV2V** (Xu et al., ICRA 2022): Pioneering V2V collaborative perception work; Griffin fills the AGC gap.
- **DAIR-V2X** (Yu et al., CVPR 2022): Real-world V2I dataset, but camera height is fixed at 20–25m.
- **BEVFormer** (Li et al., 2022): Serves as the unified backbone for all baselines.
- **V2X-ViT, Where2comm**: Representatives of BEV-level intermediate fusion; this work exposes their limitations in AGC scenarios.
- **CoopTrack** (Zhong et al., ICCV 2025): Instance-level fusion proves more robust in variable-height scenarios.
- **Insights**: AGC scenarios demand entirely new fusion design philosophy—simply migrating V2V/V2I methods is insufficient as scale and perspective mismatches induced by altitude variation constitute the core challenges.

## Rating

- Novelty: ⭐⭐⭐⭐ — While the AGC dataset concept is not entirely new, the occlusion-aware annotation and systematic robustness evaluation are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive evaluation involving 6 methods $\times$ 4 altitudes $\times$ 3 types of disturbances.
- Writing Quality: ⭐⭐⭐⭐⭐ — Deep coverage of dataset construction details and thorough experimental analysis.
- Value: ⭐⭐⭐⭐⭐ — Fills the critical data gap of AGC perception; the 'altitude robustness' findings provide invaluable guidelines for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](../../ICCV2025/3d_vision/event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2026\] VGA: Empowering Aerial-Ground Localization by Visual Geometry Alignment](../../CVPR2026/3d_vision/vga_empowering_aerial-ground_localization_by_visual_geometry_alignment.md)
- [\[AAAI 2026\] Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable](redundant_queries_in_detr-based_3d_detection_methods_unnecessary_and_prunable.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](../../CVPR2025/3d_vision/aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)

</div>

<!-- RELATED:END -->
