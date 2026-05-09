---
title: >-
  [Paper Note] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark
description: >-
  [AAAI 2026][3D Vision][aerial-ground cooperative perception] This paper presents Griffin, the first aerial-ground cooperative (AGC) 3D perception dataset and benchmark framework, comprising 250+ dynamic scenes (37K+ frames) generated via CARLA-AirSim joint simulation. Griffin features realistic UAV dynamics, variable cruise altitudes (20–60 m), occlusion-aware annotations, and a systematic robustness evaluation protocol.
tags:
  - AAAI 2026
  - 3D Vision
  - aerial-ground cooperative perception
  - UAV-vehicle collaboration
  - 3D object detection
  - multi-object tracking
  - collaborative perception dataset
date: 2026-05-08
content_hash: 37f9a437b6ab0a0d
---

# Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark

**Conference**: AAAI 2026
**arXiv**: [2503.06983](https://arxiv.org/abs/2503.06983)
**Code**: [https://github.com/wang-jh18-SVM/Griffin](https://github.com/wang-jh18-SVM/Griffin)
**Area**: 3D Vision / Collaborative Perception
**Keywords**: aerial-ground cooperative perception, UAV-vehicle collaboration, 3D object detection, multi-object tracking, collaborative perception dataset

## TL;DR

This paper presents Griffin, the first aerial-ground cooperative (AGC) 3D perception dataset and benchmark framework, comprising 250+ dynamic scenes (37K+ frames) generated via CARLA-AirSim joint simulation. Griffin features realistic UAV dynamics, variable cruise altitudes (20–60 m), occlusion-aware annotations, and a systematic robustness evaluation protocol.

## Background & Motivation

### State of the Field

Collaborative perception has emerged as a critical direction for overcoming the limitations of single-agent systems (occlusion, limited field of view). Major paradigms include:
- **V2V (vehicle-to-vehicle)**: OPV2V, V2V4Real, etc.
- **V2I (vehicle-to-infrastructure)**: DAIR-V2X, V2X-Seq, etc.
- **UAV collaboration**: CoPerception-UAVs, UAV3D, etc.

### Root Cause

V2V/V2I systems require large-scale infrastructure investment and widespread vehicular networking, imposing high economic barriers. **Aerial-ground cooperation (AGC)**—pairing UAVs with ground vehicles—offers a more flexible and cost-effective alternative that can be deployed on demand and provides an unobstructed bird's-eye view. However, AGC perception research is hindered by the **lack of high-quality public datasets and benchmarks**.

### Limitations of Prior Work

| Issue | Affected Datasets |
|-------|------------------|
| Idealized communication and localization (noise-free) | UAV3D, AeroCollab3D, Air-Co-Pred, AirV2X |
| Simplified UAV models (fixed orientation/altitude) | V2U-COO, UAV3D, Air-Co-Pred |
| No occlusion-aware annotations | CoPerception-UAVs, UAV3D, AeroCollab3D, AirV2X |
| No tracking IDs | AGC-Drive |
| 2D annotations only | CoPeD |

**Critical gap**: No existing AGC dataset simultaneously provides occlusion-aware annotations, realistic noise simulation, multi-altitude support, and tracking IDs.

### Starting Point

This paper constructs Griffin—the first AGC perception dataset to simultaneously support occlusion-aware 3D annotations, realistic UAV dynamics, multi-altitude settings, and simulation of communication interference and localization errors—alongside a unified detection and tracking benchmark framework.

## Method

### Overall Architecture

Griffin consists of three components:
1. **Dataset**: CARLA-AirSim joint simulation → multi-sensor data collection → occlusion-aware annotation
2. **Benchmark framework**: Standardized implementations of four fusion paradigms (early / intermediate BEV-level / intermediate instance-level / late)
3. **Evaluation protocol**: Accuracy + communication efficiency + robustness (latency, packet loss, localization error)

### Key Designs

#### 1. Data Collection and Scene Diversity

**Function**: Generate synchronized multi-agent data via CARLA-AirSim joint simulation.

**Sensor configuration**:
- **Ground vehicle**: 4 wide-FOV RGB cameras (108.8°, 1920×1080) + 80-line LiDAR (10 Hz, vertical FOV −25° to 15°)
- **Aerial UAV**: 5 downward-facing cameras (SWaP-constrained; no LiDAR)

**Scene diversity**:
- 4 CARLA maps (2 urban + 2 suburban)
- Weather: sunny/rainy/foggy × noon/sunset/night × wind speed 0–9 m/s
- Altitude: Griffin-Random (20–60 m), Griffin-25m/40m/55m (each ±2 m)
- 255 scene clips, ~15 s each; 37.7K frames, 339.3K images, 914.8K 3D annotations in total

**UAV dynamics realism**: AirSim's physics engine is used to simulate UAV motion; pitch/roll angle distributions are centered near zero rather than sharply peaked—reflecting the continuous micro-adjustments and wind-resistance corrections of real UAVs.

**Design Motivation**:
- CARLA provides rich environments and traffic flows; AirSim provides a realistic UAV physics model
- The LiDAR-free UAV configuration mirrors practical deployments (e.g., BYD-DJI solutions, where small UAVs carry payloads <1 kg)
- Variable altitudes and weather conditions test method generalization

#### 2. Occlusion-Aware Annotation

**Function**: Quantify the visibility ratio of each target to each agent and filter out invisible targets.

**Mechanism**:
1. Collect RGB and instance segmentation images (perfectly aligned)
2. Sample points within each 3D bounding box and project them onto the segmentation image
3. Compare the semantic class and instance ID of projected pixels against those of the target
4. Compute the per-agent visibility ratio
5. **Collaborative perception GT**: retain targets visible to at least one agent

**Design Motivation**:
- Many datasets filter annotations by distance only (yellow boxes), ignoring heavily occluded targets (red boxes)
- Skipping occlusion filtering introduces annotation noise from invisible targets, degrading model training quality
- Experimental validation: training without occlusion filtering reduces Early Fusion AP from 0.607 to 0.586

#### 3. Benchmark Framework

**Function**: Implement four fusion paradigms on a unified backbone (BEVFormer + ResNet-50).

**Four fusion strategies**:

| Fusion Type | Representative Methods | Communication (BPS) | Characteristics |
|-------------|----------------------|---------------------|-----------------|
| Early Fusion | Raw image transmission | 3.11×10⁸ | Performance upper bound; extremely high bandwidth |
| BEV-level intermediate | V2X-ViT, Where2comm | 3.3–8.0×10⁵ | Scene-level BEV features transmitted after compression |
| Instance-level intermediate | UniV2X, CoopTrack | 0.56–1.17×10⁵ | Sparse object queries; lower bandwidth |
| Late Fusion | Detection result transmission | 1.56×10³ | Extremely low bandwidth; limited performance |

**Evaluation protocol**:
- **Accuracy**: NuScenes AP and AMOTA
- **Communication efficiency**: bytes per second (BPS)
- **Robustness**: communication latency (0–400 ms), packet loss rate (0–50%), localization error (translation 0–2.5 m, rotation 0–5°)

### Loss & Training

- AdamW optimizer, learning rate $2\times10^{-4}$, batch size 8
- Distributed training on 4× NVIDIA RTX 3090 GPUs
- Input images downsampled from 1920×1080 to 960×540
- Objects merged into 3 categories (car, pedestrian, two-wheeler)
- Perception range: 102.4 m × 102.4 m region centered on the ego vehicle

## Key Experimental Results

### Main Results

#### Per-Method Performance Across Altitude Datasets

| Method | Griffin-25m AP/AMOTA | Griffin-55m AP/AMOTA | Communication (BPS) |
|--------|---------------------|---------------------|---------------------|
| No Fusion | 0.375/0.365 | 0.335/0.359 | 0 |
| Early Fusion | **0.607/0.670** | 0.483/0.522 | 3.11×10⁸ |
| V2X-ViT | 0.465/0.508 | 0.350/0.379 | 8.00×10⁵ |
| Where2comm | 0.396/0.406 | 0.317/0.353 | 3.30×10⁵ |
| **CoopTrack** | 0.479/0.488 | 0.364/0.402 | **1.17×10⁵** |
| UniV2X | 0.419/0.456 | 0.323/0.349 | 5.58×10⁴ |
| Late Fusion | 0.378/0.377 | 0.306/0.332 | 1.56×10³ |

#### Griffin-Random (Mixed Altitude 20–60 m)

| Method | AP | vs. No Fusion |
|--------|----|--------------|
| No Fusion | 0.459 | — |
| Early Fusion | 0.583 | +0.124 |
| V2X-ViT | 0.400 | **−0.059** |
| Where2comm | 0.406 | −0.053 |
| CoopTrack | 0.468 | +0.009 |
| UniV2X | 0.402 | −0.057 |

### Ablation Study

#### Impact of Occlusion-Aware Annotation (Griffin-25m)

| Model | Annotation Type | AP | AMOTA |
|-------|-----------------|----|-------|
| Early Fusion | Occlusion-aware (baseline) | 0.607 | 0.670 |
| Early Fusion | No filtering | 0.586 (↓) | 0.636 (↓) |
| Vehicle Side | Occlusion-aware | 0.477 | 0.457 |
| Vehicle Side | No filtering | 0.412 (↓) | 0.433 (↓) |

#### Communication Robustness

| Latency (ms) | Early Fusion AP Drop | Intermediate Fusion |
|--------------|---------------------|---------------------|
| 100 | ~10% | Still outperforms No Fusion |
| 200 | ~20% | Detection marginally better; tracking still improved |
| 400 | >30% | Tracking maintains advantage |

#### Localization Robustness

| Translation Error std (m) | V2X-ViT | UniV2X |
|---------------------------|---------|--------|
| 0.5 | Normal | Normal |
| 1.5 | Below No Fusion | **Still outperforms No Fusion** |
| 2.5 | Severe degradation | **Still maintains advantage** |

### Key Findings

1. **Altitude variation has a profound impact on collaborative perception**: cooperative gain is greatest at 25 m and degrades with increasing altitude; under mixed altitudes (20–60 m), most intermediate fusion methods **perform worse than the single-agent baseline**.
2. **Instance-level fusion is more robust than BEV-level fusion**: CoopTrack is the only intermediate fusion method to maintain positive gain on Griffin-Random, as instance-level methods decouple geometric transformations from semantic features, making them more robust to viewpoint inconsistency.
3. **Where2comm and UniV2X perform poorly in AGC scenarios**: UAV bird's-eye views produce sparse object distributions, leading to under-training of spatial confidence maps and sparse queries based on positive sample detection.
4. **Packet loss has less impact than latency**: packet loss only causes information absence (reducing gain), whereas latency introduces spatial misalignment.
5. **UniV2X is most robust to localization errors**: selective fusion and instance-level filtering down-weight unreliable signals.
6. **Occlusion-aware annotation matters**: omitting filtering degrades both collaborative and single-agent model performance.
7. **Tracking is more robust to latency than detection**: temporal information helps mitigate inter-frame alignment issues.

## Highlights & Insights

1. **"Altitude variation invalidates collaborative perception" is a profound AGC-specific finding**: this issue does not arise in V2V/V2I settings but is critical for AGC.
2. **The occlusion-aware annotation method is concise and effective**: leveraging the simulator's instance segmentation ground truth to quantify visibility ratios avoids the cost of manual annotation.
3. **The robustness evaluation covers aggressive ranges** (2.5 m translation / 5° rotation / 400 ms latency / 50% packet loss): far exceeding standard evaluation ranges, revealing true failure boundaries of each method.
4. **The CARLA-AirSim joint simulation framework** cleverly exploits the complementary strengths of both simulators (CARLA's environments + AirSim's UAV physics).
5. **In-depth comparison of BEV-level vs. instance-level fusion**: provides clear guidance on fusion strategy selection for AGC scenarios.

## Limitations & Future Work

1. **Simulation-to-real domain gap**: despite efforts to approximate reality (LiDAR-free UAVs, noise injection), the sim-to-real gap persists.
2. **Only car category evaluated**: results for pedestrians and two-wheelers are not presented in the main paper.
3. **Fixed backbone (ResNet-50 BEVFormer)**: stronger single-agent detectors may alter the relative ranking of fusion methods.
4. **Differential impact of weather on individual methods is not analyzed**: although the dataset covers diverse weather conditions, the analysis does not group results by weather type.
5. **Height-adaptive and scale-aware fusion mechanisms** should be developed to address the core challenge.
6. More advanced late fusion strategies could be explored to achieve better cost-effectiveness under extremely low bandwidth constraints.

## Related Work & Insights

- **OPV2V** (Xu et al., ICRA 2022): pioneering work in V2V collaborative perception; Griffin fills the AGC gap.
- **DAIR-V2X** (Yu et al., CVPR 2022): real-world V2I dataset, but with fixed camera height of 20–25 m.
- **BEVFormer** (Li et al., 2022): unified backbone used for all baselines.
- **V2X-ViT, Where2comm**: representative BEV-level intermediate fusion methods; this paper reveals their limitations in AGC scenarios.
- **CoopTrack** (Zhong et al., ICCV 2025): instance-level fusion proves more robust under altitude variation.
- **Insight**: AGC scenarios require entirely new fusion design philosophies—V2V/V2I methods cannot be directly transferred; scale and viewpoint inconsistency induced by altitude variation is the core challenge.

## Rating

- Novelty: ⭐⭐⭐⭐ — AGC datasets are not a wholly new concept, but occlusion-aware annotation and systematic robustness evaluation are notable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 methods × 4 altitudes × 3 perturbation types; evaluation is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Dataset construction details and experimental analysis are both highly thorough.
- Value: ⭐⭐⭐⭐⭐ — Fills a critical data gap in AGC perception research; the "altitude robustness" finding provides important guidance for future work.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](../../ICCV2025/3d_vision/event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[AAAI 2026\] Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable](redundant_queries_in_detr-based_3d_detection_methods_unnecessary_and_prunable.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)

<!-- RELATED:END -->
