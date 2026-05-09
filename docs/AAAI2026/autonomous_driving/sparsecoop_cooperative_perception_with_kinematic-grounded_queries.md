---
title: >-
  [Paper Note] SparseCoop: Cooperative Perception with Kinematic-Grounded Queries
description: >-
  [AAAI 2026][Autonomous Driving][Cooperative Perception] This paper proposes SparseCoop—the first fully sparse cooperative perception framework—which abandons dense BEV representations entirely through kinematic-grounded queries (KGQ), a coarse-to-fine aggregation module, and a cooperative instance denoising strategy. SparseCoop achieves state-of-the-art performance on V2X-Seq and Griffin datasets with minimal communication overhead and maximum computational efficiency (AP 0.530, transmission cost only 3.17×10⁴ BPS).
tags:
  - AAAI 2026
  - Autonomous Driving
  - Cooperative Perception
  - Sparse Query
  - "3D Object Detection & Tracking"
  - V2X Communication
  - Vehicle-Road Cooperation
date: 2026-05-08
content_hash: 25ac56e744e815c7
---

# SparseCoop: Cooperative Perception with Kinematic-Grounded Queries

**Conference**: AAAI 2026
**arXiv**: [2512.06838](https://arxiv.org/abs/2512.06838)
**Code**: [github.com/wang-jh18-SVM/SparseCoop](https://github.com/wang-jh18-SVM/SparseCoop)
**Area**: Autonomous Driving
**Keywords**: Cooperative Perception, Sparse Query, 3D Object Detection & Tracking, V2X Communication, Vehicle-Road Cooperation

## TL;DR

This paper proposes SparseCoop—the first fully sparse cooperative perception framework—which abandons dense BEV representations entirely through kinematic-grounded queries (KGQ), a coarse-to-fine aggregation module, and a cooperative instance denoising strategy. SparseCoop achieves state-of-the-art performance on V2X-Seq and Griffin datasets with minimal communication overhead and maximum computational efficiency (AP 0.530, transmission cost only 3.17×10⁴ BPS).

## Background & Motivation

### The Necessity of Cooperative Perception

Single-vehicle systems are inherently constrained by sensor field-of-view limitations, long-range perception degradation, and severe occlusion—challenges that constitute critical bottlenecks for the safe deployment of autonomous driving. Cooperative perception addresses these issues by creating collective perception systems through information exchange among multiple agents (V2V, V2I, V2D).

### Three Core Challenges in Existing Approaches

#### 1. Fundamental Flaws of Dense BEV Features
Mainstream methods share dense BEV feature maps to provide a unified spatial grid, but suffer from:
- **Communication and computation costs that scale quadratically with perception range**
- **Abstract scene-level features that are difficult to align precisely across agents** (especially under temporal asynchrony and viewpoint discrepancies)

#### 2. New Problems in Sparse Query Methods
Emerging sparse query methods are more efficient but face:
- **Insufficient geometric representation**: queries are typically anchored by a single reference point, which cannot handle large viewpoint rotations and temporal offsets
- **Suboptimal fusion strategies**: simple linear networks (limited expressiveness) or global attention (ignoring fine-grained context from ego-vehicle sensor data)
- **Training instability**: sparse co-observed objects across different agent viewpoints and occlusion patterns lead to insufficient positive training samples

#### 3. Limitations of Pseudo-Sparse Methods
Many "sparse" methods still rely on dense BEV components, inheriting the associated computational scaling issues.

## Method

### Overall Architecture

The complete SparseCoop pipeline:

1. **Sparse Instance Extraction** (executed independently per agent): generates kinematic-grounded queries (KGQ) from multi-view image features
2. **Kinematic-Grounded Association**: leverages the rich state vectors of KGQs for precise spatio-temporal alignment and matching across agents
3. **Coarse-to-Fine Aggregation**: first performs coarse fusion on matched pairs, then refines all instances through multi-context refinement
4. **Cooperative Instance Denoising** (training only): initializes denoising queries from perturbed GT boxes to provide stable supervision signals

### Key Designs

#### 1. Kinematic-Grounded Query (KGQ)

**Function**: Defines a rich, explicit state vector for each detected instance, replacing simple reference point representations.

**Core Definition**: Each KGQ is defined as $\{\mathcal{F}, \mathcal{S}\}$, where $\mathcal{F}$ is a semantic feature vector and $\mathcal{S}$ is an 11-dimensional explicit state vector:

$$\mathcal{S} = (x, y, z, l, w, h, \sin(\theta), \cos(\theta), v_x, v_y, v_z)$$

This encodes 3D position, dimensions, heading angle, and velocity, carrying far richer geometric and kinematic information than a single reference point.

**Design Motivation**: Simple reference-point anchoring cannot accommodate the large viewpoint rotations and temporal offsets inherent in cooperative perception. The explicit state vector not only supports precise spatio-temporal alignment but also provides multi-dimensional cues for matching and fusion.

#### 2. Kinematic-Grounded Association (KGA)

**Function**: Enables robust instance matching across different agents.

**Spatio-Temporal Alignment**:
- **Latency Compensation**: uses velocity $(v_x, v_y, v_z)$ from the state vector to predict the state of cooperative instances to the ego vehicle's current timestamp via a constant-velocity motion model
- **Coordinate Projection**: projects instances into the ego coordinate frame via transformation matrix $\mathbf{T}_{co \to ego}$
- **Feature Update**: updates feature vectors using a rotation-aware MLP

**Geometric-Appearance Matching (GAM)**: constructs a pairwise cost matrix $C$ combining two complementary components:
- **Geometric Similarity**: weighted L1 distance between state vectors
- **Appearance Similarity**: cosine distance between feature vectors

The association yields three groups of instances: (1) matched pairs, (2) unmatched ego instances, and (3) unmatched cooperative instances.

**Interaction Range Design**: An interaction range $R_{int}$ is defined (optimally 30m on V2X-Seq, 15m on Griffin), within which fusion is performed. Cooperative instances outside this range are directly output, preventing low-quality ego data from contaminating high-quality cooperative detections.

#### 3. Coarse-to-Fine Aggregation (CFA)

**Function**: Effectively fuses information from both matched and unmatched instances.

**Coarse Fusion**: applies a lightweight linear network to fuse feature vectors of matched pairs:

$$\mathcal{F}_{\text{fused}} = \text{MLP}([\mathcal{F}_{\text{ego}}; \widetilde{\mathcal{F}_{\text{co}}}])$$

**Multi-Context Refinement**: fused KGQs and all unmatched KGQs undergo iterative refinement, with each refinement stage comprising:
- **Temporal Cross-Attention**: links current-frame instances with those from the previous frame to understand motion and maintain tracking consistency
- **Cooperative Cross-Attention**: interacts with the full set of aligned cooperative KGQs to obtain information about occluded regions
- **Self-Attention**: captures relationships among all instances in the current frame to reason about scene layout and suppress duplicate detections
- **Deformable Aggregation**: samples from ego multi-scale image features, grounding abstract instance representations in raw visual data

**Design Motivation**: Unlike methods that rely solely on temporal or cooperative context, this work argues that **ego image features are equally critical for cooperative perception**—deformable aggregation refines localization directly from raw visual data.

#### 4. Cooperative Instance Denoising (CID)

**Function**: Addresses training instability caused by scarce positive supervision signals in sparse cooperative perception.

**Problem Analysis**:
- In V2X-Seq, a large proportion of GT objects are visible to only one agent (approximately 58% are visible only from the roadside)
- Even when the same object is visible to both agents, early-training predictions may be too inaccurate to be matched

**Noise Injection**:
- **Observation Noise**: simulates sensor errors by applying uniform perturbations to GT attributes in the local coordinate frame (position ±2.0m, others ±0.5)
- **Transformation Noise (novel contribution)**: simulates calibration errors and temporal asynchrony by adding random rotations (σ=2°) and translations (σ=1m) to transformation matrices

**Denoising Pipeline**:
- Denoising instances are matched directly via tracking ID, providing a large number of stable matched pairs
- A customized attention mask strictly isolates the normal and denoising pipelines to prevent information leakage

### Loss & Training

- Standard detection and tracking losses based on the Sparse4D framework
- The denoising pipeline shares network weights with the normal pipeline while maintaining attention isolation
- Denoising instances provide additional matched-pair supervision during training
- High-confidence KGQs are assigned tracking IDs and propagated to subsequent frames via a recurrent mechanism

## Key Experimental Results

### Main Results

#### Performance Comparison on V2X-Seq and Griffin-25m

| Method | V2X-Seq AP↑ | V2X-Seq AMOTA↑ | TC (BPS)↓ | Griffin AP↑ | Griffin AMOTA↑ | FPS↑ |
|--------|------------|----------------|-----------|------------|----------------|------|
| No Fusion | 0.166 | 0.130 | 0 | 0.375 | 0.365 | 8.10 |
| Early Fusion | 0.243 | 0.209 | 8.19×10⁷ | 0.607 | 0.670 | 5.17 |
| V2X-ViT (ECCV22) | 0.268 | 0.287 | 2.56×10⁶ | 0.465 | 0.508 | 7.56 |
| CoopTrack (ICCV25) | 0.390 | 0.328 | 5.64×10⁴ | 0.479 | 0.488 | 6.23 |
| **SparseCoop** | **0.530** | **0.421** | **3.17×10⁴** | **0.559** | **0.509** | **11.64** |
| Gain (vs. CoopTrack) | **+35.9%** | **+28.4%** | **-43.8%** | **+16.7%** | **+4.3%** | **+86.8%** |

SparseCoop achieves comprehensive superiority across all metrics: detection AP improves by 35.9%, tracking AMOTA by 28.4%, while transmission cost decreases by 43.8% and inference speed improves by 86.8% (11.64 vs. 6.23 FPS).

### Ablation Study

#### Contribution of Each Module (V2X-Seq)

| Configuration | AP↑ | AMOTA↑ | Notes |
|---------------|-----|--------|-------|
| **Full Model** | **0.530** | **0.421** | - |
| w/o Latency Compensation (LC) | 0.505 | 0.414 | AP -4.7% |
| w/o Geometric-Appearance Matching (GAM) | 0.502 | 0.414 | AP -5.3% |
| w/o Coarse Feature Fusion (CFF) | 0.489 | 0.375 | **AMOTA -10.9%** |
| w/o Multi-Context Refinement (MCR) | 0.512 | 0.379 | AMOTA -10.0% |
| w/o Observation Noise (ON) | 0.521 | 0.416 | AP -1.7% |
| w/o Transformation Noise (TN) | 0.531 | 0.394 | AMOTA -6.4% |
| **w/o All Denoising** | **0.521** | **0.352** | **AMOTA -16.4%** |

Coarse fusion and denoising have the largest impact on tracking performance (AMOTA drops by 10.9% and 16.4%, respectively), demonstrating the critical role of fusion quality and training stability.

### Key Findings

1. **Fully sparse is feasible and efficient**: SOTA is achievable without dense BEV, with a transmission cost of only 3.17×10⁴ BPS
2. **Interaction range is a critical hyperparameter**: too large causes contamination by low-quality data; too small leads to duplicate detections (optimal: 30m on V2X-Seq, 15m on Griffin)
3. **Excellent robustness to communication latency**: at 200ms latency, SparseCoop's AP surpasses all methods including zero-latency early fusion, attributable to kinematic compensation
4. **Transformation noise is more important than observation noise**: removing transformation noise causes a 6.4% AMOTA drop, as it directly simulates cross-agent calibration and asynchrony errors

## Highlights & Insights

- **Complete departure from BEV**: the first genuinely fully sparse cooperative perception framework
- **Elegant KGQ design**: the 11-dimensional state vector simultaneously serves latency compensation, coordinate transformation, matching, and fusion localization
- **Deep understanding of interaction range**: not all cooperative data should be fused—far-field instances are directly output while near-field instances are carefully fused
- **Clever denoising strategy**: leverages prior knowledge from GT boxes to supply stable matched pairs during training while strictly isolating pipelines to prevent information leakage

## Limitations & Future Work

- Currently limited to two agents (one-to-one); extension to multi-agent cooperation is needed
- The constant-velocity motion model has limited compensation accuracy for non-uniform motion
- The interaction range requires manual tuning for different datasets
- The framework uses camera input only; extension to LiDAR and radar modalities is a natural direction
- Noise parameters in the denoising strategy need to be calibrated to match actual system errors

## Related Work & Insights

- **Sparse4D series**: foundational framework for sparse instance extraction; SparseCoop builds cooperative modules on top of it
- **DN-DETR/MaskDINO**: denoising training techniques transferred from object detection to cooperative perception
- **V2X-Seq/Griffin**: standard evaluation benchmarks for cooperative perception
- **QUEST/CoopTrack**: pioneering sparse query cooperative methods whose key limitations SparseCoop addresses

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (fully sparse + KGQ + cooperative denoising; systematic innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (two datasets, comprehensive ablation, latency robustness analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear logic, explicit problem-solution correspondence)
- Value: ⭐⭐⭐⭐⭐ (significant advancement for the cooperative perception paradigm, balancing performance, efficiency, and robustness)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](../../ICCV2025/autonomous_driving/cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [\[AAAI 2026\] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving](radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)

</div>

<!-- RELATED:END -->
