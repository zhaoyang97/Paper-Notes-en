---
title: >-
  [Paper Note] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts
description: >-
  [ICCV 2025][Autonomous Driving][Monocular 3D Detection] This paper proposes DUO (Dual Uncertainty Optimization), the first test-time adaptation framework that jointly minimizes semantic uncertainty and geometric uncertai…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Monocular 3D Detection"
  - "Test-Time Adaptation"
  - "Uncertainty Optimization"
  - "Domain Shift"
  - "Convex Optimization"
date: 2026-05-08
content_hash: d35073ca4fbc427d
---

# Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts

**Conference**: ICCV 2025
**arXiv**: [2508.20488](https://arxiv.org/abs/2508.20488)  
**Code**: [GitHub](https://github.com/hzcar/DUO)  
**Area**: Autonomous Driving
**Keywords**: Monocular 3D Detection, Test-Time Adaptation, Uncertainty Optimization, Domain Shift, Convex Optimization

## TL;DR

This paper proposes DUO (Dual Uncertainty Optimization), the first test-time adaptation framework that jointly minimizes semantic uncertainty and geometric uncertainty, achieving robust monocular 3D object detection via conjugate focal loss and normal field constraints.

## Background & Motivation

Monocular 3D object detection (M3OD) is critical for safety-sensitive applications such as autonomous driving. However, **domain shifts** caused by weather variations and sensor discrepancies during real-world deployment lead to severe performance degradation. Test-time adaptation (TTA) addresses this by updating model parameters online during inference, with entropy minimization being the predominant strategy for uncertainty reduction.

Nevertheless, existing TTA methods overlook the **dual uncertainty** inherent to M3OD:

**Semantic uncertainty**: ambiguity in category predictions

**Geometric uncertainty**: instability in spatial localization

The authors identify two critical failure modes through empirical analysis:
- **Low-score object neglect**: Entropy minimization provides ineffective supervision for difficult objects with low detection scores, resulting in missed detections.
- **Spatial perception collapse**: Directly minimizing depth uncertainty causes multi-head depth estimators to degenerate into a single deterministic head, undermining robust spatial understanding.

These observations expose fundamental limitations of existing methods in the M3OD setting and motivate the design of DUO.

## Method

### Overall Architecture

DUO adopts a dual-branch design with two core contributions:
1. **Conjugate Focal Loss (CFL)**: Label-free semantic uncertainty optimization grounded in convex optimization theory.
2. **Normal Consistency Loss (NCL)**: Stabilization of geometric representations through normal field consistency.

The two branches form a complementary cycle: enhanced spatial perception improves semantic classification, while robust semantic predictions further refine spatial understanding.

### Key Design 1: Conjugate Focal Loss (CFL)

**Legendre-Fenchel Structure**: The focal loss is reformulated as a convex optimization problem $\mathcal{L}_{FL} = f(h) - y^\top g(h)$, where $f(h) = \alpha \log s$ and $g(h) = \alpha h + \alpha((1-p)^\gamma - 1)\log p$.

**Problem Reformulation**: Under the assumption that the representation $h$ from the pretrained model is near a local optimum, the optimization is recast as finding the conjugate function $f^*(y)$.

**Higher-Order Approximation**: Via chain rule and higher-order approximation, the label-free estimate is derived as:
$$y_0 \approx (I + \gamma(1-\log p) \cdot pp^\top - \gamma \log p \cdot \text{diag}(p))^{-1} p$$

**Final CFL Formulation**:
$$\mathcal{L}_{CFL}(x) = -\alpha(1-p)^\gamma (I + \gamma(1-\log p) \cdot p^\top p - \gamma \log p \cdot \text{diag}(p))^{-1} p \log p$$

CFL offers three advantages over the original focal loss:
- **Dynamic vs. static weighting**: Beyond the $(1-p)^\gamma$ class-imbalance factor, it dynamically adjusts weights across all classes via matrix inversion.
- **No ground-truth labels required**: Operates entirely on predicted probabilities.
- **Hyperparameter compatibility**: $\alpha$ and $\gamma$ are reused from the source-stage focal loss, requiring no additional tuning.

### Key Design 2: Semantics-Guided Normal Field Constraint

**Normal Field Computation**: Starting from depth map $D$, spatial gradients $\nabla D_x, \nabla D_y$ are efficiently computed using Sobel operators to derive the normal field $\mathbf{N}(u,v)$.

**Normal Consistency Loss (NCL)**:
$$\mathcal{L}_{NCL}(u,v) = (\psi_x(u,v) + \psi_y(u,v)) \cdot \exp(-\|\nabla \text{I}(u,v)\|_2)$$
where $\psi_x$ and $\psi_y$ enforce normal consistency along horizontal and vertical directions respectively, and the edge-aware weight $\exp(-\|\nabla I\|_2)$ preserves discontinuities at boundaries.

**Semantics-Guided Mask**: Using semantic uncertainty $U_i$ computed by CFL, low-uncertainty regions are selected via an exponential moving average threshold to construct the guidance mask $\mathcal{M}$:
$$\mathcal{M}(u,v) = \max_{i \in R} s_i \cdot \mathbb{I}_{\text{inside}}(u,v | \mathcal{B}_i)$$
ensuring that geometric constraints are applied only to regions with low semantic uncertainty.

### Loss & Training

The overall optimization objective is:
$$\min_\theta \sum_{x \in I} \mathcal{L}_{CFL}(x) + \lambda \sum_{(u,v) \in I} \mathcal{M}(u,v) \cdot \mathcal{L}_{NCL}(u,v)$$
where $\lambda = 0.7$.

## Key Experimental Results

### Main Results: KITTI-C (MonoFlex Baseline, Severity 5)

| Method | Car Avg | Ped. Avg | Cyc. Avg |
|--------|---------|----------|----------|
| MonoFlex (no adaptation) | 4.54 | 0.88 | 0.83 |
| TENT | 19.68 | 6.30 | 4.62 |
| EATA | 20.03 | 6.41 | 4.71 |
| DeYO | 20.30 | 6.50 | 4.65 |
| MonoTTA | 20.87 | 6.72 | 4.77 |
| **DUO (Ours)** | **22.97** | **7.19** | **5.10** |

Average improvement on the Car category: +2.1 AP₃D|R₄₀.

### Main Results: KITTI-C (MonoGround Baseline)

Car category: DUO achieves 24.73 Avg, compared to 22.57 for MonoTTA (+2.2 gain).

### nuScenes Real-World Scenarios (MonoFlex)

| Task | Source | TENT | MonoTTA | **DUO** |
|------|--------|------|---------|---------|
| D→N | 1.53 | 3.33 | 6.92 | **9.05** |
| N→D | 2.75 | 3.45 | 3.68 | **5.41** |
| S→R | 6.86 | 8.53 | 9.47 | **11.54** |
| R→S | 10.91 | 11.61 | 12.55 | **13.21** |

Average improvement over existing methods: +18%.

### Ablation Study

| CFL | NCL | Guided M | Car | Ped. | Cyc. | Avg |
|-----|-----|----------|-----|------|------|-----|
| ✗ | ✗ | ✗ | 4.54 | 0.88 | 0.83 | 2.08 |
| ✔ | ✗ | ✗ | 20.98 | 6.60 | 4.32 | 10.63 |
| ✗ | ✔ | ✔ | 16.49 | 6.23 | 4.87 | 9.20 |
| ✔ | ✔ | ✔ | **22.97** | **7.19** | **5.10** | **11.75** |

### Key Findings

- NCL alone (without semantic guidance) yields only marginal and unstable improvements; it must be combined with the semantic mask $\mathcal{M}$ to be effective.
- CFL substantially improves the detection score distribution for low-confidence objects, addressing the "high-score bias" of entropy minimization.
- Normal field constraints lead to consistent uncertainty reduction across all depth heads, preventing the model collapse caused by direct optimization.
- When the two branches are combined, semantic and geometric uncertainty decrease most rapidly in tandem, validating the effectiveness of the complementary cycle.

## Highlights & Insights

1. **Theoretical Innovation**: This work is the first to introduce the Legendre-Fenchel duality from convex optimization into TTA loss design, deriving a label-free conjugate focal loss with both theoretical guarantees and empirical effectiveness.
2. **Dual Uncertainty Perspective**: The paper clearly distinguishes between semantic and geometric uncertainty in M3OD and demonstrates that both are amplified under domain shift while exhibiting complementarity.
3. **Observation-Driven Design**: Each design choice is grounded in explicit empirical observations — CFL addresses low-score object neglect, while NCL combined with semantic guidance resolves spatial perception collapse.
4. **Zero Hyperparameter Overhead**: The $\alpha$ and $\gamma$ parameters of CFL can be directly reused from source-stage training, eliminating the need for target-domain tuning.
5. **Simplicity and Efficiency**: Normal field computation relies solely on Sobel operators without requiring additional training or data, making the approach suitable for real-time TTA.

## Limitations & Future Work

- Validation is limited to two baselines (MonoFlex and MonoGround); applicability to more modern 3D detectors (e.g., Transformer-based methods) remains to be explored.
- The corruption types in KITTI-C are synthetically generated and may not fully capture the diversity of real-world domain shifts.
- The normal field constraint depends on depth map quality and may be limited when depth estimation itself severely degrades under extreme domain shifts.
- The framework is tailored to M3OD; extension to other 3D vision tasks (e.g., 3D semantic segmentation, BEV perception) is not discussed.

## Related Work & Insights

- **Monocular 3D Detection**: MonoDLE and PGD identify depth estimation as the bottleneck; MonoFlex integrates multiple depth predictions; MonoGround exploits ground plane priors; MonoCD leverages multi-head complementarity.
- **Test-Time Adaptation**: TENT pioneers entropy minimization; SAR incorporates sharpness-aware optimization; DeYO introduces probability shift under augmentation; ReCAP models regional uncertainty; MonoTTA separately optimizes uncertainty for positive and negative classes.
- **Uncertainty Estimation**: Semantic uncertainty (prediction entropy) and geometric uncertainty (depth uncertainty regression) each have rich literature in their respective domains, but their joint optimization in the TTA setting is introduced for the first time in this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Deriving an unsupervised loss from a convex optimization perspective is a novel theoretical contribution; the dual uncertainty viewpoint fills a gap in 3D TTA.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — The Legendre-Fenchel duality derivation is rigorous, the normal field constraint design is well-motivated, and the complementary cycle is supported by both theory and experiments.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 13 corruption types, 4 real-world scenarios, and 2 baseline models with thorough ablations; diversity of test sets and detectors could be further expanded.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, and the observation → method → validation narrative flows logically.
- **Recommendation**: ⭐⭐⭐⭐ — Represents a significant contribution to 3D TTA with solid theoretical and experimental foundations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations](monosowa_scalable_monocular_3d_object_detector_without_human_annotations.md)
- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)

</div>

<!-- RELATED:END -->
