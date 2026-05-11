---
title: >-
  [Paper Note] A Prediction-as-Perception Framework for 3D Object Detection
description: >-
  [CVPR 2026][Autonomous Driving][3D Perception] Inspired by the brain's predictive perception mechanism, this paper proposes the PAP framework…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "3D Perception"
  - "Object Detection"
  - "Prediction-as-Perception"
  - "nuScenes"
  - "End-to-End"
date: 2026-05-08
content_hash: a095023eb82b42fe
---

# A Prediction-as-Perception Framework for 3D Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.12599](https://arxiv.org/abs/2603.12599)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: 3D Perception, Object Detection, Prediction-as-Perception, Autonomous Driving, nuScenes, End-to-End

## TL;DR

Inspired by the brain's predictive perception mechanism, this paper proposes the PAP framework, which injects trajectory prediction outputs from previous frames as queries into the current frame's perception module, achieving a 10% improvement in tracking accuracy and a 15% speedup in inference on UniAD.

---

## Background & Motivation

**Predictive Perception in the Brain**: Neuroscience research indicates that the brain does not passively receive sensory signals; instead, it continuously generates predictions about future inputs and iteratively refines its internal model via "prediction errors." For example, when tracking a flying bird, humans anticipate the next position before focusing their gaze.

**Absence of Predictive Priors in Existing Perception Models**: Queries in mainstream 3D detection models (Sparse4D, StreamPETR, DETR3D, etc.) are either randomly initialized each frame or propagated temporally in a straightforward manner, without leveraging explicit trajectory prediction results to guide current-frame perception.

**Disconnection Between Perception and Prediction**: In the conventional detect→track→predict pipeline, each module is trained independently, causing errors to accumulate stage by stage. Even end-to-end models typically employ a unidirectional information flow (perception→prediction), lacking a feedback loop from prediction back to perception.

**Inefficiency of Random Queries**: Attention-based detectors generate a large number of random queries per frame, the vast majority of which are far from actual object locations, resulting in slow convergence and wasted computation.

**Loss of Temporal Cues**: Randomly initialized queries cannot carry knowledge of object motion trends from prior frames, making ID switches more likely during tracking.

**Research Hypothesis**: Incorporating predicted future positions from the prediction module as part of the next frame's perception queries can simultaneously improve detection accuracy and inference efficiency.

---

## Method

### Overall Architecture

The PAP (Prediction-As-Perception) framework consists of a **perception module** and a **prediction module**, which exchange information via queries to form a closed-loop iterative pipeline:

> Current-frame image + previous-frame prediction queries → Perception module → Detection/tracking result queries → Prediction module → Future position queries → Stored in query bank → Retrieved by next-frame perception module

For the first frame, where no historical predictions are available, all queries are initialized randomly.

### Key Design 1: Injecting Prediction Queries into the Perception Module

- **Function**: In each frame's perception module, prediction queries output by the previous frame's prediction module replace some or all random queries.
- **Mechanism**: Predicted coordinates are mapped via an embedding layer to match the dimension of perception queries and then concatenated directly, formulated as $q_i^T \in (q_{random}^T \cup q_{predict}^{T-1})$, after which they are fed into the reference point network $c_i^T = \varnothing^{ref}(q_i^T)$.
- **Design Motivation**: Prediction queries naturally reside near regions where objects are likely to appear, substantially reducing futile search compared to random queries, while preserving temporal motion cues that benefit tracking continuity.

### Key Design 2: Prediction Module and Query Embedding

- **Function**: Detection result queries output by the perception module are passed to the prediction module, which outputs multi-step future position coordinates that are then embedded into queries usable by the next frame.
- **Mechanism**: $c_{predict}^T = \text{PRED}(\text{PECP}(c_i^T))$, $q_{predict}^T = \phi^{embd}(c_{predict}^T)$, where $\phi^{embd}$ is a linear embedding layer.
- **Design Motivation**: This design decouples the choice of prediction module — any module capable of outputting future coordinates can be integrated into PAP without modifying its internal structure or loss function.

### Key Design 3: Integration with UniAD

- **Function**: Prediction queries are taken from UniAD's MotionFormer output, aligned in dimension, and fed together with Track Queries into the TrackFormer.
- **Mechanism**: Since UniAD already facilitates inter-module interaction via queries, PAP only requires adding one feedback path from MotionFormer → TrackFormer, leaving the Planning module and all other losses unchanged.
- **Design Motivation**: By leveraging UniAD's existing end-to-end architecture with minimal intrusion, the PAP concept is validated under fair experimental conditions (all hyperparameters identical to the original model).

---

## Loss & Training

- The perception module's loss is kept identical to that of the original model (TrackFormer in UniAD); learning of prediction queries is driven by back-propagation through the joint perception and prediction loss.
- All training hyperparameters are identical to those of the original UniAD, ensuring a fair comparison.
- Training environment: 4× A100 GPUs, 64-core CPU, 256 GB RAM.
- Training time is reduced from 91h to 78h (↓14%), as prediction queries accelerate detection convergence.

---

## Key Experimental Results

### Table 1: Overall Comparison of UniAD vs. UniAD+PAP on nuScenes val

| Metric | UniAD | UniAD+PAP | Change |
|--------|-------|-----------|--------|
| AMOTA ↑ | 0.359 | **0.395** | +10.0% |
| AMOTP ↓ | 1.32 | **1.22** | -7.6% |
| Recall ↑ | 0.467 | **0.493** | +5.6% |
| IDS ↓ | 906 | **826** | -8.8% |
| Training Time | 91h | **78h** | -14.3% |
| FPS ↑ | 14 | **16** | +14.3% |

### Table 2: Per-Category Performance of UniAD+PAP

| Category | AMOTA | AMOTP | Recall | IDS |
|----------|-------|-------|--------|-----|
| Bicycle | 0.372 | 1.297 | 0.453 | 15 |
| Bus | 0.465 | 1.225 | 0.535 | 8 |
| Car | **0.613** | **0.744** | **0.667** | 405 |
| Motor | 0.438 | 1.253 | 0.500 | 24 |
| Pedestrian | 0.411 | 1.192 | 0.487 | 342 |
| Trailer | 0.330 | 1.551 | 0.201 | 4 |
| Truck | 0.411 | 1.267 | 0.611 | 28 |

The Car category achieves the best metrics (AMOTA 0.613), while the Pedestrian category records the highest IDS (342), reflecting the greater randomness of pedestrian motion and the associated prediction difficulty.

---

## Highlights & Insights

- **Biologically Inspired and Elegantly Simple**: Adding a single "prediction→perception" feedback path yields consistent improvements across all metrics, with a clear and intuitive rationale.
- **Plug-and-Play**: Both the perception and prediction modules can be replaced with stronger off-the-shelf models, endowing the framework with high generality.
- **Simultaneous Speedup**: Replacing random queries with prediction queries reduces futile attention computation, improving FPS by 14% and reducing training time by 14% — a rare outcome in model improvements that typically incur additional computational cost.
- **No Additional Supervision**: No new annotations or auxiliary tasks are required; learning of prediction queries is entirely driven by the existing losses.

---

## Limitations & Future Work

1. **Validated Only on UniAD**: The perception and prediction modules of UniAD are not state-of-the-art; whether PAP maintains its gains on stronger baselines such as Sparse4Dv3 and StreamPETR remains unclear.
2. **Lack of Ablation Studies**: The effects of key hyperparameters — such as the proportion of prediction queries, query bank size, and prediction horizon — are not analyzed.
3. **Single Dataset**: Experiments are conducted solely on nuScenes; generalization to larger-scale datasets such as Waymo and Argoverse2 has not been verified.
4. **First-Frame Degradation**: All queries for the first frame are initialized randomly, offering no benefit from PAP. This has limited impact in long sequences but warrants attention in short-sequence scenarios.
5. **Prediction Error Propagation**: If the prediction module produces large deviations, the injected queries may mislead the perception module; no mechanism currently filters queries based on prediction confidence.

---

## Related Work & Insights

- **BEV Detection (BEVDet, BEVDepth)**: These methods lift features to 3D via depth estimation, but explicit depth estimation is prone to inaccuracies; PAP follows a query-based route and is complementary to BEV approaches.
- **Query-Based Detection (DETR3D, PETR, Sparse4D)**: The PAP framework can be directly applied to these models by replacing random queries with prediction queries.
- **End-to-End Autonomous Driving (UniAD)**: PAP further tightens the perception–prediction–planning closed loop.
- **Trajectory Prediction (THOMAS, AutoBot, GoHome)**: These models can serve directly as the prediction module within PAP.
- **Broader Implications**: The approach can be extended to occupancy prediction (using historical occupancy flow predictions to initialize the current-frame occupancy decoder queries) and 4D scene flow estimation, among other tasks.

---

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐ |
| Experimental Thoroughness | ⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐ |

## Related Work & Insights

| Method | Perception→Prediction | Prediction→Perception | End-to-End | Temporal Query |
|--------|-----------------------|-----------------------|------------|----------------|
| DETR3D | ✗ | ✗ | ✗ | ✗ |
| StreamPETR | ✓ | ✗ | ✗ | Propagation |
| Sparse4Dv3 | ✓ | ✗ | ✗ | Propagation |
| UniAD | ✓ | ✗ | ✓ | Propagation |
| **UniAD+PAP** | ✓ | **✓** | ✓ | **Predictive** |

Unlike the temporal query propagation in StreamPETR and Sparse4D, PAP's queries are processed by an explicit trajectory prediction module, incorporating reasoning about future positions rather than merely continuing past features. The original UniAD design routes information unidirectionally from perception to prediction and planning; PAP closes the loop by adding a prediction→perception feedback path, making the end-to-end system more complete.

## Inspirations & Connections

1. **Extension to Occupancy Prediction**: Historical occupancy flow predictions can serve as initial queries for the current-frame occupancy decoder, reducing the search space in dense prediction.
2. **Extension to 4D Scene Flow**: In scene flow estimation, motion predictions from previous frames can initialize the matching search window for the current frame, reducing computational cost.
3. **Integration with World Models**: Replacing PAP's prediction module with a more powerful world model (e.g., OccWorld) can provide higher-quality prediction queries.
4. **Query Confidence Filtering**: PAP currently accepts all prediction queries unconditionally; incorporating prediction uncertainty estimation would enable filtering of low-quality queries and further improve robustness.
5. **Multi-Modal Fusion**: The PAP framework is not limited to vision-only settings; LiDAR-camera fusion detectors (e.g., BEVFusion) can equally benefit from the prediction feedback path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] On the Feasibility and Opportunity of Autoregressive 3D Object Detection](on_the_feasibility_and_opportunity_of_autoregressive_3d_object_detection.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)

</div>

<!-- RELATED:END -->
