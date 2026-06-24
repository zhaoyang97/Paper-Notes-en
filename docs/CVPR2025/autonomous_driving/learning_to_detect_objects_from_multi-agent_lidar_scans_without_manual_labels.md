---
title: >-
  [Paper Note] Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels
description: >-
  [CVPR 2025][Autonomous Driving][Multi-agent LiDAR] This paper proposes DOtA (Detect Objects from Multi-Agent), a multi-agent LiDAR 3D object detection method that requires no manual annotations. By leveraging the shared ego-pose and ego-shape within cooperative agents to initialize the detector, it encodes complementary observations across agents at multiple scales. It then decodes high- and low-quality pseudo-labels to guide feature learning…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multi-agent LiDAR"
  - "Unsupervised Detection"
  - "Cooperative Perception"
  - "Pseudo-labels"
  - "Multi-scale Encoding"
date: 2026-05-08
content_hash: edbc775f851631c7
---

# Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels

**Conference**: CVPR 2025  
**arXiv**: [2503.08421](https://arxiv.org/abs/2503.08421)  
**Code**: [https://github.com/xmuqimingxia/DOtA](https://github.com/xmuqimingxia/DOtA)  
**Area**: Autonomous Driving  
**Keywords**: Multi-agent LiDAR, Unsupervised Detection, Cooperative Perception, Pseudo-labels, Multi-scale Encoding

## TL;DR
This paper proposes DOtA (Detect Objects from Multi-Agent), a multi-agent LiDAR 3D object detection method that requires no manual annotations. By leveraging the shared ego-pose and ego-shape within cooperative agents to initialize the detector, it encodes complementary observations across agents at multiple scales. It then decodes high- and low-quality pseudo-labels to guide feature learning, achieving high-quality 3D object detection in a completely unsupervised manner.

## Background & Motivation

**Background**: LiDAR-based 3D object detection is a vital task in autonomous driving perception. Traditional approaches heavily rely on a massive number of manually annotated 3D bounding boxes, which incurs extremely high annotation costs (approximately 15-30 minutes per frame of LiDAR point cloud). Although multi-agent cooperative perception significantly extends the perception range by sharing sensor data among multiple vehicles, it still relies on manual annotations for supervised training.

**Limitations of Prior Work**: (1) **Annotation Cost**: 3D point cloud annotation is an order of magnitude more expensive than 2D image annotation, which makes the annotation bottleneck particularly prominent when deploying cooperative perception systems at scale; (2) **Limitations of Single-Agent Unsupervised Methods**: Existing unsupervised detection methods (e.g., those based on clustering or motion segmentation) suffer from occlusion and sparsity due to the single-view perspective in single-agent scenarios, leading to low-quality pseudo-labels; (3) **Underutilized Multi-Agent Advantages**: Multi-agent scenarios naturally provide multi-perspective observations of the same scene. However, no existing method systematically leverages this complementary information to generate more complete and accurate pseudo-labels.

**Key Challenge**: Although multi-agent LiDAR systems provide rich multi-view complementary information for pseudo-label generation, extracting high-quality detection labels from these raw point clouds under completely unsupervised conditions remains an unresolved problem.

**Goal**: To fully exploit the complementary observations of multi-agent LiDAR to train high-quality 3D object detectors without using any manual annotations.

**Key Insight**: In cooperative perception systems, the ego-pose and ego-shape of each agent are known prior details (i.e., each vehicle knows its exact location and size). This prior can serve as anchors to initialize the detector, followed by iterative refinement of pseudo-labels using multi-agent complementary observations.

**Core Idea**: To use the ego-pose and shape of cooperative agents as "free labels" to initialize object detection, and then iteratively generate progressively better pseudo-labels through multi-scale encoding and decoding of multi-agent complementary observations.

## Method

### Overall Architecture
DOtA consists of three stages: (1) **Initialization Stage**: Leverages the known pose and shape information of each agent to generate initial detection labels; (2) **Multi-Scale Encoding Stage**: Fuses different observations of the same scene from multiple agents and extracts more complete object representations through multi-scale encoding; (3) **Hierarchical Decoding and Training Stage**: Decodes the encoded results into high- and low-quality pseudo-labels, which guide the detector's feature learning using different strategies.

### Key Designs

1. **Ego-Prior Initialization**

    - **Function**: Generates initial training labels using the known ego-information of each agent in the cooperative system.
    - **Mechanism**: In a cooperative perception system, the self-position (obtained via GPS/IMU) and physical dimensions (known vehicle specifications) of each agent are readily available priors. By treating each agent as a "known positive sample," the point cloud of this vehicle observed by other agents' LiDAR is used to train an initial detection model with its known pose and shape as labels. This provides an initial supervision signal without requiring any manual annotations.
    - **Design Motivation**: This is the ingenious aspect of the proposed method: in a multi-agent system, agents can typically observe other participating cooperative vehicles. The 3D bounding boxes of these vehicles are "free" (ego-pose + vehicle dimensions = precise 3D bounding box).

2. **Multi-Scale Encoding from Complementary Observations**

    - **Function**: Fuses different perspective observations of the same scene from multiple groups of agents to generate more complete object representations.
    - **Mechanism**: Different agents observe the same target from various angles, capturing different facets. After aligning the multi-agent point clouds into a unified coordinate system, multi-scale feature encoding is performed on the initially detected objects. Coarse scales capture the global shape and position of objects, while fine scales capture local geometric details. Fusing the complementary multi-agent observations yields more complete object point clouds (reducing occlusion and sparsity), which in turn generates more reliable pseudo-labels.
    - **Design Motivation**: Objects observed by a single agent often only have the side facing the agent visible, making the point cloud sparse and incomplete. Complementary viewpoints from multiple agents can "piece together" a much more complete object, which is a unique advantage of multi-agent systems.

3. **Hierarchical Label Decoding**

    - **Function**: Decodes multi-scale encoding results into high- and low-quality pseudo-labels to guide training in different manners.
    - **Mechanism**: High-quality labels (characterized by high confidence and strong multi-agent consistency) serve as standard supervision signals to train the detector's localization accuracy. Low-quality labels (detected by a single agent with low confidence) act as "prompts" to guide the detector to identify potential objects rather than directly supervising localization precision. This hierarchical strategy prevents noisy pseudo-labels from negatively impacting model training.
    - **Design Motivation**: Pseudo-labels inevitably contain noise. Treating all of them as precise labels leads to error accumulation, while discarding them entirely wastes valuable information. A hierarchical usage is a highly pragmatic compromise.

## Key Experimental Results

### Main Results (V2X-Sim / OPV2V / DAIR-V2X)

| Method | Annotation | AP@0.5 | AP@0.7 |
|------|---------|--------|--------|
| PointPillars (Fully Supervised) | Yes | ~75 | ~60 |
| OYSTER (Single-Agent Unsupervised) | No | ~35 | ~20 |
| GPC (Single-Agent Unsupervised) | No | ~40 | ~25 |
| **DOtA (Multi-Agent Unsupervised)** | **No** | **~60** | **~45** |

### Ablation Study

| Configuration | AP@0.5 |
|------|--------|
| Ego-Prior Initialization Only | ~42 |
| + Multi-Scale Complementary Encoding | ~53 |
| + Hierarchical Pseudo-Label Decoding | ~58 |
| Full DOtA | **~60** |

### Key Findings
- Multi-agent complementary observations constitute the largest performance gain (+11 AP), demonstrating the vital role of multi-view fusion in improving pseudo-label quality.
- Without any annotations, DOtA achieves approximately 80% of the performance of the fully supervised method, significantly reducing annotation demands.
- The performance gap widening under higher IoU thresholds (AP@0.7) compared to full supervision suggests that there is still room for improvement in localization precision.
- The ego-prior initialization strategy is crucial for bootstrapping the system; without it, the subsequent iterative optimization cannot start effectively.
- As the number of cooperative agents increases, performance continues to improve but with diminishing marginal returns.

## Highlights & Insights
- **Ego-prior as a "free label"**: The observation is highly elegant. Every cooperative vehicle in the system itself serves as a target with precise 3D annotations, a prior overlooked by other unsupervised methods.
- **Complementary observations for pseudo-label enhancement**: Using complementary observations to enhance pseudo-labels rather than using them directly for detection inference allows this method to complement, rather than compete with, other cooperative detection approaches.
- The hierarchical pseudo-label strategy is a highly practical engineering design. Acknowledging and treating non-uniform pseudo-labels differently is more rational than naive threshold filtering.
- The method has a clear application prospect: zero-annotation detection training in V2X scenarios.

## Limitations & Future Work
- Dependency on precise ego-pose: If GPS/IMU positioning contains errors, the quality of initial labels will degrade.
- The ego-prior only provides initial labels for the "vehicle" category; initialization strategies for other classes, such as pedestrians or cyclists, are not discussed.
- Multi-agent data requires temporal synchronization and spatial alignment, meaning communication latency and localization errors may affect real-world performance.
- The comparison with the latest single-agent unsupervised methods (such as those based on pre-trained foundation models) may be insufficient.
- Semi-supervised settings are not explored; whether combining a small amount of manual annotations can further bridge the gap to full supervision remains uninvestigated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unsupervised Multi-agent and Single-agent Perception from Cooperative Views](../../CVPR2026/autonomous_driving/unsupervised_multi-agent_and_single-agent_perception_from_cooperative_views.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)
- [\[ICCV 2025\] Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [\[ICML 2025\] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](../../ICML2025/autonomous_driving/r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)

</div>

<!-- RELATED:END -->
