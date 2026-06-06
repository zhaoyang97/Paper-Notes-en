---
title: >-
  [Paper Note] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation
description: >-
  [CVPR 2026][Self-Supervised Learning][Scene Flow] This paper proposes TeFlow — the first method to introduce multi-frame supervision into self-supervised feed-forward scene flow estimation. By constructing a motion candi…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Scene Flow"
  - "Self-Supervised"
  - "Multi-frame Supervision"
  - "Temporal Aggregation"
  - "Feed-forward Network"
  - "Point Cloud"
date: 2026-05-08
content_hash: e02b4da0f0def8b9
---

# TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation

**Conference**: CVPR 2026
**arXiv**: [2602.19053](https://arxiv.org/abs/2602.19053)  
**Code**: [github.com/KTH-RPL/OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow)  
**Area**: Self-Supervised Learning / Autonomous Driving
**Keywords**: Scene Flow, Self-Supervised, Multi-frame Supervision, Temporal Aggregation, Feed-forward Network, Point Cloud

## TL;DR

This paper proposes TeFlow — the first method to introduce multi-frame supervision into self-supervised feed-forward scene flow estimation. By constructing a motion candidate pool via temporal aggregation and aggregating temporally consistent supervision signals through consensus voting, TeFlow achieves a Three-way EPE of 3.57 cm on Argoverse 2 (on par with the optimization-based method Floxels) while maintaining real-time inference (8 s vs. 24 min), representing a 22.3% improvement over SeFlow++.

## Background & Motivation

**Background**: Scene flow estimation aims to predict the 3D motion of each point in LiDAR point clouds. Existing self-supervised methods fall into two categories: (1) **Optimization-based methods** (NSFP, EulerFlow) — which optimize scene-specific models using multi-frame long-horizon constraints, achieving high accuracy but with prohibitive latency (hours to days); (2) **Feed-forward methods** (SeFlow, ZeroFlow) — which achieve efficient single-pass inference, but whose training objectives rely solely on two-frame point correspondences, making them susceptible to unstable supervision signals caused by occlusions, noise, and sparse observations.

**Key Challenge**: Multi-frame supervision has the potential to provide more stable training signals, but naively extending two-frame objectives to multiple frames is ineffective — inter-frame point correspondences vary dramatically, producing inconsistent signals. As shown in Figure 1b of the paper, two-frame supervision signal directions fluctuate violently over time; even when the true motion is smooth, two-frame estimates oscillate sharply due to occlusion and noise.

**Limitations of Prior Work**: ZeroFlow generates pseudo-labels via knowledge distillation from a slow "teacher," requiring 7.2 GPU-months of computation. SeFlow improves the two-frame loss function but remains bounded by the two-frame signal ceiling. Multi-frame architectures (Flow4D, DeltaFlow) are effective under supervised learning, but are still constrained by two-frame objectives in the self-supervised setting.

**Key Insight**: Rather than designing a better two-frame loss, TeFlow **mines temporally consistent motion cues across multiple frames** — constructing a motion candidate pool followed by consensus voting to produce stable multi-frame supervision signals, enabling feed-forward models to fully leverage the temporal modeling capacity of multi-frame architectures under self-supervision for the first time.

## Method

### Overall Architecture

TeFlow builds upon the DeltaFlow multi-frame backbone network. Given 5 LiDAR frames as input (ego-motion aligned), it predicts a residual flow $\mathcal{F}_{res}$. During training, point clouds are first partitioned into static and dynamic regions (provided by DUFOMap), and dynamic points are clustered into clusters $\mathcal{C}_j$ (pre-computed by HDBSCAN). For each dynamic cluster, TeFlow generates reliable supervision targets $\bar{\mathbf{f}}_{\mathcal{C}_j}$ via temporal aggregation, and training is performed using a combination of a static loss and a geometric consistency loss.

### Key Designs

1. **Motion Candidate Generation**
    - Function: Generate diverse motion hypotheses for each dynamic cluster, balancing stability with geometry-driven evidence from data.
    - Mechanism: The candidate pool consists of two sources — (a) **Internal candidates** $\hat{\mathbf{f}}_{\mathcal{C}_j}$: the cluster-averaged flow from the current network prediction, serving as a stable anchor; (b) **External candidates** $\mathbf{f}^{t'}_{\mathcal{C}_j,k}$: for each temporal frame $t'$, the top-K nearest neighbor correspondences with the largest displacements are retrieved, normalized by the temporal interval: $\mathbf{f}^{t'}_{\mathcal{C}_j,k} = \frac{\mathcal{NN}(\mathbf{p}_k, \mathcal{P}_{t',d}) - \mathbf{p}_k}{t' - t}$
    - Design Motivation: Internal candidates prevent training from drifting; external candidates mine true motion cues from multi-frame geometry. Selecting top-K largest displacements filters noisy points. Temporal normalization ensures comparability across candidates from different time intervals.

2. **Consensus Voting**
    - Function: Extract the most reliable motion estimate from the candidate pool, filtering inconsistent noisy candidates.
    - Mechanism: A consensus matrix $\mathbf{M}_{ab} = \mathbf{1}[\cos(\mathbf{f}_a, \mathbf{f}_b) > \tau_{cos}]$ is constructed to measure directional consistency, combined with reliability weights $w_i = \gamma^{m_i}(1 + \|\mathbf{f}_i\|_2^2)$ (temporal decay plus displacement magnitude weighting). The voting score $\mathbf{S} = \mathbf{M}\mathbf{w}$ selects the highest-scoring candidate as the consensus winner $a^\dagger$. The final supervision target is the weighted average of all candidates consistent with the winner's direction: $\bar{\mathbf{f}}_{\mathcal{C}_j} = \frac{\sum_b \mathbf{M}_{a^\dagger b} w_b \mathbf{f}_b}{\sum_b \mathbf{M}_{a^\dagger b} w_b}$
    - Design Motivation: A single candidate is unreliable; voting aggregation exploits majority consistency. Temporal decay $\gamma=0.9$ prioritizes nearby frames; large-displacement candidates are more informative. The resulting consensus signal is substantially more stable than two-frame signals (cf. Figure 1b).

3. **Dynamic Cluster Loss**
    - Function: Supervise dynamic objects of varying sizes equitably, preventing large objects from dominating training due to their higher point counts.
    - Mechanism: A combination of point-level loss (averaged L2 over all points) and cluster-level loss (intra-cluster average followed by cross-cluster average): $\mathcal{L}_{dcls} = \frac{1}{|\mathcal{P}_\mathcal{C}|}\sum_j\sum_{\mathbf{p}_i \in \mathcal{C}_j}\|\hat{\mathbf{f}}_i - \bar{\mathbf{f}}_{\mathcal{C}_j}\|^2_2 + \frac{1}{N_c}\sum_j(\frac{1}{|\mathcal{C}_j|}\sum_{\mathbf{p}_i \in \mathcal{C}_j}\|\hat{\mathbf{f}}_i - \bar{\mathbf{f}}_{\mathcal{C}_j}\|^2_2)$
    - Design Motivation: Point-level loss alone causes small objects such as pedestrians to be overwhelmed (ablation shows a 53% increase on the PED category, Table 5); cluster-level loss alone is insufficient for fine-grained alignment of large objects (OTHER category error increases by 82%). The combination achieves optimal performance.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{dcls} + \mathcal{L}_{static} + \mathcal{L}_{geom}$

- $\mathcal{L}_{static}$: Drives residual flow of static points toward zero.
- $\mathcal{L}_{geom}$: Multi-frame Chamfer distance ensuring warped point clouds align geometrically with neighboring frames.
- Training configuration: Adam optimizer, lr=0.002, batch size=20, 10×RTX 3080, 15 epochs, approximately 15–20 hours.

## Key Experimental Results

### Main Results: Argoverse 2 Test Set Leaderboard

| Method | Type | #Frames | Runtime/seq | Three-way EPE↓ | Dynamic Norm↓ | PED↓ |
|------|------|:---:|:---:|:---:|:---:|:---:|
| NSFP | Optimization | 2 | 60 min | 6.06 | 0.422 | 0.722 |
| EulerFlow | Optimization | all | 1440 min | 4.23 | 0.130 | 0.195 |
| Floxels | Optimization | 13 | 24 min | 3.57 | 0.154 | 0.195 |
| ZeroFlow | Feed-forward | 3 | 5.4 s | 4.94 | 0.439 | 0.808 |
| SeFlow | Feed-forward | 2 | 7.2 s | 4.86 | 0.309 | 0.464 |
| SeFlow++ | Feed-forward | 3 | 10 s | 4.40 | 0.264 | 0.367 |
| **TeFlow** | **Feed-forward** | **5** | **8 s** | **3.57** | **0.205** | **0.253** |

- TeFlow EPE of 3.57 cm matches Floxels (an optimization-based method) while being 150× faster (8 s vs. 24 min).
- Dynamic metric improves 22.3% over SeFlow++; pedestrian category error decreases by 31%.

### Ablation Study: Number of Frames and Loss Terms

| Loss Combination | Dynamic Norm Mean↓ | CAR↓ | PED↓ | Three-way EPE↓ |
|---------|:---:|:---:|:---:|:---:|
| $\mathcal{L}_{geom}$ only | 0.386 | 0.317 | 0.297 | 8.85 |
| $\mathcal{L}_{geom} + \mathcal{L}_{static}$ | 0.458 | 0.321 | 0.481 | 6.37 |
| $\mathcal{L}_{dcls}$ only | 0.303 | 0.254 | 0.285 | 8.53 |
| $\mathcal{L}_{static} + \mathcal{L}_{dcls}$ | 0.313 | 0.233 | 0.296 | 4.84 |
| **All three terms** | **0.265** | **0.198** | **0.295** | **4.43** |

| #Frames | Dynamic Norm Mean↓ | Three-way EPE Mean↓ |
|:---:|:---:|:---:|
| 2 (SeFlow) | 0.408 | 6.35 |
| 2 (TeFlow) | 0.353 | 5.98 |
| 4 | 0.283 | 4.57 |
| **5** | **0.265** | **4.43** |
| 6 | 0.269 | 4.55 |
| 8 | 0.300 | 5.40 |

### Key Findings

- Even with 2 frames, TeFlow outperforms SeFlow by 13.5%, attributable to the candidate pool and cluster-level loss.
- 5 frames is the optimal window; performance slightly degrades at 6 frames and noticeably at 8 frames, suggesting that distant frames introduce noise.
- The dynamic cluster loss alone performs strongly on dynamic objects (Mean 0.303) but yields high static EPE of 8.53, necessitating the static loss.
- Candidate pool ablation: internal only (0.455) < external only (0.321) < combined (0.265), confirming the complementarity of internal anchoring and external geometric evidence.
- State-of-the-art results are also achieved on nuScenes: Dynamic Norm 0.395 vs. SeFlow++ 0.509, with a 33.8% reduction in pedestrian error.

## Highlights & Insights

- **Precise core insight**: The key difficulty in self-supervised feed-forward scene flow is not architecture but supervision signal quality — mining temporal consistency is the right approach.
- **Elegant candidate pool and voting mechanism**: Unreliable multi-source motion estimates are aggregated into reliable signals without additional networks or complex optimization.
- **Simple yet effective cluster-level loss**: A single additional loss term resolves the large-small object imbalance, yielding substantial gains especially for small objects such as pedestrians.
- **Pareto-optimal efficiency–accuracy trade-off**: TeFlow achieves the highest accuracy among real-time methods and the fastest inference among high-accuracy methods, successfully bridging the gap between the two paradigms.

## Limitations & Future Work

- Relies on external modules for static/dynamic segmentation (DUFOMap) and dynamic clustering (HDBSCAN); segmentation errors may propagate through the pipeline.
- Performance degrades beyond 5 frames, suggesting the consensus mechanism does not yet exploit distant frames with sufficient granularity.
- Candidate normalization assumes linear motion, limiting candidate quality for curvilinear motion (e.g., turning vehicles).
- Application of the temporal aggregation strategy at inference time (currently used only during training) remains unexplored.

## Related Work & Insights

- **vs. EulerFlow**: EulerFlow optimizes a continuous ODE and achieves high accuracy (EPE 4.23) but requires 1440 minutes per sequence. TeFlow achieves 3.57 cm in only 8 seconds, making it the only viable high-accuracy solution for real-world deployment.
- **vs. SeFlow/SeFlow++**: These methods improve the two-frame loss but remain bounded by the two-frame signal ceiling. TeFlow improves signal quality at the source.
- **vs. ZeroFlow**: Knowledge distillation requires 7.2 GPU-months to generate pseudo-labels. TeFlow is fully end-to-end self-supervised.
- **Insights**: The multi-frame signal mining paradigm via consensus voting is transferable to other self-supervised visual tasks (optical flow, depth estimation). The cluster-level loss concept is broadly applicable to any 3D perception task involving object scale imbalance.

## Rating

⭐⭐⭐⭐⭐ (5/5)

Overall assessment: TeFlow precisely identifies the core bottleneck of self-supervised feed-forward scene flow (unstable multi-frame supervision signals) and proposes a concise and elegant temporal aggregation solution. Experiments are comprehensive (Argoverse 2 + nuScenes + Waymo), ablations are thorough (number of frames / loss combinations / candidate pool / loss formulations), and both quantitative and qualitative results are convincing. This work is the first to achieve optimization-level accuracy in self-supervised feed-forward scene flow while maintaining real-time efficiency, establishing a new Pareto frontier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](../../ICML2026/self_supervised/a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)
- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)

</div>

<!-- RELATED:END -->
