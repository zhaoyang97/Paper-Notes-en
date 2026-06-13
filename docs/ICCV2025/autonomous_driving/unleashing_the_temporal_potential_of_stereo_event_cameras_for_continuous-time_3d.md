---
title: >-
  [Paper Note] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception
description: >-
  [ICCV 2025][Autonomous Driving][Event Camera] This paper proposes the first 3D object detection framework relying solely on stereo event cameras. Through a semantic-geometric dual filtering module and object-centric ROI…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Event Camera"
  - "Stereo Vision"
  - "3D Object Detection"
  - "Continuous-Time Perception"
  - "Semantic-Geometric Dual Filtering"
date: 2026-05-08
content_hash: a4240c73dc89f3cb
---

# Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception

**Conference**: ICCV 2025
**arXiv**: [2508.02288](https://arxiv.org/abs/2508.02288)  
**Code**: [GitHub](https://github.com/mickeykang16/Ev-Stereo3D)  
**Area**: Autonomous Driving
**Keywords**: Event Camera, Stereo Vision, 3D Object Detection, Continuous-Time Perception, Semantic-Geometric Dual Filtering

## TL;DR
This paper proposes the first 3D object detection framework relying solely on stereo event cameras. Through a semantic-geometric dual filtering module and object-centric ROI alignment, it enables continuous-time 3D detection during blind time periods, significantly outperforming methods that depend on synchronized sensors (Ev-3DOD) in dynamic large-motion scenarios. Its pedestrian AP3D even surpasses methods that use LiDAR+RGB+Event.

## Background & Motivation
3D object detection is a fundamental task in autonomous driving. Although LiDAR and RGB cameras are widely used, they are constrained by fixed frame rates (10–20 Hz), resulting in perceptual blind time periods in high-speed scenarios where no data is collected between frames, leading to detection latency.

Event cameras offer a solution through their asynchronous nature and high temporal resolution:

**Limitations of Ev-3DOD**: Recent methods integrate event cameras with LiDAR/RGB for continuous-time detection, but they heavily rely on 3D geometric information from the previous LiDAR frame during blind time. When rapid motion causes drastic scene changes, historical geometric information becomes inapplicable and errors accumulate over time, leading to frequent detection failures.

**Newly Appearing Objects**: Methods relying on synchronized sensors can only detect new objects during active time (when LiDAR/RGB is available); objects that appear in the field of view during blind time cannot be captured.

**Sparsity of Event Data**: Event data lacks semantic and geometric information, making accurate classification and regression for 3D detection directly challenging.

**Core Idea**: Completely abandon synchronized sensors and compute 3D geometric information using only stereo event cameras → mutually enhance complementary information through semantic-geometric dual filtering → improve regression accuracy via object-centric ROI alignment → achieve 3D detection at arbitrary time points during blind time.

## Method

### Overall Architecture
The system consists of four components: (1) a geometric plane-sweep volume (constructing a 3D geometric volume from stereo event data) → (2) semantic-geometric dual filtering (mutually enhancing semantic and depth information) → (3) a global 3D detector (anchor-based detection in BEV) → (4) object-centric ROI alignment (local fine-grained regression).

The input consists of left and right event streams within a time window $\Delta\tau$, converted into a voxel grid representation $E_L, E_R$. Detection at arbitrary time points is achieved by adjusting $\Delta\tau$.

### Key Designs
1. **Geometric Plane-Sweep Volume**:

    - Function: Constructs a 3D geometric volume from stereo event data.
    - Mechanism: A modified PSMNet is used as the feature extractor with an additional feature head to disentangle semantic features $F^{sem}$ and geometric features $F^{geo}$. The geometric features are used to construct the plane-sweep volume:
    $\mathcal{V}_{geo}(u,v,w) = (F_L^{geo}(u,v) \| F_R^{geo}(u - \frac{fL}{d(w)}, v))$
      A depth probability volume $\mathcal{P}_{geo}$ is generated via 3D convolution + softmax and then projected into a unified 3D voxel space $\mathbf{V}_{geo}^{3D}$.
    - Design Motivation: Disentangling semantic and geometric features allows the network to focus on their respective tasks — semantic features activate well on objects in the 2D plane, while geometric features seek stereo correspondences across the entire scene.

2. **Semantic-Geometric Dual Filtering**:

    - Function: Refines depth using semantic information while enhancing semantic features with depth information.
    - Mechanism:
        - **Semantics-Guided Depth Refinement**: An initial depth $D_{init}$ is computed; a similarity score $S(u,v)$ is computed via semantic feature matching; combined with a depth confidence $C(u,v)$ (variance of the depth probability distribution), a refined depth volume $\tilde{\mathcal{P}}_{geo}$ is generated through neighborhood probability weighting:
       $$W_m(u,v) = S_m(u,v) \cdot \text{sigmoid}(C_m(u,v))$$
       $$\tilde{\mathcal{P}}_{geo}(u,v,w) = \sum_{m=1}^{M} \mathcal{P}_{geo}(u,v,w) \cdot \text{softmax}_m(W_m(u,v))$$
        - **Geometry-Filtered Semantic Volume**: The refined depth $\tilde{D}$ is used to warp right-camera semantic features to the left camera; a Transformer channel-attention mechanism handles occlusion and misalignment:
       $$\tilde{F}_L^{sem} = F_L^{sem} + \text{MLP}(\mathbb{A}) + \mathbb{A}$$
       The semantic volume is projected to 3D using a depth probability mask: $\mathbf{V}_{sem}^{3D}(x,y,z) = \tilde{F}_L^{sem}(u,v) \cdot \tilde{\mathcal{P}}_{geo}(u,v,d^{-1}(z))$
    - Design Motivation: Event data is spatially sparse; neither semantic nor geometric features alone are sufficient. Dual filtering allows the two to complement and mutually enhance each other.

3. **Object-Centric ROI Alignment**:

    - Function: Performs object-level local fine-grained refinement of global detection results.
    - Mechanism: The BEV projection $\mathbf{B}_{sem}^{2D}$ of the semantic volume $\mathbf{V}_{sem}^{3D}$ is used (as semantic features focus on objects and event data provides rich edge information). Global detection bounding boxes are divided into $k \times k$ grids, and ROI pooling is applied on the semantic BEV to obtain object-centric local features $\hat{\mathbf{B}}_{sem}^{2D}$. An MLP predicts local offsets $\Delta P_L$, which are added to the global predictions to yield the final result: $\tilde{P}_G = g_d(P_G, \Delta P_L)$.
    - Design Motivation: Single-stage global regression is limited in accuracy under the ambiguities caused by object motion and ego-motion, necessitating object-level local alignment.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{depth}^{init} + \mathcal{L}_{depth}^{refine} + \mathcal{L}_{2D} + \mathcal{L}_{cls} + \mathcal{L}_{reg}^{global} + \mathcal{L}_{reg}^{local}$$
- $\mathcal{L}_{depth}^{refine}$: smooth L1 loss for refined depth.
- $\mathcal{L}_{reg}^{global/local}$: global/local 3D regression losses.

Training and evaluation are conducted on the DSEC-3DOD dataset with 100 FPS annotations. Event stream slice duration is $\Delta\tau = 10$ ms.

## Key Experimental Results

### Main Results (Blind Time 3D Detection)

| Modality | Method | VEH AP3D Easy↑ | VEH APBEV Easy↑ | PED AP3D Easy↑ | PED APBEV Easy↑ |
|------|------|---------------|----------------|---------------|----------------|
| LiDAR | VoxelNeXt | 12.66 | 31.46 | 10.59 | 12.77 |
| LiDAR+RGB | LoGoNet | 17.65 | 32.55 | 11.66 | 15.09 |
| LiDAR+RGB+Event | Ev-3DOD | **29.53** | **49.31** | 18.42 | **29.06** |
| RGB-Stereo | LIGA | 14.26 | 27.25 | 6.02 | 8.73 |
| **Event-Stereo** | **Ours** | 23.47 | 40.13 | **19.86** | 22.91 |

Without any LiDAR/RGB input, the proposed method surpasses Ev-3DOD in pedestrian AP3D (19.86 vs. 18.42).

### Ablation Study

| Configuration | mAP Easy 3D↑ | mAP Easy BEV↑ | Note |
|------|-------------|--------------|------|
| Geo only (G) | 12.92 | 20.78 | Baseline |
| G + Semantic (S) | 14.60 | 24.85 | +1.68/+4.07 |
| G + SDR | 14.56 | 21.73 | Semantics-guided depth refinement |
| G+S + SDR | 15.95 | 26.12 | Both combined |
| G+S + SDR (DSGF) | 15.85 | 27.78 | Full dual filtering |
| G+S + DSGF + OCRA | **21.66** | **31.58** | +ROI alignment, large gain |

### Motion Scale Experiment (Key)

| Motion Scale | Time Slice | Method | VEH AP3D/BEV | PED AP3D/BEV |
|-------------|-----------|------|-------------|-------------|
| ×2 | ×20 | Ev-3DOD | 15.96/31.13 | 1.47/2.63 |
| ×2 | ×20 | **Ours** | **23.47/40.13** | **19.86/22.91** |
| ×4 | ×20 | Ev-3DOD | 6.36/11.70 | 0.39/0.43 |
| ×4 | ×20 | **Ours** | **22.72/39.81** | **19.04/22.71** |

As the motion scale increases, Ev-3DOD suffers a drastic performance collapse (PED AP3D: 18.42 → 1.47 → 0.39), whereas the proposed method remains stable (19.86 → 19.86 → 19.04).

### Key Findings
- Stereo event cameras demonstrate significant potential for 3D detection during blind time: continuous 3D structure estimation is achievable without LiDAR geometric information.
- Pedestrian detection surpasses Ev-3DOD (LiDAR+RGB+Event), as pedestrian detection requires fine-grained details where the high temporal resolution of event data combined with the complementary semantic feature scheme proves more effective.
- The motion scale experiment reveals a critical difference: methods relying on synchronized sensors degrade severely under large motion, while the asynchronous event-based method remains stable.
- The ROI alignment module contributes the largest gain (mAP 3D: 15.85 → 21.66), likely because the sparsity of event data makes global regression particularly difficult.
- Ev-3DOD performance degrades with temporal distance from active time; the proposed method is not subject to this constraint.
- The method can detect newly appearing objects during blind time, which Ev-3DOD cannot.

## Highlights & Insights
- **First pure event-stereo 3D object detection system**: demonstrates the feasibility of event cameras independently performing 3D detection.
- The **semantic-geometric dual filtering** complementary design is elegant: semantics disambiguate depth estimation, while depth enhances spatial awareness for semantics.
- The **motion scale experiment design** is innovative: large-motion scenarios are simulated via temporal-axis scaling, overcoming the limited motion dynamics of the DSEC-3DOD dataset.
- Continuous-time detection capability (variable $\Delta\tau$) allows the system to adapt to different speed scenarios.
- Surpassing multi-modal fusion methods in pedestrian AP3D is a surprisingly strong result.

## Limitations & Future Work
- Vehicle AP3D still falls short of Ev-3DOD (23.47 vs. 29.53); geometric estimation for large objects still requires LiDAR-level precision.
- Validation is performed on only one dataset (DSEC-3DOD); generalizability requires confirmation on additional datasets.
- The semantic information in event data is inherently weaker than that in RGB, limiting classification performance.
- Computational cost is not reported in detail; it remains questionable whether stereo matching combined with dual filtering meets real-time requirements for autonomous driving.
- The motion scale experiment approximates large-motion scenarios via temporal-axis scaling, which may differ from real large-motion conditions.

## Related Work & Insights
- This work validates that event cameras can serve not merely as auxiliary sensors but as primary sensors for 3D perception.
- The dual filtering concept is generalizable to other multi-modal or multi-task fusion settings.
- The choice to use semantic BEV features (rather than geometric features) for ROI alignment is instructive — object edge information is more important under sparse data conditions.
- This work lays an important foundation for future purely event-driven autonomous driving systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First pure event-stereo 3D detection system; both problem formulation and method design are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation study with a cleverly designed motion scale experiment, but limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, method figures are intuitive, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Demonstrates the transformative potential of event cameras in autonomous driving; open-source release facilitates follow-up research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes](trafficloc_localizing_traffic_surveillance_cameras_in_3d_scenes.md)
- [\[ICCV 2025\] Towards Open-World Generation of Stereo Images and Unsupervised Matching](towards_open-world_generation_of_stereo_images_and_unsupervised_matching.md)
- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](../../AAAI2026/autonomous_driving/rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)

</div>

<!-- RELATED:END -->
