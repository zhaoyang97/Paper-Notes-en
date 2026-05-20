---
title: >-
  [Paper Note] OpenBox: Annotate Any Bounding Boxes in 3D
description: >-
  [NeurIPS 2025][Autonomous Driving][3D auto-annotation] This paper proposes OpenBox, a two-stage automatic 3D bounding box annotation pipeline that first maps instance-level information from 2D visual foundation models to…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "3D auto-annotation"
  - "open-vocabulary"
  - "visual foundation models"
  - "point cloud"
date: 2026-05-08
content_hash: 4024f84a073faffa
---

# OpenBox: Annotate Any Bounding Boxes in 3D

**Conference**: NeurIPS 2025
**arXiv**: [2512.01352](https://arxiv.org/abs/2512.01352)  
**Code**: Available (to be released)  
**Area**: Autonomous Driving / 3D Object Detection
**Keywords**: 3D auto-annotation, open-vocabulary, visual foundation models, point cloud, autonomous driving

## TL;DR

This paper proposes OpenBox, a two-stage automatic 3D bounding box annotation pipeline that first maps instance-level information from 2D visual foundation models to 3D point clouds via cross-modal instance alignment, then adaptively generates high-quality 3D bounding boxes based on the physical state of each object (static rigid / dynamic rigid / deformable), without requiring any self-training iterations.

## Background & Motivation

**Background**: 3D object detection is a core component of autonomous driving, yet large-scale 3D annotation is prohibitively costly. Unsupervised and open-vocabulary 3D object detection has attracted increasing attention in recent years.

**Limitations of Prior Work**:
   - Existing unsupervised methods (MODEST, OYSTER, CPD) treat all objects uniformly during bounding box generation, ignoring the physical properties of objects (rigid vs. deformable, static vs. dynamic), leading to poor annotation quality.
   - Most methods require multiple rounds of self-training iterations to refine annotations, incurring substantial computational overhead.
   - Multi-modal methods such as LiSe fuse 3D boxes from different modalities at the output level, lacking geometric alignment.

**Key Challenge**: LiDAR provides precise geometry but lacks semantics, while 2D images are semantically rich but lack 3D information.

**Goal**: Automatically generate high-quality, multi-category 3D bounding box annotations without manual labeling or self-training iterations.

**Key Insight**: Leverage 2D visual foundation models (Grounding DINO + SAM2) for strong instance-level semantic information, combined with physical-state classification to enable adaptive box generation.

**Core Idea**: Cross-modal instance alignment combined with physical-state-aware adaptive 3D bounding box generation.

## Method

### Overall Architecture

OpenBox consists of two main stages:

**Stage 1: Cross-modal Instance Alignment**
- Extracts instance-level features from 2D images and maps them to 3D point clouds
- Employs Context-aware Refinement to improve the quality of instance point clouds

**Stage 2: Adaptive 3D Bounding Box Generation**
- Classifies instances by physical type and generates bounding boxes with tailored strategies

### Key Designs

1. **Instance-level Feature Extraction and Back-projection**

    - **Function**: Maps 2D detection and segmentation results into 3D point cloud space.
    - **Mechanism**:
        - Grounding DINO is used for open-vocabulary 2D detection, yielding bounding boxes $\mathcal{B}$ and category labels $\mathcal{C}$
        - SAM2 is used for instance segmentation, yielding segmentation masks $\mathcal{M}$ and tracking IDs $\mathcal{T}$
        - 3D points are projected onto 2D masks via camera projection matrix $\Pi_j$ to obtain instance-level point clouds
        - Adaptive erosion is applied to handle mask boundary noise

2. **Context-aware Refinement**

    - **Function**: Addresses noisy points caused by projection errors.
    - **Design Motivation**: LiDAR points are frequently projected onto background objects (e.g., guardrails, walls) that occlude foreground instances, leading to inaccurate back-projection.
    - **Mechanism**:
        - Ground-removed raw LiDAR point clouds are clustered with HDBSCAN to obtain $\{\mathcal{R}_1, ..., \mathcal{R}_{N'}\}$
        - The bidirectional nearest-neighbor inclusion ratio between each LiDAR cluster $\mathcal{R}_k$ and instance point cloud $\mathcal{F}_i$ is computed
        - A cluster is retained when mutual overlap ratios satisfy thresholds $\alpha, \beta$; otherwise it is discarded:
    $\frac{|\{p \in \mathcal{R}_k \mid \text{dist}(p, \mathcal{F}_i) < \delta\}|}{|\mathcal{R}_k|} > \alpha$

3. **Physical-type Classification and Adaptive Box Generation**

    - **Function**: Generates 3D bounding boxes with distinct strategies based on the physical properties of each object.
    - **Classification**: ChatGPT is used to determine rigidity/deformability from semantic category; the PP score is used to estimate motion state (static/dynamic).
    - **Three strategies by instance type**:
        - **Static rigid bodies**: Multi-frame point cloud aggregation → SDF surface reconstruction → surface-aware noise filtering (vertex-level voting) → surface-normal-assisted size adjustment → 3D–2D IoU alignment for optimal box selection
        - **Dynamic rigid bodies**: Single-frame point cloud → orientation estimation from inter-frame position differences → visibility-guided box expansion (ray–normal dot product to determine expansion direction) → standard size statistical constraints
        - **Deformable objects (pedestrians, cyclists)**: Single-frame point cloud → closeness-to-edge algorithm to tightly fit the visible region

4. **Surface-aware Refinement**

    - **Function**: Further denoises the aggregated point cloud for static rigid bodies.
    - **Mechanism**: SDF (VDBFusion) is used to reconstruct surface $\mathbf{S}$; for each vertex, the counts of nearby foreground and background points are tallied; vertices dominated by foreground points are retained to form the refined surface $\mathbf{S}_{\text{ref}}$.

### Loss & Training

OpenBox is an annotation pipeline and does not involve end-to-end training. The generated annotations are used to train downstream 3D detectors:
- WOD: Voxel R-CNN
- Lyft: PointRCNN
- nuScenes: CenterPoint
- Built upon OpenPCDet and MMDetection3D frameworks

## Key Experimental Results

### Main Results

#### WOD Validation Set (AP_3D, L1)

| Method | Modality | Vehicle IoU0.5/0.7 | Pedestrian IoU0.3/0.5 | Cyclist IoU0.3/0.5 |
|--------|----------|-------------------|----------------------|--------------------|
| DBSCAN | LiDAR | 2.32/0.29 | 0.51/0.00 | 0.28/0.03 |
| MODEST | LiDAR | 18.51/6.46 | 11.83/0.17 | 1.47/1.14 |
| OYSTER | LiDAR | 30.48/14.66 | 4.33/0.18 | 1.27/0.33 |
| CPD | LiDAR | 57.79/37.40 | 21.91/16.31 | 5.83/5.06 |
| **OpenBox*** | LiDAR+Cam | **70.49/32.41** | **57.95/17.11** | 20.81/2.15 |
| Human | — | 93.31/75.70 | 87.25/77.93 | 58.84/54.88 |

#### Lyft Validation Set (AP_3D, class-agnostic, IoU=0.25)

| Method | 0–30m | 30–50m | 50–80m | 0–80m |
|--------|-------|--------|--------|-------|
| MODEST | 45.4 | 10.8 | 0.4 | 18.0 |
| LiSe | 54.0 | 22.8 | 1.2 | 27.5 |
| **OpenBox** | **62.3** | **50.6** | **19.5** | **43.3** |
| Human | 82.6 | 70.3 | 49.6 | 69.1 |

#### nuScenes Validation Set (AP_3D)

| Method | Car | Pedestrian | Cyclist |
|--------|-----|-----------|---------|
| UNION | 30.1 | 41.6 | 0.0 |
| **OpenBox** | **40.9** | **62.7** | **5.2** |

### Ablation Study

#### Point-level Refinement Ablation (WOD Vehicle, AP_3D@IoU=0.4)

| Surface-aware | Context-aware | AP_3D |
|:---:|:---:|:---:|
| ✓ | | 30.34 |
| | ✓ | 32.52 |
| ✓ | ✓ | **38.65** |

#### Box-level Refinement Ablation

| Visibility-based | 3D–2D IoU | AP_3D |
|:---:|:---:|:---:|
| ✓ | | 30.49 |
| | ✓ | 34.71 |
| ✓ | ✓ | **38.65** |

### Key Findings

- OpenBox surpasses the previous SOTA (LiSe) on the Lyft dataset by **+15.8%** AP_3D (27.5→43.3); in direct annotation quality comparison, OpenBox exceeds LiSe by **+19.94%** relative to human annotations.
- On WOD, Vehicle AP_3D@0.5 reaches 70.49%, approximately 2.3× that of CPD (30.30).
- The improvement on the Pedestrian class is especially pronounced (WOD: 57.95 vs. CPD's 14.28), as OpenBox can detect stationary pedestrians whereas CPD only annotates moving objects.
- The largest advantage appears in long-range scenarios (50–80m): AP_3D of 19.5 vs. 1.2 for LiSe.
- OpenBox can also annotate open-vocabulary categories (strollers, fire hydrants, dogs, etc.), surpassing the predefined category sets of existing datasets.

## Highlights & Insights

- **Physical-property-aware adaptive box generation** is the core innovation—objects in different physical states require fundamentally different processing strategies, an intuition that is straightforward yet overlooked by prior methods.
- **Eliminating self-training iterations** substantially reduces computational cost while achieving superior annotation quality.
- **Open-vocabulary capability** allows the method to transcend fixed category sets, which is of significant practical importance for real-world driving safety.
- The method cleverly exploits surface normals and visibility rays to determine box expansion directions, avoiding brute-force search.

## Limitations & Future Work

- **Adverse weather** (rain, fog) degrades the reliability of 2D visual models, and 3D annotations inherit the resulting errors.
- **Deformable objects** (pedestrians, cyclists) exhibit large pose variation; the method falls back to fixed statistical category sizes, yielding less precise box dimensions.
- **Long-range scenarios** suffer from overly sparse LiDAR points, making box fitting unstable.
- Cyclist class performance remains relatively weak (using the "bicycle" prompt leads to undersized boxes).
- The method depends on ChatGPT for object-type classification and size queries, introducing an external dependency.

## Related Work & Insights

- **3D transfer from 2D foundation models**: The combination of Grounding DINO and SAM2 has become a standard paradigm for 2D→3D knowledge transfer.
- **SDF surface reconstruction**: The application of VDBFusion in automatic annotation demonstrates the complementarity of classical geometric methods and deep learning.
- **PP Score**: The Persistence Point Score is an effective tool for estimating the motion state of point clouds.
- Insight: Physical-property awareness is an underappreciated yet important prior in 3D scene understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ The physical-state-aware adaptive box generation strategy is both novel and practically effective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three large-scale datasets, two evaluation settings, and comprehensive ablation studies
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and illustrations are intuitive
- Value: ⭐⭐⭐⭐⭐ Annotation quality significantly surpasses the state of the art, with substantial practical value for reducing 3D annotation costs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LabelAny3D: Label Any Object 3D in the Wild](labelany3d_label_any_object_3d_in_the_wild.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[CVPR 2026\] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](../../CVPR2026/autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [\[NeurIPS 2025\] 3EED: Ground Everything Everywhere in 3D](3eed_ground_everything_everywhere_in_3d.md)
- [\[ICLR 2026\] SEAL: Segment Any Events with Language](../../ICLR2026/autonomous_driving/segment_any_events_with_language.md)

</div>

<!-- RELATED:END -->
