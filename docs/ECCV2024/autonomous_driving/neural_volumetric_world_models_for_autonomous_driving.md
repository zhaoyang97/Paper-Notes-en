---
title: >-
  [Paper Note] Neural Volumetric World Models for Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][Voxel World Models] This paper proposes NeMo (Neural Volumetric World Model), an end-to-end autonomous driving framework based on volumetric representation. It represents scenes via 3D voxels, models dynamics through a motion flow module, and integrates future predictions via temporal attention. Trained in a self-supervised manner, NeMo outperforms prior methods by over 18% in driving performance on both nuScenes and CARLA.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Voxel World Models"
  - "Motion Flow Estimation"
  - "End-to-End Planning"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: 58a1cfca5b0994aa
---

# Neural Volumetric World Models for Autonomous Driving

**Conference**: ECCV 2024  
**Authors**: Zanming Huang, Jimuyang Zhang, Eshed Ohn-Bar  
**Code**: None  
**Area**: Autonomous Driving / 3D World Models  
**Keywords**: Voxel World Models, Autonomous Driving, Motion Flow Estimation, End-to-End Planning, Self-Supervised Learning  

## TL;DR

This paper proposes NeMo (Neural Volumetric World Model), an end-to-end autonomous driving framework based on volumetric representation. It represents scenes via 3D voxels, models dynamics through a motion flow module, and integrates future predictions via temporal attention. Trained in a self-supervised manner, NeMo outperforms prior methods by over 18% in driving performance on both nuScenes and CARLA.

## Background & Motivation

**Background**: Perception and planning systems in autonomous driving have made rapid progress in recent years. Current mainstream methods primarily employ Bird's Eye View (BEV) as a 2D spatial representation of the scene, performing detection, prediction, and planning within this representation space.

**Limitations of Prior Work**: BEV representation compresses the 3D world into a 2D ground plane, discarding critical vertical dimension information. This leads to several issues: (1) an inability to accurately model occlusion relationships (e.g., vehicles blocking each other); (2) failure to capture fine-grained vertical motions (e.g., hand gestures of pedestrians or height variations of traffic lights); (3) insufficient representation capability under sloped terrains and multi-level interchanges; and (4) limited capability to handle partial observability.

**Key Challenge**: While real-world driving environments are inherently 3D, mainstream methods suffer from information bottlenecks due to their reliance on 2D BEV representations. Although 3D volumetric representations are theoretically more suitable, they suffer from high computational overhead and sparse training signals (since most voxels are empty). Efficiently utilizing 3D volumetric representations to improve end-to-end driving remains an open question.

**Goal**: (1) How to construct a world model based on 3D volumetric representation for autonomous driving? (2) How to model complex dynamic scenes (multiple moving objects, different motion patterns) in volumetric space? (3) How to effectively utilize the predictions of the volumetric world model for motion planning? (4) How to train the entire system in a self-supervised manner to ensure scalability?

**Key Insight**: The authors argue that 3D volumetric representations provide a more faithful scene modeling, preserving full spatial structures, occlusion relations, and motion characteristics. The key lies in designing an efficient volumetric world model architecture and training it via self-supervised tasks (e.g., image reconstruction and occupancy prediction), thereby avoiding reliance on expensive 3D annotations.

**Core Idea**: Replace BEV with 3D voxel representations, integrate motion flow modules and temporal attention to construct a volumetric world model, and leverage self-supervised training to achieve high-fidelity 3D scene understanding and driving planning.

## Method

### Overall Architecture

The inputs to NeMo are multi-view camera image sequences, and the outputs are planned trajectories for the ego-vehicle. The pipeline consists of the following stages: (1) a visual encoder lifts multi-view image features into 3D volumetric features (lifting to voxel space); (2) a motion flow module estimates 3D motion fields in the volumetric space; (3) future volumetric states are predicted via temporal propagation based on motion flows; (4) a temporal attention module integrates historical and predicted volumetric features; and (5) a planning head outputs ego-vehicle trajectories based on the fused volumetric features. Training is supervised by self-supervised losses from image reconstruction and occupancy prediction.

### Key Designs

1. **3D Volumetric Feature Lifting**:

    - **Function**: Convert multi-view 2D image features into a unified 3D volumetric feature representation.
    - **Mechanism**: Once image features for each view are extracted by a 2D backbone, known camera parameters are used to "lift" the 2D features into the 3D voxel space. Specifically, an LSS-like (Lift-Splat-Shoot) method is used: the probability distribution along the depth direction is predicted for each pixel, and 2D features are distributed to corresponding 3D voxel locations weighted by these probabilities. Features from multiple views are fused into a unified voxel grid using summation or averaging.
    - **Design Motivation**: Compared to BEV, volumetric representation preserves complete vertical dimension information, enabling better representation of occlusions, multi-level structures, and irregular terrains. Despite higher computational costs, it models the 3D world more faithfully.

2. **Motion Flow Module**:

    - **Function**: Estimate the motion vector of each voxel within the 3D volumetric space, modeling the 3D motion of all objects in the scene.
    - **Mechanism**: Given volumetric features of two consecutive frames, $V_t$ and $V_{t-1}$, the motion flow module estimates a voxel-wise 3D motion field $F_{t \rightarrow t+1}$ using a 3D convolutional network, which represents the displacement of each voxel from time $t$ to $t+1$. This motion field can warp the current volume to a future state, thereby predicting future scenes. The motion field captures both the motion of dynamic objects (vehicles, pedestrians) and implicitly encodes the effect of ego-motion.
    - **Design Motivation**: Traditional methods performing motion prediction in 2D BEV space lose motion information in the vertical direction. A 3D motion field captures motion characteristics across all three dimensions (such as vertical gestures of pedestrians or height variations of vehicles on slopes), providing more complete dynamic modeling. Additionally, motion consistency can serve as a self-supervised signal to constrain temporal feature learning.

3. **Temporal Attention Module**:

    - **Function**: Fuse predicted future volumetric features with historical volumetric features to provide comprehensive spatiotemporal information for planning.
    - **Mechanism**: Use the transformer attention mechanism to aggregate multi-frame volumetric features along the temporal dimension. Specifically, the volumetric feature of the current frame is used as Query, while the features of historical and predicted future frames serve as Key and Value. The attention weights learn where each voxel should retrieve information from—for static objects (e.g., buildings), information is accumulated across multiple frames to enhance robustness; for dynamic objects, the model focuses more on recent frames to capture the latest motion state.
    - **Design Motivation**: The planning task requires understanding both "what became of the past" and "what will happen in the future." Temporal attention enables the planning head to obtain features integrated with predictive information, rather than looking only at the current frame, facilitating more proactive decision-making.

### Loss & Training

NeMo adopts a self-supervised training paradigm. The core losses include: (1) **Image Reconstruction Loss**: reconstructs multi-view images from volumetric features via volume rendering, compared with ground-truth images using L2 + perceptual losses; (2) **Occupancy Prediction Loss**: utilizes LiDAR point clouds as weak supervision signals to predict the occupancy state of the 3D space; (3) **Motion Flow Consistency Loss**: enforces consistency between volumetric features from consecutive frames after motion flow warping. The planning component utilizes an imitation learning loss, represented as the L2 distance between predicted trajectories and expert trajectories. The entire system is trained end-to-end.

## Key Experimental Results

### Main Results

| Dataset | Metric | NeMo | Prev. SOTA | Gain |
|--------|------|------|---------|------|
| nuScenes | Driving Score ↑ | SOTA | BEV-based | +18% |
| nuScenes | Route Completion ↑ | SOTA | BEV-based | Significant Improvement |
| CARLA | Driving Score ↑ | SOTA | Multiple methods | +18%+ |
| CARLA | Infraction Score ↑ | SOTA | Multiple methods | Significant safety improvement |

### Ablation Study

| Configuration | Driving Score | Description |
|------|--------------|------|
| Full NeMo | Best | Full model |
| BEV instead of volumetric representation | Significant drop | Validates the advantage of volumetric representation |
| w/o Motion flow module | Moderate drop | Inability to predict dynamic scene changes |
| w/o Temporal attention | Drop | Lack of temporal information fusion |
| w/o Image reconstruction loss | Drop | Reduced self-supervised signals lead to degraded feature quality |
| Reduced volumetric resolution | Slight drop | Still effective under coarse resolution |

### Key Findings
- The advantages of the 3D volumetric representation over BEV are most pronounced in scenarios with severe occlusions and complex terrains.
- The motion flow module contributes the most in dynamic and dense scenarios (e.g., intersections) while providing limited gains in static environments.
- Self-supervised training (image reconstruction + occupancy prediction) unlocks the potential of volumetric representation, avoiding the need for expensive 3D annotations.
- The overall improvement of over 18% is highly significant, validating that transitioning from 2D BEV to 3D volumetric representations is a meaningful paradigm shift.

## Highlights & Insights
- **Paradigm shift from BEV to Volumetric**: The paper clearly demonstrates why 3D volumetric representations outperform 2D BEVs, providing compelling experimental evidence with a performance gain of over 18%. This shift is highly influential for the entire autonomous driving perception-and-planning community.
- **Ingenious design of the self-supervised training paradigm**: The image reconstruction loss helps the model learn complete 3D geometry and appearance, while motion consistency guides it to learn dynamics. Eliminating the dependency on 3D annotations enables the system to be trained on massive scales of driving videos.
- **Volumetric motion flow estimation** can be easily transferred to other tasks requiring 3D dynamic understanding, such as robotic manipulation and indoor navigation.

## Limitations & Future Work
- The computational and memory footprints of 3D volumetric representations are significantly larger than those of BEV, limiting real-time deployment on embedded platforms.
- The current volumetric resolution is constrained by GPU memory, leading to insufficient accuracy in distant regions; sparse voxels or octree structures could be potential solutions.
- The method relies solely on camera inputs without directly fusing LiDAR point clouds. Although LiDAR supervision signals are used, integrating LiDAR spatial features could further boost performance.
- The motion flow is assumed to be rigid, which is insufficient for detailed modeling of deformable objects (e.g., articulated movements of pedestrians).
- Performance under extreme weather conditions (heavy rain, fog) and night scenes remains to be validated.

## Related Work & Insights
- **vs UniAD**: UniAD performs end-to-end planning in the BEV space but is constrained by 2D representation. NeMo provides richer 3D details through volumetric representation, particularly in modeling the vertical dimension.
- **vs OccWorld**: OccWorld also utilizes occupancy grids as a world model but focuses primarily on occupancy prediction tasks. NeMo places more emphasis on leveraging the volumetric world model to improve planning performance.
- **vs MUVO**: MUVO also explores volumetric representations in autonomous driving. NeMo distinguishes itself by incorporating a motion flow module and temporal attention to better model dynamic scenes.
- **vs SelfD**: SelfD similarly employs self-supervised training but operates in 2D space. NeMo extends the self-supervised paradigm into 3D volumetric space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of volumetric world models, motion flow, and temporal attention is novel in end-to-end driving.
- Experimental Thoroughness: ⭐⭐⭐⭐ Double validation on nuScenes and CARLA; the 18%+ improvement is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Well-justified motivation (why 3D volumetric representation is preferred over BEV) and clear technical descriptions.
- Value: ⭐⭐⭐⭐⭐ Proposes a paradigm shift from BEV to volumetric representations, providing directional guidance for the autonomous driving domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[ECCV 2024\] NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)
- [\[ICLR 2026\] DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](../../ICLR2026/autonomous_driving/drivevla-w0_world_models_amplify_data_scaling_law_in_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)

</div>

<!-- RELATED:END -->
