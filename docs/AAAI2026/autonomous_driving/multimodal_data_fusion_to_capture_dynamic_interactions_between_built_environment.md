---
title: >-
  [Paper Note] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults
description: >-
  [AAAI 2026][Autonomous Driving][Multimodal Data Fusion] This paper proposes a **multimodal data fusion framework** that integrates eye-tracking, motion sensors (IMU), physiological monitoring (EDA/HRV), GPS…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Multimodal Data Fusion"
  - "Built Environment"
  - "Older Adult Walking"
  - "SLAM Trajectory Reconstruction"
  - "Urban Planning"
date: 2026-05-08
content_hash: d383a2a3952c5453
---

# Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults

**Conference**: AAAI 2026
**arXiv**: [2601.11545](https://arxiv.org/abs/2601.11545)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Multimodal Data Fusion, Built Environment, Older Adult Walking, SLAM Trajectory Reconstruction, Urban Planning

## TL;DR

This paper proposes a **multimodal data fusion framework** that integrates eye-tracking, motion sensors (IMU), physiological monitoring (EDA/HRV), GPS, and video recording to dynamically characterize interactions between vulnerable older adults (with knee osteoarthritis or fall history) and the urban built environment. Through AI-driven data fusion, the framework identifies urban street segments that significantly influence walking behavior and perception at a microscopic scale, providing evidence-based support for **age-friendly urban planning**.

## Background & Motivation

### Problem Definition
As cities evolve toward human-centered and data-driven design, understanding how the built environment (BE) affects the daily activities of vulnerable populations becomes critical. For older adults with mobility impairments (e.g., knee osteoarthritis, history of falls), microscopic features of the built environment—such as pavement material, pedestrian crossings, lighting, and spatial layout—directly determine their perceived safety and mobility.

### Limitations of Prior Work

**Surveys/Interviews/Field Audits**: Subjective reports are subject to recall bias, limited sample sizes, and high time costs.

**GIS/Street-View Image Analysis**: Provides aggregate statistical assessments but cannot capture moment-to-moment behavioral, perceptual, and physiological responses.

**Single Sensing Modality**: Using only motion sensors or physiological sensors alone fails to provide a comprehensive understanding of human–environment interaction.

**Neglect of Individual Differences**: Large-scale spatial analyses may obscure heterogeneous responses across individuals in identical environments.

### Core Problem
- A **high spatiotemporal resolution** approach is needed to capture real-time behavioral–perceptual–physiological responses.
- Advances in wearable sensors, computer vision, and SLAM technology make multimodal fusion feasible.
- Urban planners require **microscale evidence-based indicators** to complement traditional environmental data.

## Method

### Overall Architecture

The multimodal data acquisition and fusion framework comprises the following layers:
1. **Wearable Sensing**: Eye-tracking (Pupil Labs Neon), motion sensors (ZurichMOVE IMU + Axivity A6), physiological monitoring (Empatica EmbracePlus).
2. **Environmental Perception**: Chest-mounted GoPro third-person video, GPS localization.
3. **Trajectory Reconstruction**: VINS-Fusion SLAM + GPS anchor fusion.
4. **Environment Representation**: AI vision models (Mask2Former, Grounding DINO, SAM, OpenPose) for automated BE feature extraction.
5. **Multimodal Fusion**: Temporal/distance alignment to generate spatiotemporal feature segments.

### Key Designs

#### 1. Multimodal Data Acquisition System

**Function**: Designs a complete sensor configuration for 60–90 minute naturalistic walking experiments, enabling synchronized capture of multiple human and environmental signals.

**Sensor Suite**:
- **Pupil Labs Neon Eye-Tracker**: Records gaze vectors, head kinematics, and first-person video → analyzes visual attention and BE element interaction.
- **ZurichMOVE IMU**: Mounted on both feet, wrists, torso, and head → full-body motion analysis.
- **Axivity A6**: Mounted on the lower back, high-frequency accelerometer and gyroscope → gait parameters (step count, step frequency variability, symmetry).
- **Empatica EmbracePlus**: Wrist-worn, records EDA (electrodermal activity), skin temperature, and heart rate → stress and autonomic regulation.
- **GoPro**: Researcher-operated third-person trailing video → external environmental context.
- **GPS**: Global positioning.

All data streams are **temporally synchronized**, enabling high temporal resolution multimodal alignment analysis.

**Design Motivation**: Each sensor captures a distinct dimension of information—gaze = cognitive processing, gait = motor adaptation, physiology = emotional/stress state, video = environmental context. Only by fusing all modalities can human–environment interaction be comprehensively understood.

#### 2. Trajectory Reconstruction and Fusion (SLAM + GPS)

**Function**: Achieves continuous high-accuracy trajectory reconstruction in urban environments with frequent GPS dropouts (dense tree canopies, high-rise buildings).

**Mechanism**:
1. VINS-Fusion algorithm applied to chest-mounted video for visual-inertial SLAM → locally consistent, smooth trajectory.
2. Reliable GPS segments selected as spatial anchors.
3. Umeyama similarity transformation used to align the local SLAM trajectory to global GPS coordinates.

**Design Motivation**: SLAM provides local accuracy (sub-meter) but suffers from drift; GPS provides global positioning but fails in indoor/occluded areas. The fusion approach combines the strengths of both: global accuracy from GPS and local continuity from SLAM.

#### 3. AI-Driven Built Environment Representation

**Function**: Automatically extracts microscale built environment features from video and trajectory data.

**Pavement Material Classification**:
1. Grounding DINO generates region proposals.
2. SAM produces precise segmentation masks.
3. A linear probe classifies 14 urban surface categories (concrete, asphalt, tile, brick, etc.).

**Pathway Width Estimation**:
1. OpenPose detects participant skeleton keypoints and foot positions.
2. Foot–ground contact regions are identified.
3. Pixel distances are converted to meters using participant height as reference.

**Gaze Behavior Analysis**:
1. The I-DT algorithm detects fixations and saccades (adaptive threshold + optical flow compensation for head motion).
2. Each fixation is projected onto the corresponding frame and cross-referenced with Mask2Former semantic segmentation results → identifies BE elements attracting visual attention.
3. Aggregated metrics: fixation frequency, mean duration, horizontal/vertical dispersion.

**Physiological Signal Processing**:
- EDA decomposed into phasic and tonic components; peak indices quantify arousal frequency.
- HRV metrics (RMSSD, pNN10) quantify autonomic regulation.
- Signals are temporally mapped onto the reconstructed trajectory → identifies microscale environments that elicit stress responses.

**Gait Parameterization**:
- IMU data undergoes time-domain stride segmentation.
- Step count, mean stride time, and stride time variability (STV) are computed.
- Elevated STV or asymmetric gait may indicate adaptive responses to environmental obstacles.

### Loss & Training

This paper presents a system/framework contribution and does not involve end-to-end model training. Each AI component uses pretrained models:
- Mask2Former: pretrained on Mapillary Vistas.
- Grounding DINO + SAM: used in a zero-shot manner.
- OpenPose: standard human pose estimation implementation.
- VINS-Fusion: standard visual-inertial SLAM implementation.

## Key Experimental Results

### Main Results (Preliminary)

This is a methodology paper; experiments are **proof-of-concept** in nature:

**Trajectory Reconstruction Evaluation**:
- Field-tested in Singapore's Yio Chu Kang district.
- Test scenarios: covered walkways, underground passages, semi-enclosed public spaces.
- SLAM maintains smooth, locally consistent motion tracking in GPS-denied environments.
- The fused trajectory demonstrates continuous and stable motion across indoor–outdoor transitions.

**Pavement Feature Extraction**:

| Function | Input | Output | Performance |
|---|---|---|---|
| Material Classification | Third-person video frames | 14-class pavement material labels | Accurately identifies concrete, asphalt, granite, etc. |
| Width Estimation | OpenPose skeleton + video | Continuous width variation curve | Reveals narrow-to-wide segment transitions |
| Trajectory Annotation | SLAM+GPS fused trajectory | Material/width spatiotemporal mapping | Supports per-meter BE condition analysis |

### Ablation Study (Multimodal Complementarity Analysis)

Although no traditional ablation experiments are conducted, the paper systematically demonstrates the complementarity of each modality:

| Modality | Captured Dimension | Key Metrics | Application Value |
|---|---|---|---|
| Eye-tracking | Cognitive perception | Fixation frequency/duration/dispersion | Which BE elements attract attention |
| IMU/Accelerometer | Motor adaptation | Step frequency variability/symmetry | Effect of pavement conditions on gait |
| EDA/HRV | Emotional stress | EDA peaks/RMSSD | Which environments elicit stress |
| Third-person video | Environmental context | Pavement material/width | Objective quantification of BE conditions |
| GPS/SLAM | Spatial localization | Sub-meter trajectory | Precise behavior–environment correspondence |

### Key Findings

1. **SLAM+GPS fusion** achieves reliable continuous trajectory reconstruction in tropical urban environments with frequent GPS dropout.
2. **AI vision models can automatically extract microscale BE features** (pavement material, width), replacing manual field audits.
3. **Multimodal fusion potential**: Simultaneous EDA peaks and wide gaze dispersion in narrow corridors may reflect increased cognitive load or perceived risk.
4. **Framework scalability**: Can be integrated into urban digital twins to support multi-participant hotspot analysis.

## Highlights & Insights

1. **Interdisciplinary Innovation**: Combining computer vision, robotics (SLAM), psychophysiology, and urban planning represents a novel research paradigm.
2. **"Experiential" urban assessment** replaces "observational" assessment: traditional methods evaluate urban spaces from a third-party perspective, while this paper centers on the **first-person experience of users**, better reflecting actual needs.
3. **The interactive dashboard prototype** is a valuable tool design: researchers and planners can navigate by time or distance and intuitively link physiological arousal to specific BE elements.
4. **Focus on vulnerable populations** reflects inclusive design principles: data are collected not for the "average person" but for those most in need of support.

## Limitations & Future Work

1. **Only preliminary results** (proof-of-concept); statistical correlation analysis and large-sample validation are absent.
2. **Small participant sample**; specific participant counts and statistical significance are not reported.
3. Lacks **quantitative comparison** with traditional methods (surveys, GIS audits).
4. **Privacy and ethical concerns**: video recording and eye-tracking in public spaces may face ethical review challenges.
5. **Sensor burden**: multiple wearable devices and a trailing researcher may alter participants' natural behavior (Hawthorne effect).
6. Classification accuracy for pavement material is not quantitatively reported.
7. Computational cost may be high: real-time processing demands of SLAM combined with multiple AI models.

## Related Work & Insights

- **VINS-Fusion**: Visual-inertial SLAM algorithm providing locally accurate trajectories in GPS-denied environments.
- **Grounding DINO + SAM**: A zero-shot object detection and segmentation combination used for pavement region extraction.
- **Mapillary Vistas + Mask2Former**: Street-view semantic segmentation dataset and model.
- Insights: Multimodal sensing has broad applicability in **human factors engineering** and **universal design**. Similar frameworks can be extended to other vulnerable populations (visually impaired individuals, wheelchair users) and other environmental assessment tasks (industrial safety, disaster relief route evaluation).

## Rating

- Novelty: ⭐⭐⭐⭐ — The application of multimodal fusion frameworks to age-friendly urban planning is genuinely novel.
- Experimental Thoroughness: ⭐⭐ — Only proof-of-concept preliminary results; statistical analysis and large-sample validation are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation and framework description are clear; figures and diagrams are intuitive.
- Value: ⭐⭐⭐ — The approach is conceptually inspiring but practical utility remains to be validated; leans more toward a position paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](../../NeurIPS2025/autonomous_driving/layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[AAAI 2026\] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)

</div>

<!-- RELATED:END -->
