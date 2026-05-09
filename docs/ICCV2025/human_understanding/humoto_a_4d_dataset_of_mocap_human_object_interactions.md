---
title: >-
  [Paper Note] HUMOTO: A 4D Dataset of Mocap Human Object Interactions
description: >-
  [ICCV 2025][Human Understanding][HOI dataset] This paper presents HUMOTO, a high-fidelity 4D human-object interaction dataset comprising 735 sequences (7,875 seconds at 30fps), covering 63 precisely modeled objects with 72 articulated parts. It introduces an LLM-driven scene scripting pipeline and a multi-sensor capture system, achieving significantly superior hand pose accuracy and interaction quality compared to existing datasets.
tags:
  - ICCV 2025
  - Human Understanding
  - HOI dataset
  - motion capture
  - hand pose
  - multi-object interaction
  - LLM script generation
date: 2026-05-08
content_hash: 962ca4df2616e8e2
---

# HUMOTO: A 4D Dataset of Mocap Human Object Interactions

**Conference**: ICCV 2025
**arXiv**: [2504.10414](https://arxiv.org/abs/2504.10414)
**Code**: [https://jiaxin-lu.github.io/humoto/](https://jiaxin-lu.github.io/humoto/)
**Area**: Human Understanding / Human-Object Interaction
**Keywords**: HOI dataset, motion capture, hand pose, multi-object interaction, LLM script generation

## TL;DR

This paper presents HUMOTO, a high-fidelity 4D human-object interaction dataset comprising 735 sequences (7,875 seconds at 30fps), covering 63 precisely modeled objects with 72 articulated parts. It introduces an LLM-driven scene scripting pipeline and a multi-sensor capture system, achieving significantly superior hand pose accuracy and interaction quality compared to existing datasets.

## Background & Motivation

4D human-object interaction (HOI) data is critical for computer vision, robotics, graphics, and generative AI. However, existing datasets exhibit notable deficiencies:

- **Single-object limitation**: Most datasets (GRAB, BEHAVE, OMOMO, etc.) cover only single-object interactions.
- **Absence of hand motion**: Many datasets lack fine-grained hand motion (e.g., BEHAVE and OMOMO use only standard hand templates).
- **Semantic discontinuity**: Existing sequences are often isolated and purposeless, lacking coherent task-level logic.
- **Poor interaction quality**: Significant hand-object penetration or complete separation is common.

Acquiring high-quality 4D HOI data is costly, requiring sophisticated motion capture equipment and extensive manual cleanup. HUMOTO aims to address the gap in multi-object, whole-body-plus-hand, semantically meaningful task sequence data.

## Method

### Overall Architecture

Data acquisition pipeline: Scene design → LLM script generation → Mocap + camera recording → Data processing → Multi-stage quality assurance → Text annotation

### Key Designs

1. **Scene-Driven LLM Scripting**: Drawing inspiration from filmmaking workflows, 63 objects are first grouped logically into "rooms" (e.g., kitchen, study), and these object groups are then provided to an LLM to generate coherent interaction scripts. The LLM generates content hierarchically: first determining the scene theme, then elaborating specific action sequences (e.g., "open drawer to retrieve item → organize on table → prepare meal"), ensuring each action has a clear purpose and natural transition, avoiding isolated or meaningless motions.

2. **Multi-Sensor Capture System**:
   - **Body mocap**: Rokoko smart motion capture suit with matching gloves; 30fps inertial sensor network tracking full-body skeleton and finger joints.
   - **Object tracking**: Dual Kinect RGB-D cameras to maximize coverage and minimize occlusion; FoundationPose algorithm for 6DoF object pose estimation.
   - **Custom environment**: Wooden elevated stage (reducing magnetic interference from metal on inertial sensors); dual-computer UDP time synchronization.
   - **Occlusion handling**: SAM2 combined with manual annotation for object masks; dynamic reset mechanism to handle tracking loss caused by rapid motion.

3. **Multi-Stage Quality Assurance**:
   - **Technical refinement**: Professional animators correct capture artifacts (drift, tracking errors) and ensure interaction logical consistency.
   - **Independent validation**: A separate team verifies natural and plausible interactions and fixes joint jitter and foot sliding issues.
   - Both stages iterate until all quality criteria are met.
   - **Text annotation**: Three-level annotation — short title, brief script, and detailed long script.

### Loss & Training

As a dataset paper, no training loss is involved. However, a comprehensive HOI dataset quality evaluation metric suite is proposed:
- **Body motion**: Foot Sliding, Jerk (smoothness of acceleration), MSNR (motion signal-to-noise ratio, with Mixamo as baseline), Coherence, Diversity.
- **Object motion**: Jerk (manipulation smoothness).
- **Interaction quality**: Penetration depth, Contact Entropy (diversity of contact state distribution), State Consistency.

## Key Experimental Results

### Main Results (Dataset Quality Comparison)

| Dataset | Foot Sliding (cm)↓ | Jerk (m/s³)↓ | MSNR (dB)→ | Object Jerk↓ | Penetration (cm)↓ | Contact Entropy↑ |
|---|---|---|---|---|---|---|
| BEHAVE | 4.556 | 4.08 | 5.51 | 10.40 | 0.0606 | 2.2915 |
| OMOMO | 2.130 | 15.10 | 12.37 | 27.40 | 0.0602 | 1.9468 |
| IMHD | 1.474 | 1.14 | 14.20 | 24.06 | 0.1172 | 2.4265 |
| ParaHome | 3.008 | 9.19 | 1.82 | 0.08 | 0.2167 | 1.0254 |
| **HUMOTO** | **0.958** | **1.87** | **9.42** | **1.13** | **0.0068** | **1.4587** |
| Mixamo (ref.) | 3.184 | 8.14 | 10.88 | — | — | — |

### Ablation Study (Dataset Statistics Comparison)

| Dataset | Duration (h) | Subjects | Objects | Hands | Body | Max Scene Objects | Capture Setting |
|---|---|---|---|---|---|---|---|
| GRAB | 3.8 | 10 | 51 | ✓ | ✓ | 1 | Standing |
| BEHAVE | 4.2 | 8 | 20 | ✗ | ✓ | 1 | Portable |
| OMOMO | 10.1 | 17 | 15 | ✗ | ✓ | 1 | Portable |
| ParaHome | 8.1 | 38 | 22 | ✓ | ✓ | 22 | Room |
| **HUMOTO** | **2.2** | **1** | **63** | **✓** | **✓** | **15** | **Scene** |

### Key Findings

1. **Minimal foot sliding**: HUMOTO's 0.958 cm significantly outperforms all other datasets (second-best IMHD: 1.474 cm), attributed to a rigorous mocap pipeline and professional cleanup.
2. **Lowest penetration**: Penetration depth of 0.0068 cm is an order of magnitude lower than BEHAVE (0.0606 cm) and OMOMO (0.0602 cm), maintaining high precision even with fine-grained hand poses.
3. **Natural object manipulation**: Object Jerk of only 1.13, far below OMOMO (27.40) and IMHD (24.06), indicating smooth and realistic manipulation.
4. **MSNR close to Mixamo baseline**: 9.42 dB approaches the professionally animated Mixamo reference of 10.88 dB.
5. **Perceptual evaluation**: 82% of participants gave HUMOTO the highest quality rating; 96% preferred HUMOTO over BEHAVE in overall quality.

## Highlights & Insights

- LLM-driven scene scripting is a novel contribution to the data capture pipeline, importing filmmaking storyboard concepts into mocap planning to ensure semantic coherence and task completeness across sequences.
- The combination of multi-sensor capture, wooden elevated stage, and electromagnetic mocap suit is a practical engineering solution effectively addressing occlusion and magnetic interference.
- Penetration depth is an order of magnitude lower than competing datasets, which is critical for improving the precision of hand-object interaction modeling.
- Downstream application demonstrations are extensive: motion generation (MotionGPT's poor performance on HUMOTO prompts highlights dataset difficulty), robotic grasping (comparison with DexGraspNet), and pose estimation (both 4D Humans and TRAM fail).

## Limitations & Future Work

- **Single performer**: Only one actor is involved, potentially introducing bias in body shape and motion style, limiting cross-subject generalization.
- **Limited total duration**: At 2.2 hours, despite high quality, the scale is modest compared to OMOMO (10.1 h) and ParaHome (8.1 h).
- **High manual cleanup cost**: Each sequence requires multiple rounds of professional animator review and correction, making large-scale expansion difficult.
- **Artist-modeled objects**: Objects are created by artists rather than 3D-scanned, which may introduce subtle geometric discrepancies from real objects.
- No contact force or torque information is provided, limiting utility for robot learning tasks requiring force feedback.

## Related Work & Insights

- **GRAB**: A pioneering whole-body HOI dataset, but limited to upper body, single objects, and standing postures.
- **BEHAVE / OMOMO**: More complex scenes but lacking hand pose data.
- **ParaHome**: Multi-object interaction in home scenes, but labeled markers on hands interfere with natural motion.
- **FoundationPose**: Used for 6DoF object tracking; effective but requires a dynamic reset mechanism as auxiliary support.
- **SAM2**: Assists in generating object segmentation masks, improving tracking robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ LLM script-driven design and multi-sensor approach are novel, though the core contribution is the dataset rather than an algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative metrics are comprehensive, perceptual evaluation is rigorous, and downstream application demonstrations are extensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, the capture pipeline is described in detail, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Fills the gap in multi-object fine-grained hand HOI data, with significant value for motion generation, robotics, and embodied AI.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](../../CVPR2026/human_understanding/hum4d_markerless_motion_capture.md)
- [\[ICCV 2025\] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions](tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)
- [\[ICCV 2025\] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation](mdd_a_dataset_for_text-and-music_conditioned_duet_dance_generation.md)
- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)

<!-- RELATED:END -->
