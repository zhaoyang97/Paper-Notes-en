---
title: >-
  [Paper Note] PoseSOR: Human Pose Can Guide Our Attention
description: >-
  [ECCV 2024][Human Understanding][Salient Object Ranking] This paper introduces human pose information to the Salient Object Ranking (SOR) task for the first time. By proposing a Pose-Aware Interaction (PAI) module and a Pose-Driven Ranking (PDR) module, it models the relationship between human activities and attention shifts, significantly improving SOR performance in complex scenes and achieving SOTA results.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Salient Object Ranking"
  - "Human Pose"
  - "Attention Guidance"
  - "High-Level Interaction"
  - "Pose-Awareness"
date: 2026-05-08
content_hash: ee2d5bebaa9886ee
---

# PoseSOR: Human Pose Can Guide Our Attention

**Conference**: ECCV 2024  
**Code**: [https://github.com/guanhuankang/ECCV24PoseSOR](https://github.com/guanhuankang/ECCV24PoseSOR)  
**Area**: Human Understanding / Saliency Detection  
**Keywords**: Salient Object Ranking, Human Pose, Attention Guidance, High-Level Interaction, Pose-Awareness

## TL;DR
This paper introduces human pose information to the Salient Object Ranking (SOR) task for the first time. By proposing a Pose-Aware Interaction (PAI) module and a Pose-Driven Ranking (PDR) module, it models the relationship between human activities and attention shifts, significantly improving SOR performance in complex scenes and achieving SOTA results.

## Background & Motivation

**Background**: Salient Object Ranking (SOR) aims to study how human observers shift attention between different objects in a scene. Existing methods mainly rely on explicit visual saliency cues to accomplish this task, such as low-level and mid-level visual features including spatial frequency, semantic context, and color contrast.

**Limitations of Prior Work**: These methods based on visual saliency cues often perform poorly when handling real-world scenes involving human activities and interactions. For example, when multiple people in a scene perform different activities (such as running, pointing, or conversing), relying solely on spatial frequency and semantic context is insufficient to accurately determine where an observer's attention will shift. Existing methods overlook a critical attention-guiding factor — human pose and gestures.

**Key Challenge**: The attention of human observers is often "reflexively" guided by the poses and gestures of people in the scene. For instance, observers tend to follow others' head orientations or running/walking directions to anticipate what is about to happen. Such high-level interaction information is crucial for understanding attention shifts, but existing SOR methods do not utilize this cue at all.

**Goal**: (1) How to integrate human pose knowledge into salient object queries to learn high-level interaction relationships; (2) how to use poses as directional cues to predict the direction of attention shifts.

**Key Insight**: The authors observe that when people view images, their attention is naturally guided by the poses of characters in the image. For example, seeing a running person makes us subconsciously look in the direction of their run. This intuitive observation inspires incorporating human pose as critical prior knowledge for the SOR task.

**Core Idea**: Utilizing human pose information as high-level interaction cues and directional guidance to enhance salient object ranking, compensating for the limitations of traditional visual saliency cues in complex human interaction scenes.

## Method

### Overall Architecture
PoseSOR is a human pose-aware salient object ranking model. The overall pipeline is: the input image first goes through feature extraction to obtain visual features, while drawing pose information (keypoint coordinates and skeleton structures) of characters in the scene via a pose estimator. Then, pose knowledge is integrated into the object ranking process through two core modules: the Pose-Aware Interaction (PAI) module and the Pose-Driven Ranking (PDR) module. Finally, it outputs the ranking results of all salient objects in the scene, reflecting the priority of human attention.

### Key Designs

1. **Pose-Aware Interaction (PAI) Module**:

    - **Function**: Integrates human pose knowledge into salient object queries, enabling the model to learn high-level interaction relationships between people and the surrounding environment.
    - **Mechanism**: First, keypoint information for each person (head, arms, torso, legs, etc.) is obtained from the pose estimator, and these pose features are encoded into pose embeddings. Then, the pose embeddings are fused with salient object queries through a cross-attention mechanism. Specifically, the pose embeddings act as key and value, while the object queries act as query. Attention weights are used to decide which human pose information each object should attend to. Consequently, the object queries incorporate contextual information about "what the people in the scene are doing".
    - **Design Motivation**: Traditional methods determine saliency solely by looking at the visual properties of the target itself. However, human activities (such as pointing, gazing, and walking towards something) are actually important cues for guiding attention. The PAI module enables the model to "understand" human behavioral intentions.

2. **Pose-Driven Ranking (PDR) Module**:

    - **Function**: Uses pose knowledge as directional cues to predict the direction of human attention shifts, thereby performing object ranking.
    - **Mechanism**: Extracts directional information from human poses, including head orientation (indicating gaze direction), limb movement direction (representing travel direction), etc. These directional details are encoded into directional vectors and matched against the spatial position relationships of each object in the scene. Objects with higher matching degrees are assigned higher ranking priorities. Inside the module, a spatial attention map is used to model the correspondence between directional cues and object positions.
    - **Design Motivation**: Human attention shifts have distinct directionality — observers tend to look for the next target of attention along the direction of another person's gaze or movement. PDR leverages this intuition to use pose direction as an essential basis for ranking.

3. **Multi-Scale Feature Fusion and Ranking Head**:

    - **Function**: Synthesizes the outputs of PAI and PDR to generate the final ranking prediction on multi-scale features.
    - **Mechanism**: The model extracts features at different scales to capture objects of various sizes. After PAI and PDR operate independently at each scale, they are fused through a Feature Pyramid Network (FPN). The ranking head uses these fused features to assign a ranking score to each detected salient object.
    - **Design Motivation**: Objects in a scene vary in size, and multi-scale processing ensures that both small and large objects receive accurate rankings.

### Loss & Training
The model is trained using ranking loss functions, including a pairwise ranking loss to optimize the relative ranking relationships between objects, and a segmentation loss to assist in object localization. Multi-task joint training allows ranking prediction and object segmentation to mutually reinforce each other.

## Key Experimental Results

### Main Results

| Dataset | Metric | PoseSOR | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| ASSR | SA-SOR (↑) | **0.713** | 0.685 | +4.1% |
| ASSR | MAE (↓) | **0.034** | 0.039 | -12.8% |
| IRSR | SA-SOR (↑) | **0.698** | 0.672 | +3.9% |
| IRSR | MAE (↓) | **0.041** | 0.047 | -12.8% |

### Ablation Study

| Configuration | SA-SOR | Description |
|------|--------|------|
| Full model (PAI+PDR) | **0.713** | Full model |
| w/o PAI | 0.691 | Drops 3.1% without pose interaction |
| w/o PDR | 0.695 | Drops 2.5% without directional ranking |
| w/o Pose (baseline) | 0.678 | Drops 4.9% without using pose information |

### Key Findings
- The PAI module contributes the most, indicating that high-level interaction modeling is crucial for SOR.
- The directional guidance of PDR yields more significant improvements in complex scenes involving multi-person activities.
- In simple scenes (with few objects and no people), the performance gains from pose information are limited, but the advantages are prominent in complex human interaction scenes.
- The accuracy of pose estimation has some impact on the final ranking, but the model exhibits good robustness against pose noise.

## Highlights & Insights
- **Pose as an attention prior** is a highly natural and ingenious idea. Humans are indeed "guided" in their attention by the poses of others. This cognitive psychology discovery is introduced into the computer vision SOR task for the first time, opening up a new research direction.
- **The complementary design of PAI and PDR** is elegant: PAI is responsible for "understanding what people are doing", while PDR is responsible for "predicting where attention goes", utilizing pose information from different perspectives.
- Pose information can be transferred to other attention-related tasks, such as gaze guidance in Visual Question Answering (VQA), key person detection in video summarization, etc.

## Limitations & Future Work
- When there are no people in the scene, pose information cannot provide assistance, and the model degrades to an ordinary SOR method.
- The model relies on the accuracy of the external pose estimator; failures in pose estimation will affect the ranking quality.
- Only the pose in static images is considered; dynamic pose information in videos (such as motion trajectories and gesture changes) is not utilized.
- It is worth exploring the integration of eye-tracking data for more precise attention modeling.

## Related Work & Insights
- **vs ASSR**: ASSR uses purely visual saliency cues for ranking and lacks an understanding of human activities. PoseSOR performs better in complex scenes by introducing pose information.
- **vs RankNet**: RankNet uses a general pairwise ranking strategy, whereas PoseSOR's ranking is driven by pose direction, offering stronger interpretability.
- This work inspires the direction of "action-intent-guided visual understanding" and can serve as a baseline for tasks such as human-computer interaction and social scene understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce human pose into the SOR task, with unique observations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple SOR benchmarks with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the story flows smoothly.
- Value: ⭐⭐⭐⭐ Opens up a new direction of pose-guided attention, providing inspiration for the SOR field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ICLR 2026\] EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?](../../ICLR2026/human_understanding/emoprefer_can_large_language_models_understand_human_emotion_preferences.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)

</div>

<!-- RELATED:END -->
