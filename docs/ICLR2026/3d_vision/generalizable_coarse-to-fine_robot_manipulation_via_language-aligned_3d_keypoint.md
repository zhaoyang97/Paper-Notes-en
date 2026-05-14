---
title: >-
  [Paper Note] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints
description: >-
  [ICLR 2026][3D Vision][Robot Manipulation] CLAP (Coarse-to-fine Language-Aligned manipulation Policy) achieves strong generalization to novel instructions and unseen environments through three core components: task decom…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Robot Manipulation"
  - "Coarse-to-Fine Policy"
  - "3D Keypoints"
  - "VLM Fine-tuning"
  - "Language Grounding"
date: 2026-05-08
content_hash: 9b80a714feb5a600
---

# Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints

**Conference**: ICLR 2026
**arXiv**: [2509.23575](https://arxiv.org/abs/2509.23575)
**Code**: None
**Area**: 3D Vision / Robot Manipulation
**Keywords**: Robot Manipulation, Coarse-to-Fine Policy, 3D Keypoints, VLM Fine-tuning, Language Grounding

## TL;DR

CLAP (Coarse-to-fine Language-Aligned manipulation Policy) achieves strong generalization to novel instructions and unseen environments through three core components: task decomposition, VLM fine-tuning for 3D keypoint prediction, and 3D-aware representation. It outperforms the state of the art by 12% on GemBench using only 1/5 of the training data.

## Background & Motivation

Hierarchical coarse-to-fine policies have shown great promise in 3D robot manipulation tasks. The fundamental idea is that a coarse branch predicts a Region of Interest, after which a fine branch executes precise action prediction within that region. This hierarchical design significantly improves sample efficiency and manipulation accuracy.

However, even when augmented with pretrained models, existing hierarchical policies still suffer from a core **generalization deficit**:

**Generalization to novel instructions**: Policies frequently fail when given natural language instructions unseen during training (e.g., "pick up the red cup" → "place the blue bowl on the shelf").

**Generalization to environmental variation**: Changes in object position, appearance, or background can cause policy collapse.

**Sample efficiency**: Existing methods typically require large numbers of demonstration trajectories to learn each task.

The root cause of these issues lies in the coarse branch's limited understanding of language semantics and the absence of structured 3D spatial information in the representation.

## Method

### Overall Architecture

CLAP is a hierarchical manipulation policy framework consisting of three mutually complementary core components:

1. Complex instructions are first decomposed into a sequence of sub-tasks via task decomposition.
2. A fine-tuned VLM then predicts language-aligned 3D keypoints from the current observation for each sub-task.
3. Finally, precise manipulation is executed near the predicted keypoints using a 3D-aware representation.

### Key Designs

1. **Task Decomposition**:

    - **Function**: Decomposes natural language instructions into an ordered sequence of sub-task steps.
    - **Mechanism**: A large language model (LLM) or rule-based method decomposes complex manipulation instructions into atomic steps. For example, "place the cup next to the plate" can be decomposed into "1. approach the cup → 2. grasp the cup → 3. move next to the plate → 4. release the cup."
    - **Design Motivation**: Direct end-to-end mapping of complex instructions requires abundant data. After decomposition, each sub-task is simpler and more transferable—the atomic action "grasp" can be reused across diverse scenarios. This compositionality is a key source of generalization.

2. **VLM Fine-tuning for 3D Keypoint Prediction**:

    - **Function**: Fine-tunes a vision-language model (VLM) to predict the 3D keypoint location of the target object given the current visual observation and a sub-task description.
    - **Mechanism**: A pretrained VLM (e.g., a CLIP-based model) is fine-tuned on robot manipulation data. The input consists of an RGB image and a sub-task text description; the output is the keypoint coordinates in 3D space. The keypoints are language-aligned—for "grasp the red cup" versus "push the red cup," the predicted keypoint locations differ (corresponding to the handle and the side of the cup, respectively).
    - **Design Motivation**: VLMs possess rich visual-language priors, encoding knowledge of what a "cup" looks like and where "grasping" should be applied. Fine-tuning adapts these priors to the robot manipulation domain while preserving generalization to novel concepts. Predicting 3D rather than 2D keypoints ensures awareness of depth and spatial relationships.

3. **3D-Aware Representation**:

    - **Function**: Constructs a manipulation-oriented 3D spatial representation.
    - **Mechanism**: Multi-view RGB images and depth information are combined to build 3D local features centered on the predicted keypoints. The fine branch performs action prediction based on this 3D representation rather than on raw images.
    - **Design Motivation**: Robot manipulation is inherently a 3D task—grasp poses and placement positions are defined in 3D space. Pure 2D representations lack depth information and are prone to failure under viewpoint changes or object occlusion. A 3D representation provides a more robust foundation for spatial reasoning.

### Loss & Training

- **VLM Fine-tuning**: Keypoint prediction is trained with a regression loss (L1 or L2 distance) to align predicted 3D coordinates with annotated keypoint positions.
- **Policy Learning**: The fine branch employs Behavior Cloning (BC), learning a mapping from the 3D representation near the keypoint to end-effector actions.
- **Data Efficiency**: As few as 10 real-world demonstrations suffice to train an effective policy—far fewer than the hundreds typically required by conventional methods.

## Key Experimental Results

### Experimental Setup
- Simulation benchmark: GemBench (a manipulation benchmark specifically designed for generalization evaluation)
- Real-world experiments: Physical robot platform with 10 demonstrations
- Evaluation metric: Manipulation success rate
- Generalization dimensions: Novel instructions, novel object appearances, novel environment layouts

### Main Results

| Method | GemBench Avg. Success Rate | Training Trajectories | Notes |
|--------|---------------------------|----------------------|-------|
| SOTA (best baseline) | ~X% | ~5N | Requires many demonstrations |
| **CLAP** | **X + 12%** | **N (1/5)** | Higher success rate with less data |

CLAP exceeds the best prior method on GemBench by an average of 12 percentage points while using only 1/5 of the training trajectories.

### Real-Robot Experiments

| Setting | Success Rate | Notes |
|---------|-------------|-------|
| Training scenarios | High | Learned from only 10 demonstrations |
| Novel instructions | Successful generalization | Language-aligned keypoints correctly identify new targets |
| Novel environments | Successful generalization | 3D representation is robust to layout changes |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| w/o Task Decomposition | Success rate drops | Direct processing of complex instructions is ineffective |
| w/o VLM Fine-tuning (pretrained VLM only) | Success rate drops | Pretrained VLM is insufficiently adapted to manipulation |
| 2D representation instead of 3D | Success rate drops | Lack of depth information impairs precise manipulation |

### Key Findings

1. **All three components are indispensable**: Task decomposition, VLM fine-tuning, and 3D representation each contribute a distinct dimension of generalization.
2. **Extremely low data requirements**: Only 10 demonstrations are needed for real-world deployment, which is highly valuable for practical applications.
3. **Language alignment is critical**: Keypoints carry not only spatial location but also semantic information—the same object produces different keypoints under different instructions.

## Highlights & Insights

- **The ideal combination of low data and strong generalization**: By fully leveraging pretrained VLM priors, the method reduces sample requirements to a minimum while maintaining strong generalization.
- **Clean hierarchical design**: The coarse branch (VLM keypoint prediction) and fine branch (3D local action prediction) have well-defined, complementary roles.
- **Bridging language and 3D space**: Fine-tuning VLMs to map language semantics onto 3D keypoints provides an effective bridge between NLP and robot manipulation.
- **Practical orientation**: The ability to deploy with only 10 demonstrations gives this approach high real-world applicability.

## Limitations & Future Work

1. **Robustness of task decomposition**: If the LLM produces an inaccurate decomposition (e.g., omitting a critical step or incorrect ordering), the entire pipeline fails.
2. **Expressiveness of keypoints**: A single 3D keypoint may be insufficient to describe complex manipulations, such as tasks requiring bimanual coordination or multi-contact interactions.
3. **VLM fine-tuning data**: Although policy learning requires few demonstrations, VLM fine-tuning may still demand a non-trivial amount of annotated data.
4. **Dynamic environments**: The method appears designed for static or slowly changing environments; its adaptability to rapidly dynamic scenes (e.g., moving objects) remains unknown.
5. **Long-horizon tasks**: Error accumulation may become problematic when task decomposition yields a long sequence of sub-tasks.
6. **Limits of open-vocabulary generalization**: Although generalization to novel instructions is demonstrated, the boundary of generalization to entirely unseen object categories (never encountered during training) has not been thoroughly explored.

## Related Work & Insights

- **Relation to PerAct/RVT**: PerAct and RVT employ voxelized 3D representations for manipulation but lack a language-guided keypoint mechanism. CLAP's coarse-to-fine design is an effective complement to such approaches.
- **Relation to SayCan/Code-as-Policies**: These methods use LLMs for task planning without addressing generalization at the level of low-level manipulation policies. CLAP's task decomposition is conceptually similar but focuses more on the execution layer.
- **Trend of VLMs in robotics**: Works such as RT-2 and Octo also integrate VLMs into robot systems, but predominantly in an end-to-end fashion. CLAP's hierarchical approach (VLM → keypoints → local policy) offers a more controllable and data-efficient alternative.
- **Universality of 3D keypoints**: Keypoints as an intermediate representation for manipulation exhibit strong generality; future work could explore richer keypoint representations, such as oriented keypoints or keypoint graphs.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](../../ICCV2025/3d_vision/robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[CVPR 2026\] 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image](../../CVPR2026/3d_vision/3d-fixer_coarse-to-fine_in-place_completion_for_3d_scenes_from_a_single_image.md)
- [\[ICLR 2026\] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
