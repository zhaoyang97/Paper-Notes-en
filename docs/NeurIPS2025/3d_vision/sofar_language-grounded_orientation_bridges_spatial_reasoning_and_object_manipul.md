---
title: >-
  [Paper Note] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation
description: >-
  [NeurIPS 2025][3D Vision][semantic orientation] This paper introduces the concept of **Semantic Orientation**, which describes object directions using natural language (e.g., the "insertion direction" of a USB plug or the "handle direction" of a cup). It constructs the large-scale OrienText300K dataset to train the PointSO model for zero-shot orientation prediction, and integrates these components into the SoFar system for 6-DoF scene understanding and robotic manipulation.
tags:
  - NeurIPS 2025
  - 3D Vision
  - semantic orientation
  - 6-DoF manipulation
  - spatial reasoning
  - 3D scene graph
  - point cloud
date: 2026-05-08
content_hash: 5d5a86870a6dee15
---

# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation

**Conference**: NeurIPS 2025
**arXiv**: [2502.13143](https://arxiv.org/abs/2502.13143)
**Authors**: Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong et al. (Tsinghua University, SJTU, Galbot, Peking University, UIUC, ShanghaiTech University)
**Code**: [SoFar](https://github.com/SoFar-LGO/SoFar)
**Area**: 3D Vision
**Keywords**: semantic orientation, 6-DoF manipulation, spatial reasoning, 3D scene graph, point cloud

## TL;DR
This paper introduces the concept of **Semantic Orientation**, which describes object directions using natural language (e.g., the "insertion direction" of a USB plug or the "handle direction" of a cup). It constructs the large-scale OrienText300K dataset to train the PointSO model for zero-shot orientation prediction, and integrates these components into the SoFar system for 6-DoF scene understanding and robotic manipulation.

## Background & Motivation

Existing spatial reasoning methods primarily focus on **positional relationships** (left/right/front/back) while neglecting object **orientation**—a critical factor for precise 6-DoF manipulation. For example:
- Cutting bread with a knife requires knowing the blade direction
- Uprighting a fallen wine glass requires knowing which way the rim faces
- Plugging in a charger requires aligning with the socket orientation

Conventional orientation representations rely on quaternions or Euler angles, which depend on predefined reference frames or template models, and suffer from significant limitations:

**Lack of semantic grounding**: Rotation matrices cannot express functional semantics such as "handle direction" or "insertion direction."

**Poor generalizability**: Per-category coordinate frame definitions are required, making it impossible to handle novel objects in open-world settings.

**Missing VLM capability**: Current vision-language models (GPT-4V, LLaVA, etc.) struggle to understand and output precise object orientations.

The core insight is that humans intuitively describe orientations in natural language (e.g., "blade facing down"). This **semantically grounded orientation representation** can bridge geometric reasoning and functional semantics, enabling reference-frame-free, open-vocabulary orientation understanding.

## Method

### Semantic Orientation Definition
For an object $X$ and a language description $\ell$, the semantic orientation is defined as a unit direction vector:
$$\mathbf{s}_{\ell}^{X} = \mathcal{F}(X, \ell) \in S(2)$$
where $\ell$ can be a generic direction (front/up), an object part (handle/cap), or an interaction action (pour/insert). A single object may be associated with multiple semantic orientations, forming the set $S_X = \{\mathbf{s}_{\ell_1}^X, \ldots, \mathbf{s}_{\ell_n}^X\}$.

### OrienText300K Dataset
Constructed from Objaverse (~800K 3D models) through rigorous filtering and annotation:

**Data Filtering** (6 criteria): standard orthographic views, no ground assistance, valid objects, high quality, identifiable, non-scene-level. GPT-4o is used for automatic filtering with an accuracy of 88.3%.

**Data Annotation**: GPT-4o is used as a discriminator to interpret semantic content from six-view images and generate semantic–view pairs. Annotation accuracy is 97.1%.

The final dataset contains 350K+ clean samples rendered into 8M+ high-quality images.

### PointSO Model Architecture
A cross-modal 3D-language fusion model based on a plain Transformer:

1. **3D Encoding**: FPS-sampled seed points → KNN grouping → lightweight PointNet for local geometric feature extraction
2. **Language Encoding**: CLIP (frozen) for global text feature extraction
3. **Cross-modal Fusion**: Text tokens are element-wise added to point tokens at each layer (simple yet most effective)
4. **Prediction Head**: MLP maps the [CLS] token to a direction vector
5. **Loss Function**: Negative cosine similarity $\mathcal{L}_{\text{cos}}$

### SoFar System Framework
A complete reasoning system integrating PointSO with foundation models (SAM, Florence-2):

**6-DoF Scene Graph Construction**:
1. VLM extracts task-relevant object phrases from the language query
2. Florence-2 + SAM performs language-conditioned segmentation to obtain 3D point clouds
3. VLM generates orientation descriptions; PointSO predicts semantic orientations
4. Scene graph $\mathcal{G} = (\mathbf{V}, \mathbf{E})$ is constructed, with nodes containing object ID, 3D position, bounding box, and semantic orientation set

**Chain-of-Thought Spatial Reasoning**:
1. Analyze the scene and query to identify relevant objects
2. Compute target positions and orientations
3. Translation $\mathbf{t}_i = \tilde{\mathbf{c}}_i - \mathbf{c}_i$; rotation is estimated from initial and target semantic orientations via the Kabsch-Umeyama algorithm

**Low-Level Motion Execution**: GSNet generates grasp candidates → optimal grasp selection → OMPL plans collision-free trajectories

## Key Experimental Results

### Table 1: Open6DOR 6-DoF Object Rearrangement Evaluation

| Method | Position L0 | Position L1 | Position Overall | Rotation L0 | Rotation L1 | Rotation L2 | Rotation Overall | 6-DoF Position | 6-DoF Rotation | 6-DoF Overall | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4V | 46.8 | 39.1 | 45.2 | 9.1 | 6.9 | 11.7 | 9.2 | - | - | - | - |
| Dream2Real | 17.2 | 11.0 | 15.9 | 37.3 | 27.6 | 26.2 | 31.3 | 26.2 | 18.7 | 13.5 | 358.3s |
| Open6DOR-GPT | 78.6 | 60.3 | 74.9 | 45.7 | 32.5 | 49.8 | 41.1 | 84.8 | 40.0 | 35.6 | 126.3s |
| SoFar-LLaVA | 86.3 | 57.9 | 78.7 | 62.5 | 30.2 | 67.1 | 48.6 | 83.0 | 48.2 | 40.3 | 9.6s |
| **SoFar** | **96.0** | **81.5** | **93.0** | **68.6** | **42.2** | **70.1** | **57.0** | **92.7** | **52.7** | **48.7** | **8.5s** |

SoFar comprehensively outperforms all baselines on perception tasks: position overall 93.0% (vs. Open6DOR-GPT 74.9%, +18.1%); rotation overall 57.0% (+15.9%); 6-DoF overall 48.7% (+13.1%); inference time only 8.5s (vs. 126.3s, ~15× speedup).

On execution tasks (Open6DOR V2), SoFar also surpasses Octo and OpenVLA, achieving a 6-DoF overall success rate of 18.4% vs. 8.0%/8.2%.

### Table 4: SimplerEnv Google Robot Evaluation (Success Rate)

| Method | Training Data | Pick Coke Can Avg | Move Near Avg | Open/Close Drawer Avg | Overall Avg |
|---|---|---|---|---|---|
| RT-1-X | OXE | 0.490 | 0.323 | 0.294 | 0.397 |
| RT-2-X | OXE | 0.823 | 0.792 | 0.353 | 0.661 |
| OpenVLA | OXE | 0.545 | 0.477 | 0.177 | 0.411 |
| **SoFar** | **Zero-Shot** | **0.907** | **0.740** | **0.297** | **0.676** |

Under the Visual Matching setting, SoFar achieves an overall success rate of **74.9%**, surpassing all baselines (including methods trained on OXE data) in a zero-shot transfer setting. On the Pick Coke Can task, SoFar reaches 92.3%.

### Table 2: PointSO Semantic Orientation Prediction Accuracy

| Model | 45° | 30° | 15° | 5° | Mean |
|---|---|---|---|---|---|
| PointSO-S | 77.34 | 74.22 | 67.97 | 60.94 | 70.12 |
| PointSO-B | 79.69 | 77.34 | 70.31 | 62.50 | 72.46 |
| PointSO-L | 81.25 | 78.13 | 72.66 | 65.63 | 74.42 |

Even under the strict 5° threshold, PointSO-L achieves 65.63% accuracy. Under robustness evaluations (single-view, noisy input, random rotation), the model maintains strong performance.

### Table 6: 6-DoF SpatialBench Spatial Reasoning Evaluation

| Method | Position Relative | Position Absolute | Orientation Relative | Orientation Absolute | Overall |
|---|---|---|---|---|---|
| GPT-4o | 49.4 | 28.4 | 44.2 | 25.8 | 36.2 |
| SpatialBot | 50.9 | 21.6 | 39.6 | 22.9 | 32.7 |
| RoboPoint | 43.8 | 30.8 | 33.8 | 25.8 | 33.5 |
| **SoFar** | **59.6** | **33.8** | **54.6** | **31.3** | **43.9** |

SoFar achieves an overall score of 43.9%, outperforming the second-best method by more than **18%** (relative), with a particularly pronounced advantage in orientation reasoning.

## Highlights & Insights
- **Semantic Orientation Concept**: The first work to bind object orientation with natural language semantics, proposing a reference-frame-free, open-vocabulary orientation representation that bridges the gap between positional and orientational reasoning.
- **Large-Scale Data with Automatic Annotation**: OrienText300K (350K+ models, 8M+ images) is constructed via GPT-4o automatic filtering and annotation, avoiding expensive robot data collection, with annotation accuracy of 97.1%.
- **Zero-Shot Generalization**: SoFar operates in a zero-shot manner across Open6DOR, SimplerEnv, and real-world tasks, yet outperforms VLA models trained on large-scale robot trajectory data (Octo, OpenVLA).
- **Cross-Embodiment Generalization**: Supports diverse end-effectors including parallel grippers, suction cups, and dexterous hands, as well as multiple task types including manipulation, navigation, and VQA.
- **Inference Efficiency**: 8.5s inference time, approximately 15× faster than Open6DOR-GPT and 42× faster than Dream2Real.

## Limitations & Future Work
- **Cascading Errors in Modular System**: Errors from individual sub-modules (segmentation, grasping, orientation prediction) accumulate; for instance, unstable grasps can cause rotational deviation during placement.
- **Execution Success Rate Has Room for Improvement**: On Open6DOR V2 execution tasks, the 6-DoF success rate is only 18.4%, with some objects being inherently difficult to manipulate.
- **Dependency on Depth Input**: RGB-D images are required; an additional depth estimation module is needed for RGB-only settings.
- **Limited Closed-Loop Capability**: Closed-loop behavior is approximated via VLM error detection and retry (up to 3 attempts) rather than true closed-loop control.
- **Single-Source Data for OrienText300K**: Data sourced exclusively from Objaverse may introduce category distribution bias.

## Related Work & Insights
- **Spatial Reasoning**: SpatialVLM, SpatialBot, and RoboPoint focus on positional relationships without modeling orientation; SoFar extends reasoning to full 6-DoF.
- **Robotic Manipulation**: RT-1/RT-2 (end-to-end policies), VoxPoser (3D value maps), and CoPa (generalizable manipulation) do not explicitly model semantic orientation.
- **VLA Models**: Octo, OpenVLA, and SpatialVLA require large-scale robot data for training, limiting generalizability; SoFar surpasses them in a zero-shot setting.
- **Object Pose Estimation**: Traditional methods rely on CAD models or category-level templates and cannot handle novel open-world objects; semantic orientation provides a more flexible alternative.
- **Scene Graphs**: Conventional scene graphs encode object relationships but lack orientation information; SoFar's 6-DoF scene graph is the first to incorporate semantic orientations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The semantic orientation concept is original; binding natural language to 3D direction vectors is a novel and elegant formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers simulation (Open6DOR V1/V2, SimplerEnv) + real-world (60 tasks, 100+ objects) + VQA (SpatialBench), with comprehensive multi-baseline comparisons and ablations.
- Writing Quality: ⭐⭐⭐⭐ — Concepts are clearly presented with rich illustrations and complete system descriptions; the large content volume leads to some compression of details.
- Value: ⭐⭐⭐⭐⭐ — Introduces a new representational paradigm for 6-DoF robotic manipulation; the dataset and benchmark are likely to facilitate future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [\[CVPR 2026\] Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models](../../CVPR2026/3d_vision/scalable_object_relation_encoding_for_better_3d_spatial_reasoning_in_large_langu.md)
- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)

</div>

<!-- RELATED:END -->
