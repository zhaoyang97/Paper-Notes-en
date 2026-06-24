---
title: >-
  [Paper Note] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models
description: >-
  [CVPR 2025][3D Vision][Monocular Depth Perception] This paper proposes the DepthCues benchmark to systematically evaluate the depth perception capabilities of 20 large-scale pre-trained vision models across six human monocular depth cue tasks (elevation, light-shadow, occlusion, perspective, size, and texture gradient), revealing the emergence of human-like depth cues in modern vision models.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Monocular Depth Perception"
  - "Vision Foundation Models"
  - "Depth Cues"
  - "Benchmark Evaluation"
  - "Representation Probing"
date: 2026-05-08
content_hash: 9c4f9d90fb32c320
---

# DepthCues: Evaluating Monocular Depth Perception in Large Vision Models

**Conference**: CVPR 2025  
**arXiv**: [2411.17385](https://arxiv.org/abs/2411.17385)  
**Code**: [https://danier97.github.io/depthcues](https://danier97.github.io/depthcues)  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Perception, Vision Foundation Models, Depth Cues, Benchmark Evaluation, Representation Probing

## TL;DR

This paper proposes the DepthCues benchmark to systematically evaluate the depth perception capabilities of 20 large-scale pre-trained vision models across six human monocular depth cue tasks (elevation, light-shadow, occlusion, perspective, size, and texture gradient), revealing the emergence of human-like depth cues in modern vision models.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Large-scale pre-trained vision models (such as DINOv2, Stable Diffusion) exhibit surprising capabilities in monocular depth estimation tasks, despite not receiving explicit depth supervision during the pre-training phase. A core question is: **how does depth perception emerge in these models?** The human visual system relies on various monocular depth cues (elevation, light-shadow, occlusion, perspective, size, texture gradient) to perceive depth, but existing benchmarks only evaluate the depth estimation accuracy of models, without exploring whether models understand these underlying visual cues. From the perspective of human vision science, this paper constructs a specialized benchmark to bridge this research gap.

## Method

### Overall Architecture

The DepthCues benchmark comprises six tasks, each corresponding to a human monocular depth cue. The evaluation protocol employs **feature probing**: freezing the feature extractor $\phi(\cdot)$ of the pre-trained model and training a lightweight probing head $g_\theta$ on top to solve a specific task, using the probing performance to measure the model's understanding of that cue.

### Key Designs

1. **Design of Six Depth Cue Tasks**:
    - Function: Comprehensively cover the core cues of human monocular depth perception.
    - Mechanism: Each task is adapted from existing datasets to test elevation (horizon estimation), light-shadow (object-shadow association), occlusion (whether an object is occluded), perspective (vanishing point estimation), size (3D object volume comparison), and texture gradient (depth ordering of textured planes).
    - Design Motivation: Deconstruct abstract "depth perception" into quantifiable sub-abilities, enabling a more fine-grained analysis.

2. **Task-Specific Feature Extraction**:
    - Function: Extract feature representations suitable for different tasks from pre-trained models.
    - Mechanism: For object-level tasks (occlusion, light-shadow, size, texture), mask pooling is used to extract regional features $\mathbf{f} = \frac{\sum_{h,w} M_A \odot \text{up}(\phi(I))}{\sum_{h,w} M_A}$; for global tasks (elevation, perspective), the complete feature map is used.
    - Design Motivation: Different cues require different levels of spatial information; a unified feature extraction approach would discard task-relevant information.

3. **Multi-Level Probing and Evaluation**:
    - Function: Fairly compare models with different architectures and pre-training configurations.
    - Mechanism: MLP probing heads are used for binary classification tasks, while attention probing heads are used for regression tasks; the optimal layer is searched for each model, with results averaged over 5 random seeds.
    - Design Motivation: Nonlinear probers are better suited than linear probers for capturing complex depth cue representations.

### Loss & Training

- Binary classification tasks (light-shadow, occlusion, size, texture gradient): Binary Cross-Entropy loss.
- Regression tasks (elevation, perspective): Mean Squared Error loss.
- Hyperparameter search is employed to determine the optimal probing layer, and training is conducted independently for each task.

## Key Experimental Results

### Main Results

| Model | Average Rank | Depth Est. NYUv2 Acc(%) | Characteristics |
|------|---------|---------------------|------|
| DepthAnythingv2 | 1 | Strongest | Dedicated depth model, most comprehensive across all six cues |
| DINOv2-b14 | Top-5 | 87.78 | Self-supervised model, cues emerge even without explicit depth supervision |
| Stable Diffusion | Top-5 | - | Generative models also possess depth understanding |
| CLIP | Last | 43.78 | Vision-language models exhibit the weakest depth cues |

### Ablation Study

| Configuration | NYUv2 Acc(%) | DIW WHDR(%) | Description |
|------|-------------|-------------|------|
| Original DINOv2 | 87.78 | 11.99 | Baseline |
| DINOv2+DC Fine-tuned | 87.06 | 11.95 | Slightly decreases after fine-tuning |
| concat(DINOv2, DINOv2+DC) | **88.46** | **11.72** | Concatenating original + fine-tuned features is optimal |
| Original CLIP | 43.78 | 35.25 | Baseline |
| concat(CLIP, CLIP+DC) | **44.32** | **33.53** | Explicitly injecting cues improves depth perception |

### Key Findings

- **Human-like depth cues are stronger in larger and newer models**: Self-supervised/generative models like DINOv2 and Stable Diffusion display strong depth cues.
- **Multi-view models excel at texture gradient cues**: Multi-view models like CroCo and DUSt3R rank in the top four on texture-grad.
- **DepthCues performance is highly correlated with depth estimation**: The Spearman correlation is significant, validating the effectiveness of the benchmark.
- **No single model is optimal across all six cues**: Every model has its own weaknesses.
- **SigLIP is far superior to CLIP**: Although both are trained on image-text matching, SigLIP uses 9 times more data, resulting in significantly better cue understanding.

## Highlights & Insights

- First to systematically introduce human vision science theories of monocular depth cues into the analysis of foundation models.
- Discovering that fine-tuning on DepthCues can enhance models' depth perception capabilities, even with extremely sparse annotations (<35K images), hinting at a new direction that does not rely on dense depth supervision.
- Humans achieve 95%±1.48% accuracy on the test set, indicating the task design is reasonable and solvable by humans.

## Limitations & Future Work

- Only public pre-trained weights are analyzed, without directly studying the causal impacts of pre-training objectives and datasets on cue emergence.
- Depth cues associated with ego-motion and scene motion are not covered.
- The texture gradient task uses synthetic data, which may result in lower correlation with other cues.
- The data scale of DepthCues is limited; to achieve improvements, original features must be concatenated post-fine-tuning, indicating limitations in generalization.

## Related Work & Insights

- Complementary to 3D perception probing works such as Probe3D and GeoMeter, but more focused on the systematic analysis of monocular depth cues.
- Intuition from fine-tuning experiments: Designing larger-scale cue-learning datasets or incorporating cue learning as an auxiliary pre-training task for depth estimation can be considered.
- Provides a new perspective for understanding the success of models like Depth Anything V2 and Marigold.

## Rating

- Novelty: ⭐⭐⭐⭐ The research perspective analyzing model depth perception from human vision science is novel, but the probing methodology itself is standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 20 models, 6 tasks, and multiple correlation analyses and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and deeply analyzed with effective visual designs.
- Value: ⭐⭐⭐⭐ Provides a valuable benchmark and analytical tools to the community, though practical application value warrants further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
