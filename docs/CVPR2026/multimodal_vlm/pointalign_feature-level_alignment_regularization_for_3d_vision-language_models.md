---
title: >-
  [Paper Note] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][3D point cloud understanding] PointAlign is proposed to apply feature-level alignment regularization to point cloud tokens at intermediate LLM layers (aligned with Q-Former outputs) in 3D VLMs…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "3D point cloud understanding"
  - "vision-language models"
  - "feature alignment"
  - "geometric information preservation"
  - "regularization"
date: 2026-05-08
content_hash: 8b1b0941d327ad6d
---

# PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.00412](https://arxiv.org/abs/2603.00412)  
**Code**: [https://github.com/yharoldsu0627/PointAlign](https://github.com/yharoldsu0627/PointAlign)  
**Area**: Multimodal VLM
**Keywords**: 3D point cloud understanding, vision-language models, feature alignment, geometric information preservation, regularization

## TL;DR
PointAlign is proposed to apply feature-level alignment regularization to point cloud tokens at intermediate LLM layers (aligned with Q-Former outputs) in 3D VLMs. By training only a lightweight alignment projector and LoRA adapters, the method effectively prevents geometric information from degrading during language modeling, achieving a 7.50pp improvement on open-vocabulary classification.

## Background & Motivation
**Background**: 3D vision-language models (3D VLMs) are critical for applications such as robotics, autonomous driving, and AR, yet remain limited by the scarcity of 3D-text paired data.

**Limitations of Prior Work**: Existing methods (PointLLM, ShapeLLM, MiniGPT-3D) rely solely on next-token prediction loss for training, with only language tokens providing supervision signals. This leads to:
   - Low utilization efficiency of limited 3D data
   - Progressive degradation and loss of valuable geometric information in intermediate representations as they propagate through LLM layers

**Key Challenge**: The language modeling objective only rewards geometric features that directly contribute to next-token prediction, while structural cues useful for spatial reasoning but irrelevant to the current language task are discarded during training.

**Goal**: To explicitly supervise point cloud tokens at intermediate LLM layers to preserve fine-grained 3D geometric-semantic information, without introducing any inference overhead.

**Key Insight**: Q-Former outputs are observed to encode both geometric and semantic information (owing to point cloud-text paired training), making them ideal targets for internal supervision.

**Core Idea**: A consistency loss is employed to align point cloud tokens at intermediate LLM layers with frozen Q-Former outputs via a lightweight alignment projector, which is discarded at inference time with zero additional cost.

## Method

### Overall Architecture
Two-stage training: Stage 1 follows MiniGPT-3D pretraining; Stage 2 freezes the encoder, Q-Former, and projector, training only LoRA and the alignment projector. The alignment projector is used exclusively during training and discarded at inference.

### Key Designs

1. **Alignment Target Selection — Q-Former Output $\bar{Q}$**:

    - Why not use point cloud encoder outputs? The encoder captures only geometric features and lacks semantic information.
    - Why not use deep LLM representations? Deep representations may have already lost 3D information.
    - Q-Former outputs, under direct supervision, retain the richest combination of geometric and semantic information, making them the optimal alignment target.

2. **Alignment Projector $f_\pi$ (3-layer Linear + SiLU)**:

    - Maps point cloud tokens $T_{pc}^{(\ell)}$ at LLM layer $\ell$ into the Q-Former feature space.
    - Architecture: $\mathbb{R}^C \to \mathbb{R}^{d_h} \to \mathbb{R}^{d_h} \to \mathbb{R}^{D_1}$, with only 8.39M parameters.
    - Completely discarded at inference with zero overhead.

3. **Alignment Loss**:

    - Cosine similarity loss: $\mathcal{L}_{align} = -\frac{1}{o}\sum_{i=1}^{o} \frac{\tilde{Q}_i^\top \bar{Q}_i}{\|\tilde{Q}_i\|_2 \|\bar{Q}_i\|_2}$
    - Focuses on feature direction rather than magnitude, better suited for cross-space alignment.
    - Gradients through Q-Former output $\bar{Q}$ are detached to prevent backpropagation from affecting frozen modules.
    - Total loss: $\mathcal{L}_{total} = \mathcal{L}_{ntp} + \lambda \mathcal{L}_{align}$

### Loss & Training
Stage 2 jointly trains LoRA and the alignment projector using $\mathcal{L}_{total} = \mathcal{L}_{ntp} + \lambda \mathcal{L}_{align}$, updating only a minimal number of parameters.

## Key Experimental Results

### Main Results (3D Object Classification)

| Model | LLM Size | ModelNet40 Avg | Objaverse Avg | Overall Avg |
|------|---------|:---------:|:---------:|:---------:|
| PointLLM-7B | 7B | 50.85 | 62.50 | 56.68 |
| PointLLM-13B | 13B | 52.19 | 62.25 | 57.22 |
| MiniGPT-3D (Baseline) | 2.7B | 61.24 | 66.75 | 64.00 |
| **PointAlign (Ours)** | **2.7B** | **61.17** | **71.00** | **66.08** |

### Ablation Study

| Configuration | Objaverse Avg | Notes |
|------|:----------:|------|
| Baseline (MiniGPT-3D) | 66.75 | No alignment regularization |
| + Align to encoder features | 67.50 | Geometric information only; limited benefit |
| + Align to Q-Former outputs | **71.00** | Geometric + semantic information; best performance |
| + Align to deep LLM features | 68.25 | Partial loss of 3D information |

### Key Findings
- Achieves **7.50pp** improvement on the most challenging open-vocabulary Objaverse classification (66.75→71.00 on 2-prompt avg) and **4.88pp** on 3D captioning.
- Performance on ModelNet40 is nearly unchanged (−0.07pp), indicating that alignment regularization primarily benefits difficult or open-ended scenarios.
- A 2.7B-parameter model surpasses the 13B PointLLM, demonstrating the data efficiency of the approach.
- Regarding alignment layer $\ell$: intermediate layers yield the best results, with performance degrading for both shallower and deeper choices.

## Highlights & Insights
- **Training trick with zero inference overhead**: The alignment projector is used only during training and discarded at inference — a design pattern worth adopting in other VLMs, analogous to the use of auxiliary heads in knowledge distillation.
- **Extension of 2D VLM findings to 3D**: Prior work in 2D VLMs has shown that visual representations degrade in deeper layers without explicit visual supervision; this paper extends that finding to 3D point clouds and provides a concrete solution.
- **Data efficiency**: Under conditions of extreme 3D data scarcity, internal alignment regularization maximizes the utility of limited data.

## Limitations & Future Work
- The method is built upon MiniGPT-3D; the relatively small model scale may limit the performance ceiling.
- The alignment layer $\ell$ requires hyperparameter tuning, and different model architectures may require different settings.
- Validation is limited to single-object understanding; scene-level multi-object 3D understanding is not addressed.
- Future work could explore multi-layer alignment (aligning multiple intermediate layers rather than a single one) or dynamic alignment layer selection.

## Related Work & Insights
- **vs. PointLLM**: PointLLM achieves 3D-text alignment through full model fine-tuning, incurring high computational cost (200+ GPU-hours); PointAlign surpasses its performance by training only a small number of parameters.
- **vs. representation supervision methods in 2D VLMs**: Reconstruction-based methods in 2D (supervising intermediate representations by recovering visual inputs) tend to capture low-level textures; 3D understanding requires capturing structural relationships and geometric configurations, making direct alignment with Q-Former semantic features more appropriate.

## Supplementary Analysis
- The alignment projector contains only 8.39M parameters, negligible compared to MiniGPT-3D's 2.7B.
- During Stage 2 training, gradients through Q-Former output $\bar{Q}$ are detached — a critical design choice, as omitting this would allow the alignment loss to modify the Q-Former via backpropagation.
- On open-vocabulary Objaverse classification, instruction-based evaluation shows a larger gain (65→72.5, +7.5pp), suggesting that alignment regularization is especially beneficial for understanding free-form instructions.
- Feature similarity visualizations demonstrate that in the baseline, the cosine similarity between point cloud tokens at intermediate layers and Q-Former outputs decreases with depth, whereas PointAlign maintains stable similarity throughout.
- The method is built on MiniGPT-3D's Q-Former architecture; adaptation to other 3D VLMs using direct projection schemes (e.g., PointLLM) would require adjustments to the alignment target.

## Rating
- Novelty: ⭐⭐⭐ Clear motivation, moderate technical novelty
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation analysis with convincing feature quality visualizations
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly articulated motivation
- Value: ⭐⭐⭐⭐ Practically informative for the 3D VLM community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
