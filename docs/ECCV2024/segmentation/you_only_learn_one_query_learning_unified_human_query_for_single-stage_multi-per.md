---
title: >-
  [Paper Note] You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception
description: >-
  [ECCV 2024][Segmentation][Human-Centric Perception] Proposes the HQNet framework, which learns a unified Human Query representation to simultaneously perform multiple human-centric perception tasks—including pedestrian detection, instance segmentation, 2D pose estimation, 3D Mesh recovery, and attribute recognition—within a single-stage, single-model system. It also establishes the first comprehensive multi-task human perception benchmark, COCO-UniHuman.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Human-Centric Perception"
  - "Unified Query"
  - "Multi-Task Learning"
  - "DETR"
  - "Instance Segmentation"
date: 2026-05-08
content_hash: fd63511a8c7c12a5
---

# You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception

**Conference**: ECCV 2024  
**arXiv**: [2312.05525](https://arxiv.org/abs/2312.05525)  
**Code**: [https://github.com/lishuhuai527/COCO-UniHuman](https://github.com/lishuhuai527/COCO-UniHuman)  
**Area**: Image Segmentation  
**Keywords**: Human-Centric Perception, Unified Query, Multi-Task Learning, DETR, Instance Segmentation

## TL;DR

Proposes the HQNet framework, which learns a unified Human Query representation to simultaneously perform multiple human-centric perception tasks—including pedestrian detection, instance segmentation, 2D pose estimation, 3D Mesh recovery, and attribute recognition—within a single-stage, single-model system. It also establishes the first comprehensive multi-task human perception benchmark, COCO-UniHuman.

## Background & Motivation

**Background**: Human-Centric Perception (HCP) covers a variety of tasks such as pedestrian detection, human segmentation, 2D keypoint estimation, 3D human reconstruction, and attribute recognition. Mature methods exist for each task individually, but most are single-task models or multi-stage pipelines (e.g., detection followed by person-by-person analysis).

**Limitations of Prior Work**:
   - **Lack of a Unified Benchmark**: Different HCP tasks use independent datasets (e.g., COCO only has detection/pose/segmentation, CelebA only has facial attributes) with varying scales, viewpoints, and scenarios, preventing multi-task evaluation under a unified environment.
   - **Defects of Multi-stage Methods**: (1) Early decision-making issue—the entire pipeline heavily relies on the detector, and detection failure is irrecoverable; (2) Execution time scales linearly with the number of people, making multi-person scenarios highly inefficient; (3) The inherent relationships among different HCP tasks are neglected.
   - **Dataset Bias**: Datasets for different tasks have intrinsic scale differences (scene-level images vs. cropped images) and domain biases (laboratory vs. surveillance viewpoints). Naive joint training introduces substantial bias.

**Key Challenge**: Different HCP tasks require features of different granularities—detection requires global semantics, attributes require global + local semantics, segmentation requires fine-grained semantics, and pose estimation requires fine-grained localization. The challenge lies in accommodating these multi-granularity demands within a unified representation.

**Goal**: Design a single-stage, single-model framework to simultaneously handle all representative HCP tasks and construct an associated unified evaluation benchmark.

**Key Insight**: Drawing inspiration from query learning in DETR-based architectures, each human instance is encoded as a unified Human Query that carries multi-granularity information for use by all tasks.

**Core Idea**: Learn an all-in-one Human Query representation that encodes global/local appearance features and coarse/fine localization features for each human instance, completing unified multi-task inference via a shared Transformer decoder and lightweight task-specific heads.

## Method

### Overall Architecture

HQNet consists of four key components:
- **Backbone**: (such as ResNet-50 / Swin-L / ViT-L): Extra multi-scale image features.
- **Shared Transformer Encoder**: Enhances feature representation, outputting enhanced multi-scale features and positional encodings.
- **Task-Shared Transformer Decoder**: Uses Deformable Attention and Mixed Query Selection to initialize positional queries, progressively refining the content query (i.e., Human Query) through multiple decoding layers. All tasks share the same decoder.
- **Lightweight Task-Specific Heads**: Individual MLP/FC heads for each task that generate final predictions starting from the shared Human Query.

Core philosophy: Maximize weight sharing (backbone + encoder + decoder) and branch out only at the final prediction heads, ensuring scalability.

### Key Designs

1. **Human Query (Unified Human Query Representation)**:

    - **Function**: Expands the object query in DETR into a unified representation carrying multi-granularity human info.
    - **Mechanism**: The query is composed of a positional query (4D anchor box encoding center coordinates, width, and height) and a content query. The content query, which is the Human Query, interacts with image features via multi-layer deformable attention in the shared decoder to encode instance-level global/local appearance features and coarse/fine localization features.
    - **Design Motivation**: Different HCP tasks focus on different feature granularities. A sufficiently rich unified representation can simultaneously serve all tasks while leveraging multi-task synergy.

2. **HumanQuery-Instance Matching (HQ-Ins Matching)**:

    - **Function**: Performs query-GT matching during training by integrating losses from multiple tasks.
    - **Mechanism**: The matching cost is expressed as $\lambda_{cls}L_{cls} + \lambda_{det}L_{det} + \lambda_{seg}L_{seg} + \lambda_{pose}L_{pose}$, synthetically considering classification, detection, segmentation, and pose losses.
    - **Design Motivation**: Traditional DETR only uses detection loss for matching, which may lead to a scenario where one person's pose is matched to another person. HQ-Ins Matching ensures each query consistently maps to the same ground-truth (GT) instance across all tasks via multi-task joint constraints.

3. **Gender-aided human Model Selection (GaMS)**:

    - **Function**: Leverages gender prediction results to select the corresponding SMPL human model (male/female/neutral) for 3D Mesh recovery.
    - **Mechanism**: Selects different versions of the SMPL model during training and inference based on ground-truth gender labels/predictions.
    - **Design Motivation**: Prior works were forced to use a neutral model due to the lack of gender annotations. COCO-UniHuman provides gender labels, which can be leveraged to improve 3D reconstruction accuracy via cross-task synergy.

4. **Task-Specific Head Design (Three Paradigms)**:

    - **Coordinate Prediction Tasks** (detection, pose): Shared reference point, MLP regressing normalized offsets.
    - **Dense Prediction Tasks** (segmentation): Dot product of Human Query and high-resolution pixel embedding map to generate instance-aware pixel-wise classifications.
    - **Classification Tasks** (gender, age): Direct mapping of Human Query to classification predictions.

### Loss & Training

- **Detection Head**: Category classification + bbox regression ($L_1$ + GIoU)
- **Segmentation Head**: Dice loss + Focal loss (referencing Mask DINO)
- **Pose Head**: Keypoint coordinate regression + confidence prediction + auxiliary heatmap loss (used during training)
- **Attribute Head**: Gender—binary classification BCE; Age—85-class classification + softmax expected value estimation
- **3D Mesh Head**: SMPL pose/shape parameter regression
- **Contrastive DeNoising (CDN)**: Applied only to the detection task to accelerate convergence
- Training setup: 100 epochs, COCO-UniHuman train set, DINO data augmentation strategies

### COCO-UniHuman Dataset

Extended based on the COCO dataset:
- 200K images, 273K human instances
- Newly added annotations: **Gender** (body-based annotation), **Apparent Age** (a two-stage strategy: coarse-grained age group first, then fine-grained apparent age, averaged over votes from 10 annotators), **3D Mesh** (SMPL pseudo-GT generated using the EFT method)
- The first multi-person multi-task benchmark simultaneously covering classification (gender/age), detection (human/face), segmentation, and pose (2D/3D).

## Key Experimental Results

### Main Results (COCO-UniHuman val)

| Model | Backbone | Det. AP | Seg. AP | Pose AP | Gender AP | Age AP |
|------|----------|---------|---------|---------|-----------|--------|
| Faster R-CNN | R-50 | 65.3 | — | — | — | — |
| DINO | R-50 | 73.3 | — | — | — | — |
| Mask DINO | R-50 | 72.3 | 64.8 | — | — | — |
| Mask R-CNN | R-50 | 66.7 | 58.4 | — | — | — |
| PETR | R-50 | — | — | 68.8 | — | — |
| ViTPose† (top-down) | ViT-L | — | — | 78.2 | — | — |
| **HQNet (D+S+P+C)** | **R-50** | **74.9** | **65.8** | **69.3** | **56.0** | **53.8** |
| **HQNet** | **Swin-L** | **77.3** | **68.1** | **72.6** | **57.9** | **56.2** |
| **HQNet** | **ViT-L** | **78.0** | **68.6** | **75.3** | **58.0** | **58.0** |

### OCHuman Dataset (Crowded Occlusion Scenarios)

| Model | Backbone | Det. AP | Seg. AP | Pose AP | Description |
|------|----------|---------|---------|---------|------|
| Mask R-CNN | R-50-FPN | — | 16.9 | — | Two-stage method |
| CondInst | R-50-FPN | — | 20.1 | — | Single-stage segmentation |
| SBL† (top-down) | R-50 | — | — | 30.4 | Top-down pose estimation |
| HrHRNet† (bottom-up) | HRNet-w32 | — | — | 39.4 | Bottom-up pose estimation |
| CID† | HRNet-w32 | — | — | 44.0 | Single-stage pose estimation |
| **HQNet** | **R-50** | **29.5** | **31.1** | **40.0** | Significantly outperforms models of equivalent scale / parameters |
| **HQNet** | **ViT-L** | **35.8** | **38.8** | **45.6** | Comprehensive SOTA |

### 3D Mesh Reconstruction Ablation

| Method | Backbone | MPJPE↓ | PA-MPJPE↓ | Description |
|------|----------|--------|-----------|------|
| HMR | R-50 (GT bbox) | 109.62 | 72.03 | Using GT bboxes |
| HMR+ | R-50 (GT bbox) | 78.06 | 50.36 | Using GT bboxes |
| ROMP | R-50 | 119.52 | 72.27 | Single-stage without GT bboxes |
| HQNet w/o GaMS | R-50 | 87.00 | 54.92 | Without gender-aided selection |
| **HQNet w/ GaMS** | **R-50** | **84.74** | **50.80** | Gender-aided model selection |
| **HQNet** | **ViT-L** | **76.31** | **48.26** | Strongest configuration |

### Generalization Verification

| New Task | Method | Key Metric | Description |
|--------|------|----------|------|
| Face detection (finetune) | Faster R-CNN | AP 43.9 | Trained from scratch |
| Face detection (finetune) | ZoomNet | AP 58.2 | Specialized model |
| Face detection (finetune) | **HQNet (R-50)** | **AP 68.4** | Frozen backbone fine-tuning |
| MOT (zero-shot) | FairMOT (finetune) | IDF1 63.2 | Requires fine-tuning on MOT data |
| MOT (zero-shot) | **HQNet (R-50)** | **IDF1 64.6** | No MOT training data required |
| MOT (zero-shot) | **HQNet (ViT-L)** | **IDF1 69.1** | No MOT training data required |

### Key Findings

- **Multi-task synergy is effective**: Fully joint training of HQNet (D+S+P+C) shows improvements in all metrics compared to single-task subsets (D+S, D+S+P), proving positive synergy among tasks.
- **Significance of HQ-Ins Matching**: Utilizing multi-task joint matching avoids matching detection and pose results to different individuals.
- **GaMS Effectiveness**: Leveraging gender prediction to select the SMPL model reduces MPJPE from 87.00 to 84.74.
- **Strong Generalization**: Human Query performs excellently on unseen tasks like face detection and Multi-Object Tracking (MOT), demonstrating that the learned representations are generalizable.
- **High Efficiency & Scalability**: Multi-task weight sharing (backbone, encoder, decoder) ensures that each new task only introduces a lightweight head, resulting in minimal parameter overhead.
- **HQNet with R-50 (trained on COCO) significantly outperforms specialized methods of the same backbone on OCHuman**: Seg AP 31.1 vs CondInst 20.1, Pose AP 40.0 vs SBL 30.4.

## Highlights & Insights

- The design philosophy of **"one query for all"** is simple yet powerful; Human Query simultaneously encodes appearance, position, and structural info, which each task decodes as needed.
- **Long-term value of the COCO-UniHuman dataset**: The first unified multi-person benchmark covering four major categories and seven HCP tasks, filling a gap in the community. The body-based age annotation and two-stage annotation strategy are also valuable contributions.
- **Design choice of a shared decoder**: Unlike prior works utilizing task-specific decoders, HQNet employs a fully shared decoder, maximizing knowledge sharing while maintaining scalability.
- **Zero-shot MOT capability**: Despite not being trained on MOT, Human Query can serve as Re-ID features to achieve competitive tracking performance, suggesting that the learned representation possesses deep instance-discriminative capability.
- **Practicality**: Single-stage inference efficiency does not scale with the number of persons, making it suitable for practical deployment.

## Limitations & Future Work

- Imbalanced gender distribution in the dataset (male:female $\approx$ 65:35), which might introduce bias.
- Age distribution is biased toward 25-35 years, potentially leading to inadequate estimation capabilities for children and elderly individuals.
- The 3D Mesh ground truth relies on EFT pseudo-annotations rather than real ground truths, which limits overall quality.
- Compared to top-down methods (particularly ViTPose), a performance gap in pose estimation still exists (an inherent limitation of single-stage methods).
- The absolute AP values for attribute prediction (gender/age) remain relatively low (56-58%), leaving substantial room for improvement.
- Small-category scale people were excluded from the evaluation, meaning long-distance, small-scale human sensing has not been fully assessed.
- HQ-Ins Matching increases the computational complexity of instance matching.

## Related Work & Insights

- **DINO** [ICLR 2023]: The primary technical foundation of HQNet, utilizing Mixed Query Selection + CDN + Deformable Attention.
- **Mask DINO** [CVPR 2023]: A unified DETR framework for detection and segmentation; the segmentation head design of HQNet references this work.
- **UniHCP** [CVPR 2023]: Large-scale multi-task human pre-training, but still limited to single-task inference; HQNet realizes true single-stage multi-task inference.
- **PETR** [CVPR 2022]: Single-stage multi-person pose estimation; HQNet's pose head references its joint decoder layer design.
- **Insight**: Shared query is an effective paradigm for achieving multi-task unification, which can be extended to more entity-centric perception tasks (such as hand pose estimation, action recognition, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Although extending the DETR query philosophy, this is the first systematic implementation of unified HCP multi-tasking. The COCO-UniHuman benchmark is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 tasks, 3 datasets (COCO-UniHuman/OCHuman/PoseTrack21/Human-Art), comparisons with extensive baselines, and comprehensive generalization validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the dataset comparison in Table 1 and overall experiments in Table 2 are professionally formatted, though some details require referring to the supplementary materials.
- **Value**: ⭐⭐⭐⭐⭐ A trinity of dataset, method, and benchmark, significantly driving the unified human perception field. The generalization capability of Human Query demonstrates its potential as a universal human representation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](../../NeurIPS2025/segmentation/hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](../../NeurIPS2025/segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](../../CVPR2025/segmentation/mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
