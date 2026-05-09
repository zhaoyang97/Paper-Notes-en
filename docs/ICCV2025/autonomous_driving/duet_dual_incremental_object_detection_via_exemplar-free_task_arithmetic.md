---
title: >-
  [Paper Note] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic
description: >-
  [ICCV 2025][Autonomous Driving][Incremental Object Detection] This paper proposes DuET, a framework that, for the first time, addresses both class-incremental and domain-incremental object detection simultaneously (Dual Incremental Object Detection, DuIOD) via exemplar-free Task Arithmetic model merging. It introduces a Directional Consistency Loss to mitigate sign conflicts, achieving substantial improvements over existing methods on the Pascal Series and Diverse Weather Series benchmarks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Incremental Object Detection
  - Class Incremental
  - Domain Incremental
  - Task Arithmetic
  - Model Merging
  - Catastrophic Forgetting
  - YOLO11
  - RT-DETR
date: 2026-05-08
content_hash: 2d8bca9ce33d6b69
---

# DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic

**Conference**: ICCV 2025
**arXiv**: [2506.21260](https://arxiv.org/abs/2506.21260)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: Incremental Object Detection, Class Incremental, Domain Incremental, Task Arithmetic, Model Merging, Catastrophic Forgetting, YOLO11, RT-DETR

## TL;DR

This paper proposes DuET, a framework that, for the first time, addresses both class-incremental and domain-incremental object detection simultaneously (Dual Incremental Object Detection, DuIOD) via exemplar-free Task Arithmetic model merging. It introduces a Directional Consistency Loss to mitigate sign conflicts, achieving substantial improvements over existing methods on the Pascal Series and Diverse Weather Series benchmarks.

## Background & Motivation

### The Dual Challenge in Real-World Scenarios

Real-world object detection systems (e.g., autonomous driving, surveillance) must continuously learn new categories while adapting to environmental changes (illumination, weather, style, and other domain shifts). Existing methods address only one of these dimensions:

- **Class-Incremental Object Detection (CIOD)**: Incrementally learns new categories but assumes a fixed domain, leading to severe performance degradation on unseen domains.
- **Domain-Incremental Object Detection (DIOD)**: Adapts to new domains but assumes a fixed category set, making it unable to detect new classes.

Both families of methods fail when class shifts and domain shifts occur simultaneously—precisely the scenario most common in practice.

### Problem Formulation of DuIOD

The authors propose **Dual Incremental Object Detection (DuIOD)** as a more realistic setting: the model must handle a sequence of incremental tasks, each introducing new categories $\mathcal{C}_t$ and a new domain $\mathcal{D}_t$, without retaining any historical training data (exemplar-free). This introduces three core challenges:

**Catastrophic Forgetting**: Learning new knowledge erases old knowledge.

**Domain Generalization**: Maintaining performance on old categories in new domains.

**Background Shift**: Old categories are unannotated in new tasks and are treated as background during training.

### Limitations of Prior Work

- CL-DETR (CIOD SOTA): Relies on the DETR architecture and exemplar replay; performs poorly under severe domain shift.
- LDB (DIOD SOTA): Freezes the base model and learns domain biases; cannot handle category changes.
- LwF / ERD: General continual learning methods that degrade significantly in multi-stage DuIOD.

## Method

### Overall Architecture

The DuET framework consists of two core components:

1. **DuET Module**: Updates shared parameters (backbone + neck) by dynamically fusing old and new task vectors layer by layer, balancing knowledge retention and new knowledge absorption.
2. **Incremental Head**: Concatenates detection head parameters across tasks to expand the model's categorical detection capacity.

The overall pipeline is as follows:

1. **Base Task $\mathcal{T}_1$**: Fine-tune a pretrained detector on the first task to obtain parameters $\theta_1$.
2. **Parameter Decomposition**: Decompose model parameters into shared parameters $\theta_s$ (backbone + neck) and task-specific parameters $\theta_\tau$ (detection heads).
3. **Incremental Task $\mathcal{T}_t, t \geq 2$**:
    - Sequential fine-tuning: Initialize from $\theta_{t-1}$ and train with the total loss to obtain $\theta_t$.
    - Compute old and current task vectors: $\tau_{\text{old}} = \theta_{s_{t-1}} - \theta_{s_0}$, $\tau_{\text{curr}} = \theta_{s_t} - \theta_{s_0}$.
    - Merge shared parameters via the DuET Module.
    - Concatenate detection head parameters via the Incremental Head.
4. **Inference**: Perform detection using the merged incremental weights.

### Key Design 1: DuET Module — Layer-wise Dynamic Task Vector Fusion

The DuET Module is the core of the framework. For each layer $l$, it computes a retention factor $\alpha_l$ and an adaptation factor $\beta_l$ to fuse old and new task vectors.

The p-factor is first computed to measure the relative importance of old versus current updates:

$$p_l = \frac{\|\tau_{\text{old}}^l\| - \|\tau_{\text{curr}}^l\|}{\|\tau_{\text{old}}^l + \tau_{\text{curr}}^l\| + \epsilon}$$

After $\tanh$ mapping and clamping, the dynamic coefficients are:

$$\alpha_l = \alpha_{\text{base}} + \text{clamp}(\gamma \cdot \tanh(p_l), -\gamma, \gamma), \quad \beta_l = 1 - \alpha_l$$

The merged shared parameters for each layer are then:

$$(\theta_{s_t}^l)_{\text{incre}} = \theta_{s_0}^l + \alpha_l \cdot \tau_{\text{old}}^l + \beta_l \cdot \tau_{\text{curr}}^l$$

**Design Intuition**: When the old task vector norm dominates in a given layer, $\alpha_l$ is larger, prioritizing the preservation of old knowledge (stability); otherwise, the model absorbs more new knowledge (plasticity). This avoids the high computational cost of second-order methods such as Fisher Merging.

### Key Design 2: Incremental Head — Task-Specific Parameter Concatenation

Detection head parameters are not merged but directly concatenated across current and historical task-specific parameters:

$$(\theta_{\tau_t})_{\text{incre}} = [\theta_{\tau_t}; (\theta_{\tau_{t-1}})_{\text{incre}}]$$

This allows the model to simultaneously output detection results for all learned categories and constitutes a simple yet effective incremental expansion strategy.

### Key Design 3: Detector Agnosticism

The parameter decomposition strategy of DuET is architecture-agnostic:
- **YOLO11**: backbone + neck as $\theta_s$, detection head as $\theta_\tau$.
- **RT-DETR**: Analogously applicable.
- **Deformable DETR**: Analogously applicable.

This enables real-time detectors such as YOLO11 and RT-DETR to serve as incremental detectors for the first time.

### Loss & Training

For the base task ($t=1$), only the standard detection loss $\mathcal{L}_{\text{Detector}}$ is used.

For incremental tasks ($t \geq 2$), the total loss is:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Detector}} + \lambda_{\text{Distill}} \mathcal{L}_{\text{Distill}}^* + \lambda_{\text{DC}} \mathcal{L}_{\text{DC}}$$

**Directional Consistency Loss $\mathcal{L}_{\text{DC}}$**:

$$\mathcal{L}_{\text{DC}} = \sum_{i \in \theta_s} \text{ReLU}\left[-\left((\tau_{s_t}^{(i)} - \tau_{s_{t-1}}^{(i)}) \cdot (\tau_{s_{t-1}}^{(i)} - \tau_{s_{t-2}}^{(i)})\right)\right]$$

The dot product measures the directional consistency of consecutive incremental updates: if the current update direction is opposite to the previous step (sign conflict), the ReLU term produces a penalty. This effectively reduces sign conflicts by approximately 34% during model merging.

**Modified Distillation Loss $\mathcal{L}_{\text{Distill}}^*$**: A standard distillation strategy to preserve knowledge of previous tasks.

### Evaluation Metric: Retention-Adaptability Index (RAI)

The authors propose a new evaluation metric that jointly measures retention and adaptability:

$$\text{RAI} = \frac{\text{Avg RI} + \text{Avg GI}}{2}$$

- **Avg RI (Average Retention Index)**: Ratio of the final model's mAP on old categories in old domains to the mAP at the time of initial learning; measures the degree of forgetting.
- **Avg GI (Average Generalization Index)**: Ratio of the model's mAP on unseen categories to a reference model; measures cross-domain generalization capability.

## Key Experimental Results

### Datasets

| Dataset Series | Domains | # Categories | Source |
|---|---|---|---|
| Pascal Series | VOC, Clipart, Watercolor, Comic | 3–20 | Pascal VOC, Cross-Domain Det |
| Diverse Weather Series | Daytime Sunny, Night Sunny, Daytime Foggy | 7 | BDD-100k, FoggyCityscapes, Adverse-Weather |

### Main Results: Two-Stage and Multi-Stage Results (Table 2)

| Method | Base Detector | Pascal 2-Stage RAI | Pascal 4-Stage RAI | Weather 2-Stage RAI | Weather 3-Stage RAI |
|---|---|---|---|---|---|
| Sequential FT | YOLO11n | 6.81% | 5.53% | 22.94% | 15.26% |
| LwF | YOLO11n | 53.19% | 34.84% | 38.88% | 25.86% |
| ERD | YOLO11n | 56.17% | 47.95% | 59.92% | 42.00% |
| LDB | ViTDet | 42.83% | 52.83% | 11.76% | 27.96% |
| CL-DETR | Def. DETR | 54.51% | 54.18% | 57.09% | 53.86% |
| **DuET** | **YOLO11n** | **65.99%** | **65.95%** | **72.51%** | **65.25%** |

**Key Findings**:
- DuET substantially outperforms all baselines across all experiments: +13.12% RAI on Pascal multi-stage and +11.39% RAI on Weather multi-stage.
- DuET maintains Avg RI at 87–89%, indicating minimal forgetting.
- DuET has only 2.58M parameters, far fewer than CL-DETR (39.85M) and LDB (110.52M).

### Cross-Detector Generalization (Table 3) — Weather Two-Stage

| Base Detector | # Params | GFLOPs | Avg RI | Avg GI | RAI |
|---|---|---|---|---|---|
| ViTDet | 110.52M | 1829.6 | 27.55% | 28.22% | 27.89% |
| Deformable DETR | 39.85M | 11.8 | 84.45% | 33.45% | 58.95% |
| RT-DETR-l | 32.00M | 103.4 | 47.73% | 21.00% | 34.37% |
| RT-DETR-x | 65.49M | 222.5 | 56.39% | 24.15% | 40.27% |
| **YOLO11n** | **2.58M** | **6.3** | **88.06%** | **56.95%** | **72.51%** |
| YOLO11x | 56.84M | 194.4 | 96.88% | 42.41% | 69.18% |

**Key Finding**: YOLO11n achieves the best RAI with the fewest parameters and lowest computational cost, demonstrating that DuET's task arithmetic strategy is highly compatible with lightweight detectors.

### Ablation Study (Table 4) — Pascal Two-Stage, YOLO11n

| Configuration | Avg RI | Avg GI | RAI |
|---|---|---|---|
| No Incremental (Static) | 0.5% | 9.13% | 4.82% |
| + Sequential Fine-tuning | 0.75% | 12.86% | 6.81% |
| + Incremental Head | 24.75% | 33.36% | 29.06% |
| + DuET Module | 75.00% | 37.26% | 56.13% |
| + $\mathcal{L}_{\text{Distill}}^*$ | 87.06% | 37.75% | 62.41% |
| + $\mathcal{L}_{\text{DC}}$ (Full DuET) | **87.44%** | **44.54%** | **65.99%** |

**Key Findings**:
- The Incremental Head contributes the largest single RAI jump (+22.25%).
- The DuET Module further substantially improves Avg RI (24.75% → 75.00%).
- $\mathcal{L}_{\text{DC}}$ particularly boosts Avg GI (+6.79%), effectively improving generalization.
- Every component is indispensable.

## Highlights & Insights

1. **Valuable Problem Formulation**: DuIOD is more realistic than pure CIOD or DIOD and constitutes an important new research direction.
2. **Novel Application of Task Arithmetic to Detection**: This work is the first to introduce Task Arithmetic into incremental object detection and validates its detector-agnostic applicability.
3. **Elegant Directional Consistency Loss**: Constraining the consistency of consecutive update directions via dot products to mitigate sign conflicts is simple and effective, reducing sign conflicts by 34% on average.
4. **Lightweight and Efficient**: YOLO11n with only 2.58M parameters and 6.3 GFLOPs can serve as a real-time incremental detector, offering strong practical utility.
5. **Well-Designed Evaluation Metric**: RAI jointly captures retention and generalization, providing a more comprehensive assessment than existing forgetting-only metrics.
6. **Parameter Efficient**: No exemplar buffer or generative replay is required; only task vectors and the shared parameter baseline need to be stored.

## Limitations & Future Work

1. **Category–Domain Binding Assumption**: Each incremental task pairs new categories with a new domain; more complex real-world combinations (new categories in an old domain, the same category across multiple domains, etc.) are not addressed.
2. **Requires Storing Baseline Weights $\theta_{s_0}$**: Task vector computation depends on the initial pretrained weights, and storage overhead grows linearly with the number of layers.
3. **Poor Performance on ViTDet**: RAI of only 27.89% suggests that DuET's layer-wise fusion strategy may not generalize to all architectures.
4. **Low Avg GI Overall**: Even for DuET, Avg GI peaks at only 56.95%, indicating substantial room for improvement in cross-domain generalization.
5. **Only 2–4 Stages Evaluated**: Performance on longer incremental sequences (10+ tasks) remains unknown.
6. **$\mathcal{L}_{\text{DC}}$ Requires at Least Three Tasks**: Directional consistency necessitates comparing updates across three consecutive steps, limiting its contribution in two-stage experiments.

## Related Work & Insights

- **Task Arithmetic [Ilharco et al., 2023]**: Modifies pretrained models via arithmetic operations on task vectors — DuET extends this paradigm to incremental detection.
- **TIES-Merging [Yadav et al., 2023]**: Resolves sign conflicts via orthogonal constraints — inspires DuET's $\mathcal{L}_{\text{DC}}$.
- **MagMax [Marczak et al.]**: Mitigates forgetting through important parameter selection — DuET's layer-wise p-factor serves as a more dynamic alternative.
- **CL-DETR [Liu et al., 2023]**: CIOD SOTA using knowledge distillation + exemplar replay — DuET surpasses it without any exemplars.
- **LDB [Chen et al., 2024]**: DIOD SOTA that learns domain biases — incapable of handling category changes in DuIOD.

**Insight**: The "vector space" perspective of Task Arithmetic offers a new lens for continual learning — treating the weight differences between new and old task models as directional vectors and balancing stability and plasticity through simple linear combinations, thereby avoiding complex regularization or replay strategies.

## Rating

| Dimension | Score (1–5) | Notes |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | First to define DuIOD + first to introduce Task Arithmetic into incremental detection |
| Technical Depth | ⭐⭐⭐⭐ | Layer-wise dynamic fusion + DC Loss + comprehensive evaluation framework |
| Experimental Thoroughness | ⭐⭐⭐⭐ | 7 experiments + 6 detectors + detailed ablation |
| Practical Value | ⭐⭐⭐⭐⭐ | Detector-agnostic + lightweight + exemplar-free; highly practical |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure with informative figures |
| Overall | ⭐⭐⭐⭐ | Strong problem formulation, concise and effective method, solid experiments |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ICCV 2025\] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)

<!-- RELATED:END -->
