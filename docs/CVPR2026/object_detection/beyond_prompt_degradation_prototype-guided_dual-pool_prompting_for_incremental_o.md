---
title: >-
  [Paper Note] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection
description: >-
  [CVPR 2026][Object Detection][Incremental Object Detection] This paper proposes the PDP framework, which addresses prompt degradation in incremental object detection caused by prompt coupling and prompt drift via decoupl…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Incremental Object Detection"
  - "Prompt Learning"
  - "Dual-Pool Paradigm"
  - "Prototype Pseudo-Labels"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: efb75e82c8535735
---

# Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.02286](https://arxiv.org/abs/2603.02286)
**Code**: [Available](https://github.com/zyt95579/PDP_IOD/tree/main)
**Area**: Object Detection
**Keywords**: Incremental Object Detection, Prompt Learning, Dual-Pool Paradigm, Prototype Pseudo-Labels, Catastrophic Forgetting

## TL;DR

This paper proposes the PDP framework, which addresses prompt degradation in incremental object detection caused by prompt coupling and prompt drift via decoupled dual-pool prompting (shared pool + private pool) and Prototypical Pseudo-Label Generation (PPG), achieving state-of-the-art performance on COCO and VOC.

## Background & Motivation

Incremental Object Detection (IOD) requires a model to continuously learn new categories without access to old data while retaining detection performance on previously learned categories. Prompt-based methods have attracted attention for their replay-free, parameter-efficient nature, but suffer from two core issues:

**Prompt Coupling**: Existing methods adopt a single-pool paradigm that stores task-generic and task-specific prompts in the same pool, causing competition and interference within a limited parameter space.

**Prompt Drift**: Under the IOD setting, foreground objects of old classes are annotated as "background" in subsequent tasks. This supervisory inconsistency forces already-optimized prompts to drift toward incorrect semantic directions.

Existing pseudo-label methods rely on fixed confidence thresholds and cannot adapt to inter-class distribution differences, further exacerbating drift.

## Method

### Overall Architecture

PDP is built upon Deformable-DETR and adopts a teacher–student distillation architecture. Its core consists of two modules:
- **Decoupled Dual-Pool Prompting (DDP)**: a decoupled dual-pool prompting mechanism
- **Prototypical Pseudo-Label Generation (PPG)**: prototype-guided pseudo-label generation

### Key Designs

#### 1. Decoupled Dual-Pool Prompting (DDP)

**Shared Pool**:
- Contains learnable prompts $P_s \in \mathbb{R}^{N_s \times L_p \times D}$, key vectors $K_s$, and a query adapter $A_s$
- Shared across all tasks and continuously optimized to capture general visual knowledge
- Facilitates stable forward knowledge transfer

**Private Pool**:
- Each task is assigned independent prompt parameters $(P_p^t, K_p^t, A_p^t)$
- Only the current task's parameters are updated during training; parameters of previous tasks are frozen
- The private pool size $N_p$ is dynamically adjusted according to the number of new classes

**Prompt Retrieval and Integration**:
- A query extractor produces query vectors, which are then modulated via Hadamard product with the query adapter
- Cosine similarity with key vectors from both pools is computed to aggregate prompts via weighted summation
- Retrieved prompts $P_r$ are injected into the Transformer decoder via Prefix-Tuning

**Inter-Pool Diversity Constraint**: A directional decoupling loss maximizes angular separation between the two pool vectors:

$$\mathcal{L}_{DDL} = \lambda_{ddl} \cdot \frac{2}{|N_s||N_p|} \sum_{i,j} \max(0, \theta_{ddl} - \theta_{i,j})$$

where $\theta_{ddl} = 90°$, ensuring that shared and private prompts learn complementary, orthogonal representations.

#### 2. Prototypical Pseudo-Label Generation (PPG)

**Class Prototype Space Construction**:
- Query embeddings $f_i$ of correctly classified instances are extracted from the last decoder layer
- These are stored in a class-specific memory bank $F_c$, and class prototypes are computed as $p_c = \frac{1}{|F_c|}\sum_{f_i \in F_c} f_i$
- Prototypes are updated only during the last epoch of each task to ensure sufficient feature convergence

**Hierarchical Verification Mechanism**:
- **Easy samples**: Detections with confidence $s_i > \tau_h$ (0.5) are directly used as reliable pseudo-labels
- **Hard samples**: For intermediate confidence $\tau_l < s_i < \tau_h$ (0.2 < $s_i$ < 0.5), feature similarity to the class prototype is computed; detections exceeding the threshold are retained
- Both categories are merged to form the high-quality pseudo-label set $Y_{ppg}$

### Loss & Training

The total loss comprises the DETR detection loss, query regularization, directional decoupling loss, and distillation loss:

$$\mathcal{L} = \mathcal{L}_{DETR} + \mathcal{L}_Q + \mathcal{L}_{DDL} + \mathcal{L}_{DKD}(Y_{ppg})$$

- $\lambda_{ddl} = 0.15$, $\lambda_Q = 0.1$
- Shared pool: 100 prompts; private pool size equals the total number of dataset categories
- Confidence thresholds: $\tau_h = 0.5$, $\tau_l = 0.2$; prototype similarity threshold: $\theta_s = 0.5$

## Key Experimental Results

### Main Results

**Table 1: MS-COCO Multi-Step Incremental Setting (4 Tasks)**

| Method | Task4 mAP@P | Task4 mAP@C | Task4 mAP@A |
|--------|-------------|-------------|-------------|
| MD-DETR | 51.5 | 52.7 | 50.2 |
| OWOBJ | 49.4 | 38.8 | 43.9 |
| **PDP (Ours)** | **61.3 (+9.8)** | **55.8 (+3.1)** | **59.4 (+9.2)** |

**Table 3: PASCAL VOC Three Incremental Settings**

| Method | 10+10 mAP@A | 15+5 mAP@A | 19+1 mAP@A |
|--------|-------------|------------|------------|
| MD-DETR | 73.2 | 76.7 | 76.1 |
| RGR | 75.8 | 73.4 | 75.4 |
| **PDP (Ours)** | **78.7 (+2.9)** | **78.0 (+1.3)** | **79.4 (+3.3)** |

### Ablation Study

**Module Contributions (Table 4, COCO Task4 mAP@A)**:

| Configuration | mAP@P | mAP@A |
|---------------|-------|-------|
| Private Pool only (PP) | 46.0 | 46.0 |
| PP + SP + DDL | 56.9 | 55.1 |
| PP + PPG | 59.9 | 58.3 |
| PP + SP + PPG + DDL (Full) | **61.3** | **59.4** |

PPG improves old knowledge retention mAP@P by +13.9% while boosting new class adaptation mAP@C by +2.7%.

### Key Findings

1. PPG remains stable across three prototype similarity thresholds (0.5/0.6/0.7), indicating that hard samples naturally exhibit high similarity to their class prototypes.
2. Shared pool $N_s=100$ and private pool $N_p=80$ constitute the optimal configuration; an oversized shared pool (160) introduces redundancy and degrades performance.
3. PDP achieves mAP@P of 70.1% on the PASCAL VOC 19+1 setting, with qualitative visualizations confirming accurate detection of previously learned objects.

## Highlights & Insights

1. **Precise Problem Formulation**: This work is the first to decompose prompt degradation into two independently addressable sub-problems: coupling and drift.
2. **Prototype Space as an Alternative to Confidence Thresholds**: Similarity matching against class prototypes in the embedding space avoids the failure modes of fixed thresholds under inconsistent inter-class distributions.
3. **End-to-End Framework**: Unlike PseDet, which requires additional inference and clustering steps, PDP is fully end-to-end.

## Limitations & Future Work

1. PDP is slightly outperformed by PseDet in the 70+10 two-step setting (42.9 vs. 44.7 AP), leaving room for improvement under large-step incremental scenarios.
2. Prototypes are updated only in the last epoch of each task, so early-stage prototypes may be insufficiently accurate.
3. The private pool grows linearly with the number of tasks; parameter management warrants attention in long task sequences.
4. Validation is limited to DETR-based architectures; applicability to single-stage detectors such as YOLO remains unexplored.

## Related Work & Insights

- **MD-DETR**: The baseline architecture for PDP, which uses a single memory bank with task-ID-based prompt isolation.
- **DualPrompt**: Distinguishes between General and Expert Prompts but still manages them within a single pool.
- **PseDet**: Employs k-means adaptive thresholding for pseudo-label generation; not end-to-end.
- The dual-pool design philosophy is transferable to continual learning tasks such as incremental segmentation and incremental instance segmentation.

## Rating

- **Novelty**: ★★★★☆ — The combination of dual-pool decoupling and prototype-guided pseudo-labels is original.
- **Technical Depth**: ★★★★☆ — The method design is complete, with clearly motivated components.
- **Experimental Thoroughness**: ★★★★★ — Multiple datasets, multiple settings, and detailed ablations.
- **Writing Quality**: ★★★★☆ — Figures are clear; the problem–solution correspondence is explicit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)

</div>

<!-- RELATED:END -->
