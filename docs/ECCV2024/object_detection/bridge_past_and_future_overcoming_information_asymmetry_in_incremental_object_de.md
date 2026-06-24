---
title: >-
  [Paper Note] Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection
description: >-
  [ECCV2024][Object Detection][incremental object detection] This paper proposes the Bridge Past and Future (BPF) method, which bridges past stages via pseudo-labels, excludes potential future objects using an attention mechanism, and incorporates dual-teacher distillation (Distillation with Future) to resolve the optimization goal inconsistency caused by cross-stage information asymmetry in incremental object detection.
tags:
  - "ECCV2024"
  - "Object Detection"
  - "incremental object detection"
  - "knowledge distillation"
  - "information asymmetry"
  - "catastrophic forgetting"
  - "pseudo labeling"
date: 2026-05-08
content_hash: a67bcc93e791de81
---

# Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection

**Conference**: ECCV2024  
**arXiv**: [2407.11499](https://arxiv.org/abs/2407.11499)  
**Code**: [iSEE-Laboratory/BPF](https://github.com/iSEE-Laboratory/BPF)  
**Area**: Object Detection  
**Keywords**: incremental object detection, knowledge distillation, information asymmetry, catastrophic forgetting, pseudo labeling

## TL;DR

This paper proposes the Bridge Past and Future (BPF) method, which bridges past stages via pseudo-labels, excludes potential future objects using an attention mechanism, and incorporates dual-teacher distillation (Distillation with Future) to resolve the optimization goal inconsistency caused by cross-stage information asymmetry in incremental object detection.

## Background & Motivation

Incremental Object Detection (IOD) requires models to continuously learn new object classes while preserving the detection capability for old classes, without access to old data. Unlike incremental classification, object detection faces a unique challenge—objects from past, current, and future stages may co-occur in the same image, but only annotations for current classes are provided in the current stage.

This causes a severe **information asymmetry** problem:
- **Past to Current Asymmetry**: Old-class objects lack annotations in the current stage and are erroneously trained as background, exacerbating catastrophic forgetting.
- **Current to Future Asymmetry**: Future-class objects also lack annotations currently and are likewise treated as background. When the future stage arrives, the model must correct this erroneous perception, increasing the difficulty of learning new classes.

Existing methods (e.g., MMA, PPAS) primarily focus on preventing forgetting via strong regularization, but ignore the inconsistency of optimization goals caused by class co-occurrence across different stages, which limits the overall performance on both old and new classes.

## Core Problem

How to overcome cross-stage information asymmetry in incremental object detection to maintain a consistent optimization direction throughout the incremental learning process? Specifically, three sub-problems need to be solved:

1. How to recover the supervision signals of old-class objects in the current stage (compensating for the missing past information).
2. How to avoid forcibly classifying potential future objects as background (reserving space for future learning).
3. How to preserve old-class knowledge while promoting new-class learning during the distillation process (comprehensive distillation).

## Method

### Overall Architecture

BPF is built upon Faster R-CNN and comprises three core components: Bridge Past, Bridge Future, and Distillation with Future (DwF).

### 1. Bridge Past

The old model $\mathcal{M}_{t-1}$ is utilized as a **pseudo-label generator** to infer high-confidence predictions of old classes on current training images, thereby replenishing the missing old-class supervision signals:

- Select high-confidence predictions of old classes $\mathcal{C}_{1:t-1}$ from the old model (threshold $\eta=0.75$), followed by NMS de-duplication.
- Exclude predictions that highly overlap with current stage annotations (IoU threshold $\lambda_1=0.7$), ensuring a clear boundary between new and old classes.
- Merge the filtered pseudo-labels with current annotations to symmetrically train the current detector.

Unlike MMA which merges background and old-class probabilities, BPF explicitly separates old-class modeling from the background, retaining the discriminative capability between old classes and background.

### 2. Bridge Future

Identify regions in the background that may contain future-class objects using a feature map attention mechanism, and exclude them from negative samples:

- Compute the spatial attention map from the backbone feature map: $A_i = \text{Softmax}(\sum_{c=1}^{C}|F_i|^p)$
- Compute the attention score for each region proposal: $a_{i,j}^{roi} = \text{Avg}(\text{RoIPool}(A_i, r_j))$
- Regions with high attention scores and high objectness scores are treated as potential future objects and are excluded from the negative samples used for training the RoI Head.

This ensures that salient object regions are not hard-coded as background in the current stage, reserving space for learning in future stages.

### 3. Distillation with Future (DwF)

Introduce a **dual-teacher distillation** architecture, using the old model $\mathcal{M}_{t-1}$ and an intermediate model $\mathcal{M}_t^{im}$ fully supervised on current data as two teachers:

- **Region Division**: Divide the distillation regions into $\mathcal{R}_1$ (old-class regions) and $\mathcal{R}_2$ (new-class regions) based on the IoU between proposals and current annotations (threshold $\lambda_2=0.5$).
- **$\mathcal{R}_1$ (Old-class regions)**: Use the old model as the primary teacher, and utilize the intermediate model to refine its background probability into fine-grained probabilities of current classes.
- **$\mathcal{R}_2$ (New-class regions)**: Use the intermediate model as the primary teacher, and utilize the old model to supplement old-class knowledge into its background probability.

The core idea is to leverage the rich information contained in the background probabilities—the background probability of the old model contains current-class information, and that of the intermediate model contains old-class information—to achieve class-wise distillation via probability reorganization.

## Key Experimental Results

### PASCAL VOC 2007 (Single-step Incremental, mAP@0.5)

| Setting | BPF (1-20) | MMA (1-20) | Gain |
|------|-----------|-----------|------|
| 19-1 | **74.1** | 70.7 | +3.4 |
| 15-5 | **72.7** | 69.9 | +2.8 |
| 10-10 | **72.9** | 66.6 | +6.3 |
| 5-15 | **73.0** | 59.6 | +13.4 |

### PASCAL VOC 2007 (Multi-step Incremental, mAP@0.5)

| Setting | BPF (1-20) | MMA (1-20) |
|------|-----------|-----------|
| 10-5 (3 tasks) | **68.7** | 64.2 |
| 5-5 (4 tasks) | **62.5** | 38.9 |
| 15-1 (6 tasks) | **66.9** | 64.1 |

### MS COCO 2017

| Setting | BPF AP | MMA AP |
|------|--------|--------|
| 40-40 | 34.4 | 33.0 |
| 70-10 | **36.2** | 30.2 |

Under rehearsal-free conditions, BPF outperforms methods including those that use exemplar replay (e.g., ABR) across almost all settings, with a particularly remarkable gain of 13.4 percentage points over MMA in the 5-15 setting.

### Ablation Study

- Bridge Past: Crucial for preserving old classes; explicit pseudo-labels are significantly better than MMA's background merging scheme.
- Bridge Future: Consistently brings a 0.3-0.6 mAP gain across all settings with minimal cost.
- DwF Distillation: Region-adaptive division with $\lambda_2=0.5$ outperforms using the old model alone ($\lambda_2=1.0$), validating the effectiveness of dual-teacher complementarity.

## Highlights & Insights

1. **Profound Problem Analysis**: For the first time, the information asymmetry across the three stages of past-current-future in IOD is systematically analyzed, revealing that inconsistent optimization goals are the root cause of the performance bottleneck.
2. **Exquisite Bidirectional Bridging Design**: Using pseudo-labels for the past and attention exclusion for the future, the solutions in both directions are intuitively complementary and elegantly implemented.
3. **Theoretical Elegance of DwF Distillation**: Leveraging the decomposability of background probabilities, class-wise distillation is realized through probability reorganization, which is much finer than simply merging backgrounds.
4. **Outperforming Replay Methods without Memory Replay**: BPF surpasses exemplar-storing methods such as ABR under strict rehearsal-free settings, offering high practical value.

## Limitations & Future Work

1. **Only Validated on Faster R-CNN**: Although the paper mentions scalability to transformer-based detectors (DETR series), experimental validation is not provided.
2. **Pseudo-label Quality Depends on the Old Model**: If the old model has poor performance or generalizes poorly on current images, pseudo-labels may introduce noise.
3. **Attention Mechanism in Bridge Future Cannot Distinguish Categories**: Relying solely on feature saliency judgment may erroneously exclude hard samples of current categories.
4. **Extra Training Overhead from Intermediate Model**: DwF requires training an additional intermediate model solely on current data, which increases training time.
5. **Insignificant Improvement in COCO 40-40 Setting**: The advantage is less pronounced in equal-partition scenarios of large-scale datasets.

## Related Work & Insights

| Method | Core Idea | Difference from BPF |
|------|---------|--------------|
| MMA | Merges background and old classes into a single entity for UKD distillation | BPF explicitly separates old classes and background, retaining discriminative capability |
| PPAS | Pseudo-positive-aware sampling to avoid putting old classes into background | BPF considers both past and future, and introduces an additional distillation mechanism |
| ABR | Uses exemplar replay (exemplar-based) | BPF requires no storage of old exemplars, better aligning with privacy and storage constraints |
| PseudoRM | Pseudo labeling + memory replay | BPF requires no replay, and adds Bridge Future and DwF distillation |

## Insights & Connections

- **The analytical perspective of "information asymmetry" is highly valuable**: Many problems in incremental learning can be attributed to cross-stage missing information, and this analytical framework can be extended to tasks like incremental segmentation and incremental instance segmentation.
- **The decomposability of background probability is a clever insight**: In incremental learning, the background class is essentially a "trash-can" class that contains substantial useful information, which can be appropriately decomposed and utilized.
- **Dual-teacher complementary distillation can be extended to other scenarios**: Such as utilizing expert teachers of different modalities in multi-modal fusion, or model complementarity across different clients in federated learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — The information asymmetry perspective and bidirectional bridging design are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluations across various VOC/COCO settings, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Achieves SOTA without memory replay, offering strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Incremental Object Detection via Future-Aware Decoupled Cross-Head Distillation](../../CVPR2026/object_detection/incremental_object_detection_via_future-aware_decoupled_cross-head_distillation.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2026\] Parameterized Prompt for Incremental Object Detection](../../CVPR2026/object_detection/parameterized_prompt_for_incremental_object_detection.md)
- [\[CVPR 2026\] Towards an Incremental Unified Multimodal Anomaly Detection: Augmenting Multimodal Denoising From an Information Bottleneck Perspective](../../CVPR2026/object_detection/towards_an_incremental_unified_multimodal_anomaly_detection_augmenting_multimoda.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](../../CVPR2026/object_detection/beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)

</div>

<!-- RELATED:END -->
