---
title: >-
  [Paper Note] Rectify the Regression Bias in Long-Tailed Object Detection
description: >-
  [ECCV 2024][Object Detection][Long-Tailed Object Detection] This work first reveals and systematically addresses the overlooked **regression bias** problem in long-tailed object detection. Due to insufficient samples, the parameters of class-specific regression heads for rare categories suffer from poor generalization. By incorporating an additional class-agnostic regression branch for trade-off, this method achieves state-of-the-art performance on datasets such as LVIS.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Long-Tailed Object Detection"
  - "Regression Bias"
  - "Class-Agnostic"
  - "LVIS"
  - "Bounding Box Regression"
date: 2026-05-08
content_hash: b90ee1eb9f49d0cb
---

# Rectify the Regression Bias in Long-Tailed Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2401.15885](https://arxiv.org/abs/2401.15885)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Long-Tailed Object Detection, Regression Bias, Class-Agnostic, LVIS, Bounding Box Regression

## TL;DR

This work first reveals and systematically addresses the overlooked **regression bias** problem in long-tailed object detection. Due to insufficient samples, the parameters of class-specific regression heads for rare categories suffer from poor generalization. By incorporating an additional class-agnostic regression branch for trade-off, this method achieves state-of-the-art performance on datasets such as LVIS.

## Background & Motivation

Long-tailed object detection suffers from extreme class distribution imbalance. Existing methods (e.g., EQLv2, SeeSaw, ECM) almost exclusively focus on alleviating **classification bias** via strategies like re-weighting, over-sampling, and balanced grouping to prevent tail categories from being suppressed. However, apart from the classification branch, the **regression branch** is equally crucial in the detection pipeline, yet it has received little attention.

The authors discovered a key phenomenon through experiments:
- The loss of the **RCNN regression branch** exhibits severe category-wise discrepancy, where the regression loss for rare categories is significantly higher than that for frequent ones, indicating poor regression quality.
- Conversely, the loss of the **RPN regression branch** is almost balanced (with similar losses across all classes).

The key difference between the two is that the RPN regression head is **class-agnostic** (all classes share the same set of parameters), whereas the RCNN regression head is **class-specific** (each class has its own parameters). When using class-specific regression heads, rare categories fail to learn robust regression parameters due to extremely scarce training samples and challenges such as scale shift.

Furthermore, the authors found that the **object scale shift** between the training and validation sets is much larger for rare categories than for frequent ones, further exacerbating the difficulty of regression.

A key verification experiment: after replacing the classifier with ground-truth (GT) labels, the $AP_r$ of rare categories with a class-agnostic regression head dramatically increases from 0.7 to 54.6, even exceeding the 40.0 of frequent categories! This strongly demonstrates that rare categories indeed require **class-agnostic regression**.

## Method

### Overall Architecture

In the standard Faster R-CNN two-stage detection framework, the regression branch of the RCNN head maintains an independent linear layer $W_i$ for each category $i$ to predict bounding box offsets. The authors propose three schemes to mitigate regression bias, with the core idea being the introduction of parameter sharing among classes to varying degrees.

### Key Designs

1. **Class-Agnostic Branch (CAB)**: The most concise and effective scheme. An additional shared regression head $W_0$ is added for all categories. The final regression head for each class becomes a weighted combination of both:

    $W_i' = \alpha W_0 + (1-\alpha) W_i$
   
   where $\alpha$ is a trade-off hyperparameter. Rare classes gain better generalization ability from the shared head $W_0$, while frequent classes can still benefit from the class-specific head $W_i$. Experiments show that $\alpha=0.5$ achieves optimal performance. This scheme does not require any dataset statistics and simultaneously leverages both class-agnostic prior knowledge and class-specific fine-grained knowledge.

2. **Clustering Heads**: Based on the observation in Fig. 1(c) that object scale statistics of different categories show similarities. $C$ categories are sorted by instance counts or average scales and then partitioned into $K$ groups, where each group shares a single regression head:

    - Sorting: Sort $W_1, \dots, W_C$ by the number of instances or scale.
    - Grouping: Neighboring classes are partitioned into the same group, with $N = C/K$ classes per group.
    - Replacement: Classes within the same group share the same regression matrix $W_{g_i}$.
   
   Using scale statistic clustering ($k=100$) yields better results, with APr increasing from 13.4 to 16.7.

3. **Merging Heads**: The most straightforward scheme, which merges regression heads based on the predefined LVIS rare/common/frequent splits. For example, allowing all rare classes to share a single $W_{rare}$. Interestingly, merging only the common classes yields the largest gain in APr (from 14.2 to 17.7). The authors hypothesize that this is related to the category distribution shift between the training and validation sets.

### Loss & Training

- Adopts the same default training strategies as each baseline, only modifying the structure of the regression head.
- Employs FP16 mixed-precision training and a warmup strategy.
- Implemented within the MMDetection framework.
- All experiments were run 3 times on 8 RTX 3090 GPUs, and the results were averaged to reduce variance.

## Key Experimental Results

### Main Results

**Comparison of the three variants (Baseline: CE + Mask-RCNN R50-FPN, LVIS1.0)**

| Method | AP | APr | APb | APrb |
|------|-----|------|------|------|
| Baseline (class-specific) | 23.7 | 14.2 | 24.7 | 13.4 |
| + CAB ($\alpha=0.5$) | **25.1** | **17.5** | **27.0** | **18.0** |
| Merging (c) | 25.5 | 17.7 | 27.2 | 17.2 |
| Clustering ($k=100$, scale) | 25.2 | 16.7 | 26.9 | 16.7 |

**Combining CAB with existing long-tailed methods (LVIS1.0, Mask-RCNN R50-FPN)**

| Method | +CAB | APb | APrb | AP | APr |
|------|------|------|------|-----|------|
| RFS  | ✗ | 24.7 | 13.4 | 23.7 | 14.2 |
| RFS  | ✓ | 27.0 (+2.3) | 18.0 (+4.6) | 25.1 | 17.5 |
| EQLv2 | ✗ | 26.0 | 16.1 | 25.2 | 17.4 |
| EQLv2 | ✓ | 28.1 (+2.1) | 20.4 (+4.3) | 26.0 | 19.5 |
| SeeSaw | ✗ | 27.3 | 18.2 | 26.9 | 19.6 |
| SeeSaw | ✓ | 28.9 (+1.6) | 19.9 (+1.7) | 27.7 | 20.2 |
| ECM | ✗ | 27.7 | 17.7 | 27.2 | 19.6 |
| ECM | ✓ | 29.1 (+1.4) | 18.4 (+0.7) | 27.8 | 19.1 |

**Comparison with SOTA (SeeSaw + CAB = "Ours")**

| Architecture | Backbone | Method | AP | APb |
|------|----------|------|-----|------|
| Mask-RCNN | R50 | ECM | 27.2 | 27.7 |
| Mask-RCNN | R50 | **Ours** | **27.7** | **28.9** |
| Mask-RCNN | R101 | ECM | 28.6 | 29.3 |
| Mask-RCNN | R101 | **Ours** | **29.0** | **30.7** |
| Cascade R-CNN | Swin-T | **Ours** | **34.6** | **38.2** |
| Cascade R-CNN | Swin-B | **Ours** | **39.9** | **44.2** |

### Ablation Study

| $\alpha$ Value | AP | APr | APb | APrb |
|-----|-----|------|------|------|
| 0.0 (baseline) | 23.7 | 14.2 | 24.7 | 13.4 |
| 0.2 | 24.1 | 15.8 | 25.4 | 15.1 |
| **0.5** | **25.1** | **17.5** | **27.0** | **18.0** |
| 0.8 | 24.4 | 17.0 | 25.9 | 16.4 |
| 1.0 (Pure agnostic) | 24.7 | 16.7 | 26.7 | 18.3 |

### Key Findings

- CAB brings consistent and significant improvements to all existing long-tailed detection methods, especially for rare classes.
- "RFS+CAB" (using CE loss) achieves detection accuracy almost comparable to the SeeSaw method, demonstrating that regression improvement is a powerful complement to classification improvement.
- The method is equally effective on both COCO-LT (artificial long-tailed) and COCO2017 (relatively balanced) datasets.
- The advantages are even more pronounced under stricter evaluation metrics (APboundary, APfixed_bbox).
- The method can be directly generalized to the mask branch design.

## Highlights & Insights

- **Novel Problem Formulation**: This work first identifies and systematically studies the regression bias in long-tailed object detection, whereas all previous methods focused on classification bias.
- **Clear and Intuitive Idea**: The hypothesis is naturally derived by comparing the regression losses of RPN (agnostic) versus RCNN (specific).
- **GT Experiments**: The experimental design of replacing the classifier with ground-truth (GT) labels in Table 1 is elegant, cleanly decoupling the impact of classification and regression.
- **Extremely Simple Method**: The core modification consists of only a single line of formula (weighted combination), which is plug-and-play.
- **Strong Generalization**: The method is effective across different datasets, metrics, and branches (e.g., mask).

## Limitations & Future Work

- The optimal value of $\alpha$ may vary depending on the dataset/class distribution, and is currently fixed at 0.5.
- The merging scheme (Table 2c) even outperforms CAB under certain metrics, but the authors chose the simpler CAB and did not further explore adaptive merging.
- The grouping strategy of the clustering scheme is relatively simple (equal division); more fine-grained clustering might yield further improvements.
- The paper does not discuss whether regression bias likewise exists in anchor-free detectors (e.g., FCOS).

## Related Work & Insights

- Complementary to the ideas of BaGS and the EQL series: while the former addresses classification bias, this work addresses regression bias, and the two can be superimposed.
- Similar ideas can be extended to other tasks involving long-tailed distributions (e.g., pose estimation, keypoint detection).
- Inspired by [37] but drawing opposite conclusions: [37] argued that performance degradation is mainly caused by classification, neglecting the impact of regression.

## Rating

- Novely: ⭐⭐⭐⭐☆ (Novel problem identification, relatively simple method)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple methods, architectures, backbones, datasets, and metrics)
- Writing Quality: ⭐⭐⭐⭐☆ (Clear presentation, rich figures and tables)
- Value: ⭐⭐⭐⭐☆ (A plug-and-play improvement technique, highly inspiring for the field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](../../CVPR2025/object_detection/simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[ECCV 2024\] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
