---
title: >-
  [Paper Note] On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines
description: >-
  [ECCV 2024 (Oral)][Object Detection][Object Detection Calibration] This paper systematically reveals significant flaws in existing evaluation frameworks, evaluation metrics, and the use of Temperature Scaling in object detector calibration research. It proposes a principled joint evaluation framework along with post-hoc calibration methods tailored specifically for object detection (Platt Scaling and Isotonic Regression), demonstrating that correctly designed and evaluated po…
tags:
  - "ECCV 2024 (Oral)"
  - "Object Detection"
  - "Object Detection Calibration"
  - "Post-hoc Calibration"
  - "D-ECE"
  - "Platt Scaling"
  - "Isotonic Regression"
date: 2026-05-08
content_hash: 28c1b1e3da4a14dd
---

# On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

**Conference**: ECCV 2024 (Oral)  
**arXiv**: [2405.20459](https://arxiv.org/abs/2405.20459)  
**Code**: [https://github.com/fiveai/detection_calibration](https://github.com/fiveai/detection_calibration)  
**Area**: Object Detection / Model Calibration  
**Keywords**: Object Detection Calibration, Post-hoc Calibration, D-ECE, Platt Scaling, Isotonic Regression

## TL;DR

This paper systematically reveals significant flaws in existing evaluation frameworks, evaluation metrics, and the use of Temperature Scaling in object detector calibration research. It proposes a principled joint evaluation framework along with post-hoc calibration methods tailored specifically for object detection (Platt Scaling and Isotonic Regression), demonstrating that correctly designed and evaluated post-hoc calibrators far outperform recent train-time calibration methods.

## Background & Motivation

**Background**: The reliable deployment of object detectors requires the confidence scores output by the model to be calibrated, meaning that the predicted probability should accurately reflect the actual correctness. This is particularly crucial for safety-critical applications such as autonomous driving and medical imaging. In recent years, researchers have primarily explored detector calibration from two directions: (1) designing new training loss functions to train calibrated detectors from scratch (e.g., Cal-DETR); (2) using post-hoc Temperature Scaling (TS) to adjust the output probabilities of already trained detectors.

**Limitations of Prior Work**: Through extensive analysis, the authors find that current calibration research suffers from several severe issues: (1) **Unsound evaluation framework**—the existing frameworks measure calibration error without fully considering the task-specific characteristics of object detection, such as NMS post-processing and localization quality; (2) **Flawed evaluation metrics**—the commonly used Detection Expected Calibration Error (D-ECE) exhibits systematic bias under realistic operating conditions, which can lead to erroneous conclusions; (3) **Improper use of Temperature Scaling**—directly transferring TS from classification tasks to object detection, while neglecting unique aspects of detection such as foreground/background imbalance and multi-threshold operations.

**Key Challenge**: These flaws have led to a widespread but incorrect conclusion: train-time calibration methods outperform post-hoc calibration methods. In fact, once correctly designed and evaluated, post-hoc methods far exceed train-time methods in performance with extremely low computational cost.

**Goal**: (1) Identify and correct the flaws in existing evaluation frameworks and metrics; (2) Propose post-hoc calibration methods suitable for object detection; (3) Establish a fair benchmark to compare train-time and post-hoc calibration methods.

**Key Insight**: Starting from evaluation methodology, this work first corrects the issues in evaluation, and then proposes simple yet effective post-hoc calibration baselines under a fair evaluation framework.

**Core Idea**: Once the evaluation flaws in object detection calibration research are corrected, extremely cheap post-hoc calibration methods (Isotonic Regression) perform significantly better than complex train-time methods.

## Method

### Overall Architecture

This work operates on two levels: (1) Evaluation level—proposing a principled joint evaluation framework to simultaneously measure both the calibration and accuracy of detectors; (2) Method level—adapting classic post-hoc calibration methods (Platt Scaling and Isotonic Regression) to the object detection task. The overall pipeline is: using a trained detector to generate detection results $\rightarrow$ fitting a post-hoc calibrator on the validation set $\rightarrow$ evaluating the calibrated detection results on the test set.

### Key Designs

1. **Principled Joint Evaluation Framework**:

    - **Function**: Fairly and simultaneously measure both the accuracy and calibration of a detector
    - **Mechanism**: Employs Localization-Recall-Precision (LRP) Error as the primary accuracy metric, combined with Localization-aware ECE (LaECE) as the primary calibration error metric. LRP Error comprehensively considers three dimensions (localization, recall, and precision), which more completely reflects detector performance. LaECE accounts for the localization quality (IoU) of detection boxes when computing calibration errors, avoiding the flaw of D-ECE where localization factors are ignored. Additionally, the validation set is divided into minival and minitest, where the former is used to fit the calibrator and the latter is used for evaluation, preventing data leakage.
    - **Design Motivation**: Traditional evaluation only uses a combination of AP and D-ECE, lacking an intrinsic connection between the two metrics, and D-ECE is biased in detection scenarios. LRP and LaECE can more accurately reflect the calibration quality of the detector in real-world deployment.

2. **Platt Scaling for Object Detection**:

    - **Function**: Calibrates the confidence outputs of the detector by learning a linear transformation
    - **Mechanism**: For a detector's confidence score $s$, parameters $a$ and $b$ are learned to map it to a calibrated probability via $\sigma(a \cdot s + b)$ (where $\sigma$ is the sigmoid function). Adaptive modifications are made for object detection characteristics: (1) using the IoU threshold of matched detection boxes and ground truths as the boundary for positive/negative sample classification; (2) learning calibration parameters independently for each class; (3) performing calibration on detection results after NMS to align with actual deployment scenarios.
    - **Design Motivation**: Platt Scaling is a classic post-hoc calibration method for classification tasks, but directly applying it to detection tasks requires addressing detection-specific issues such as localization matching, multi-class handling, and NMS.

3. **Isotonic Regression for Object Detection**:

    - **Function**: Calibrates the confidence outputs of the detector via non-parametric monotonic transformation
    - **Mechanism**: Isotonic Regression learns a non-decreasing piecewise constant function that maps original confidence scores to calibrated probabilities. Compared to the linear assumption of Platt Scaling, it can fit more complex calibration curves. It is similarly adapted for the detection task: generating calibration training data based on IoU matching results, fitting independently per class, and operating on detection results after NMS. Additionally, threshold optimization is introduced to simultaneously improve both calibration and detection accuracy.
    - **Design Motivation**: The mapping between a detector's confidence and actual accuracy can be highly non-linear, and the non-parametric nature of Isotonic Regression is better suited for capturing such complex relationships.

### Loss & Training

Post-hoc methods do not involve training the detector itself, only fitting calibration parameters on the validation set. Platt Scaling optimizes parameters $a, b$ by maximizing log-likelihood; Isotonic Regression finds the optimal monotonic mapping by minimizing weighted least squares. Both are highly efficient, with the fitting process taking only a few seconds. The authors also propose an improved version of the LaECE metric (LaECE-v2), which modifies the computation of localization quality weights on top of the original LaECE.

## Key Experimental Results

### Main Results

Comparison of calibration results on the COCO dataset using the D-DETR detector:

| Method | Type | D-ECE↓ | LaECE↓ | LRP Error↓ | AP↑ |
|------|------|--------|--------|-----------|-----|
| D-DETR (Uncalibrated) | Baseline | High | High | Baseline | Baseline |
| Cal-DETR | Train-time | Medium | Medium | Slightly Decreased | Slightly Decreased |
| TS (Temperature Scaling) | Post-hoc | Unstable | Unstable | Degraded | Degraded |
| Platt Scaling (Ours) | Post-hoc | Significantly Reduced | Significantly Reduced | Improved | Maintained |
| Isotonic Regression (Ours) | Post-hoc | **Best (>7↓ vs Cal-DETR)** | **Best** | **Best** | **Maintained** |

A consistent trend is also observed on the Cityscapes and LVIS datasets, with Isotonic Regression performing the best across all datasets and detectors.

### Ablation Study

| Configuration | D-ECE | LaECE | Description |
|------|-------|-------|------|
| Fitting without class division | Medium | Medium | Ignores calibration differences between classes |
| Fitting independently per class | Better | Better | Calibrates parameters for each class separately |
| Calibration before NMS | Worse | Worse | Mismatches the actual deployment scenario |
| Calibration after NMS | **Best** | **Best** | Aligns with the actual deployment workflow |
| Without threshold optimization | Medium | Medium | Only performs probability calibration |
| With threshold optimization | **Best** | **Best** | Simultaneously optimizes both calibration and accuracy |

### Key Findings

- Under correct evaluation, post-hoc calibrators (especially Isotonic Regression) far outperform the train-time calibration method Cal-DETR, with a D-ECE gap of over 7 points.
- Temperature Scaling exhibits unstable performance in object detection, partly due to the extreme imbalance of foreground and background distributions in detection tasks.
- The existing evaluation frameworks led to the erroneous conclusion that "train-time methods are superior"—this conclusion is completely reversed after correcting the evaluation.
- Post-hoc calibration methods incur almost negligible computational costs, whereas train-time methods require completely retraining the detector.

## Highlights & Insights

- **Methodological contribution outweighs technical innovation**: The most valuable aspect is revealing the flaws in evaluation methods across the entire field, which has a greater impact than proposing a new algorithm.
- **A victory for simple methods**: Isotonic Regression is a classic method from the last century, yet after adaptation, it easily beats all complex train-time methods.
- **Highly practical**: Post-hoc calibrators can be attached as a plug-and-play component to any pre-trained detector, requiring no retraining.
- **ECCV Oral Paper**: Indicates that the reviewers recognized the high importance of this "course-correcting" type of work.

## Limitations & Future Work

- Post-hoc methods require a calibration validation set, which may be a limitation in data-scarce scenarios.
- Presently validated only under COCO-style evaluations, leaving more application scenarios (such as online learning and domain adaptation) to be explored.
- Only class-level calibration was explored, while instance-level calibration (considering object size, occlusion, etc.) remains a potential direction.
- The LaECE metric itself may still have room for further improvement.

## Related Work & Insights

- **Cal-DETR** (CVPR 2024): A train-time calibration method, which is significantly outperformed by the post-hoc method proposed in this paper.
- **Temperature Scaling** (ICML 2017): A classic post-hoc calibration method, but not directly applicable to detection tasks.
- **LRP Error** (TPAMI 2022): A metric for comprehensively evaluating detector performance.
- Insight: Before pursuing complex methods, first ensure the correctness of the evaluation framework—flawed evaluation can mislead the entire research direction.

## Rating

- Novelty: ⭐⭐⭐⭐ (profound insights at the methodological level, rather than just technical innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multiple datasets, multiple detectors, and exhaustive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (thorough analysis and rigorous argumentation)
- Value: ⭐⭐⭐⭐⭐ (corrects a misconception within the field and provides practical plug-and-play tools)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ECCV 2024\] Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)
- [\[CVPR 2026\] Explaining Object Detectors via Collective Contribution of Pixels](../../CVPR2026/object_detection/explaining_object_detectors_via_collective_contribution_of_pixels.md)
- [\[ICLR 2026\] Interference-Isolated Elastic Weight Consolidation and Knowledge Calibration for Incremental Object Detection](../../ICLR2026/object_detection/interference-isolated_elastic_weight_consolidation_and_knowledge_calibration_for.md)
- [\[CVPR 2026\] RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection](../../CVPR2026/object_detection/rhcnet_residual-guided_hierarchical_calibration_network_for_robust_underwater_ob.md)

</div>

<!-- RELATED:END -->
