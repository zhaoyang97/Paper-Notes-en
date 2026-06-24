---
title: >-
  [Paper Note] Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation
description: >-
  [ECCV 2024][Autonomous Driving][Source-Free Domain Adaptation] To address the performance degradation issue in the late training stage of Source-Free Unsupervised 3D Domain Adaptation (SFUDA) for 3D semantic segmentation, this paper proposes regularization strategies and validation criteria based on reference model consistency to achieve stable and robust adaptation.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Source-Free Domain Adaptation"
  - "3D Semantic Segmentation"
  - "LiDAR"
  - "Training Stability"
  - "Hyperparameter Selection"
date: 2026-05-08
content_hash: 31481f9b908650b9
---

# Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation

**Conference**: ECCV 2024  
**arXiv**: [2409.04409](https://arxiv.org/abs/2409.04409)  
**Code**: Yes  
**Area**: Autonomous Driving  
**Keywords**: Source-Free Domain Adaptation, 3D Semantic Segmentation, LiDAR, Training Stability, Hyperparameter Selection

## TL;DR

To address the performance degradation issue in the late training stage of Source-Free Unsupervised 3D Domain Adaptation (SFUDA) for 3D semantic segmentation, this paper proposes regularization strategies and validation criteria based on reference model consistency to achieve stable and robust adaptation.

## Background & Motivation

**Source-Free Unsupervised Domain Adaptation (SFUDA)** is one of the most challenging settings in domain adaptation. When adapting to the target domain, only a model pre-trained on the source domain is available without access to source data, and the target domain data contains no labels. This setting is highly practical in scenarios like autonomous driving, where source data is typically unshareable due to data privacy, storage limitations, or legal requirements.

In the SFUDA of 3D semantic segmentation (LiDAR point cloud segmentation), researchers face a common yet tricky issue: **performance degradation in the late stage of training**. Specifically, performance improves initially during adaptation but starts to decline with further training, sometimes dropping below the level of the initial model. This is because SFUDA is intrinsically an **under-constrained ill-posed problem**. Without label information, the model is prone to deviating further in the positive feedback loop of pseudo-labels.

The key consequence of this problem is that, during practical deployment, one cannot determine the right moment to stop training without a labeled validation set to monitor performance. If this problem is not solved, SFUDA methods remain unreliable in practice.

## Method

### Overall Architecture

This paper proposes two complementary strategies: (1) **regularization strategies** to heavily constrain the learning problem during adaptation, fundamentally slowing down performance degradation; (2) **agreement-based stopping/validation criteria** using reference models to determine when to stop training and how to select hyperparameters based on prediction consistency with the original source model.

### Key Designs

1. **Learning Problem Regularization**:
    - Function: Constrains the adaptation process to prevent the model from drifting too far.
    - Mechanism: Introduces multiple regularizations during the self-training process: (a) Source model knowledge retention: training on target samples using pseudo-labels while constraining the model parameters from deviating too far from the source model (conceptually similar to EWC); (b) Pseudo-label quality control: setting a dynamic confidence threshold to train only with high-confidence pseudo-labels; (c) Data augmentation regularization: requiring consistent predictions for different augmented versions of the same point cloud.
    - Design Motivation: The root cause of performance degradation in SFUDA is the positive feedback accumulation of pseudo-label noise. Regularization limits this accumulation from multiple perspectives.

2. **Agreement-based Criterion**:
    - Function: Evaluates model quality and selects hyperparameters on the unlabeled target domain.
    - Mechanism: Treats the source model as a reference model, evaluating the adaptation quality by calculating the prediction agreement between the current adapted model and the reference model on the target domain. Key assumption: a good adaptation should both improve target domain performance (differing from the reference model) and maintain basic semantic understanding capabilities (retaining consensus with the reference model). Too low agreement indicates excessive deviation, while too high agreement indicates no adaptation.
    - Design Motivation: In traditional settings, validation requires labeled data; the agreement-based criterion proposed here offers a label-free alternative.

3. **Dual-purpose: Training Stopping and Hyperparameter Selection**:
    - Function: Makes SFUDA methods operationally viable in real-world deployment.
    - Mechanism: (a) Training stopping: monitoring the agreement curve and stopping training when agreement begins to drop abnormally; (b) Hyperparameter selection: running adaptation for different hyperparameter configurations and choosing the configuration with the optimal agreement metric. These two features remove the need for labeled target domain data in decision-making.
    - Design Motivation: Hyperparameter sensitivity is another practical barrier in SFUDA. A label-free validation criterion is essential.

### Loss & Training

- Self-training loss: Pseudo-label-based cross-entropy loss (filtered with confidence thresholding).
- Consistency regularization: KL divergence between predictions of augmented versions.
- EWC-like parameter regularization: Constrains critical parameters from deviating from the source model.
- Training strategy: Dynamic threshold scheme, using a higher (conservative) threshold initially and gradually relaxing it later.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| nuScenes→SemanticKITTI | mIoU | SOTA | CoSMix, etc. | +2-5% |
| SynLiDAR→SemanticKITTI | mIoU | SOTA | GIPSO, etc. | +3-6% |
| SynLiDAR→nuScenes | mIoU | SOTA | Multiple baselines | +2-4% |
| Long-term Training Preservation | Degradation Magnitude↓ | Significantly reduced | Baselines degraded severely | Significant stability improvement |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Without Regularization | Post-training degradation | Collapses in later stages with baseline methods |
| Regularization Only | Slowed degradation | But manual determination of stopping point is still needed |
| Agreement-based Stopping Only | Partial improvement | But insufficient regularization might lead to suboptimal stopping points |
| Regularization + Agreement Criterion | Optimal and most stable | Complete method |

### Key Findings

- Performance degradation in the late stage of training is a common issue for all SFUDA methods and should not be neglected.
- The combination of regularization and automatic stopping criteria makes SFUDA genuinely usable in practice.
- The effectiveness of the agreement-based validation criterion using a reference model is surprisingly strong — even outperforming validation with a small amount of randomly labeled data on the target domain.
- The proposed method can be applied as a "plug-in" to any existing SFUDA method to stably improve its performance.

## Highlights & Insights

- Directly addresses practical deployment issues of SFUDA (when to stop, how to tune hyperparameters) instead of solely pursuing metric improvements.
- The "plug-in" design allows the method to seamlessly improve any existing SFUDA approach in a plug-and-play manner.
- The agreement-based validation criterion using the reference model is a simple yet powerful tool.
- The paper title "Train Till You Drop" vividly describes the problem — the longer the training runs, the worse the performance gets.

## Limitations & Future Work

- The effectiveness of the agreement-based criterion depends on the quality of the source model; if the source model is poor, the reference may be misleading.
- The choice of regularization strength still requires some empirical judgment.
- Validation was mainly performed on 3D LiDAR data; the applicability to 2D image SFUDA requires further study.
- Adaptive strategy tuning for regularization strength can be explored.
- Online/streaming SFUDA scenarios represent a valuable direction for future expansion.

## Related Work & Insights

- **CoSMix / GIPSO**: Representative methods for 3D domain adaptation, but both suffer from training degradation issues.
- **SHOT / TENT**: 2D SFUDA methods; the strategies proposed in this paper may also be applicable.
- **EWC**: A parameter regularization method in continual learning, borrowed here for SFUDA.
- Insights: The reliability and deployability of SFUDA methods are as important as the pursuit of performance metrics.

## Rating

- Novelty: ⭐⭐⭐ The ideas of regularization and stopping criteria are not entirely new, but their application to the SFUDA degradation problem is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple 3D domain adaptation scenarios.
- Writing Quality: ⭐⭐⭐⭐ Vivid description of the problem (excellent title) and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ Addresses key pain points in SFUDA deployment, and the method is easy to integrate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)
- [\[CVPR 2026\] PanDA: Unsupervised Domain Adaptation for Multimodal 3D Panoptic Segmentation in Autonomous Driving](../../CVPR2026/autonomous_driving/panda_unsupervised_domain_adaptation_for_multimodal_3d_panoptic_segmentation_in_.md)
- [\[ECCV 2024\] LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment](livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)
- [\[ECCV 2024\] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)

</div>

<!-- RELATED:END -->
