---
title: >-
  [Paper Note] Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework
description: >-
  [ECCV 2024][Human Understanding][Human Motion Prediction] Proposes the HoCoTTA framework, which achieves robust adaptation for human motion prediction in continuously changing target domains through multi-domain homeostasis assessment and isolated parameter optimization strategies, effectively mitigating catastrophic forgetting and error accumulation.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Motion Prediction"
  - "Continual Test-Time Adaptation"
  - "Domain Shift"
  - "Knowledge Distillation"
  - "Fisher Information Matrix"
date: 2026-05-08
content_hash: cce5f7a12645ae87
---

# Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Human Motion Prediction, Continual Test-Time Adaptation, Domain Shift, Knowledge Distillation, Fisher Information Matrix

## TL;DR
Proposes the HoCoTTA framework, which achieves robust adaptation for human motion prediction in continuously changing target domains through multi-domain homeostasis assessment and isolated parameter optimization strategies, effectively mitigating catastrophic forgetting and error accumulation.

## Background & Motivation

**Background**: Human motion prediction is a crucial task in computer vision and robotics, aiming to predict future human movements based on observed historical pose sequences. Existing methods often suffer from significant performance degradation when deployed directly to a target domain after training on a source domain, due to domain discrepancies (such as differing motion styles, environmental conditions, etc.).

**Limitations of Prior Work**: In recent years, domain adaptation solutions have typically relied on an unrealistic assumption that the target domain is static and unchanging. However, in real-world scenarios, the test distribution continuously changes over time. For instance, a motion prediction system deployed across different scenes encounters constantly changing users, action types, and environmental conditions. Existing Test-Time Adaptation (TTA) methods suffer from catastrophic forgetting (overwriting knowledge from previously adapted domains) and error accumulation (gradual amplification of errors during adaptation) when faced with such continual domain shifts.

**Key Challenge**: The model must simultaneously achieve two objectives under continuous domain shifts: (1) rapidly adapt to the characteristics of the new domain, and (2) preserve useful knowledge accumulated from previous domains. A fundamental tension exists between these two goals—over-adaptation to the new domain leads to the loss of old knowledge, while over-protection of old knowledge hinders adaptation to the new domain.

**Goal**: Formulating human motion prediction under continual domain shifts, specifically decomposed into: (1) How to distinguish between domain-invariant and domain-specific knowledge? (2) How to selectively update domain-specific parameters while preserving domain-invariant ones? (3) How to perform these operations at test time without labels?

**Key Insight**: The authors draw inspiration from the biological concept of "homeostasis"—the capacity of organisms to maintain an internal stable state despite continuous external environmental changes. Analogously, in model adaptation, domain-invariant parameters should remain stable like the core systems of an organism, while domain-specific parameters adapt flexibly to cope with new environments.

**Core Idea**: Evaluating parameter sensitivity to domain shifts using the Fisher Information Matrix to divide parameters into domain-invariant and domain-specific categories, isolating the optimization of domain-specific parameters to adapt to new domains while preserving domain-invariant knowledge.

## Method

### Overall Architecture
The overall pipeline of HoCoTTA follows the paradigms of knowledge distillation and parameter isolation. The input is a human pose sequence from a continuously shifting target domain, and the output is the predicted future motion. The framework comprises three core stages: (1) multi-domain homeostasis assessment, which analyzes the uncertainty of model parameters when facing new domain samples; (2) parameter sensitivity measurement, which identifies domain-sensitive and domain-invariant parameters via the Fisher Information Matrix; and (3) isolated parameter optimization, which selectively updates domain-specific parameters while freezing domain-invariant ones.

### Key Designs

1. **Multi-Domain Homeostasis Assessment**:

    - **Function**: Assess the degree of uncertainty of the current model parameters when facing samples from a new domain.
    - **Mechanism**: When the model encounters test samples from a new domain, it first estimates the uncertainty of the current parameters. High parameter uncertainty indicates sensitivity to domain shifts, identifying it as a domain-specific parameter. Low uncertainty indicates stability across different domains, identifying it as a domain-invariant parameter. This assessment does not require target domain labels and is implemented entirely at test time via the statistical properties of the model output.
    - **Design Motivation**: In continual TTA, not all parameters require updating. Blindly updating all parameters wastes computational resources and causes catastrophic forgetting of domain-invariant knowledge. Through homeostasis assessment, the subset of parameters that need adjustment can be pinpointed accurately.

2. **Fisher Information-based Parameter Sensitivity**:

    - **Function**: Quantify the sensitivity of each parameter to domain shifts to achieve accurate separation of domain-invariant and domain-specific parameters.
    - **Mechanism**: Compute the Fisher Information Matrix (FIM) to measure parameter sensitivity. The diagonal elements of the FIM reflect the impact of each parameter on the loss function—a larger Fisher information value indicates that the parameter has a greater impact on the current prediction and is more sensitive to domain shifts. Based on FIM values, thresholds are set to partition parameters into domain-sensitive (to be updated) and domain-invariant (to be preserved) categories. Formally, for a parameter $\theta_i$, a larger Fisher information value $F_i$ suggests that it is more domain-specific.
    - **Design Motivation**: Compared to simpler metrics like gradient magnitude or parameter change magnitude, FIM provides a more theoretically grounded measure of sensitivity. It directly correlates parameter variation with its influence on the model's output distribution, offering greater reliability than heuristic methods.

3. **Isolated Parameter Optimization Strategy**:

    - **Function**: Selectively update domain-specific parameters to adapt to new domains, while strictly protecting domain-invariant parameters from modification.
    - **Mechanism**: Based on the FIM analysis, parameters are divided into two groups. For domain-specific parameters (high Fisher information values), gradient updates are performed using a self-supervised loss to adapt to the new domain. For domain-invariant parameters (low Fisher information values), their gradients are frozen, completely preserving the knowledge learned during source domain training. Additionally, a knowledge distillation mechanism is used, leveraging a teacher model (updated via EMA) to provide extra stability constraints.
    - **Design Motivation**: This design directly addresses the core tension in continual TTA—mitigating forgetting through physical isolation rather than soft regularization, which is more thorough than traditional methods like EWC.

### Loss & Training
During the training phase, the model is trained on the source domain using standard motion prediction losses. In the test-time adaptation phase, self-supervised objectives (such as motion consistency constraints and temporal smoothness constraints) drive the update of domain-specific parameters. A knowledge distillation loss is employed to maintain consistency between the teacher and child models, preventing excessive drift during adaptation. The overall loss is a weighted combination of the adaptation loss and the distillation loss.

## Key Experimental Results

### Main Results

| Dataset/Setup | Metric | HoCoTTA | Prev. SOTA | Gain |
|-------------|------|---------|----------|------|
| Continual Domain Shift Setting | Average Prediction Error | **Best** | Second Best Method | Substantial Improvement |
| Multi-Domain Sequence TTA | Long-Term Adaptation Stability | **Best** | Significant Degradation | Strong Anti-Forgetting Capability |

### Ablation Study

| Configuration | Prediction Error | Description |
|------|---------|------|
| Full HoCoTTA | Lowest | Complete Model |
| w/o Homeostasis Assessment | Significantly Increased | Inability to distinguish domain-invariant/specific parameters |
| w/o Parameter Isolation | Increased | Jointly updating all parameters leads to forgetting |
| w/o Fisher Information | Increased | Replacing with simple gradients yields poor results |
| Full Parameter Update TTA | Highest | Most severe catastrophic forgetting |

### Key Findings
- The parameter isolation strategy is key to resisting forgetting—removing isolation leads to significant error accumulation under continual domain shifts.
- The Fisher Information Matrix identifies domain-sensitive parameters more accurately than a simple gradient norm.
- In scenarios with severe domain changes, the advantages of HoCoTTA are more prominent, demonstrating excellent robustness.
- The EMA update rate of the teacher model needs to be carefully tuned; an excessively fast update rate will cause the teacher model to also drift towards the new domain.

## Highlights & Insights
- **Ingenious Application of the Homeostasis Metaphor**: Mapping the biological concept of homeostasis to the parameter space and using the FIM as an "internal thermometer" to monitor parameter states. This cross-domain analogy provides clear intuition and theoretical support for the method design.
- **Label-Free Parameter Analysis at Test Time**: Discriminating between domain-invariant and domain-specific parameters solely via the FIM without target domain labels, which is highly valuable for real-world deployment.
- **Generalization Potential of the Continual TTA Paradigm**: The framework of parameter isolation combined with FIM analysis is not limited to motion prediction; theoretically, it can be extended to any vision task facing continual domain shifts, such as object detection in autonomous driving or robotic manipulation in diverse environments.

## Limitations & Future Work
- The computation of the FIM may incur high computational overhead for large models, necessitating the exploration of more efficient approximation methods.
- Using a fixed threshold for parameter partitioning may not adapt well to scenarios where the rate of domain drift varies—dynamic thresholding strategies merit further investigation.
- The experiments are primarily validated on human motion prediction; generalization to other sequential forecasting tasks (e.g., trajectory prediction, gesture recognition) remains to be verified.
- The current method assumes discrete domain shifts; performance under gradual domain drifts requires further investigation.

## Related Work & Insights
- **vs CoTTA**: CoTTA utilizes augmentation consistency and stochastic restoration to combat forgetting but does not explicitly distinguish between domain-invariant and domain-specific parameters, leading to accumulated errors under large shifts. HoCoTTA achieves more precise parameter selection via the FIM.
- **vs EWC (Elastic Weight Consolidation)**: EWC also uses the FIM to protect important parameters, but it is designed under a supervised continual learning setting. HoCoTTA generalizes this to the unlabeled TTA setting. Furthermore, while EWC protects parameters "important to past tasks," HoCoTTA protects "cross-domain invariant" parameters, making them fundamentally different in concept.
- **vs TENT**: TENT performs TTA by minimizing entropy, but it updates all Batch Normalization (BN) parameters without selection, making it prone to forgetting under continual drift.
- **vs DUA**: DUA adapts to domain shifts by maintaining BN statistics, but the effectiveness of BN statistics is limited in temporal tasks such as human motion prediction.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of the homeostasis concept and FIM-based parameter isolation in continual TTA is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated under multiple domain shift settings with relatively complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with an illustrative metaphor of homeostasis.
- Value: ⭐⭐⭐⭐ Continual domain shift is a critical problem in practical deployment, and the methodology is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](../../CVPR2025/human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[AAAI 2026\] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](../../AAAI2026/human_understanding/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)
- [\[ECCV 2024\] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [\[ICLR 2026\] HUMOF: Human Motion Forecasting in Interactive Social Scenes](../../ICLR2026/human_understanding/humof_human_motion_forecasting_in_interactive_social_scenes.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)

</div>

<!-- RELATED:END -->
