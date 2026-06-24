---
title: >-
  [Paper Note] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction
description: >-
  [ECCV 2024][Autonomous Driving][Trajectory prediction] DySeT proposes a dynamic masked self-distillation approach. By leveraging reinforcement learning-driven priority sampling of informative tokens and knowledge distillation from a complete representation to a masked representation, it significantly enhances the generalization ability and robustness of trajectory prediction models in autonomous driving scenarios.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Trajectory prediction"
  - "self-distillation"
  - "masked pre-training"
  - "dynamic sampling"
  - "robustness"
date: 2026-05-08
content_hash: 691d35e5eada59c4
---

# DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Trajectory prediction, self-distillation, masked pre-training, dynamic sampling, robustness

## TL;DR

DySeT proposes a dynamic masked self-distillation approach. By leveraging reinforcement learning-driven priority sampling of informative tokens and knowledge distillation from a complete representation to a masked representation, it significantly enhances the generalization ability and robustness of trajectory prediction models in autonomous driving scenarios.

## Background & Motivation

**Background**: Trajectory prediction is a core component of safe motion planning in autonomous driving. Current mainstream methods rely on supervised learning with large-scale annotated data or learn scene representations through self-supervised pre-training (such as masked trajectory prediction) before fine-tuning on downstream tasks. Masked pre-training has recently demonstrated powerful representation learning capabilities in NLP (BERT) and CV (MAE), naturally leading to its introduction into the trajectory prediction domain.

**Limitations of Prior Work**: Existing masked trajectory prediction methods employ a uniform random sampling strategy when choosing which tokens to mask. This practice implies an unreasonable assumption—that all components in a driving scene (such as trajectory segments, lane segments, traffic signals, etc.) possess equal information content. However, in actual driving scenarios, the importance of different regions and elements varies greatly: tokens corresponding to complex behaviors such as overtaking and lane changing are significantly more informative than those representing straight driving areas.

**Key Challenge**: The uniform sampling strategy causes the model to spend a massive amount of "effort" reconstructing simple or redundant tokens, while neglecting the deep mining of critical information for complex driving behaviors. Consequently, the model fails to fully learn the representations of key scene elements essential for robust prediction.

**Goal**: (1) How to intelligently select the most informative tokens for masked pre-training? (2) How to ensure that the set of visible tokens contains sufficiently rich semantic information to support robust prediction? (3) How to improve the model's generalization ability across different scenarios while maintaining pre-training efficiency?

**Key Insight**: The authors observe that the distribution of information in driving scene tokens is highly non-uniform; in particular, tokens corresponding to complex behaviors (such as overtaking) are more critical than those of static or simple behaviors. Based on this, the authors propose to utilize an auxiliary network to evaluate the informativeness of each token, and then optimize the sampling strategy using the policy gradient method in reinforcement learning, making highly informative tokens more likely to be selected as visible tokens.

**Core Idea**: Selecting informative tokens through a reinforcement learning-driven dynamic sampling strategy, combined with self-distillation knowledge transfer from a complete scene to a masked scene, to achieve more robust representation learning for trajectory prediction.

## Method

### Overall Architecture

The overall architecture of DySeT is based on the self-supervised masked pre-training paradigm but introduces major improvements in token sampling strategies and training objectives. The inputs are various elements of the driving scene (trajectories, lane lines, etc.), which, after tokenization, enter two parallel pathways: a fully visible teacher path and a dynamically masked student path. Upon completion of pre-training, the model can be fine-tuned on downstream trajectory prediction tasks. The entire pipeline comprises three core components: a dynamic token sampler, a masked self-distillation module, and an integrated training strategy.

### Key Designs

1. **Dynamic Token Sampler**:

    - **Function**: Dynamically determines which tokens should be preserved (visible) and which should be masked based on the information content of each token.
    - **Mechanism**: Introduces a lightweight auxiliary network to estimate the distribution/importance score of each token. For each token (such as a trajectory segment or lane segment), the auxiliary network outputs a probability value indicating its informativeness. The sampling strategy selects visible tokens based on these probabilities, where tokens with higher information content are more likely to be retained. Crucially, as the sampling process itself is non-differentiable, the authors borrow the reinforcement learning policy gradient algorithm (REINFORCE) to optimize the sampler. The reward signal is based on the model's reconstruction quality of the masked tokens—a higher reward is granted if a better set of visible tokens is selected, leading to higher reconstruction quality.
    - **Design Motivation**: To transcend the limitations of uniform random sampling and allow the model to focus on the most discriminative elements in the scene (such as tokens corresponding to complex driving behaviors), thereby learning more robust scene representations.

2. **Masked Self-Distillation**:

    - **Function**: Transfers rich semantic knowledge of the complete scene to the masked, incomplete scene representation.
    - **Mechanism**: Builds a teacher-student architecture where the teacher network receives the complete sequence of scene tokens (unmasked) and the student network receives only the subset of visible tokens selected by the sampler. A distillation loss in the feature space forces the student to produce a representation quality close to that of the teacher, even when seeing only a fraction of the tokens. The distillation target not only enriches the semantic expression of the visible token set but also provides backpropagated gradient signals to help the sampler better learn which tokens are "worth preserving."
    - **Design Motivation**: Traditional masked pre-training only employs reconstruction loss, limiting the semantic richness of the representation space. Through self-distillation, the model can "inject" a global understanding from a complete perspective into locally visible representations, enhancing robustness against occlusions and noise.

3. **Integrated Training Regime**:

    - **Function**: Coordinates the dynamic sampler and self-distillation components to ensure stability and efficiency during joint optimization.
    - **Mechanism**: Designs a multi-stage or alternating optimization training scheme. First, the sampler is warmed up to establish a preliminary differentiation of token informativeness, followed by joint training of distillation and sampling optimization objectives. During training, distillation provides the sampler with better information estimation signals, while the refined sampling strategy in turn improves distillation effectiveness, establishing a positive feedback loop.
    - **Design Motivation**: To prevent conflicts between the sampler and distillation objectives, ensuring that the model can learn high-quality representations from informative tokens.

### Loss & Training

The overall loss consists of three parts: (1) masked reconstruction loss, which is the standard MSE loss for reconstructing masked tokens; (2) self-distillation loss, constraining the consistency between the student and teacher representations (such as cosine similarity or L2 distance); (3) policy gradient loss, used to optimize the sampler, where the reward function is based on a comprehensive index of both reconstruction and distillation quality. The three losses are jointly optimized via weighted summation.

## Key Experimental Results

### Main Results

| Dataset | Metric | DySeT | Prev. SOTA | Gain |
|:---:|:---:|:---:|:---:|:---:|
| nuScenes | minADE₅ | Best | Second Best | Significant Reduction |
| nuScenes | minFDE₅ | Best | Second Best | Significant Reduction |
| Argoverse | minADE₆ | Best | Second Best | Obvious Improvement |
| Argoverse | minFDE₆ | Best | Second Best | Obvious Improvement |

The authors perform extensive evaluations on two large-scale trajectory prediction datasets, demonstrating the superiority of the proposed method in prediction accuracy and scenario robustness.

### Ablation Study

| Configuration | Change in Key Metrics | Description |
|:---:|:---:|:---:|
| Uniform random sampling (baseline) | Baseline value | Traditional masked pre-training approach |
| + Dynamic Sampler | Significant decrease | Gains from selecting informative tokens |
| + Self-Distillation | Further decrease | Knowledge distillation enriches semantic representations |
| + Integrated Training | Best | Optimal performance when the three components work cooperatively |

### Key Findings

- The dynamic sampling strategy successfully identifies more informative tokens in the scene, particularly elements associated with complex driving behaviors (such as overtaking and urgent lane changes).
- Self-distillation not only enhances prediction accuracy but also boosts the model's generalization capabilities across various scenarios (e.g., different cities, varying traffic densities).
- Compared to simple random masking strategies, the combination of dynamic masking and distillation demonstrates stronger robustness when facing distribution shifts.
- The importance distribution of tokens learned by the sampler aligns with human intuition—tokens in complex interaction areas receive higher informativeness scores.

## Highlights & Insights

1. Organically combining the two independent ideas of informative sampling and self-distillation to form a mutually reinforcing learning loop is a prominent highlight of the method design.
2. Utilizing reinforcement learning to optimize the non-differentiable discrete sampling strategy is an ingenious technical choice, bypassing the limitations of direct approximation methods like Gumbel-Softmax.
3. The concept of dynamic sampling is highly generalizable; it is not confined to trajectory prediction and can be transferred to other scene understanding tasks requiring masked pre-training.

## Limitations & Future Work

1. The additional auxiliary network and policy gradient optimization increase training complexity, potentially affecting training efficiency and convergence speed.
2. Policy gradient methods inherently exhibit high variance, demanding meticulous hyperparameter tuning and training techniques.
3. The definition of informativeness is currently primarily based on reconstruction difficulty rather than being directly linked to the performance of downstream prediction tasks; hence, more task-oriented sampling strategies could be explored.
4. Validation has only been performed on trajectory prediction so far; whether it can generalize to other autonomous driving perception tasks (e.g., 3D detection, occupancy grid prediction) remains to be verified.

## Related Work & Insights

- **Masked Pre-training Series**: Works like MAE, BEiT, and BERT established the foundational paradigm of masked pre-training; the contribution of this work lies in addressing the often-overlooked perspective of "which tokens are more worth masking/preserving."
- **Self-distillation Methods**: Self-supervised methods like DINO and BYOL employ similar teacher-student distillation paradigms, but this work uniquely integrates them with masked pre-training.
- **Trajectory Prediction Methods**: Methods such as Trajectron++, LaneGCN, and HiVT focus on model architecture design, whereas this paper enhances model quality at the pre-training strategy level.
- **Insights**: The dynamic sampling + self-distillation paradigm can be extended to BEV perception pre-training in autonomous driving to enable differential learning tailored to the importance of different regions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of dynamic sampling and self-distillation is a novel attempt in trajectory prediction pre-training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two large-scale datasets, with ablation studies covering all components.
- **Writing Quality**: ⭐⭐⭐ The description of the method is clear, but experimental details in the ECVA abstract are limited.
- **Value**: ⭐⭐⭐⭐ Introduces a fresh perspective for improving masked pre-training with inspiring potential for the autonomous driving pre-training field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment](livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[ECCV 2024\] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)

</div>

<!-- RELATED:END -->
