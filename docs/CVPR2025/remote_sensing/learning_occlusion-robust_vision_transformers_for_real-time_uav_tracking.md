---
title: >-
  [Paper Note] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking
description: >-
  [CVPR 2025][Remote Sensing][UAV tracking] The ORTrack framework is proposed to learn occlusion-robust ViT feature representations through random masking based on spatial Cox processes (imposing mask constraints during training and achieving zero overhead during inference). An adaptive feature distillation method is designed to compress large models into a lightweight student model ORTrack-D, achieving the best balance of state-of-the-art accuracy and real-time speed across se…
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "UAV tracking"
  - "occlusion robustness"
  - "Vision Transformer"
  - "knowledge distillation"
  - "spatial Cox process"
date: 2026-05-08
content_hash: da1c8ceba4a261f0
---

# Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking

**Conference**: CVPR 2025  
**arXiv**: [2504.09228](https://arxiv.org/abs/2504.09228)  
**Code**: [https://github.com/wuyou3474/ORTrack](https://github.com/wuyou3474/ORTrack)  
**Area**: Video Understanding / Object Tracking  
**Keywords**: UAV tracking, occlusion robustness, Vision Transformer, knowledge distillation, spatial Cox process

## TL;DR

The ORTrack framework is proposed to learn occlusion-robust ViT feature representations through random masking based on spatial Cox processes (imposing mask constraints during training and achieving zero overhead during inference). An adaptive feature distillation method is designed to compress large models into a lightweight student model ORTrack-D, achieving the best balance of state-of-the-art accuracy and real-time speed across several UAV tracking benchmarks.

## Background & Motivation

**Background**: UAV tracking has shifted in recent years from DCF (Discriminative Correlation Filter) methods to single-stream architectures based on ViT (e.g., OSTrack, MixFormer) due to their structural simplicity and high accuracy, which have become mainstream. Lightweight methods such as Aba-ViTrack have achieved real-time UAV tracking.

**Limitations of Prior Work**: Occlusion events are highly frequent in UAV scenarios—obstacles like buildings and trees often block the target. Existing single-stream ViT trackers lack dedicated occlusion handling strategies, resulting in a significant drop in tracking accuracy when occlusion occurs. Furthermore, lightweight strategies like Aba-ViTrack utilize variable token numbers, leading to unstructured memory access and high actual inference latency.

**Key Challenge**: There is a critical need to enhance the occlusion robustness of ViT without sacrificing real-time inference speed—introducing complex occlusion handling modules compromises efficiency, while neglecting occlusion degrades accuracy.

**Goal**: (1) To enable ViT to learn occlusion-robust feature representations without introducing inference overhead; (2) To compress highly capable large models into deployment-friendly lightweight models.

**Key Insight**: If random masking is applied to template images during training while enforcing consistency between features extracted from the masked and unmasked templates, the model can naturally acquire features insensitive to occlusion. The key innovation lies in utilizing a spatial Cox process (rather than uniform random masking) to model the mask distribution, concentrating masks near the target center to more realistically simulate actual occlusions.

**Core Idea**: Simulate occlusions using random masking driven by a spatial Cox process during training to minimize the discrepancy between masked and unmasked features, thereby gaining occlusion robustness with zero extra overhead during inference.

## Method

### Overall Architecture

ORTrack consists of a two-stage sequential training process: the first stage trains a teacher model with occlusion-robust representation learning, and the second stage compresses the teacher into a student model using adaptive knowledge distillation. The entire framework is built upon a single-stream ViT tracking architecture, where the concatenated template $Z$ and search image $X$ are fed into the ViT, and the output tokens are processed by the prediction head to obtain the target bounding box.

### Key Designs

1. **Occlusion-Robust Representation Learning (ORR) based on Spatial Cox Process**:

    - Function: Simulates occlusion during training to allow ViT to learn occlusion-invariant features.
    - Mechanism: Applies a random mask $\mathfrak{m}(Z)$ to the template image $Z$ to obtain $Z'$, then separately feeds $(Z, X)$ and $(Z', X)$ into the ViT to minimize the MSE between their corresponding template tokens. The key lies in the mask distribution: instead of the uniform masking in MAE, a spatial Cox process is used to generate non-uniform masks. The Cox process is a "doubly stochastic Poisson process"—first generating a random intensity function $\lambda(x,y) = \Gamma e^{-(x^2+y^2)} / \int e^{-(x^2+y^2)}$, which is a bell-shaped function that concentrates the mask on the central region of the template (where the target is located). The random variable $\Gamma$ introduces randomness into the masking ratio, exposing the model to diverse occlusion patterns during training.
    - Design Motivation: Uniform masking is suboptimal on target templates, which typically contain background. Uniform masking might primarily mask the background rather than the target itself. The bell-shaped intensity function of the Cox process increases the probability of masking the target region, better simulating real occlusions. **No masking is required during inference, incurring zero extra overhead.**

2. **Adaptive Feature Knowledge Distillation (AFKD)**:

    - Function: Compresses the teacher model into a smaller student model while maintaining performance.
    - Mechanism: The student model shares the same structure as the teacher but has fewer ViT layers. The distillation loss is the MSE of the output tokens between teacher and student, scaled by an adaptive weight $\varpi = \alpha + \beta(\mathcal{L}_{iou} - \overline{\mathcal{L}_{iou}})$, where $\mathcal{L}_{iou}$ is the GIoU loss between student predictions and the ground truth. When the task is difficult (loss exceeds the average), the distillation strength is increased to guide the student; when the task is simple (loss is below the average), distillation is weakened to preserve the student's generalization capability.
    - Design Motivation: Traditional distillation treats all samples uniformly, but excessive mimicry can cause the student to inherit spurious correlations from the teacher. The adaptive strategy allows the student to "maintain autonomy in well-learned tasks" and "learn from the teacher in challenging tasks."

3. **Prediction Head and Loss Function**:

    - Function: Outputs the target localization bounding box.
    - Mechanism: Reshapes the output tokens of the search image into a 2D feature map, which is then processed by 4 Conv-BN-ReLU layers to predict classification scores, local offsets, and box sizes. Training loss: weighted Focal Loss (classification) + L1 + GIoU (regression) + MSE (ORR constraint).
    - Design Motivation: Standard prediction head from OSTrack is adopted for simplicity and efficiency.

### Loss & Training

Teacher stage: $\mathcal{L}_T = \mathcal{L}_{pred} + \gamma \mathcal{L}_{orr}$, where $\gamma = 2 \times 10^{-4}$. Student stage: freeze teacher weights, $\mathcal{L}_S = \mathcal{L}_{pred} + \mathcal{L}_{afkd}$. The models are trained for 300 epochs using the AdamW optimizer with an initial learning rate of $4 \times 10^{-5}$.

## Key Experimental Results

### Main Results

| Benchmark | Method | Prec. | Succ. | GPU FPS |
|------|------|-------|-------|---------|
| UAVDT | ORTrack-DeiT | **83.4** | **60.1** | 226 |
| UAVDT | Aba-ViTrack | 83.4 | 59.9 | 182 |
| VisDrone2018 | ORTrack-DeiT | **88.6** | **66.8** | 206 |
| VisDrone2018 | AQATrack (CVPR24) | 87.2 | 66.9 | 53 |
| DTB70 | ORTrack-DeiT | **86.2** | **66.4** | 226 |
| UAV123 | ORTrack-DeiT | 84.3 | 66.4 | 226 |

ORTrack-D-DeiT (distilled version) achieves an average Prec. of 83.7% and Succ. of 63.7% across 4 benchmarks, with speeds reaching 292 FPS (GPU) / 65 FPS (CPU).

### Ablation Study

| Configuration | UAVDT Prec. | UAVDT Succ. | Note |
|------|-------------|-------------|------|
| Baseline (w/o ORR) | 80.8 | 57.4 | Baseline ViT-tiny tracker |
| + Uniform Mask ORR | 82.1 | 58.5 | MAE-style uniform masking provides limited improvement |
| + Cox Process Mask ORR | **83.4** | **60.1** | Cox masking significantly outperforms uniform masking |
| w/o AFKD (Ordinary Distillation) | 81.5 | 58.9 | Non-adaptive distillation yields average performance |
| + AFKD | **82.5** | **59.7** | Adaptive distillation outperforms fixed-weight distillation |

### Key Findings

- Cox process masking improves Prec. by approximately 1.3% compared to MAE uniform masking, validating the hypothesis that center-focused masking better simulates actual occlusion.
- ORR only increases computation during training, resulting in absolutely zero overhead during inference, which makes it highly suitable for resource-constrained UAV deployment.
- The distilled ORTrack-D-DeiT drops only about 1% in Prec. but gains a 30% speedup (226 -> 292 FPS), showing high practicality.
- On VisDrone2018, ORTrack-DeiT at 206 FPS outperforms all deep trackers (such as AQATrack at 53 FPS), achieving 4x the speed with comparable accuracy.

## Highlights & Insights

- **Modeling occlusion with spatial Cox processes is a highly elegant design**: Unlike simple uniform masking or fixed-pattern masking, the Cox process concurrently models spatial non-uniformity (centers are more likely to be masked) and masking ratio randomness (the Poisson distribution of $\Gamma$), better simulating the diversity of real-world occlusions. This idea can be migrated to other tasks requiring occlusion robustness (e.g., person re-identification, pose estimation).
- **"Training-time constraint, inference-time zero-overhead" paradigm**: ORR serves only as an auxiliary loss during training without altering the network architecture. It can be easily integrated out-of-the-box into any ViT tracker, demonstrating high practicality.
- **"Hard-easy split" concept in adaptive distillation**: The GIoU deviation is employed as a proxy metric for task difficulty, which is simple, intuitive, and effective. This prevents overfitting, which typically arises from overly mimicking the teacher on simple samples.

## Limitations & Future Work

- The Cox process assumes that occlusion is concentrated at the center of the template, but actual occlusions can come from any direction (e.g., lateral occlusion); a more generalized occlusion simulation might require conditional mask distributions.
- A very small weight coefficient of $\gamma = 2 \times 10^{-4}$ is used to balance the ORR loss, suggesting that the magnitude and gradient direction of this loss might not be fully compatible with the tracking loss. A more in-depth gradient analysis is worth exploring.
- Distillation is applied only using the MSE of the final layer's features. Multi-layer feature distillation or relation distillation could bring further improvements.
- Evaluation has not been conducted on large-scale generic tracking benchmarks like LaSOT; generalization capability outside UAV scenarios remains to be validated.

## Related Work & Insights

- **vs Aba-ViTrack**: Aba-ViTrack improves efficiency using adaptive token computation but introduces unstructured memory access issues. ORTrack maintains a fixed token number and accelerates through distillation, making it more hardware-friendly. While both achieve comparable accuracy, ORTrack-D is faster.
- **vs DropMAE**: DropMAE also employs masking to enhance tracking robustness, but utilizes an MAE reconstruction framework. ORTrack avoids reconstruction and only constrains feature consistency, which is more lightweight.
- **vs SGDViT**: SGDViT improves tracking from a visual representation perspective but is slow (111 FPS). ORTrack comprehensively outperforms it in terms of both accuracy and speed.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing the spatial Cox process for occlusion simulation is theoretically novel, and the adaptive distillation is also well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 UAV benchmarks + generic tracking comparison + detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivation is rigorous, but the Cox process section might be slightly dense for readers without a strong mathematics background.
- Value: ⭐⭐⭐⭐ High practical utility; the "training constraint + zero inference overhead" paradigm can be widely transferred.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[ICML 2025\] ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts](../../ICML2025/remote_sensing/explora_parameter-efficient_extended_pre-training_to_adapt_vision_transformers_u.md)
- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](../../CVPR2026/remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization.md)

</div>

<!-- RELATED:END -->
