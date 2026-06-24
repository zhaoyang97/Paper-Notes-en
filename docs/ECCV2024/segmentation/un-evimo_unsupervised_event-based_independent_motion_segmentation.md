---
title: >-
  [Paper Note] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation
description: >-
  [ECCV 2024][Segmentation][Event Camera] The first label-free independent motion object (IMO) segmentation framework for event cameras. It leverages optical flow and geometric constraints to generate pseudo-labels for training a segmentation network, achieving performance comparable to supervised methods on the EVIMO dataset.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Event Camera"
  - "Independent Motion Segmentation"
  - "Unsupervised"
  - "Pseudo-label"
  - "optical flow"
date: 2026-05-08
content_hash: ac01e511a261f267
---

# Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2312.00114](https://arxiv.org/abs/2312.00114)  
**Code**: [https://www.cis.upenn.edu/~ziyunw/un_evmoseg/](https://www.cis.upenn.edu/~ziyunw/un_evmoseg/)  
**Area**: Motion Segmentation / Event Camera  
**Keywords**: Event Camera, Independent Motion Segmentation, Unsupervised, Pseudo-label, optical flow

## TL;DR

The first label-free independent motion object (IMO) segmentation framework for event cameras. It leverages optical flow and geometric constraints to generate pseudo-labels for training a segmentation network, achieving performance comparable to supervised methods on the EVIMO dataset.

## Background & Motivation

Event cameras feature high temporal resolution, high dynamic range, and low power consumption, making them highly suitable for motion segmentation tasks requiring rapid response. However, existing event-based motion segmentation methods heavily rely on annotated data, which is extremely costly to acquire (for instance, the EVIMO dataset requires a Vicon multi-camera system to track objects and project them to generate masks).

Biological visual systems (such as insects hunting or humans avoiding obstacles while driving) can detect independently moving objects without explicit labels. Inspired by this, the authors pose a Core Problem: **Can unsupervised motion segmentation be learned using event cameras solely by observing motion patterns?**

Limitations of Prior Work:

**Mixture model methods** (e.g., EMSGC, EVIMO) require pre-defined fixed numbers of motion models and parametric forms, limiting their generalization capabilities.

**Supervised methods** (e.g., SpikeMS, EVDodgeNet) require a large amount of annotated IMO masks.

**Optimization-based methods** (e.g., EMSGC) require per-scene parameter tuning, resulting in extremely low inference efficiency.

## Method

### Overall Architecture

Un-EVIMO consists of two core modules:
1. **Geometric Self-Labeling Module** (Training Phase): Utilizes optical flow and depth information to estimate camera motion via RANSAC, computes the residual optical flow field, and generates IMO pseudo-labels through adaptive thresholding.
2. **Event Motion Segmentation Network** (Inference Phase): Takes only the event stream as input and directly predicts the binary IMO segmentation mask via a feedforward UNet.

Key Advantage: While depth and optical flow are required during training, only the event stream is needed at inference time, free from any additional sensor inputs.

### Key Designs

1. **Optical Flow Estimation with Independent Motion**: E-RAFT pretrained on DSEC exhibits poor optical flow estimation in IMO regions due to the lack of independently moving objects in its training data. The authors fine-tune E-RAFT using optical flow predicted by RAFT from grayscale images as supervision, enabling E-RAFT to correctly estimate optical flow in IMO regions. The EPE decreases from 11.15 with the baseline E-RAFT to 1.55 on the Table scene.

2. **Robust Camera Motion Estimation (RANSAC)**: IMO motion is inconsistent with camera motion; direct optimization would be biased by close-range fast-moving objects. The authors utilize a complete 6-DOF rigid body motion field model, sampling 3 points via RANSAC to solve the linear equation $A\theta = b$ (Eq. 7), and apply SVD to solve the overdetermined least-squares problem for all inlier pixels. The maximum iteration is 300, or the stopping probability reaches 0.999. Translation errors of the camera pose estimation achieve sub-centimeter accuracy (0.0082m, 0.0075m) in the Table and Floor scenes.

3. **Adaptive Geometric Thresholding (Otsu's Method)**: The residual optical flow $r(q_i) = \|\Psi(q_i) - \Psi_{cam}(q_i)\|_2$ typically displays a bimodal distribution—one peak corresponding to the rigid background (low residual) and another to the IMO (high residual). Otsu's method is adopted to automatically select the threshold by maximizing inter-class variance, avoiding the failure of fixed thresholds due to noise and depth variations across different scenes. Additionally, a two-stage confidence filtering is introduced: training samples are discarded if the total variance is too large (indicating indistinct optical flow boundaries) or the inter-class variance is too small.

4. **Optional Depth Input and Parametric Optical Flow**: Depth is only used during the training phase for pseudo-label generation. The authors also provide a depth-free alternative—using a 6-DOF or 12-DOF quadratic parametric optical flow model (Eq. 10-12), which, despite a slight decline in performance, still outperforms EMSGC.

### Loss & Training

- **Focal Loss**: Since IMOs typically occupy only a small area of the frame, causing a severe class imbalance, Focal Loss is used instead of standard cross-entropy.
- **Event Volume Representation**: A 15-channel event volume is used, where events are assigned to discrete spatio-temporal bins via bilinear interpolation kernels to preserve rich temporal information.
- **Network Architecture**: A UNet-like structure with a ResNet34 encoder (pretrained on ImageNet), where the bottleneck layer aggregates global features to differentiate between global camera motion and local IMO motion.
- **Optimizer**: Adam, learning rate $2 \times 10^{-4}$.

## Key Experimental Results

### Main Results

Event-masked IoU evaluation on the EVIMO dataset (Eq. 13, 40Hz evaluation frequency):

| Scene | Baseline CNN (Supervised) | EVIMO (Supervised) | SpikeMS (Supervised) | EMSGC Top30% (Unsupervised) | **Un-EVIMO (Unsupervised)** |
|------|:---:|:---:|:---:|:---:|:---:|
| Table | 66±23 | 79±6 | 50±8 | 55±17 | **50±21** |
| Box | 50±23 | 70±5 | 65±8 | 24±28 | **45±24** |
| Floor | 74±13 | 59±9 | 53±16 | 18±29 | **56±15** |
| Wall | 60±20 | 78±5 | 63±6 | 24±33 | **53±19** |
| Fast | 52±24 | 67±3 | 38±10 | 43±27 | **44±21** |

### Ablation Study

| Configuration | Table | Box | Floor | Wall | Fast | Description |
|------|:---:|:---:|:---:|:---:|:---:|------|
| (a) E-RAFT without fine-tuning | 32±23 | 28±21 | 35±19 | 42±22 | 27±23 | Pretrained optical flow lacks IMO |
| (b) 6-DOF parametrization | 43±26 | 42±25 | 51±21 | 47±23 | 37±24 | No depth required, simplified model |
| (c) 12-DOF parametrization | 47±24 | 40±25 | 56±18 | 49±22 | 37±25 | More flexible parametrization |
| **Full Model** | **50±21** | **45±24** | **56±15** | **53±19** | **44±21** | Depth + full motion field |

### Key Findings

1. **Optical Flow Quality is the Key Bottleneck**: Ablation experiments using un-fine-tuned E-RAFT exhibit a severe performance drop (from 50 to 32 on Table), demonstrating that accurate optical flow in IMO regions is crucial.
2. **Real-time Inference**: The total inference time of Un-EVIMO is only 6.57ms (3.35ms preprocessing + 3.22ms inference), which is significantly faster than EMSGC (9529ms) and SpikeMS (120ms).
3. **High Accuracy in Camera Pose Estimation**: The translation error is at the sub-centimeter level, and the rotation error is around 0.03 rad, proving the robustness of the flow+RANSAC approach.
4. **On Synthesized Motion Blur Videos**, the performance of supervised RGB methods degrades severely (Table IoU drops to 24), highlighting the advantages of event cameras in high-speed scenarios.

## Highlights & Insights

1. **Scalability of Geometric Self-Labeling**: Unconstrained by semantic information, this purely geometric method can generalize to arbitrary scenes without requiring object scanning or Vicon systems.
2. **Training-Inference Decoupling**: While depth and optical flow are required during training to generate pseudo-labels, only the event stream is needed during inference—an elegant knowledge distillation formulation.
3. **No Assumption on the Number of Objects**: Unlike mixture model methods, the pixel-wise classification approach used in this work naturally handles an arbitrary number of IMOs.
4. **Complete Motion Field Model**: Avoids simplifying geometry by utilizing the complete 6-DOF rigid motion field equations, ensuring theoretical correctness.

## Limitations & Future Work

1. **Lack of Temporal Consistency**: The current method predicts independently on single event slices, potentially leading to inconsistent predictions across consecutive frames (Fig. 5b). Temporal constraints or CRFs could be introduced.
2. **Missed Detections of Static/Slow-Moving IMOs**: Objects with very small residual optical flow are difficult to detect, requiring the integration of historical motion information.
3. **Blurry Boundaries**: Pseudo-labels are inherently noisy, causing the predicted mask boundaries to be less sharp compared to supervised methods.
4. **Depth Dependency**: Although inference does not require depth, optimal performance during the training phase still relies on depth input. Parametric optical flow models offer a potential depth-free alternative.

## Related Work & Insights

- Comparison with EMSGC highlights the advantages of end-to-end learning over frame-by-frame optimization (achieving a 1400x+ speedup without requiring parameter tuning).
- The self-labeling concept aligns with Yang & Ramanan (CVPR) on self-supervised segmentation based on scene flow errors. The key difference is that the high temporal resolution of event cameras allows for more accurate optical flow estimation.
- Inspiration: This framework can be generalized to other self-labeling scenarios—any signal that can be separated via geometric constraints can serve as a pseudo-label.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply geometric self-labeling to event-based IMO segmentation. The design decoupling training and inference is clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers quantitative evaluation across multiple scenes, ablation studies, speed comparisons, and failure case analyses, although evaluated on only one dataset (EVIMO).
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are complete and clear. The narrative logic from motion field equations to RANSAC and Otsu thresholding flows smoothly.
- **Value**: ⭐⭐⭐⭐ Real-time inference + label-free training provides significant import for practical event camera applications (such as autonomous driving and UAV obstacle avoidance).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders](colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](../../ICCV2025/segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)

</div>

<!-- RELATED:END -->
