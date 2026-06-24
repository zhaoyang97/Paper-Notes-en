---
title: >-
  [Paper Note] Perceptual Inductive Bias is What You Need Before Contrastive Learning
description: >-
  [CVPR 2025][3D Vision][Contrastive Learning] Inspired by David Marr's multi-stage visual processing theory, this paper proposes adding a "pre-pretraining" stage prior to standard contrastive learning. By using foreground-background segmented shape silhouettes and intrinsic image decomposition (albedo + shading) as perceptual inductive biases, this approach achieves 2x faster convergence on ResNet18 and comprehensive improvements across downstream tasks such as segmentation…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Contrastive Learning"
  - "Perceptual Inductive Bias"
  - "Shape Prototypes"
  - "Intrinsic Image Decomposition"
  - "Pre-pretraining"
date: 2026-05-08
content_hash: 9dbe8f01ebf242eb
---

# Perceptual Inductive Bias is What You Need Before Contrastive Learning

**Conference**: CVPR 2025  
**arXiv**: [2506.01201](https://arxiv.org/abs/2506.01201)  
**Code**: None  
**Area**: Self-Supervised Learning / 3D Vision  
**Keywords**: Contrastive Learning, Perceptual Inductive Bias, Shape Prototypes, Intrinsic Image Decomposition, Pre-pretraining

## TL;DR

Inspired by David Marr's multi-stage visual processing theory, this paper proposes adding a "pre-pretraining" stage prior to standard contrastive learning. By using foreground-background segmented shape silhouettes and intrinsic image decomposition (albedo + shading) as perceptual inductive biases, this approach achieves 2x faster convergence on ResNet18 and comprehensive improvements across downstream tasks such as segmentation, depth estimation, and recognition.

## Background & Motivation

**Background**: Contrastive learning (e.g., SimCLR, MoCo, BYOL) represents the dominant paradigm for self-supervised representation learning, which learns semantic representations by maximizing the mutual information between different views of the same image. These methods typically learn high-level semantic spaces directly from raw images, bypassing the construction of mid-level visual representations.

**Limitations of Prior Work**: End-to-end contrastive learning suffers from two major limitations: (1) slow convergence, requiring a large number of epochs to learn high-quality representations; (2) texture bias in the learned representations, where models favor texture shortcuts over shape information, unlike the human visual system. Additionally, a trade-off exists between semantic-level classification tasks and pixel-level segmentation/depth estimation tasks, making it difficult for a single contrastive learning framework to optimize both simultaneously.

**Key Challenge**: Human vision processes information in stages—first perceiving boundaries and surface properties (mid-level representations) before forming semantic object representations. However, contemporary contrastive learning directly leaps to the semantic layer, neglecting the inductive biases of mid-level perceptual constructs, which renders the representations insensitive to shape, depth, and surface characteristics.

**Goal**: To validate and exploit Marr's multi-stage theory—first constructing mid-level representations at the boundary and surface levels, then training semantic representations—in order to: (1) accelerate the convergence of contrastive learning, (2) improve performance on downstream tasks (classification, segmentation, and depth estimation), and (3) enhance shape bias and robustness.

**Key Insight**: The authors draw on developmental psychology findings showing that infants learn vocabulary through shape prototypes, indicating that shape perception is critical in early visual development. In addition, intrinsic image decomposition (albedo and shading) encodes surface material properties and implicit 2.5D information respectively, which should yield differentiated gains for different downstream tasks.

**Core Idea**: To design three perceptual constructs—Shape Prototypes, Reflectance, and Shading—as inductive biases injected into contrastive learning prior to pretraining. A hybrid coarse-to-fine strategy is proposed: first accelerate initial learning using shape prototypes, then switch back to standard contrastive learning for fine-grained refinement.

## Method

The overall methodology comprises three independent perceptual construct components that can be used either individually or in combination. The core mechanism is to introduce mid-level representations of early visual stages (shape silhouettes, intrinsic images) into the contrastive learning framework as additional "views" or "prototypes", thereby equipping the network with the inductive biases of the human visual system.

### Overall Architecture

The inputs are images from ImageNet-100. During the preprocessing phase, TRACER is used offline to generate foreground-background segmented shape silhouettes, and the Retinex algorithm decomposes images into albedo and shading maps. During the training phase, depending on the configuration, S-PCL (Shape-Prototypical Contrastive Learning), ReflCL (Reflectance Contrastive Learning), ShadCL (Shading Contrastive Learning), or MidVCL (a combination of the three) is selected to conduct 100 epochs of pre-pretraining, followed by a switch to standard MoCoV2 for 300 epochs of training.

### Key Designs

1. **Shape-Prototypical Contrastive Learning (S-PCL)**:

    - **Function**: Learn clustering prototypes based on shape silhouettes to guide representation learning.
    - **Mechanism**: Images are fed into an online encoder to obtain representation $V$, while shape silhouettes are fed into a momentum encoder to obtain representation $U$. $K$-Means clustering is performed on $U$ to obtain $K$ shape prototypes $S = \{s_1, ..., s_K\}$. The ShapeProtoNCE loss is used to maximize the mutual information between the image representation $v_i$ and its corresponding shape prototype $s_p$, combined with the standard InfoNCE. Multiple clusterings (with different values of $K$) are averaged to achieve multi-granularity prototypes.
    - **Design Motivation**: Humans rely on the global shape envelope of an object for recognition, categorizing objects with similar shapes together. By clustering shape silhouettes to obtain prototypes, the network can rapidly build shape perception capabilities. However, experiments reveal that S-PCL saturates in performance after 100 epochs—exactly demonstrating that shape bias serves as a "starter" rather than a final solution, necessitating subsequent semantic contrastive learning for refinement.

2. **Intrinsic Image View Contrastive Learning (ReflCL / ShadCL)**:

    - **Function**: Utilize intrinsic images (albedo or shading) as augmented views in contrastive learning.
    - **Mechanism**: Raw images extract representations via the online encoder, while intrinsic images (albedo or shading) extract representations via the momentum encoder to compute the InfoNCE loss between them. The final loss is the sum of the standard two-view InfoNCE and the intrinsic image InfoNCE. Albedo maps preserve object surface colors/materials while removing illumination effects; shading maps capture implicit interactions between 3D shape and lighting.
    - **Design Motivation**: Albedo maps aid in instance boundary detection (segmentation based on material differences), contributing to segmentation and recognition but not to depth estimation; shading maps contain rich implicit 2.5D information, improving depth estimation but providing limited help for segmentation/recognition. This differentiated gain validates the hypothesis that different perceptual constructs contribute differently to various downstream tasks.

3. **Hybrid Coarse-to-Fine Strategy (Hybrid Coarse-to-Fine)**:

    - **Function**: Accelerate startup using perceptual biases first, then refine via semantic contrast.
    - **Mechanism**: Training is divided into two phases—the first 100 epochs use S-PCL/MidVCL for pre-pretraining, and the subsequent 300 epochs switch to standard MoCoV2/PCL. Shape prototypes quickly establish shape-perception capability in the early stages, but continued usage saturates or even hinders finer-grained semantic learning. This mirrors human development, where shape perception is developed before vocabulary and concepts.
    - **Design Motivation**: S-PCL's behavior on the AMI (Adjusted Mutual Information) metric aligns with this—it grows rapidly in the early stages and declines later. This suggests that shape clustering is initially highly correlated with semantic categories, but as training progresses, finer-grained semantic discrimination requires breaking through the boundaries of shape prototypes.

### Loss & Training

The overall loss function is formulated as $\mathcal{L} = \mathcal{L}_{InfoNCE} + \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{ShapeProtoNCE, K_i} + \alpha \mathcal{L}_{Shad} + \beta \mathcal{L}_{Refl}$, where $\alpha, \beta$ denote the weights for shading and reflectance losses. Training strategy: pre-pretrain with S-PCL/MidVCL for 100 epochs, followed by standard contrastive learning for 300 epochs. ResNet18 is utilized as the encoder, outputting a 256-D representation.

## Key Experimental Results

### Main Results

| Method | Epochs | IN-100 Top-1 | IN-1k Top-1 | ADE20K mIoU | Depth RME |
|------|--------|-------------|-------------|-------------|-----------|
| SimCLR | 400 | 77.2 | 40.8 | 30.4 | 0.1420 |
| MoCoV2 | 400 | 77.0 | 41.6 | 30.4 | 0.1434 |
| BYOL | 400 | 75.8 | 42.9 | 30.9 | 0.1458 |
| S-PCL | 100 | **70.2** | **37.2** | - | - |
| S-PCL+MoCoV2 | 400 | **78.0** | **43.9** | **31.9** | 0.1398 |
| MidVCL+MoCoV2 | 400 | 77.8 | 43.8 | **31.9** (tie) | **0.1354** |

### Ablation Study

| Configuration | IN-100 Top-1 (100ep) | IN-100 Top-1 (400ep) |
|------|---------------------|---------------------|
| S-PCL Training Alone | 70.2 (Best) | 71.8 (Saturated) |
| MoCoV2 Training Alone | 61.7 | 77.0 |
| S-PCL→MoCoV2 | - | **78.0** (Best) |
| MidVCL→MoCoV2 | - | 77.8 |

### Key Findings

- S-PCL performs best in the early stage (100 epochs), outperforming all baselines by approximately 5-8 percentage points; however, its performance saturates or even declines after 400 epochs (71.8% vs. 77% baseline), indicating that the shape bias needs to be released in a timely manner.
- The hybrid strategy S-PCL+MoCoV2 achieves optimal or near-optimal results across all downstream tasks, while accelerating convergence speed by 2x.
- Reflectance assists in classification and segmentation, shading assists in depth estimation, and their combination (MidVCL) is optimal for both depth and segmentation.
- S-PCL and MidVCL improve segmentation by approximately 1.4 and 1.7 mIoU points on ADE20K, respectively, with even more pronounced improvements on Cityscapes (68.3% vs. 63.4% baseline).

## Highlights & Insights

- Translates cognitive science theories (Marr's vision theory, infant shape development) into concrete algorithm designs, featuring highly solid theoretical motivation.
- The differentiated gains of the three perceptual constructs across various tasks (reflectance $\rightarrow$ recognition/segmentation, shading $\rightarrow$ depth, shape $\rightarrow$ all) provide intriguing insights.
- The concept of "pre-pretraining" is simple yet effective, maintaining the main training framework unchanged, making it highly integrable.
- The "early rise and late decline" behavior of S-PCL, along with its analogy to infant development, offers a novel biological perspective for curriculum learning.

## Limitations & Future Work

- Experiments are limited to ResNet18 and ImageNet-100, lacking validation on larger scales (full ImageNet-1k training) and larger architectures (such as ViT).
- Shape silhouettes rely on the pretrained TRACER segmentation model, introducing external data dependencies.
- The quality of intrinsic image decomposition from the Retinex algorithm is limited; utilizing more advanced decomposition methods could potentially yield better results.
- The epoch for stage transition (100 to 300) might not be the optimal partitioning, and sensitivity analysis regarding this hyperparameter is lacking.
- The additional preprocessing overhead introduced by the hybrid strategy (e.g., generating silhouettes, decomposing intrinsic images) may become a bottleneck on large-scale datasets.

## Related Work & Insights

- Relationship with **PCL (Prototypical Contrastive Learning)**: S-PCL shifts prototypical clustering from the semantic space to the shape space.
- Studies on **texture bias** indicate that most discriminative models remain heavily biased towards textures; the proposed shape bias method offers a self-supervised path to address this.
- Insight: Similar "staged/curriculum-based" inductive biases can be introduced into other self-supervised learning frameworks (such as MAE, DINO v2).

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The cognitive-science-inspired pre-pretraining strategy is highly novel. |
| Experimental Thoroughness | 3 | Limited to ResNet18 and ImageNet-100, which is relatively small in scale. |
| Writing Quality | 4 | Clear motivation and detailed experimental analysis. |
| Value | 3 | Requires additional preprocessing, limiting its general applicability. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale](you_see_it_you_got_it_learning_3d_creation_on_pose-free_videos_at_scale.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [\[ICLR 2026\] YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting](../../ICLR2026/3d_vision/yonosplat_you_only_need_one_model_for_feedforward_3d_gaussian_splatting.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](../../AAAI2026/3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[ECCV 2024\] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](../../ECCV2024/3d_vision/pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)

</div>

<!-- RELATED:END -->
