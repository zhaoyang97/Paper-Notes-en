---
title: >-
  [Paper Note] ProtoDepth: Unsupervised Continual Depth Completion with Prototypes
description: >-
  [CVPR 2025][3D Vision][Continual Learning] ProtoDepth proposes a prototype-based continual learning method. By freezing the pre-trained model and learning a lightweight prototype set for each new domain to modulate latent features, it reduces the forgetting rate by over 50% in both indoor and outdoor scenarios.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Continual Learning"
  - "Depth Completion"
  - "Prototype Learning"
  - "Catastrophic Forgetting"
  - "Unsupervised"
date: 2026-05-08
content_hash: 4206293ca7e97b10
---

# ProtoDepth: Unsupervised Continual Depth Completion with Prototypes

**Conference**: CVPR 2025  
**arXiv**: [2503.12745](https://arxiv.org/abs/2503.12745)  
**Code**: [protodepth.github.io](https://protodepth.github.io/)  
**Area**: 3D Vision  
**Keywords**: Continual Learning, Depth Completion, Prototype Learning, Catastrophic Forgetting, Unsupervised

## TL;DR

ProtoDepth proposes a prototype-based continual learning method. By freezing the pre-trained model and learning a lightweight prototype set for each new domain to modulate latent features, it reduces the forgetting rate by over 50% in both indoor and outdoor scenarios.

## Background & Motivation

Depth completion (predicting dense depth maps from RGB images and sparse point clouds) is widely used in autonomous driving and robotics. The unsupervised learning paradigm (which does not require ground truth depth) naturally suits continual learning scenarios. However, when models are trained on a sequence of non-stationary distributed data, catastrophic forgetting occurs—previously learned knowledge degrades severely after training on a new domain.

Limitations of existing continual learning methods:
- **Regularization methods** (e.g., EWC): Mitigate forgetting by penalizing changes to important parameters, but have limited effectiveness under large domain shifts.
- **Replay methods**: Store past data for periodic retraining, but are limited by memory and privacy constraints.
- **Architecture methods**: Allocate task-specific subnetworks, but the parameter size can exceed that of the original model.

Key Insight: If all weights of the pre-trained model are frozen, **zero forgetting** can be guaranteed. The problem then becomes how to adapt to new domains using lightweight parameters. Unlike prompts in NLP, images lack a natural tokenization scale, making prompt-based methods applicable only to ViTs. The prototype-based method proposed in this paper is **architecture-agnostic**, working for both CNNs and Transformers.

## Method

### Overall Architecture

ProtoDepth freezes the pre-trained depth completion model $f_\theta$ and learns a prototype set for each newly encountered domain $\mathcal{D}_k$. The prototype set modulates the latent features to adapt to the new domain distribution via a global multiplicative bias and a local additive bias. For the domain-agnostic setting (where the domain identity is unknown at test time), a domain descriptor is additionally learned to automatically select the best-matching prototype set.

### Key Designs

**1. Global Prototype + Local Prototype**

- **Function**: The global prototype learns the transformation from the pre-trained data distribution to the new domain distribution; the local prototype captures fine-grained features, which can be selectively queried based on the input.
- **Mechanism**: The global prototype implements a multiplicative bias using $1 \times 1$ depthwise convolution to scale each channel of the latent features. The local prototype operates via a key-value mechanism: a frozen query calculates cosine similarity with the keys in the prototype set, and the corresponding values are select-weighted and injected into the latent features as an additive bias.
- **Design Motivation**: Domain shift can be modeled as a global distribution offset (e.g., systematic biases of different sensors) coupled with local content differences (e.g., geometric properties of different scenes). The combination of global and local components flexibly expresses various cross-domain differences.

**2. Domain Descriptor and Prototype Set Selection Mechanism**

- **Function**: Automatically selects the most appropriate prototype set for an input sample when the domain identity is unknown during testing.
- **Mechanism**: A descriptor vector is learned for each domain. During inference, the input sample is passed through the frozen encoder to extract a descriptor. The cosine similarity between this descriptor and all learned domain descriptors is calculated, and the prototype set corresponding to the best-matching domain is selected.
- **Design Motivation**: Unlike the domain-incremental setting that requires prior knowledge of domain identity, the domain-agnostic setting is closer to real-world scenarios. Descriptor learning is achieved through a contrastive loss—descriptors of samples from the same domain are close, while those from different domains are far apart.

**3. Training Objective Design**

- **Function**: Adds contrastive learning loss of domain descriptors on top of the unsupervised depth completion loss.
- **Mechanism**: The unsupervised loss is $\mathcal{L} = w_{ph}\ell_{ph} + w_{sz}\ell_{sz} + w_{sm}\ell_{sm}$ (photometric consistency + sparse depth consistency + local smoothness regularization). The domain descriptor is trained using an InfoNCE-like contrastive loss to cluster sample descriptors within the same domain.
- **Design Motivation**: The three unsupervised losses are standard SfM self-supervised signals, requiring no ground truth depth. The contrastive loss makes the descriptor space discriminative, supporting domain identification during inference.

### Loss & Training

$$\mathcal{L} = w_{ph}\ell_{ph} + w_{sz}\ell_{sz} + w_{sm}\ell_{sm} + \lambda_{desc}\mathcal{L}_{desc}$$

Where $\ell_{ph}$ combines L1 and SSIM to measure the photometric reconstruction error between adjacent frames, $\ell_{sz}$ measures the L1 distance between the predicted depth and the sparse point cloud, and $\ell_{sm}$ applies edge-aware smoothness constraints to the depth gradient.

## Key Experimental Results

### Main Results: Outdoor Sequence (KITTI $\rightarrow$ Waymo $\rightarrow$ VKITTI), VOICED Backbone

| Method | Avg Forgetting MAE (%) ↓ | Avg Forgetting RMSE (%) ↓ | Avg MAE (mm) ↓ | Avg RMSE (mm) ↓ |
|------|---------------|----------------|-------------|-------------|
| Finetuned | 8.828 | 6.131 | 63.352 | 125.28 |
| EWC | 9.439 | 8.014 | 63.787 | 126.706 |
| Replay | 6.154 | 4.688 | 64.305 | 126.714 |
| ProtoDepth-A | 2.439 | 3.598 | 56.971 | 118.132 |
| **ProtoDepth** | **0.000** | **0.000** | **56.359** | **115.153** |

### Ablation Study: Domain-Incremental vs Domain-Agnostic

| Setting | VOICED MAE Forgetting (%) | FusionNet MAE Forgetting (%) |
|------|-------------------|---------------------|
| ProtoDepth (Domain-incremental) | 0.000 | 0.000 |
| ProtoDepth-A (Domain-agnostic) | 2.439 | 1.282 |

### Key Findings

- Under the **domain-incremental** setting, ProtoDepth achieves **zero forgetting** (frozen model + independent prototype sets).
- Under the **domain-agnostic** setting, the forgetting rate is reduced by 52.2% (indoor) and 53.2% (outdoor), outperforming all baselines.
- Each domain only adds $<5\%$ of the original model's parameters, which is much lighter than existing architecture-based methods.
- The prototype-based method is architecture-agnostic, demonstrating effectiveness on both VOICED (CNN) and FusionNet (CNN + Sparse Conv).
- The selection accuracy of the domain descriptor is above 90%, which is key to the effectiveness of the domain-agnostic setting.

## Highlights & Insights

1. The **frozen + prototype paradigm** shifts continual learning from "how to update parameters safely" to "how to adapt to new domains with lightweight biases," fundamentally eliminating the forgetting problem.
2. The dual bias design of **global multiplicative + local additive** is intuitively clear: the global component captures systematic domain differences (e.g., sensor calibration), while the local component captures content-dependent feature variations.
3. ProtoDepth is the first to introduce the prototype-based method into unsupervised continual 3D reconstruction, expanding the application boundaries of prototype learning.

## Limitations & Future Work

- The domain descriptor needs to be trained jointly during training, which prevents fully online learning of new domains.
- The size and quantity of prototype sets need to be set manually and are not adaptively adjusted.
- Currently, it only considers the depth completion task and has not been extended to other 3D perception tasks (e.g., semantic understanding).
- The adaptation performance under extreme domain gaps (e.g., indoor $\rightarrow$ extreme weather outdoor) remains to be validated.

## Related Work & Insights

- **Relation to VPT/L2P**: Prompt-based methods concatenate learnable tokens to the ViT input, which is not applicable to CNNs. ProtoDepth's prototypes operate in the latent space and are architecture-agnostic.
- **Relation to LoRA**: LoRA fine-tunes weights via low-rank matrices, which can still cause minor forgetting. ProtoDepth freezes all weights to achieve zero forgetting.
- **Insight**: For continual learning scenarios, the philosophy of "keeping the model frozen and only adding lightweight adapters" is worth extending to more computer vision tasks.

## Rating

⭐⭐⭐⭐

The paper presents the first dedicated method for unsupervised continual depth completion. The prototype paradigm is elegantly designed and theoretically clear. The results of zero forgetting (domain-incremental) and >50% reduction in forgetting (domain-agnostic) are impressive. The parameter overhead is extremely low (<5% per domain). The limitations lie in the flexibility of domain descriptor learning and the extension of the task scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[ICLR 2026\] ORCaS: Unsupervised Depth Completion via Occluded Region Completion as Supervision](../../ICLR2026/3d_vision/orcas_unsupervised_depth_completion_via_occluded_region_completion_as_supervisio.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[ICCV 2025\] Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration](../../ICCV2025/3d_vision/omni-dc_highly_robust_depth_completion_with_multiresolution_depth_integration.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)

</div>

<!-- RELATED:END -->
