---
title: >-
  [Paper Note] Efficient Rectified Flow for Image Fusion
description: >-
  [NeurIPS 2025][Image Generation][Image Fusion] This paper proposes RFfusion, which introduces Rectified Flow into image fusion for the first time, enabling training-free one-step sampling. A two-stage fusion-oriented VAE training strategy is further designed, achieving comprehensive superiority over existing diffusion-based fusion methods in both speed and quality.
tags:
  - NeurIPS 2025
  - Image Generation
  - Image Fusion
  - Rectified Flow
  - diffusion model
  - VAE
  - One-Step Sampling
date: 2026-05-08
content_hash: 6bf1fc3f19c5e7c0
---

# Efficient Rectified Flow for Image Fusion

**Conference**: NeurIPS 2025
**arXiv**: [2509.16549](https://arxiv.org/abs/2509.16549)
**Code**: [zirui0625/RFfusion](https://github.com/zirui0625/RFfusion)
**Area**: Image Generation
**Keywords**: Image Fusion, Rectified Flow, diffusion model, VAE, One-Step Sampling

## TL;DR

This paper proposes RFfusion, which introduces Rectified Flow into image fusion for the first time, enabling training-free one-step sampling. A two-stage fusion-oriented VAE training strategy is further designed, achieving comprehensive superiority over existing diffusion-based fusion methods in both speed and quality.

## Background & Motivation

Image fusion aims to merge images from different modalities (e.g., infrared/visible, multi-exposure, multi-focus) into a single image containing complementary information, with broad applications in object detection, medical diagnosis, and related fields. In recent years, diffusion models (DDPM) have achieved notable progress in image fusion by leveraging powerful generative priors. Methods such as DDFM and CCF inject fusion priors into the diffusion process via posterior sampling, effectively improving fusion quality.

However, the fundamental bottleneck of diffusion-based approaches lies in **extremely low inference efficiency**: DDFM requires hundreds of sampling steps, taking approximately 22 seconds per image, while CCF requires as long as 62 seconds. Reducing the number of sampling steps leads to a significant degradation in fusion quality. Existing distillation-based acceleration methods lack cross-task generalization, and latent diffusion methods based on VAEs perform poorly when applied directly, due to an inherent mismatch between the reconstruction objective used during training and the fusion objective. Accordingly, there is an urgent need for an efficient sampling approach that preserves fusion quality while maintaining generalization capability.

## Core Problem

1. **Sampling efficiency**: How can diffusion-based fusion models be compressed from hundreds of sampling steps to a single step without additional training?
2. **VAE objective mismatch**: Conventional VAEs optimize for pixel-level reconstruction, whereas fusion tasks require capturing cross-modal complementary semantic information. How can this gap be bridged?
3. **Multi-task generalization**: Can a single set of model parameters achieve strong performance across multiple tasks, including infrared-visible image fusion (IVIF), multi-exposure fusion (MEF), and multi-focus fusion (MFF)?

## Method

### Overall Architecture

RFfusion consists of two core components: a Rectified Flow-based one-step fusion network and a two-stage fusion-oriented VAE.

### 1. One-Step Fusion via Rectified Flow

Rectified Flow treats the forward process as a linear transformation between two data distributions, constructing straight-line trajectories via linear interpolation:

$$x_t = (1-t) x_0 + t \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$

The training objective is to learn a velocity field $v_\theta(x_t, t)$ that predicts the shortest path from noise to the real image. A key observation is that using the visible image rather than pure Gaussian noise as input yields better fusion results.

Drawing on the posterior sampling mechanism of DDFM, the fusion prior is injected into the sampling process through the velocity field:

$$v_\theta(f_t | i, v) = v_\theta(f_t) + \nabla_{f_t} \log p(i, v | f_t)$$

where the fusion prior guides generation by computing an observation loss between the fused image and the input images. Because Rectified Flow operates under an ODE framework (without stochastic noise injection), the sampling trajectory is deterministic and naturally suited for one-step inference.

### 2. Two-Stage Fusion-Oriented VAE Training

**Stage 1 — Frequency-Aware Reconstruction Training**: The VAE encoder and decoder are fine-tuned independently of the fusion process. A frequency-domain similarity loss $\mathcal{L}_{fre}$ is introduced, which applies FFT to convert images to the frequency domain and computes the spectral discrepancy between input and reconstructed images. The core motivation is that the complementary semantic information relevant to fusion is closely related to the high- and low-frequency components of images; the frequency loss guides the VAE to attend to fusion-relevant semantics during reconstruction.

**Stage 2 — Fusion-Adaptive Joint Training**: The VAE is integrated into the overall fusion framework for joint training, with only the decoder being fine-tuned. A fusion-specific loss function is employed:

$$\mathcal{L}_{fusion} = \lambda_{int}\mathcal{L}_{int} + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{grad}\mathcal{L}_{grad} + \lambda_{color}\mathcal{L}_{color} + \lambda_{mask}\mathcal{L}_{mask}$$

where $\mathcal{L}_{mask}$ is a newly proposed saliency mask loss that uses a saliency weight map to guide the network toward salient regions, enhancing the preservation of complementary information in the fused image.

## Key Experimental Results

### Inference Efficiency Comparison (V100 GPU, RoadScene Dataset)

| Method | Inference Time (s) | Params (M) | SF↑ | AG↑ |
|--------|--------------------|-----------|------|------|
| DDFM | 22.03 | 552.81 | 9.689 | 3.981 |
| CCF | 62.47 | 552.81 | 10.14 | 3.882 |
| Diff-IF | 2.457 | 23.47 | 13.90 | 5.179 |
| **RFfusion** | **0.308** | **65.57** | **14.00** | **5.218** |

Speed improvement of **71.5×** over DDFM and **202.8×** over CCF.

### Infrared-Visible Image Fusion (M3FD Dataset)

Compared to the baseline DDFM, MI improves by +0.449, VIF by +0.071, and SSIM by +0.047. Gains are even more pronounced on the TNO+RoadScene dataset: MI +1.150, VIF +0.398, SSIM +0.714.

### Multi-Task Generalization

Evaluated on IVIF, MEF, and MFF using a **single checkpoint** without any task-specific fine-tuning. MI reaches 6.528 (1st) on the MEFB dataset and CC reaches 0.977 (1st) on the MFIF dataset.

### Ablation Study

Applying the complete two-stage training strategy improves PSNR from 59.41 to 61.81; jointly using $\mathcal{L}_{fre}$ and $\mathcal{L}_{mask}$ yields the best overall performance.

## Highlights & Insights

1. **First application of Rectified Flow to image fusion**, enabling training-free one-step sampling with an inference time of only 0.308 seconds.
2. **Elegant two-stage VAE training strategy**: Stage 1 bridges the gap between reconstruction and fusion objectives via frequency loss; Stage 2 enhances the decoder's fusion adaptability through joint training.
3. **Strong multi-task generalization**: a single model achieves state-of-the-art performance across IVIF, MEF, and MFF tasks.
4. Parameter count of only 65.57M, substantially smaller than the 552.81M of DDFM/CCF.

## Limitations & Future Work

1. The method still relies on a Rectified Flow model pre-trained on generic image generation tasks rather than being purpose-built for fusion, which may constrain further improvements in fusion quality.
2. Stage 1 VAE training is conducted only on the LLVIP and MSRS infrared-visible datasets; adaptability to other modalities remains to be validated.
3. The paper does not discuss performance on high-resolution inputs or feasibility for practical deployment scenarios such as edge devices.
4. Future work could explore Rectified Flow models trained specifically for fusion tasks, rather than solely leveraging pre-trained weights.

## Related Work & Insights

| Dimension | DDFM | CCF | Diff-IF | RFfusion |
|-----------|------|-----|---------|----------|
| Sampling Steps | ~1000 | ~1000 | Multi-step | **1** |
| Additional Training Required | No | No | Yes | **No** |
| Latent Space Operation | No | No | No | **Yes** |
| Multi-Task Generalization | Limited | Moderate | Limited | **Strong** |
| Inference Speed | Slow | Very Slow | Moderate | **Fast** |

The key distinction is that RFfusion replaces the convoluted sampling trajectory of DDPM with the straight-line paths of Rectified Flow, eliminating multi-step sampling; the fusion-oriented VAE further reduces computational cost by operating in the latent space.

**Broader implications:**

1. **Rectified Flow as a general acceleration tool**: The core idea of linearizing sampling trajectories for one-step inference is transferable to other diffusion-based low-level vision tasks, such as image inpainting and super-resolution.
2. **Two-stage adaptation for objective mismatch**: The two-stage progressive adaptation strategy is a general and practical solution to the common problem of misalignment between VAE reconstruction objectives and downstream task objectives.
3. **Value of frequency-domain loss in fusion**: The correlation between frequency components and cross-modal complementary information provides a meaningful prior for loss design in fusion tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of Rectified Flow to fusion; the two-stage VAE design is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across three fusion task categories, multiple benchmark datasets, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure, detailed mathematical derivations, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Two-order-of-magnitude speedup with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)

<!-- RELATED:END -->
