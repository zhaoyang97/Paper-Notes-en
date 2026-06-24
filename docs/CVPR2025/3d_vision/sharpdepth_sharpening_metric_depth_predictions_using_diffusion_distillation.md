---
title: >-
  [Paper Note] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation
description: >-
  [CVPR 2025][3D Vision][Monocular Depth Estimation] SharpDepth is proposed to inject fine-grained edge detail knowledge from generative depth models (e.g., Lotus) into the predictions of discriminative metric depth models (e.g., UniDepth) via diffusion distillation. By leveraging noise-aware gating and label-free training, it achieves an optimal balance between metric accuracy and edge sharpness.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Metric Depth"
  - "Diffusion Distillation"
  - "Edge Sharpening"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: 8de23b9bbea27661
---

# SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation

**Conference**: CVPR 2025  
**arXiv**: [2411.18229](https://arxiv.org/abs/2411.18229)  
**Code**: None  
**Area**: 3D Vision / Depth Estimation  
**Keywords**: Monocular Depth Estimation, Metric Depth, Diffusion Distillation, Edge Sharpening, Zero-shot Generalization

## TL;DR

SharpDepth is proposed to inject fine-grained edge detail knowledge from generative depth models (e.g., Lotus) into the predictions of discriminative metric depth models (e.g., UniDepth) via diffusion distillation. By leveraging noise-aware gating and label-free training, it achieves an optimal balance between metric accuracy and edge sharpness.

## Background & Motivation

- Zero-shot monocular metric depth estimation suffers from complementary drawbacks of two classes of methods: (1) **Discriminative methods** (e.g., UniDepth, Metric3D) are trained on sparse ground-truth annotations of real data, yielding high metric accuracy but overly smooth depth maps that lack edge details; (2) **Generative methods** (e.g., Marigold, Lotus) generate sharp edges leveraging diffusion model priors, but can only predict affine-invariant depth and suffer from the synthetic-to-real domain gap.
- Existing depth refinement methods (e.g., BetterDepth, PatchRefiner) rely on training on synthetic datasets, leading to domain gaps and limited generalization in real-world scenes.
- Core Problem: Is it possible to distill sharp details from generative models into metric depth models without using ground-truth annotations?
- SharpDepth requires only about 90,000 unlabeled real images for training, which is 1/100 to 1/150 of the training data size of existing methods.

## Method

### Overall Architecture

Given an input image, pretrained metric depth model $f_D$ (UniDepth) and diffusion depth model $f_G$ (Lotus) are used to predict metric depth $d$ and affine-invariant depth $\tilde{d}$, respectively. The normalized error map $e$ between them is calculated, where high-error regions indicate areas requiring sharpening. The SharpDepth model $\mathbf{G}_\theta$ is initialized based on the Lotus architecture, taking the depth latent variables processed by noise-aware gating and image latent variables as inputs.

### Key Designs

**Design 1: Noise-aware Gating**

- **Function**: Guides the diffusion model to focus precisely on uncertain regions that require sharpening.
- **Mechanism**: The discrepancy map $e$ between the metric depth and affine-invariant depth is computed to perform selective noise addition on the depth latent $z_d$: $z'_d = \hat{e} \odot \epsilon + (1-\hat{e}) \odot z_d$. High-discrepancy regions are covered with substantial noise (requiring reconstruction by the model), while low-discrepancy regions retain the original information (requiring no modification).
- **Design Motivation**: Unlike conventional diffusion models that uniformly add noise to all pixels, selective noise addition allows the model to differentiate between reliable and uncertain regions, concentrating computational resources on locations that need improvement the most. During training, an EMA model replaces $f_G$ to achieve iterative refinement.

**Design 2: SDS Depth Prior Distillation**

- **Function**: Injects the fine edge knowledge of pretrained diffusion depth models into the sharpener.
- **Mechanism**: The standard SDS loss is modified into an $x_0$-prediction formulation to match Lotus: $\nabla_\theta \mathcal{L}_{SDS} \triangleq \mathbb{E}_{t,\epsilon}[w^t(\hat{z} - f_G(\hat{z}^t; z_i, t))]$. By avoiding backpropagation through the diffusion model U-Net, the large diffusion depth prior is distilled into the sharpener.
- **Design Motivation**: Direct inference using diffusion models is slow and only provides relative depth; SDS distillation can efficiently extract high-frequency detail knowledge from their image priors during the training stage.

**Design 3: Noise-aware Reconstruction Loss**

- **Function**: Ensures that the sharpened depth maintains metric accuracy, preventing translation drift toward the affine-invariant depth.
- **Mechanism**: $\mathcal{L}_{recons} = \|e \odot (\hat{d} - d)\|$, weighted by the discrepancy map $e$, where high-discrepancy regions yield larger gradients and low-discrepancy regions have almost zero gradient. It acts as a regularization to keep the output close to the original metric depth.
- **Design Motivation**: Pure SDS distillation causes the network to inherit the shortcomings of diffusion models (e.g., metric inaccuracy); the reconstruction loss serves as an anchor to prevent the degradation of metric accuracy.

### Loss & Training

$$\mathcal{L}_{total} = \lambda_{SDS} \cdot \mathcal{L}_{SDS} + \lambda_{recons} \cdot \mathcal{L}_{recons}$$

Where $\lambda_{SDS} = 1.0$ and $\lambda_{recons} = 0.3$. Both losses focus on high-discrepancy regions through different mechanisms.

## Key Experimental Results

### Metric Depth Accuracy (Zero-Shot)

| Method | KITTI δ₁↑ | NYUv2 δ₁↑ | ETH3D δ₁↑ | nuScenes δ₁↑ |
|------|-----------|-----------|-----------|-------------|
| UniDepth | 0.98 | 0.98 | 0.25 | 0.84 |
| Metric3Dv2 | — | — | — | — |
| Lotus (Aligned to GT) | 0.88 | 0.97 | 0.96 | 0.51 |
| **SharpDepth** | **Close to UniDepth** | **Close to UniDepth** | **Significant Gain** | **Improved** |

### Depth Edge Quality (DBE/PDBE)

| Method | Sintel ↓ | UnrealStereo ↓ | Spring ↓ | iBims ↓ |
|------|----------|----------------|----------|---------|
| UniDepth | High | High | High | High |
| Lotus | Low | Low | Low | Low |
| **SharpDepth** | **Close to Lotus** | **Close to Lotus** | **Close to Lotus** | **Close to Lotus** |

### Key Findings

- SharpDepth achieves an optimal trade-off on the accuracy-sharpness curve (Fig. 2).
- Trained only on ~90K unlabeled real images, the training data size is approximately 1% of that used by discriminative methods.
- Noise-aware gating is significantly superior to uniform noise addition across all metrics.
- Using EMA self-iterative refinement performs better than using a fixed Lotus model.
- Point cloud reconstruction quality is significantly superior to UniDepth (e.g., fine-grained structures like hedgehog spines and keyboard keys).

## Highlights & Insights

1. **Ingenious Label-Free Depth Training Design**: By using the prediction discrepancy between two pretrained models as pseudo-supervision signals, the need for ground-truth annotations is completely eliminated.
2. **Deep Insight into Noise-aware Gating**: Altering the uniform noise addition of traditional diffusion models to region-adaptive noise addition provides a new paradigm for diffusion model refinement tasks.
3. **Complementary Design of SDS + Reconstruction Losses**: While one pursues sharpness, the other anchors accuracy, elegantly resolving the conflict between the two objectives.

## Limitations & Future Work

- It relies on two pretrained depth models (UniDepth + Lotus), increasing system complexity.
- The quality of the discrepancy map directly depends on the complementarity of the two baseline models; if both models make the same error, it cannot be detected.
- During inference, all three models (UniDepth, Lotus, and SharpDepth) must be run, which introduces higher latency.
- Lightweight sharpener architectures and end-to-end training schemes can be explored in future work.

## Related Work & Insights

- **UniDepth** [Piccinelli et al.] and **Metric3Dv2** [Hu et al.] are representative methods in metric depth.
- **Marigold** [Ke et al.] and **Lotus** [He et al.] are representative methods in diffusion-based depth estimation.
- Distillation techniques like **SDS** [Poole et al., DreamFusion] are transferred from the 3D generation domain to depth refinement.
- This work provides a valuable example of "enhancing discriminative models using generative models".

## Rating

⭐⭐⭐⭐ — The motivation of the method is clear and the design is elegant, with the label-free training scheme holding practical value. Noise-aware gating is the core innovation, and the experiments convincingly validate both accuracy and edge quality. However, requiring three models during inference increases deployment complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2026\] The Midas Touch for Metric Depth](../../CVPR2026/3d_vision/the_midas_touch_for_metric_depth.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[CVPR 2026\] Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors](../../CVPR2026/3d_vision/dense_metric_depth_completion_from_sparse_direct_time-of-flight_sensors.md)
- [\[ICLR 2026\] DepthLM: Metric Depth from Vision Language Models](../../ICLR2026/3d_vision/depthlm_metric_depth_from_vision_language_models.md)

</div>

<!-- RELATED:END -->
