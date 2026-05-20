---
title: >-
  [Paper Note] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection
description: >-
  [CVPR 2026][3D Vision][Weakly-supervised monocular 3D detection] This paper proposes VirPro—an adaptive multimodal pre-training paradigm that provides scene-aware semantic supervision signals for weakly-supervised monocu…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Weakly-supervised monocular 3D detection"
  - "probabilistic prompt learning"
  - "multimodal pre-training"
  - "visual-language alignment"
  - "CLIP"
date: 2026-05-08
content_hash: 6fff74583fcee1b9
---

# VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection

**Conference**: CVPR 2026
**arXiv**: [2603.17470](https://arxiv.org/abs/2603.17470)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Weakly-supervised monocular 3D detection, probabilistic prompt learning, multimodal pre-training, visual-language alignment, CLIP

## TL;DR

This paper proposes VirPro—an adaptive multimodal pre-training paradigm that provides scene-aware semantic supervision signals for weakly-supervised monocular 3D detection via visually guided probabilistic prompts (Adaptive Prompt Bank + Multi-Gaussian Prompt Modeling). VirPro can be seamlessly integrated into existing WS-M3D frameworks, achieving up to 4.8% AP improvement on KITTI.

## Background & Motivation

Monocular 3D object detection relies heavily on expensive 3D annotations due to the absence of explicit depth information. Existing weakly-supervised methods follow three main directions:

1. **Pseudo 3D label generation**: Aligning 2D bounding boxes with LiDAR point clouds to generate 3D pseudo-labels
2. **3D knowledge distillation**: Transferring knowledge from strong models to monocular detectors
3. **Text-visual alignment**: Borrowing ideas from CLIP to use deterministic text descriptions as auxiliary weak supervision signals

Representative methods such as CAW3D adopt **hand-crafted static text prompts** (e.g., "a photo of a car") as weak supervision. However, such **deterministic, scene-agnostic** text descriptions fail to capture the visual diversity of object appearance and spatial layout across different scenes, limiting the model's ability to learn scene-aware representations.

**Core Insight**: If prompts can adaptively reflect visual diversity across scenes, more robust scene-aware representations can be achieved without requiring additional manual annotations.

## Core Problem

How to design prompt supervision signals that embrace cross-scene visual diversity, enabling robust scene-aware representations without additional manual annotations?

## Method

VirPro adopts a **two-stage training pipeline**: Stage 1 performs pre-training of probabilistic prompts and visual-text alignment; Stage 2 transfers the learned scene-aware priors to the monocular encoder via knowledge distillation.

### 3.1 Adaptive Prompt Bank (APB)

**Design Motivation**: Relying solely on visual features and a single category prompt is insufficient to model diverse scene contexts in weakly-supervised monocular 3D detection. Multiple diverse prompts provide complementary semantic cues that enhance language-visual alignment.

**Design**: For the $i$-th target query token $o_i$, $N_p$ probabilistic prompt templates are generated:

$$p_i^t = \{a_1^t, a_2^t, \ldots, a_L^t \mid o_i\}, \quad t = 1, \ldots, N_p$$

where $\{a_1^t, \ldots, a_L^t\}$ are $L$ **learnable scenario descriptors**, randomly initialized and jointly optimized during training.

**Key Design—Random Position Insertion Strategy**: Unlike ProDA, which fixes the target token position (beginning/middle/end), VirPro allows target-related tokens to be **randomly placed** within the template, encouraging the model to capture more robust contextual associations—particularly critical under weak supervision.

In practice, 32 learnable prompts are initialized per RoI, and 8 are randomly sampled and normalized to form RoI-specific text embeddings.

### 3.2 Multi-Gaussian Prompt Modeling (MGPM)

This is the core module of VirPro, which models each scene prompt as an **independent isotropic Gaussian distribution**, enabling semantic diversity and structured disentanglement.

**Probabilistic Modeling**: For the $i$-th target and its $N_p$ scene prompts, the distribution is defined as:

$$\mathcal{P}(z_i^{(1:N_p)} \mid p_i) \sim \left\{\mathcal{N}\left(\boldsymbol{\mu}_i^{(t)}, (\boldsymbol{\sigma}_i^{(t)})^2 \mathbf{I}\right)\right\}_{t=1}^{N_p}$$

**Dual-Decoder Parameter Estimation**:

| Component | Function | Computation | Input Source |
|-----------|----------|-------------|--------------|
| Textual Prompt Decoder | Estimates Gaussian mean $\boldsymbol{\mu}$ | $\mu_i^t = \phi_\mu(q_i^t) + \text{SelfAttn}_\mu(q_i^t; P_i)$ | Self-attention within the prompt set |
| Cross-Modal Visual-Text Decoder | Estimates Gaussian variance $\boldsymbol{\sigma}$ | $\sigma_i^t = \phi_\sigma(q_i^t) + \text{CrossAttn}_\sigma(q_i^t; F)$ | Cross-attention over visual-language features $F$ |

**Core Idea**: The mean is produced by text-side self-attention, capturing canonical semantics; the variance is injected from visual features via cross-attention, expressing **visual uncertainty**. This allows prompts to maintain stability in category semantics while adapting to scene-level visual variation.

**Stochastic Sampling and Reparameterization**: For each scene $t$, $N_s$ random samples are drawn from the learned distribution:

$$z_{i,j}^{(t)} \sim \mathcal{N}\left(\boldsymbol{\mu}_i^{(t)}, (\boldsymbol{\sigma}_i^{(t)})^2 \mathbf{I}\right), \quad j = 1, \ldots, N_s$$

The reparameterization trick is used to ensure end-to-end differentiability:

$$\hat{z}_{i,j}^{(t)} = \boldsymbol{\mu}_i^{(t)} + \boldsymbol{\sigma}_i^{(t)} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 3.3 RoI Contrastive Matching

Object-level image-text contrastive learning is employed to ensure all targets within the same scene share a consistent global context while remaining distinguishable from targets in different scenes.

- **Text embedding** $\mathbf{e}_i^{\text{txt}}$: Obtained via **max pooling** over sampled prompt distributions $\hat{z}_{i,j}^{(t)}$
- **Image embedding** $\mathbf{e}_i^{\text{img}}$: Extracted from the monocular 3D encoder, spatially aligned with the 2D detector
- Positive pairs: $(\mathbf{e}_i^{\text{txt}}, \mathbf{e}_i^{\text{img}})$ for the same target

Contrastive loss:

$$\mathcal{L}_{\text{contrast}} = \frac{1}{N} \sum_{i=1}^{N} \ell_i$$

Four RoIs are randomly selected per scene to construct contrastive pairs, with temperature parameter initialized at $\tau = 0.07$.

### 3.4 Learning Objectives

The **probabilistic prompt learning loss** consists of two components:

1. **Diversity loss**—encourages semantic differentiation among scene prompts via orthogonality:
$$\mathcal{L}_{\text{div}} = \frac{1}{K} \sum_{i=1}^{K} \|\tilde{P}_i \tilde{P}_i^\top - \mathbf{I}\|_2^2$$

2. **KL divergence regularization**—prevents variance collapse by constraining prompt distributions toward a standard Gaussian prior:
$$\mathcal{L}_{\text{prompt}} = \mathcal{L}_{\text{div}} + \frac{1}{N_p} \sum_{t=1}^{N_p} \text{KL}\left(\mathcal{P}(\hat{\boldsymbol{z}}_i^{(t)} \mid p_i^{(t)}) \| \mathcal{N}(\mathbf{0}, \mathbf{I})\right)$$

**Two-stage losses**:

| Stage | Loss Function | Description |
|-------|---------------|-------------|
| Stage 1 | $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{contrast}} + \alpha \mathcal{L}_{\text{prompt}}$ | Probabilistic prompt learning + RoI contrastive alignment |
| Stage 2 | $\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{mse}} + \lambda \mathcal{L}_{3D}$ | Knowledge distillation (MSE) + pseudo-label 3D supervision |

Stage 2 adopts the Dual-to-One Distillation (D2OD) from CAW3D, introducing no additional inference overhead.

### Overall Pipeline Summary

1. **APB stage**: Multiple learnable prompt templates are generated per RoI, with target tokens randomly inserted
2. **MGPM stage**: The text decoder estimates Gaussian means; the visual-text cross-modal decoder estimates variances; diverse prompt embeddings are generated via sampling
3. **RoI contrastive matching**: Max-pooled aggregations are used for object-level contrastive learning, reinforcing intra-scene consistency and inter-scene discriminability
4. **Knowledge distillation**: Scene-aware priors learned during pre-training are distilled into the monocular encoder

## Key Experimental Results

### KITTI Val Set (Car Category, AP @ IoU=0.5, $R_{40}$)

| Method | Supervision | $\text{AP}_{\text{BEV}}$ Easy | $\text{AP}_{\text{BEV}}$ Mod | $\text{AP}_{\text{BEV}}$ Hard | $\text{AP}_{\text{3D}}$ Easy | $\text{AP}_{\text{3D}}$ Mod | $\text{AP}_{\text{3D}}$ Hard |
|--------|-------------|------|------|------|------|------|------|
| WeakM3D | Weak (w/o 2D GT) | 58.20 | 38.02 | 30.17 | 50.16 | 29.94 | 23.11 |
| **VirPro+WeakM3D** | - | **55.09** | **38.76** | **31.12** | **50.97** | **31.95** | **24.27** |
| GGA+PGD | Weak (w/ 2D GT) | 57.20 | 40.11 | 34.96 | 51.48 | 35.73 | 30.49 |
| **VirPro+GGA+PGD** | - | **60.11** | **42.95** | **37.50** | **54.72** | **39.49** | **33.32** |

VirPro+GGA+PGD outperforms GGA+PGD by **+3.76 $\text{AP}_{\text{3D}}$** on Moderate and **+2.83 $\text{AP}_{\text{3D}}$** on Hard.

### KITTI Test Set (Car Category)

| Method | $\text{AP}_{\text{BEV}}$ Easy | Mod | Hard | $\text{AP}_{\text{3D}}$ Easy | Mod | Hard |
|--------|------|------|------|------|------|------|
| WeakM3D | 11.82 | 5.66 | 4.08 | 5.03 | 2.26 | 1.63 |
| **VirPro+WeakM3D** | **12.23** | **5.92** | **4.33** | **5.41** | **2.52** | **1.81** |
| GGA+PGD | 14.87 | 9.26 | 7.09 | 7.09 | 4.27 | 3.26 |
| **VirPro+GGA+PGD** | **15.59** | **9.58** | **7.29** | **7.95** | **4.96** | **3.64** |

### Ablation Study

- **Prompt design**: Multi-probabilistic prompts (M.P.P) > single probabilistic prompt (S.P.P) > hand-crafted prompts (H.C.P)
- **Prompt aggregation strategy**: Max pooling significantly outperforms MLP / Concat+MLP / Add, leading by 1.15+ in $\text{AP}_{\text{3D}}$ Hard
- **Image-text fusion strategy**: Cross-attention achieves best performance ($\text{AP}_{\text{3D}}$ Hard: 25.05), substantially outperforming Add (22.37) and Concat (21.88)
- **Latent space structure**: VirPro achieves higher Calinski-Harabasz and Silhouette scores than CAW3D, indicating more compact intra-scene and more separable inter-scene RoI embeddings

## Highlights & Insights

1. **Plug-and-play**: VirPro, as a pre-training paradigm, can be seamlessly integrated into multiple WS-M3D frameworks (WeakM3D, GGA+PGD, etc.) without additional inference overhead
2. **Probabilistic modeling of visual uncertainty**: The disentangled design—mean capturing canonical semantics and variance encoding visual uncertainty—is elegant and principled
3. **Simplicity of max pooling**: The parameter-free max pooling over probabilistic prompts outperforms complex MLP-based fusion, reflecting a "less is more" design philosophy
4. **Latent space visualization**: Inter-scene centroid distance distributions and clustering metrics quantitatively demonstrate the improved latent space structure resulting from probabilistic prompts

## Limitations & Future Work

1. **RoI quality bottleneck**: Probabilistic prompt quality is constrained by the accuracy of the 2D detector; inaccurate 2D detections introduce biased visual cues
2. **Rectangular box assumption**: RoI feature extraction using rectangular crops inevitably introduces background noise, as real objects are rarely perfectly rectangular
3. **Fixed resolution constraint**: RoI feature extraction is subject to fixed image resolution and predefined cropping strategies, limiting cross-domain robustness
4. **Evaluation limited to KITTI**: Experiments are conducted solely on KITTI; generalization to larger-scale datasets such as nuScenes remains unverified
5. **Computational cost**: The two-stage training requires 25 epochs of Stage 1 pre-training, incurring higher training costs compared to end-to-end approaches

## Related Work & Insights

- **vs. CAW3D**: CAW3D employs hand-crafted static prompts; VirPro replaces them with learnable probabilistic prompts that provide richer scene-aware semantics
- **vs. ProDA**: ProDA first models prompts as multivariate Gaussians in the output space for zero-shot classification; VirPro focuses on RoI-level individualized modeling tailored for weakly-supervised 3D detection
- **vs. APP**: APP models prompt uncertainty in the input space, constrained by the sparsity of natural language; VirPro operates in the output space with injected visual features
- **vs. GGA**: GGA uses static text prompts generated by LLMs; VirPro's visually guided probabilistic prompts offer greater adaptability

The probabilistic prompt modeling approach is generalizable to other weakly-supervised visual tasks (e.g., weakly-supervised semantic and instance segmentation). The disentanglement of "mean = semantics + variance = visual uncertainty" has broad applicability in multimodal learning. The scene-aware contrastive learning design is transferable to other 3D perception tasks in autonomous driving.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The design of probabilistic prompt modeling with visually guided variance is novel; introducing probabilistic prompt learning into weakly-supervised 3D detection is a first
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablation studies are thorough, but evaluation is limited to KITTI; validation on nuScenes and Waymo is absent
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, figures are intuitive, and the overall logic is coherent
- **Value**: ⭐⭐⭐⭐ — The plug-and-play pre-training paradigm has strong practical utility, though its impact is limited by the relatively niche area of weakly-supervised 3D detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[AAAI 2026\] VPN: Visual Prompt Navigation](../../AAAI2026/3d_vision/vpn_visual_prompt_navigation.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)

</div>

<!-- RELATED:END -->
