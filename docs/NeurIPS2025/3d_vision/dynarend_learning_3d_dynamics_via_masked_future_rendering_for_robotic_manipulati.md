---
title: >-
  [Paper Note] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation
description: >-
  [NeurIPS 2025][3D Vision][Representation Learning] DynaRend is proposed to jointly learn 3D geometry, semantics, and dynamics on triplane representations via differentiable volumetric rendering…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Representation Learning"
  - "Robotic Manipulation"
  - "Differentiable Rendering"
  - "Triplane"
  - "Dynamics Prediction"
  - "Pre-training"
date: 2026-05-08
content_hash: 215681ad08c561c0
---

# DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation

**Conference**: NeurIPS 2025
**arXiv**: [2510.24261](https://arxiv.org/abs/2510.24261)  
**Code**: Not released  
**Area**: 3D Vision
**Keywords**: Representation Learning, Robotic Manipulation, Differentiable Rendering, Triplane, Dynamics Prediction, Pre-training

## TL;DR

DynaRend is proposed to jointly learn 3D geometry, semantics, and dynamics on triplane representations via differentiable volumetric rendering, using two complementary objectives — masked reconstruction and future prediction — enabling efficient transfer to downstream robotic manipulation tasks after pre-training.

## Background & Motivation

Learning generalizable robotic manipulation policies faces a fundamental bottleneck: the scarcity of real-world training data. Existing self-supervised pre-training methods exhibit three categories of limitations:

1. **2D visual pre-training** (e.g., MAE, contrastive learning): captures only static semantics, lacking 3D geometric understanding and dynamics modeling.
2. **Video prediction models** (e.g., VPP, VidMan): leverage large-scale video to learn 2D dynamics but lack explicit 3D spatial awareness.
3. **Explicit 3D dynamics** (e.g., ManiGaussian): use dynamic Gaussians to capture spatial and dynamic information, but suffer from high representational complexity, difficulty integrating with downstream policies, and heavy reliance on densely calibrated novel-view supervision.

The core idea is to use **differentiable volumetric rendering** as a bridge to simultaneously learn geometry (via RGB + depth), semantics (via visual foundation model distillation), and dynamics (via future frame prediction) on a compact triplane representation, unifying all three within a single pre-training framework.

## Core Problem

How to design a representation learning framework that jointly learns 3D-aware geometric, semantic, and dynamic representations without requiring extensive annotations or dense viewpoints, while enabling efficient transfer to robotic manipulation.

## Method

### 3D Scene Triplane Representation

A point cloud is reconstructed from multi-view RGB-D images, encoded by an MLP, and projected onto three orthogonal planes via axis-aligned max pooling:

$$\mathbf{f}_{xy} \in \mathbb{R}^{H \times W \times C}, \quad \mathbf{f}_{xz} \in \mathbb{R}^{H \times D \times C}, \quad \mathbf{f}_{yz} \in \mathbb{R}^{W \times D \times C}$$

The triplane resolution is set to $16 \times 16 \times 16$, balancing efficiency and expressiveness.

### Masked Future Prediction

A random subset of triplane features is masked and replaced with learnable embeddings. Combined with CLIP-encoded language instructions, the masked triplane is processed sequentially by:

1. **Reconstruction network** $\mathcal{E}_{\text{recon}}$: recovers the complete 3D representation of the current scene $\mathcal{V}_{\text{now}}$.
2. **Prediction network** $\mathcal{E}_{\text{pred}}$: predicts the 3D representation of the nearest future keyframe $\mathcal{V}_{\text{future}}$.

Both networks consist of 4-layer Transformers with SwiGLU, QK Norm, and RoPE.

### Differentiable Volumetric Rendering Supervision

Current and future triplanes are rendered independently. $N$ points are sampled along each ray; features $\mathbf{v}_i$ are queried via bilinear interpolation on the three planes and decoded by MLP heads into density $\sigma$, RGB $\mathbf{c}$, and semantic features $\mathbf{s}$:

$$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}(\mathbf{v}_i, \mathbf{d}), \quad \hat{\mathbf{D}}(\mathbf{r}) = \sum_{i=1}^N w_i t_i$$

where $w_i = T_i(1 - \exp(\sigma(\mathbf{v}_i)\delta_i))$ and $T_i = \exp(-\sum_{j=1}^{i-1} \sigma(\mathbf{v}_j)\delta_j)$.

Semantic features are supervised using features extracted by RADIOv2.5; depth is optimized with SiLog loss.

### View Augmentation

The pre-trained generative model See3D is employed to synthesize novel-view RGB-D pairs from existing views as additional supervision, addressing the limited camera viewpoints in real-world scenarios.

### Pre-training and Fine-tuning Losses

Pre-training loss: $\mathcal{L}_{\text{pretrain}} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{pred}} \mathcal{L}_{\text{pred}}$

During fine-tuning, triplane features are processed via convolution and upsampling to generate action heatmaps, supervised with cross-entropy over translation, rotation, and gripper state:

$$\mathcal{L}_{\text{finetune}} = \lambda_{\text{trans}} \text{CE}(\mathbf{a}_{\text{trans}}, \hat{\mathbf{a}}_{\text{trans}}) + \lambda_{\text{rot}} \text{CE}(\mathbf{a}_{\text{rot}}, \hat{\mathbf{a}}_{\text{rot}}) + \lambda_{\text{gripper}} \text{CE}(\mathbf{a}_{\text{gripper}}, \hat{\mathbf{a}}_{\text{gripper}})$$

## Key Experimental Results

### RLBench 18-Task Benchmark

| Method | Avg. Success Rate↑ | Avg. Rank↓ | Inference Speed |
|---|---|---|---|
| PerAct | 49.4 | 5.1 | 4.9 FPS |
| RVT | 62.9 | 4.3 | 11.6 FPS |
| 3D-MVP | 67.5 | 3.2 | 11.6 FPS |
| 3D Diffuser Actor | 81.3 | 2.2 | 1.4 FPS |
| RVT-2 | 81.4 | 2.2 | 20.6 FPS |
| **DynaRend** | **83.2** | **1.5** | **19.6 FPS** |

Compared to the RVT baseline, the success rate improves by 32.3% (62.9→83.2).

### Colosseum Generalization Benchmark

DynaRend demonstrates superior robustness over baselines across 12 types of environmental perturbations (color, texture, size, and lighting variations).

### Real-World Experiments

Effectiveness and practical utility are validated across 5 real-world manipulation tasks.

### Ablation Study

- Removing the prediction objective (reconstruction only) leads to a significant drop in success rate, confirming the importance of dynamics modeling.
- Removing the masking strategy degrades performance; masked reconstruction compels the model to learn more complete 3D representations.
- View augmentation is critical for real-world deployment.

## Highlights & Insights

1. **Unified three-in-one pre-training**: geometry, semantics, and dynamics are jointly captured within a volumetric rendering supervision framework in an elegant design.
2. **Lightweight and efficient**: the triplane representation is compact, with an inference speed of 19.6 FPS approaching the fastest method, RVT-2.
3. **View synthesis augmentation**: a generative model is used to synthesize novel views, addressing the limited calibrated viewpoints in real-world settings.
4. Pre-training requires only ~60k steps and fine-tuning ~30k steps, making the training cost manageable.

## Limitations & Future Work

- The fixed triplane resolution of $16^3$ may be insufficient for fine-grained manipulation tasks.
- Pre-training requires multi-view RGB-D input; single-arm camera setups necessitate additional depth sensors.
- The impact of pseudo-label quality introduced by See3D view augmentation on final performance warrants further analysis.
- Evaluation is limited to the keyframe manipulation paradigm; continuous control scenarios are not addressed.

## Related Work & Insights

- **vs. 3D-MVP**: 3D-MVP employs MAE for 3D reconstruction pre-training but lacks dynamics modeling; DynaRend adds a future prediction branch.
- **vs. VPP/VidMan**: video diffusion models model only 2D dynamics, whereas DynaRend models dynamics in 3D space.
- **vs. ManiGaussian**: explicit 3D Gaussian representations are complex and require dense novel-view supervision; DynaRend uses a compact triplane representation and supports view synthesis augmentation.
- **vs. GNFactor/SPA**: these methods focus solely on static 3D consistency pre-training without dynamic prediction.

The concept of *rendering as supervision* is particularly inspiring — explicit 3D annotations are not required; 3D representations can be learned by comparing rendered RGB, depth, and semantics against 2D observations. The triplane representation achieves a favorable balance between efficiency and expressiveness for robotic manipulation and merits further exploration.

## Rating

- ⭐ Novelty: 8/10 — First robotic pre-training framework to jointly perform masked reconstruction, future prediction, and volumetric rendering on triplanes.
- ⭐ Experimental Thoroughness: 9/10 — Covers RLBench 18+71 tasks, Colosseum generalization, real-world experiments, and detailed ablations.
- ⭐ Writing Quality: 8/10 — Clear motivation, intuitive architecture diagrams, and comprehensive comparisons.
- ⭐ Value: 8/10 — Meaningful contribution to the intersection of 3D representation learning and robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation](enerverse_envisioning_embodied_future_space_for_robotics_manipulation.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](../../CVPR2026/3d_vision/hyperbolic_multiview_pretraining_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
