---
title: >-
  [Paper Note] Deeply Supervised Flow-Based Generative Models
description: >-
  [ICCV 2025][Image Generation][flow matching] DeepFlow introduces deep supervision and a VeRA (Velocity Refiner with Acceleration) module between Transformer layers of flow-based models…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "flow matching"
  - "deep supervision"
  - "velocity alignment"
  - "acceleration learning"
  - "training efficiency"
date: 2026-05-08
content_hash: c37131a56b73ce7d
---

# Deeply Supervised Flow-Based Generative Models

**Conference**: ICCV 2025

**arXiv**: [2503.14494](https://arxiv.org/abs/2503.14494)

**Area**: Diffusion Models / Image Generation

**Keywords**: flow matching, deep supervision, velocity alignment, acceleration learning, training efficiency

## TL;DR

DeepFlow introduces deep supervision and a VeRA (Velocity Refiner with Acceleration) module between Transformer layers of flow-based models, aligning intermediate-layer velocity features via second-order ODE dynamics. Without relying on any external pretrained model, it achieves an 8× training speedup and significant FID improvement.

## Background & Motivation

- **Limitations of flow-based models**: Mainstream flow-based models (e.g., SiT) learn velocity fields via linear interpolation but predict velocity only from the final layer output, **failing to leverage the rich feature representations in intermediate layers**, which leads to slow training convergence and limited representational capacity.
- **Shortcomings of external alignment methods**: Methods such as REPA improve training by aligning internal features with representations from external self-supervised models (DINO), but **rely entirely on external models** and overlook the self-correction potential of inter-layer features within flow-based models.
- **Core Problem**: Can flow-based models be improved by internally aligning velocity representations across Transformer layers, without depending on external models?

## Method

### Overall Architecture

DeepFlow builds upon the SiT/DiT architecture and introduces three core designs:

1. **Branch partitioning**: Transformer blocks are evenly divided into $k$ branches (e.g., 2T denotes 2 branches), each with a velocity prediction head appended at its end.
2. **Deep supervision**: Each branch independently predicts velocity and is trained with distinct timestep conditioning.
3. **VeRA module**: A lightweight velocity refinement module is inserted between adjacent branches to explicitly align intermediate velocity features.

### Key Designs

**Deep Supervision**: Transformer blocks are evenly split into $k$ branches, each conditioned on an independent timestep and supervised via its own velocity head. The loss is a weighted multi-branch MSE over velocity predictions, with lower weights (e.g., 0.2) for intermediate branches and 1.0 for the final branch.

**VeRA Module** consists of three sub-modules:

1. **Acceleration Learning (ACC MLP)**: A lightweight MLP generates acceleration features from the velocity features of the preceding branch, trained using second-order ODE dynamics. The core formulation is a second-order Taylor expansion: position + velocity × time + 0.5 × acceleration × time², with the objective of reconstructing the clean image.
2. **Time-interval conditioning**: Velocity and acceleration features are concatenated and modulated via time-interval-conditioned AdaLN-Zero, enabling the features to be aware of the temporal gap between adjacent branches.
3. **Cross-spatial attention**: Cross-attention is applied between the modulated velocity feature space and the original patchified image space to integrate spatial information.

### Loss & Training

Total loss = deep supervision velocity loss + $\lambda$ × acceleration loss (second-order ODE reconstruction error). During inference, all branches share a unified timestep.

## Key Experimental Results

### Main Results

| Model | Epoch | SSL | FID | sFID | IS |
|------|-------|-----|------|-------|-----|
| SiT-B/2 | 80 | None | 29.7 | 6.2 | 51.0 |
| **DeepFlow-B/2-2T** | **80** | **None** | **23.1** | **5.6** | **60.3** |
| SiT-XL/2 | 800 | None | 9.8 | 7.3 | 128.2 |
| **DeepFlow-XL/2-3T** | **400** | **None** | **7.2** | **5.1** | **138.5** |
| SiT-XL/2 + REPA | 800 | DINOv2 | 5.7 | 6.4 | 171.0 |
| **DeepFlow-XL/2-3T + SSL** | **400** | **DINOv2** | **5.0** | **5.2** | **162.0** |

- Achieves **FID 1.77** on ImageNet-256 (400 epochs + SSL + CFG), surpassing SiT-XL's 1.80 (800 epochs).
- Achieves **FID 1.96** on ImageNet-512 (200 epochs + SSL), surpassing SiT-XL + SSL's 2.08.

### Ablation Study

| Component | FID |
|------|------|
| SiT-B/2 baseline | 34.4 |
| + Deep supervision | 33.0 |
| + Time-interval conditioning | 31.1 |
| + Inter-layer acceleration learning | 29.9 |
| + Cross-spatial attention | **28.1** |

**Key Findings**:

- Deep supervision alone reduces the intermediate/final layer feature distance from 7.7 to 7.2; adding VeRA further reduces it to 2.9.
- DeepFlow-B without SSL alignment achieves performance comparable to SiT-B aligned with DINOv1.
- On text-to-image tasks, DeepFlow consistently outperforms SiT across FID, FDD, IS, CLIP score, and GenEval.

## Highlights & Insights

1. **Internal alignment as a substitute for external alignment**: This work is the first to demonstrate that flow-based models can achieve alignment quality comparable to external DINO-based methods through internal inter-layer velocity alignment, reducing dependence on pretrained models.
2. **Second-order dynamics perspective**: Inter-layer feature refinement is formulated as a physical acceleration problem, where the second-order ODE from velocity to acceleration provides an elegant theoretical framework.
3. **8× training speedup**: The substantial improvement in training efficiency at comparable performance levels has significant practical value for large-scale training.
4. **Lightweight design**: The VeRA module introduces minimal additional parameters (only 6M for DeepFlow-XL: 681M vs. 675M), incurring negligible inference overhead.
5. **Complementarity with external alignment**: DeepFlow + REPA (DINOv2) yields further performance gains, indicating that internal and external alignment are orthogonal and mutually complementary.

## Limitations & Future Work

- Ablation studies are conducted primarily at the Base scale, with limited component-level analysis at XL scale.
- The optimal value of the time-interval hyperparameter requires adjustment depending on model scale.
- Inference still relies on a unified timestep; multi-branch multi-timestep inference strategies remain unexplored.
- There is no theoretical guidance for selecting the number of branches $k$.

## Related Work & Insights

- **SiT** [Ma et al., 2024]: Flow matching + DiT; the direct baseline for DeepFlow.
- **REPA** [Yu et al., 2024]: Aligns internal features with external self-supervised models; complementary to DeepFlow.
- **DiT** [Peebles and Xie, 2023]: Transformer-based diffusion model; a foundational work in this line of research.
- **Deep Supervision** [Lee et al., 2015]: Multi-layer supervision strategy originally proposed for discriminative tasks.

## Rating

| Dimension | Score |
|------|------|
| Novelty | 4/5 |
| Effectiveness | 5/5 |
| Practicality | 5/5 |
| Writing Quality | 4/5 |
| Overall | 4.5/5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](../../NeurIPS2025/image_generation/gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)

</div>

<!-- RELATED:END -->
