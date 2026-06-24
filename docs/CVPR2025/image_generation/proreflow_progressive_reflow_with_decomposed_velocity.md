---
title: >-
  [Paper Note] ProReflow: Progressive Reflow with Decomposed Velocity
description: >-
  [CVPR 2025][Image Generation][Flow Matching] This paper proposes progressive Reflow (gradually straightening diffusion trajectories from multi-window to few-window) and aligned v-prediction (prioritizing direction over magnitude in velocity matching), enabling SDv1.5 to achieve generation quality close to 32-step DDIM with only 4-step sampling.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Progressive Reflow"
  - "Velocity Decomposition"
  - "Few-step Generation"
  - "Diffusion Acceleration"
date: 2026-05-08
content_hash: 2dddc870d1dc7290
---

# ProReflow: Progressive Reflow with Decomposed Velocity

**Conference**: CVPR 2025  
**arXiv**: [2503.04824](https://arxiv.org/abs/2503.04824)  
**Code**: [GitHub](https://github.com/)  
**Area**: Image Generation/Diffusion Model Acceleration  
**Keywords**: Flow Matching, Progressive Reflow, Velocity Decomposition, Few-step Generation, Diffusion Acceleration

## TL;DR

This paper proposes progressive Reflow (gradually straightening diffusion trajectories from multi-window to few-window) and aligned v-prediction (prioritizing direction over magnitude in velocity matching), enabling SDv1.5 to achieve generation quality close to 32-step DDIM with only 4-step sampling.

## Background & Motivation

Flow matching enables few-step or even single-step generation by straightening the sampling trajectories of pre-trained diffusion models. The standard Reflow training pipeline directly trains models to predict consistent velocities across all timesteps. However, this strategy does not fully exploit the potential of rectified flow:

- **Velocity discrepancy is too large**: Pre-trained diffusion models exhibit significant velocity discrepancies across different timesteps (validated by L2 distance and cosine similarity). Eliminating all discrepancies directly presents an optimization challenge.
- **Direction is more crucial than magnitude**: Velocity can be decomposed into direction and magnitude. For trajectory "straightness", direction matching is more critical than magnitude matching, yet the MSE loss in existing methods treats both equally.
- Although PeRFlow proposes piecewise straightening, it attempts to approximate complex trajectories into piecewise linear segments within a single stage, which still presents a difficult optimization target.

Key Observation: Pre-trained models exhibit high velocity similarity at adjacent timesteps, which provides the foundation for a curriculum learning strategy that first straightens trajectories within local windows and then progressively merges these windows.

## Method

### Overall Architecture

ProReflow is based on the piecewise Reflow framework of PeRFlow, introducing two key improvements: progressive Reflow (multi-stage reduction from multi-window to few-window) and aligned v-prediction (incorporating a direction-matching loss into the MSE loss).

### Key Designs

#### Key Design 1: Progressive Reflow

- **Function**: Alleviates the optimization difficulty of Reflow through multi-stage curriculum learning.
- **Mechanism**: The diffusion process is first divided into $N$ local time windows to be straightened. Subsequently, adjacent windows are merged into a larger window via Cross-windows Reflow, progressively scaling from $N \to N/2 \to N/4 \to \ldots$ until the target number of windows is reached. For instance, the original trajectory is first approximated using 8 windows (easy), then merged to 4 windows (medium), and ultimately achieves few-step generation.
- **Design Motivation**: Directly transitioning from complex trajectories to a 4-segment linear approximation has a smaller "learning rate" $\beta$ (difficult). Introducing intermediate representations (8 windows) ensures smaller spans and more stable optimization across stages, akin to introducing intermediate models in knowledge distillation.

The core of Cross-windows Reflow is to guide the two trajectory segments of adjacent windows $[t_{k-1}, t_k]$ and $[t_k, t_{k+1}]$ into a single straight line directly connecting $z_{t_{k-1}}$ to $z_{t_{k+1}}$.

#### Key Design 2: Aligned V-Prediction

- **Function**: Increases the weight of direction matching in velocity matching.
- **Mechanism**: In addition to the standard MSE loss, a cosine similarity loss is introduced to constrain the consistency of velocity directions. The total loss is defined as $\mathcal{L} = \mathcal{L}_{MSE} + \alpha \cdot (1 - \cos(v_{pred}, v_{target}))$, where $\alpha$ controls the strength of direction matching.
- **Design Motivation**: Experimental results validate that direction noise impacts FID more severely than magnitude noise—under the same perturbation magnitude, directional perturbation consistently causes greater FID degradation than magnitude perturbation. Treating direction and magnitude equally via MSE loss is suboptimal.

#### Key Design 3: Multi-stage Training Strategy

- **Function**: Achieves progressive knowledge transfer from the teacher model to the student model.
- **Mechanism**: ProReflow-I performs a single-stage merge (8 windows $\to$ 4 windows), and ProReflow-II conducts a two-stage merge (8 $\to$ 4 $\to$ 2 windows). Each stage utilizes the model from the previous stage to generate paired (noise, image) data for training the subsequent stage.
- **Design Motivation**: Based on privileged information distillation theory, intermediate representations can transfer knowledge more effectively.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{MSE} + \alpha \cdot \mathcal{L}_{cos}$, where $\mathcal{L}_{MSE} = \|v_{target} - v_\theta(z_t, t)\|^2$ and $\mathcal{L}_{cos} = 1 - \frac{v_{target} \cdot v_\theta}{\|v_{target}\| \|v_\theta\|}$.

## Key Experimental Results

### Main Results: SDv1.5 on MSCOCO-2014 val

| Method | Steps | FID ↓ |
|------|------|-------|
| Teacher (DDIM) | 32 | 10.05 |
| InstaFlow (1-ReFlow) | 4 | 23.40 |
| 2-ReFlow | 4 | 21.64 |
| PeRFlow | 4 | 11.90 |
| **ProReflow-I** | 4 | **10.70** |

### MSCOCO-2017 Comparison

| Method | 4-step FID | 2-step FID |
|------|---------|---------|
| 2-ReFlow | 23.13 | 46.32 |
| PeRFlow | 14.15 | 28.92 |
| **ProReflow-I** | **12.19** | - |
| **ProReflow-II** | - | **24.59** |

### Ablation Study

| Configuration | FID (4 steps) |
|------|----------|
| PeRFlow baseline | 14.15 |
| + Progressive Reflow only | 12.95 |
| + Aligned v-prediction only | 13.10 |
| + Combination of both | **12.19** |

### Key Findings

- ProReflow-I achieves an FID of 10.70 with 4 steps, approaching the 32-step DDIM teacher model (10.05).
- Compared to 2-ReFlow, it reduces FID by 10.94 in 4 steps and 21.73 in 2 steps.
- Both Progressive Reflow and aligned v-prediction are independently effective, with their combination yielding the best performance.
- Direction perturbation experiments clearly demonstrate the criticality of velocity direction to generation quality.

## Highlights & Insights

1. **Ingenious Application of Curriculum Learning in Generative Models**: Transitioning from easy local straightening to difficult global straightening aligns with the natural principles of learning.
2. **Compelling Observation with Respect to Direction vs. Magnitude**: The critical importance of direction matching is clearly illustrated through comparative experiments.
3. **Preserving Sampler Generality**: Without altering the network architecture, the trained model can be used with standard samplers.

## Limitations & Future Work

- Multi-stage training increases training costs (requiring multiple rounds of paired data generation and training).
- Validated only on SDv1.5 and SDXL, its applicability to newer architectures (such as DiT) remains unknown.
- The quality of single-step generation still has room for improvement.
- The coefficient $\alpha$ for aligned v-prediction requires manual tuning.

## Related Work & Insights

- **InstaFlow**: Explored extending ReFlow to large-scale text-to-image models for the first time.
- **PeRFlow**: Proposed piecewise reflow for segment-wise straightening.
- **Progressive Distillation**: A classic work in progressive distillation.
- The progressive paradigm can be generalized to other tasks requiring trajectory straightening.

## Rating

⭐⭐⭐⭐ — The design of Progressive Reflow is elegant and effective, with the observation of direction matching well-supported by experiments. Reaching performance close to the 32-step teacher model in just 4 steps holds significant practical value. However, the overhead of multi-stage training is a cost that needs consideration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Terminal Velocity Matching](../../ICLR2026/image_generation/terminal_velocity_matching.md)
- [\[ICML 2025\] Progressive Tempering Sampler with Diffusion](../../ICML2025/image_generation/progressive_tempering_sampler_with_diffusion.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](../../ICML2026/image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICLR 2026\] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference](../../ICLR2026/image_generation/meancache_from_instantaneous_to_average_velocity_for_accelerating_flow_matching_.md)
- [\[ICML 2026\] Direct 3D-Aware Object Insertion via Decomposed Visual Proxies](../../ICML2026/image_generation/direct_3d-aware_object_insertion_via_decomposed_visual_proxies.md)

</div>

<!-- RELATED:END -->
