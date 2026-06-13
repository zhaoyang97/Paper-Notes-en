---
title: >-
  [Paper Note] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency
description: >-
  [NeurIPS 2025][Image Generation][flow matching] This work is the first to introduce frequency-domain consistency constraints into flow-based visuomotor policies. By projecting action chunk velocity fields into the freque…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "flow matching"
  - "visuomotor policy"
  - "one-step generation"
  - "frequency consistency"
  - "robotic manipulation"
date: 2026-05-08
content_hash: 117c01251d654f4b
---

# FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency

**Conference**: NeurIPS 2025
**arXiv**: [2506.08822](https://arxiv.org/abs/2506.08822)  
**Code**: Not released  
**Area**: Image Generation
**Keywords**: flow matching, visuomotor policy, one-step generation, frequency consistency, robotic manipulation

## TL;DR

This work is the first to introduce frequency-domain consistency constraints into flow-based visuomotor policies. By projecting action chunk velocity fields into the frequency domain via DCT and imposing an adaptive frequency component loss, it achieves high-quality one-step action generation at 93.5 Hz, outperforming existing one-step generation methods on both simulation and real-robot tasks.

## Background & Motivation

- Generative model-based visuomotor policies (diffusion / flow matching) have achieved notable progress in robotic manipulation, but their **multi-step iterative sampling** introduces high inference latency, limiting real-time deployment.
- Existing acceleration methods (Consistency Policy, OneDP, SDM, etc.) largely adapt acceleration techniques from image generation. However, image generation produces **independent samples**, whereas robotic manipulation requires generating action trajectories with **temporal coherence** — a critical distinction that has been overlooked.
- Extensive research in time-series and speech processing shows that frequency-domain features are more effective than time-domain features for modeling non-stationary and oscillatory patterns. Action chunks sampled at high frequency are inherently temporal signals, and frequency-domain representations can more finely distinguish subtle variations in smooth trajectories.
- Accordingly, the authors propose imposing consistency constraints on flow matching from a **frequency-domain perspective**, being the first to exploit the temporal structure of action chunks to accelerate one-step action generation.

## Core Problem

How to leverage the temporal structure of action chunks within the flow matching framework to achieve high-quality one-step action generation **without requiring a pretrained teacher model**?

## Method

### 1. Basic Flow Matching Objective

Given initial noise $a_0 \sim \mathcal{N}(0, I)$ and expert action $a_1$, a velocity field $v_\theta(t, a_t)$ is learned to map noise to actions:

$$\mathcal{L}_{\text{fm}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \|v_\theta(t, a_t) - (a_1 - a_0)\|_2$$

where $a_t = (1-t) \cdot a_0 + t \cdot a_1$ is the linear interpolation. While this basic objective yields a functional policy model, multi-step sampling is still required to generate high-quality actions.

### 2. Frequency Consistency Constraint Objective (Core Contribution)

**Core Idea**: Treating the multi-dimensional action chunk as a **temporal signal** (rather than a static vector), consistency constraints are imposed on velocity vectors at different timesteps in the frequency domain, promoting straight-line flow and one-step generation.

Specific steps:

**(a)** Construct interpolated states $a_r, a_s$ at two random timesteps $s, r \in [0,1]$

**(b)** Project the velocity vector into the frequency domain using the type-II Discrete Cosine Transform (DCT):

$$F(v_t)_k = \sum_{n=0}^{H-1} v_t(n) \cdot \cos\left[\frac{\pi}{N}\left(n+\frac{1}{2}\right)k\right]$$

**(c)** The frequency consistency loss comprises two terms:
- **Velocity consistency**: directly constrains the velocity fields at different timesteps to be consistent in the frequency domain
- **Trajectory consistency**: trajectories propagated from different starting points via their respective velocity fields should converge to the same target point

$$\mathcal{L}_{\text{freq}} = \mathbb{E}\left[\text{Sim}(v_\theta(s, a_s), v_\theta(r, a_r))\right] + \mathbb{E}\left[\text{Sim}(a_s + (u-s)v_\theta(s, a_s), a_r + (u-r)v_\theta(r, a_r))\right]$$

### 3. Adaptive Frequency Component Loss (Second Contribution)

**Design Motivation**: Robotic manipulation sequences alternate between low-dynamic phases (e.g., moving to a target) and high-dynamic phases (e.g., skill transitions, contact interactions), with substantially different frequency component distributions across phases.

Inspired by Focal Loss, an adaptive weighting scheme is designed to assign higher weights to frequency components with larger discrepancies:

$$w_k = \frac{\exp(\|F(v_r)_k - F(v_s)_k\|_2)}{\sum_{j=0}^{H-1} \exp(\|F(v_r)_j - F(v_s)_j\|_2)}$$

The final similarity function is a weighted sum:

$$\text{Sim}(v_r, v_s) = \sum_{k=0}^{H-1} w_k \cdot \|F(v_r)_k - F(v_s)_k\|_2$$

### 4. Total Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fm}} + \mathcal{L}_{\text{freq}}$$

The two losses respectively handle (1) learning an accurate noise-to-action mapping and (2) enforcing temporally consistent frequency-domain flow, jointly enabling reliable one-step action generation.

### 5. Model Architecture

- A standard 1D CNN-based U-Net is used as the backbone (for fair comparison with prior work)
- 2D inputs are encoded with ResNet-18; 3D inputs use a lightweight MLP to encode point clouds
- Can serve as a policy head for VLA models (integrated with OpenVLA)

## Key Experimental Results

### Robomimic (2D input, 5 tasks)

| Method | NFE | Transport | Tool Hang |
|--------|-----|-----------|-----------|
| DDPM | 15 | 80% | 52% |
| Consistency Policy | 1 | 78% | 70% |
| Consistency-FM | 1 | 84% | 80% |
| IMLE Policy | 1 | 90% | 81% |
| **FreqPolicy** | **1** | **90%** | **85%** |

- Best among all one-step methods; Transport is 6% higher than Consistency-FM, Tool Hang is 5% higher

### MetaWorld + Adroit (3D input, 53 tasks)

| Method | NFE | Avg. Success Rate |
|--------|-----|-------------------|
| DP3 (10 steps) | 10 | 76.1% |
| SDM (requires teacher) | 1 | 74.8% |
| FlowPolicy | 1 | 77.2% |
| **FreqPolicy** | **1** | **78.5%** |

- No pretrained teacher model required; surpasses SDM by 3.7 percentage points on MetaWorld

### VLA Integration (LIBERO, 40 tasks)

| Method | NFE | Avg. Success Rate | Inference Speed |
|--------|-----|-------------------|-----------------|
| OpenVLA-DP | 50 | 68.1% | 0.32 Hz |
| OpenVLA-FM | 10 | 93.7% | 1.26 Hz |
| OpenVLA-FM | 1 | 93.5% | 5.92 Hz |
| **OpenVLA-FreqPolicy** | **1** | **94.8%** | **6.05 Hz** |

- One-step FreqPolicy outperforms 10-step FlowMatching (94.8% vs. 93.7%) at 5× the speed

### Real Robot (3 long-horizon tasks)
- FreqPolicy achieves the highest or tied-highest success rate on all tasks at an inference frequency of **93.5 Hz**
- Task 1 (fruit sorting): 70% success @ 93.5 Hz (vs. Diffusion Policy: 55% @ 19.8 Hz)

### Ablation Study

| Configuration | Transport | Tool Hang |
|---------------|-----------|-----------|
| Vanilla FM (1-step) | 84% | 76% |
| + Spatial consistency constraint | 90% | 78% |
| + Frequency consistency (all bands) | 92% | 82% |
| + **Adaptive frequency loss** | **92%** | **88%** |

- From vanilla to full FreqPolicy, Tool Hang improves by 12 percentage points

## Highlights & Insights

1. **Novel perspective**: First to treat action chunks as temporal signals and impose consistency constraints in the frequency domain, breaking away from the convention of directly porting image-generation acceleration techniques to robotics.
2. **No teacher model required**: Unlike SDM / OneDP, which rely on pretrained diffusion teachers, FreqPolicy has a simpler training pipeline.
3. **Well-motivated adaptive weighting**: The Focal Loss-inspired adaptive frequency component weighting dynamically adjusts focus based on the manipulation phase.
4. **Broad experimental coverage**: 93 simulation tasks + 3 real-robot tasks, covering 2D/3D inputs, with additional validation as a VLA policy head.
5. **Extremely fast inference**: Real-robot inference at 93.5 Hz, far exceeding multi-step methods.

## Limitations & Future Work

1. **Only validated with flow matching**: The core idea should be extensible to diffusion-based policies, but this is not verified in the paper.
2. **Doubled computational cost**: The frequency consistency loss requires forward passes over two random samples, making training approximately twice as expensive as basic FM.
3. **No ablation on the choice of DCT**: The rationale for choosing type-II DCT over FFT or wavelet transforms is not sufficiently discussed.
4. **Limited real-robot experiments**: Only 3 tasks in a single manipulation scenario; generalization is not thoroughly validated.
5. **Sensitivity to action chunk length $H$ not analyzed**: The frequency resolution of the DCT representation directly depends on $H$, yet the effect of different chunking sizes on performance is not examined.

## Related Work & Insights

| Method | Policy Type | Requires Teacher | NFE | Consistency Domain |
|--------|-------------|-----------------|-----|--------------------|
| Consistency Policy | Diffusion | ✅ | 1–3 | Denoising trajectory |
| SDM / OneDP | Diffusion | ✅ | 1 | Distribution matching |
| FlowPolicy | Flow Matching | ❌ | 1 | Spatial velocity |
| **FreqPolicy** | **Flow Matching** | **❌** | **1** | **Frequency-domain velocity** |

- FreqPolicy is most closely related to FlowPolicy; the key distinction is shifting from spatial-domain to frequency-domain constraints, exploiting the temporal structure of action chunks.
- Compared to teacher-dependent methods (SDM / OneDP), FreqPolicy offers a simpler training pipeline with superior performance.

The frequency-domain consistency idea may transfer to other temporal generation tasks (motion planning, trajectory prediction). The adaptive frequency weighting design is also relevant to weighted learning across frequency bands in time-series forecasting. The VLA integration approach (with OpenVLA) demonstrates potential for synergy with large models, and future integration with stronger VLAs such as π₀ warrants attention.

## Rating
- Novelty: ⭐⭐⭐⭐ (Frequency-domain consistency constraint is a novel perspective, though the core idea is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (93 simulation tasks + 3 real-robot tasks + VLA integration, with complete ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated)
- Value: ⭐⭐⭐⭐ (Practically significant for real-time robotic deployment, with substantial inference speedup)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](../../AAAI2026/image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[NeurIPS 2025\] Riemannian Consistency Model](riemannian_consistency_model.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](../../ICLR2026/image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](../../ICLR2026/image_generation/cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)

</div>

<!-- RELATED:END -->
