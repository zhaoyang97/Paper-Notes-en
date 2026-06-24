---
title: >-
  [Paper Note] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control
description: >-
  [ICCV 2025][Autonomous Driving][Model Quantization] This paper proposes SQIL (Saliency-Aware Quantized Imitation Learning), which identifies task-critical states via saliency scoring and applies weighted distillation during quantization-aware training. SQIL recovers full-precision performance for 4-bit quantized VLA policy models in robotic manipulation and autonomous driving, while achieving 2.5–3.7× inference speedup.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Model Quantization"
  - "Imitation Learning"
  - "Saliency-Awareness"
  - "Vision-Language-Action Models"
  - "Edge Deployment"
date: 2026-05-08
content_hash: 5ce0a88206fc70ec
---

# Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control

**Conference**: ICCV 2025
**arXiv**: [2505.15304](https://arxiv.org/abs/2505.15304)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: Model Quantization, Imitation Learning, Saliency-Awareness, Vision-Language-Action Models, Edge Deployment

## TL;DR

This paper proposes SQIL (Saliency-Aware Quantized Imitation Learning), which identifies task-critical states via saliency scoring and applies weighted distillation during quantization-aware training. SQIL recovers full-precision performance for 4-bit quantized VLA policy models in robotic manipulation and autonomous driving, while achieving 2.5–3.7× inference speedup.

## Background & Motivation

Deep neural network-based policy models (e.g., the VLA model OpenVLA) have demonstrated strong performance in robotic manipulation and autonomous driving, yet their rapidly growing model sizes (from 0.07M parameters in D4RL to 7.6B in OpenVLA) pose significant challenges for real-time deployment on resource-constrained devices.

Model quantization is an effective approach to reduce inference cost. However, the authors find that the impact of quantization on imitation learning (IL) policies fundamentally differs from that in conventional classification or NLP tasks:

**Most timesteps are minimally affected**: Quantization errors cause only slight action deviations at the majority of timesteps.

**Critical states are severely degraded**: At task-critical states (e.g., grasping or releasing objects during fine manipulation), quantization errors lead to large action deviations that ultimately cause task failure.

This phenomenon is distinct from distributional shift—the deviation originates from model quantization rather than data mismatch, and only a small number of critical states fail rather than entire trajectories collapsing. Conventional PTQ and QAT methods cannot selectively address these critical states, and the QAT+LoRA 4-bit quantization attempted in the original OpenVLA work also yields inconsistent results.

## Method

### Overall Architecture

SQIL consists of two core components: **SIS (Saliency-based State-Importance Score)** for identifying task-critical states, and **QRD (Quantization-Robust Action Distillation)** for applying weighted distillation of the full-precision policy's action distribution at those states. The total loss is:

$$\mathcal{L}^{\text{SQIL}}(\theta) = \mathcal{L}^{\text{QAT}}(\theta) + \mathcal{L}^{\text{QRD}}(\theta)$$

### Key Designs

1. **SIS (Saliency-based State-Importance Score)**: Local perturbations (Gaussian blur) are applied to visual inputs to measure the sensitivity of policy outputs. For each location $k$ of state $s_t$, the action deviation under perturbation is computed as:

$$S_\pi(s_t, k) = \frac{1}{2}\|\pi(s_t) - \pi(\phi(s_t, k))\|^2$$

The SIS is the mean saliency across all locations: $SIS^{s_t}_\pi = \mathbb{E}_k[S_{\pi^{FP}}(s_t, k)]$. A high SIS value indicates that the state is sensitive to visual perturbations, typically corresponding to critical fine-manipulation moments (e.g., grasping, releasing). Compared to vision-language-based keyframe (KF) detection, which activates only at coarse-grained subtask boundaries, SIS captures fine-grained robot–environment interactions and yields a 1.1% higher success rate than KF in experiments.

2. **QRD (Quantization-Robust Action Distillation)**: The action distribution of the full-precision policy guides the quantized model, with SIS-based weighting assigning greater training emphasis to critical states:

$$\mathcal{L}^{\text{QRD}}(\theta) = \alpha_t \cdot \mathbb{E}_{\tau_i \sim \mathcal{D}_E}\left[\frac{1}{|\tau_i|}\sum_{s_t \in \tau_i} D(\pi^Q(s_t), \pi^{FP}(s_t))\right]$$

where $\alpha_t = \beta$ (when $SIS > T$, i.e., top 20%) or $\alpha_t = 1$ (otherwise). $D$ denotes L2 distance and $\beta > 1$ is an additional weighting coefficient. The key distinction from conventional knowledge distillation is the selective weighting—strong distillation is applied only to critical states.

3. **QAT+QRD Synergy**: QAT maximizes the log-likelihood of expert actions in the quantized model, while QRD aligns the overall action distribution of the quantized model with that of the full-precision model. Their combination enables the quantized policy to both preserve expert behavior and recover the decision patterns of the full-precision model. Action distribution visualizations confirm this: PTQ severely deviates from the reference, QAT produces overly sharp peaks, QRD recovers distributional shape but may neglect expert actions, and SQIL combines the advantages of both.

### Loss & Training

- Total loss: $\mathcal{L}^{\text{SQIL}} = \mathcal{L}^{\text{QAT}} + \mathcal{L}^{\text{QRD}}$
- SIS can be precomputed once offline, requiring no repeated evaluation
- Existing expert datasets and training hyperparameters are reused; no additional data collection is needed
- QLoRA (r=32) is applied to OpenVLA, fine-tuning 110M trainable parameters
- Hyperparameters $D(\cdot)$, $\beta$, and $T$ are insensitive to convergence and are kept identical across all tasks

## Key Experimental Results

### Main Results

**Robotic Manipulation (OpenVLA + LIBERO benchmark, INT4 Weight-Only)**

| Method | Quantizer | Spatial | Object | Goal | Long |
|--------|-----------|---------|--------|------|------|
| FP | - | 84.0% | 83.9% | 76.6% | 50.7% |
| PTQ | AWQ | 80.1% | 81.3% | 74.3% | 47.2% |
| QAT | AWQ | 80.9% | 82.4% | 75.7% | 47.3% |
| **SQIL** | AWQ | **83.9%** | **83.5%** | **76.3%** | **49.2%** |
| SQIL | QuaRot | 83.8% | 83.7% | 76.3% | 49.4% |

**Autonomous Driving (CILRS, W4A4, NoCrash-dense)**

| Method | Bit-width | tt Succ. | tn Succ. | nt Succ. | nn Succ. |
|--------|-----------|---------|---------|---------|---------|
| FP | FP | 82% | 74% | 80% | 68% |
| PTQ | W4A4 | 34% | 43% | 36% | 29% |
| QAT | W4A4 | 62% | 58% | 58% | 48% |
| **SQIL** | W4A4 | **80%** | **72%** | **72%** | **68%** |

### Ablation Study

| Method | SIS vs. KF | LIBERO Avg. Success Rate |
|--------|------------|--------------------------|
| QAT | - | 71.6% |
| SQIL (KF) | Keyframe | 72.6% |
| SQIL (SIS) | Saliency | **73.2%** |

**Deployment Efficiency (Edge Devices)**

| Platform | Model | Speedup | Energy Savings |
|----------|-------|---------|----------------|
| Jetson AGX Orin | OpenVLA INT4 | 2.5× | 2.5× |
| RTX 2080Ti | CILRS W4A4 | 3.7× | 3.1× |

### Key Findings

- SQIL recovers near-full-precision success rates across both AWQ and QuaRot quantizers.
- Saliency map visualizations confirm that SQIL restores the distorted attention distributions of quantized models.
- SQIL remains effective on a real UR5 robot (77% vs. FP 79%).
- Consistent improvements are observed when applied to alternative architectures such as π₀.
- Quantized models maintain generalization across varying lighting conditions and language instructions.

## Highlights & Insights

- **Precise diagnosis of quantization failure**: Not all timesteps are affected; only a small subset of critical states (fine-manipulation moments) constitute the bottleneck—a significant finding in its own right.
- **Strong generalizability**: SQIL operates as a plug-and-play solution effective across robotic manipulation, autonomous driving, and physics simulation.
- **Elegant saliency score design**: Action sensitivity to visual perturbations is exploited to automatically discover critical states, eliminating the need for manual annotation or environment interaction.
- **High practical deployment value**: A 2.5× speedup and reduced energy consumption are demonstrated on real edge devices such as the Jetson AGX Orin.

## Limitations & Future Work

- SIS precomputation requires forward inference with the full-precision model, incurring additional offline preparation cost.
- The current top-20% threshold selection is relatively simple; whether adaptive threshold strategies could yield further improvement merits exploration.
- Only 4-bit quantization is evaluated; performance under lower bit-widths (e.g., 2-bit) remains unknown.
- For policy models with non-visual inputs (e.g., pure state vector inputs), the saliency score computation would require adaptation.

## Related Work & Insights

- Unlike RL quantization methods such as LPPD, SQIL targets the IL setting and requires no environment interaction feedback.
- Reframing quantization as a "critical-state protection" problem provides a new perspective for future IL model compression research.
- The saliency scoring paradigm is generalizable to other model compression directions, including pruning and distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of quantization effects on imitation learning, with a novel critical-state-aware quantization scheme.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Coverage spans three domains (robotics/driving/simulation), multiple quantizers, and real hardware deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical structure, precise problem formulation, and high-quality visualizations.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to the edge deployment of large VLA models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](../../ECCV2024/autonomous_driving/risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[ICCV 2025\] Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations](resonance_learning_to_predict_social-aware_pedestrian_trajectories_as_co-vibrati.md)
- [\[ICCV 2025\] Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks](wavelet_policy_lifting_scheme_for_policy_learning_in_long-horizon_tasks.md)
- [\[ICML 2026\] CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving](../../ICML2026/autonomous_driving/coirl-ad_collaborative-competitive_imitation-reinforcement_learning_in_latent_wo.md)

</div>

<!-- RELATED:END -->
