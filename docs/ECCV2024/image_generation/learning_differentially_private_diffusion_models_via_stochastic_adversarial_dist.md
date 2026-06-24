---
title: >-
  [Paper Note] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation
description: >-
  [ECCV 2024][Image Generation] The DP-SAD framework is proposed to train differentially private diffusion models via stochastic adversarial distillation. It leverages the diffusion model's timesteps to dilute the impact of DP noise, introduces a discriminator to accelerate convergence, and combines the gradient chain rule with DP's post-processing property to reduce the introduction of randomness, achieving SOTA privacy-preserving image generation quality without requiring pre…
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 4002033829dacfdb
---

# Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation

**Conference**: ECCV 2024  
**arXiv**: [2408.14738](https://arxiv.org/abs/2408.14738)  
**Area**: Image Generation

## TL;DR

The DP-SAD framework is proposed to train differentially private diffusion models via stochastic adversarial distillation. It leverages the diffusion model's timesteps to dilute the impact of DP noise, introduces a discriminator to accelerate convergence, and combines the gradient chain rule with DP's post-processing property to reduce the introduction of randomness, achieving SOTA privacy-preserving image generation quality without requiring pre-training.

## Background & Motivation

In privacy-sensitive areas (e.g., healthcare, finance), data sharing is restricted. Differentially private (DP) generative models offer a solution by releasing the trained generative model rather than the raw data. Existing approaches face three major challenges:

**Difficulties in GAN Training**: Privacy constraints aggravate the already unstable GAN training process.

**Amplification of High-Dimensional Noise**: As data and model dimensions increase, more DP noise is required to maintain the same level of privacy.

**Severe Damage from Direct Noise Addition**: DPSGD adds noise to all gradients, introducing excessive randomness.

Existing DP diffusion models (e.g., DP-DM, DP-LDM) require pre-training on large datasets, and directly training diffusion models with DPSGD leads to excessive privacy budget consumption.

## Method

### Overall Architecture

DP-SAD consists of three components:
1. **Teacher Model $\epsilon_\psi$**: Trained directly on private data without protection, used only to guide student training and is not released.
2. **Student Model $\epsilon_\theta$**: Learns from the teacher via distillation, achieving DP by applying clipping and noise addition to the gradients.
3. **Discriminator $\epsilon_\phi$**: Distinguishes between the teacher and student outputs, forming adversarial training to accelerate convergence.

### Key Designs

**Timestep Dilution of DP Noise**: The core innovation lies in utilizing the $T$ timesteps of the diffusion model to dilute the impact of DP noise. The DP noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ is amortized by $B \cdot T$:

$$\bar{g} \approx \frac{1}{B}\sum_{i=1}^{B} CLIP\left(\frac{\partial \mathcal{L}^{i,r}}{\partial \theta}, C\right) + \frac{\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})}{B \cdot T}$$

Increasing $T$ does not increase the privacy budget but reduces the impact of noise on gradients.

**Stochastic Timestep Sampling**: Replacing the average gradient over $T$ steps with the gradient of a single randomly selected timestep $r$ drastically reduces computational overhead (no need to run the entire $T$-step diffusion chain for each sample). The intermediate state $x_r$ is obtained directly via the forward process instead of reverse inference from noise.

**Chain Rule + Post-Processing Property**: Utilizing the gradient chain rule $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial x_{\theta}} \cdot \frac{\partial x_\theta}{\partial \theta}$, clipping and noise addition are only applied to $\frac{\partial \mathcal{L}}{\partial x_\theta}$. The term $\frac{\partial x_\theta}{\partial \theta}$ requires no noise injection due to the post-processing property of DP, thereby reducing the introduction of randomness.

**Adversarial Discriminator**: Concatenates teacher and student outputs as the discriminator input, with teacher labels as $[1,0]$ and student labels as $[0,1]$, driving the student output to closely match the teacher.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{dis} + \lambda \mathcal{L}_{adv}$$

- $\mathcal{L}_{dis}$: MSE distillation loss = MSE(teacher output, student output) + MSE(real data, student output)
- $\mathcal{L}_{adv}$: adversarial loss = $\log(1 - \epsilon_\phi(\mathcal{C}(x_{\psi}, x_{\theta})))$
- $\lambda = 1$ (the optimal trade-off coefficient, determined via ablation)

## Key Experimental Results

### Main Results

CelebA 64×64 perceptual quality comparison ($\varepsilon=10$, while methods not requiring pre-training use $\varepsilon=10^4$):

| Method | Requires Pre-training | IS ↑ | FID ↓ |
|---|---|---|---|
| DP-GAN | No | 1.00 | 403.94 |
| PATE-GAN | No | 1.00 | 397.62 |
| GS-WGAN | No | 1.00 | 384.78 |
| DP-MERF | No | 1.36 | 327.24 |
| DPGEN | No | 1.48 | 55.91 |
| DP-LDM | Yes | N/A | 14.30 |
| **DP-SAD** | **No** | **2.37** | **11.26** |

Classification accuracy comparison ($\varepsilon=1$ / $\varepsilon=10$):

| Method | MNIST | FMNIST | CelebA-H | CelebA-G |
|---|---|---|---|---|
| DP-GAN | 0.404/0.801 | 0.105/0.610 | 0.533/0.521 | 0.345/0.392 |
| GS-WGAN | 0.143/0.808 | 0.166/0.658 | 0.590/0.614 | 0.420/0.523 |
| DPGEN | 0.905/0.936 | 0.828/0.878 | 0.700/0.884 | 0.661/0.815 |
| DP-DM (Pre-trained) | 0.952/0.981 | 0.794/0.862 | N/A | N/A |
| **DP-SAD** | **0.962/0.976** | **0.844/0.896** | **0.915/0.928** | **0.826/0.841** |

### Ablation Study

Impact of structural coefficient $\lambda$ (CelebA, $\varepsilon=10$):

| $\lambda$ | 0.0 | 0.2 | 0.5 | **1.0** | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| FID ↓ | 14.63 | 13.41 | 12.38 | **11.68** | 12.11 | 13.55 |
| IS ↑ | 2.12 | 2.18 | 2.26 | **2.37** | 2.35 | 2.31 |

Impact of timestep $T$: As $T$ increases, IS steadily rises and FID consistently decreases, verifying the effectiveness of timestep dilution of DP noise. In experiments, $T=500$ is selected to balance efficiency and quality.

Ablation on model conditioning: The combination of student conditioning + discriminator conditioning yields the best performance.

### Key Findings

- **No pre-training required** while still outperforming the pre-trained DP-LDM (FID: 11.26 vs 14.30), demonstrating the strength of the proposed method itself.
- Outperforms other training-from-scratch methods by at least **16 percentage points** on complex tasks (CelebA, $\varepsilon=1$).
- Increasing $T$ acts as a "free" means to improve the privacy-utility trade-off—improving generation quality without exhausting more privacy budget.
- The method permits a **small batch size + large $T$** instead of a large batch size, enabling training in resource-constrained scenarios.

## Highlights & Insights

1. **A New Utility for Diffusion Timesteps**: This is the first work to discover and utilize diffusion timesteps to dilute the impact of DP noise, establishing a novel connection between diffusion model architecture and privacy protection.
2. **Rigorous Privacy Proofs**: Provides a complete privacy bound analysis using Rényi DP and the Gaussian mechanism.
3. **High Practicality**: Does not require large-scale pre-training datasets and supports resource-constrained scenarios.
4. **Gradient Chain Rule + Post-Processing**: Cleverly exploits the post-processing property of DP to reduce noise injection into model parameter gradients.

## Limitations & Future Work

- The teacher model is trained on private data without protection—although not released, it poses a potential risk.
- Validated only on relatively low resolutions (32x32, 64x64); the performance on high resolutions remains unknown.
- Requires an additional clustering step (MoCo + k-means) for unlabeled data.
- A gap may exist between the theoretical privacy bound and the actual level of privacy protection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of timestep dilution of DP noise is novel, and the adversarial distillation framework is well-designed.
- **Practicality**: ⭐⭐⭐⭐ — Non-reliance on pre-training, support for resource-constrained scenarios, SOTA performance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 11 baselines, 3 datasets, and thorough ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivation and complete privacy analysis, though the writing could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
