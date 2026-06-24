---
title: >-
  [Paper Note] Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion
description: >-
  [ICML2025][Image Generation][Flow Matching] This paper proposes the Multidimensional Adaptive Coefficient (MAC), a plug-and-play module for flow/diffusion models. MAC extends traditional one-dimensional time scheduling coefficients to multidimensional, sample-adaptive coefficients. By optimizing the inference trajectory through adversarial training, MAC achieves a SOTA FID of 1.37 with 5 NFEs on conditional CIFAR-10 generation.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Flow Matching"
  - "Diffusion acceleration"
  - "inference trajectory optimization"
  - "multidimensional adaptive coefficient"
  - "adversarial training"
date: 2026-05-08
content_hash: 8f0dc1f411dddec3
---

# Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion

**Conference**: ICML2025  
**arXiv**: [2404.14161](https://arxiv.org/abs/2404.14161)  
**Code**: To be confirmed  
**Area**: Flow Model Acceleration / image_generation  
**Keywords**: Flow Matching, Diffusion acceleration, inference trajectory optimization, multidimensional adaptive coefficient, adversarial training

## TL;DR

This paper proposes the Multidimensional Adaptive Coefficient (MAC), a plug-and-play module for flow/diffusion models. MAC extends traditional one-dimensional time scheduling coefficients to multidimensional, sample-adaptive coefficients. By optimizing the inference trajectory through adversarial training, MAC achieves a SOTA FID of 1.37 with 5 NFEs on conditional CIFAR-10 generation.

## Background & Motivation

Flow and diffusion models have demonstrated excellent performance and training stability in generative tasks, but lack two key attributes compared to simulation-based methods (such as NeuralODE):

**Dimensional Freedom**: Traditional interpolation coefficients $\alpha_0(t), \alpha_1(t) \in \mathbb{R}$ in flow/diffusion are scalars, imposing the same time scheduling across all data dimensions.

**Trajectory Adaptivity**: During inference, all samples share the same step size and trajectory direction, lacking dynamic adjustments for different samples.

Existing trajectory optimization methods (e.g., straightness constraints in Rectified Flow, OT pairing) define optimality criteria beforehand and lack dimensional flexibility in the inference schedule. The motivation of this work is to integrate the advantages of simulation methods into the flow/diffusion framework while maintaining training efficiency.

## Method

### Core Idea: From One-Dimensional to Multidimensional Adaptive Coefficients

The interpolation path of traditional flow/diffusion is:

$$x(t) = \alpha_0(t) x_0 + \alpha_1(t) x_1, \quad \alpha_0(t), \alpha_1(t) \in \mathbb{R}$$

MAC extends the coefficients from scalars to diagonal matrices $\gamma(t) \in \mathbb{R}^{d \times 2}$ to allow different dimensions to have different time schedulings:

$$x(t) = \gamma_0(t) \odot x_0 + \gamma_1(t) \odot x_1, \quad \gamma_0(t), \gamma_1(t) \in \mathbb{R}^d$$

Furthermore, a parameterized MAC $\gamma_\phi(t, \mathbf{x}_{\theta,\phi}^{\mathcal{S}})$ is introduced, enabling adaptive adjustment of coefficients based on different inference trajectories.

### Parameterization Design of MAC

MAC is modeled using weighted sinusoidal basis functions (similar to Fourier expansion):

- **Basis Functions**: $b_m(t) = \sin(\pi m (t/T)^{1/q})$
- **Weight Network**: $w_\phi(x_T) = s \cdot \text{LPF} \circ \tanh(\text{nn}_\phi(x_T))$, where $\text{nn}_\phi$ is a U-Net
- **Low-Pass Filter LPF**: Gaussian convolution is used to eliminate high-frequency noise and ensure smooth coefficients
- **tanh Constraint**: Restricts the output range to $(-1, 1)$, supporting direct sampling of weights from a uniform distribution during pre-training.

Key Design: $\gamma_\phi$ requires only a single forward pass at $t=T$ to compute the entire inference schedule.

### Inference Trajectory Optimization

**Problem 1 (Optimizing MAC Only)**: Given a fixed vector field $\theta$, optimize $\phi^* = \arg\min_\phi \mathbb{D}(\rho_0, \hat{\rho}_{0,\theta,\phi})$

**Problem 2 (Joint Optimization)**: Simultaneously optimize the vector field and the inference schedule $\theta^*, \phi^* = \arg\min_{\theta,\phi} \mathbb{D}(\rho_0, \hat{\rho}_{0,\theta,\phi})$

### Loss & Training

Using hinge loss + StyleGAN-XL discriminator $D_\psi$, three sets of loss functions are updated respectively:

- $\mathcal{L}_\phi$: Adversarially optimizes MAC parameters by simulating the entire inference process of $G_{\theta,\phi}$
- $\mathcal{L}_\theta$: Adversarially optimizes the vector field model $H_\theta$
- $\mathcal{L}_\psi$: Optimizes the discriminator

### Optional γ-Pre-training

Pre-train $H_\theta$ with randomly sampled multidimensional coefficients $\gamma \sim \Gamma_h$ to adapt it to multidimensional inputs. The pre-training phase restricts multidimensionality when $t$ is large, while the adversarial phase removes this restriction. This step is optional—MAC is directly compatible with standard $\alpha$ pre-trained models.

## Key Experimental Results

### 2D Synthetic Data (Optimizing φ Only, Freezing θ)

| Method | Gaussian→8Gaussians (NFE=5) | Gaussian→Moons (NFE=5) |
|------|:---:|:---:|
| SI_α | 0.763 | 0.882 |
| SI_γ + opt φ（MAC） | **0.721** | **0.682** |
| SI_α^OT | 0.457 | 0.245 |
| SI_γ^OT + opt φ（MAC） | **0.399** | **0.230** |

Optimizing MAC alone reduces the $\mathcal{W}_2$ distance across all configurations.

### Effect of γ-Pre-training (CIFAR-10 FID↓)

| Method | NFE=100 | NFE=200 |
|------|:---:|:---:|
| SI_α | 4.75 | 4.30 |
| SI_γ | **3.98** | **3.63** |
| FM_α | 4.52 | 4.07 |
| FM_γ | **3.59** | **3.42** |

Multidimensional pre-training consistently improves performance across all frameworks (SI/FM/DDPM).

### SOTA Comparison on CIFAR-10

| Model | NFE | FID uncond. | FID cond. |
|------|:---:|:---:|:---:|
| EDM_α（Karras 2022） | 35 | 1.98 | 1.79 |
| CTM_α + adv θ（Kim 2024） | 2 | 1.87 | 1.63 |
| **EDM_γ + adv θ,φ (MAC, Ours)** | **5** | **1.69** | **1.37** |

Achieves a SOTA conditional generation FID of **1.37** (5 NFE) on CIFAR-10.

### ImageNet-64 Conditional Generation

| Model | NFE | FID | FD_DINOv2 |
|------|:---:|:---:|:---:|
| CTM_α + adv θ | 2 | 1.73 | 157.7 |
| **EDM_α + adv θ,φ (MAC)** | **5** | **1.48** | **70.2** |

FD_DINOv2 significantly outperforms CTM (70.2 vs 157.7), demonstrating that MAC possesses a greater advantage in perceptual quality.

### Adaptivity Ablation (CIFAR-10, 10 NFE)

| Conditional Input of γ_φ | SI_γ FID | DDPM_γ FID |
|------|:---:|:---:|
| Constant $\mathbf{1}_d$ (No adaptivity) | 7.84 | 26.09 |
| Random $z \sim \rho_T$ | 6.48 | 23.31 |
| Actual Starting Point $x_T \sim \rho_T$ | **4.14** | **10.04** |

Using the actual starting point of inference as the MAC input yields the best performance, validating the importance of sample adaptivity.

## Highlights & Insights

1. **Novel Perspective**: Proposes that the optimality of the inference trajectory should not be defined by pre-defined criteria (such as straightness) but measured by the final generative quality after simulation.
2. **Plug-and-Play**: As a modular component, MAC is compatible with any flow/diffusion framework such as DDPM/FM/EDM/SI without modification to the backbone architecture.
3. **Efficient Training**: The parameter size of the MAC network is far smaller than the primary model, and $\gamma_\phi$ requires only 1 NFE to compute the entire inference schedule.
4. **Search Space Expansion**: Expands scheduling from a scalar to a dimension-level adaptive schedule, permitting non-linear curved trajectories and sample-level adaptive step sizes.
5. **Elegant Theory**: Eliminates coarse coefficients through a Fourier basis + low-pass filter design, demonstrating a well-justified hypothesis space.

## Limitations & Future Work

1. **FID vs. Perceptual Quality**: FID performance drops sharply at highly restricted NFEs such as 4 steps (e.g., DDPM_γ achieves FID=72.64 at 4 NFE), showing a gap remains before practical 1- or 2-step generation.
2. **Discriminator Dependence**: The use of the StyleGAN-XL discriminator for adversarial training introduces additional training complexity and hyperparameter tuning overheads.
3. **Resolution Limits**: Experiments are verified only up to 128×128 or 64×64 resolution, leaving scalability to higher resolutions (e.g., 256/512) unexplored.
4. **Diagonal Matrix Simplification**: Restricting $\gamma$ to a diagonal matrix for computational efficiency may discard interactions among dimensions.
5. **U-Net Dependency**: The MAC network utilizes a U-Net architecture, leaving more lightweight alternatives unexplored.

## Related Work & Insights

- **Rectified Flow / OT Pairing**: Traditional strategies utilize straightness as the predefined optimality standard, whereas MAC provides an alternative path free from predefined criteria.
- **CTM (Kim et al., 2024)**: As a direct competitor in adversarial distillation, MAC outperforms it with better FID and FD_DINOv2 at 5 NFEs.
- **NeuralODE**: The theoretical motivation behind MAC stems from introducing the dimensional freedom and adaptivity of simulation-based methods into flow/diffusion.
- **Insight**: This work implies that trajectory optimality in flow/diffusion models remains an under-explored direction, potentially inspiring subsequent research in adaptive sampling strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ — The concept of a multidimensional adaptive coefficient is novel, and optimizing inference trajectories from a coefficient design perspective is an under-explored direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — It covers 4 frameworks, 4 datasets, and multiple ablation studies, providing convincing SOTA results.
- Writing Quality: ⭐⭐⭐⭐ — The mathematical formulation is clean, the notations are consistent, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ — Its plug-and-play property and training efficiency make it highly potential for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](../../CVPR2025/image_generation/rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)
- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](sada_stability-guided_adaptive_diffusion_acceleration.md)
- [\[ICLR 2026\] SDErasure: Concept-Specific Trajectory Shifting for Concept Erasure via Adaptive Diffusion Classifier](../../ICLR2026/image_generation/sderasure_concept-specific_trajectory_shifting_for_concept_erasure_via_adaptive_.md)
- [\[ICLR 2026\] FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing](../../ICLR2026/image_generation/flowalign_trajectory-regularized_inversion-free_flow-based_image_editing.md)
- [\[ICLR 2026\] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](../../ICLR2026/image_generation/flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)

</div>

<!-- RELATED:END -->
