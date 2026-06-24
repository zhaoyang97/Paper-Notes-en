---
title: >-
  [Paper Note] Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems
description: >-
  [ECCV 2024][LLM Evaluation][Electromagnetic Inverse Scattering] A solution for the Electromagnetic Inverse Scattering Problem (EISP) is proposed based on Implicit Neural Representations (INR). By modeling the relative permittivity of the scatterer as a continuous implicit representation and optimizing it within a forward framework, this approach effectively avoids the difficulties of inverse estimation and low-resolution issues caused by discretization.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Electromagnetic Inverse Scattering"
  - "Implicit Neural Representation"
  - "Computational Imaging"
  - "Forward Estimation"
  - "Non-invasive Imaging"
date: 2026-05-08
content_hash: 12299c2114dbab8f
---

# Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems

**Conference**: ECCV 2024  
**arXiv**: [2407.09352](https://arxiv.org/abs/2407.09352)  
**Code**: Yes ([https://luo-ziyuan.github.io/Imaging-Interiors](https://luo-ziyuan.github.io/Imaging-Interiors))  
**Area**: LLM Evaluation  
**Keywords**: Electromagnetic Inverse Scattering, Implicit Neural Representation, Computational Imaging, Forward Estimation, Non-invasive Imaging

## TL;DR

A solution for the Electromagnetic Inverse Scattering Problem (EISP) is proposed based on Implicit Neural Representations (INR). By modeling the relative permittivity of the scatterer as a continuous implicit representation and optimizing it within a forward framework, this approach effectively avoids the difficulties of inverse estimation and low-resolution issues caused by discretization.

## Background & Motivation

The Electromagnetic Inverse Scattering Problem (EISP) is an important topic in the field of computational imaging. By penetrating the surface of objects with electromagnetic waves, the relative permittivity distribution inside the scatterer can be determined non-invasively, thereby achieving internal structure imaging. Compared with X-rays and MRI, electromagnetic waves offer a low-cost and safe means of non-invasive imaging.

However, solving EISP faces two core difficulties:

**Difficulties in Inverse Estimation**: Back-calculating relative permittivity from measured scattered fields is highly non-linear and ill-posed due to multiple scattering effects.

**Curse of Discretization**: Continuous space must be discretized into finite elements or grids for numerical computation, leading to loss of details and degradation in resolution.

Limitations of Prior Work:
- **Traditional Iterative Methods** (e.g., SOM, CSI): Discretize permittivity into a matrix format for optimization, failing to resolve the low-resolution issue caused by discretization.
- **Deep Learning Methods** (e.g., BPS, PGAN): Employ a two-stage process—first obtaining a coarse image using traditional methods, then refining it with an image translation network, but the second stage ignores the physical scattering data.

## Method

### Overall Architecture

The core idea is to solve the EISP during a forward estimation process, thereby avoiding the difficulties of inverse estimation. The specific approach is as follows:

1. Map spatial coordinates to the relative permittivity $\varepsilon_r$ using an MLP $F_\theta$.
2. Map spatial coordinates and transmitter positions to the induced current $J$ using another MLP $H_\phi$.
3. Optimize these two implicit representations in the forward process to match the calculated scattered field with the measured values.

The use of two MLPs circumvents the complex matrix inversion computation in Equation (7), significantly reducing the computational cost.

### Key Designs

**1. Continuous Representation of Relative Permittivity**

$$\varepsilon_r(\mathbf{x}) = F_\theta(\gamma(\mathbf{x}))$$

where $\gamma$ represents the positional encoding that projects low-dimensional coordinates into a high-dimensional space to enhance fitting capability.

**2. Continuous Representation of Induced Current**

$$J(\mathbf{x}, \mathbf{x}^t) = H_\phi(\gamma(\mathbf{x}), \gamma(\mathbf{x}^t))$$

The induced current depends simultaneously on the spatial coordinates and transmitter positions, a design that faithfully reflects physical relationships.

**3. Stochastic Spatial Sampling Strategy**

The ROI region is divided into an $M \times M$ grid, where each sampling point is randomly drawn from a Gaussian distribution:
$$x_m^{\text{sample}} \sim \mathcal{N}(x_m, \sigma^2)$$

This probabilistic sampling ensures comprehensive consideration of each spatial position during the optimization process, avoiding sampling bias from fixed discrete locations.

### Loss & Training

**Data Loss**: Compares the calculated scattered field with the measured scattered field.
$$\mathcal{L}_{\text{data}} = \sum_{p=1}^{N_t} \|\hat{\mathbf{E}}_p^s - \mathbf{E}_p^s\|^2$$

**State Loss**: Compares the induced current queried directly from $H_\phi$ with the induced current calculated through physical relationships.
$$\mathcal{L}_{\text{state}} = \sum_{p=1}^{N_t} \|\hat{\mathbf{J}}_p - \mathbf{J}_p\|^2$$

**Total Loss**:
$$\mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}$$

where $\mathcal{L}_{\text{TV}}$ is the total variation regularization term. Hyperparameter settings: $\lambda_{\text{data}}=1.0$, $\lambda_{\text{state}}=1.0$, $\lambda_{\text{TV}}=0.01$.

Implementation details: Two 8-layer MLPs, 256 channels, ReLU activation, ROI discretized to 64×64, Adam optimizer, learning rate of $5 \times 10^{-4}$, and 4K iterations.

## Key Experimental Results

### Main Results

| Method | Circular(5% noise)-RRMSE↓ | Circular(5%)-SSIM↑ | MNIST(5%)-RRMSE↓ | MNIST(5%)-SSIM↑ | Fresnel-RRMSE↓ |
|------|-------------------------|---------------------|-------------------|-----------------|----------------|
| **Ours** | **0.016** | **0.968** | **0.017** | **0.972** | **0.127** |
| PGAN | 0.021 | 0.957 | 0.090 | 0.918 | 0.167 |
| Physics-Net | 0.024 | 0.945 | 0.079 | 0.938 | 0.168 |
| BPS | 0.027 | 0.964 | 0.098 | 0.912 | 0.166 |
| Gs SOM | 0.034 | 0.926 | 0.101 | 0.853 | 0.135 |
| BP | 0.048 | 0.916 | 0.171 | 0.750 | 0.180 |

### Ablation Study

| Configuration | Parameters↓ | Iteration Time↓ | RRMSE↓ | SSIM↑ | PSNR↑ |
|------|---------|--------------|--------|-------|-------|
| Dual MLPs (Full Method) | 1,019,139 | 117 ms | 0.038 | 0.909 | 30.36 |
| Single MLP | 493,313 | 289 ms | 0.053 | 0.876 | 27.32 |

### Key Findings

1. The proposed method comprehensively outperforms traditional (BP, SOM) and deep learning (BPS, PGAN) baselines on both synthetic data and the real-world Fresnel database.
2. Accurate reconstruction is maintained even under a 30% noise level, demonstrating excellent robustness.
3. Trained using 64x64 resolution, yet allows flexible sampling during inference to yield higher-resolution images.
4. Superior performance is still achieved in sparse measurement scenarios using only 25% of the standard measurement data.
5. The method can be naturally extended to 3D scenarios.

## Highlights & Insights

- **Forward Estimation Preempting Inverse Estimation**: The ill-posed inverse problem is ingeniously transformed into an optimization problem within a forward framework.
- **Dual-MLP Strategy**: Permittivity and induced current are represented separately, bypassing the computational bottleneck of matrix inversion.
- **Inherent Advantages of INR**: The resolution flexibility of continuous representations naturally overcomes the curse of discretization.
- **Stochastic Sampling** enhances the model's comprehensive coverage of spatial positions.

## Limitations & Future Work

- Optimization is required individually for each target (case-by-case), making batch inference impossible.
- The number of optimization iterations still incurs some computational overhead (4K iterations).
- Validations in practical application scenarios, such as medical diagnostics, are yet to be conducted.
- Shortening optimization times by improving the INR architecture (e.g., using acceleration methods like TensoRF) could be considered.

## Related Work & Insights

- Applying INR (the core technology of the NeRF family) to electromagnetic imaging, a non-visual domain, demonstrates the broad applicability of INRs.
- The paradigm of forward optimization combined with physical constraints can be generalized to other inverse problems.
- The idea of using dual networks to separately model different physical quantities is highly informative for other physics-driven problems.

## Rating

- **Novelty**: ★★★★☆ — The combination of INR and forward estimation is a pioneering effort in EISP.
- **Practicality**: ★★★☆☆ — Case-by-case optimization limits practical deployment efficiency.
- **Experimental Thoroughness**: ★★★★★ — Synthetic/real/noise/sparse/3D/ablation scenarios are all thoroughly covered.
- **Writing Quality**: ★★★★☆ — Clear exposition of physical background and methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [\[ICLR 2026\] ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems](../../ICLR2026/llm_evaluation/acadreason_exploring_the_limits_of_reasoning_models_with_academic_research_probl.md)
- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](../../ICML2025/llm_evaluation/leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ICLR 2026\] Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions?](../../ICLR2026/llm_evaluation/inverse_ifeval_can_llms_unlearn_stubborn_training_conventions_to_follow_real_ins.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)

</div>

<!-- RELATED:END -->
