---
title: >-
  [Paper Note] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
description: >-
  [CVPR 2026][3D Vision][Physical Property Estimation] This paper proposes PhysGS, which integrates Bayesian inference into a 3D Gaussian Splatting pipeline. By leveraging vision-language model priors and multi-view confidence-weighted updates, PhysGS enables per-point probabilistic estimation and uncertainty quantification of physical properties (friction, hardness, density, stiffness), achieving a 22.8% improvement over NeRF2Physics in APE for mass estimation and a 61.2% reduction in Shore hardness error.
tags:
  - CVPR 2026
  - 3D Vision
  - Physical Property Estimation
  - Bayesian Inference
  - 3D Gaussian Splatting
  - Uncertainty Quantification
  - Vision-Language Models
date: 2026-05-08
content_hash: 421165c32b9f6fd9
---

# PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation

**Conference**: CVPR 2026
**arXiv**: [2511.18570](https://arxiv.org/abs/2511.18570)
**Code**: [Project Page](https://samchopra2003.github.io/physgs)
**Area**: 3D Vision
**Keywords**: Physical Property Estimation, Bayesian Inference, 3D Gaussian Splatting, Uncertainty Quantification, Vision-Language Models

## TL;DR
This paper proposes PhysGS, which integrates Bayesian inference into a 3D Gaussian Splatting pipeline. By leveraging vision-language model priors and multi-view confidence-weighted updates, PhysGS enables per-point probabilistic estimation and uncertainty quantification of physical properties (friction, hardness, density, stiffness), achieving a 22.8% improvement over NeRF2Physics in APE for mass estimation and a 61.2% reduction in Shore hardness error.

## Background & Motivation
**Background**: Understanding the physical properties of environments (friction, hardness, elasticity, density, etc.) is critical for safe robotic interaction. Existing 3D reconstruction methods (NeRF, 3DGS) focus primarily on geometry and appearance, and cannot infer underlying physical properties.

**Limitations of Prior Work**:
   - Methods such as NeRF2Physics perform zero-shot regression using language embeddings but do not model uncertainty, leading to brittle predictions in ambiguous regions (e.g., dirt vs. asphalt).
   - Existing approaches typically estimate only one or two physical properties (e.g., friction or elasticity) and lack generality.
   - Outdoor scenes remain largely unexplored.
   - Two sources of uncertainty—aleatoric (sensor noise) and epistemic (insufficient model knowledge)—are not explicitly modeled.

**Key Challenge**: How to estimate multiple physical properties within a unified framework from visual sensors, while simultaneously quantifying the reliability of such estimates.

**Key Insight**: Each Gaussian primitive is treated as a probabilistic entity whose belief over physical properties is continuously refined through Bayesian posterior updates.

**Core Idea**: A Dirichlet-Categorical model is used to fuse discrete material classification, while a Normal-Inverse-Gamma (NIG) prior models aleatoric and epistemic uncertainty over continuous properties.

## Method

### Overall Architecture
Multi-view images → SAM part-level segmentation → VLM (GPT-5) predicts material labels, density, and other physical properties with confidence scores → Bayesian inference fuses multi-view evidence → 3DGS reconstruction → per-point physical property field.

### Key Designs

1. **Dirichlet-Categorical Material Classification**

    - **Function**: Fuses discrete material label predictions from a VLM into posterior probabilities via Bayesian updates.
    - **Mechanism**: The Dirichlet distribution serves as the conjugate prior to the Categorical likelihood. Posterior parameters are updated recursively as $\tilde{\alpha}_i \leftarrow \alpha_i(0) + \sum_{m: c_m=i} \lambda p_m$, where $p_m$ is the VLM confidence for the $m$-th prediction. The posterior predictive probability is $f(z=i | Z, \boldsymbol{\alpha}) = \tilde{\alpha}_i / \sum_j \tilde{\alpha}_j$.
    - **Design Motivation**: Single-view VLM predictions may be inconsistent; cross-view fusion is necessary. The conjugacy of the Dirichlet-Categorical model enables closed-form fusion, which is well-suited for online updates.

2. **Bayesian Estimation of Continuous Properties**

    - **Function**: Estimates the mean and variance of continuous physical properties (e.g., friction coefficient, density) per material class.
    - **Mechanism**: Three confidence-weighted accumulators are maintained: $W_i = \sum p_m$, $S_i = \sum p_m \psi_m$, and $Q_i = \sum p_m \psi_m^2$. The posterior mean is $\mu_i = S_i / W_i$ and variance is $\sigma_i^2 = Q_i/W_i - \mu_i^2$. The final property distribution is a Gaussian mixture: $f(\psi | Z, \boldsymbol{\alpha}) = \sum_i \frac{\tilde{\alpha}_i}{\sum_j \tilde{\alpha}_j} \mathcal{N}(\mu_i, \sigma_i^2)$.
    - **Design Motivation**: Incremental online updates avoid storing historical observations, making the approach particularly suitable for streaming reconstruction scenarios.

3. **Normal-Inverse-Gamma Uncertainty Decomposition**

    - **Function**: Decomposes total predictive uncertainty into aleatoric uncertainty (sensor/perceptual noise) and epistemic uncertainty (insufficient model knowledge).
    - **Mechanism**: A NIG distribution $p(\mu_i, \sigma_i^2 | \tau_i, \kappa_i, \alpha_i, \beta_i)$ is used as the joint prior over mean $\mu_i$ and variance $\sigma_i^2$. Uncertainty is decomposed as $\text{Var}[\psi_i] = \underbrace{\mathbb{E}[\sigma_i^2]}_{\text{aleatoric}} + \underbrace{\text{Var}[\mu_i]}_{\text{epistemic}}$, where $\mathbb{E}[\sigma_i^2] = \tilde{\beta}_i / (\tilde{\alpha}_i - 1)$ and $\text{Var}[\mu_i] = \mathbb{E}[\sigma_i^2] / \tilde{\kappa}_i$.
    - **Design Motivation**: Robotic decision-making requires explicit uncertainty awareness. High aleatoric uncertainty indicates perceptual difficulty, while high epistemic uncertainty suggests that additional observations are needed.

4. **3DGS Semantic Property Mapping**

    - **Function**: Maps the Bayesian-inferred material-property correspondences back onto the 3D Gaussian field.
    - **Mechanism**: Scene images are re-colored according to the Bayesian-determined material assignments and used as semantic inputs to build the 3DGS. Each voxel is associated with a predicted property value, supporting per-point queries (e.g., friction) and global aggregation (e.g., total mass).

### Loss & Training
- Uses the Nerfstudio `splatfacto-big` variant, trained for 20k iterations on an RTX A5000.
- VLM: GPT-5 with structured visual-text prompts.

## Key Experimental Results

### Main Results — Mass Estimation (ABO-500 Test Set, 100 Objects)

| Method | ADE↓(kg) | ALDE↓ | APE↓ | MnRE↑ |
|--------|---------|--------|------|-------|
| Image2mass | 12.496 | 1.792 | 0.976 | 0.341 |
| 2D CNN | 15.431 | 1.609 | 14.459 | 0.362 |
| LLaVA | 17.328 | 1.893 | 1.837 | 0.306 |
| NeRF2Physics | 8.730 | **0.771** | 1.061 | **0.552** |
| **PhysGS** | **8.254** | 0.999 | **0.819** | 0.474 |

### Ablation Study (ABO-500 Validation Set)

| Configuration | ADE↓ | APE↓ | Note |
|---------------|------|------|------|
| NeRF2Physics | 9.786 | 0.931 | Baseline |
| PhysGS (w/o Bayesian) | 9.728 | 0.717 | Without Bayesian updates |
| **PhysGS (w/ Bayesian)** | **9.187** | **0.715** | Full model |

### Key Findings
- PhysGS reduces APE by 22.8% relative to NeRF2Physics (1.061→0.819) and ADE by 5.5%.
- Bayesian inference reduces ADE by 5.6% compared to the variant without Bayesian updates.
- Shore hardness error is reduced by 61.2%; kinetic friction coefficient error is reduced by 18.1%.
- Regions with high aleatoric uncertainty correspond to areas with large sensor noise or difficult material recognition; regions with high epistemic uncertainty correspond to areas with insufficient evidence—consistent with intuition.

## Highlights & Insights
- **Theoretical elegance of Bayesian inference + 3DGS**: Treating each Gaussian primitive as a probabilistic entity refined through posterior updates is a natural and principled modeling choice. Confidence-weighted online updates require no storage of historical data, making the approach well-suited for incremental reconstruction.
- **Practical value of uncertainty decomposition**: The aleatoric vs. epistemic distinction is highly relevant for downstream robotic decision-making. In regions of high epistemic uncertainty, a robot should proceed more cautiously or seek additional observations.
- **VLMs as a source of physical priors**: Leveraging GPT-5 as a zero-shot prior for material recognition and property estimation is a viable approach; while individual predictions are limited in accuracy, multi-view Bayesian fusion substantially improves overall estimates.

## Limitations & Future Work
- VLM-based physical property estimation is zero-shot and lacks domain calibration, potentially introducing large biases for uncommon materials.
- Mass estimation relies on density estimation × volume integration; errors in volume estimation propagate and compound.
- NeRF2Physics still outperforms PhysGS on ALDE and MnRE, suggesting that Bayesian updates introduce bias under certain metrics.
- The pipeline requires SAM segmentation and VLM inference, incurring higher computational cost than purely geometry-based methods.
- Validation on outdoor scenes is primarily qualitative, without quantitative benchmarks.

## Related Work & Insights
- **vs. NeRF2Physics**: NeRF2Physics uses CLIP embeddings and kernel regression for property estimation without uncertainty modeling. PhysGS outperforms it on most metrics, but falls short on ALDE and MnRE.
- **vs. GaussianProperty**: Also built on 3DGS + VLM, but lacks Bayesian updates and uncertainty decomposition.
- **vs. uncertainty-aware methods (e.g., EVORA, MEM)**: These model uncertainty in 2D perception; PhysGS is the first work to jointly address physical property estimation and uncertainty quantification within 3DGS.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Bayesian inference, 3DGS, and physical property estimation is innovative, and the NIG decomposition provides theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐ The dataset and baselines are somewhat limited; outdoor scenes lack quantitative evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are complete and clear; problem motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ Uncertainty-aware physical property estimation has significant implications for robotics.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_model_for_feed-forward_4d_synthesis.md)
- [\[AAAI 2026\] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation](../../AAAI2026/3d_vision/cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)

<!-- RELATED:END -->
