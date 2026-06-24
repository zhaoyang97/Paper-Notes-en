---
title: >-
  [Paper Note] Learning Extremely High Density Crowds as Active Matters
description: >-
  [CVPR 2025][Time Series][High-density crowd] This paper models extremely high-density crowds ($\ge 5 \text{ people/m}^2$) as active matter, proposing a neural stochastic differential equation system that combines a novel "crowd material" stress model with Toner-Tu active forces. The system learns and predicts crowd dynamics directly from in-the-wild video optical flow using a hybrid Eulerian-Lagrangian CrowdMPM framework.
tags:
  - "CVPR 2025"
  - "Time Series"
  - "High-density crowd"
  - "active matter"
  - "Material Point Method"
  - "Neural Stochastic Differential Equations"
  - "optical flow prediction"
date: 2026-05-08
content_hash: b5c0b729e6eaa2a0
---

# Learning Extremely High Density Crowds as Active Matters

**Conference**: CVPR 2025  
**arXiv**: [2503.12168](https://arxiv.org/abs/2503.12168)  
**Code**: None  
**Area**: Time-series Analysis / Crowd Dynamics  
**Keywords**: High-density crowd, active matter, Material Point Method, Neural Stochastic Differential Equations, optical flow prediction

## TL;DR
This paper models extremely high-density crowds ($\ge 5 \text{ people/m}^2$) as active matter, proposing a neural stochastic differential equation system that combines a novel "crowd material" stress model with Toner-Tu active forces. The system learns and predicts crowd dynamics directly from in-the-wild video optical flow using a hybrid Eulerian-Lagrangian CrowdMPM framework.

## Background & Motivation

**Background**: Video crowd analysis and prediction are long-standing problems in computer vision. Existing approaches are divided into empirical modeling (interpretable but inaccurate) and data-driven methods (accurate but lacking interpretability). Recent hybrid methods combine neural networks with differential equations, but they primarily target low-density scenarios.

**Limitations of Prior Work**: Extremely high-density crowd scenarios ($>5 \text{ people/m}^2$) face three major difficulties: (1) data is scarce and of low quality—CCTV videos are highly noisy, making it difficult to track individuals or count; (2) trajectory-based methods are infeasible at this density; (3) high-density crowd dynamics are extremely complex, exhibiting wave-like spatio-temporal perturbations that can lead to fatal stampedes.

**Key Challenge**: Existing methods either require precise trajectory data (which is unobtainable under high density) or are pure black-box models (making them unusable for simulation and analysis). Meanwhile, high-density crowds exhibit unique active matter properties—where humans, as self-propelled particles under physical constraints, still demonstrate autonomous motion—which is fundamentally different from low-density dynamics.

**Goal**: To design a learnable physical model capable of learning high-density crowd dynamics directly from in-the-wild video optical flow, while maintaining interpretability and simulation capability.

**Key Insight**: The authors observe that high-density crowds resemble active matter (composed of self-propelled particles in a continuous medium subject to random forces), thereby drawing inspiration from continuum mechanics and active matter theory for modeling.

**Core Idea**: To model the crowd as a novel "crowd material" characterized by three key properties: elastic asymmetry, exponential resistance, and compression dominance. This is integrated with a Toner-Tu equation to capture random active forces, forming a neural stochastic differential equation system solved via MPM.

## Method

### Overall Architecture
The input consists of optical flow estimates from video frames, treated as noisy observations of the underlying continuous velocity field. The system simultaneously solves the equations on an Eulerian grid and Lagrangian particles using the Material Point Method (MPM). The grid discretizes space, while the particles represent individual pedestrians. The model is learned end-to-end by minimizing the discrepancy with the observed optical flow.

### Key Designs

1. **CrowdMPM (Crowd Material Point Method)**:

    - **Function**: Solves the conservation equations of the crowd continuum.
    - **Mechanism**: Adopts a hybrid Eulerian-Lagrangian scheme, where particles are no longer mere quadrature points but represent actual individuals. The three-step update loop is: P2G (Particle-to-Grid transfer of mass and momentum) $\rightarrow$ GO (Grid Operations to solve momentum equations and apply boundary conditions) $\rightarrow$ G2P (Grid-to-Particle update of particle velocity, position, and deformation gradient).
    - **Design Motivation**: Pure Eulerian methods fail to model individual behaviors, while pure Lagrangian methods cannot guarantee full spatial coverage. MPM combines the strengths of both, perfectly addressing the need to "model Lagrangian behaviors (individual active forces) when only Eulerian data (optical flow) is available."

2. **Crowd Material Stress Model $\sigma^{cm}$**:

    - **Function**: Captures the unique stress-strain relationship of crowds as a continuum.
    - **Mechanism**: Modeled using three characteristics: elastic asymmetry (crowds disperse easily but are hard to compress, implemented via weakly compressible fluid stress), exponential resistance (repulsive forces within the comfort distance grow logarithmically as $f_r = -k\log(d_{pp'})$, simulating the exponential increase in human resistance when close), and compression dominance (separating the compressive forces between particles from shear/rotational forces, implemented through projected traction forces). Key parameters $k$ and Young's modulus $\epsilon$ are predicted by neural networks based on particle positions, velocities, and neighborhoods.
    - **Design Motivation**: Crowds differ from homogeneous materials like water—people cannot overlap, maintain a comfort distance, and can slide past each other at close range. These three characteristics precisely correspond to these empirical observations.

3. **Toner-Tu Active Force Model $f^{act}$**:

    - **Function**: Captures the random active forces generated by individual self-propulsion within the crowd.
    - **Mechanism**: Based on the Toner-Tu equation describing the collective dynamics of active matter, it divides the forces into a motion alignment term $\alpha v$ (learned by $NN_\alpha$) and a residual random force term. Since the distribution of the latter is non-Gaussian, it is assumed to be Gaussian in a latent space and modeled using the decoder of a Conditional Variational Autoencoder (CVAE), with the terms of the TT equation and a latent variable $z$ as inputs.
    - **Design Motivation**: Pedestrians in high-density crowds exhibit autonomous behaviors such as recovering balance and following neighbors, which manifest as systematic random forces that cannot be captured by material stress alone.

### Loss & Training
The model is fully differentiable and trained end-to-end via MSE between the predicted and observed optical flows (optimized using Adam). Since it is essentially parameterized PDE learning, it does not require massive training data.

## Key Experimental Results

### Main Results

| Dataset | Metric (Errvel) | Ours (mean) | BaselineI | HINN | SimVP | Gain |
|--------|-------------|-----------|-----------|------|-------|------|
| Drill1 | Errvel | **0.5284** | 0.7555 | 0.5618 | 2.2364 | Best |
| Drill2 | Errvel | **1.0721** | 1.3319 | 1.1187 | 5.6415 | 18.69% vs. second best |
| Drill3 | Errvel | **1.6461** | 2.1150 | 2.6590 | 2.9760 | Best |
| Hajj | Errvel | 0.6591 | 0.9354 | 1.1600 | **0.6212** | Close to best |
| Hellfest | Errvel | **3.0457** | 3.5151 | 7.1427 | 4.9703 | Best |
| Marathon | Errvel | **1.4927** | 2.8778 | 4.3488 | 1.6636 | Best |

### Ablation Study

| Configuration | Description |
|------|------|
| The proposed model shows significant advantages in long-term prediction | The margin of improvement over other methods widens as the prediction horizon increases |
| Methods perform closely in the Hajj scenario | Due to slow crowd detour, the dynamics are simple |
| Optical flow metric is not optimal on Marathon | Because the moving crowd only occupies part of the space, and optical flow noise in other areas is filtered out by P2G |

### Key Findings
- The proposed method achieves the best performance on 5 out of 6 datasets under the Errvel metric, achieving an 18.69% improvement over the second best on Drill2.
- Its advantage is more pronounced in long-term predictions, reflecting the extrapolation capability of the physical model.
- Serving as a continuous-time physical model, it can be used for simulation and analysis, offering strong interpretability.
- The Hajj dataset is relatively simple (slow circular motion), and the differences among various methods are minor.

## Highlights & Insights
- Modeling crowds as "active matter" and designing specific material constitutive models is a highly novel physical modeling perspective. The three properties of elastic asymmetry, exponential resistance, and compression dominance precisely summarize the physical behavior of high-density crowds.
- The "particle-as-individual" design in CrowdMPM seems simple, but it crucially unifies macro-continuum models with micro-individual behaviors.
- Using a CVAE to learn non-Gaussian random forces in the TT equation is a clever approach—it maintains the physical framework while granting sufficient expressive power.

## Limitations & Future Work
- The model depends on the quality of optical flow estimation, and noisy optical flow can affect modeling accuracy.
- Currently, modeling is limited to the 2D plane, without considering height information and 3D effects.
- The dataset scale is relatively small (mostly laboratory sessions or YouTube videos), and the generalization capability remains to be validated.
- Future work could consider extending to unified modeling across different density ranges, achieving a smooth transition from low to high density.

## Related Work & Insights
- **vs HINN**: HINN uses neural networks embedded with fluid dynamics but ignores the self-propelled nature of crowds. The proposed active matter modeling aligns more closely with the intrinsic nature of crowds.
- **vs SimVP/TAU**: Pure data-driven video prediction methods may suffice in simple scenarios (e.g., Hajj) but fall significantly short compared to physics-driven methods in complex and chaotic scenes.
- **vs Trajectory-based Methods**: This approach completely bypasses the challenge of obtaining individual trajectories under high density.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Modeling crowds as active matter is a highly original, interdisciplinary idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 datasets, multiple baselines, but the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ The physical modeling part is clear, but formulas are dense and require a strong background.
- Value: ⭐⭐⭐⭐ Highly practical significance for high-density crowd safety analysis.
---
title: >-
  [Paper Interpretation] Learning Extremely High Density Crowds as Active Matters
description: >-
  [CVPR 2025][Time Series][Extremely High-Density Crowd] Analogizing extremely high-density crowds to "active matter" in physics, learning collective crowd dynamic behavior patterns from noisy in-the-wild videos for crowd analysis and prediction.
tags:
  - CVPR 2025
  - Time Series
  - Extremely High-Density Crowd
  - Active Matter
  - Crowd Dynamics
  - In-the-wild Video Learning
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](../../NeurIPS2025/time_series/statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](../../NeurIPS2025/time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[ICML 2026\] OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density](../../ICML2026/time_series/olivia_harmonizing_time_series_foundation_models_with_power_spectral_density.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](../../NeurIPS2025/time_series/frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)

</div>

<!-- RELATED:END -->
