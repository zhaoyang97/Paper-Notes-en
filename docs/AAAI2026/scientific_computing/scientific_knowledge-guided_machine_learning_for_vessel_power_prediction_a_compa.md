---
title: >-
  [Paper Note] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study
description: >-
  [AAAI 2026][Scientific Computing][vessel power prediction] A hybrid modeling framework combining a physics baseline with a data-driven residual is proposed. The sea trial power curve (propeller law $P=cV^n$) serves as th…
tags:
  - "AAAI 2026"
  - "Scientific Computing"
  - "vessel power prediction"
  - "hybrid modeling"
  - "physical residual learning"
  - "PINN"
  - "XGBoost"
  - "extrapolation generalization"
date: 2026-05-08
content_hash: 10429ed06498c35f
---

# Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study

**Conference**: AAAI 2026
**arXiv**: [2602.18403](https://arxiv.org/abs/2602.18403)  
**Authors**: Orfeas Bourchas, George Papalambrou
**Code**: None  
**Area**: Scientific ML / Marine Engineering
**Keywords**: vessel power prediction, hybrid modeling, physical residual learning, PINN, XGBoost, extrapolation generalization

## TL;DR

A hybrid modeling framework combining a physics baseline with a data-driven residual is proposed. The sea trial power curve (propeller law $P=cV^n$) serves as the baseline, and XGBoost/NN/PINN models learn the residual correction, significantly improving extrapolation stability and physical consistency in sparse data regions.

## Background & Motivation

1. **Tightening emission regulations**: IMO constraints on shipping carbon emissions are increasingly stringent, making accurate main engine power prediction a prerequisite for fuel optimization and compliance.
2. **Limitations of traditional empirical models**: Sea trial curves and calm-water resistance models capture only the cubic speed–power relationship, failing to reflect complex operational deviations such as wind, waves, and hull fouling.
3. **Poor extrapolation of pure data-driven models**: Models such as XGBoost and ANN achieve high accuracy within the training distribution, but predictions in sparse regions—such as high-speed or extreme operating conditions—often violate the propeller law, exhibiting non-monotonic or flat curves.
4. **Training difficulties of PINNs**: Although physics-informed neural networks can incorporate physical constraints, training on noisy measured data is extremely challenging, and capturing steep gradients and discontinuities remains difficult.
5. **Insights from residual hybrid strategies**: Recent work in physics-informed learning, such as Stacked Residual approaches, demonstrates that decomposing complex predictions into a physics baseline plus residual correction substantially simplifies the learning task.
6. **Lack of systematic cross-architecture comparison**: Prior work has not compared XGBoost, NN, and PINN models under a unified framework before and after the introduction of a physics baseline.

### Mechanism

**Goal**:

## Method

### Overall Architecture

The power prediction is decomposed into two components:

$$\hat{P}(\bar{X}) = P_{\text{sea trial}}(V, T) + f(\bar{X})$$

- **Physics baseline** $P_{\text{sea trial}}(V, T)$: A calm-water power curve obtained from sea trials, of the form $P = cV^n$, fitted separately for ballast and laden conditions, with intermediate drafts computed by linear interpolation:

$$P_{\text{sea trial}}(V,T) = \left(1-\frac{T-T_b}{T_l-T_b}\right) P_b(V) + \frac{T-T_b}{T_l-T_b} P_l(V)$$

- **Residual regressor** $f(\bar{X})$: A data-driven model that learns only the difference (residual) between the measured power and the physics baseline, with inputs including wind velocity components $W_x, W_y$, hull aging time, and other factors not covered by the baseline.

### Hybrid Configurations for Three Regressors

1. **Hybrid XGBoost**: XGBoost predicts the residual; hyperparameters are optimized via RandomizedSearchCV (learning rate, tree depth, number of estimators, L1/L2 regularization).
2. **Hybrid NN**: A simple fully connected network predicts the residual, trained with Adam for 1000 epochs; hyperparameters are optimized via WandB Bayesian sweep (learning rate, number of layers, number of neurons).
3. **Hybrid PINN**: A physical loss term is added on top of the NN. The total loss is:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{P.\text{law}}$$

where the physical loss enforces the propeller law constraint via partial differentiation with respect to speed:

$$\mathcal{L}_{P.\text{law}} = \sum_{i}\left(\frac{\partial f}{\partial V}\bigg|_{x_i} - \left[3cV_i^2 - \text{baseline derivative term}\right]\right)^2$$

$\lambda=100$ is a fixed weight; all derivatives are computed via PyTorch autograd.

### Input Features

| Symbol | Description | Unit |
|--------|-------------|------|
| $V$ | Speed through water (S.T.W.) | kn |
| $T$ | Mean draft | m |
| Trim | Trim | m |
| $W_x, W_y$ | cos/sin components of true wind speed | kn |
| $t$ | Logging time (UTC) | — |

## Experiments

### Dataset

Approximately 40,000 data points from five months of operational data from a real vessel, split 80/10/10 into training/validation/test sets.

### Quantitative Results

### Main Results

| Metric | XGBoost Baseline | XGBoost Hybrid | NN Baseline | NN Hybrid | PINN Baseline | PINN Hybrid |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Train MAE [kW] | 114.2 | 143.3 | 181.0 | 235.3 | 147.0 | 214.7 |
| Train RMSE [kW] | 180.5 | 199.9 | 229.7 | 292.1 | 205.5 | 248.3 |
| Val MAE [kW] | 114.2 | 143.3 | 156.3 | 214.1 | 135.5 | 167.2 |
| Val RMSE [kW] | 180.5 | 199.9 | 212.9 | 272.4 | 204.2 | 219.5 |
| Test MAE [kW] | 122.2 | 148.8 | 162.7 | 219.3 | 144.3 | 171.2 |
| Test RMSE [kW] | 195.0 | 208.2 | 225.1 | 284.3 | 211.9 | 229.5 |

Hybrid models exhibit slightly higher global errors than baseline models (within 1% of the vessel's maximum power), though global metrics fail to capture extrapolation behavior in sparse regions.

### Hyperparameter Comparison

### Ablation Study

| Hyperparameter | XGBoost Base | XGBoost Hybrid | NN Base | NN Hybrid | PINN Base | PINN Hybrid |
|----------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Learning rate | 0.25 | 0.1 | 1e-4 | 3e-4 | 1e-4 | 3e-4 |
| Layers/depth | depth=10 | depth=10 | 6 layers | 4 layers | 8 layers | 6 layers |
| Neurons/estimators | 40 | 500 | 256 | 64 | 256 | 256 |
| Regularization | L1=1, L2=0 | L1=1, L2=100 | — | — | λ=100 | λ=100 |

### Qualitative Extrapolation Analysis

Extrapolation behavior is evaluated across the speed range of 8–17 kn under ballast draft and 5 kn wind conditions at wind directions of 0°/90°/180°:
- **XGBoost Baseline**: Non-monotonic or flat power–speed curves frequently appear in high-speed regions, violating physical expectations.
- **NN Baseline**: Overestimation in high-speed regions and anomalous sensitivity to wind direction.
- **PINN Baseline**: Benefits from physical loss, yet still exhibits deviations in sparse regions.
- **All Hybrid models**: Power–speed curves are smooth and monotonically increasing, maintaining physical consistency beyond the training envelope.

## Key Findings

- The core value of the hybrid framework lies not in reducing global error metrics, but in **regularizing model behavior in sparse and extrapolation regions**.
- Hybrid PINN achieves the best performance across all architectures, combining competitive accuracy with the strongest physical consistency.
- Restricting the ML task to residual prediction is equivalent to a strong regularization: the model need not rediscover the dominant speed–power relationship.
- In practical applications such as weather routing and speed optimization, decisions are often made in regions with the sparsest historical data coverage—precisely where the hybrid model's advantage is greatest.

### Key Findings
- The primary components/modules contribute the most critical performance improvements.

## Highlights & Insights

- **Concise and effective decomposition**: Using the sea trial curve as a physics baseline and learning the residual with ML is intuitive and highly practical for engineering deployment.
- **Unified cross-architecture validation**: Effectiveness is demonstrated across three fundamentally different architectures—XGBoost, NN, and PINN—establishing the generality of the framework.
- **Derivation of PINN physical loss**: The physical constraint for the PINN (partial derivative of the residual function with respect to speed) is re-derived within the hybrid framework, with complete and rigorous mathematical derivation.
- **Focus on extrapolation over fitting**: Rather than pursuing lower training error, the paper targets physical credibility of model predictions under unseen operating conditions, reflecting a precise problem formulation.

## Limitations & Future Work

- The physics baseline uses only linear interpolation between two draft conditions (ballast and laden), which is a simplification of the actual nonlinear draft–power relationship.
- $\lambda=100$ is fixed throughout; the effect of adaptive weight scheduling on PINN training is not explored.
- Experimental data come from a single vessel over five months; generalization to different vessel types and longer time horizons remains to be verified.
- The cumulative effect of hull fouling over time is not explicitly modeled (it appears only as an input feature).
- The hybrid model depends on the availability of sea trial data, which is not standardized across all vessels.
- No comparison is made with more advanced time-series models such as Transformer or Mamba.

## Related Work & Insights

- **Classical ML prediction**: Gaussian Process, SVR (Bassam et al. 2023; Gkerekos et al. 2019), ANN/LSTM (Guo et al. 2023; Cai et al. 2024; Chen et al. 2024), XGBoost (Nguyen et al. 2023; Agand et al. 2023; Fan et al. 2024).
- **PINNs in the maritime domain**: Raissi et al. (2017) introduced the PINN framework; Lang et al. (2024) and Bourchas & Papalambrou (2025) applied it to vessel speed/power prediction.
- **Residual hybrid strategies**: Howard et al. (2023) proposed Stacked Networks to improve PINN training; Eshkofti & Barreau (2025) proposed Vanishing Stacked-Residual PINN.

## Rating

- Novelty: ⭐⭐⭐ The framework is intuitive; the core contribution lies in systematic comparison rather than methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across three architectures × two configurations, including both quantitative and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, complete mathematical derivation, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides a practical tool for vessel power prediction with direct applicability to weather routing and energy efficiency planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/scientific_computing/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/scientific_computing/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[ICLR 2026\] Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](../../ICLR2026/scientific_computing/learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)

</div>

<!-- RELATED:END -->
