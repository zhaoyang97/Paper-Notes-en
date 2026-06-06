---
title: >-
  [Paper Note] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Probabilistic Time Series Forecasting] PPM utilizes a lightweight encoder to infer context-aware Gaussian priors from historical sequences and "pushes forward" these priors into complete predicti…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Probabilistic Time Series Forecasting"
  - "Non-stationarity"
  - "Parametric Prior"
  - "Push-forward Mapping"
  - "KDE-NLL"
date: 2026-05-08
content_hash: 071909a818c7a054
---

# Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2605.23402](https://arxiv.org/abs/2605.23402)  
**Code**: https://github.com/ljl8336/PPM (Available)  
**Area**: Time Series  
**Keywords**: Probabilistic Time Series Forecasting, Non-stationarity, Parametric Prior, Push-forward Mapping, KDE-NLL  

## TL;DR
PPM utilizes a lightweight encoder to infer context-aware Gaussian priors from historical sequences and "pushes forward" these priors into complete predictive distributions using a two-layer MLP. Through joint training with KDE-NLL and mean MSE, it outperforms diffusion models like DeepAR and NsDiff across seven time-series benchmarks while achieving $2\times$ to $100\times$ faster inference.

## Background & Motivation
**Background**: Probabilistic forecasting for multivariate time series primarily follows two paths. The parametric approach, such as DeepAR assuming fixed Gaussian likelihoods or BetterDeepAR learning time-varying covariance, benefits from stability and efficiency due to strong inductive biases. The deep generative approach, such as TimeGrad, TMDM, and NsDiff, gradually denoises trajectories from noise; these are flexible but slow and data-intensive.

**Limitations of Prior Work**: While diffusion models can theoretically approximate any distribution from any prior, under finite samples and limited computation, the **form of the prior severely impacts trajectory reachability**. TMDM uses $\mathcal{N}(f(\bm{x}),\mathbf{I})$ as an endpoint, forcing the variance to be an identity matrix. NsDiff improves this by using sliding window variance as a prior, but the window length is a fixed hyperparameter that cannot adapt to rapidly changing aleatoric uncertainty. The paper illustrates this using Traffic data: traffic volume variance differs vastly between early morning (trough) and evening (peak), a discrepancy the fixed priors of TMDM/NsDiff fail to capture.

**Key Challenge**: Parametric methods possess strong inductive biases but weak expressiveness, while generative models offer expressiveness but lack structural priors and suffer from slow inference. These advantages are complementary yet have been treated as opposing paradigms.

**Goal**: (1) Construct a data-adaptive prior that varies with input; (2) Retain the expressiveness of generative models to fit complex non-Gaussian conditional distributions; (3) Avoid the $T$-step iterative process of diffusion during inference.

**Key Insight**: Instead of requiring a generative model to learn a transport map starting from uninformative noise, it is more efficient to "leverage" a context-aware Gaussian prior from the historical window using a parametric estimator (MLP). Then, a learned non-linear mapping "pushes forward" this structured prior into the final predictive distribution. This significantly reduces the burden on the transport map, allowing it to be completed in a single forward pass.

**Core Idea**: Utilize parametric estimation to create an **adaptive prior**, followed by a push-forward MLP for **one-step** conditional distribution generation, with hybrid training using KDE density estimation and mean MSE to anchor the first moment.

## Method

### Overall Architecture
The input is a historical window $\bm{x}\in\mathbb{R}^{H\times C}$, and the output is a sample-based predictive distribution over the prediction horizon $\bm{y}\in\mathbb{R}^{L\times C}$. The pipeline consists of three stages:

1.  **Parametric Prior Induction**: An MLP encoder $f_\theta(\bm{x})$ outputs latent variables $(\bm{\mu}, \bm{\sigma})\in\mathbb{R}^{C\times D}$ for each channel, defining a diagonal Gaussian conditional prior $p_\theta(\bm{z}|\bm{x})=\mathcal{N}(\bm{z};\bm{\mu},\text{diag}(\bm{\sigma}^2))$.
2.  **Distribution Push-forward**: Samples $\bm{z}^{(k)}=\bm{\mu}+\bm{\sigma}\odot\bm{\epsilon}^{(k)}$ are obtained via reparameterization and mapped through a channel-independent two-layer GeLU MLP $g_\phi:\mathbb{R}^{C\times D}\to\mathbb{R}^{L\times C}$ to produce $K$ trajectory samples $\hat{\bm{y}}^{(k)}=g_\phi(\bm{z}^{(k)})$. The predictive distribution is formalized as the push-forward measure $q_\phi(\bm{y}|\bm{x})=(g_\phi)_\# p_\theta(\bm{z}|\bm{x})$.
3.  **Hybrid Objective Optimization**: Gaussian KDE is applied to the $K$ samples to estimate the marginal density $\hat{q}_h(y_{c,t}|\bm{x})$, and NLL is calculated using log-sum-exp. An additional MSE term based on the sample mean anchors the first moment. Inference only requires one pass of $f_\theta$ and $K$ passes of $g_\phi$, without KDE.

### Key Designs

1.  **Context-aware Parametric Prior**:
    - **Function**: Directly derives a data-dependent Gaussian prior from the historical window, replacing the rigid "fixed or sliding window" prior settings in diffusion models.
    - **Mechanism**: A lightweight MLP $f_\theta$ maps $\bm{x}$ to $(\bm{\mu},\bm{\sigma})$ in a latent space. An over-complete latent dimension $D$ is used to allow the prior enough capacity to encode rich contextual semantics. Samples $\bm{z}^{(k)}=\bm{\mu}+\bm{\sigma}\odot\bm{\epsilon}^{(k)}$ maintain differentiability. The framework is backbone-agnostic.
    - **Design Motivation**: The authors demonstrate that the true variance of Traffic data fluctuates heavily within a day; by making the prior "respond to $\bm{x}$," the model resolves time-varying aleatoric uncertainty.

2.  **Push-forward MLP (One-step Generation)**:
    - **Function**: Maps the structured Gaussian prior directly to a complex, non-Gaussian, and potentially multi-modal predictive distribution, replacing $T$-step iterative denoising.
    - **Mechanism**: A two-layer channel-independent MLP with GeLU activation projects the $D$-dimensional latent code to the $L$-dimensional prediction window. Theoretically, the push-forward measure $(g_\phi)_\# p_\theta$ is dense for any conditional distribution with finite first moments under the $W_1$ metric (Theorem 5.2).
    - **Design Motivation**: Diffusion iterations bridge generic noise and complex data. Since the prior is already "informed" by the context via parametric estimation, the transport map requires no iterative refinement, reducing inference complexity from $O(BKT)$ to $O(B+BK)$.

3.  **KDE-NLL + Mean MSE Hybrid Objective**:
    - **Function**: Jointly optimizes distribution calibration and point prediction accuracy.
    - **Mechanism**: Since the model produces samples rather than explicit densities, Gaussian KDE estimates marginal densities: $\hat{q}_h(y_{c,t}|\bm{x})=\frac{1}{Kh}\sum_k \mathcal{K}\big(\tfrac{y_{c,t}-\hat{y}_{c,t}^{(k)}}{h}\big)$. Additionally, $\mathcal{L}_{\text{MM}}=\|\bm{y}-\frac{1}{K}\sum_k\hat{\bm{y}}^{(k)}\|^2$ anchors the first moment. Total loss: $\mathcal{L}_{\text{total}}=\alpha\mathcal{L}_{\text{NLL}}+\mathcal{L}_{\text{MM}}$ ($\alpha=0.1, h=0.3$).
    - **Design Motivation**: Pure KDE-NLL may suffer from vanishing gradients when samples are far from reality in early training. Theorem 5.1 provides error bounds for finite $(K,h)$, showing a bias of $O(h^2/\varepsilon)$ and variance of $O(1/(\varepsilon h\sqrt{K}))$. MSE provides stable gradients to "find the mean," while KDE-NLL shapes the distribution using responsibility-weighted residuals.

### Loss & Training
- $K=100$ trajectories are sampled during training. KDE bandwidth $h=0.3$ and weight $\alpha=0.1$.
- Encoder $f_\theta$ and mapping $g_\phi$ are both MLPs, trained with end-to-end backpropagation.
- Inference approximates the distribution with 100 samples via a single pass of $f_\theta$ and $g_\phi$ without iteration.

## Key Experimental Results

### Main Results
Testing on seven real-world datasets (ETTh1/h2/m1/m2, Weather, Electricity, Traffic) against 6 SOTA baselines (DeepAR, TimeGrad, TimeDiff, D3VAE, DiffusionTS, TMDM, NsDiff).

| Dataset | Metric | PPM | 2nd Place (NsDiff) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Electricity | CRPS | 0.206 | 0.286 | **-28.0%** |
| Electricity | QICE | 2.435 | 7.595 | **-67.9%** |
| Traffic | CRPS | 0.252 | 0.367 | **-31.3%** |
| Traffic | QICE | 2.744 | 8.366 | **-67.2%** |
| ETTh1 | CRPS | 0.337 | 0.417 | -19.2% |
| ETTh2 | MSE | 0.376 | 0.448 | -16.1% |
| Weather | CRPS | 0.215 | 0.240 (TMDM) | -10.4% |

CRPS achieved SOTA on all 7 datasets; QICE achieved SOTA on 6 out of 7. MSE/MAE are SOTA across all datasets. On complex datasets like Traffic (Variance=14.225), the performance gain is most significant.

### Ablation Study

| Config | ETTm1 MSE | ETTm1 CRPS | ETTm1 QICE | Elec MSE | Elec CRPS | Elec QICE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Full PPM | **0.381** | **0.314** | **1.782** | **0.182** | **0.206** | **2.435** |
| w/o NLL | 0.371 | 0.345 | 6.342 | 0.182 | 0.257 | 8.317 |
| w/o MM (Mean MSE) | 0.407 | 0.324 | 1.912 | 0.191 | 0.213 | 5.697 |

Prior form ablation (Traffic): Using Raw Gaussian/Uniform as prediction results in QICE of 2.414/3.715. Adding push-forward results in 2.744 (Gaussian) and 2.616 (Uniform), indicating that push-forward is **insensitive to the specific prior form**.

### Key Findings
- **MM loss stabilizes point prediction**: Removing MM significantly increases MSE (ETTm1: 0.381 → 0.407).
- **NLL loss shapes the distribution**: Removing NLL severely degrades QICE (Electricity: 2.435 → 8.317) and CRPS.
- **2×–100× Inference Speed**: By compressing $T$-step denoising into a single-step mapping, complexity is reduced by factor of $\Theta(T)$.
- **Superiority in Non-stationarity**: On Traffic (highest Fourier variance), gain over NsDiff is maximal (CRPS -31.3%), validating the context-aware prior.
- **Higher MI Lower Bound**: PPM's mutual information lower bound between latent $\bm{z}$ and input $\bm{x}$ is consistently higher than baselines.

## Highlights & Insights
- **Upgrading Priors**: Moving from rigid Gaussian/sliding windows to learnable, context-aware distributions addresses the mismatch in non-stationary scenarios at the source.
- **"Doing Less" for Speed and Accuracy**: Collapsing $T$-step denoising into one-step push-forward follows the paradigm of "exchanging iterations for inductive bias," which is transferable to other generative tasks.
- **Complementary Objectives**: The "dense anchor (MSE) + sparse shaping (KDE-NLL)" combination is a robust strategy for sample-based likelihood learning.
- **Theoretical Grounding**: Theorem 5.1 provides principled guidance for co-tuning $h$ and $K$ by defining bias and variance as functions of these hyperparameters.

## Limitations & Future Work
- **KDE Bandwidth Sensitivity**: A single fixed $h$ may enlarge estimation errors in extreme regimes. Learnable or local multi-bandwidth KDE is a potential extension.
- **Marginal vs. Joint Dependence**: NLL is currently computed per $(c,t)$ marginal density; explicit modeling of cross-variable or cross-time joint dependencies is missing.
- **Calibration Evaluation**: Relies on CRPS/QICE; detailed reliability diagrams or quantile coverage over time would provide deeper non-stationary calibration checks.
- **Backbone Diversity**: While Transformer backbones are mentioned in the appendix, maintaining speed advantages with complex architectures remains a practical consideration.

## Related Work & Insights
- **vs. TMDM (NeurIPS'24)**: Both learn priors, but TMDM freezes variance to identity and uses diffusion; PPM uses adaptive variance and single-step mapping.
- **vs. NsDiff (ICLR'25)**: NsDiff uses fixed window variance; PPM uses end-to-end learned $\bm{\sigma}(\bm{x})$ to adaptively track fast dynamics.
- **vs. Flow-based Models**: PPM avoids the architectural constraints of invertibility and Jacobian tractability by using KDE to estimate density.
- **vs. DeepAR**: PPM can be viewed as replacing DeepAR's restrictive Gaussian output with a flexible push-forward MLP while retaining parametric efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICML 2026\] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](../../NeurIPS2025/time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[ICML 2026\] U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecasting](u-cast_a_surprisingly_simple_and_efficient_frontier_probabilistic_ai_weather_for.md)

</div>

<!-- RELATED:END -->
