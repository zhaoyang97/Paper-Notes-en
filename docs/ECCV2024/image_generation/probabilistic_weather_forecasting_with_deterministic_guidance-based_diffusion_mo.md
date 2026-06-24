---
title: >-
  [Paper Note] Probabilistic Weather Forecasting with Deterministic Guidance-Based Diffusion Model
description: >-
  [ECCV 2024][Image Generation][Probabilistic Weather Forecasting] This paper proposes DGDM (Deterministic Guidance Diffusion Model), which jointly trains a deterministic prediction branch and a Brownian Bridge-based probabilistic-diffusion branch. By utilizing deterministic forecasting results to truncate the reverse diffusion process, the model controls the range of uncertainty while achieving both accurate and probabilistic weather forecasting…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Probabilistic Weather Forecasting"
  - "Deterministic Guidance"
  - "Diffusion Model"
  - "Brownian Bridge"
  - "Truncated Diffusion"
date: 2026-05-08
content_hash: cdcb80f79deed959
---

# Probabilistic Weather Forecasting with Deterministic Guidance-Based Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2312.02819](https://arxiv.org/abs/2312.02819)  
**Code**: [https://github.com/DongGeun-Yoon/DGDM](https://github.com/DongGeun-Yoon/DGDM)  
**Area**: Diffusion Models / Weather Forecasting  
**Keywords**: Probabilistic Weather Forecasting, Deterministic Guidance, Diffusion Model, Brownian Bridge, Truncated Diffusion

## TL;DR
This paper proposes DGDM (Deterministic Guidance Diffusion Model), which jointly trains a deterministic prediction branch and a Brownian Bridge-based probabilistic-diffusion branch. By utilizing deterministic forecasting results to truncate the reverse diffusion process, the model controls the range of uncertainty while achieving both accurate and probabilistic weather forecasting, reaching SOTA performance in both global and regional forecasting tasks.

## Background & Motivation

**Background**: Weather forecasting requires both deterministic results for immediate decision-making and probabilistic results to assess uncertainty. Traditional Numerical Weather Prediction (NWP) achieves ensemble forecasting by running multiple simulations with minor perturbations to initial conditions, which is extremely computationally expensive. Recently, data-driven methods like GraphCast and Pangu-Weather have surpassed NWP in deterministic global forecasting, but they are inherently deterministic models and cannot provide probabilistic predictions.

**Limitations of Prior Work**: Deterministic models (e.g., TAU, SimVP) achieve high prediction accuracy but fail to capture the multiple possibilities of weather, outputting a single blurry prediction—essentially the "average" of all possible futures. Probabilistic models (e.g., RaMViD, VDM) can generate diverse prediction samples based on diffusion processes, but suffer from insufficient accuracy, and the excessive sample diversity makes it difficult to determine which sample is closest to reality. Data-driven weather forecasting faces a trade-off between determinism (high accuracy but no probabilistic prediction) and probability (diverse but low accuracy).

**Key Challenge**: Deterministic models cannot express sub-grid scale processes (such as turbulence and tropical cumulus convection, which are stochastic phenomena smaller than the model's grid resolution), whereas probabilistic models, though capable of expressing uncertainty, suffer from uncontrollable prediction quality, making excessive diversity a burden instead of an asset.

**Goal**: (1) How to simultaneously obtain high-accuracy and probabilistic weather forecasts; (2) How to control the uncertainty range of probabilistic predictions; (3) How to utilize deterministic results to improve the quality of probabilistic predictions.

**Key Insight**: NWP ensemble forecasting is inherently "generating probability by perturbing from a deterministic starting point". Inspired by this, the authors propose using the deterministic prediction as an intermediate starting point for the reverse process of the diffusion model rather than starting from pure noise, thereby constraining the prediction space while maintaining diversity.

**Core Idea**: Utilizing the prediction results from the deterministic branch as the intermediate starting point of the diffusion model's reverse process for truncated diffusion, which both preserves probabilistic diversity and improves prediction accuracy while accelerating inference.

## Method

### Overall Architecture
DGDM consists of two branches: the deterministic branch (DB) and the probabilistic branch (PB). During training, DB employs a non-autoregressive encoder-translator-decoder structure to predict future weather, while PB models the transition between initial weather conditions and future weather using a Brownian Bridge diffusion process. The two branches are trained jointly in an end-to-end manner. During inference, the prediction results of DB are used as the intermediate starting point of PB's reverse process, controlling the uncertainty range of predictions via truncated diffusion.

### Key Designs

1. **Deterministic Branch (DB)**:

    - **Function**: Provides high-accuracy deterministic weather forecasting while offering guidance information for the probabilistic branch.
    - **Mechanism**: Adopts a non-autoregressive architecture to avoid error accumulation inherent in autoregressive predictions. It consists of an encoder $e(\cdot)$, a spatial-temporal translator $st(\cdot)$, and a decoder $d(\cdot)$, similar to the structure of TAU. Given input $x$ (current weather conditions), the loss function is $L_{DB} = \|y - d(st(e(x)))\|^2$. Crucially, the intermediate feature $z = st(e(x))$ of the translator is extracted for the probabilistic branch to use via a cross-attention mechanism.
    - **Design Motivation**: The non-autoregressive architecture performs better over fixed forecast horizons. Moreover, DB serves not only as an independent predictor but also as an information source for PB—the spatial-temporal features it extracts are injected into PB via cross-attention, enabling probabilistic predictions to leverage the precise dynamics captured by the deterministic model.

2. **Probabilistic Branch (PB)**:

    - **Function**: Generates diverse probabilistic predictions to capture weather uncertainty.
    - **Mechanism**: Employs a Brownian Bridge diffusion process instead of the standard DDPM. The forward process of the Brownian Bridge is conditioned on the starting point $x_0 = y$ (the ground-truth future weather) and the endpoint $x_T = x$ (a replica of the current weather conditions), with the intermediate state distribution being $q(x_t|x_0, x_T) = \mathcal{N}((1-m_t)x_0 + m_tx_T, \delta_t I)$, where $m_t = t/T$. To enable spatial-temporal modeling in the 3D-UNet, the features $z$ extracted by DB are injected into each layer of PB via cross-attention. The training objective is $L_{PB} = \mathbb{E}\|m_t(x_T - x_0) + \delta_t\epsilon - \epsilon_\theta(x_t, t, z)\|^2$. The total loss is $L_{total} = L_{PB} + L_{DB}$.
    - **Design Motivation**: The Brownian Bridge is better suited for weather forecasting than standard DDPM because it naturally constrains the start and end points of the diffusion process—the mapping between the initial weather conditions and future weather. Cross-attention injection of DB features allows PB to exploit the precise dynamic information already captured by the deterministic model.

3. **Sequential Variance Schedule + Truncated Diffusion**:

    - **Function**: Dynamically allocates diffusion steps based on the forecast horizon to reflect the growth of uncertainty over time, while utilizing deterministic results to accelerate and constrain the reverse process.
    - **Mechanism**: **Sequential Variance Schedule (SVS)**: In weather forecasting, longer forecast horizons carry greater uncertainty. SVS allocates different numbers of diffusion steps for each prediction time step—fewer steps for near-term forecasts (low uncertainty) and more steps for long-term forecasts (high uncertainty). The formula is $\text{SVS} = \{T - (\hat{L}-i) \cdot S : i=1,...,\hat{L}\}$, where $S$ is the step size. **Truncated Diffusion**: During inference, substituting the DB prediction $\hat{y}$ into the formula $\hat{x}_t = (1-m_t)\hat{y} + m_tx_T + \delta_t\epsilon$, the reverse process starts from the intermediate state $\hat{x}_t$ instead of the endpoint $x_T$. This controls the range of diversity (around $\hat{y}$) and substantially reduces the number of diffusion steps required for inference.
    - **Design Motivation**: SVS aligns with the physical principles of meteorology—near-term weather is more predictable. Truncated diffusion elegantly blends determinism and probability, using the deterministic outcome as an "anchor", where probability is only responsible for expressing uncertainty around the anchor rather than searching the entire potential space from scratch.

### Loss & Training
Joint training: $L_{total} = L_{PB} + L_{DB}$. Adam optimizer, learning rate of 3e-4 for DB and 1e-4 for PB. 1000 forward steps, 200 reverse steps truncated to 100 steps. Training is stabilized using EMA (decay rate of 0.995). The number of training epochs varies by dataset: 2000 epochs for Moving MNIST, 200 epochs for PNW-Typhoon, and 50 epochs for WeatherBench. Trained on a single NVIDIA A100 GPU.

## Key Experimental Results

### Main Results (Moving MNIST)

| Model | Diversity | MAE↓ | MSE↓ | SSIM↑ | FVD↓ |
|------|--------|------|------|-------|------|
| TAU (Deterministic) | ✗ | 51.46 | 15.68 | 0.966 | 28.17 |
| RaMViD (Probabilistic) | ✓ | 123.76 | 81.26 | 0.878 | 12.06 |
| DGDM-Best | ✗ | **47.31** | **19.14** | **0.966** | **7.43** |
| DGDM-SB | ✓ | 50.21 | 20.96 | 0.962 | 7.46 |

### WeatherBench Global Forecasting

| Model | Temp MSE↓ | Humidity MSE↓ | Wind Speed MSE↓ |
|------|---------|---------|---------|
| TAU | 1.162 | 31.831 | 1.5925 |
| RaMViD | 1.908 | 39.028 | 2.7639 |
| DGDM-Best | **1.025** | **28.572** | **1.5914** |

### Ablation Study (Moving MNIST)

| Configuration | MAE↓ (Probabilistic) | FVD↓ (Probabilistic) | Description |
|------|------------|------------|------|
| DB only | 58.13 | 18.50 | Deterministic only, no diversity |
| PB only | 123.20 | 8.80 | Has diversity but poor accuracy |
| DB+PB (w/o Brownian Bridge) | 110.07 | 14.74 | No endpoint constraints |
| DB+PB+Brownian Bridge | 52.04 | 10.06 | Brownian Bridge significantly improves accuracy |
| + Last Frame (LF) Replication | 50.35 | 9.31 | Further improvement |
| +SVS (Full Model) | **50.22** | **8.28** | SVS improves FVD and accelerates inference |

### Key Findings
- When DB and PB are jointly trained, the performance of DB itself is also improved—indicating that the probabilistic branch has a regularizing effect on the deterministic branch.
- Truncated diffusion not only controls diversity but also improves accuracy: truncating to 100 steps reduces MAE by 11% compared to 200 non-truncated steps.
- Fewer truncation steps lead to a lower STD (more controllable uncertainty) but a slightly higher MAE—allowing adjustment of the accuracy-diversity trade-off as needed.
- SVS enables faster inference for short-term forecasts (fewer steps required) while also improving the FVD metric.
- DGDM shows a more pronounced advantage in regional high-resolution forecasting (PNW-Typhoon), with typhoon details (such as the eye and cloud structures) significantly outperforming both purely deterministic and purely probabilistic models.

## Highlights & Insights
- **Deterministic guidance-based truncated diffusion** is the most ingenious design of this work. It simultaneously solves three challenges: improving accuracy (anchoring on deterministic results), controlling diversity (truncating limits the search space), and accelerating inference (reducing reverse steps). This concept can be transferred to any generative task requiring a balanced trade-off between accuracy and diversity.
- **SVS (Sequential Variance Schedule)** reflects respect for physical laws, where uncertainty grows linearly with the forecast horizon. This practice of encoding domain knowledge into the model architecture is highly valuable.
- The mutual benefit of jointly training DB and PB is an intriguing finding—gradient signals from the probabilistic branch exert a regularizing effect on the deterministic branch.

## Limitations & Future Work
- DGDM is fundamentally still a probabilistic model, and choosing which sample is closest to ground truth remains a challenge—a limitation the paper circumvents by using a "Best" selection strategy.
- Currently, only short-term forecasts (10-12 hours) have been validated; its effectiveness on medium-to-long-range forecasts (3-10 days) remains unknown.
- The introduced PNW-Typhoon dataset only covers the East Asian region, leaving its generalizability to be validated.
- Future work could attempt to use the deterministic branch to predict the uncertainty itself (e.g., predicting variance) to achieve adaptive truncation.
- Comparisons with the latest large-scale weather foundation models (e.g., GenCast, Aardvark Weather) are currently missing.

## Related Work & Insights
- **vs TAU/SimVP**: Purely deterministic models perform strongly in MAE/SSIM but poorly in FVD, reflecting their inability to generate clear details. DGDM achieves better texture quality through its probabilistic branch.
- **vs RaMViD/MCVD**: Purely probabilistic models suffer from uncontrollable diversity, whereas DGDM effectively restricts the range of diversity via truncated diffusion.
- **vs GraphCast/Pangu-Weather**: These large models show superior performance in global forecasting but lack the ability to make probabilistic predictions. The approach of DGDM could serve as a probabilistic extension for them.

## Rating
- Novelty: ⭐⭐⭐⭐ Deterministic guidance-based truncated diffusion and SVS are both highly novel designs
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation on three datasets (MMNIST/PNW/WeatherBench) with exhaustive ablations
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with highly detailed method derivations
- Value: ⭐⭐⭐⭐ Provides an elegant solution for integrating deterministic and probabilistic weather forecasting

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
