---
title: >-
  [Paper Note] MIRA: Medical Time Series Foundation Model for Real-World Health Data
description: >-
  [NeurIPS 2025][Medical Imaging][Time series foundation model] This paper presents MIRA, a foundation model specifically designed for irregular medical time series. Through continuous-time rotary position encoding (CT-RoP…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Time series foundation model"
  - "irregular sampling"
  - "medical signals"
  - "Neural ODE"
  - "Mixture-of-Experts"
date: 2026-05-08
content_hash: 23c3e7c4d708e176
---

# MIRA: Medical Time Series Foundation Model for Real-World Health Data

**Conference**: NeurIPS 2025
**arXiv**: [2506.07584](https://arxiv.org/abs/2506.07584)
**Code**: [GitHub](https://github.com/Microsoft/MIRA)
**Area**: Medical Signals
**Keywords**: Time series foundation model, irregular sampling, medical signals, Neural ODE, Mixture-of-Experts

## TL;DR

This paper presents MIRA, a foundation model specifically designed for irregular medical time series. Through continuous-time rotary position encoding (CT-RoPE), frequency-specific Mixture-of-Experts (MoE), and a Neural ODE-based extrapolation module, MIRA is pretrained on 454 billion observation points and achieves zero-shot forecasting performance that reduces average error by 8% and 6% in OOD and in-distribution (ID) settings, respectively.

## Background & Motivation

Medical time series data (ECG, EEG, vital signs, laboratory tests) are essential for understanding patients' dynamic physiological states. However, building effective medical time series foundation models faces three core challenges:

**Irregular sampling**: ECG recordings are captured at millisecond intervals, while laboratory tests may be separated by hours, resulting in highly non-uniform temporal spacing.

**Frequent missing values**: Clinical workflows lead to substantial amounts of missing observations.

**Heterogeneous sampling rates**: Different devices and clinical contexts produce signals with vastly different frequencies.

Existing general-purpose time series foundation models (Chronos, Moirai, TimesFM, etc.) typically assume uniform time intervals and cannot handle irregular clinical data. Emerging medical time series foundation models achieve cross-dataset generalization only within narrow domains (e.g., EEG sleep monitoring) and are unable to handle continuous-time forecasting or irregular sampling.

**Core Problem**: How to build a unified medical time series foundation model capable of handling irregular sampling, heterogeneous frequencies, and frequent missing values?

## Method

### Overall Architecture

MIRA adopts a decoder-only architecture comprising three core components: Continuous-Time Rotary Position Encoding (CT-RoPE), sparse temporal Mixture-of-Experts (MoE) layers, and a Neural ODE-based continuous dynamics extrapolation module. A channel-independent setting is employed to process univariate time series.

### Key Designs

1. **Continuous-Time Rotary Position Encoding (CT-RoPE)**: Standard RoPE assumes discrete, uniformly spaced token indices and cannot accommodate continuous timestamps. CT-RoPE directly maps continuous timestamps $t \geq 0$ to rotation angles:

$$\theta_i(t) = \omega_i \cdot t, \quad \omega_i = 10000^{-2i/d}$$

Planar rotations are applied to each pair of components of the input embedding $\mathbf{x} \in \mathbb{R}^d$. A key property is that attention scores depend only on time differences:

$$\langle q_m, k_n \rangle = x_m^\top (W^Q)^\top R_\Theta(t_n - t_m) W^K x_n$$

This enables the model to capture relative positional relationships in continuous time while preserving the efficiency of standard dot-product attention.

2. **Frequency-Specific Mixture-of-Experts (MoE)**: Medical time series exhibit distinct dynamics across multiple temporal frequencies (smooth long-term trends vs. rapid short-term fluctuations). The MoE layer replaces the standard FFN sub-layer, routing each token to $K$ experts:

$$\text{MoE}(\bar{\mathbf{u}}_t^l) = g_{N+1,t} \cdot \text{FFN}_{N+1}(\bar{\mathbf{u}}_t^l) + \sum_{i=1}^N g_{i,t} \cdot \text{FFN}_i(\bar{\mathbf{u}}_t^l)$$

where $\text{FFN}_{N+1}$ is a shared expert (global residual pathway) and $g_{i,t}$ is computed via softmax with top-$K$ selection. The shared expert weights are computed through an independent sigmoid gate to capture universal patterns.

3. **Continuous Dynamics Extrapolation Module (Neural ODE)**: An autoregressive Transformer cannot access the timestamp of the target token at inference time. This module employs a Neural ODE to extrapolate the hidden state from the current timestamp $t_N$ to the target timestamp $t_{N+1}$:

$$h(t_{N+1}) = h(t_N) + \int_{t_N}^{t_{N+1}} f(s - t_N, h(s); \theta_{\text{ODE}}) \, ds$$

Numerical integration is performed using the Dormand–Prince (RK45) method, enabling prediction at arbitrary unseen time points.

### Loss & Training

- **Primary loss**: Huber loss, robust to outliers and noisy measurements, with threshold $\delta$ controlling the L2/L1 transition.
- **Load balancing loss**: $\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^N f_i r_i$, preventing expert collapse and promoting uniform utilization.
- **Pretraining corpus**: 454 billion time points drawn from publicly available datasets including MIMIC-III/IV, PTB-XL, Sleep-EDF, and WAVES.
- Training continues from a Time-MoE checkpoint using up to 8 NVIDIA H/A100 GPUs.

## Key Experimental Results

### Main Results (OOD Zero-Shot Forecasting)

| Dataset | Metric | MIRA_large | Time-MoE_large | Moirai_large | Chronos_large | Best Supervised |
|---------|--------|------------|----------------|--------------|---------------|-----------------|
| Heart Rate | RMSE(×10⁻¹) | **1.392** | 0.833 | 2.098 | 1.218 | 0.774 (Contiformer) |
| MIT-BIH | RMSE | **0.130** | 0.135 | 0.593 | 0.350 | 0.453 (ODE-RNN) |
| CDC-IHA | RMSE(×10¹) | **4.401** | 4.748 | 6.788 | 15.986 | 5.211 (Contiformer) |
| JH COVID-19 | RMSE(×10²) | **0.336** | 0.402 | 0.614 | 3.478 | 0.323 (Contiformer) |
| ILI | RMSE | **1.041** | 0.951 | 1.499 | 1.870 | 0.391 (Contiformer) |

MIRA_large achieves the best RMSE across all OOD datasets, with an average error reduction of approximately 8%.

### ID Zero-Shot Forecasting

| Dataset | Metric | MIRA_large | Time-MoE_large | Chronos_large |
|---------|--------|------------|----------------|--------------|
| SleepEDF | RMSE(×10²) | **0.189** | 0.244 | 0.413 |
| PTB-XL | RMSE | **0.121** | 0.109 | 0.229 |
| MIMIC-III | RMSE | **0.102** | 0.103 | 0.151 |
| MIMIC-IV | RMSE | **0.081** | 0.082 | 0.319 |
| WAVES | RMSE | **0.129** | 0.141 | 0.182 |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Remove CT-RoPE | Significant performance degradation; positional encoding is critical for irregular time series |
| Replace MoE with standard FFN | Reduced capacity to capture multi-frequency temporal dynamics |
| Replace Neural ODE with linear extrapolation | Unable to accurately predict at arbitrary time points |
| Medical pretraining vs. general pretraining | All model variants consistently outperform their general-domain counterparts after continued pretraining on medical corpora |

### Key Findings

- **Domain-specific pretraining is critical**: After continued pretraining on medical corpora, all model variants consistently outperform their general-domain counterparts; smaller models even surpass larger general-purpose models.
- **Robustness to missing data**: As the missing rate increases from 10% to 90%, MIRA's error grows gradually, significantly outperforming baselines such as Moirai.
- **Favorable scalability**: Performance improves steadily from MIRA_small (73M) to MIRA_large (455M parameters).
- On multiple datasets, MIRA's zero-shot performance approaches or even surpasses fully supervised fine-tuned models.

## Highlights & Insights

- **Precise targeting of real-world challenges**: Irregular sampling, heterogeneous frequencies, and frequent missing values are central challenges in clinical data; MIRA is the first foundation model to address all three simultaneously within a unified architecture.
- **Elegant CT-RoPE design**: RoPE is naturally extended to the continuous-time domain, preserving the favorable properties of relative position modeling.
- **Substantial data scale**: The pretraining corpus of 454 billion time points is among the largest medical time series pretraining datasets to date.
- **MoE + Neural ODE combination**: Frequency-specific experts handle multi-scale temporal dynamics, while the ODE module enables continuous-time extrapolation — a well-motivated design.

## Limitations & Future Work

- The channel-independent design may discard inter-variable correlation information, which is important in ICU multi-parameter monitoring scenarios.
- The Neural ODE module incurs considerable computational overhead due to adaptive-step ODE solvers, potentially limiting real-time applicability.
- Evaluation is primarily focused on forecasting tasks and does not cover other clinically relevant tasks such as classification and anomaly detection.
- The pretraining data originates from a limited number of sources (5 public datasets); performance may improve further with more diverse data.

## Related Work & Insights

Compared to general-purpose time series foundation models such as Time-MoE, Moirai, and Chronos, MIRA's core distinction lies in its native support for irregular time series. Compared to irregular time series models such as ContiFormer and Neural-CDE, MIRA is the first to reach foundation model scale. Key insight: the unique characteristics of medical data — irregularity and missing values — require architectural-level native support rather than post-hoc remedies such as interpolation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ CT-RoPE and the MoE+ODE combination offer notable design contributions, though each individual component is not entirely novel in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 7 OOD and 5 ID datasets against 13 baselines, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with sufficient technical detail.
- **Value**: ⭐⭐⭐⭐⭐ Fills a significant gap in medical time series foundation models; both datasets and benchmarks are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)

</div>

<!-- RELATED:END -->
