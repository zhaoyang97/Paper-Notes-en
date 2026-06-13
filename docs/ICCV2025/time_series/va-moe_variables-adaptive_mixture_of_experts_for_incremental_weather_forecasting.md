---
title: >-
  [Paper Note] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting
description: >-
  [ICCV 2025][Time Series][Weather Forecasting] This paper proposes a novel incremental weather forecasting paradigm and the VA-MoE framework. Through a variables-adaptive MoE architecture and index embedding mechanism…
tags:
  - "ICCV 2025"
  - "Time Series"
  - "Weather Forecasting"
  - "Incremental Learning"
  - "Mixture of Experts"
  - "Variable-Adaptive"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 3b1c76ba29d08949
---

# VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting

**Conference**: ICCV 2025
**arXiv**: [2412.02503](https://arxiv.org/abs/2412.02503)  
**Code**: [https://github.com/chenhao-zju/VAMoE](https://github.com/chenhao-zju/VAMoE)  
**Area**: Time Series Forecasting / Weather Prediction
**Keywords**: Weather Forecasting, Incremental Learning, Mixture of Experts, Variable-Adaptive, Catastrophic Forgetting

## TL;DR

This paper proposes a novel incremental weather forecasting paradigm and the VA-MoE framework. Through a variables-adaptive MoE architecture and index embedding mechanism, VA-MoE achieves forecasting accuracy comparable to full training with only 25% trainable parameters and 50% of the initial training data.

## Background & Motivation

Data-driven AI weather forecasting models (e.g., Pangu-Weather, GraphCast) have achieved remarkable progress, yet share a fundamental assumption: all variables are simultaneously available at both training and inference time. In practice:

**Variable heterogeneity**: Upper-air variables (e.g., temperature profiles) are sparsely sampled via radiosondes/satellites, while surface variables (e.g., precipitation, wind speed) are densely updated in near real-time.

**High retraining cost**: Incorporating new variables (e.g., satellite aerosol data) requires complete retraining — Pangu-Weather demands 64 days × 192 V100 GPUs.

**Catastrophic forgetting**: Incrementally introducing new variables causes pretrained parameters to drift toward the new distribution, severely degrading performance on existing variables.

This paper is the first to propose the Incremental Weather Forecasting (IWF) paradigm and addresses the dynamic variable expansion problem with VA-MoE.

## Method

### Overall Architecture

A two-stage training paradigm is adopted:
- **Initial stage**: Five categories of upper-air variables (Z/Q/U/V/T, each with 13 pressure levels, 65 channels total) are trained on 40 years of data, with each category handled by a Channel-Adaptive Expert (CAE).
- **Incremental stage**: Existing experts are frozen; a new CAE_SV is added to handle 5 surface variables (u10/v10/t2m/msl/sp), training only the newly added modules on 20 years (half of the original 40 years) of data.

The model is built on a Transformer backbone, replacing the FFN layer in each Transformer block with the VA-MoE module.

### Key Designs

1. **Variable Index Embedding**: A one-hot index embedding $\mathbf{I}_h \in \mathbb{R}^{5 \times N}$ is introduced to guide expert learning of variable affinity. The index embedding is encoded into a latent space via a linear layer and element-wise multiplied with input features at the channel level within the CAE module, encouraging experts to develop domain-specific specialization. During the incremental stage, the index embedding is expanded from $\mathbb{R}^{(N \times l) \times N}$ to $\mathbb{R}^{(N \times l + M \times r) \times (N+M)}$.

2. **Channel-Adaptive Expert (CAE)**: Each CAE handles a specific variable type. The key procedure is: channel-wise multiplication of index embedding with input features → GateEmbed layer → SoftMax + TopK selection of top-K ranked channels → obtain GateIndex and GateWeight → select and weight input features → feed into Expert MLP. The formulation is: $\mathbf{I}_Z^{topk}, \mathbf{W}_Z^{topk} = \text{TOP}_k(\text{SoftMax}(\text{MLP}_Z(\mathbf{X}_h^t \odot \mathbf{I}_Z)))$. This is an auxiliary-loss-free design.

3. **Shared Expert**: Processes global features across all variables in parallel; its output is summed with each CAE output and then projected back to the original channel dimension via an upsampling linear layer: $(X')_h^t = \text{Expert}_{shared}(X_h^t) + \text{Linear}_{up}(X_h^{t,fused})$.

### Loss & Training

- **Dynamic prediction loss**: Channel-wise learnable weights $\mathbf{w}$ are introduced to dynamically assign gradients across variables: $Obj_{pred} = (\hat{X}^{t+1} - X^{t+1})^2 / e^{\mathbf{w}} + \mathbf{w}$. Fast-changing variables (e.g., temperature) receive larger gradients, while slow-changing variables (e.g., geopotential height) have their weights gradually adjusted.
- **Reconstruction loss**: $Obj_{recon} = (\hat{X}^t - X^t)^2$, ensuring the encoder-decoder focuses on feature encoding and decoding.
- **Total loss**: $Obj_{final} = Obj_{pred} + \lambda \cdot Obj_{recon}$
- AdamW optimizer; learning rate 0.0002 (initial stage) and 0.00005 (incremental stage); 100 epochs per stage; batch size 16.
- Training on 16 A100 GPUs.

## Key Experimental Results

### Main Results — Surface Variable Forecasting

RMSE comparison on ERA5 for 5 surface variables (T2M/U10/V10/MSL/SP):

| Method | Training | T2M-6h | T2M-72h | T2M-120h | U10-72h | U10-120h |
|--------|----------|--------|---------|----------|---------|----------|
| Pangu-Weather | Full | 0.82 | 1.09 | 1.53 | 1.63 | 2.54 |
| GraphCast | Full | 0.51 | 0.94 | 1.37 | 1.51 | 2.37 |
| FuXi | Full | 0.55 | 0.99 | 1.41 | 1.50 | 2.36 |
| VA-MoE | Full | 0.57 | 1.03 | 1.42 | **1.41** | **2.25** |
| VA-MoE(IL) 40yr | Incremental | 0.58 | 1.05 | 1.45 | 1.47 | 2.33 |
| VA-MoE(IL) 20yr | Incremental | 0.73 | 1.17 | 1.57 | 1.58 | 2.49 |

VA-MoE achieves the best long-range forecasting performance on U10 and V10. Incremental training (40 years of data, half the iterations) nearly matches full training.

### Ablation Study — Architecture Comparison

Upper-air variable 500 hPa forecast (1.5° resolution):

| Method | Params (M) | Z500-6h | Z500-72h | Z500-120h | T500-72h | T500-120h |
|--------|-----------|---------|----------|-----------|----------|-----------|
| ViT | 307 | 33.38 | 209.4 | 517.81 | 1.18 | 2.40 |
| ViT+MoE(light) | 609 | 37.92 | 207.11 | 405.73 | 1.23 | 2.02 |
| ViT+MoE | 1113 | 28.31 | 169.61 | 356.02 | 1.07 | 1.83 |
| **VA-MoE** | **665** | **20.59** | **139.02** | **302.13** | **0.92** | **1.59** |
| VA-MoE(IL) | 137 | 20.29 | 138.52 | 301.41 | 0.93 | 1.60 |

VA-MoE with 665M parameters significantly outperforms ViT+MoE with 1113M parameters. The incremental variant (only 137M trainable parameters) achieves nearly identical performance to full training.

### Key Findings

- After incremental training, VA-MoE(IL) slightly improves long-range Z500 forecasting over the initial training version, confirming the absence of catastrophic forgetting.
- Incremental training using only 20 years of data (50% of initial data) and 25% of the training iterations still maintains acceptable accuracy.
- The combination of index embedding and CAE enables domain-specific expert specialization without any auxiliary loss.
- Visualization of 6-hour global forecasts shows maximum absolute errors of 0.08% for Z500 and 0.22% for T850.

## Highlights & Insights

- **First proposal of the incremental weather forecasting paradigm**: Introduces incremental learning into the weather forecasting domain and establishes a quantitative benchmark.
- **Auxiliary-loss-free expert specialization**: Index embeddings drive expert diversity, eliminating the load-balancing auxiliary losses typical in conventional MoE designs.
- **Dynamic prediction loss**: Adaptive weight learning that accounts for the distributional characteristics of different variables is more principled than treating all variables uniformly.
- **Simple and effective freezing strategy**: During the incremental stage, only new experts and the shared expert are trained while all original experts are completely frozen, cleanly preventing catastrophic forgetting.

## Limitations & Future Work

- Only a single incremental step (from upper-air to surface variables) is validated; multi-step incremental scenarios are not explored.
- The incremental stage still requires 50% of the original upper-air variable data, making it not entirely data-free incremental learning.
- Surface variables consist of only 5 single-level fields, which differ substantially in complexity from the 13-level upper-air variables.
- A performance gap remains versus the strongest baselines (e.g., GraphCast) on certain variables such as T2M.
- Forecasting capability for extreme weather events is not investigated.

## Related Work & Insights

- Distinction from EWMoE: VA-MoE targets incremental learning scenarios and adopts variable-level expert assignment rather than task-level assignment.
- Expert Gate and Lifelong-MoE provided inspiration for MoE designs in visual incremental learning.
- The proposed approach offers reference value for other spatiotemporal forecasting tasks that require dynamic variable expansion, such as traffic and energy forecasting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The incremental weather forecasting paradigm and variable-adaptive MoE design are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation on ERA5 with comparisons against multiple state-of-the-art methods.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear and the framework is described systematically.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution to the scalability challenges of AI-based meteorological modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](../../AAAI2026/time_series/m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](../../CVPR2026/time_series/stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/time_series/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](../../NeurIPS2025/time_series/graph-based_neural_space_weather_forecasting.md)

</div>

<!-- RELATED:END -->
